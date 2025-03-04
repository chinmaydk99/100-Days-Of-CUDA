#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define THREADS_PER_BLOCK 256 
#define CHUNK_SIZE 65536 
#define MAX_CHUNKS 4 

void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}


__global__ void MaxLogitsKernel(
    const float* __restrict__ logits,  // [nrows, vocab_size]
    float* __restrict__ max_per_chunk, // [nrows, num_chunks]
    int nrows,
    int vocab_size,
    int num_chunks
) {
    extern __shared__ float shared_max[];
    
    int row = blockIdx.x;
    int chunk_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (row >= nrows) return;
    
    int chunk_start = chunk_idx * CHUNK_SIZE;
    int chunk_end = min(chunk_start + CHUNK_SIZE, vocab_size);
    const float* row_logits = logits + row * vocab_size;
    
    shared_max[tid] = -INFINITY;
    
    for (int i = chunk_start + tid; i < chunk_end; i += blockDim.x) {
        shared_max[tid] = fmaxf(shared_max[tid], row_logits[i]);
    }
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        max_per_chunk[row * num_chunks + chunk_idx] = shared_max[0];
    }
}


__global__ void SumExpKernel(
    const float* __restrict__ logits,      // [nrows, vocab_size]
    const float* __restrict__ max_per_row, // [nrows]
    float* __restrict__ sum_exp_per_chunk, // [nrows, num_chunks]
    float* __restrict__ local_max_per_chunk, // [nrows, num_chunks]
    int nrows,
    int vocab_size,
    int num_chunks
) {
    extern __shared__ float shared_sum[];
    
    int row = blockIdx.x;
    int chunk_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (row >= nrows) return;
    
    int chunk_start = chunk_idx * CHUNK_SIZE;
    int chunk_end = min(chunk_start + CHUNK_SIZE, vocab_size);
    const float* row_logits = logits + row * vocab_size;
    
    float row_max = max_per_row[row];
    float chunk_max = local_max_per_chunk[row * num_chunks + chunk_idx];
    float sum_exp = 0.0f;
    
    for (int i = chunk_start + tid; i < chunk_end; i += blockDim.x) {
        sum_exp += expf(row_logits[i] - row_max);
    }
    
    shared_sum[tid] = sum_exp;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        sum_exp_per_chunk[row * num_chunks + chunk_idx] = shared_sum[0];
    }
}

__global__ void CrossEntropyLossKernel(
    const float* __restrict__ logits,      // [nrows, vocab_size]
    const int* __restrict__ labels,        // [nrows]
    const float* __restrict__ max_per_row, // [nrows]
    const float* __restrict__ sum_exp_per_chunk, // [nrows, num_chunks]
    float* __restrict__ loss,              // [nrows]
    int nrows,
    int vocab_size,
    int num_chunks
) {
    int row = blockIdx.x;
    
    if (threadIdx.x != 0 || row >= nrows) return;
    
    float row_max = max_per_row[row];
    float sum_exp = 0.0f;
    
    // Sum all chunks
    for (int i = 0; i < num_chunks; i++) {
        sum_exp += sum_exp_per_chunk[row * num_chunks + i];
    }
    
    // Compute logsumexp
    float logsumexp = row_max + logf(sum_exp);
    
    // Get correct logit
    int label = labels[row];
    if (label >= 0 && label < vocab_size) {
        float correct_logit = logits[row * vocab_size + label];
        loss[row] = logsumexp - correct_logit;
    } else {
        loss[row] = 0.0f;
    }
}

void compute_cross_entropy_forward(float *h_logits, int *h_labels, float *h_loss, int n_rows, int vocab_size) {
    int num_chunks = (vocab_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
    if (num_chunks > MAX_CHUNKS) {
        std::cerr << "Warning: Vocabulary size exceeds maximum supported chunks. "
                  << "Adjusting to " << MAX_CHUNKS << " chunks." << std::endl;
        num_chunks = MAX_CHUNKS;
    }
    
    float *d_logits, *d_loss;
    int *d_labels;
    float *d_max_per_chunk, *d_max_per_row;
    float *d_sum_exp_per_chunk;
    
    cudaMalloc(&d_logits, n_rows * vocab_size * sizeof(float));
    cudaMalloc(&d_labels, n_rows * sizeof(int));
    cudaMalloc(&d_loss, n_rows * sizeof(float));
    cudaMalloc(&d_max_per_chunk, n_rows * num_chunks * sizeof(float));
    cudaMalloc(&d_max_per_row, n_rows * sizeof(float));
    cudaMalloc(&d_sum_exp_per_chunk, n_rows * num_chunks * sizeof(float));
    
    checkCudaError(cudaGetLastError(), "Memory allocation failed");
    
    cudaMemcpy(d_logits, h_logits, n_rows * vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, n_rows * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError(cudaGetLastError(), "Memory copy to device failed");
    
    cudaMemset(d_loss, 0, n_rows * sizeof(float));

    int shared_memory_size = THREADS_PER_BLOCK * sizeof(float);
    
    dim3 grid_dim(n_rows, num_chunks);
    MaxLogitsKernel<<<grid_dim, THREADS_PER_BLOCK, shared_memory_size>>>(
        d_logits, d_max_per_chunk, n_rows, vocab_size, num_chunks
    );
    checkCudaError(cudaGetLastError(), "Max logits kernel launch failed");
    cudaDeviceSynchronize();
    
    float *h_max_per_chunk = new float[n_rows * num_chunks];
    cudaMemcpy(h_max_per_chunk, d_max_per_chunk, n_rows * num_chunks * sizeof(float), cudaMemcpyDeviceToHost);
    float *h_max_per_row = new float[n_rows];
    
    for (int row = 0; row < n_rows; row++) {
        float max_val = h_max_per_chunk[row * num_chunks];
        for (int i = 1; i < num_chunks; i++) {
            max_val = std::max(max_val, h_max_per_chunk[row * num_chunks + i]);
        }
        h_max_per_row[row] = max_val;
    }
    
    cudaMemcpy(d_max_per_row, h_max_per_row, n_rows * sizeof(float), cudaMemcpyHostToDevice);
    
    SumExpKernel<<<grid_dim, THREADS_PER_BLOCK, shared_memory_size>>>(
        d_logits, d_max_per_row, d_sum_exp_per_chunk, d_max_per_chunk, n_rows, vocab_size, num_chunks
    );
    checkCudaError(cudaGetLastError(), "Sum exp kernel launch failed");
    cudaDeviceSynchronize();
    
    CrossEntropyLossKernel<<<n_rows, 1>>>(
        d_logits, d_labels, d_max_per_row, d_sum_exp_per_chunk, d_loss, n_rows, vocab_size, num_chunks
    );
    checkCudaError(cudaGetLastError(), "Loss kernel launch failed");
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_loss, d_loss, n_rows * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError(cudaGetLastError(), "Memory copy from device failed");
    
    cudaFree(d_logits);
    cudaFree(d_labels);
    cudaFree(d_loss);
    cudaFree(d_max_per_chunk);
    cudaFree(d_max_per_row);
    cudaFree(d_sum_exp_per_chunk);
    
    delete[] h_max_per_chunk;
    delete[] h_max_per_row;
}

int main() {
    int n_rows = 2, vocab_size = 256000; // Large vocab size
    float *h_logits = new float[n_rows * vocab_size];
    int *h_labels = new int[n_rows];
    float *h_loss = new float[n_rows];
    

    srand(42); 

    for (int i = 0; i < n_rows * vocab_size; i++) {
        h_logits[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
    h_labels[0] = 150000; 
    h_labels[1] = 200000;
    
    float *cpu_loss = new float[n_rows];
    for (int row = 0; row < n_rows; row++) {
        int label = h_labels[row];
        if (label < 0 || label >= vocab_size) {
            cpu_loss[row] = 0.0f;
            continue;
        }
        

        float max_val = h_logits[row * vocab_size];
        for (int i = 1; i < vocab_size; i++) {
            max_val = std::max(max_val, h_logits[row * vocab_size + i]);
        }

        double sum_exp = 0.0;
        for (int i = 0; i < vocab_size; i++) {
            sum_exp += std::exp(static_cast<double>(h_logits[row * vocab_size + i] - max_val));
        }
        
        float logsumexp = max_val + std::log(sum_exp);
        
        cpu_loss[row] = logsumexp - h_logits[row * vocab_size + label];
    }
    
    compute_cross_entropy_forward(h_logits, h_labels, h_loss, n_rows, vocab_size);
    
    std::cout << "Cross-Entropy Loss Results:" << std::endl;
    std::cout << "----------------------------" << std::endl;
    for (int i = 0; i < n_rows; i++) {
        std::cout << "Row " << i << " - GPU: " << h_loss[i] << ", CPU: " << cpu_loss[i]
                  << ", Diff: " << std::abs(h_loss[i] - cpu_loss[i]) << std::endl;
    }
    
    delete[] h_logits;
    delete[] h_labels;
    delete[] h_loss;
    delete[] cpu_loss;
    
    return 0;
}