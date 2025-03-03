#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define THREADS_PER_BLOCK 32
#define CHUNK_SIZE 65536  // Each chunk is 64K
#define MAX_CHUNKS 4  // 256K vocab split into 4 chunks

__global__ void CrossEntropyForward_Kernel(
    const float* __restrict__ logits, // [nrows, vocab_size]
    const int* __restrict__ labels,   // [nrows]
    float* loss,                      // [nrows]
    float* logsumexp_chunks,          // [nrows, MAX_CHUNKS]
    int nrows,
    int vocab_size,
    int num_chunks
) {
    extern __shared__ float shared_data[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int chunk_idx = blockIdx.y;  // Each block processes a chunk

    // Guard against out-of-bounds row access
    if (row >= nrows) return;

    const float* row_logits = logits + row * vocab_size;
    float* logsumexp_row = logsumexp_chunks + row * num_chunks;

    int chunk_start = chunk_idx * CHUNK_SIZE;
    int chunk_end = fmin(chunk_start + CHUNK_SIZE, vocab_size);

    // Step 1: Compute max value in chunk
    float local_max = -INFINITY;
    for (int i = tid + chunk_start; i < chunk_end; i += blockDim.x) {
        local_max = fmaxf(local_max, row_logits[i]);
    }

    shared_data[tid] = local_max;
    __syncthreads();

    // Parallel reduction to find chunk max
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_data[0];

    // Step 2: Compute sum(exp(x - max))
    float sum_exp = 0.0f;
    for (int i = tid + chunk_start; i < chunk_end; i += blockDim.x) {
        sum_exp += expf(row_logits[i] - max_val);
    }

    shared_data[tid] = sum_exp;
    __syncthreads();

    // Parallel reduction to compute sum(exp(x - max))
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    sum_exp = shared_data[0];

    // Compute logsumexp for this chunk
    float logsumexp_chunk = max_val + logf(sum_exp);
    if (tid == 0) {
        logsumexp_row[chunk_idx] = logsumexp_chunk;
        // Remove debug printf to improve performance
    }

    __syncthreads();

    // Step 3: Compute final logsumexp across chunks (only in the first chunk's block)
    if (chunk_idx == 0) {
        __shared__ float final_max;
        __shared__ float final_sum;

        // First, find max across all chunks
        if (tid < num_chunks && tid > 0) {
            shared_data[tid] = logsumexp_row[tid];
        } else if (tid == 0) {
            shared_data[0] = logsumexp_row[0];
        }
        __syncthreads();

        // Find max logsumexp across chunks
        if (tid == 0) {
            float max_logsumexp = shared_data[0];
            for (int i = 1; i < num_chunks; i++) {
                if (i < num_chunks) {
                    max_logsumexp = fmaxf(max_logsumexp, shared_data[i]);
                }
            }
            final_max = max_logsumexp;
        }
        __syncthreads();

        // Compute sum(exp(logsumexp_chunk - max_logsumexp))
        if (tid < num_chunks) {
            shared_data[tid] = (tid < num_chunks) ? expf(logsumexp_row[tid] - final_max) : 0.0f;
        }
        __syncthreads();

        // Reduce to get final sum
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride && tid < num_chunks) {
                shared_data[tid] += (tid + stride < num_chunks) ? shared_data[tid + stride] : 0.0f;
            }
            __syncthreads();
        }

        if (tid == 0) {
            final_sum = shared_data[0];
        }
        __syncthreads();

        // Step 4: Each thread checks if its chunk contains the correct label
        __shared__ float correct_logit;
        if (tid == 0) {
            int label = labels[row];
            if (label >= 0 && label < vocab_size) {
                correct_logit = row_logits[label];
                
                // Compute the final loss
                float final_logsumexp = final_max + logf(final_sum);
                loss[row] = final_logsumexp - correct_logit;
            } else {
                // Handle invalid label
                loss[row] = 0.0f;
            }
        }
    }
}

void compute_cross_entropy_forward(float *h_logits, int *h_labels, float *h_loss, int n_rows, int vocab_size) {
    float *d_logits, *d_loss, *d_logsumexp_chunks;
    int *d_labels;

    // Calculate number of chunks needed for the vocabulary
    int num_chunks = (vocab_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
    if (num_chunks > MAX_CHUNKS) {
        std::cerr << "Warning: Vocabulary size exceeds maximum supported chunks. "
                  << "Increase MAX_CHUNKS or CHUNK_SIZE." << std::endl;
        num_chunks = MAX_CHUNKS;
    }

    // Allocate device memory
    cudaMalloc(&d_logits, n_rows * vocab_size * sizeof(float));
    cudaMalloc(&d_labels, n_rows * sizeof(int));
    cudaMalloc(&d_loss, n_rows * sizeof(float));
    cudaMalloc(&d_logsumexp_chunks, n_rows * num_chunks * sizeof(float));

    // Initialize loss to zeros
    cudaMemset(d_loss, 0, n_rows * sizeof(float));
    
    // Copy input data to device
    cudaMemcpy(d_logits, h_logits, n_rows * vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, n_rows * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate shared memory size
    int shared_memory_size = THREADS_PER_BLOCK * sizeof(float);

    // Launch kernel
    dim3 grid_dim(n_rows, num_chunks);  // Launch one block per (row, chunk)
    CrossEntropyForward_Kernel<<<grid_dim, THREADS_PER_BLOCK, shared_memory_size>>>(
        d_logits, d_labels, d_loss, d_logsumexp_chunks, n_rows, vocab_size, num_chunks
    );
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(error) << std::endl;
    }
    
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_loss, d_loss, n_rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_logits);
    cudaFree(d_labels);
    cudaFree(d_loss);
    cudaFree(d_logsumexp_chunks);
}

int main() {
    int n_rows = 2, vocab_size = 256000; // Large vocab size
    float *h_logits = new float[n_rows * vocab_size];
    int *h_labels = new int[n_rows];
    float *h_loss = new float[n_rows];

    // Initialize random seed
    srand(42);  // Fixed seed for reproducibility

    // Initialize logits with random values
    for (int i = 0; i < n_rows * vocab_size; i++) {
        h_logits[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
    h_labels[0] = 150000; // Example labels
    h_labels[1] = 200000;

    // Compute CPU reference result (for comparison)
    float *cpu_loss = new float[n_rows];
    for (int row = 0; row < n_rows; row++) {
        int label = h_labels[row];
        if (label < 0 || label >= vocab_size) {
            cpu_loss[row] = 0.0f;
            continue;
        }

        // Find max for numerical stability
        float max_val = -INFINITY;
        for (int i = 0; i < vocab_size; i++) {
            max_val = std::max(max_val, h_logits[row * vocab_size + i]);
        }

        // Compute sum of exp(logits - max)
        double sum_exp = 0.0;
        for (int i = 0; i < vocab_size; i++) {
            sum_exp += std::exp(h_logits[row * vocab_size + i] - max_val);
        }

        // Compute logsumexp
        float logsumexp = max_val + std::log(sum_exp);
        
        // Compute loss
        cpu_loss[row] = logsumexp - h_logits[row * vocab_size + label];
    }

    // Run CUDA implementation
    compute_cross_entropy_forward(h_logits, h_labels, h_loss, n_rows, vocab_size);

    // Print and compare results
    std::cout << "Cross-Entropy Loss Results:" << std::endl;
    std::cout << "----------------------------" << std::endl;
    for (int i = 0; i < n_rows; i++) {
        std::cout << "Row " << i << " - GPU: " << h_loss[i] << ", CPU: " << cpu_loss[i] 
                  << ", Diff: " << std::abs(h_loss[i] - cpu_loss[i]) << std::endl;
    }

    // Clean up
    delete[] h_logits;
    delete[] h_labels;
    delete[] h_loss;
    delete[] cpu_loss;
    
    return 0;
}