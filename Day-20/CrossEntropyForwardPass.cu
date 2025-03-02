#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define THREADS_PER_BLOCK 32

__global__ void CrossEntropyForward_Kernel(
    const float* __restrict__ logits, //nrows, vocab_size
    const int* __restrict__ labels, //nrows
    float* loss, //nrows
    int nrows,
    int vocab_size
){
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* row_logits = logits + row*vocab_size;

    extern __shared__ float shared_data[];

    float local_max = -INFINITY;
    for(int i = tid; i < vocab_size; i += blockDim.x){
        local_max = fmaxf(local_max, row_logits[i]);
    }

    shared_data[tid] = local_max;
    __syncthreads();

    for(int stride= blockDim.x/2 ; stride > 0; stride /= 2){
        if(tid < stride){
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid+stride]);
        }
        __syncthreads();
    }

    float max_val = shared_data[0];

    float sum_exp = 0.0f;
    for(int i = tid; i < vocab_size; i += blockDim.x){
        sum_exp += expf(row_logits[i]-max_val);
    }

    shared_data[tid] = sum_exp;
    __syncthreads();

    for(int stride= blockDim.x/2 ; stride > 0; stride /= 2){
        if(tid < stride){
            shared_data[tid] += shared_data[tid+stride];
        }
        __syncthreads();
    }

    sum_exp = shared_data[0];

    float logsumexp = max_val + logf(sum_exp);

    __shared__ float correct_logit;
    correct_logit = 0.0f;
    __syncthreads();

    if(tid == 0){
        int label = labels[row];
        correct_logit = (label >= 0 && label < vocab_size) ? row_logits[label] : 0.0f;
    }
    __syncthreads();

    if (tid == 0) {
        loss[row] = (labels[row] >= 0) ? (logsumexp - correct_logit) : 0.0f;
    }
}

void compute_cross_entropy_forward(float *h_logits, int *h_labels, float *h_loss, int n_rows, int vocab_size){
    float *d_logits, *d_loss;
    int *d_labels;

    cudaMalloc(&d_logits, n_rows * vocab_size * sizeof(float));
    cudaMalloc(&d_labels, n_rows * sizeof(int));
    cudaMalloc(&d_loss, n_rows * sizeof(float));

    cudaMemcpy(d_logits, h_logits, n_rows * vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, n_rows * sizeof(int), cudaMemcpyHostToDevice);
    
    int shared_memory_size = (THREADS_PER_BLOCK+1) * sizeof(float);

    CrossEntropyForward_Kernel<<<n_rows, THREADS_PER_BLOCK, shared_memory_size>>>(d_logits, d_labels, d_loss, n_rows, vocab_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_loss, d_loss, n_rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_logits);
    cudaFree(d_labels);
    cudaFree(d_loss);
}

int main() {
    int n_rows = 2, vocab_size = 10; 
    float h_logits[] = {
        2.1, 1.0, 0.5, -0.5, 1.2, 0.8, 0.2, 0.3, -1.0, 0.7,
        1.8, 2.4, -0.2, 1.1, 0.9, 0.5, -0.3, 0.6, 0.0, 1.0
    };
    int h_labels[] = {2, 5}; // Example labels (correct class index)
    float h_loss[2];

    compute_cross_entropy_forward(h_logits, h_labels, h_loss, n_rows, vocab_size);

    for (int i = 0; i < n_rows; i++) {
        printf("Cross-Entropy Loss for row %d: %f\n", i, h_loss[i]);
    }

    return 0;
}