#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define THREADS_PER_BLOCK 32 

__global__ void LogSumExp_Kernel(
    const float* __restrict__ logits,
    float* __restrict__ logsumexp,
    int n_rows,
    int vocab_size    
){
    int row = blockIdx.x; // Each Block processes one row of logits
    int tid = threadIdx.x;

    extern __shared__ float shared_data[];

    // Offset to start of current row
    const float* row_logits = logits + row*vocab_size; // Because logits is of shape [nrows, vocab_size]


    // Computing Max Logit for each row

    // Stride of blockDim.x ensures coalesced memory acces
    float local_max = -INFINITY;
    for(int i = tid; i < vocab_size; i += blockDim.x){
        local_max = fmaxf(local_max, row_logits[i]);
    }

    shared_data[tid] = local_max;
    __syncthreads();

    float max_val = 0.0f;
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tid < stride){
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid+stride]);
        }
        __syncthreads();
    }
    max_val = shared_data[0]; // Final Max Value of the row after reduction

    // Computing Sum of Exp(Logits - Max)
    float sum_exp = 0.0f;
    for(int i = tid; i < vocab_size; i += blockDim.x){
        sum_exp += expf(row_logits[i] - max_val);
    }

    shared_data[tid] = sum_exp;
    __syncthreads();

    for(int stride = blockDim.x/2 ; stride > 0; stride /= 2){
        if(tid < stride){
            shared_data[tid] += shared_data[tid+stride];
        }
        __syncthreads();
    }

    sum_exp = shared_data[0];

    if(tid == 0){
        logsumexp[row] = max_val + logf(sum_exp);
    }
}

void compute_logsumexp(float *h_logits, float *h_logsumexp, int nrows, int vocab_size){
    float *d_logits, *d_logsumexp;

    cudaMalloc(&d_logits, nrows*vocab_size*sizeof(float));
    cudaMalloc(&d_logsumexp, nrows*sizeof(float));

    cudaMemcpy(d_logits, h_logits, nrows*vocab_size*sizeof(float), cudaMemcpyHostToDevice);
    
    int shared_memory_size = THREADS_PER_BLOCK*sizeof(float); 

    LogSumExp_Kernel<<<nrows, THREADS_PER_BLOCK, shared_memory_size>>>(d_logits, d_logsumexp, nrows, vocab_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_logsumexp, d_logsumexp, nrows*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_logits);
    cudaFree(d_logsumexp);
}

int main() {
    int n_rows = 2, vocab_size = 10; 
    float h_logits[] = {
        2.1, 1.0, 0.5, -0.5, 1.2, 0.8, 0.2, 0.3, -1.0, 0.7,
        1.8, 2.4, -0.2, 1.1, 0.9, 0.5, -0.3, 0.6, 0.0, 1.0
    };
    float h_logsumexp[2];

    compute_logsumexp(h_logits, h_logsumexp, n_rows, vocab_size);

 
    for (int i = 0; i < n_rows; i++) {
        printf("Log-Sum-Exp for row %d: %f\n", i, h_logsumexp[i]);
    }

    return 0;
}