#include <cuda_runtime.h>
#include <iostream>

#define N 1024
#define THREADS_PER_BLOCK 256
#define SEGMENT_SIZE THREADS_PER_BLOCK

__global__ void KoggeStoneScanDB(float *input, float *output){
    __shared__ float buffer_A[SEGMENT_SIZE];
    __shared__ float buffer_B[SEGMENT_SIZE];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){
        buffer_A[threadIdx.x] = input[i];
    }else{
        buffer_A[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        if(threadIdx.x >= stride){
            buffer_B[threadIdx.x] = buffer_A[threadIdx.x] + buffer_A[threadIdx.x - stride]; 
        }else{
            buffer_B[threadIdx.x] = buffer_A[threadIdx.x];
        }

        __syncthreads();

        // Swapping the buffers
        if(threadIdx.x < SEGMENT_SIZE){
            buffer_A[threadIdx.x] = buffer_B[threadIdx.x];
        }
    }

    if(i < N){
        output[i] = buffer_A[threadIdx.x];
    }
}

int main() {
    float h_input[N], h_output[N];
    float *d_input, *d_output;


    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;  
    }

    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    KoggeStoneScanDB<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Prefix Sum (Partial Output):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "h_output[" << i << "] = " << h_output[i] << std::endl;
    }


    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
