#include <cuda_runtime.h>
#include <iostream>

#define N 1024
#define THREADS_PER_BLOCK 256
#define SEGMENT_SIZE THREADS_PER_BLOCK

__global__ void KoggeStoneScan(float *input, float *output) {
    __shared__ float input_ds[SEGMENT_SIZE];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;


    // Load into shared memory
    if (i < N) {
        input_ds[threadIdx.x] = input[i];
    } else {
        input_ds[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Parallel reduce method
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp = 0.0f;

        if (threadIdx.x >= stride) { 
            temp = input_ds[threadIdx.x] + input_ds[threadIdx.x - stride];
        }

        __syncthreads();

        if (threadIdx.x >= stride) {
            input_ds[threadIdx.x] = temp;
        }
    }
    __syncthreads();

    // Ultimately write into the output
    if (i < N) {
        output[i] = input_ds[threadIdx.x];
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
    KoggeStoneScan<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Prefix Sum (Partial Output):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "h_output[" << i << "] = " << h_output[i] << std::endl;
    }


    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
