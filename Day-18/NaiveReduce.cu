#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

__global__ void SumReducedKernel(float *input, float *output, int N){
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x*2;
    __shared__ float input_ds[BLOCK_SIZE];

    if(i < N){
        input_ds[threadIdx.x] = input[i];
    }else{
        input_ds[threadIdx.x] = 0.0f;
    }

    if(i + blockDim.x < N){
        input_ds[threadIdx.x] += input[i + blockDim.x];
    }
    __syncthreads();

    for(unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(threadIdx.x < stride){
            input_ds[threadIdx.x] += input_ds[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[blockIdx.x] = input_ds[0];
    }
}

int main() {
    int N = 1 << 20; // 1M elements
    float *h_input, *h_output;
    float *d_input, *d_output;

    int gridSize = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2); // ceil(N/ 2* BLOCK_SIZE)

    h_input = (float*) malloc(N * sizeof(float));
    h_output = (float*) malloc(gridSize * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, gridSize * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    SumReducedKernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "First partial sum: " << h_output[0] << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}