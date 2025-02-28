#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024
#define WARP_SIZE 32

__global__ void WarpSumReducedKernel(float *input, float *output, int N){
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

    for(unsigned int stride = blockDim.x / 2; stride > WARP_SIZE; stride /= 2){
        if(threadIdx.x < stride){
            input_ds[threadIdx.x] += input_ds[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float sum = 0.0f;
    // Loading value from next warp into current warp. We do this we will have only 32 threads active in the warp.
    if(threadIdx.x < WARP_SIZE){
        sum += input_ds[threadIdx.x] + input_ds[threadIdx.x + WARP_SIZE];
    }

    // Now using shuffle down sync to get final value by reading from and writing to register
    for(unsigned int stride = WARP_SIZE; stride > 0; stride /= 2){
        sum += __shfl_down_sync(0xffffffff, sum, stride);
    }

    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
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

    WarpSumReducedKernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "First partial sum: " << h_output[0] << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}