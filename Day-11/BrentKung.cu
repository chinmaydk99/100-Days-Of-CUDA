#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Total number of elements
#define THREADS_PER_BLOCK 256  // Number of threads per block
#define SECTION_SIZE (2 * THREADS_PER_BLOCK)

__global__ void BrentKungScanKernel(float *input, float *output){
    __shared__ float XY[SECTION_SIZE];

    unsigned int i = 2*blockIdx.x*blockDim.x + threadIdx.x; // 2 elements per thread

    if(i < N){
        XY[threadIdx.x] = input[i];
    }else{
        XY[threadIdx.x] = 0.0f;
    }

    // Loading the second element
    if(i+blockDim.x < N){
        XY[threadIdx.x + blockDim.x] = input[i+blockDim.x];
    }else{
        XY[threadIdx.x + blockDim.x] = 0.0f;
    }

    __syncthreads();

    // Up Sweep Phase
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * stride * 2 -1;
        if(index < SECTION_SIZE){
            XY[index] += XY[index - stride];
        }
    }

    // Down sweep: Sum distribution phase
    for(int stride = SECTION_SIZE/4; stride > 0; stride /= 2){
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * stride * 2 -1;
        if(index + stride < SECTION_SIZE){
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();

    // Adding the elements to output
    if(i < N){
        output[i] = XY[threadIdx.x];
    }

    if(i + blockDim.x < N){
        output[i+blockDim.x] = XY[threadIdx.x + blockDim.x];
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

    int numBlocks = (N + SECTION_SIZE - 1) / SECTION_SIZE;
    BrentKungScanKernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Prefix Sum (First 10 Elements):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "h_output[" << i << "] = " << h_output[i] << std::endl;
    }

    if (h_output[N - 1] == N) {
        std::cout << "Prefix Sum Computed Successfully!" << std::endl;
    } else {
        std::cout << "Error in Computation. Expected " << N << " but got " << h_output[N - 1] << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
