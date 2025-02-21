#include <iostream>
#include <cuda_runtime.h>

#define N 1024  
#define THREADS_PER_BLOCK 256  
#define SECTION_SIZE (2 * THREADS_PER_BLOCK) 

__global__ void BlockScanKernel(float *input, float *output, float *block_sums, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];

    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    XY[threadIdx.x] = (i < N) ? input[i] : 0.0f;
    XY[threadIdx.x + blockDim.x] = (i + blockDim.x < N) ? input[i + blockDim.x] : 0.0f;

    __syncthreads();

    for (unsigned int stride = 1; stride < SECTION_SIZE; stride *= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < SECTION_SIZE) {
            XY[index] += XY[index - stride];
        }
    }

    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();

    if (i < N) output[i] = XY[threadIdx.x];
    if (i + blockDim.x < N) output[i + blockDim.x] = XY[threadIdx.x + blockDim.x];

    if (threadIdx.x == 0 && block_sums != nullptr) {
        block_sums[blockIdx.x] = XY[SECTION_SIZE - 1];
    }
}

__global__ void BlockSumsScanKernel(float *block_sums, unsigned int num_blocks) {
    __shared__ float XY[SECTION_SIZE];  // Shared memory buffer for block sums

    unsigned int i = threadIdx.x;

    XY[i] = (i < num_blocks) ? block_sums[i] : 0.0f;
    __syncthreads();

    for (unsigned int stride = 1; stride < num_blocks; stride *= 2) {
        __syncthreads();
        if (i >= stride) {
            XY[i] += XY[i - stride];
        }
    }

    __syncthreads();

    if (i < num_blocks) {
        block_sums[i] = XY[i];
    }
}

__global__ void AddBlockSumsKernel(float *output, float *block_sums, unsigned int N) {
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    float block_offset = (blockIdx.x > 0) ? block_sums[blockIdx.x - 1] : 0.0f;

    if (i < N) {
        output[i] += block_offset;
    }
    if (i + blockDim.x < N) {
        output[i + blockDim.x] += block_offset;
    }
}

int main() {
    float h_input[N], h_output[N]; 
    float *d_input, *d_output, *d_block_sums; 

    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));
    int num_blocks = (N + SECTION_SIZE - 1) / SECTION_SIZE;
    cudaMalloc((void**)&d_block_sums, num_blocks * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);


    BlockScanKernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_output, d_block_sums, N);

    BlockSumsScanKernel<<<1, num_blocks>>>(d_block_sums, num_blocks);

    AddBlockSumsKernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_output, d_block_sums, N);


    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Prefix Sum (First 10 Elements):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "h_output[" << i << "] = " << h_output[i] << std::endl;
    }

    if (h_output[N - 1] == N) {
        std::cout << "Segmented Scan Completed Successfully!" << std::endl;
    } else {
        std::cout << "Error in Computation. Expected " << N << " but got " << h_output[N - 1] << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_block_sums);

    return 0;
}
