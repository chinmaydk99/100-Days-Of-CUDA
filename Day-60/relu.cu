#include <cuda_runtime.h>
#include <cmath>

__global__ void relu(const float* __restrict__ input, 
float* __restrict__ output, size_t n , size_t m){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i < m && j < n){
        output[i*n + j] = fmaxf(0.0f, input[i*n + j]);
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    dim3 blockDim(16,16);
    dim3 gridDim((m + blockDim.x - 1)/blockDim.x, (n + blockDim.y - 1)/blockDim.y);

    relu<<<gridDim, blockDim>>>(input, output, n , m);
    cudaDeviceSynchronize();

}