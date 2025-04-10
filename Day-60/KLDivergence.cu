#include <cuda_runtime.h>

__global__ void kldivergence(const float* __restrict__ predictions,
const float* __restrict__ targets, float* output, size_t n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < n){
        float sum = 0.0f;
        sum = targets[idx] * (__logf(targets[idx]) - __logf(predictions[idx]));

        output[idx] = sum;
    }
}

// Note: predictions, targets, output are all device pointers to float32 arrays
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {    
    dim3 blockDim(256);
    dim3 gridDim((n+blockDim.x-1)/ blockDim.x);

    kldivergence<<<gridDim, blockDim>>>(predictions, targets, output, n);
    cudaDeviceSynchronize();
}