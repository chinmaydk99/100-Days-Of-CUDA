#include <cuda_runtime.h>
#include <cmath>

__global__ void hinge_loss(const float* __restrict__ predictions, 
const float* __restrict__ targets,float* output, size_t n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < n){
        output[i] = fmaxf(0, 1 - predictions[i]*targets[i]);
    }
}

// Note: predictions, targets, output are all device pointers to float32 arrays
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {    
    dim3 blockDim(256);
    dim3 gridDim((n+blockDim.x-1)/blockDim.x);

    hinge_loss<<<gridDim, blockDim>>>(predictions, targets, output, n);
    cudaDeviceSynchronize();
}