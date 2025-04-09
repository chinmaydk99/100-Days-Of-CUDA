#include <cuda_runtime.h>
#include <cmath>

__global__ void conv1d_kernel(const float* __restrict__ A, 
const float* __restrict__ B, float* __restrict__ C, size_t N, size_t K){
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    int radius = (K-1)/2;

    if(index < N){
        float sum = 0.0f;

        #pragma unroll
        for(int k= 0; k < K; ++k){
            int input_index = index + k - radius;
            
            if(input_index >= 0 && input_index < N){
                sum += A[input_index] * B[k];
            }
        }
        C[index] = sum;  
    }
}

// Note: A, B, C are all device pointers to float32 arrays
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    dim3 blockDim(256);
    dim3 gridDim(N+255/256);
    conv1d_kernel<<<gridDim, blockDim>>>(A, B, C, N, K);

    cudaDeviceSynchronize();
}