#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float* __restrict__ A,
const float* __restrict__ B, float* __restrict__ C, size_t size, size_t K){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z  + threadIdx.z;

    if(row < size && col < size && d < size){
        float sum = 0.0f;
        int radius = (K-1)/2;

        for(int kr = 0; kr < K ; ++kr){
            for(int kc = 0; kc < K; ++ kc){
                for(int kd = 0; kd < K; ++kd){
                    int curr_row = row + kr - radius;
                    int curr_col = col + kc - radius;
                    int curr_d = d + kd - radius;

                    if(curr_row >= 0 && curr_row < size && curr_col >= 0 && curr_col < size && curr_d >= 0 && curr_d < size){
                        sum += A[curr_row*size * size + curr_col * size + curr_d] * B[kr * K * K + kc * K + kd];
                    }
                }
            }
        }
        C[row * size * size + col * size + d] = sum;
    }
}

// Note: A, B, C are all device pointers to float32 arrays
extern "C" void solution(const float* A, const float* B, float* C, size_t size, size_t K) {    
    dim3 blockDim(8,8,4);
    dim3 gridDim((size+blockDim.x - 1)/ blockDim.x, (size + blockDim.y - 1)/ blockDim.y, (size + blockDim.z -1)/ blockDim.z);

    conv3d_kernel<<<gridDim, blockDim>>>(A, B, C, size, K);
    cudaDeviceSynchronize();
}