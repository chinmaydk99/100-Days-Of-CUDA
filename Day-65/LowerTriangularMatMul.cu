#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while (0)

__global__ void lower_triangular_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        if (i >= j) {
            float sum = 0.0f;
            for (int k = j; k <= i; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

extern "C" void solution(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int N
) {
    CUDA_CHECK(cudaMemset(d_C, 0, (size_t)N * N * sizeof(float)));

    dim3 blockDim(16, 16);
    dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x,
        (N + blockDim.y - 1) / blockDim.y
    );

    lower_triangular_matmul_kernel<<<gridDim, blockDim>>>(
        d_A,
        d_B,
        d_C,
        N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}