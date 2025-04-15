#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while (0)

__device__ float device_op_identity(float x) {
    return x;
}

__device__ float device_op_square(float x) {
    return x * x;
}

__device__ float device_op_cube(float x) {
    return x * x * x;
}

typedef float (*DeviceProcessFunc)(float);

__constant__ DeviceProcessFunc device_func_table[3] = {
    device_op_identity,
    device_op_square,
    device_op_cube
};
const int NUM_DEVICE_FUNCS = 3;

__global__ void apply_func_ptr_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ op_indices,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    while (idx < n) {
        int op_idx = op_indices[idx];

        if (op_idx >= 0 && op_idx < NUM_DEVICE_FUNCS) {
            DeviceProcessFunc func_ptr = device_func_table[op_idx];
            output[idx] = func_ptr(input[idx]);
        } else {
            output[idx] = input[idx];
        }

        idx += stride;
    }
}

extern "C" void solution(
    const float* d_input,
    float* d_output,
    const int* d_op_indices,
    int n
) {
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    blocks_per_grid = max(1, blocks_per_grid);

    apply_func_ptr_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_input,
        d_output,
        d_op_indices,
        n
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}