#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while (0)


__global__ void histogram_2d_atomic_kernel(
    const float* __restrict__ input_x,
    const float* __restrict__ input_y,
    unsigned int* __restrict__ histogram_2d,
    int n,
    int num_bins_x,
    int num_bins_y,
    float min_val_x,
    float max_val_x,
    float min_val_y,
    float max_val_y
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float range_x = max_val_x - min_val_x;
    float range_y = max_val_y - min_val_y;

    float bin_width_inv_x = (range_x > 1e-9f) ? (float)num_bins_x / range_x : 0.0f;
    float bin_width_inv_y = (range_y > 1e-9f) ? (float)num_bins_y / range_y : 0.0f;

    while (idx < n) {
        float value_x = input_x[idx];
        float value_y = input_y[idx];

        int bin_index_x = -1;
        int bin_index_y = -1;

        if (value_x >= min_val_x && value_x < max_val_x) {
            bin_index_x = static_cast<int>(floorf((value_x - min_val_x) * bin_width_inv_x));
            bin_index_x = max(0, min(num_bins_x - 1, bin_index_x));
        }

        if (value_y >= min_val_y && value_y < max_val_y) {
            bin_index_y = static_cast<int>(floorf((value_y - min_val_y) * bin_width_inv_y));
            bin_index_y = max(0, min(num_bins_y - 1, bin_index_y));
        }

        if (bin_index_x >= 0 && bin_index_y >= 0) {
            int linear_bin_index = bin_index_y * num_bins_x + bin_index_x;
            atomicAdd(&histogram_2d[linear_bin_index], 1);
        }

        idx += stride;
    }
}


extern "C" void solution(
    const float* d_input_x,
    const float* d_input_y,
    unsigned int* d_histogram_2d,
    int n,
    int num_bins_x,
    int num_bins_y,
    float min_val_x,
    float max_val_x,
    float min_val_y,
    float max_val_y
) {

    long long total_bins = (long long)num_bins_x * num_bins_y;
    CUDA_CHECK(cudaMemset(d_histogram_2d, 0, total_bins * sizeof(unsigned int)));

    int threads_per_block = 256;
    int min_blocks = 1024;
    int blocks_per_grid = min(min_blocks, (n + threads_per_block - 1) / threads_per_block);
    blocks_per_grid = max(1, blocks_per_grid);

    histogram_2d_atomic_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_input_x,
        d_input_y,
        d_histogram_2d,
        n,
        num_bins_x,
        num_bins_y,
        min_val_x,
        max_val_x,
        min_val_y,
        max_val_y
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}