#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while (0)


__device__ int sturm_sequence_count(
    float lambda,
    const float* __restrict__ d,
    const float* __restrict__ e,
    int n
) {
    int sign_changes = 0;
    float p_prev = 1.0f;
    float p_curr = d[0] - lambda;

    if (p_curr < 0.0f) {
        sign_changes++;
    } else if (p_curr == 0.0f) {
         p_curr = -1.0e-30f;
         sign_changes++;
    }

    for (int i = 1; i < n; ++i) {
        float e_sq = e[i] * e[i];
        float p_next = (d[i] - lambda) * p_curr - e_sq * p_prev;

        if ((p_next > 0.0f && p_curr < 0.0f) || (p_next < 0.0f && p_curr > 0.0f)) {
            sign_changes++;
        } else if (p_next == 0.0f) {
            p_next = -1.0e-30f;
            sign_changes++;
        }

        p_prev = p_curr;
        p_curr = p_next;
    }

    return sign_changes;
}


__global__ void compute_eigenvalue_bisection_kernel(
    const float* __restrict__ d,
    const float* __restrict__ e,
    float* __restrict__ eigenvalues,
    const float* __restrict__ left_bounds,
    const float* __restrict__ right_bounds,
    const int* __restrict__ eigenvalue_indices,
    int n,
    float tolerance,
    int max_iterations
) {
    int block_id = blockIdx.x;
    int k = eigenvalue_indices[block_id];
    float left = left_bounds[block_id];
    float right = right_bounds[block_id];

    for (int iter = 0; iter < max_iterations; ++iter) {
        if (fabsf(right - left) <= tolerance) {
            break;
        }
        float mid = left + (right - left) * 0.5f;
        int count = sturm_sequence_count(mid, d, e, n);

        if (count <= k) {
            left = mid;
        } else {
            right = mid;
        }
    }

    eigenvalues[block_id] = left + (right - left) * 0.5f;
}


extern "C" void solution(
    const float* d_d,
    const float* d_e,
    float* d_eigenvalues,
    const float* d_left_bounds,
    const float* d_right_bounds,
    const int* d_eigenvalue_indices,
    int n,
    int nev,
    float tolerance,
    int max_iterations
) {

    dim3 blockDim(1);
    dim3 gridDim(nev);

    compute_eigenvalue_bisection_kernel<<<gridDim, blockDim>>>(
        d_d,
        d_e,
        d_eigenvalues,
        d_left_bounds,
        d_right_bounds,
        d_eigenvalue_indices,
        n,
        tolerance,
        max_iterations
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}