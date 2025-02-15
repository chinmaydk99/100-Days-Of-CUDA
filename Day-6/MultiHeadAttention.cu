#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>

#define TILE_SIZE 16
#define WARP_SIZE 32

__device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}
__global__ void multiHeadAttentionOptimized(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ Wq,
    const float* __restrict__ Wk,
    const float* __restrict__ Wv,
    float* __restrict__ output,
    int seq_len,
    int d_model,
    int num_heads
) {
    int head = blockIdx.x; // Which attention head
    int query = blockIdx.y; // Which query token
    int d_head = d_model / num_heads;
    int tid = threadIdx.x;
    float sqrt_d = sqrtf((float)d_head);

    // Shared memory tiles
    extern __shared__ float shared_mem[]; // Dynamically allocated
    float* Q_tile = shared_mem; // [d_head]
    float* K_tile = shared_mem + d_head; // [seq_len]
    float* V_tile = shared_mem + d_head + seq_len; // [seq_len]

    // **Step 1: Compute Q_proj, K_proj, V_proj using weight matrices**
    float Q_proj[TILE_SIZE], K_proj[TILE_SIZE], V_proj[TILE_SIZE];

    for (int k = 0; k < d_head; k++) {
        Q_proj[k] = 0.0f;
        K_proj[k] = 0.0f;
        V_proj[k] = 0.0f;
        for (int j = 0; j < d_model; j++) {
            Q_proj[k] += Q[query * d_model + j] * Wq[j * d_model + head * d_head + k];
            K_proj[k] += K[tid * d_model + j] * Wk[j * d_model + head * d_head + k];
            V_proj[k] += V[tid * d_model + j] * Wv[j * d_model + head * d_head + k];
        }
    }
    __syncthreads();

    // **Step 2: Compute dot product Q * Kᵀ and store in shared memory**
    float scores[TILE_SIZE] = {0.0f};
    for (int k = 0; k < seq_len; k += TILE_SIZE) {
        // Load a tile of K into shared memory
        for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
            if (k + i < seq_len)
                K_tile[i] = K_proj[i];
        }
        __syncthreads();

        // Compute dot product for this tile
        for (int i = 0; i < TILE_SIZE; i++) {
            scores[k + i] += Q_proj[i] * K_tile[i];
        }
        __syncthreads();
    }

    // Scale scores
    for (int i = 0; i < seq_len; i++) {
        scores[i] /= sqrt_d;
    }
    __syncthreads();

    // **Step 3: Warp reduction for softmax max computation**
    float max_score = scores[0];
    for (int i = 1; i < seq_len; i++) {
        max_score = fmaxf(max_score, scores[i]);
    }
    max_score = warpReduceSum(max_score);
    __syncthreads();

    // Compute softmax
    float sum_exp = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        scores[i] = expf(scores[i] - max_score);
        sum_exp += scores[i];
    }
    sum_exp = warpReduceSum(sum_exp);
    __syncthreads();

    for (int i = 0; i < seq_len; i++) {
        scores[i] /= sum_exp;
    }
    __syncthreads();

    // **Step 4: Compute the output = softmax(QKᵀ) * V**
    float output_local[TILE_SIZE] = {0.0f};
    for (int k = 0; k < seq_len; k += TILE_SIZE) {
        // Load tile of V into shared memory
        for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
            if (k + i < seq_len)
                V_tile[i] = V_proj[i];
        }
        __syncthreads();

        // Compute weighted sum
        for (int i = 0; i < TILE_SIZE; i++) {
            output_local[i] += scores[k + i] * V_tile[i];
        }
        __syncthreads();
    }

    // Store final output
    for (int i = tid; i < d_head; i += blockDim.x) {
        output[query * d_model + head * d_head + i] = output_local[i];
    }
}
