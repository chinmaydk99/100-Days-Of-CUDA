#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>

#define TILE_SIZE 16
#define WARP_SIZE 32

// Warp reduction for softmax sum
__device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Optimized Multi-Head Attention Kernel
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
    int head = blockIdx.x; 
    int query = blockIdx.y; 
    int d_head = d_model / num_heads;
    int tid = threadIdx.x;
    float sqrt_d = sqrtf((float)d_head);

    // Shared memory tiles
    extern __shared__ float shared_mem[];
    float* Q_tile = shared_mem; 
    float* K_tile = shared_mem + d_head;
    float* V_tile = shared_mem + d_head + seq_len;

    // Compute Q_proj, K_proj, V_proj
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

    // Compute dot product Q * Kᵀ
    float scores[TILE_SIZE] = {0.0f};
    for (int k = 0; k < seq_len; k += TILE_SIZE) {
        for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
            if (k + i < seq_len)
                K_tile[i] = K_proj[i];
        }
        __syncthreads();

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

    // Warp reduction for softmax
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

    // Compute output = softmax(QKᵀ) * V
    float output_local[TILE_SIZE] = {0.0f};
    for (int k = 0; k < seq_len; k += TILE_SIZE) {
        for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
            if (k + i < seq_len)
                V_tile[i] = V_proj[i];
        }
        __syncthreads();

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

int main() {
    int seq_len = 4;
    int d_model = 8;
    int num_heads = 2;
    int d_head = d_model / num_heads;
    int size = seq_len * d_model;

    // Initialize Q, K, V, and weight matrices
    std::vector<float> h_Q(size, 0.01f);
    std::vector<float> h_K(size, 0.02f);
    std::vector<float> h_V(size, 0.03f);
    std::vector<float> h_Wq(size, 0.1f);
    std::vector<float> h_Wk(size, 0.1f);
    std::vector<float> h_Wv(size, 0.1f);
    std::vector<float> h_output(size, 0.0f);

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_Wq, *d_Wk, *d_Wv, *d_output;
    cudaMalloc(&d_Q, size * sizeof(float));
    cudaMalloc(&d_K, size * sizeof(float));
    cudaMalloc(&d_V, size * sizeof(float));
    cudaMalloc(&d_Wq, size * sizeof(float));
    cudaMalloc(&d_Wk, size * sizeof(float));
    cudaMalloc(&d_Wv, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_Q, h_Q.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wq, h_Wq.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wk, h_Wk.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wv, h_Wv.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch
    dim3 grid(num_heads, seq_len);
    dim3 block(seq_len);
    multiHeadAttentionOptimized<<<grid, block>>>(d_Q, d_K, d_V, d_Wq, d_Wk, d_Wv, d_output, seq_len, d_model, num_heads);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Multi-Head Attention Output (CUDA Optimized):\n";
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            std::cout << h_output[i * d_model + j] << " ";
        }
        std::cout << "\n";
    }

    // Free memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_Wq);
    cudaFree(d_Wk);
    cudaFree(d_Wv);
    cudaFree(d_output);

    return 0;
}
