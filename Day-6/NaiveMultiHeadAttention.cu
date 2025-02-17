#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define MAX_SEQ_LEN 256

__global__ void multiHeadAttention(
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
){
    int head = blockIdx.x;
    int query = blockIdx.y;

    int d_head = d_model / num_heads;
    
    int head_offset = head*d_head;

    int threadId = threadIdx.x;

    __shared__ float scores[MAX_SEQ_LEN];

    float sqrt_d = sqrtf((float)d_head);

    // Step 1: Calculating the projections
    float Q_proj[MAX_SEQ_LEN], K_proj[MAX_SEQ_LEN], V_proj[MAX_SEQ_LEN];

    for (int k = 0; k < d_head; k++) {
        Q_proj[k] = 0.0f;
        K_proj[k] = 0.0f;
        V_proj[k] = 0.0f;

        for (int j = 0; j < d_model; j++) {
            Q_proj[k] += Q[threadId * d_model + j] * Wq[j * d_model + head_offset + k];
            K_proj[k] += K[threadId * d_model + j] * Wk[j * d_model + head_offset + k];
            V_proj[k] += V[threadId * d_model + j] * Wv[j * d_model + head_offset + k];
        }
    }
    __syncthreads();

    // Step 2: Computing the dot product between Q and K tranpose
    if(threadId < seq_len){
        
    }

}
int main(){
    int seq_len = 4;
    int d_model = 8;
    int num_heads = 2;
    int d_head = d_model / num_heads;
    int size = seq_len * d_model;

    vector<float> Q_h(size, 0.01f);
    vector<float> K_h(size, 0.02f);
    vector<float> V_h(size, 0.03f);
    vector<float> Wq_h(size, 0.1f);
    vector<float> Wk_h(size, 0.1f);
    vector<float> Wv_h(size, 0.1f);
    vector<float> output_h(size, 0.0f);

    float *Q_d, *K_d, *V_d, *Wq_d, *Wk_d, *Wv_d, *output_d;
    cudaMalloc(&Q_d, size * sizeof(float));
    cudaMalloc(&K_d, size * sizeof(float));
    cudaMalloc(&V_d, size * sizeof(float));
    cudaMalloc(&Wq_d, size * sizeof(float));
    cudaMalloc(&Wk_d, size * sizeof(float));
    cudaMalloc(&Wv_d, size * sizeof(float));
    cudaMalloc(&output_d, size * sizeof(float));

    cudaMemcpy(Q_d, Q_h.data(), size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(K_d, K_h.data(), size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V_h.data(), size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Wq_d, Wq_h.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Wk_d, Wk_h.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Wv_d, Wv_h.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridSize(num_heads, seq_len);
    dim3 blockSize(seq_len);
    multiHeadAttention<<<gridSize, blockSize>>>(Q_d, K_d, V_d, Wq_d, Wk_d, Wv_d, output_d, seq_len, d_model, num_heads);
    cudaDeviceSynchronize();

    cudaMemcpy(output_h.data(), output_d, size*sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Multi-Head Attention Output (CUDA Naïve):\n";
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            cout << output_h[i * d_model + j] << " ";
        }
        cout << "\n";
    }

    cudaFree(Q_d);
    cudaFree(K_d);
    cudaFree(V_d);
    cudaFree(Wq_d);
    cudaFree(Wk_d);
    cudaFree(Wv_d);
    cudaFree(output_d);
    
    return 0;

}   