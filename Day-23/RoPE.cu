#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <algorithm>

#define ROPE_GROUP_SIZE 4
#define MAX_SEQ_LEN 1024
#define MAX_HEAD_DIM 64
#define MAX_CACHE_SIZE (MAX_HEAD_DIM*MAX_SEQ_LEN)

__constant__ float const_cos_cache[MAX_CACHE_SIZE];
__constant__ float const_sin_cache[MAX_CACHE_SIZE];

__global__ void apply_rope_cuda(float *X, int n_rows, int num_heads, int head_dim, int seq_len){
    int rowIdx = blockIdx.x;
    int groupIdx = blockIdx.y;

    int half_dim = head_dim/2;
    int tid = threadIdx.x; // Handles one element in range [0,half_dim)

    if(tid < half_dim){
        int token_idx = rowIdx % seq_len; // rowIdx = batch_size * seq_len

        for(int groupOffset = 0; groupOffset < ROPE_GROUP_SIZE; groupOffset ++){
            int head = groupIdx*ROPE_GROUP_SIZE + groupOffset; // Navigating to current head within the group

            if(head < num_heads){
                int offset = rowIdx*(num_heads * head_dim) + head*head_dim;

                float x1 = X[offset + tid]; // Accessing the first half element
                float x2 = X[offset + tid + half_dim]; // Accessing the second half element
                
                // Getting sine and cosine for this 
                int cache_index = token_idx*half_dim + tid;
                float cos_val = const_cos_cache[cache_index];
                float sin_val = const_sin_cache[cache_index];

                // Applying the RoPE transformation
                float new_x1 = x1*cos_val - x2*sin_val;
                float new_x2 = x1*sin_val + x2*cos_val;

                X[offset + tid] = new_x1;
                X[offset + tid + half_dim] = new_x2;
            }
        }
    }
}

int main(){
    int batch = 2;
    int seq_len = 512;
    int n_heads = 8;
    int head_dim = 64;
    int half_dim = head_dim / 2;

    // Reshasping from [batch, seq_len, n_heads, head_dim] -> [batch*seq_len, n_heads*head_dim]
    // Idea taken from the unsloth implementation

    int n_rows = batch*seq_len;
    int total_elements = n_rows * n_heads * head_dim;
    size_t bytes = total_elements*sizeof(float);

    float* h_Q = new float[total_elements];
    float* h_V = new float[total_elements];
    for (int i = 0; i < total_elements; i++) {
        h_Q[i] = static_cast<float>(i % 10);
        h_V[i] = static_cast<float>((i + 5) % 10); 
    }

    size_t cache_bytes = seq_len * half_dim * sizeof(float);
    float *h_cos_cache = new float[seq_len *half_dim];
    float *h_sin_cache = new float[seq_len*half_dim];
    float base = 1000.0f;
    
    // Calculating sin and cos values sequentially since I don't see the benefit of parallelizing this 
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float exponent = static_cast<float>(i) / static_cast<float>(half_dim);
            float theta = powf(base, -exponent);
            float angle = pos * theta;
            h_cos_cache[pos * half_dim + i] = cosf(angle);
            h_sin_cache[pos * half_dim + i] = sinf(angle);
        }
    }

    // Storing this in constant memory
    cudaMemcpyToSymbol(const_cos_cache, h_cos_cache, cache_bytes);
    cudaMemcpyToSymbol(const_sin_cache, h_sin_cache, cache_bytes);

    float *d_Q, *d_V;
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice);

    // Grid
    /// Along x - one block per token(n_rows)
    /// Along y - one block per head group
    int n_groups = (n_heads + ROPE_GROUP_SIZE -1)/ ROPE_GROUP_SIZE;
    dim3 gridSize(n_rows, n_groups);
    dim3 blockSize(half_dim);

    apply_rope_cuda<<<gridSize, blockSize>>>(d_Q, n_rows, n_heads, head_dim,seq_len);
    cudaDeviceSynchronize();

    apply_rope_cuda<<<gridSize, blockSize>>>(d_V, n_rows, n_heads, head_dim,seq_len);
    cudaDeviceSynchronize();

    cudaMemcpy(h_Q, d_Q, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V, d_V, bytes, cudaMemcpyDeviceToHost);
    
    std::cout << "Verifying RoPE transformation for Q:" << std::endl;
    for (int row = 0; row < std::min(3, n_rows); row++) {  
        std::cout << "Token " << row << ": ";
        for (int i = 0; i < std::min(10, n_heads * head_dim); i++) {  
            std::cout << h_Q[row * (n_heads * head_dim) + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Verifying RoPE transformation for V:" << std::endl;
    for (int row = 0; row < std::min(3, n_rows); row++) {
        std::cout << "Token " << row << ": ";
        for (int i = 0; i < std::min(10, n_heads * head_dim); i++) {  
            std::cout << h_V[row * (n_heads * head_dim) + i] << " ";
        }
        std::cout << std::endl;
}

    
    delete[] h_Q;
    delete[] h_V;
    delete[] h_cos_cache;
    delete[] h_sin_cache;
    cudaFree(d_Q);
    cudaFree(d_V);

}