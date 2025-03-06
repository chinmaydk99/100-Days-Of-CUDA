#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib> 

static const int B_r = 64;
static const int B_c = 64;

__global__ void flash_attention_kernel(
    const float *Q,
    const float *K,
    const float *V,
    float * O,
    float * L,
    int batch_size,
    int num_heads,
    int seq_len,
    int d
){
    int batch_id = blockIdx.z;
    int query_block_id = blockIdx.x;
    int head_id = blockIdx.y;

    int local_row = threadIdx.x; // Local position within the block
    int global_query_idx =  query_block_id*B_r + local_row;

    int tid_y = threadIdx.y;

    // Total Number of key value blocks
    int T_c = (seq_len + B_c - 1)/ B_c;

    // Starting point for current batch and head
    size_t base_offset = ((size_t)batch_id * num_heads + head_id ) * seq_len * d;

    // Shared Memory Allocation
    __shared__ float Q_shared[B_r][128];
    __shared__ float K_shared[128][B_c]; // Storing in Transposed form
    __shared__ float V_shared[B_c][128];

    // Softmax accumulator
    __shared__ float m_prev[B_r];
    __shared__ float m_curr[B_r];
    __shared__ float l_prev[B_r];
    __shared__ float l_curr[B_r];

    // Output accumulation
    __shared__ float O_shared[B_r][128];

    // Loading the current query block into shared memory
    if(global_query_idx < seq_len && tid_y < d){
        int q_idx = base_offset + global_query_idx * d + tid_y;
        Q_shared[local_row][tid_y] = Q[q_idx];
    }

    // Initialising softmax accumulators and output. I'll use only one thread for this to avoid redundant operations
    if(tid_y == 0 && global_query_idx < seq_len){
        m_curr[local_row] = -INFINITY;
        l_curr[local_row] = 0.0f;

        for(int feat = 0; feat < d; feat ++){
            O_shared[local_row][feat] = 0.0f;
        }
    }

    __syncthreads();

    // Processing each key value block
    for(int j = 0; j < T_c; j++){
        // Saving current softmax states as previous states
        if(tid_y == 0 && global_query_idx < seq_len){
            m_prev[local_row] = m_curr[local_row];
            l_prev[local_row] = l_curr[local_row];
        }
        __syncthreads();

        // Load K and V from global to shared memory
        int key_block_start = j * B_c; // Fixed: Use B_c instead of T_c

        if(key_block_start + local_row < seq_len){ // Fixed: Changed tid_y to local_row
            for(int feat = tid_y; feat < d; feat += blockDim.y){
                int k_idx = base_offset + (key_block_start + local_row) * d + feat;
                if(local_row < B_c && key_block_start + local_row < seq_len){
                    K_shared[feat][local_row] = K[k_idx];
                }
            }

            for(int feat = tid_y; feat < d; feat += blockDim.y){
                int v_idx = base_offset + (key_block_start + local_row) * d + feat;
                if(local_row < B_c && key_block_start + local_row < seq_len){
                    V_shared[local_row][feat] = V[v_idx];
                }
            }
        }

        __syncthreads();

        // Each thread handles one query row
        // First pass is to obtain the max value
        if(tid_y == 0 && global_query_idx < seq_len){
            float m_i_j = m_prev[local_row];

            for(int key_idx = 0; key_idx < B_c && key_idx + key_block_start < seq_len; key_idx++){
                float s = 0.0f;
                for(int feat = 0; feat < d; feat++){
                    s += Q_shared[local_row][feat] * K_shared[feat][key_idx]; // Fixed: Q_shared not Q
                }
                s *= sqrtf((float)d);
                
                m_i_j = fmaxf(m_i_j, s);
            }

            // Computing Normalization score using new max
            float l_i_j = 0.0f;
            if(l_prev[local_row] > 0){
                l_i_j = expf(m_prev[local_row] - m_i_j) * l_prev[local_row];
            }

            float P_sums[128];
            for(int key_idx = 0; key_idx < B_c && key_idx + key_block_start < seq_len; key_idx++){
                float s = 0.0f;
                for(int feat = 0; feat < d; feat++){
                    s += Q_shared[local_row][feat] * K_shared[feat][key_idx]; // Fixed: Q_shared not Q
                }
                s *= sqrtf((float)d); 

                float p_ij = expf(s - m_i_j);
                P_sums[key_idx] = p_ij;

                l_i_j += p_ij;
            }

            for(int feat = 0; feat < d; feat++){
                float output = 0.0f;
                // Scaling previous output by change in max
                if (l_prev[local_row] > 0) {
                    output = expf(m_prev[local_row] - m_i_j) * O_shared[local_row][feat];
                }

                // Add contribution by current block
                for (int key_idx = 0; key_idx < B_c && key_block_start + key_idx < seq_len; key_idx++) {
                    output += (P_sums[key_idx] / l_i_j) * V_shared[key_idx][feat];
                }
                
                O_shared[local_row][feat] = output;
            }
            m_curr[local_row] = m_i_j;
            l_curr[local_row] = l_i_j;
        }
        __syncthreads();
    }
    
    if (global_query_idx < seq_len) {
        for (int feat = tid_y; feat < d; feat += blockDim.y) {
            int out_idx = base_offset + global_query_idx * d + feat;
            O[out_idx] = O_shared[local_row][feat];
        }

        if (tid_y == 0) {
            int l_idx = (batch_id * num_heads * seq_len) + (head_id * seq_len) + global_query_idx;
            L[l_idx] = m_curr[local_row] + logf(l_curr[local_row]);
        }
    }
}


// Host function
void flash_attention_forward(
    const float *h_Q,
    const float *h_K,
    const float *h_V,
    float *h_O,
    float *h_L,
    int batch_size,
    int num_heads,
    int seq_len,
    int d
){
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    
    size_t qkv_size = (size_t)batch_size * num_heads * seq_len * d * sizeof(float);
    size_t out_size = (size_t)batch_size * num_heads * seq_len * d * sizeof(float);
    size_t log_size = (size_t)batch_size * num_heads * seq_len * sizeof(float);

    cudaMalloc(&d_Q, qkv_size);
    cudaMalloc(&d_K, qkv_size);
    cudaMalloc(&d_V, qkv_size);
    cudaMalloc(&d_O, out_size);
    cudaMalloc(&d_L, log_size);

    cudaMemcpy(d_Q, h_Q, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_size, cudaMemcpyHostToDevice);

    cudaMemset(d_O, 0, out_size);
    
    int Tr = (seq_len + B_r - 1) / B_r;  // Number of query blocks
    int grid_y = num_heads;
    int grid_z = batch_size;
    
    dim3 gridDim(Tr, grid_y, grid_z);
    dim3 blockDim(B_r, 16);

    flash_attention_kernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, d_L,
                                                  batch_size, num_heads, seq_len, d);

    cudaMemcpy(h_O, d_O, out_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_L, d_L, log_size, cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
}


int main() {
    int batch_size = 1;
    int num_heads = 8;
    int seq_len = 512;
    int d = 128;

    size_t total_elements = (size_t)batch_size * num_heads * seq_len * d;
    float *h_Q = new float[total_elements];
    float *h_K = new float[total_elements];
    float *h_V = new float[total_elements];
    float *h_O = new float[total_elements];
    float *h_L = new float[batch_size * num_heads * seq_len];

    for (size_t i = 0; i < total_elements; i++) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    flash_attention_forward(h_Q, h_K, h_V, h_O, h_L,
                            batch_size, num_heads, seq_len, d);

    std::cout << "Output O (first 10 values): ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_O[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "LogSumExp L (first 5 values): ";
    for (int i = 0; i < 5; i++) {
        std::cout << h_L[i] << " ";
    }
    std::cout << std::endl;

    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;
    delete[] h_L;

    return 0;
}