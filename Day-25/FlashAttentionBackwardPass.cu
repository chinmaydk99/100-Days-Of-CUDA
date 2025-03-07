#include <iostream>
#include <cuda_runtime.h>


#define B_r 64
#define B_c 64

__global__ float(
    const float *Q,
    const float *K, 
    const float *V,
    const float *O,
    const float *dO,
    const float *L, // [batch_size, num_heads, seq_len] : We have one value per query block
    const float *D, // [batch_size, num_heads, seq_len] : row wise sum of element wise product of dO and O
    float *dQ,
    float *dV,
    float *dV,
    int batch_size,
    int num_heads,
    int seq_len,
    int d
){
    int q_id  = blockIdx.x;
    int key_id =  blockIdx.y;
    int batch_head_id = blockIdx.z;

    // Unrolling the batch_size * num_heads dimension
    int batch_id = batch_head_id / d;
    int head_id = batch_head_id % d;

    int local_row = threadIdx.x;
    int tid_y = threadIdx.y;

    int global_query_idx = q_id * B_r + local_row;
    int global_key_idx = key_id * B_c + local_row;

    // Base offsets for current batch and head
    size_t base_offset = ((size_t)batch_id * num_heads + head_id) * seq_len * d;
    
    // Base offset for logsumexp L , which is of shape [batch_size, num_heads, seq_len]
    size_t l_base_offset = ((size_t)batch_id * num_heads + head_id) * seq_len;

    // Shared memory for block matrices
    __shared__ float Q_shared[B_r][128];
    __shared__ float K_shared[B_c][128];
    __shared__ float V_shared[B_c][128];
    __shared__ float dO_shared[B_r][128];
    __shared__ float L_shared[B_r];
    __shared__ float D_shared[B_r]; //rowsum(dO ⊙ O)

    // Shared memory for intermediate computations
    __shared__ float S_shared[B_r][B_c];    // Attention Scores
    __shared__ float P_shared[B_r][B_c];    // Probabilities
    __shared__ float dP_shared[B_r][B_c];  // Gradient of probabilities

    // Loading data into shared memory
    if(global_query_idx < seq_len && tid_y < d){
        size_t query_idx = base_offset + global_query_idx*d + tid_y;
        Q_shared[local_row][tid_y] = Q[query_idx];
        dO_shared[local_row][tid_y] = dO[query_idx];
    }

    // Load L and D for this query
    if (tid_y == 0 && global_query_idx < seq_len) {
        L_shared[local_row] = L[l_base_offset + global_query_idx];
        D_shared[local_row] = D[l_base_offset + global_query_idx];
    }
    
    __syncthreads();

    // Loading Key and Value
    if(tid_y < d && global_key_idx < seq_len){
        size_t kv_idx =  base_offset + global_key_idx*d  + tid_y;
        K_shared[local_row][tid_y] = Q[kv_idx];
        V_shared[local_row][tid_y] = dO[kv_idx];
    }

    __syncthreads();

    // Computing S = QK^T and P = exp(S-L) for this block
    if(global_query_idx < seq_len && local_row < B_r){
        // For each key block
        for(int c = 0; c < B_c && global_key_idx*B_c + c < seq_len; c++){
            float s_ij = 0.0f;
            for(int k = 0; k < d; k ++){
                s_ij += Q_shared[local_row][k] * K_shared[c][k];
            }

            s_ij /= sqrt(d);

            S_shared[local_row][c] = s_ij;

            P_shared[local_row] = expf(s_ij - L[local_row]);
        }
    }
}