#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>

#define B_r 64
#define B_c 64

// Kernel to compute D = rowsum(dO ⊙ O)
__global__ void compute_D_kernel(
    const float *O,
    const float *dO,
    float *D,
    int batch_size,
    int num_heads,
    int seq_len,
    int d
) {
    int batch_head_idx = blockIdx.z;
    int batch_id = batch_head_idx / num_heads;
    int head_id = batch_head_idx % num_heads;
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx < seq_len) {
        size_t base_offset = ((size_t)batch_id * num_heads + head_id) * seq_len * d;

        float sum = 0.0f;
        for (int i = 0; i < d; i++) {
            size_t idx = base_offset + query_idx * d + i;
            sum += dO[idx] * O[idx];
        }

        size_t d_idx = ((size_t)batch_id * num_heads + head_id) * seq_len + query_idx;
        D[d_idx] = sum;
    }
}

__global__ void flash_attention_backward_kernel(
    const float *Q,
    const float *K, 
    const float *V,
    const float *O,
    const float *dO,
    const float *L, // [batch_size, num_heads, seq_len] : We have one value per query block
    const float *D, // [batch_size, num_heads, seq_len] : row wise sum of element wise product of dO and O
    float *dQ,
    float *dK,
    float *dV,
    int batch_size,
    int num_heads,
    int seq_len,
    int d
){
    int query_block_id  = blockIdx.x;
    int key_block_id =  blockIdx.y;
    int batch_head_idx = blockIdx.z;

    // Unrolling the batch_size * num_heads dimension
    int batch_id = batch_head_idx / num_heads;
    int head_id = batch_head_idx % num_heads;

    int local_row = threadIdx.x;
    int tid_y = threadIdx.y;

    int global_query_idx = query_block_id * B_r + local_row;
    int global_key_idx = key_block_id * B_c + local_row;

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
        size_t query_idx = base_offset + global_query_idx * d + tid_y;
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
        size_t kv_idx =  base_offset + global_key_idx * d  + tid_y;
        K_shared[local_row][tid_y] = K[kv_idx];  // Fixed: K instead of Q
        V_shared[local_row][tid_y] = V[kv_idx];  // Fixed: V instead of dO
    }

    __syncthreads();

    // Computing S = QK^T and P = exp(S-L) for this block
    // For S/P calculations, each thread handles a query row × all keys computation (a row of the attention matrix)
    if(global_query_idx < seq_len && local_row < B_r){
        // For each key block
        for(int c = 0; c < B_c && key_block_id * B_c + c < seq_len; c++){
            float s_ij = 0.0f;
            for(int k = 0; k < d; k++){
                s_ij += Q_shared[local_row][k] * K_shared[c][k];
            }

            s_ij /= sqrtf(d);  // Fixed: sqrtf instead of sqrt

            S_shared[local_row][c] = s_ij;

            // Fixed: Use L_shared instead of L
            P_shared[local_row][c] = expf(s_ij - L_shared[local_row]);
        }
    }

    __syncthreads();

    // Computing dV = dV + P^T(dO)
    // For dQ/dK calculations, each thread handles a query/key position × a specific feature dimension (a single element of the gradient matrix)
    if(global_key_idx < seq_len && tid_y < d){
        // Loop through all query blocks, no need to go into d dimensions
        float dv_sum = 0.0f;
        for(int r = 0; r < B_r && query_block_id * B_r + r < seq_len; r++){
            dv_sum += P_shared[r][local_row] * dO_shared[r][tid_y];
        }

        size_t dv_idx = base_offset + global_key_idx * d + tid_y;
        atomicAdd(&dV[dv_idx], dv_sum);
    }

    __syncthreads();

    // Computing dP = dO.V^T
    if(global_query_idx < seq_len && local_row < B_r){
        for(int c = 0; c < B_c && key_block_id * B_c + c < seq_len; c++){
            float dp_sum = 0.0f;
            for(int k = 0; k < d; k++){
                dp_sum += dO_shared[local_row][k] * V_shared[c][k];
            }
            dP_shared[local_row][c] = dp_sum;
        }
    }
    
    __syncthreads();

    // Computing dS = dP ⊙ P - P ⊙ (P^T · dP)
    // Each thread with a valid query index computes gradients for all key positions in the current block
    if(global_query_idx < seq_len && local_row < B_r){
        for(int c = 0; c < B_c && key_block_id * B_c + c < seq_len; c++){
            float ds_ij = P_shared[local_row][c] * (dP_shared[local_row][c] - D_shared[local_row]);
            S_shared[local_row][c] = ds_ij;
        }
    }

    __syncthreads();
    
    // dQ = dS.K
    // Compute dQ = dS·K (line 15 in algorithm)
    if (global_query_idx < seq_len && tid_y < d) {
        float dq_sum = 0.0f;
        for (int c = 0; c < B_c && key_block_id * B_c + c < seq_len; c++) {
            dq_sum += S_shared[local_row][c] * K_shared[c][tid_y];
        }
        dq_sum /= sqrtf(d);  // Scale by sqrt(d)
        
        // Atomically add to global dQ (needed for accumulation across blocks)
        size_t dq_idx = base_offset + global_query_idx * d + tid_y;
        atomicAdd(&dQ[dq_idx], dq_sum);
    }
    
    // Compute dK = dS^T·Q (line 16 in algorithm)
    if (global_key_idx < seq_len && tid_y < d) {
        float dk_sum = 0.0f;
        for (int r = 0; r < B_r && query_block_id * B_r + r < seq_len; r++) {
            dk_sum += S_shared[r][local_row] * Q_shared[r][tid_y];
        }
        dk_sum /= sqrtf(d);  // Scale by sqrt(d)
        
        // Atomically add to global dK (needed for accumulation across blocks)
        size_t dk_idx = base_offset + global_key_idx * d + tid_y;
        atomicAdd(&dK[dk_idx], dk_sum);
    }
}

__global__ void flash_attention_kernel(
    const float *Q,
    const float *K,
    const float *V,
    float *O,
    float *L,
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

        for(int feat = 0; feat < d; feat++){
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
        int key_block_start = j * B_c;

        if(key_block_start + local_row < seq_len){ 
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
                    s += Q_shared[local_row][feat] * K_shared[feat][key_idx];
                }
                s /= sqrtf((float)d);
                
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
                    s += Q_shared[local_row][feat] * K_shared[feat][key_idx];
                }
                s /= sqrtf((float)d); 

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
    
    int Tr = (seq_len + B_r - 1) / B_r;
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

void flash_attention_backward(
    const float *h_Q,
    const float *h_K,
    const float *h_V,
    const float *h_O,
    const float *h_L,
    const float *h_dO,
    float *h_dQ,
    float *h_dK,
    float *h_dV,
    int batch_size,
    int num_heads,
    int seq_len,
    int d
) {
    float *d_Q, *d_K, *d_V, *d_O, *d_L, *d_dO, *d_dQ, *d_dK, *d_dV, *d_D;
    
    size_t qkv_size = (size_t)batch_size * num_heads * seq_len * d * sizeof(float);
    size_t log_size = (size_t)batch_size * num_heads * seq_len * sizeof(float);
    
    cudaMalloc(&d_Q, qkv_size);
    cudaMalloc(&d_K, qkv_size);
    cudaMalloc(&d_V, qkv_size);
    cudaMalloc(&d_O, qkv_size);
    cudaMalloc(&d_L, log_size);
    cudaMalloc(&d_dO, qkv_size);
    cudaMalloc(&d_dQ, qkv_size);
    cudaMalloc(&d_dK, qkv_size);
    cudaMalloc(&d_dV, qkv_size);
    cudaMalloc(&d_D, log_size); 
    
    cudaMemcpy(d_Q, h_Q, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_O, h_O, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, h_L, log_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dO, h_dO, qkv_size, cudaMemcpyHostToDevice);
    
    cudaMemset(d_dQ, 0, qkv_size);
    cudaMemset(d_dK, 0, qkv_size);
    cudaMemset(d_dV, 0, qkv_size);
    
    dim3 d_grid((seq_len + 255) / 256, 1, batch_size * num_heads);
    dim3 d_block(256, 1, 1);
    compute_D_kernel<<<d_grid, d_block>>>(d_O, d_dO, d_D, batch_size, num_heads, seq_len, d);
    
    int T_r = (seq_len + B_r - 1) / B_r;  // Number of query blocks
    int T_c = (seq_len + B_c - 1) / B_c;  // Number of key/value blocks
    
    dim3 gridDim(T_r, T_c, batch_size * num_heads);
    dim3 blockDim(B_r, 16);  // Using 16 threads per row for parallel processing
    
    flash_attention_backward_kernel<<<gridDim, blockDim>>>(
        d_Q, d_K, d_V, d_O, d_dO, d_L, d_D,
        d_dQ, d_dK, d_dV,
        batch_size, num_heads, seq_len, d
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaMemcpy(h_dQ, d_dQ, qkv_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dK, d_dK, qkv_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dV, d_dV, qkv_size, cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
    cudaFree(d_dO);
    cudaFree(d_dQ);
    cudaFree(d_dK);
    cudaFree(d_dV);
    cudaFree(d_D);
}

int main() {
    int batch_size = 1;
    int num_heads = 8;
    int seq_len = 128;
    int d = 32;

    size_t total_elements = (size_t)batch_size * num_heads * seq_len * d;
    size_t total_output_size = (size_t)batch_size * num_heads * seq_len;

    
    float *h_Q = new float[total_elements];
    float *h_K = new float[total_elements];
    float *h_V = new float[total_elements];
    float *h_O = new float[total_elements];
    float *h_L = new float[total_output_size];


    float *h_dO = new float[total_elements];
    float *h_dQ = new float[total_elements];
    float *h_dK = new float[total_elements];
    float *h_dV = new float[total_elements];

    for (size_t i = 0; i < total_elements; i++) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
        h_dO[i] = static_cast<float>(rand()) / RAND_MAX; 
    }

    std::cout << "Running forward pass..." << std::endl;
    flash_attention_forward(h_Q, h_K, h_V, h_O, h_L,
                           batch_size, num_heads, seq_len, d);

    std::cout << "Running backward pass..." << std::endl;
    flash_attention_backward(h_Q, h_K, h_V, h_O, h_L, h_dO, h_dQ, h_dK, h_dV,
                            batch_size, num_heads, seq_len, d);

    std::cout << "Gradient dQ (first 10 values): ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_dQ[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Gradient dK (first 10 values): ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_dK[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Gradient dV (first 10 values): ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_dV[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;
    delete[] h_L;
    delete[] h_dO;
    delete[] h_dQ;
    delete[] h_dK;
    delete[] h_dV;

    return 0;
}