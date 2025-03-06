#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

static const int B_r = 64;
static const int B_c = 64

__global__ void flash_attention_kernel(
    const float *Q,
    const float *K, 
    const float *V, 
    float *O, 
    float *L, 
    int batch_size, 
    int num_heads, 
    int seq_len, 
    int d){
        // Index Calculations
        int batch_id = blockIdx.z;
        int query_block_id = blockIdx.x;
        int head_id = blockIdx.y;

        int local_row = threadIdx.x; // Which amongst the Br rows are we in
        int global_query_idx = query_block_id * B_r + local_row; // This is absolute position within query blocks. MAx value - seq_len
        
        int tid_y = threadIdx.y; 

        // Shape of Q is [batch_size, num_heads, seq_len, d] and elements are laid contigously
        // For each batch and head we have seq_len*d floats
        // First flattening along batch and head dimensions (batch_id * num_heads + head_id), then we multiply by seq_len*d
        
        size_t base_offset = ((size_t)batch_id*num_heads + head_id) * seq_len * d;


        __shared__ float Q_shared[B_r][d]; 
        __shared__ float K_shared[d][B_c]; // Loading K in transposed form
        __shared__ float V_shared[B_c][d];

        __shared__ float row_max[B_r];
        __shared__ float row_sum[B_r];


        // Loading Query Block into shared memory
        if(global_query_idx < seq_len && tid_y < d){
            // base offset stands at beginning of current batch and head. We need to flatten along seq_len(global_query_idx) and d dimensions to reach current feature_dim
            int q_idx = base_offset + global_query_idx*dim + tid_y;
            Q_shared[local_row][tid_y] = Q[q_idx];
        }

        if(feature_index == 0 && global_query_idx < seq_len){
            row_max[local_row] = -INFINITY; 
            row_sum[local_row] = 0.0f;
        }

        __syncthreads();


        // Loading key and value tiles into shared memory

        int num_tiles = (seq_len + B_c - 1)/ B_c;
        for(int tileIdx = 0; tileIdx < num_tiles; tileIdx++){
            int global_key_idx = tileIdx * B_c + tid_y;

            if(global_key_idx < seq_len){
                // Loop such that if needed each thread covers multiple features
                for(int feat = tid_y; feat <d; feat += blockDim.y){
                    int k_idx = base_offset + global_key_idx * d + feat;
                    K_shared[feat][tid_y] = K[k_idx]; // Loading in transposed form
                }

                for (int feat = tid_y; feat < d; feat += blockDim.y) {
                    int v_idx = base_offset + global_key_idx * d + feat;
                    V_shared[tid_y][feat] = V[v_idx];
                }
            }

            __syncthreads();

            // Compute dot product and softmax accumulators

            if(tid_y == 0 && global_query_idx < seq_len){
                // Computing score for each key in tile
                for(int pos = 0; pos < B_c; pos ++){ // Shape of K_shared is [d, B_c]
                    // Pos is the tile Idx
                    float s = 0.0f;
                    for(int j = 0; j < d; j++){
                    // This gives the key position within the tile
                        s += Q_shared[local_row][j] * K_shared[j][pos];
                    }
                    s *= rsqrtf((float)d);

                    float old_max = row_max[local_row];
                    float new_max = fmaxf(old_max, s);
                    row_sum[local_row] = expf(old_max - new_max) * row_sum[local_row] + expf(s - new_max); // Applying the softmax correction
                    row_max[local_row] = new_max;
                }
            }
            __syncthreads();

        }

       
        __shared__ float O_shared[B_r][d];
        // Initializing this for each query row
        if(tid_y == 0 && global_query_idx < seq_len){
            for(int feat = 0; feat < d; feat++){
                O_shared[local_row][feat] = 0.0f;
            }
        }
        __syncthreads();

        // Looping over the tiles to compute the weighted sum over V
        for(int tileIdx = 0; tileIdx < num_tiles; tileIdx++){
            __syncthreads();

            if(tid_y == 0 && global_query_idx < seq_len){
                for(int pos = 0; pos < B_c; pos++){
                    float s = 0.0f;
                    for(int j = 0; j < d; j++){
                        s += Q_shared[local_row][j] * K_shared[j][pos];
                    }
                    s *= rqrtf((float)d);

                    // Computing the softmax probability
                    float weight = expf(s - row_max[local_row]) / row_sum[local_row];

                    for(int feat = 0; feat < d; feat ++){
                        O_shared[local_row][feat] += weight * V_shared[pos][feat];
                    }
                }
            }
            __syncthreads();
        }

        // Final output from shared to global memory
        if(global_query_idx < seq_len){
            for(int feat)
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
    int d // head_dim
){
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    
    size_t qkv_size = (size_t)batch_size * num_heads * seq_len * d * sizeof(float);
    size_t out_size = (size_t)batch_size * num_heads * seq_len * d * sizeof(float);
    size_t log_size = (size_t)batch_size * num_heads * seq_len * sizeof(float); // We will have one per per token

    cudaMalloc(&d_Q, qkv_size);
    cudaMalloc(&d_K, qkv_size);
    cudaMalloc(&d_V, qkv_size);
    cudaMalloc(&d_O, out_size);
    cudaMalloc(&d_L, log_size);

    cudaMemcpy(d_Q, h_Q, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_size, cudaMemcpyHostToDevice);

    int grid_x = (seq_len + B_r - 1)/ B_r; // Total number of sequence blocks
    int grid_y = num_heads;
    int grid_z = batch_size;
    
    dim3 gridDim(grid_x, grid_y, grid_z);
    dim3 blockDim(32,32); // B_r*d may exceed 1024. Subject to further changes later

    flash_attention_kernel<<<gridDim, blockDim>>>(
        d_Q, d_K, d_V, d_O, d_L,
        batch_size, num_heads, seq_len, d
    );

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
    int d = 128;  // Head dimension

    size_t total_elements = (size_t)batch_size * num_heads * seq_len * d;
    float *h_Q = new float[total_elements];
    float *h_K = new float[total_elements];
    float *h_V = new float[total_elements];
    float *h_O = new float[total_elements];
    float *h_L = new float[batch_size * num_heads * seq_len];

    // Initialize Q, K, V with random data.
    for (size_t i = 0; i < total_elements; i++) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    flash_attention_forward(h_Q, h_K, h_V, h_O, h_L,
                            batch_size, num_heads, seq_len, d);

    // Print first 10 output values for a quick check.
    std::cout << "Output O (first 10 values): ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_O[i] << " ";
    }
    std::cout << std::endl;

    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;
    delete[] h_L;

    return 0;
}