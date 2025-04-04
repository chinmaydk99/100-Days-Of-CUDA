#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath> 
#include <vector>
#include <numeric>
#include <limits>
#include <cassert>
#include <algorithm> 
#include <float.h>


#define WARP_SIZE 32

#define DEVICE_MAX(a, b) fmaxf((a), (b))
#define DEVICE_MIN(a, b) fminf((a), (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__,       \
                    __LINE__, cudaGetErrorString(err));                   \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// --- Warp Shuffle Intrinsics ---
#define CUDA_SHFL_XOR_SYNC(var, lane_mask) \
    __shfl_xor_sync(0xFFFFFFFF, var, lane_mask)

#define CUDA_SHFL_SYNC(var, src_lane) \
    __shfl_sync(0xFFFFFFFF, var, src_lane)


// --- Block Reduction Utilities ---

// Sum reduction across a block
template <int NUM_WARPS_TPL>
__device__ inline float block_sum(float* red_smem, float thread_sum) {
    constexpr int NUM_WARPS = NUM_WARPS_TPL;
    int warp_idx = threadIdx.x / WARP_SIZE;
    int lane_idx = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        thread_sum += CUDA_SHFL_XOR_SYNC(thread_sum, mask);
    }
    if (lane_idx == 0) red_smem[warp_idx] = thread_sum;
    __syncthreads();

    float warp_sum = (lane_idx < NUM_WARPS) ? red_smem[lane_idx] : 0.0f;
    #pragma unroll
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
        warp_sum += CUDA_SHFL_XOR_SYNC(warp_sum, mask);
    }
    return CUDA_SHFL_SYNC(warp_sum, 0);
}

// Max reduction across a block
template <int NUM_WARPS_TPL>
__device__ inline float block_fmaxf(float* red_smem, float thread_max) {
    constexpr int NUM_WARPS = NUM_WARPS_TPL;
    int warp_idx = threadIdx.x / WARP_SIZE;
    int lane_idx = threadIdx.x % WARP_SIZE;
    float max_val = thread_max;

    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        max_val = DEVICE_MAX(max_val, CUDA_SHFL_XOR_SYNC(max_val, mask));
    }
    if (lane_idx == 0) red_smem[warp_idx] = max_val;
    __syncthreads();

    // FIX: Use -FLT_MAX for initialization
    max_val = (lane_idx < NUM_WARPS) ? red_smem[lane_idx] : -FLT_MAX;
     #pragma unroll
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
        max_val = DEVICE_MAX(max_val, CUDA_SHFL_XOR_SYNC(max_val, mask));
    }
    return CUDA_SHFL_SYNC(max_val, 0);
}


// --- Simplified Paged Attention V1 Kernel ---
template <int NUM_THREADS_TPL, int HEAD_SIZE_TPL>
__global__ void paged_attention_v1_kernel_standalone(
    float* __restrict__ out,           // [num_seqs, num_heads, head_size]
    const float* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const float* __restrict__ k_cache, // Flat pool [...]
    const float* __restrict__ v_cache, // Flat pool [...]
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,     // [num_seqs]
    const int max_num_blocks_per_seq,
    const int kv_block_stride,   // Elements between blocks
    const int kv_head_stride,    // Elements between heads within a block
    const int BLOCK_SIZE
) {
    // Use template parameters
    constexpr int NUM_THREADS = NUM_THREADS_TPL;
    constexpr int HEAD_SIZE = HEAD_SIZE_TPL;

    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    const int num_heads = gridDim.x;
    const int num_queries_per_kv = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / num_queries_per_kv;

    const int thread_idx = threadIdx.x;
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    const int warp_idx = thread_idx / WARP_SIZE;
    

    const int start_block_idx = 0;
    const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
    const int end_block_idx = num_seq_blocks;
    const int num_tokens_to_process = seq_len;

    // --- Step 5: Grouping & Vec (Simplified) ---
    const int THREAD_GROUP_SIZE = 1;
    constexpr int CURRENT_VEC_SIZE = 1;
    using Q_vec = float;
    using K_vec = float;
    using V_vec = float;

    // These become simpler with THREAD_GROUP_SIZE = 1
    const int num_elems_per_thread = HEAD_SIZE / THREAD_GROUP_SIZE; // = HEAD_SIZE
    const int num_vecs_per_thread = num_elems_per_thread / CURRENT_VEC_SIZE; // = HEAD_SIZE

    // --- Step 7: Prepare Shared Memory & Init Registers ---
    extern __shared__ char shared_mem[];
    float* logits = reinterpret_cast<float*>(shared_mem);
    float* reduction_smem = logits + num_tokens_to_process;
    float* output_smem = reduction_smem + NUM_WARPS;

    float qk_max = -FLT_MAX;

    float output_accumulators[num_elems_per_thread]; // Size = HEAD_SIZE
    #pragma unroll
    for (int i = 0; i < num_elems_per_thread; ++i) {
        output_accumulators[i] = 0.0f;
    }

    // --- Step 8: QK Calculation Loop ---
    const float* q_head_ptr = q + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;

    for (int block_idx = start_block_idx; block_idx < end_block_idx; ++block_idx) { // Simplified loop
        const int block_table_idx = seq_idx * max_num_blocks_per_seq + block_idx;
        const int64_t physical_block_number = (int64_t)block_tables[block_table_idx];
        const float* k_block_head_ptr = k_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride;

        for (int token_offset_in_block = 0; token_offset_in_block < BLOCK_SIZE; ++token_offset_in_block) {
            const int current_token_idx = block_idx * BLOCK_SIZE + token_offset_in_block;

            if (current_token_idx < seq_len) {
                const float* k_token_ptr = k_block_head_ptr + token_offset_in_block * HEAD_SIZE;

                // Calculate QK dot product - now fully per-thread (THREAD_GROUP_SIZE=1)
                float qk_dot = 0.0f;
                #pragma unroll
                // Use simple loop as num_vecs_per_thread == HEAD_SIZE
                for (int j = 0; j < HEAD_SIZE; ++j) {
                    if ( (j / NUM_THREADS) == (thread_idx / NUM_THREADS) && (j % NUM_THREADS == thread_idx % NUM_THREADS) ) {
                        float q_val = q_head_ptr[j];
                        float k_val = k_token_ptr[j]; 
                        qk_dot += q_val * k_val;
                    }
                }

                // --- Reduce QK dot product across block ---
                // Now we need to sum the partial `qk_dot` from each thread.
                reduction_smem[thread_idx] = qk_dot;
                __syncthreads();

                float final_qk_dot = 0.0f;
                if (thread_idx == 0) {
                     for (int t = 0; t < NUM_THREADS; ++t) final_qk_dot += reduction_smem[t];
                     float qk_val = final_qk_dot * scale;
                     logits[current_token_idx] = qk_val;
                     // Thread 0 updates its qk_max based on the final computed logit
                     qk_max = DEVICE_MAX(qk_max, qk_val);
                }
                __syncthreads(); // Sync before next token

            } else { // Padding
                if (thread_idx == 0) {
                     // FIX: Use -FLT_MAX
                     logits[current_token_idx] = -FLT_MAX;
                }
                __syncthreads();
            }
        } // End loop token_offset_in_block
    } // End loop block_idx

    // --- Step 9: Find Global Max Logit ---

    qk_max = block_fmaxf<NUM_WARPS>(reduction_smem, thread_idx == 0 ? qk_max : -FLT_MAX);
    // --- Step 10: Calculate Exp Sum ---
    float exp_sum = 0.0f;
    for (int i = thread_idx; i < num_tokens_to_process; i += NUM_THREADS) {
        float val = expf(logits[i] - qk_max);
        logits[i] = val;
        exp_sum += val;
    }
    exp_sum = block_sum<NUM_WARPS>(reduction_smem, exp_sum);

    // --- Step 11: Normalize Logits ---
    float inv_exp_sum = 1.0f / (exp_sum + 1e-6f);
    for (int i = thread_idx; i < num_tokens_to_process; i += NUM_THREADS) {
        logits[i] *= inv_exp_sum;
    }
    __syncthreads();

    // --- Step 12: Aggregate Value (V) Vectors ---
    #pragma unroll // Reset accumulators
    for (int i = 0; i < num_elems_per_thread; ++i) output_accumulators[i] = 0.0f;

    for (int block_idx = start_block_idx; block_idx < end_block_idx; ++block_idx) {
        const int block_table_idx = seq_idx * max_num_blocks_per_seq + block_idx;
        const int64_t physical_block_number = (int64_t)block_tables[block_table_idx];
        const float* v_block_head_ptr = v_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride;

        for (int token_offset_in_block = 0; token_offset_in_block < BLOCK_SIZE; ++token_offset_in_block) {
            const int current_token_idx = block_idx * BLOCK_SIZE + token_offset_in_block;

            if (current_token_idx < seq_len) {
                const float* v_token_ptr = v_block_head_ptr + token_offset_in_block * HEAD_SIZE; // Assumed Layout
                float sm_prob = logits[current_token_idx]; // Read softmax prob from shared mem

                 // Accumulate V * sm_prob using grid-stride loop over HEAD_SIZE
                 #pragma unroll
                 for (int j = 0; j < HEAD_SIZE; ++j) { // Each thread iterates all of HEAD_SIZE
                    // Check if this thread is responsible for element j
                    if ( (j / NUM_THREADS) == (thread_idx / NUM_THREADS) && (j % NUM_THREADS == thread_idx % NUM_THREADS) ) {
                        float v_val = v_token_ptr[j]; // Load V element
                        // Since THREAD_GROUP_SIZE=1, num_elems_per_thread=HEAD_SIZE.
                        // The index 'j' corresponds directly to the accumulator index.
                        output_accumulators[j] += sm_prob * v_val;
                    }
                 }
            }
        } // End token_offset_in_block loop
    } // End block_idx loop for V aggregation
    __syncthreads(); // Ensure all accumulation registers are updated

    // --- Step 13: Reduce Accumulators and Write Output ---
    // Each thread holds the final value for the elements it calculated in V aggregation.
    float* out_head_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    #pragma unroll
    for (int j = 0; j < HEAD_SIZE; ++j) { // Grid-stride write
        if ( (j / NUM_THREADS) == (thread_idx / NUM_THREADS) && (j % NUM_THREADS == thread_idx % NUM_THREADS) ) {
            out_head_ptr[j] = output_accumulators[j];
        }
    }
}


// --- Host Code ---

#include <vector>
#include <iostream>
#include <random>
#include <limits>
#include <cassert>
#include <map>

void paged_attention_cpu(
    float* out_cpu, const float* q, const float* k_cache, const float* v_cache,
    const int* block_tables, const int* seq_lens,
    int num_seqs, int num_heads, int head_size, int block_size,
    int max_num_blocks_per_seq, int num_kv_heads, float scale,
    int total_blocks, int kv_block_stride, int kv_head_stride)
{
    #pragma omp parallel for collapse(2)
    for (int s = 0; s < num_seqs; ++s) {
        for (int h = 0; h < num_heads; ++h) {
            int seq_len = seq_lens[s];
             float* out_vec = out_cpu + s * num_heads * head_size + h * head_size;
             std::fill(out_vec, out_vec + head_size, 0.0f); // Initialize output
            if (seq_len == 0) continue;

            const int* sequence_block_table = block_tables + s * max_num_blocks_per_seq;
            int kv_head_idx = h / (num_heads / num_kv_heads);
            const float* q_vec = q + s * num_heads * head_size + h * head_size;

            std::vector<float> logits(seq_len);
            float max_logit = -FLT_MAX; // Use FLT_MAX

            // Calculate Logits
            for (int token_idx = 0; token_idx < seq_len; ++token_idx) {
                int block_idx = token_idx / block_size;
                int token_offset_in_block = token_idx % block_size;
                assert(block_idx < max_num_blocks_per_seq);
                int physical_block_id = sequence_block_table[block_idx];
                assert(physical_block_id >= 0 && physical_block_id < total_blocks);

                const float* k_token_ptr = k_cache + (size_t)physical_block_id * kv_block_stride +
                                           (size_t)kv_head_idx * kv_head_stride +
                                           (size_t)token_offset_in_block * head_size;

                float qk_dot = 0.0f;
                for (int d = 0; d < head_size; ++d) qk_dot += q_vec[d] * k_token_ptr[d];
                float logit = qk_dot * scale;
                logits[token_idx] = logit;
                max_logit = std::max(max_logit, logit); 
            }

            // Calculate Softmax
            float exp_sum = 0.0f;
            std::vector<float> probs(seq_len);
            for (int token_idx = 0; token_idx < seq_len; ++token_idx) {
                float val = expf(logits[token_idx] - max_logit);
                probs[token_idx] = val;
                exp_sum += val;
            }
            float inv_exp_sum = 1.0f / (exp_sum + 1e-6f);
            for (int token_idx = 0; token_idx < seq_len; ++token_idx) probs[token_idx] *= inv_exp_sum;

            // Aggregate Values
             // std::fill already done at start
            for (int token_idx = 0; token_idx < seq_len; ++token_idx) {
                 int block_idx = token_idx / block_size;
                 int token_offset_in_block = token_idx % block_size;
                 int physical_block_id = sequence_block_table[block_idx];
                 assert(physical_block_id >= 0 && physical_block_id < total_blocks);

                 const float* v_token_ptr = v_cache + (size_t)physical_block_id * kv_block_stride +
                                            (size_t)kv_head_idx * kv_head_stride +
                                            (size_t)token_offset_in_block * head_size;

                 float prob = probs[token_idx];
                 for(int d=0; d<head_size; ++d) out_vec[d] += prob * v_token_ptr[d];
            }
        }
    }
}


// Helper function type for kernel launch
typedef void (*paged_attn_kernel_t)(
    float*, const float*, const float*, const float*, int, float,
    const int*, const int*, int, int, int, int);

int main() {
    // --- Parameters ---
    int num_seqs = 8;
    int num_heads = 12;
    int head_size = 128; 
    int block_size = 16;
    int max_seq_len_alloc = 256;
    int num_kv_heads = num_heads;
    int total_physical_blocks = 512;
    const int HOST_NUM_THREADS = 128;

    // ... (Parameter printing) ...
    printf("Parameters:\n");
    printf("  Num Seqs: %d\n", num_seqs);
    printf("  Num Heads: %d\n", num_heads);
    printf("  Head Size: %d\n", head_size);
    printf("  Block Size: %d\n", block_size);
    printf("  Max Seq Len Alloc: %d\n", max_seq_len_alloc);
    printf("  Num KV Heads: %d\n", num_kv_heads);
    printf("  Total Physical Blocks: %d\n", total_physical_blocks);
    printf("  Threads Per Block: %d\n", HOST_NUM_THREADS);


    // Derived sizes
    int max_num_blocks_per_seq = DIVIDE_ROUND_UP(max_seq_len_alloc, block_size);
    float scale = 1.0f / sqrtf(static_cast<float>(head_size));

    // --- Simulate Block Allocation & Generate Data 
    srand(42);
    std::vector<int> seq_lens(num_seqs);
    std::vector<std::vector<int>> block_tables_host(num_seqs, std::vector<int>(max_num_blocks_per_seq));
    std::vector<int> physical_block_usage(total_physical_blocks, 0);
    int physical_blocks_allocated = 0;
    auto allocate_block = [&]() -> int { /* ... */
        for(int i=0; i<total_physical_blocks; ++i) {
            if(physical_block_usage[i] == 0) {
                physical_block_usage[i] = 1;
                physical_blocks_allocated++;
                return i;
            }
        }
        return -1;
    };
    printf("\nGenerating sequence lengths and block tables...\n");
    for (int i = 0; i < num_seqs; ++i) { /* ... */
        seq_lens[i] = (rand() % max_seq_len_alloc) + 1;
        int num_blocks_needed = DIVIDE_ROUND_UP(seq_lens[i], block_size);
        printf("  Seq %d: Len = %d, Blocks Needed = %d, Blocks = [", i, seq_lens[i], num_blocks_needed);
        for (int j = 0; j < num_blocks_needed; ++j) {
            int physical_block = allocate_block();
            if (physical_block == -1) { fprintf(stderr, "Error: Ran out of physical blocks!\n"); return EXIT_FAILURE; }
            block_tables_host[i][j] = physical_block;
            printf(" %d", physical_block);
        }
        for (int j = num_blocks_needed; j < max_num_blocks_per_seq; ++j) block_tables_host[i][j] = -1;
        printf(" ]\n");
    }
    printf("Total physical blocks allocated: %d\n", physical_blocks_allocated);


    size_t q_size = (size_t)num_seqs * num_heads * head_size;
    int kv_block_stride_host = num_kv_heads * block_size * head_size;
    int kv_head_stride_host = block_size * head_size;
    size_t kv_cache_pool_size = (size_t)total_physical_blocks * kv_block_stride_host;
    size_t block_tables_size = (size_t)num_seqs * max_num_blocks_per_seq;
    size_t seq_lens_size = num_seqs;
    size_t out_size = (size_t)num_seqs * num_heads * head_size;
    std::vector<float> h_q(q_size);
    std::vector<float> h_k_cache(kv_cache_pool_size);
    std::vector<float> h_v_cache(kv_cache_pool_size);
    std::vector<int> h_block_tables(block_tables_size);
    std::vector<float> h_out_gpu(out_size);
    std::vector<float> h_out_cpu(out_size);
    for(int i=0; i<num_seqs; ++i) for(int j=0; j<max_num_blocks_per_seq; ++j) h_block_tables[i * max_num_blocks_per_seq + j] = block_tables_host[i][j];


     printf("\nInitializing host data (Q, K cache, V cache)...\n");
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dist(-1.0, 1.0);
    for(size_t i=0; i<q_size; ++i) h_q[i] = dist(gen);
    int kv_token_stride_host = head_size;
    for(int pb=0; pb<total_physical_blocks; ++pb) { /* ... */
         if(physical_block_usage[pb]) {
            for(int kvh=0; kvh<num_kv_heads; ++kvh) {
                for(int t=0; t<block_size; ++t) {
                    for(int d=0; d<head_size; ++d) {
                        size_t k_idx = (size_t)pb * kv_block_stride_host + (size_t)kvh * kv_head_stride_host + (size_t)t * kv_token_stride_host + d;
                        size_t v_idx = k_idx;
                        h_k_cache[k_idx] = dist(gen);
                        h_v_cache[v_idx] = dist(gen);
                    } } } } }
    printf("Host data initialized.\n");

    printf("\nAllocating device memory...\n");
    float *d_q, *d_k_cache, *d_v_cache, *d_out;
    int *d_block_tables, *d_seq_lens;
    CUDA_CHECK(cudaMalloc(&d_out, out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q, q_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_cache, kv_cache_pool_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_cache, kv_cache_pool_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_tables, block_tables_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seq_lens, seq_lens_size * sizeof(int)));
    printf("Device memory allocated.\n");

    printf("\nCopying data H->D...\n");
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_cache, h_k_cache.data(), kv_cache_pool_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_cache, h_v_cache.data(), kv_cache_pool_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_tables, h_block_tables.data(), block_tables_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_lens, seq_lens.data(), seq_lens_size * sizeof(int), cudaMemcpyHostToDevice));
    printf("Data copied.\n");


    // --- Kernel Launch ---
    dim3 gridDim(num_heads, num_seqs, 1);
    dim3 blockDim(HOST_NUM_THREADS, 1, 1);

    int max_len_in_batch = 0;
    for(int len : seq_lens) max_len_in_batch = std::max(max_len_in_batch, len);
    // Shared mem: logits + reduction_smem workspace
    size_t shared_mem_bytes = (max_len_in_batch + (HOST_NUM_THREADS / WARP_SIZE)) * sizeof(float);
    // If output reduction uses shared mem, add HEAD_SIZE floats. Add some buffer.
    shared_mem_bytes += (head_size * sizeof(float)) + 256; // Add HEAD_SIZE + buffer


    printf("\nLaunching kernel...\n");
    printf("  Grid: (%u, %u, %u), Block: (%u, %u, %u), Shared Mem: %zu bytes\n",
           gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, shared_mem_bytes);

    paged_attn_kernel_t kernel_func = nullptr; // Function pointer

    switch(head_size) {
        case 64:
             kernel_func = paged_attention_v1_kernel_standalone<HOST_NUM_THREADS, 64>;
             break;
        case 128:
             kernel_func = paged_attention_v1_kernel_standalone<HOST_NUM_THREADS, 128>;
             break;
        default:
             fprintf(stderr, "Error: Unsupported HEAD_SIZE %d for templated kernel launch!\n", head_size);
             exit(EXIT_FAILURE);
    }

    // Launch the selected kernel specialization
    kernel_func<<<gridDim, blockDim, shared_mem_bytes>>>(
        d_out, d_q, d_k_cache, d_v_cache, num_kv_heads, scale,
        d_block_tables, d_seq_lens, max_num_blocks_per_seq,
        kv_block_stride_host, kv_head_stride_host,
        block_size
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Kernel finished.\n");

    printf("\nCopying result D->H...\n");
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Result copied.\n");


    printf("\nRunning CPU verification...\n");
    paged_attention_cpu(
        h_out_cpu.data(), h_q.data(), h_k_cache.data(), h_v_cache.data(),
        h_block_tables.data(), seq_lens.data(),
        num_seqs, num_heads, head_size, block_size, max_num_blocks_per_seq,
        num_kv_heads, scale, total_physical_blocks, kv_block_stride_host, kv_head_stride_host
    );
    printf("CPU verification finished.\n");

    printf("\nComparing GPU vs CPU results...\n");
    double max_diff = 0.0;
    int mismatches = 0;
    for (size_t i = 0; i < out_size; ++i) { /* ... comparison ... */
        double diff = std::abs(h_out_gpu[i] - h_out_cpu[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-3) { mismatches++; if(mismatches < 20) { /* print */ } } }
    if (mismatches == 0) { printf("Result: PASS! Max difference = %.6e\n", max_diff); }
    else { printf("Result: FAIL! %d mismatches (%.2f%%). Max difference = %.6e\n", mismatches, (double)mismatches * 100.0 / out_size, max_diff); }


    printf("\nCleaning up...\n");
    CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_k_cache));
    CUDA_CHECK(cudaFree(d_v_cache)); CUDA_CHECK(cudaFree(d_block_tables)); CUDA_CHECK(cudaFree(d_seq_lens));
    printf("Cleanup complete.\n");

    return 0;
}
