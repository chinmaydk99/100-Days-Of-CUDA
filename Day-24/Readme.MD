# FlashAttention-2 CUDA Implementation Guide

This document provides a comprehensive explanation of the FlashAttention-2 algorithm and its CUDA implementation. It maps the mathematical formulation from the paper to the actual code, providing intuition and reasoning behind each step.

## Table of Contents

1. [Introduction](#introduction)
2. [Algorithm Overview](#algorithm-overview)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Implementation Walkthrough](#implementation-walkthrough)
5. [Key Optimizations](#key-optimizations)
6. [Common Pitfalls](#common-pitfalls)
7. [Performance Considerations](#performance-considerations)
8. [References](#references)

## Introduction

FlashAttention-2 is a memory-efficient attention algorithm designed to enable processing of longer sequences than traditional attention mechanisms. The core innovation lies in how it processes the attention computation in blocks, avoiding the need to materialize the full attention matrix in memory.

### Problem Statement

The standard attention mechanism requires O(N²) memory, which becomes prohibitive for long sequences. FlashAttention-2 reduces this to O(N) memory by computing attention block-by-block while maintaining mathematical equivalence to the standard approach.

## Algorithm Overview

At a high level, FlashAttention-2:

1. Divides Q, K, V matrices into blocks
2. Processes each block sequentially while maintaining running statistics
3. Updates outputs incrementally using an "online" softmax calculation
4. Avoids storing the full N×N attention matrix

### Key Concepts

- **Blocked Processing**: Computing attention in small blocks that fit in fast memory
- **Online Softmax**: Computing softmax incrementally across blocks
- **Tiling**: Moving data between HBM (High Bandwidth Memory) and SRAM (on-chip memory)

## Mathematical Foundations

### Standard Attention Calculation

The standard attention formula is:

```
O = softmax(QK^T / √d) · V
```

Where:
- Q, K, V are the query, key, and value matrices
- d is the dimension of the embedding
- Output O contains the attention-weighted values

### FlashAttention-2 Reformulation

FlashAttention-2 reformulates this calculation to work block-by-block:

1. Divide Q into Tr blocks Q₁, Q₂, ..., QTr (each of size Br × d)
2. Divide K, V into Tc blocks K₁, K₂, ..., KTc and V₁, V₂, ..., VTc (each of size Bc × d)
3. Divide output O into Tr blocks O₁, O₂, ..., OTr (each of size Br × d)

For each query block Qᵢ, we compute:

```
mᵢ⁽ʲ⁾ = max(mᵢ⁽ʲ⁻¹⁾, rowmax(Sᵢⱼ))
P̃ᵢⱼ = exp(Sᵢⱼ - mᵢ⁽ʲ⁾)
ℓᵢ⁽ʲ⁾ = exp(mᵢ⁽ʲ⁻¹⁾ - mᵢ⁽ʲ⁾) · ℓᵢ⁽ʲ⁻¹⁾ + rowsum(P̃ᵢⱼ)
Oᵢ⁽ʲ⁾ = diag(exp(mᵢ⁽ʲ⁻¹⁾ - mᵢ⁽ʲ⁾))⁻¹ · Oᵢ⁽ʲ⁻¹⁾ + P̃ᵢⱼ/ℓᵢ⁽ʲ⁾ · Vⱼ
```

Where:
- mᵢ⁽ʲ⁾ tracks the maximum value seen so far (for numerical stability)
- ℓᵢ⁽ʲ⁾ tracks the sum of exponentials (for normalization)
- P̃ᵢⱼ contains the exponentials of the current block
- Oᵢ⁽ʲ⁾ is the output accumulated so far

### Online Softmax Intuition

The key insight is that softmax can be computed incrementally across blocks:

1. As we process each new block, we might find larger values that change our normalization
2. When this happens, we need to rescale our previous computations
3. The formulas above handle this rescaling precisely, ensuring numerical stability

## Implementation Walkthrough

### Kernel Structure

The CUDA kernel is organized around these key components:

```cpp
__global__ void flash_attention_kernel(
    const float *Q, const float *K, const float *V, float *O, float *L,
    int batch_size, int num_heads, int seq_len, int d
) {
    // 1. Set up indexing
    // 2. Allocate shared memory
    // 3. Load query block
    // 4. Initialize accumulators
    // 5. Process each key/value block
    //    a. Save previous softmax state
    //    b. Load current K, V block
    //    c. Compute scores and update max
    //    d. Compute normalization and exponentials
    //    e. Update output
    // 6. Write final results to global memory
}
```

Let's examine each step in detail:

### 1. Set up indexing

```cpp
int batch_id = blockIdx.z;
int query_block_id = blockIdx.x;
int head_id = blockIdx.y;

int local_row = threadIdx.x; // Local position within the block
int global_query_idx = query_block_id*B_r + local_row;

int tid_y = threadIdx.y;

// Total Number of key value blocks
int T_c = (seq_len + B_c - 1)/ B_c;

// Starting point for current batch and head
size_t base_offset = ((size_t)batch_id * num_heads + head_id) * seq_len * d;
```

This establishes:
- Which batch and attention head we're processing
- Which query block (i) this CUDA block is handling
- The mapping from thread positions to data positions
- How many key/value blocks (j) we'll need to process

### 2. Allocate shared memory

```cpp
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
```

This allocates:
- Space for the current query block (B_r × d)
- Space for the current key block (d × B_c) in transposed format
- Space for the current value block (B_c × d)
- Arrays to track softmax statistics (max and sum values)
- Space for the output block being accumulated

### 3. Load query block

```cpp
// Loading the current query block into shared memory
if(global_query_idx < seq_len && tid_y < d){
    int q_idx = base_offset + global_query_idx * d + tid_y;
    Q_shared[local_row][tid_y] = Q[q_idx];
}
```

This loads the query block Qᵢ from global memory to shared memory. The bounds check ensures we don't access invalid memory.

### 4. Initialize accumulators

```cpp
// Initialising softmax accumulators and output
if(tid_y == 0 && global_query_idx < seq_len){
    m_curr[local_row] = -INFINITY;
    l_curr[local_row] = 0.0f;

    for(int feat = 0; feat < d; feat++){
        O_shared[local_row][feat] = 0.0f;
    }
}
```

This initializes:
- m_curr to -INFINITY (initial max value)
- l_curr to 0 (initial sum value)
- O_shared to zeros (initial output)

These correspond to the paper's initialization:
- mᵢ⁽⁰⁾ = -∞
- ℓᵢ⁽⁰⁾ = 0
- Oᵢ⁽⁰⁾ = 0

### 5. Process each key/value block

The main loop processes each key/value block (j from 0 to T_c-1):

```cpp
for(int j = 0; j < T_c; j++){
    // 5a. Save previous softmax state
    if(tid_y == 0 && global_query_idx < seq_len){
        m_prev[local_row] = m_curr[local_row];
        l_prev[local_row] = l_curr[local_row];
    }
    __syncthreads();

    // 5b. Load K, V blocks
    int key_block_start = j * B_c;
    // ... (loading code)
    __syncthreads();

    // 5c-e. Compute scores, update max, compute normalization, update output
    if(tid_y == 0 && global_query_idx < seq_len){
        // ... (computation code)
    }
    __syncthreads();
}
```

#### 5a. Save previous softmax state

Before processing each new block, we save the current state as the previous state. This gives us mᵢ⁽ʲ⁻¹⁾ and ℓᵢ⁽ʲ⁻¹⁾ needed for the update formulas.

#### 5b. Load K, V blocks

```cpp
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
```

This loads the key block Kⱼ and value block Vⱼ from global memory to shared memory. Key points:

- We load K in transposed form for efficient computation of Q·K^T
- We use blockDim.y threads per row to parallelize loading
- We have bounds checks to handle sequence boundaries

#### 5c. Compute scores and update max

```cpp
float m_i_j = m_prev[local_row];

for(int key_idx = 0; key_idx < B_c && key_idx + key_block_start < seq_len; key_idx++){
    float s = 0.0f;
    for(int feat = 0; feat < d; feat++){
        s += Q_shared[local_row][feat] * K_shared[feat][key_idx];
    }
    s *= rsqrtf((float)d);
    
    m_i_j = fmaxf(m_i_j, s);
}
```

This computes the attention scores Sᵢⱼ = Qᵢ·Kⱼ^T / √d and updates the maximum value mᵢ⁽ʲ⁾ = max(mᵢ⁽ʲ⁻¹⁾, rowmax(Sᵢⱼ)).

#### 5d. Compute normalization and exponentials

```cpp
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
    s *= rsqrtf((float)d);

    float p_ij = expf(s - m_i_j);
    P_sums[key_idx] = p_ij;

    l_i_j += p_ij;
}
```

This:
1. Updates the sum value from previous blocks: ℓᵢ⁽ʲ⁾ = exp(mᵢ⁽ʲ⁻¹⁾ - mᵢ⁽ʲ⁾) · ℓᵢ⁽ʲ⁻¹⁾
2. Computes the exponentials for the current block: P̃ᵢⱼ = exp(Sᵢⱼ - mᵢ⁽ʲ⁾)
3. Adds them to the running sum: ℓᵢ⁽ʲ⁾ += rowsum(P̃ᵢⱼ)

#### 5e. Update output

```cpp
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
```

This implements the key formula from the paper:
```
Oᵢ⁽ʲ⁾ = diag(exp(mᵢ⁽ʲ⁻¹⁾ - mᵢ⁽ʲ⁾))⁻¹ · Oᵢ⁽ʲ⁻¹⁾ + P̃ᵢⱼ/ℓᵢ⁽ʲ⁾ · Vⱼ
```

1. First, we scale the previous output by the change in max value
2. Then, we add the contribution from the current block
3. Finally, we save the updated output and softmax statistics

### 6. Write final results

```cpp
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
```

This writes the final results back to global memory:
1. The output values O
2. The logsumexp values L = m + log(ℓ)

## Key Optimizations

### 1. Memory Efficiency

FlashAttention-2 achieves memory efficiency through:
- Processing data in blocks that fit in fast on-chip memory
- Avoiding materializing the full attention matrix
- Reusing memory for different blocks
- Storing intermediate results compactly

### 2. Online Softmax

The online softmax computation is a critical optimization that:
- Allows processing arbitrary sequence lengths with fixed memory
- Maintains numerical stability through the max subtraction trick
- Correctly rescales earlier computations when new max values are found

### 3. Matrix Layout

- K is stored in transposed form in shared memory for efficient dot products
- Thread organization maximizes parallel processing

## Common Pitfalls

1. **Indexing Errors**: Confusing local vs. global indices
2. **Memory Access Patterns**: Inefficient access patterns can degrade performance
3. **Softmax Stability**: Forgetting to handle the case when l_prev = 0
4. **Boundary Conditions**: Missing checks for sequence boundaries
5. **Synchronization**: Missing __syncthreads() between memory operations
6. **Memory Limits**: Exceeding shared memory capacity 

## Performance Considerations

1. **Block Sizes**: Choose B_r and B_c to balance occupancy and memory usage
2. **Thread Organization**: Use enough threads for good parallelism, but not so many that you waste resources
3. **Memory Coalescing**: Ensure global memory accesses are coalesced
4. **Bank Conflicts**: Organize shared memory to minimize bank conflicts

## References

1. Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv.
2. Rabe, M. S., & Staats, C. (2021). Self-attention does not need O(n²) memory.
3. Hua, W., Dai, Z., Tao, L., Tian, X., Wang, D., Huang, J., ... & Yang, Z. (2022). Transformer quality in linear time.