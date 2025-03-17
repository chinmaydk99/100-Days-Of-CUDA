import triton
import triton.language as tl
import torch

@triton.jit
def swizzle_tile(pid,
    m, n,
    block_m : tl.constexpr,
    block_n : tl.constexpr,
    group_m : tl.constexpr):

    total_groups_m = tl.cdiv(m, block_m)
    total_groups_n = tl.cdiv(n, block_n)

    num_groups_per_block = group_m * total_groups_n
    group_id = pid // num_groups_per_block

    group_size = min(group_m , total_groups_m - group_id * group_m)

    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % num_groups_per_block) // group_size

    return pid_m, pid_n

@triton.jit
def splitk_gptq_kernel(
    a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
    m, n, k,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    groupsize,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Program ID for processing the M and N dimensions
    pid_mn = tl.program_id(0)
    pid_m, pid_n = swizzle_tile(pid_mn, m, n, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
    
    # Program ID for processing the K dimension (split-K)
    pid_k = tl.program_id(1)
    total_blocks_k = tl.cdiv(k, BLOCK_SIZE_K * SPLIT_K)
    
    # Calculate offsets for accessing matrices
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Ensure coalesced memory access
    offsets_am = tl.max_contiguous(tl.multiple_of(offsets_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offsets_bn = tl.max_contiguous(tl.multiple_of(offsets_n, BLOCK_SIZE_N), BLOCK_SIZE_N)
    
    # Compute pointers to input matrices
    a_ptrs = a_ptr + offsets_am[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_ptrs = b_ptr + (offsets_k[:, None] // 8) * stride_bk + offsets_bn[None, :] * stride_bn
    
    # Compute pointers to scales and zeros
    scales_ptrs = scales_ptr + offsets_n * stride_scales_n
    zeros_ptrs = zeros_ptr + (offsets_n // 8) * stride_zeros_n
    
    # Compute bit shifters for 4-bit values
    shifter = (offsets_k % 8) * 4
    zeros_shifter = (offsets_n % 8) * 4
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main loop over K dimension
    for k_idx in range(total_blocks_k):
        # Load inputs
        a = tl.load(a_ptrs)
        b_quant = tl.load(b_ptrs)
        
        # Quantization group id
        g_id = (k_idx * SPLIT_K + pid_k) * BLOCK_SIZE_K // groupsize
        
        # Load scales and zeros for current group
        group_scale_ptr = scales_ptrs + g_id * stride_scales_g
        scales = tl.load(group_scale_ptr)
        
        group_zero_ptr = zeros_ptrs + g_id * stride_zeros_g
        zeros = tl.load(group_zero_ptr)
        
        # Extract and dequantize 4-bit weights
        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) * scales
        
        b = (b_quant >> shifter[:, None]) & 0xF
        b = b * scales[None, :] - zeros[None, :]
        
        # Compute dot product and accumulate
        acc += tl.dot(a, b)
        
        # Move pointers for next iteration
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 8) * SPLIT_K * stride_bk
    
    # Convert accumulator to float16 (if needed)
    acc = acc.to(tl.float16)
    
    # Calculate output pointers
    c_ptrs = c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    
    # Atomic add for split-K reduction
    tl.atomic_add(c_ptrs, acc)

class SplitK_GPTQLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, scales, zeros, groupsize=128):
        M, K = a.shape
        _, N = b.shape
        
        # Block sizes and parallelization parameters
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 64
        GROUP_SIZE_M = 8
        SPLIT_K = 4
        
        # Calculate grid dimensions
        total_blocks_m = triton.cdiv(M, BLOCK_SIZE_M)
        total_blocks_n = triton.cdiv(N, BLOCK_SIZE_N)
        total_programs = total_blocks_m * total_blocks_n
        grid = (total_programs, SPLIT_K)
        
        # Initialize output tensor
        c = torch.zeros((M, N), dtype=a.dtype, device=a.device)
        
        # Launch kernel
        splitk_gptq_kernel[grid](
            a, b, c,
            scales, zeros,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            scales.stride(0), scales.stride(1),
            zeros.stride(0), zeros.stride(1),
            groupsize,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            SPLIT_K=SPLIT_K
        )
        
        return c
    
    @staticmethod
    def backward(ctx, grad_output):
        # Backward not implemented yet
        return None, None, None, None, None

def splitk_gptq_linear(a, b, scales, zeros, groupsize=128):
    return SplitK_GPTQLinear.apply(a, b, scales, zeros, groupsize)