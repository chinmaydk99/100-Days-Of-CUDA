import triton
import triton.language as tl
import torch 

@triton.jit
def gptq_kernel(
    a_ptr, # Input Activation function (Non quantized) Shape - [M, K]
    b_ptr, # Quantized weights - [K//8, N] since each int32 can pack in 8 4bit weights
    c_ptr, # Output matrix
    scales_ptr,
    zeros_ptr, # These two are needed for dequantization
    M, N, K, # matrix dimensions
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Quantization parameters
    stride_zeros_g, stride_zeros_n,
    stride_scales_g, stride_scales_n,
    groupsize, # Size of each quantization group
    # Kernel parameters
    BLOCK_SIZE_M : tl.constexpr,
    BLOCK_SIZE_N : tl.constexpr,
    BLOCK_SIZE_K : tl.constexpr,
    GROUP_SIZE_M : tl.constexpr):
    """
    - Full precision B - [K,N] 4 bit --> [K//8, N]
    - We store one scale per group, therefore scales = [K//GROUP_SIZE, N]
    - Zeroes - [K//GROUP_SIZE, N//8]
    """
    pid = tl.program_id(0)

    # total blocks in each dimension
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)

    # We use grouped ordering to improve cache locality
    num_blocks_in_group = GROUP_SIZE_M * num_blocks_n
    group_id = pid // num_blocks_in_group
    group_size = min(GROUP_SIZE_M, num_blocks_m - group_id * GROUP_SIZE_M) # This is to account for the last group which may have less than GROUP_SIZE_M elements

    # Calculating which output block this program is responsible for
    pid_m = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_n = (pid % num_blocks_in_group) // group_size

    # Starting indices for this block
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Handling boundaries
    offset_m = tl.multiple_of(offsets_m, BLOCK_SIZE_M)
    offset_n = tl.multiple_of(offsets_n, BLOCK_SIZE_N)

    offsets_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers for A, B, scales and zeroes
    a_ptrs = a_ptr + offset_m[:, None] * stride_am + offsets_k[None, :] * stride_ak

    b_ptrs =  b_ptr + (offsets_k[:, None] // 8) * stride_bk + offset_n[None, :] * stride_bn # We need to account for the 4-bit quantization

    scales_ptr = scales_ptr + offsets_n * stride_scales_n # Scaling happens per output feature N

    zeros_ptrs = zeros_ptr + ((offsets_n // 8) * stride_zeros_n) # zeros is indexed by N // 8 because each int32 stores 8 zero-points

    # Computing bit shift amounts for unpacking 4-bit values
    # Finding the correct 4-bit weight
    shifter = (offsets_k % 8)* 4 # each 4 bit value shifted by a multiple of 4

    zeros_shifter = (offsets_n % 8) * 4 # each 4 bit value shifted by a multiple of 4

    output = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype = tl.float32)

    # Iterate over the k blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)

        b = tl.load(b_ptrs)

        # Checking which group id this K block belongs to
        # Each group shares the scaling factor and zero point
        g_id = k // (groupsize // BLOCK_SIZE_K)

        # Loading the scale value for this group
        group_scales_ptrs = scales_ptr + g_id * stride_scales_g

        scales = tl.load(group_scales_ptrs)

        # Loading the zero value for this group
        group_zeros_ptrs = zeros_ptrs + g_id * stride_zeros_g

        zeros = tl.load(group_zeros_ptrs)

        # Extracting the correct zero pointer from the packed int 32
        zeros = (zeros >> zeros_shifter) & 0xF

        # Applying the scale to the zero points
        zeros =  (zeros + 1) * scales

        # Extracting the correct 4 bit value from b
        b = (b >> shifter[:, None]) & 0xF

        # Dequantize weights using scales and zero points
        b = b * scales[None, :] - zeros[None, :]

        output += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 8) * stride_bk

    output = output.to(tl.float16)

    # Writing the output block to C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)

    # Only store the output if it's within the bounds of the matrix
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, output, mask=mask)


class GPTQLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, scales, zeros, groupsize = 128):
        M, K = a.shape
        _, N = b.shape

        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 64
        GROUP_SIZE_M = 8

        total_blocks_m = triton.cdiv(M, BLOCK_SIZE_M)
        total_blocks_n = triton.cdiv(N, BLOCK_SIZE_N)
        total_programs = total_blocks_m * total_blocks_n
        grid = (total_programs, 1)

        c = torch.empty((M, N), dtype = a.dtype , device = a.device)

        gptq_kernel[grid](
            a, b, c,
            scales, zeros,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            scales.stride(0), scales.stride(1),
            zeros.stride(0), zeros.stride(1),
            groupsize,
            BLOCK_SIZE_M = BLOCK_SIZE_M,
            BLOCK_SIZE_N = BLOCK_SIZE_N,
            BLOCK_SIZE_K = BLOCK_SIZE_K,
            GROUP_SIZE_M = GROUP_SIZE_M
        )

        return c

def gptq_linear(a, b, scales, zeros, groupsize = 128):
    return GPTQLinear.apply(a, b, scales, zeros, groupsize)