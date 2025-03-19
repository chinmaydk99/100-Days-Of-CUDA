import torch
import triton
import triton.language as tl
import torch.nn as nn
import math
import time

@triton.jit
def fused_gptq_mlp_kernel(
    x_ptr, #input activations
    output_ptr, # Output Buffer
    gate_qweight_ptr, gate_scales_ptr, gate_zeros_ptr, # Gate Projection Weights
    up_proj_qweight_ptr, up_proj_scales_ptr, up_proj_zeros_ptr, # Up Projection Weights
    down_proj_qweight_ptr, down_proj_scales_ptr, down_proj_zeros_ptr, # Down Projection Weights
    # Matrix Dimensions
    M, # M - batch * seq_len(total number of tokens)
    N, # N - hidden_dim(input Size) 
    K, O , # K- intermediate_dim, O- Output Dimension (= N)
    stride_x_m, stride_x_n,
    stride_output_m, stride_output_o,
    # Gate Projection Stride
    stride_gate_k, stride_gate_n,
    stride_gate_scales_g, stride_gate_scales_n,
    stride_gate_zeros_g, stride_gate_zeros_n,
    # UP projection strides
    stride_up_k, stride_up_n,
    stride_up_scales_g, stride_up_scales_n,
    stride_up_zeros_g, stride_up_zeros_n,
    # Down Projection Strides
    stride_down_n, stride_down_o,
    stride_down_scales_g, stride_down_scales_o,
    stride_down_zeros_g, stride_down_zeros_o,
    groupsize,
    BLOCK_SIZE_M : tl.constexpr,
    BLOCK_SIZE_N : tl.constexpr,
    BLOCK_SIZE_K : tl.constexpr,
    BLOCK_SIZE_O : tl.constexpr,
    GROUP_SIZE_M : tl.constexpr):
    """
    Fused MLP computation for GPTQ:
    1. gate_proj = x @ gate_qweight
    2. up_proj = x @ up_proj_qweight
    3. Apply SiLU activation to gate_proj
    4. Multiply gate_proj * up_proj elementwise
    5. Conpute output = (gate_proj * up_proj) @ down_proj_qweight

    Weights are all 4-bit quantized and need to be dequantized on the fly
    """
    pid = tl.program_id(0)

    # We divide the output matrix [M,O] into blocks of size BLOCK_SIZE_M * BLOCK_SIZE_O
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_blocks_o = tl.cdiv(O, BLOCK_SIZE_O)

    num_blocks_in_group = GROUP_SIZE_M * num_blocks_o
    group_id = pid // num_blocks_in_group
    group_size = min(GROUP_SIZE_M, num_blocks_m - group_id * GROUP_SIZE_M)

    pid_m = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_o = (pid % num_blocks_in_group) // group_size

    # Block Start Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    mask = offs_m < M

    #------------------------------------------------
    # Computing Intermediate Projections (gate and up)
    #------------------------------------------------

    gate_acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    up_acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    # Shifters to obtain the 4 bit values
    shifter = (offs_k % 8) * 4 # Weights are stored in rows
    zeros_shifter = (offs_n % 8) * 4 # Zeroes are stored per column

    # Processing input blocks along the N dimension
    for n_idx in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        n_start = n_idx * BLOCK_SIZE_N
        offs_n_curr = n_start + offs_n

        # Load the input block of size [BLOCK_SIZE_M, BLOCK_SIZE_N]
        x_ptrs = x_ptr + offs_m[:, None] * stride_x_m + offs_n_curr[None, :] * stride_x_n
        x_block = tl.load(x_ptrs, mask=mask[:, None] & (offs_n_curr[None, :] < N))

        # Process K dimension for gate and up blocks
        for k_idx in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_start = k_idx * BLOCK_SIZE_K
            offs_k_curr = k_start + offs_k

            # Quantization Group ID
            g_id = k_start // groupsize

            #--------------------------
            # Gate Projection
            #--------------------------

            # Load Gate quantized weights 
            # [K//8, N]
            gate_qweight_ptrs = gate_qweight_ptr + (offs_k_curr[:, None]//8) * stride_gate_k + (offs_n_curr[None, :]) * stride_gate_n
            gate_qweight = tl.load(gate_qweight_ptrs, mask = (offs_k_curr[:, None]< K) & (offs_n_curr[None, :] < N))

            # Load gate scales and zeros for this quantisation group
            gate_scales_ptrs = gate_scales_ptr + g_id * stride_gate_scales_g + offs_n_curr * stride_gate_scales_n
            gate_scales = tl.load(gate_scales_ptrs, mask = offs_n_curr < N)

            gate_zeros_ptrs = gate_zeros_ptr + g_id * stride_gate_zeros_g + (offs_n_curr // 8) * stride_gate_zeros_n
            gate_zeros = tl.load(gate_zeros_ptrs, mask = (offs_n_curr // 8) < (N//8))

            # Dequantizing Gate Weights
            gate_zeros = (gate_zeros >> zeros_shifter) & 0xF
            gate_zeros = (gate_zeros + 1) * gate_scales

            gate_weights = (gate_qweight >> shifter[:, None]) & 0xF
            gate_weights = gate_weights * gate_scales[None, :] - gate_zeros[None, :]

            #---------------------------
            # Up Projection
            #------------------------
            up_qweight_ptrs = up_proj_qweight_ptr + (offs_k_curr[:, None] // 8) * stride_up_k + offs_n_curr[None, :] * stride_up_n
            up_qweight = tl.load(up_qweight_ptrs, mask=(offs_k_curr[:, None] < K) & (offs_n_curr[None, :] < N))
            
            # Load up scales and zeros
            up_scales_ptrs = up_proj_scales_ptr + g_id * stride_up_scales_g + offs_n_curr * stride_up_scales_n
            up_scales = tl.load(up_scales_ptrs, mask=offs_n_curr < N)
            
            up_zeros_ptrs = up_proj_zeros_ptr + g_id * stride_up_zeros_g + (offs_n_curr // 8) * stride_up_zeros_n
            up_zeros = tl.load(up_zeros_ptrs, mask=(offs_n_curr // 8) < (N // 8))
            
            # Dequantize up weights
            up_zeros = (up_zeros >> zeros_shifter) & 0xF
            up_zeros = (up_zeros + 1) * up_scales
            
            up_weights = (up_qweight >> shifter[:, None]) & 0xF
            up_weights = up_weights * up_scales[None, :] - up_zeros[None, :]

            # Compute matrix multiplications
            gate_acc += tl.dot(x_block, tl.trans(gate_weights))
            up_acc += tl.dot(x_block, tl.trans(up_weights))

    #--------------------
    # Applying SiLU and Element Wise Multiplication
    #---------------------
    gate_silu = gate_acc * (tl.sigmoid(gate_acc))
    intermediate = gate_silu * up_acc

    #---------------------
    # Down Projection
    #---------------------

    output_acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_O], dtype=tl.float32)

    shifter = (offs_n % 8) * 4
    zeros_shifter = (offs_o % 8) * 4

    for n_idx in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        n_start = n_idx * BLOCK_SIZE_N
        offs_n_curr = n_start + offs_n

        inter_block = intermediate[:, n_start : n_start + BLOCK_SIZE_N]

        g_id = n_start // groupsize

        down_qweight_ptrs = down_proj_qweight_ptr + (offs_n_curr[:, None] // 8) * stride_down_n + offs_o[None, :] * stride_down_o
        down_qweight = tl.load(down_qweight_ptrs, mask = (offs_n_curr[:, None] < N) & (offs_o[None, :] < O))

        down_scales_ptrs = down_proj_scales_ptr + g_id * stride_down_scales_g + offs_o * stride_down_scales_o
        down_scales = tl.load(down_scales_ptrs, mask=offs_o < O)

        down_zeros_ptrs = down_proj_zeros_ptr + g_id * stride_down_zeros_g + (offs_o // 8) * stride_down_zeros_o
        down_zeros = tl.load(down_zeros_ptrs, mask = (offs_o // 8) < (O // 8))

        down_zeros = (down_zeros >> zeros_shifter) & 0xF
        down_zeros = (down_zeros + 1) * down_scales
        
        down_weights = (down_qweight >> shifter[:, None]) & 0xF
        down_weights = down_weights * down_scales[None, :] - down_zeros[None, :]
        
        output_acc += tl.dot(inter_block, down_weights)
  
    output = output_acc.to(tl.float16)
    output_ptrs = output_ptr + offs_m[:, None] * stride_output_m + offs_o[None, :] * stride_output_o
    mask = (offs_m[:, None] < M) & (offs_o[None, :] < O)
    tl.store(output_ptrs, output, mask=mask)

class FusedGPTQMLP(nn.Module):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 bits = 4,
                 groupsize = 128):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bits = bits
        self.groupsize = groupsize

        self.features_per_int = 32 // bits

        # Register buffers for gate projection
        # While we want these on the GPU, we dont need gradients to be calculated which is why we register them as buffers

        # Register buffers for gate projection
        self.register_buffer('gate_qweight', torch.zeros(
            (hidden_size // self.features_per_int, intermediate_size), 
            dtype=torch.int32
        ))
        self.register_buffer('gate_scales', torch.zeros(
            (math.ceil(hidden_size / groupsize), intermediate_size), 
            dtype=torch.float16
        ))
        self.register_buffer('gate_zeros', torch.zeros(
            (math.ceil(hidden_size / groupsize), math.ceil(intermediate_size / self.features_per_int)), 
            dtype=torch.int32
        ))
        
        # Register buffers for up projection
        self.register_buffer('up_qweight', torch.zeros(
            (hidden_size // self.features_per_int, intermediate_size), 
            dtype=torch.int32
        ))
        self.register_buffer('up_scales', torch.zeros(
            (math.ceil(hidden_size / groupsize), intermediate_size), 
            dtype=torch.float16
        ))
        self.register_buffer('up_zeros', torch.zeros(
            (math.ceil(hidden_size / groupsize), math.ceil(intermediate_size / self.features_per_int)), 
            dtype=torch.int32
        ))
        
        # Register buffers for down projection
        self.register_buffer('down_qweight', torch.zeros(
            (intermediate_size // self.features_per_int, hidden_size), 
            dtype=torch.int32
        ))
        self.register_buffer('down_scales', torch.zeros(
            (math.ceil(intermediate_size / groupsize), hidden_size), 
            dtype=torch.float16
        ))
        self.register_buffer('down_zeros', torch.zeros(
            (math.ceil(intermediate_size / groupsize), math.ceil(hidden_size / self.features_per_int)), 
            dtype=torch.int32
        ))
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape

        # [b_s, seq_len, h_s] -> [b-s * seq_len, h_s]
        x_2d = x.reshape(-1, hidden_size)

        M = batch_size * seq_len
        N = hidden_size
        K = self.intermediate_size
        O = hidden_size

        output = torch.zeros((M, O), dtype=torch.float16, device=x.device)

        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 32
        BLOCK_SIZE_O = 32
        GROUP_SIZE_M = 8

        grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(O, BLOCK_SIZE_O),)
        
        fused_gptq_mlp_kernel[grid](
            x_2d, output,
            self.gate_qweight, self.gate_scales, self.gate_zeros,
            self.up_qweight, self.up_scales, self.up_zeros,
            self.down_qweight, self.down_scales, self.down_zeros,
            M, N, K, O,
            x_2d.stride(0), x_2d.stride(1),
            output.stride(0), output.stride(1),
            self.gate_qweight.stride(0), self.gate_qweight.stride(1),
            self.gate_scales.stride(0), self.gate_scales.stride(1),
            self.gate_zeros.stride(0), self.gate_zeros.stride(1),
            self.up_qweight.stride(0), self.up_qweight.stride(1),
            self.up_scales.stride(0), self.up_scales.stride(1),
            self.up_zeros.stride(0), self.up_zeros.stride(1),
            self.down_qweight.stride(0), self.down_qweight.stride(1),
            self.down_scales.stride(0), self.down_scales.stride(1),
            self.down_zeros.stride(0), self.down_zeros.stride(1),
            self.groupsize,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_O=BLOCK_SIZE_O,
            GROUP_SIZE_M=GROUP_SIZE_M
        )
        
        return output.reshape(batch_size, seq_len, hidden_size)