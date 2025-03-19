import torch
import triton
import triton.language as tl
import torch.nn as nn
from abc import abstractmethod
from logging import getLogger
import tqdm
from transformers.models.llama.modeling_llama import LlamaMLP
import math

try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

class TritonModuleMixin:
    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        pass

@triton.jit
def silu(x):
  return x*tl.sigmoid(x)

class FusedBaseModule(nn.Module, TritonModuleMixin):
    @classmethod
    @abstractmethod
    def inject_to_model(cls, *args, **kwargs): # Subclasses need to implement their own injection logic
        raise NotImplementedError()

logger = getLogger(__name__)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def quant_fused_matmul_kernel(
    a_ptr, # [M, K]
    c_ptr, # [M, N]
    b1_ptr, # [K//8, N] # The weights are quantized
    scales1_ptr, # [1, N]
    zeros1_ptr, # [1, N//8]
    g1_ptr, # Help us understand which scale and zero to use for a given K row in K, Shape [K,]
    b2_ptr, # [K//8, N]
    scales2_ptr, # [1, N]
    zeros2_ptr, # [1, N//8]
    g2_ptr, # [K,]
    M,
    N,
    K,
    bits,
    maxq, # What is this
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales, stride_zeros,
    BLOCK_SIZE_M : tl.constexpr,
    BLOCK_SIZE_N : tl.constexpr,
    BLOCK_SIZE_K : tl.constexpr,
    GROUP_SIZE_M : tl.constexpr
    ):

    features_per_byte = 32 // bits

    pid = tl.program_id(0)
    # Tile swizzling
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group =  GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(GROUP_SIZE_M , num_pid_m - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) # Offsets for rows of A and C
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) # Offsets for columns of B and C
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None]* stride_am + offs_k[None, :] * stride_ak)
    a_mask = offs_am[:, None] < M

    b1_ptrs = b1_ptr + ((offs_k[:, None] // features_per_byte)* stride_bk +
                       offs_bn[None:,] * stride_bn)
    b2_ptrs = b2_ptr + ((offs_k[:, None] // features_per_byte)* stride_bk +
                       offs_bn[None:,] * stride_bn)

    # Group Index pointers
    # Reason for offs_k ? The quantisation is performed per group (per row in K)
    g1_ptrs = g1_ptr + offs_k # (BLOCK_SIZE_K,) → Fetches group indices for `B1` in current K tile
    g2_ptrs = g2_ptr + offs_k # (BLOCK_SIZE_K,) → Fetches group indices for `B2` in current K tile

    # Shifters to extract the N bits of each element in the 32-bit word
    # These are stored per column(feature) . Scales are not packed whereas zeros are
    scales_1_ptrs = scales1_ptr + offs_bn[None,:]
    scales_2_ptr = scales2_ptr + offs_bn[None,:]
    zeros_1_ptrs = zeros1_ptr + (offs_bn[None,:] // features_per_byte)
    zeros_2_ptrs = zeros2_ptr + (offs_bn[None,:] // features_per_byte)

    shifter = (offs_k % features_per_byte) * bits # Since weights are stored per K Row
    zeros_shifter = (offs_bn % features_per_byte) * bits # This is used to extract the zeroes(stored per column)

    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)

    for k in range(0, num_pid_k): # Looping over the K tiles
        # Load the current quantisation group
        g1_idx = tl.load(g1_ptrs)
        g2_idx = tl.load(g2_ptrs)

        scales1 = tl.load(scales1_ptr + g1_idx[:, None]*stride_scales) # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
        scales2 = tl.load(scales2_ptr + g2_idx[:, None]*stride_scales) #(BLOCK_SIZE_K, BLOCK_SIZE_N,)

        zeros1 = tl.load(zeros1 + g1_idx[:, None]*stride_zeros) *(BLOCK_SIZE_K, BLOCK_SIZE_N,)
        zeros1  = (zeros1 >> zeros_shifter) & maxq
        zeros1 = zeros1 + 1

        zeros2 = tl.load(zeros2 + g2_idx[:, None]*stride_zeros)
        zeros2 = (zeros2 >> zeros_shifter) & maxq
        zeros2 = zeros2 + 1

        a = tl.load(a_ptrs, mask=a_mask, other = 0.0) # [BLOCK_SIZE_M, BLOCK_SIZE_K]
        b1 = tl.load(b1_ptrs)
        b2 = tl.load(b2_ptrs)

        # Unpack the quantized weights B1 and B2
        b1 = (b1 >> shifter[None, :]) & maxq
        b1 = (b1 - zeros1) * scales1
        acc1 += tl.dot(a, b1)

        b2 = (b2 >> shifter[None, :]) & maxq
        b2 = (b2 - zeros2) * scales2
        acc2 += tl.dot(a, b2)

        a_ptrs += BLOCK_SIZE_K # Loading Next tile of K within the same row
        b1_ptrs += (BLOCK_SIZE_K // features_per_byte) * stride_bk
        b2_ptrs += (BLOCK_SIZE_K // features_per_byte) * stride_bk
        g1_ptrs += BLOCK_SIZE_K
        g2_ptrs += BLOCK_SIZE_K

    acc1 = silu(acc1)
    c = acc1 * acc2
    c = c.to(tl.float16)

    c_ptrs = c + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

class FusedLlamaMLPQuantized(FusedBaseModule):
    def __init__(self, gate_proj, down_proj, up_proj):
        super().__init__()
        self.infeatures = gate_proj.in_features
        self.outfeatures = gate_proj.out_features
        self.intermediate_size = gate_proj.intermediate_size
        self.bits = gate_proj.bits
        self.maxq = gate_proj.maxq

        self.gate_proj = gate_proj # Expands features from d_model to d_ff
        self.down_proj = down_proj
        self.up_proj = up_proj

    def forward(self, x):
        return self.down_proj(self.mlp_triton_llama(x))

    def mlp_triton_llama(self, x):
        with torch.cuda.device(x.device):
            out_shape = x.shape[:-1] + (self.intermediate_size,) #[batch, seq_len, d_model] -> [batch, seq_len, d_ff]

            # Flatten Batch dimensions
            x = x.reshape(-1,x.shape[-1]) # [batch, seq_len, d_model] -> [batch * seq_len, d_model]

            M, K = x.shape
            N = self.intermediate_size

            c = torch.empty((M, N), device = x.device, dtype = torch.float16)

            grid = lambda META:(
                triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            )

            quant_fused_matmul_kernel[grid](
                x,
                c,
                self.gate_proj.weight,
                self.gate_proj.scales,
                self.gate_proj.qzeros,
                self.gate_proj.g_idx,
                self.up_proj.weight,
                self.up_proj.scales,
                self.up_proj.qzeros,
                self.up_proj.g_idx,
                M, N, K,
                self.bits,
                self.maxq,
                x.stride(0),
                x.stride(1),
                self.gate_proj.weight.stride(0),
                self.gate_proj.weight.stride(1),
                c.stride(0),
                c.stride(1),
                self.gate_proj.scales.stride(0),
                self.gate_proj.qzeros.stride(0),
            )

            c = c.reshape(out_shape)
            return c

    @classmethod
    def inject_to_model(cls, model, use_triton=False, **kwargs):
        """
        Replaces all LlamaMLP layers in the given model with our Triton-optimized FusedLlamaMLPQuantized model

        Args:
            model (torch.nn.Module): The model to modify.
            use_triton (bool): Whether to use Triton for acceleration.
            kwargs: Additional arguments (not used here).
        """

        if not use_triton:
            logger.warning(
                f"Skipping module injection for {cls.__name__} as currently not supported with use_triton=False."
            )
            return

        elif not TRITON_AVAILABLE:
            logger.warning(
                f"Skipping module injection for {cls.__name__} as Triton is not available. Please check your installation."
            )
            return

        for name, m in model.named_modules():
            if not isinstance(m, LlamaMLP):
                continue

            mlp = cls(m.gate_proj, m.down_proj, m.up_proj)

            if "." in name:
                parent_name = name.rsplit(".", 1)[0]
                child_name = name[len(parent_name) + 1 :]
                parent = model.get_submodule(parent_name)
            else:
                parent_name = ""
                parent = model
                child_name = name

            setattr(parent, child_name, mlp)

    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        from tqdm import tqdm

        kn_values = {}

        for _, m in model.named_modules():
            if not isinstance(m, cls):
                continue

            k = m.infeatures
            n = m.intermediate_size

            if (k, n) not in kn_values:
                kn_values[(k, n)] = m

        logger.info(f"Found {len(kn_values)} unique fused mlp KN values.")
        logger.info("Warming up autotune cache ...")
        with torch.no_grad():
            for m in tqdm(range(0, math.ceil(math.log2(seqlen)) + 1)):
                m = 2**m
                for (k, n), (modules) in kn_values.items():
                    a = torch.randn(m, k, dtype=torch.float16, device=model.device)
                    modules.mlp_triton_llama(a)
        del kn_values

