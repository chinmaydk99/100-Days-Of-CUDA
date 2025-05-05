import torch, triton, triton.language as tl
from torch.autograd.function import once_differentiable

# ─────────────────────────────────────────────────────────────
#  Kernel parameters
#     BLOCK_M × BLOCK_N  – output tile size computed by one CTA
#     BLOCK_K            – K‑dimension chunk each CTA loads per step
# ─────────────────────────────────────────────────────────────
@triton.jit
def linear_kernel(X_ptr, W_ptr, B_ptr, Y_ptr,
                  M, N, K,
                  stride_xm, stride_xk,
                  stride_wn, stride_wk,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid   = tl.program_id(axis=0)                         # each CTA handles one tile of C (M×N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid // num_pid_m
    pid_n = pid %  num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)      # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)      # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                        # [BLOCK_K]

    # pointers for output tile
    X_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)   # [M,K]
    W_ptrs = W_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)   # [N,K]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # FP32 accumulator

    # loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(X_ptrs, mask=offs_m[:, None] < M)            # [M,Kc]
        w = tl.load(W_ptrs, mask=offs_n[:, None] < N)            # [N,Kc]
        acc += tl.dot(x, tl.trans(w))                            # GEMM micro‑tile
        X_ptrs += BLOCK_K * stride_xk
        W_ptrs += BLOCK_K * stride_wk

    # add bias (broadcast on rows)
    if B_ptr is not None:
        b = tl.load(B_ptr + offs_n, mask=offs_n < N).to(acc.dtype)
        acc += b[None, :]

    # write‑back
    Y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(Y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class TritonLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None):
        M, K = x.shape
        N    = weight.shape[0]

        y = torch.empty((M, N), device=x.device, dtype=torch.float32)

        # launch‑config
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = lambda meta: (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

        linear_kernel[grid](
            x, weight, bias, y,
            M, N, K,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )
        ctx.save_for_backward(x, weight, bias)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        raise NotImplementedError("Add a backward kernel or fall back to torch.mm().")

# convenience wrapper
class LinearTriton(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        return TritonLinear.apply(x, self.weight, self.bias)
