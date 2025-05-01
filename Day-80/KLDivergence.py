import torch
import triton
import triton.language as tl

# Triton kernel
@triton.jit
def kl_kernel(P_ptr, Q_ptr, output_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    P = tl.load(P_ptr + offsets, mask=mask)
    Q = tl.load(Q_ptr + offsets, mask=mask)

    epsilon = 1e-10
    P = tl.maximum(P, epsilon)
    Q = tl.maximum(Q, epsilon)

    log_term = tl.log(P / Q)
    kl_part = P * log_term

    tl.store(output_ptr + offsets, kl_part, mask=mask)

# Python wrapper
def kl_divergence_triton(P: torch.Tensor, Q: torch.Tensor):
    assert P.shape == Q.shape and P.ndim == 1, "Only 1D tensors supported for now"
    assert P.is_cuda and Q.is_cuda, "Inputs must be CUDA tensors"

    n_elements = P.numel()
    BLOCK_SIZE = 1024
    output = torch.empty_like(P)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    kl_kernel[grid](P, Q, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return torch.sum(output)

# Example usage
P = torch.tensor([0.1, 0.4, 0.5], dtype=torch.float32, device='cuda')
Q = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32, device='cuda')

kl = kl_divergence_triton(P, Q)
print("KL Divergence:", kl.item())
