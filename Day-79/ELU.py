import triton
import triton.language as tl


@triton.jit
def elu_kernel(X_ptr,        # *float32* input
               Y_ptr,        # *float32* output
               alpha,        # scalar  α  (float32)
               n_elements,   # total number of elements = M*N
               BLOCK_SIZE: tl.constexpr):

    ## Which slice of the vector this “program” instance will handle
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    ## Guard against running past the end of the array
    mask = offsets < n_elements

    ## Load, apply ELU, store
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    y = tl.where(x > 0.0, x, alpha * (tl.exp(x) - 1.0))
    tl.store(Y_ptr + offsets, y, mask=mask)


import torch
import triton
import triton.language as tl


def elu_triton(x: torch.Tensor, alpha: float = 1.0,
               block_size: int = 1024, num_warps: int = 4):
    """
    In-place ELU for a contiguous matrix `x` (any shape, row-major).
    Returns a new tensor `y`; feel free to call with `x` twice for in-place.
    """
    assert x.is_contiguous(), "Input must be contiguous (row-major)."
    y = torch.empty_like(x)

    n_elems = x.numel()
    grid = lambda meta: (triton.cdiv(n_elems, meta['BLOCK_SIZE']),)

    elu_kernel[grid](                                   # ← launches on GPU
        x, y, alpha, n_elems,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return y
