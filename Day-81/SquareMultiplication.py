import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid // N
    col = pid % N

    # Offsets for this program instance
    row_offsets = row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for k in range(0, N, BLOCK_SIZE):
        a = tl.load(A_ptr + row_offsets[:, None] * N + (k + tl.arange(0, BLOCK_SIZE)[None, :]), mask=(row_offsets[:, None] < N) & (k + tl.arange(0, BLOCK_SIZE)[None, :] < N), other=0.0)
        b = tl.load(B_ptr + (k + tl.arange(0, BLOCK_SIZE)[:, None]) * N + col_offsets[None, :], mask=(k + tl.arange(0, BLOCK_SIZE)[:, None] < N) & (col_offsets[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    # Write back to C
    mask = (row_offsets[:, None] < N) & (col_offsets[None, :] < N)
    tl.store(C_ptr + row_offsets[:, None] * N + col_offsets[None, :], acc, mask=mask)

# === Host Code ===
def matmul_torch_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    N = A.shape[0]
    C = torch.empty((N, N), device='cuda', dtype=torch.float32)

    grid = lambda META: (N * N // (META['BLOCK_SIZE'] * META['BLOCK_SIZE']),)

    matmul_kernel[grid](
        A, B, C,
        N,
        BLOCK_SIZE=16
    )

    return C
