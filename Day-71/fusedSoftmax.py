import torch
import triton
import triton.language as tl
import random

torch.manual_seed(0)
random.seed(0)
torch.cuda.manual_seed_all(0)

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_pointers = row_start_ptr + col_offsets

    row_max = -float("inf")
    for block_start in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        col_offsets = block_start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row = tl.load(input_pointers, mask=mask, other=-float("inf"))
        block_max = tl.max(row, axis=0)
        row_max = tl.maximum(row_max, block_max)
        input_pointers += BLOCK_SIZE

    row_sum = 0.0
    input_pointers = row_start_ptr + tl.arange(0, BLOCK_SIZE) # Reset pointers
    for block_start in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        col_offsets = block_start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row = tl.load(input_pointers, mask=mask, other=0.0)
        numerator = tl.exp(row - row_max)
        row_sum += tl.sum(numerator, axis=0)
        input_pointers += BLOCK_SIZE

    inv_row_sum = 1.0 / row_sum

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    input_pointers = row_start_ptr + tl.arange(0, BLOCK_SIZE) # Reset pointers
    output_pointers = output_row_start_ptr + tl.arange(0, BLOCK_SIZE)

    for block_start in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        col_offsets = block_start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        row = tl.load(input_pointers, mask=mask, other=0.0)
        numerator = tl.exp(row - row_max)
        output = numerator * inv_row_sum

        tl.store(output_pointers, output, mask=mask)

        input_pointers += BLOCK_SIZE
        output_pointers += BLOCK_SIZE


def softmax(x: torch.Tensor):
    n_rows, n_cols = x.shape
    assert x.is_contiguous()

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    y = torch.empty_like(x)

    grid = (n_rows,)

    print(f"Launching Triton Softmax kernel with grid={grid}, BLOCK_SIZE={BLOCK_SIZE}, num_warps={num_warps}")

    softmax_kernel[grid](
        y, x,
        x.stride(0), y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


batch_size = 8
num_features = 4096
print(f"Input shape: ({batch_size}, {num_features})")

x = torch.randn(batch_size, num_features, device='cuda', dtype=torch.float32)

print("Calculating Softmax using Triton...")
y_triton = softmax(x)
print("Triton calculation complete.")

print("Calculating Softmax using PyTorch for verification...")
y_torch = torch.softmax(x, axis=1).to(torch.float32)
print("PyTorch calculation complete.")

print("Verifying results...")
if torch.allclose(y_triton, y_torch, atol=1e-2, rtol=1e-2):
    print("Verification PASSED!")
else:
    print("Verification FAILED!")
