import torch
import triton
import triton.language as tl
import random

torch.manual_seed(0)
random.seed(0)
torch.cuda.manual_seed_all(0)

@triton.jit
def layer_norm_fwd_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,
    Mean_ptr, Rstd_ptr,
    stride_x_row, stride_y_row,
    stride_w, stride_b,
    M, N,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)

    row_start_x_ptr = X_ptr + row_idx * stride_x_row
    row_start_y_ptr = Y_ptr + row_idx * stride_y_row

    mean = 0.0
    _sum = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    for block_start in range(0, N, BLOCK_SIZE_N):
        if block_start + BLOCK_SIZE_N > N:
            col_offsets = tl.arange(block_start, N)
        else:
            col_offsets = tl.arange(block_start, block_start + BLOCK_SIZE_N)

        mask = col_offsets < N
        x = tl.load(row_start_x_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        _sum += x

    mean = tl.sum(_sum, axis=0) / N

    _var = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for block_start in range(0, N, BLOCK_SIZE_N):
        if block_start + BLOCK_SIZE_N > N:
            col_offsets = tl.arange(block_start, N)
        else:
            col_offsets = tl.arange(block_start, block_start + BLOCK_SIZE_N)

        mask = col_offsets < N
        x = tl.load(row_start_x_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        _var += x * x

    var = tl.sum(_var, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Uncomment below if storing mean/rstd for backward pass
    # if Mean_ptr is not None:
    #     tl.store(Mean_ptr + row_idx, mean)
    # if Rstd_ptr is not None:
    #     tl.store(Rstd_ptr + row_idx, rstd)

    for block_start in range(0, N, BLOCK_SIZE_N):
        if block_start + BLOCK_SIZE_N > N:
            col_offsets = tl.arange(block_start, N)
        else:
            col_offsets = tl.arange(block_start, block_start + BLOCK_SIZE_N)

        mask = col_offsets < N

        x = tl.load(row_start_x_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + col_offsets, mask=mask).to(tl.float32)
        b = tl.load(B_ptr + col_offsets, mask=mask).to(tl.float32)

        x_norm = (x - mean) * rstd
        y = x_norm * w + b

        tl.store(row_start_y_ptr + col_offsets, y, mask=mask)


def layer_norm(x, weight, bias, eps=1e-5):
    M, N = x.shape
    assert x.is_contiguous()
    assert weight.is_contiguous() and weight.shape == (N,)
    assert bias.is_contiguous() and bias.shape == (N,)

    y = torch.empty_like(x)

    # mean_buf = torch.empty((M,), dtype=torch.float32, device=x.device)
    # rstd_buf = torch.empty((M,), dtype=torch.float32, device=x.device)

    BLOCK_SIZE_N = triton.next_power_of_2(N)

    grid = (M,)

    print(f"Launching Triton LayerNorm kernel with grid={grid}, BLOCK_SIZE_N={BLOCK_SIZE_N}")

    layer_norm_fwd_kernel[grid](
        x, y, weight, bias,
        None, None, # Pass None if mean/rstd aren't stored
        # mean_buf, rstd_buf, # Uncomment if storing mean/rstd
        x.stride(0), y.stride(0),
        weight.stride(0), bias.stride(0),
        M, N,
        eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return y #, mean_buf, rstd_buf


batch_size = 4
num_features = 2048
print(f"Input shape: ({batch_size}, {num_features})")

x = torch.randn(batch_size, num_features, device='cuda', dtype=torch.float32)
weight = torch.randn(num_features, device='cuda', dtype=torch.float32)
bias = torch.randn(num_features, device='cuda', dtype=torch.float32)
eps = 1e-5

print("Calculating LayerNorm using Triton...")
y_triton = layer_norm(x, weight, bias, eps)
print("Triton calculation complete.")

print("Calculating LayerNorm using PyTorch for verification...")
y_torch = torch.nn.functional.layer_norm(x, (num_features,), weight, bias, eps).to(torch.float32)
print("PyTorch calculation complete.")

print("Verifying results...")
if torch.allclose(y_triton, y_torch, atol=1e-2, rtol=1e-2):
    print("Verification PASSED!")
else:
    print("Verification FAILED!")
