import torch
import triton
import triton.language as tl
import random

torch.manual_seed(0)
random.seed(0)
torch.cuda.manual_seed_all(0)

@triton.jit
def reduce_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    block_sum = tl.sum(x, axis=0)

    tl.atomic_add(output_ptr, block_sum)

def reduce_sum_triton(x: torch.Tensor):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    if not triton.next_power_of_2(BLOCK_SIZE) == BLOCK_SIZE:
         print(f"Warning: BLOCK_SIZE ({BLOCK_SIZE}) is not a power of 2. Adjusting for tl.sum compatibility.")

    output = torch.zeros((1,), device=x.device, dtype=x.dtype)

    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    grid = (grid_size,)

    print(f"Launching Triton kernel with grid={grid}, BLOCK_SIZE={BLOCK_SIZE}")

    reduce_kernel[grid](
        x,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output[0]

N = 1 << 20
print(f"Input size: {N} elements")

h_input = torch.ones(N, device='cuda', dtype=torch.float32)

print("Calculating sum using Triton...")
triton_result = reduce_sum_triton(h_input)
print("Triton calculation complete.")

print("Calculating sum using PyTorch for verification...")
torch_result = torch.sum(h_input)
print("PyTorch calculation complete.")

print(f"Triton Result: {triton_result.item():.6f}")
print(f"PyTorch Result: {torch_result.item():.6f}")

tolerance = 1e-1
if torch.allclose(triton_result, torch_result, atol=tolerance, rtol=tolerance):
    print("Verification PASSED!")
else:
    print("Verification FAILED!")
    print(f"Difference: {abs(triton_result.item() - torch_result.item())}")
