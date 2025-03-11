import torch
import triton
import triton.language as tl

@triton.jit
def local_reduce_kernel(x_ptr, partial_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.reduce(x, axis=0, combine_fn=add_fn)
    tl.store(partial_ptr + pid, block_sum)

@triton.jit
def final_reduce_kernel(partial_ptr, out_ptr, num_blocks, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    x = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.reduce(x, axis=0, combine_fn=add_fn)
    tl.atomic_add(out_ptr, block_sum)

@triton.jit
def add_fn(a, b):
    return a + b

def hierarchical_reduction(x: torch.Tensor, BLOCK_SIZE=1024) -> torch.Tensor:
    assert x.is_cuda, "Input must be on GPU"
    N = x.numel()
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    partial_sums = torch.empty(num_blocks, device=x.device, dtype=x.dtype)
    grid = (num_blocks,)
    local_reduce_kernel[grid](x, partial_sums, N, BLOCK_SIZE=BLOCK_SIZE)
    out = torch.zeros(1, device=x.device, dtype=x.dtype)
    final_num_blocks = (num_blocks + BLOCK_SIZE - 1) // BLOCK_SIZE
    final_reduce_kernel[(final_num_blocks,)](partial_sums, out, num_blocks, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()
    return out[0]

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(20, 28)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="ms",
        plot_name="multi-stage-reduction",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.sum(x), quantiles=quantiles)
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: hierarchical_reduction(x), quantiles=quantiles)
    return ms, min_ms, max_ms

if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True)
