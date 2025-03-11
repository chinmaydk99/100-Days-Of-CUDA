import torch
import triton
import triton.language as tl

@triton.jit
def single_stage_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.reduce(x, axis=0, combine_fn=add_fn)
    tl.atomic_add(y_ptr, block_sum)

@triton.jit
def add_fn(a, b):
    return a + b

def single_stage_reduction(x: torch.Tensor, BLOCK_SIZE=1024) -> torch.Tensor:
    assert x.is_cuda, "Input must be on GPU"
    N = x.numel()
    y = torch.zeros(1, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    single_stage_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()
    return y[0]

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
        plot_name="single-stage-reduction",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.sum(x), quantiles=quantiles)
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: single_stage_reduction(x), quantiles=quantiles)
    return ms, min_ms, max_ms

if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True)
