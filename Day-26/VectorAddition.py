import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr):
    # Getting the program id
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Ensuring no out of bound data access
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask = mask)
    y = tl.load(y_ptr + offsets, mask = mask)

    output = x + y
    tl.store(output_ptr + offsets, output, mask = mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x, device = DEVICE)
    assert x.device.type == DEVICE.type and y.device.type == DEVICE.type and output.device.type == DEVICE.type

    n_elements = output.numel()

    # Specifying how many parallel programs need to be launched
    # Here I am launching a 1D block of size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Meta parameters don't change at runtime but are used to generate optimised GPU Code
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE = 1024)

    return output

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='vector-add-performance',
        args={},
    ))


def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)