#include <cuda_runtime.h>
#include <cmath>
#include <torch/extension.h>

__device__ float GELU_approximation(float x) {
    const float sqrt_2_pi = 0.7978845608f;
    const float c = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_pi * (x + c * x * x * x)));
}

__global__ void GELU_Kernel(const float *x, float *y, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = GELU_approximation(x[idx]);
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    int size = x.numel();
    auto y = torch::empty_like(x);
    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
    GELU_Kernel<<<gridSize, blockSize>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);
    cudaDeviceSynchronize();
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu_cuda", &gelu_cuda, "GELU approximation using CUDA");
}
