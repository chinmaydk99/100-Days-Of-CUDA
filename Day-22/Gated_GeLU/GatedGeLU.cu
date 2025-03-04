#include <cmath>
#include <cuda_runtime.h>
#include <torch/extensions.h>

__device__ float GELU_approximation(float x) {
    const float sqrt_2_pi = 0.7978845608f;
    const float c = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_pi * (x + c * x * x * x)));
}

__device__ float Sigmoid(float x){
    return 1.0f / (1.0f + expf(-x));
}

__global__ void Gated_GELU(const float *x, float *y, float W, float W_g, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0f;
    if (idx < size){
        float val = x[idx];
        float gelu_out = GELU_approximation(val*W);
        float sigmoid_out = Sigmoid(val*W_g);
        y[idx] = GELU_approximation(x[idx]);
    }
}

torch::Tensor gated_gelu_cuda(torch::Tensor x, float W, float W_g){
    int size = x.numel();

    auto y = torch::empty_like(x);

    dim3 blockSize(256);
    dim3 gridSize((size+blockSize.x-1)/blockSize.x);

    Gated_GELU<<<gridSize, blockSize>>>(x.data_ptr<float>(), y.data_ptr<float>(), W, W_g);
    
    cudaDeviceSynchronize();

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gated_gelu_cuda", &gated_geglu_cuda, "Gated GeLU using CUDA");
}