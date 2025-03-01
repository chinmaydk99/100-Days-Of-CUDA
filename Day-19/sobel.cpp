#include <torch/extension.h>

// Function declaration for CUDA kernel
torch::Tensor sobel_cuda_forward(torch::Tensor input);

// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // binding sobel cuda forward to the python forward function
    m.def("forward", &sobel_cuda_forward, "Sobel Filter forward pass (CUDA)");
}