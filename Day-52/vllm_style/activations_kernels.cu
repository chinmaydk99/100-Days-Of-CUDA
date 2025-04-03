#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h> // Crucial for Multi-GPU Setups

#include <cmath>

#include "cuda_compat.h" // Macros to ensure compatibility between CUDA and AMD ROCm
// Macros for abstracting differences in warp intrinsics and optimized globalmemory loads

#include "dispatch_utils.h"// Macros for dispatching kernels based on pytorch tensor data type
// Simplifies template instantiation

namespace vllm{

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&), bool act_first>
// Typename will be replaced with actual data type
// #ACT_FN is a function pointer that takes reference to ascalar_t and returns a scalar_t
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y){
    // Inline tells nvcc to inline the function into the caller to reduce kernel calling overhead
    // Effective in case of small functions
    return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
    // If act_first is true, we first apply the activation function to x and then multiply by y
    // Otherwise, we multiply x by the result of applying the activation function to y
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&), bool act_first>
// Templated for gated logics where one half is passed through activation function and then multiplied with the other half
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const int d // Size of the final dimension
){
    // Each block processess a single tokne
    const int64_t token_idx = blockIdx.x;
    
    // Grid Strid loop
    // - Each thread processes multiple dimensions
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x){
        // Calculate load index for the first x value(first half) and then offset it by idx
        const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);  
        // Calculate load index (second half. We skip over the d elements of first half and then offset it by idx
        const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);

        // Compute the output for the current token and dimension
        out[token_idx * d + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
    }
}

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x){
    // Essentially we are computing x * sigmoid(x)
    return  (T)((float)x / (1.0f + expf((float)-x)));
}

template <typename T>
__device__ __forceinline__ T gelu_kernel(const T &x){
    
    const float f = (float)x; // Cast to float for intermediate calculations
    constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f; // M_SQRT2 = sqrt(2), M_2_SQRTPI = 2/sqrt(pi)
    constexpr float KAPPA = 0.044715;

    float x_cube = f * f * f;
    float inner = BETA * (f + KAPPA * x_cube); // Approximation of erf(x)

    return (T)(0.5f * f * (1.0f + ::tanhf(inner)));
    }
}


// Launching activation and gating kernel
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)
    int d = input.size(-1) / 2; // Dimension for each half of the input
    // Assumes the last dimension is 2*d

    int64_t num_tokens = input.numel() / input.size(-1);// Total number of tokens in the input

    // Launching a grid of num_tokens blocks, each processing a single tokens
    dim3 grid(num_tokens);
    
    // Setting up block size to be the minimum of d and 1024
    dim3 block(std::min(d, 1024));

    // Optional CUDA guard ensures that all subsequent CUDA operations happen on that specific device
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    // Getting the current CUDA stream
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // CUDA kernel needs specialized C++ types whereas the host function receives a pytorch tensor
    // This helps bridge the gap between the two
    VLLM_DISPATCH_FLOATING_TYPES(                                          \
      input.scalar_type(), "act_and_mul_kernel", [&] {                   \
        vllm::act_and_mul_kernel<scalar_t, KERNEL<scalar_t>, ACT_FIRST>  \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),       \
                                         input.data_ptr<scalar_t>(), d); \
      });

void silu_and_mul(torch::Tensor& out,
                  torch::Tensor& input) 
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, true);
}

void mul_and_silu(torch::Tensor& out, 
                  torch::Tensor& input) 
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, false);
}

void gelu_and_mul(torch::Tensor& out,
                  torch::Tensor& input)
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_kernel, true);
}

void gelu_tanh_and_mul(torch::Tensor& out, 
                       torch::Tensor& input)
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_tanh_kernel, true);
}