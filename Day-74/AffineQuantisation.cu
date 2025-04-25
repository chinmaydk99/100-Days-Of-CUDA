#include <cuda_runtime.h>
#include <cuda_fp16.h> // Or other types if needed
#include <iostream>
#include <vector>
#include <cmath> // For roundf, fminf, fmaxf
#include <limits> // For numeric_limits
#include <numeric> // For iota, accumulate

#define CUDA_CHECK(call)                                                          \
{                                                                                \
    cudaError_t err = call;                                                      \
    if (err != cudaSuccess)                                                      \
    {                                                                            \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,       \
                cudaGetErrorString(err));                                        \
        exit(EXIT_FAILURE);                                                      \
    }                                                                            \
}

__global__ void quantize_affine_per_tensor_kernel(
    const float* input,
    signed char* output, // Using signed char for int8
    float scale,
    int zero_point,
    int n_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_elements) {
        float val_float = input[idx];
        float scaled_val = val_float / scale;
        float rounded_val = roundf(scaled_val); // Round to nearest integer
        float shifted_val = rounded_val + zero_point;

        // Clamp to int8 range [-128, 127]
        float clamped_val = fmaxf(-128.0f, fminf(shifted_val, 127.0f));

        output[idx] = static_cast<signed char>(clamped_val);
    }
}

// Simple host-side function to calculate scale and zero-point (symmetric quantization)
void get_scale_zp_symmetric(const std::vector<float>& data, float& scale, int& zero_point) {
    float max_abs_val = 0.0f;
    for (float val : data) {
        max_abs_val = fmaxf(max_abs_val, fabsf(val));
    }

    if (max_abs_val == 0.0f) {
        scale = 1.0f; // Avoid division by zero
    } else {
        scale = max_abs_val / 127.0f; // Map max abs value to 127
    }
    zero_point = 0; // Symmetric quantization has zero_point = 0
}

// Host-side function to dequantize for verification
void dequantize_cpu(const std::vector<signed char>& q_data, std::vector<float>& dq_data, float scale, int zero_point) {
    dq_data.resize(q_data.size());
    for (size_t i = 0; i < q_data.size(); ++i) {
        dq_data[i] = static_cast<float>(q_data[i] - zero_point) * scale;
    }
}

int main() {
    int n_elements = 1 << 20; // ~1 million elements
    std::vector<float> h_input(n_elements);
    std::vector<signed char> h_output_gpu(n_elements);
    std::vector<float> h_dequant_gpu(n_elements);

    // Generate random data (e.g., between -10 and 10)
    srand(0);
    for (int i = 0; i < n_elements; ++i) {
        h_input[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 20.0f;
    }

    float scale;
    int zero_point;
    get_scale_zp_symmetric(h_input, scale, zero_point);
    std::cout << "Input size: " << n_elements << " elements." << std::endl;
    std::cout << "Calculated Scale: " << scale << ", Zero Point: " << zero_point << std::endl;


    float* d_input;
    signed char* d_output;
    size_t input_bytes = n_elements * sizeof(float);
    size_t output_bytes = n_elements * sizeof(signed char);

    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));


    const int threads_per_block = 256;
    dim3 grid_dim((n_elements + threads_per_block - 1) / threads_per_block);
    dim3 block_dim(threads_per_block);

    std::cout << "Launching CUDA quantization kernel..." << std::endl;
    quantize_affine_per_tensor_kernel<<<grid_dim, block_dim>>>(
        d_input, d_output, scale, zero_point, n_elements
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Kernel finished." << std::endl;

    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));


    std::cout << "Dequantizing GPU output on CPU for verification..." << std::endl;
    dequantize_cpu(h_output_gpu, h_dequant_gpu, scale, zero_point);


    std::cout << "Verifying results (comparing dequantized vs original)..." << std::endl;
    double total_error = 0.0;
    double max_error = 0.0;
    for (int i = 0; i < n_elements; ++i) {
        double error = std::abs(h_input[i] - h_dequant_gpu[i]);
        total_error += error;
        max_error = std::max(max_error, error);
    }
    double mean_absolute_error = total_error / n_elements;

    // Quantization introduces error, so we check if the error is reasonable
    // A common metric is checking if the MAE is less than the scale.
    std::cout << "Mean Absolute Error (MAE): " << mean_absolute_error << std::endl;
    std::cout << "Maximum Absolute Error: " << max_error << std::endl;
    std::cout << "Scale: " << scale << std::endl;

    if (mean_absolute_error <= scale) { // Heuristic check
        std::cout << "Verification PASSED (MAE is within acceptable range for quantization)." << std::endl;
    } else {
        std::cout << "Verification potentially FAILED (MAE seems high compared to scale)." << std::endl;
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}