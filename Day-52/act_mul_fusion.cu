#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <algorithm>
#include <string.h>

// Potentially faster global memory reads by bypassing L1 cache
#define LDG(ptr) __ldg(ptr)

// Compute function
// - Given an activation function and a boolean flag, compute the output of the activation function
// - If the flag is true, the activation function is applied first and then multiplied by the second half of the input
// - If the flag is false, the input is multiplied by the result of the activation function and then the activation function is applied

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&), bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t &x, scalar_t &y){
    return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}

// Activation Device Functions

// 1. SilU Activation function
template <typename T>
__device__ __forceinline__ T silu_kernel(const T &x){
    // x * sigmoid(x)
    return (T)((float)x / (1.0f + expf((float)-x)));
}

// 2. GELU Activation function
// - Approximation of erf(x), computationally expensive
template <typename T>
__device__ __forceinline__ T gelu_kernel(const T &x){
    const float f = (float)x;

    constexpr float ALPHA = M_SQRT1_2; // 1 / sqrt(2)

    return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
}

// 3. GELU tanh approximation
// - Approximation of tanh(GELU(x))
template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T &x){
    const float f = (float)x;

    constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f; // M_SQRT2 = sqrt(2), M_2_SQRTPI = 2/sqrt(pi)
    constexpr float KAPPA = 0.044715;

    float x_cube = f * f * f;
    float inner = BETA * (f + KAPPA * x_cube); // Approximation of erf(x)

    return (T)(0.5f * f * (1.0f + ::tanhf(inner)));
}

// Kernel to avoid code duplication for different activations

template <float (*ACT_FN)(const float &), bool act_first>
__global__ void act_and_mul_kernel(
    float *out,
    const float *input,
    const int d, // Size of feature dimension
    const int num_tokens // number of input elements / d
){
    const int token_idx = blockIdx.x;

    // Grid Stride loop
    // - Each thread processes multiple dimensions
    for(int idx = threadIdx.x; idx < d ; idx += blockDim.x){
        // Calculating flattened index for the input tensor [ num_tokens , 2, d]
        // 2 because we have two halves of the input tensor , one that passes through the activation function and one that doesn't
        int inp_base_idx = token_idx * 2 * d;

        // Index for the first half of the input tensor
        int inp_x_idx = inp_base_idx + idx;

        // Index for the second half of the input tensor
        // Skipping the first d elements of the input tensor
        int inp_y_idx = inp_base_idx + d + idx;

        // Calculating flattened index for the output tensor [ num_tokens , d]
        int output_idx = token_idx * d + idx;

        const float x = LDG(&input[inp_x_idx]);
        const float y = LDG(&input[inp_y_idx]);

        // Calling the compute function
        out[output_idx] = compute<float, ACT_FN, act_first>(x, y);
    }
}

// CPU Verification function

void activation_and_mul_cpu(
    float* out_cpu,         
    const float* input,   
    int d,           
    int num_tokens,    
    const char* activation_type, 
    bool act_first
) {
    // Loop over each token
    for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
        // Loop over each feature/element in the dimension 'd'
        for (int idx = 0; idx < d; ++idx) {
            // Calculate flattened indices for the host input array
            int input_base_idx = token_idx * 2 * d;
            float x = input[input_base_idx + idx];
            float y = input[input_base_idx + d + idx];
            float activated_val = 0.0f; // Stores result of activation
            float result = 0.0f;        // Stores final result (activation * other)

            // Determine which value (x or y) to apply the activation to
            float val_to_activate = act_first ? x : y;

            // Apply the chosen activation function
            if (strcmp(activation_type, "silu") == 0) {
                activated_val = val_to_activate / (1.0f + expf(-val_to_activate));
            } else if (strcmp(activation_type, "gelu") == 0) {
                constexpr float ALPHA = M_SQRT1_2;
                activated_val = val_to_activate * 0.5f * (1.0f + erf(val_to_activate * ALPHA));
            } else if (strcmp(activation_type, "gelu_tanh") == 0) {
                constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
                constexpr float KAPPA = 0.044715;
                float v_cube = val_to_activate * val_to_activate * val_to_activate;
                float inner = BETA * (val_to_activate + KAPPA * v_cube);
                activated_val = 0.5f * val_to_activate * (1.0f + tanhf(inner));
            }
            // Perform the final multiplication based on act_first
            result = act_first ? (activated_val * y) : (x * activated_val);
            // Store the result in the host output array
            out_cpu[token_idx * d + idx] = result;
        }
    }
}

// Macro for checking CUDA API Call return values
#define CUDA_CHECK(call) do {
    cudaError_t err = call;
    if (err != cudaSuccess){
        printf("CUDA error %s:%d: '%s' failed with error \"%s\"\n", __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}while (0)


int main(){
    int num_tokens = 1024;
    int d = 4096;
    bool act_first = true;

    printf("--- Activation Kernel Test ---\n");
    printf("Parameters: num_tokens = %d, d = %d, Act_First = %s\n\n",
           num_tokens, d, act_first ? "true" : "false");

    // Host side data allocation
    size_t input_size_bytes = (size_t) 2 * num_tokens * d * sizeof(float);
    size_t output_size_bytes = (size_t) num_tokens * d * sizeof(float);

    printf("Allocating host memory (Input: %.2f MB, Output: %.2f MB)...\n",
           input_size_bytes / (1024.0 * 1024.0), output_size_bytes / (1024.0 * 1024.0));

    float *h_input = (float*)malloc(input_size_bytes);
    float *h_output_gpu = (float*)malloc(output_size_bytes);
    float *h_output_cpu = (float*)malloc(output_size_bytes);

    if (!h_input || !h_output_gpu || !h_output_cpu) {
        fprintf(stderr, "Error: Failed to allocate host memory\n");
        // Cleanup allocated memory before exiting
        free(h_input); // ok if null
        free(h_output_gpu); // ok if null
        free(h_output_cpu); // ok if null
        return EXIT_FAILURE;
    }
    printf("Host memory allocated.\n");

    // Initialize input data with random values
    printf("Initializing host input data...\n");
    srand(time(NULL));
    for (size_t i = 0; i < (size_t)num_tokens * 2 * d; ++i) {
        h_input[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f; // Random values [-1, 1]
    }
    printf("Host input initialized.\n");

    // Device side data allocation
    printf("Allocating device memory...\n");
    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_size_bytes));
    printf("Device memory allocated.\n");

    // Copy input data from host to device
    printf("Copying input data Host -> Device...\n");
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice));
    printf("Input data copied.\n");

    // Kernel Launch Parameters
    dim3 grid(num_tokens);
    dim3 block(std::min(d, 1024));

    // Test Loop for each activation
    const char* activations_to_test[] = {"silu", "gelu", "gelu_tanh"};
    int num_activations = sizeof(activations_to_test) / sizeof(activations_to_test[0]);

    for(int i = 0;i < num_activations; ++i){
        const char* current_activation = activations_to_test[i];

        printf("Launching the kernel for %s...\n", current_activation);
        cudaError_t err = cudaSuccess;

        if (strcmp(current_activation, "silu") == 0) {
            if (act_first)
                act_and_mul_kernel<silu_kernel<float>, true><<<grid, block>>>(d_output, d_input, d, num_tokens);
            else
                act_and_mul_kernel<silu_kernel<float>, false><<<grid, block>>>(d_output, d_input, d, num_tokens);
        } else if (strcmp(current_activation, "gelu") == 0) {
            if (act_first)
                act_and_mul_kernel<gelu_kernel<float>, true><<<grid, block>>>(d_output, d_input, d, num_tokens);
            else
                act_and_mul_kernel<gelu_kernel<float>, false><<<grid, block>>>(d_output, d_input, d, num_tokens);
        } else if (strcmp(current_activation, "gelu_tanh") == 0) {
             if (act_first)
                act_and_mul_kernel<gelu_tanh_kernel<float>, true><<<grid, block>>>(d_output, d_input, d, num_tokens);
             else
                act_and_mul_kernel<gelu_tanh_kernel<float>, false><<<grid, block>>>(d_output, d_input, d, num_tokens);
        } else {
             fprintf(stderr, "Error: Unknown activation '%s' in test loop!\n", current_activation);
             // Perform cleanup before returning error
             cudaFree(d_input); cudaFree(d_output);
             free(h_input); free(h_output_gpu); free(h_output_cpu);
             return EXIT_FAILURE;
        }
        launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
             fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(launch_err));
             // Cleanup before returning error
             cudaFree(d_input); cudaFree(d_output);
             free(h_input); free(h_output_gpu); free(h_output_cpu);
             return EXIT_FAILURE;
        }

        // Wait for the kernel to finish execution before proceeding
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("GPU kernel finished.\n");

        // --- Copy Data Device -> Host ---
        printf("Copying output data Device -> Host...\n");
        CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, output_size_bytes, cudaMemcpyDeviceToHost));
        printf("Output data copied.\n");

        // --- Verification ---
        printf("Running CPU verification...\n");
        activation_and_mul_cpu(h_output_cpu, h_input, d, num_tokens, current_activation, act_first);
        printf("Comparing GPU and CPU results...\n");

        bool match = true;
        float max_diff = 0.0f;
        size_t mismatches = 0;
        size_t total_elements = (size_t)num_tokens * d;

        for (size_t j = 0; j < total_elements; ++j) {
            float diff = fabsf(h_output_gpu[j] - h_output_cpu[j]);
            max_diff = fmaxf(max_diff, diff);
            if (diff > 1e-4) { // Tolerance for floating point differences
                match = false;
                mismatches++;
            }
        }

        if (match) {
            printf("Result: PASS! Max difference: %.6f\n", max_diff);
        } else {
            printf("Result: FAIL! %zu mismatches out of %zu elements. Max difference: %.6f\n",
                   mismatches, total_elements, max_diff);
        }
        printf("--- End Test: %s ---\n\n", current_activation);

     }
    

    // --- Cleanup ---
    printf("Cleaning up memory...\n");
    // Free device memory
    cudaError_t free_err_d_in = cudaFree(d_input);
    cudaError_t free_err_d_out = cudaFree(d_output);
    // Free host memory
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);

    // Check cleanup errors (less critical than compute errors, but good practice)
    if (free_err_d_in != cudaSuccess) fprintf(stderr, "Warning: cudaFree(d_input) failed: %s\n", cudaGetErrorString(free_err_d_in));
    if (free_err_d_out != cudaSuccess) fprintf(stderr, "Warning: cudaFree(d_output) failed: %s\n", cudaGetErrorString(free_err_d_out));

    printf("Cleanup complete.\n");
    printf("--- Test Finished ---\n");
    return EXIT_SUCCESS; 

}