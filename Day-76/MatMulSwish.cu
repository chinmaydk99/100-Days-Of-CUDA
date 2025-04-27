#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__global__ void matmul_bias_swish_scale_kernel(const float* input,
                                              const float* weight,
                                              const float* bias,
                                              float* output,
                                              float scaling_factor,
                                              int batch_size,
                                              int in_features,
                                              int out_features) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < out_features) {
        float z = 0.0f;

        // Compute input * weight^T
        for (int k = 0; k < in_features; ++k) {
            // Access weight matrix in a transposed manner
            z += input[row * in_features + k] * weight[col * in_features + k];
        }

        // Add bias
        z += bias[col];

        // Compute Swish activation: z * sigmoid(z)
        float sigmoid_z = 1.0f / (1.0f + expf(-z));
        float swish_z = z * sigmoid_z;

        // Scale and store the result
        output[row * out_features + col] = scaling_factor * swish_z;
    }
} 