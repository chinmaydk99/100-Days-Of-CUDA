#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 16  // Size of the tile each block will process

// CUDA Kernel: Convolution Forward Pass
__global__ void ConvLayerForward_Kernel(int C, int W_grid, int K, float* X, float* W, float* Y, int H_out, int W_out) {
    int m = blockIdx.x;  // Output feature map (channel) index
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;  // Output row index
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;  // Output column index
    int n = blockIdx.z;  // Mini-batch sample index

    // Accumulator for the convolution result
    float acc = 0.0;
    
    // Ensure threads don't go out of bounds
    if (h < H_out && w < W_out) {
        // Perform the convolution operation
        for (int c = 0; c < C; c++) {  // Loop over all input channels
            for (int p = 0; p < K; p++) {  // Filter height
                for (int q = 0; q < K; q++) {  // Filter width
                    acc += X[((n * C + c) * (H_out + K - 1) + h + p) * (W_out + K - 1) + (w + q)] *
                           W[(((m * C + c) * K + p) * K) + q];
                }
            }
        }
        // Store the result back into the output
        Y[((n * gridDim.x + m) * H_out + h) * W_out + w] = acc;
    }
}

// Function to check for CUDA errors
void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Main function
int main() {
    // Define CNN dimensions
    int N = 1;  // Batch size
    int M = 2;  // Number of output feature maps (filters)
    int C = 3;  // Number of input channels (e.g., RGB)
    int H = 32;  // Input height
    int W_in = 32;  // Input width
    int K = 3;  // Filter size (3x3)

    // Output feature map dimensions
    int H_out = H - K + 1;
    int W_out = W_in - K + 1;

    // Grid and block dimensions for tiling
    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;  // Horizontal tiles
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;  // Vertical tiles
    int T = W_grid * H_grid;  // Total number of tiles

    // Memory sizes
    size_t input_size = N * C * H * W_in * sizeof(float);
    size_t weight_size = M * C * K * K * sizeof(float);
    size_t output_size = N * M * H_out * W_out * sizeof(float);

    // Allocate host (CPU) memory
    float* h_X = (float*)malloc(input_size);
    float* h_W = (float*)malloc(weight_size);
    float* h_Y = (float*)malloc(output_size);

    // Initialize input and weights with random values
    for (int i = 0; i < N * C * H * W_in; i++) {
        h_X[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < M * C * K * K; i++) {
        h_W[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device (GPU) memory
    float *d_X, *d_W, *d_Y;
    cudaMalloc((void**)&d_X, input_size);
    cudaMalloc((void**)&d_W, weight_size);
    cudaMalloc((void**)&d_Y, output_size);

    // Copy data from host to device
    cudaMemcpy(d_X, h_X, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, weight_size, cudaMemcpyHostToDevice);

    // Define CUDA grid and block dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);  // 16x16 threads per block
    dim3 gridDim(M, T, N);  // (output feature maps, tiles, batch size)

    // Launch the kernel
    ConvLayerForward_Kernel<<<gridDim, blockDim>>>(C, W_grid, K, d_X, d_W, d_Y, H_out, W_out);
    checkCudaError("Kernel launch failed");

    // Copy result back from device to host
    cudaMemcpy(h_Y, d_Y, output_size, cudaMemcpyDeviceToHost);
    checkCudaError("Data transfer failed");

    // Display part of the output for verification
    std::cout << "Output feature map (first 10 elements):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << h_Y[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_Y);

    // Free host memory
    free(h_X);
    free(h_W);
    free(h_Y);

    return 0;
}
