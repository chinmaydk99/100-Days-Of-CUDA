#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA_ERROR(call)                                                      \
    {                                                                               \
        cudaError_t err = call;                                                     \
        if (err != cudaSuccess) {                                                   \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;    \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    }

__global__ void unrollKernel(int C, int H, int W, int K, const float* X, float* X_unroll) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    // Dimension of unrolled input is (C, K, K , H_out, W_out) but we have C*K*K threads
    int W_unroll = H_out * W_out;

    // Each thread is assigned to one column
    if (t < C * W_unroll) {
        int c = t / W_unroll;
        int w_unroll = t % W_unroll;

        int h_out = w_unroll / W_out; // Row Index
        int w_out = w_unroll % W_out; // Column Index

        int w_base = c * K * K; // Row offset in unrolled matrix corresponding to channel c
        // First K*K will be for channel 0, next K*K will be for channel 1 and so on

        // Unroll the patch into the corresponding row
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                int h_unroll = w_base + p * K + q; // Get input element corresponding to particular filter element
                X_unroll[h_unroll * W_unroll + w_unroll] = X[c * H * W + (h_out + p) * W + (w_out + q)];
            }
        }
    }
}

void printMatrix(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Input feature map dimensions
    int C = 1;
    int H = 3;
    int W = 3; 
    int K = 2;

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int unrolled_rows = C * K * K;
    int unrolled_cols = H_out * W_out;

    float* h_X = new float[C * H * W] {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    float* d_X;
    float* d_X_unroll;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_X, C * H * W * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_X_unroll, unrolled_rows * unrolled_cols * sizeof(float)));

    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_X, h_X, C * H * W * sizeof(float), cudaMemcpyHostToDevice));


    int total_threads = C * H_out * W_out;
    int block_size = 16;
    int grid_size = (total_threads + block_size - 1) / block_size;

    unrollKernel<<<grid_size, block_size>>>(C, H, W, K, d_X, d_X_unroll);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy the result back to the host
    float* h_X_unroll = new float[unrolled_rows * unrolled_cols];
    CHECK_CUDA_ERROR(cudaMemcpy(h_X_unroll, d_X_unroll, unrolled_rows * unrolled_cols * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the unrolled matrix
    std::cout << "Unrolled Matrix (Im2Col):" << std::endl;
    printMatrix(h_X_unroll, unrolled_rows, unrolled_cols);

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_X_unroll);
    delete[] h_X;
    delete[] h_X_unroll;

    return 0;
}
