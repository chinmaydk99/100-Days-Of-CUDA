#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16  

#define CHECK_CUDA_ERROR(call)                                                      
    {                                                                               
        cudaError_t err = call;                                                     
        if (err != cudaSuccess) {                                                   
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;    
            exit(EXIT_FAILURE);                                                     
        }                                                                           
    }

///////////////////////////
// Unroll Kernel 
///////////////////////////

__global__ void unrollKernel(int C, int H, int W, int K, const float* X, float* X_unroll) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;

    if (t < C * W_unroll) {
        int c = t / W_unroll;
        int w_unroll = t % W_unroll; // Column Index of unrolled matrix for output position

        int h_out = w_unroll / W_out;
        int w_out = w_unroll % W_out;

        int w_base = c * K * K; // channel offset. So in the matrix, first K*K will be channel 0, next K*K will be channel 1 and so on

        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                int h_unroll = w_base + p * K + q; // Getting the particular input element corresponding to the filter element
                X_unroll[h_unroll * W_unroll + w_unroll] = X[c * H * W + (h_out + p) * W + (w_out + q)];
            }
        }
    }
}

///////////////////////////////////////
// Tiled GEMM Kernel (with Shared Memory)
///////////////////////////////////////
__global__ void tiledGEMMKernel(int M, int N, int K, const float* A, const float* B, float* C) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    for (int tile_idx = 0; tile_idx < (K + TILE_SIZE - 1) / TILE_SIZE; tile_idx++) {
        // Load tile_A (Filter matrix) from global memory
        if (row < M && tile_idx * TILE_SIZE + threadIdx.x < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + tile_idx * TILE_SIZE + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile_B (Unrolled input matrix) from global memory
        if (col < N && tile_idx * TILE_SIZE + threadIdx.y < K) {
            tile_B[threadIdx.y][threadIdx.x] = B[(tile_idx * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Perform partial matrix multiplication
        for (int k = 0; k < TILE_SIZE; k++) {
            value += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Store the result back to global memory
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

////////////////////////////
// Matrix Printing Function
////////////////////////////
void printMatrix(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

//////////////////////////
// Main Function (Host Code)
//////////////////////////
int main() {
    // Dimensions
    int C = 1;  // Number of input channels
    int M = 1;  // Number of output channels (filters)
    int H = 3;  // Height of input feature map
    int W = 3;  // Width of input feature map
    int K = 2;  // Kernel size (2x2)

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    int unrolled_rows = C * K * K;
    int unrolled_cols = H_out * W_out;

    // Allocate and initialize input feature map
    float* h_X = new float[C * H * W] {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    // Allocate and initialize filters (Flattened)
    float* h_W = new float[M * C * K * K] {
        1, 0, 0, -1  // A simple edge-detection kernel
    };

    // Allocate output (flattened)
    float* h_Y = new float[M * unrolled_cols];

    // Device memory allocation
    float* d_X;
    float* d_X_unroll;
    float* d_W;
    float* d_Y;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_X, C * H * W * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_X_unroll, unrolled_rows * unrolled_cols * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_W, M * unrolled_rows * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Y, M * unrolled_cols * sizeof(float)));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_X, h_X, C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_W, h_W, M * unrolled_rows * sizeof(float), cudaMemcpyHostToDevice));

    // Launch unrolling kernel
    int total_threads = C * H_out * W_out;
    int block_size = 16;
    int grid_size = (total_threads + block_size - 1) / block_size;

    unrollKernel<<<grid_size, block_size>>>(C, H, W, K, d_X, d_X_unroll);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Launch tiled GEMM kernel
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((unrolled_cols + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    tiledGEMMKernel<<<gridDim, blockDim>>>(M, unrolled_cols, unrolled_rows, d_W, d_X_unroll, d_Y);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy result back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(h_Y, d_Y, M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost));

    // Print matrices
    std::cout << "Unrolled Matrix (X_unroll):" << std::endl;
    float* h_X_unroll = new float[unrolled_rows * unrolled_cols];
    CHECK_CUDA_ERROR(cudaMemcpy(h_X_unroll, d_X_unroll, unrolled_rows * unrolled_cols * sizeof(float), cudaMemcpyDeviceToHost));
    printMatrix(h_X_unroll, unrolled_rows, unrolled_cols);

    std::cout << "\nFilter Matrix (W):" << std::endl;
    printMatrix(h_W, M, unrolled_rows);

    std::cout << "\nOutput Feature Map (Y):" << std::endl;
    printMatrix(h_Y, M, unrolled_cols);

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_X_unroll);
    cudaFree(d_W);
    cudaFree(d_Y);
    delete[] h_X;
    delete[] h_W;
    delete[] h_X_unroll;
    delete[] h_Y;

    return 0;
}
