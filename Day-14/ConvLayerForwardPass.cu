#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

using namespace std;

__global__ void ConvLayerForward_Kernel(int C, int W_grid, int K, float *X, float *W, float *Y, int H_out, int W_out) {
    int m = blockIdx.x; // Output feature map index
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y; 
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    int n = blockIdx.z; // Minibatch sample index

    float acc = 0.0;

    if (h < H_out && w < W_out) {
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) { 
                    // Shape of X is (N, C, H, W), index=(((n⋅C+c)⋅H)+h)⋅W+w
                    // Shape of W is (M, C, K , K), 
                    acc += X[((n * C + c) * (H_out + K - 1) + h + p) * (W_out + K - 1) + (w + q)] *
                           W[(((m * C + c) * K + p) * K) + q];
                }
            }
        }
        // Shape of Y is (M, C, H_out, W_out)
        Y[((n * gridDim.x + m) * H_out + h) * W_out + w] = acc;
    }
}


void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(){
    int N = 1; 
    int M = 1;  
    int C = 1;  
    int H = 5;  
    int W_in = 5; 
    int K = 3;

    int H_out = H - K + 1;
    int W_out = W_in - K + 1;

    // Grid and Block Dimensions
    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;  
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int T = W_grid * H_grid; // Need to linearize this since blockIdx can only be 3d max and the other 2 dimensions are taken by minibatch and output map

    size_t input_size = N * C * H * W_in * sizeof(float);
    size_t weight_size = M * C * K * K * sizeof(float);
    size_t output_size = N * M * H_out * W_out * sizeof(float);

    float* h_X = (float*)malloc(input_size);
    float* h_W = (float*)malloc(weight_size);
    float* h_Y = (float*)malloc(output_size);

    for (int i = 0; i < N * C * H * W_in; i++) {
        h_X[i] = i + 1; 
    }

    
    for (int i = 0; i < M * C * K * K; i++) {
        h_W[i] = 1.0f; 
    }

    float *d_X, *d_W, *d_Y;
    cudaMalloc((void**)&d_X, input_size);
    cudaMalloc((void**)&d_W, weight_size);
    cudaMalloc((void**)&d_Y, output_size);

    cudaMemcpy(d_X, h_X, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, weight_size, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(M, T, N);

    ConvLayerForward_Kernel<<<gridDim, blockDim>>>(C, W_grid,K, d_X, d_W, d_Y, H_out, W_out);
    checkCudaError("Kernel launch failed");

    cudaMemcpy(h_Y, d_Y, output_size, cudaMemcpyDeviceToHost);
    checkCudaError("Data transfer failed");

    std::cout << "Input Feature Map:" << std::endl;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W_in; j++) {
            std::cout << h_X[i * W_in + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nFilter Weights:" << std::endl;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            std::cout << h_W[i * K + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nOutput Feature Map:" << std::endl;
    for (int i = 0; i < H_out; i++) {
        for (int j = 0; j < W_out; j++) {
            std::cout << h_Y[i * W_out + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_Y);

    free(h_X);
    free(h_W);
    free(h_Y);

    return 0;

}