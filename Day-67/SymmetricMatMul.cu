#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

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

__global__ void symmMatMulKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float Cvalue = 0.0f;
        for (int k = 0; k < N; ++k) {
            Cvalue += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = Cvalue;
    }
}

void matrixMulCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    int N = 256;
    size_t matrixSize = N * N;
    size_t matrixBytes = matrixSize * sizeof(float);

    std::vector<float> h_A(matrixSize);
    std::vector<float> h_B(matrixSize);
    std::vector<float> h_C_gpu(matrixSize);
    std::vector<float> h_C_cpu(matrixSize);

    srand(0);
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            float valA = static_cast<float>(rand()) / RAND_MAX;
            float valB = static_cast<float>(rand()) / RAND_MAX;
            h_A[i * N + j] = valA;
            h_A[j * N + i] = valA;
            h_B[i * N + j] = valB;
            h_B[j * N + i] = valB;
        }
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, matrixBytes));
    CUDA_CHECK(cudaMalloc(&d_B, matrixBytes));
    CUDA_CHECK(cudaMalloc(&d_C, matrixBytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), matrixBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), matrixBytes, cudaMemcpyHostToDevice));

    int TILE_DIM = 16;
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 blocksPerGrid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    std::cout << "Launching kernel..." << std::endl;
    symmMatMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Kernel finished." << std::endl;

    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, matrixBytes, cudaMemcpyDeviceToHost));

    std::cout << "Calculating CPU reference..." << std::endl;
    matrixMulCPU(h_A, h_B, h_C_cpu, N);
    std::cout << "CPU calculation finished." << std::endl;

    std::cout << "Verifying results..." << std::endl;
    bool passed = true;
    float tolerance = 1e-4;
    for (size_t i = 0; i < matrixSize; ++i) {
        if (std::abs(h_C_gpu[i] - h_C_cpu[i]) > tolerance * std::abs(h_C_cpu[i])) {
             if (std::abs(h_C_gpu[i] - h_C_cpu[i]) > tolerance ) { // Check absolute if relative is large or cpu is zero
                std::cerr << "Verification FAILED at index " << i << ": "
                        << "GPU=" << h_C_gpu[i] << ", CPU=" << h_C_cpu[i] << std::endl;
                passed = false;
                break;
             }
        }
    }

    if (passed) {
        std::cout << "Verification PASSED!" << std::endl;
    } else {
        std::cout << "Verification FAILED!" << std::endl;
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}