#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

#define INF std::numeric_limits<float>::infinity()
#define TILE_DIM 16

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

__global__ void floydWarshallKernel(float* dist, int N, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        float dist_ik = dist[i * N + k];
        float dist_kj = dist[k * N + j];

        if (dist_ik != INF && dist_kj != INF) {
             float current_dist_ij = dist[i * N + j];
             float new_dist_through_k = dist_ik + dist_kj;

             atomicMin(&dist[i * N + j], new_dist_through_k);
        }
    }
}

void floydWarshallGPU(std::vector<float>& h_dist, int N) {
    size_t matrixSizeBytes = N * N * sizeof(float);
    float* d_dist;

    CUDA_CHECK(cudaMalloc(&d_dist, matrixSizeBytes));
    CUDA_CHECK(cudaMemcpy(d_dist, h_dist.data(), matrixSizeBytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 blocksPerGrid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    std::cout << "Starting Floyd-Warshall iterations on GPU..." << std::endl;
    for (int k = 0; k < N; ++k) {
        floydWarshallKernel<<<blocksPerGrid, threadsPerBlock>>>(d_dist, N, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
         if ((k + 1) % 100 == 0 || k == N - 1) {
             std::cout << "  Completed iteration k = " << k + 1 << "/" << N << std::endl;
         }
    }
    std::cout << "GPU iterations complete." << std::endl;

    CUDA_CHECK(cudaMemcpy(h_dist.data(), d_dist, matrixSizeBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dist));
}

void floydWarshallCPU(std::vector<float>& dist, int N) {
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (dist[i * N + k] != INF && dist[k * N + j] != INF) {
                    dist[i * N + j] = std::min(dist[i * N + j], dist[i * N + k] + dist[k * N + j]);
                }
            }
        }
    }
}

int main() {
    int N = 512;
    size_t matrixSize = N * N;
    std::vector<float> h_dist_gpu(matrixSize);
    std::vector<float> h_dist_cpu(matrixSize);

    srand(0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                h_dist_gpu[i * N + j] = 0.0f;
            } else {
                if (rand() % 5 == 0) {
                    h_dist_gpu[i * N + j] = INF;
                } else {
                    h_dist_gpu[i * N + j] = static_cast<float>(rand() % 100 + 1);
                }
            }
        }
    }
    h_dist_cpu = h_dist_gpu;

    std::cout << "Graph size: " << N << "x" << N << std::endl;

    floydWarshallGPU(h_dist_gpu, N);

    std::cout << "Calculating CPU reference (this might take a while)..." << std::endl;
    floydWarshallCPU(h_dist_cpu, N);
    std::cout << "CPU calculation finished." << std::endl;

    std::cout << "Verifying results..." << std::endl;
    bool passed = true;
    float tolerance = 1e-4;
    int fail_count = 0;
    int max_fail_prints = 10;

    for (size_t i = 0; i < matrixSize; ++i) {
         bool cpu_inf = (h_dist_cpu[i] == INF);
         bool gpu_inf = (h_dist_gpu[i] == INF);

         if (cpu_inf != gpu_inf || (!cpu_inf && std::abs(h_dist_gpu[i] - h_dist_cpu[i]) > tolerance)) {
             if (fail_count < max_fail_prints) {
                 int row = i / N;
                 int col = i % N;
                 std::cerr << "Verification FAILED at index [" << row << "][" << col << "] (" << i << "): "
                           << "GPU=" << h_dist_gpu[i] << ", CPU=" << h_dist_cpu[i] << std::endl;
             }
             passed = false;
             fail_count++;
         }
    }

     if (fail_count > max_fail_prints) {
          std::cerr << "... (" << (fail_count - max_fail_prints) << " more verification failures not shown)" << std::endl;
     }


    if (passed) {
        std::cout << "Verification PASSED!" << std::endl;
    } else {
        std::cout << "Verification FAILED! (" << fail_count << " mismatches)" << std::endl;
    }

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}