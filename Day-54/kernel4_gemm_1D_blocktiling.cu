// More work per thread. Each thread handles multiple outputs

#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cstring>
#include <cmath> 

#define BLOCK_SIZE 32

#define CUDA_CHECK(call) do{   \
    cudaError_t e = call;      \
    if(e != cudaSuccess){       \
        std::cerr << "CUDA error " << e << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Fixing row within each warp while the column changes. This ensures contiguous data access in row major layout
<template const int BM, const int BN, const int BK, const int TM>
__global__ void gemm_1d_blocktiling(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta){
    const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
  const uint innerRowB = threadIdx.x / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM] = {0.0};

  // outer loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // advance blocktile
    A += BK;
    B += BK * N;

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}



void initialise_matrix(float * mat, int rows, int cols){
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister engine for generating pseudo random numbers
    std::uniform_real_distribution<> distrib(0.0f, 1.0f);

    for(int i = 0; i < rows * cols; ++i){
        mat[i] = static_cast<float>(distrib(gen));
    }
}

// Use const for input arrays
void gemm_cpu(const float *A, const float *B, float * C, int M, int N, int K, float alpha, float beta){
    for(int i = 0; i < M; i ++){
        for(int j = 0; j < N; j ++){
            float sum =  0.0f;
            for(int l = 0; l < K; l ++){
                sum += A[i * K + l] * B[l * N + j];
            }
            float initial_C = (beta == 0.0f) ? 0.0f : C[i * N + j];
            C[i * N + j] = alpha * sum + beta * initial_C;
        }
    }
}


int main(){
    int M = 1024;
    int N = 1024;
    int K = 512;
    float alpha = 1.0f;
    float beta = 1.0f;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = new float[M*K];
    float *h_B = new float[K*N];
    float *h_C = new float[M*N];
    float *h_C_cpu = new float[M*N]; // Allocate memory for CPU result

    if(!h_A || !h_B || !h_C || !h_C_cpu){ // Updated check
        std::cerr << "Memory allocation failed" <<std::endl;
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        delete[] h_C_cpu; // Added cleanup
        return -1;
    }
    std::cout<<"Host side memory allocated" << std::endl;

    initialise_matrix(h_A, M, K);
    initialise_matrix(h_B, K, N);
    initialise_matrix(h_C, M, N); // Initialize h_C since beta=1.0

    std::cout << "Host matrice initialised" << std::endl;

    memcpy(h_C_cpu, h_C, size_C); // Copy initial h_C to h_C_cpu for CPU calculation


    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    std::cout << "Device memory allocated" << std::endl;

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice)); // Copy initial h_C to device


    std::cout << "Input data copied from host to device" << std::endl;

    // Perform CPU calculation
    std::cout << "Performing CPU GEMM calculation..." << std::endl;
    gemm_cpu(h_A, h_B, h_C_cpu, M, N, K, alpha, beta);
    std::cout << "CPU GEMM calculation complete." << std::endl;


    // Cuda Event creation for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Defining the kernel dimensions

    dim3 blockSize(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridSize ((M+ BLOCK_SIZE - 1)/BLOCK_SIZE, (N + BLOCK_SIZE - 1)/BLOCK_SIZE);

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Kernel Launch
    gemm_sharedmemory<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Synchronize to ensure events are recorded
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Kernel execution time: " << milliseconds << "ms" << std::endl;

    // Destroying the events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    std::cout << "GPU Result copied back to host." << std::endl;

    // --- Verification ---
    std::cout << "Verifying GPU result against CPU result..." << std::endl;
    bool match = true;
    float tolerance = 1e-4; // Tolerance for floating-point comparison
    for (int i = 0; i < M * N; ++i) {
        if (fabs(h_C[i] - h_C_cpu[i]) > tolerance) {
            std::cerr << "Mismatch found at index " << i << ": GPU=" << h_C[i] << ", CPU=" << h_C_cpu[i] << std::endl;
            match = false;
            break; // Exit loop on first mismatch
        }
    }

    if (match) {
        std::cout << "Verification successful: GPU and CPU results match within tolerance." << std::endl;
    } else {
        std::cout << "Verification failed: GPU and CPU results do NOT match." << std::endl;
    }


    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_cpu;

    std::cout<<"Host memory freed"<< std::endl;

    return 0;
}