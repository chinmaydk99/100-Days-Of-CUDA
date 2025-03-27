#include <iostream>
#include <sstream>
#include <vector>

#include "helper.h" // In the commons folder

#include "cutlass/gemm/device/gemm.h" // Generic Gemm computation template class

// Performing alpha * AB + beta * C

cudaError_t CutlassgemmBasic(
  int M, int N, int K,
  float alpha, 
  float const *A, int lda, // A and stride_A
  float const *B, int ldb, // B and stride_A
  float beta, 
  float *C, int ldc){
    // Column major input and 128 * 128 * 8 threadblock size
    using ColumnMajor = cutlass::layout::ColumnMajor;

    using CutlassGemm = cutlass::gemm::device::Gemm<float, // Type A
                                                    ColumnMajor, // Layout A, same for B and C
                                                    float,
                                                    ColumnMajor,
                                                    float,
                                                    ColumnMajor>;

    CutlassGemm gemm_op; // Instantiating the GEMM object

    // Wrapping up everything the GEMM Kernel needs into a neat arguments object that can be passed around cleanly
    CutlassGemm::Arguments args({M, N, K},
                                {A, lda},
                                {B, ldb},
                                {C, ldc},
                                {C, ldc}, // For destination matrix D 
                                {alpha, beta});
    
    // Launching the CUTLASS GEMM Kernel
    cutlass::Status status = gemm_op(args);

    if(status != cutlass::Status::kSuccess){
      return cudaErrorUnknown;
    }

    return cudaSuccess;

  }


  // Kernel to randomly initialise input matrices

  __global__ void InitializeMatrix_kernel(
    float *matrix,
    int rows,
    int columns,
    int seed = 0
  ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < rows and j < columns){
      int offset = i + j * rows; // Column Major

      int const k = 23523;
      int const m = 32;

      float value = float(((offset + seed) * k % m) - m / 2);

      matrix[offset] = value;
    }
  }

 cudaError_t InitializeMatrix(float *matrix, int rows, int columns, int seed = 0) {

    dim3 block(16, 16);
    dim3 grid(
      (rows + block.x - 1) / block.x,
      (columns + block.y - 1) / block.y
    );

    InitializeMatrix_kernel<<< grid, block >>>(matrix, rows, columns, seed);

    return cudaGetLastError();
 }

 // Allocating device memory for matrix and filling it
cudaError_t AllocateMatrix(float **matrix, int rows, int columns, int seed = 0) {
    cudaError_t result;

    // Compute the memory needed
    size_t size_of_matrix = sizeof(float) * rows * columns;

    // Allocate memory on the GPU
    result = cudaMalloc(reinterpret_cast<void **>(matrix), size_of_matrix);

    if (result != cudaSuccess) {
      std::cerr << "Failed to allocate matrix: "
        << cudaGetErrorString(result) << std::endl;
      return result;
    }

    // Setting all elements to 0
    result = cudaMemset(*matrix, 0 , size_of_matrix);

    if (result != cudaSuccess){
      std::cerr<<"Failed to Initialize matrix: "<<
      cudaGetErrorString(result) << std::endl;
      return result;
    }

    result = InitializeMatrix(*matrix, rows, columns, seed);

    if (result != cudaSuccess){
      std::cerr << "Failed to initialize matrix: " 
      << cudaGetErrorString(result) << std::endl;
      return result;
    }

    return result;
}


// Reference Kernel

__global__ void reference_gemm_kernel(
    int M, int N, int K,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float *C, int ldc){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < M && j < N){
        float acc = 0.0f;

        for (int k = 0; k < K; ++k){
            acc += A[i + k * lda] * B[k + j * ldb];
        }

        C[i + j * ldc] = alpha * acc + beta * C[i + j * ldc];
    }
}

cudaError_t ReferenceGemm(
    int M, int N, int K,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float *C, int ldc
){
    dim3 block(16,16);
    dim3 grid(
      (M + block.x - 1) / block.x,
      (N + block.y - 1) / block.y
    );

    reference_gemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}

// Testing the code and reference implementation
cudaError_t TestCutlassGemm(int M, int N, int K, float alpha, float beta){
    cudaError_t result;

    // Leading dimensions (for column major)
    int lda = M;
    int ldb = K;
    int ldc = M;

    size_t sizeof_C = sizeof(float) * M * N;

    // Pointers to matrices in GPU device memory
    float *A;
    float *B;
    float *C_cutlass;
    float *C_reference;

    // Allocating matrices in GPU Device memory
    result = AllocateMatrix(&A, M, K, 0);

    if(result != cudaSuccess){
        return result;
    }

    result = AllocateMatrix(&B, K, N, 25);

    if(result != cudaSuccess){
      cudaFree(A);
      return result;
    }

    result = AllocateMatrix(&C_cutlass, M, N, 100);

    if (result != cudaSuccess) {
      cudaFree(A);
      cudaFree(B);
      return result;
    }

    result = AllocateMatrix(&C_reference, M, N, 100);

    if (result != cudaSuccess) {
      cudaFree(A);
      cudaFree(B);
      cudaFree(C_cutlass);
      return result;
    }

    // Copying value of C_cutlass into C_reference so that both GEMMs will operate with same initial values
    result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

    if (result != cudaSuccess) {
      std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
        << cudaGetErrorString(result) << std::endl;

      cudaFree(C_reference);
      cudaFree(C_cutlass);
      cudaFree(B);
      cudaFree(A);

      return result;
    }


    // Launching CUTLASS GEMM

    result = CutlassgemmBasic(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);

    if (result != cudaSuccess) {
      std::cerr << "CUTLASS GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;

      cudaFree(C_reference);
      cudaFree(C_cutlass);
      cudaFree(B);
      cudaFree(A);

      return result;
    }
  
    // Launching the reference GEMM
    result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

    if (result != cudaSuccess) {
      std::cerr << "Reference GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;

      cudaFree(C_reference);
      cudaFree(C_cutlass);
      cudaFree(B);
      cudaFree(A);
      return result;
    }

    // Copying to host and checking equality
    std::vector<float> host_cutlass(ldc * N, 0);
    std::vector<float> host_reference(ldc * N, 0);

    result = cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);

    if (result != cudaSuccess) {
    std::cerr << "Failed to copy CUTLASS GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
    }

    result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

    if (result != cudaSuccess) {
      std::cerr << "Failed to copy Reference GEMM results: "
        << cudaGetErrorString(result) << std::endl;

      cudaFree(C_reference);
      cudaFree(C_cutlass);
      cudaFree(B);
      cudaFree(A);

      return result;
    }

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    if (host_cutlass != host_reference) {
      std::cerr << "CUTLASS results incorrect." << std::endl;

      return cudaErrorUnknown;
    }

    return cudaSuccess;
}


int main(int argc, const char *arg[]){
    int problem[3] = {128, 128, 128};

    for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  float scalars[2] = { 1, 0 };

  for (int i = 4; i < argc && i < 6; ++i) {
    std::stringstream ss(arg[i]);
    ss >> scalars[i - 4];
  }

  cudaError_t result = TestCutlassGemm(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1]      // beta
  );

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  return result == cudaSuccess ? 0 : -1;

}


