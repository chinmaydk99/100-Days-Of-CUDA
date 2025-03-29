#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/device/gemm_batched.h"

#pragma warning( disable : 4503)

cudaError_t cutlass_batched_gemm_array(
    int m, int n, int k,
    float alpha, 
    float const * const * A,
    int lda,
    float const * const * B,
    int ldb,
    float * const * C,
    int ldc,
    float beta,
    int batch_count
) {
    using Gemm = cutlass::gemm::device::GemmArray<
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor
    >;

    Gemm gemm_op;
    
    cutlass::Status status = gemm_op({
        {m, n, k},
        A, lda,
        B, ldb,
        C, ldc,
        C, ldc,
        {alpha, beta},
        batch_count
    });

    if(status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

cudaError_t cutlass_strided_batched_sgemm(
    int m, int n, int k,
    float alpha,
    float const * A,  int lda,
    long long int batch_stride_A,
    float const * B, int ldb,
    long long int batch_stride_B,
    float * C, int ldc,
    long long int batch_stride_C,
    float beta,
    int batch_count
){
    using Gemm = cutlass::gemm::device::GemmBatched<
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor
    >;

    Gemm gemm_op;

    cutlass::Status status = gemm_op({
        {m, n, k},
        {A, lda},
        batch_stride_A,
        {B, ldb},
        batch_stride_B,
        {C, ldc},
        batch_stride_C,
        {alpha, beta},
        batch_count
    });

    if(status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}


// Reference kernel for Strided Batched GEMM

template<typename T> 
cudaError_t strided_batched_gemm_reference(
    int m,
    int n,
    int k,
    T alpha,
    std::vector<T> const &A, 
    int lda,
    long long int batch_stride_A,
    std::vector<T> const &B, 
    int ldb,
    long long int batch_stride_B,
    std::vector<T> &C,
    int ldc,
    long long int batch_stride_C,
    T beta,
    int batch_count
){
    cudaError_t result = cudaSuccess;

    // Verifying that matrix A has enough memory for batch_count matrices each with lda * k elemenys
    // Each batch of A is a complete M x K matrix stacked one after another in memory
    // Each batch operates on its own complete input matrix
    if (A.size() < size_t(lda * k * batch_count)){
        std::cout << "the size of A is too small" << std::endl;
        return cudaErrorInvalidValue;
    }
    
    // B uses an interleaved layout, where corresponding rows from different batches are adjacent
    // Here stride between consecutive batches is just k elements and not the complete matrices
    if(B.size() < size_t(ldb * n)) {
        std::cout << "the size of B is too small" << std::endl;
        return cudaErrorInvalidValue;
    }

    // Result matrix C against is a complete M * N matrix stacked sequentially
    if(C.size() < size_t(ldc * n * batch_count)) {
        std::cout << "the size of C is too small" << std::endl;
        return cudaErrorInvalidValue;
    }

    // Reference CPU implementation of the Batched GEMM computation loop
    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
    for (int n_idx = 0; n_idx < n; n_idx++) {
      for (int m_idx = 0; m_idx < m; m_idx++) {
        T accum = beta * C[batch_idx * batch_stride_C + n_idx * ldc + m_idx];
        for (int k_idx = 0; k_idx < k; k_idx++) {
          accum += alpha 
            * A[batch_idx * batch_stride_A + k_idx * lda + m_idx]
            * B[batch_idx * batch_stride_B + n_idx * ldb + k_idx];
        }
        C[batch_idx * batch_stride_C + n_idx * ldc + m_idx] = accum;
      }
    }
  }

  return result;
}


#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/device/gemm_batched.h"

#pragma warning( disable : 4503)

cudaError_t cutlass_batched_gemm_array(
    int m, int n, int k,
    float alpha, 
    float const * const * A,
    int lda,
    float const * const * B,
    int ldb,
    float * const * C,
    int ldc,
    float beta,
    int batch_count
) {
    using Gemm = cutlass::gemm::device::GemmArray<
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor
    >;

    Gemm gemm_op;
    
    cutlass::Status status = gemm_op({
        {m, n, k},
        A, lda,
        B, ldb,
        C, ldc,
        C, ldc,
        {alpha, beta},
        batch_count
    });

    if(status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

cudaError_t cutlass_strided_batched_sgemm(
    int m, int n, int k,
    float alpha,
    float const * A,  int lda,
    long long int batch_stride_A,
    float const * B, int ldb,
    long long int batch_stride_B,
    float * C, int ldc,
    long long int batch_stride_C,
    float beta,
    int batch_count
){
    using Gemm = cutlass::gemm::device::GemmBatched<
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor
    >;

    Gemm gemm_op;

    cutlass::Status status = gemm_op({
        {m, n, k},
        {A, lda},
        batch_stride_A,
        {B, ldb},
        batch_stride_B,
        {C, ldc},
        batch_stride_C,
        {alpha, beta},
        batch_count
    });

    if(status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}


// Reference kernel for Strided Batched GEMM

template<typename T> 
cudaError_t strided_batched_gemm_nn_reference(
    int m,
    int n,
    int k,
    T alpha,
    std::vector<T> const &A, 
    int lda,
    long long int batch_stride_A,
    std::vector<T> const &B, 
    int ldb,
    long long int batch_stride_B,
    std::vector<T> &C,
    int ldc,
    long long int batch_stride_C,
    T beta,
    int batch_count
){
    cudaError_t result = cudaSuccess;

    // Verifying that matrix A has enough memory for batch_count matrices each with lda * k elemenys
    // Each batch of A is a complete M x K matrix stacked one after another in memory
    // Each batch operates on its own complete input matrix
    if (A.size() < size_t(lda * k * batch_count)){
        std::cout << "the size of A is too small" << std::endl;
        return cudaErrorInvalidValue;
    }
    
    // B uses an interleaved layout, where corresponding rows from different batches are adjacent
    // Here stride between consecutive batches is just k elements and not the complete matrices
    if(B.size() < size_t(ldb * n)) {
        std::cout << "the size of B is too small" << std::endl;
        return cudaErrorInvalidValue;
    }

    // Result matrix C against is a complete M * N matrix stacked sequentially
    if(C.size() < size_t(ldc * n * batch_count)) {
        std::cout << "the size of C is too small" << std::endl;
        return cudaErrorInvalidValue;
    }

    // Reference CPU implementation of the Batched GEMM computation loop
    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
    for (int n_idx = 0; n_idx < n; n_idx++) {
      for (int m_idx = 0; m_idx < m; m_idx++) {
        T accum = beta * C[batch_idx * batch_stride_C + n_idx * ldc + m_idx];
        for (int k_idx = 0; k_idx < k; k_idx++) {
          accum += alpha 
            * A[batch_idx * batch_stride_A + k_idx * lda + m_idx]
            * B[batch_idx * batch_stride_B + n_idx * ldb + k_idx];
        }
        C[batch_idx * batch_stride_C + n_idx * ldc + m_idx] = accum;
      }
    }
  }

  return result;
}

cudaError_t run_batched_gemm(bool use_array) {

  const char* gemm_desc = use_array ? "array" : "strided batched";
  std::cout << "Running " << gemm_desc << " gemm" << std::endl;

  // Arbitrary problem size
  int const m = 520;
  int const n = 219;
  int const k = 129;
  int const batch_count = 17;

  // A, B are non-transpose, column major
  int const lda = m;
  int const ldb = k * batch_count;
  int const ldc = m;

  int const count_A = batch_count * lda * k;
  int const count_B = ldb * n;
  int const count_C = batch_count * ldc * n;

  // the memory is batched along K dimension
  long long int batch_stride_A = static_cast<long long int>(lda) * static_cast<long long int>(k);
  long long int batch_stride_B = static_cast<long long int>(k);
  long long int batch_stride_C = static_cast<long long int>(ldc) * static_cast<long long int>(n);

  // alpha and beta
  float alpha = 1.0f;
  float beta = 2.0f;

  cudaError_t result = cudaSuccess;

  // allocate the host memory
  std::vector<float> host_A(count_A);
  std::vector<float> host_B(count_B);
  std::vector<float> host_C(count_C);
  std::vector<float> result_C(count_C);

  // allocate the device memory
  float *A;
  float *B;
  float *C;

  result = cudaMalloc(&A, count_A * sizeof(float));
  if (result != cudaSuccess) {
    std::cerr << "cudaMalloc result = " << result << std::endl;
    return result;
  }
  result = cudaMalloc(&B, count_B * sizeof(float));
  if (result != cudaSuccess) {
    std::cerr << "cudaMalloc result = " << result << std::endl;
    return result;
  }
  result = cudaMalloc(&C, count_C * sizeof(float));
  if (result != cudaSuccess) {
    std::cerr << "cudaMalloc result = " << result << std::endl;
    return result;
  }

  // Limit range to avoid floating-point errors
  int const kRange = 8;

  // fill A
  for (int b_idx = 0; b_idx < batch_count; b_idx++) {
    for (int col_idx = 0; col_idx < k; col_idx++) {
      for (int row_idx = 0; row_idx < m; row_idx++) {
        host_A[row_idx + col_idx * lda + b_idx * lda * k] = static_cast<float>((row_idx + col_idx * lda + b_idx * lda * k) % kRange);
      }
    }
  }
  // fill B
  for (int b_idx = 0; b_idx < batch_count; b_idx++) {
    for (int col_idx = 0; col_idx < n; col_idx++) {
      for (int row_idx = 0; row_idx < k; row_idx++) {
        host_B[row_idx + col_idx * ldb + b_idx * k] = static_cast<float>(((n + k * ldb + batch_count * k) - (row_idx + col_idx * ldb + b_idx * k)) % kRange);
      }
    }
  }
  // fill C
  for (int b_idx = 0; b_idx < batch_count; b_idx++) {
    for (int col_idx = 0; col_idx < n; col_idx++) {
      for (int row_idx = 0; row_idx < m; row_idx++) {
        host_C[row_idx + col_idx * ldc + b_idx * ldc * n] = 1.f;
      }
    }
  }

  // ref memory
  std::vector<float> ref_A(host_A);
  std::vector<float> ref_B(host_B);
  std::vector<float> ref_C(host_C);
  // copy host memory to device
  result = cudaMemcpy(A, host_A.data(), count_A * sizeof(float), cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }
  result = cudaMemcpy(B, host_B.data(), count_B * sizeof(float), cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }
  result = cudaMemcpy(C, host_C.data(), count_C * sizeof(float), cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }

  // run cutlass
  if (use_array) {
    // allocate the host memory for the pointers to the matrices of the batch
    std::vector<float*> host_ptr_A(batch_count);
    std::vector<float*> host_ptr_B(batch_count);
    std::vector<float*> host_ptr_C(batch_count);

    // permute the batch elements to emphasize that GemmArray does not depend on matrices being separated by a fixed stride
    std::vector<size_t> permutation = {14, 11, 3, 10, 1, 13, 9, 4, 6, 16, 8, 15, 7, 12, 0, 2, 5};
    for (size_t b_idx = 0; b_idx < batch_count; b_idx++) {
      host_ptr_A[b_idx] = A + permutation[b_idx] * batch_stride_A;
      host_ptr_B[b_idx] = B + permutation[b_idx] * batch_stride_B;
      host_ptr_C[b_idx] = C + permutation[b_idx] * batch_stride_C;
    }

    // allocate the corresponding device memory
    float const **ptr_A;
    float const **ptr_B;
    float **ptr_C;

    result = cudaMalloc(&ptr_A, batch_count * sizeof(float*));
    if (result != cudaSuccess) {
      std::cerr << "cudaMalloc result = " << result << std::endl;
      return result;
    }
    result = cudaMalloc(&ptr_B, batch_count * sizeof(float*));
    if (result != cudaSuccess) {
      std::cerr << "cudaMalloc result = " << result << std::endl;
      return result;
    }
    result = cudaMalloc(&ptr_C, batch_count * sizeof(float*));
    if (result != cudaSuccess) {
      std::cerr << "cudaMalloc result = " << result << std::endl;
      return result;
    }

    // copy the matrix pointers to the device
    result = cudaMemcpy(ptr_A, host_ptr_A.data(), batch_count * sizeof(float*), cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
      std::cerr << "cudaMemcpy result = " << result << std::endl;
      return result;
    }
    result = cudaMemcpy(ptr_B, host_ptr_B.data(), batch_count * sizeof(float*), cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
      std::cerr << "cudaMemcpy result = " << result << std::endl;
      return result;
    }
    result = cudaMemcpy(ptr_C, host_ptr_C.data(), batch_count * sizeof(float*), cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
      std::cerr << "cudaMemcpy result = " << result << std::endl;
      return result;
    }

    result = cutlass_batched_gemm_array(m, n, k, alpha, ptr_A, lda, ptr_B, ldb, ptr_C, ldc, beta, batch_count);

    if (result != cudaSuccess)
      return result;
  } else {
    result = cutlass_strided_batched_sgemm(
      m, n, k, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C,
      beta, batch_count);
    if (result != cudaSuccess)
      return result;
  }

  // copy device memory to host
  result = cudaMemcpy(result_C.data(), C, count_C * sizeof(float), cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }

  //compare with reference code
  result = strided_batched_gemm_reference(m, n, k, alpha, ref_A, lda, batch_stride_A, ref_B, ldb, batch_stride_B, ref_C, ldc, batch_stride_C,
    beta, batch_count);
  if (result != 0)
    return result;

  // Expect bit-level accuracy for this simple example
  if (ref_C != result_C) {
    std::cout << "CUTLASS " << gemm_desc << " gemm does not run correctly" << std::endl;
    return cudaErrorUnknown;
  }

  // free memory
  result = cudaFree(A);
  if (result != cudaSuccess) {
    std::cerr << "cudaFree result = " << result << std::endl;
    return result;
  }
  result = cudaFree(B);
  if (result != cudaSuccess) {
    std::cerr << "cudaFree result = " << result << std::endl;
    return result;
  }
  result = cudaFree(C);
  if (result != cudaSuccess) {
    std::cerr << "cudaFree result = " << result << std::endl;
    return result;
  }

  return result;
}



    
    
    
    




    
    
    
    
