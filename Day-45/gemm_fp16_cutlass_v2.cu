#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include "cutlass/cutlass.h"
#include "cutlass/core_io.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"

// Improvements compared to basic kernel
// half_t = FP16(half precision)
// HostTensor<type, layout(column/rowmajor)> : removes need for cudaMalloc, cudaMemcpy
// Allocates host and device memory
// TensorFillRandomGaussian : Initialise tensors with values from a Gaussian distribution directly on device
// cutlass::reference::host::Gemm<..> : CPU based implementation for correctness checking

#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#pragma warning( disable : 4503)

cudaError_t cutlass_gemm_fp16(
  int M,
  int N,
  int K,
  cutlass::half_t alpha,
  cutlass::half_t const *A,
  cutlass::layout::ColumnMajor::Stride::Index lda,
  cutlass::half_t const *B,
  cutlass::layout::ColumnMajor::Stride::Index ldb,
  cutlass::half_t beta,
  cutlass::half_t *C,
  cutlass::layout::ColumnMajor::Stride::Index ldc){

    // Define the GEMM operation
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, // Element A
        cutlass::layout::ColumnMajor, // Layout A
        cutlass::half_t, // Element B
        cutlass::layout::ColumnMajor, // Layout B
        cutlass::half_t, // ElementOutput
        cutlass::layout::ColumnMajor // LayoutOutput
    >;

    Gemm gemm_op;

    cutlass::Status status = gemm_op({
      {M, N, K},
      {A, lda},
      {B, ldb},
      {C, ldc},
      {C, ldc},
      {alpha, beta}
    });

    if(status != cutlass::Status::kSuccess){
      //Small note : kSuccess is CUTLASS status message while cudaSuccess is the CUDA one
      // Transitioning from CUTLASS to CUDA
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}


// Testing the CUTLASS GEMM Kernel
// Instead of manually writing the test functions like earlier, I'll be using a host of utility functions

cudaError_t TestCutlassGemm(int M, int N, int K, cutlass::half_t alpha, cutlass::half_t beta){

    cudaError_t result;

    // Defining the matrices using HostTensor
    // cutlass::HostTensor<type, layout(row/ column major)> Matrix(cutlass::MatrixCoord(Dim1, Dim2));

    // Defining M x K cutlass::half_t A
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A(cutlass::MatrixCoord(M, K));

    // Defining K x N cutlass::half_t B
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B(cutlass::MatrixCoord(K, N));

     // Defining M-by-N matrix of cutlass::half_t C_cutlass
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C_cutlass(cutlass::MatrixCoord(M, N));

    // M-by-N matrix of cutlass::half_t C_reference
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C_reference(cutlass::MatrixCoord(M, N));
    
    // Initialising matrix with random integers
    uint64_t seed = 200;

    cutlass::half_t mean = 0.0_hf;
    cutlass::half_t stddev = 5.0_hf;

    int bits_less_than_one = 0; // Number fo bits permitted on right of decimal to be non-zero. Here truncating to integers

    cutlass::reference::device::TensorFillRandomGaussian(
      A.device_view(),
      seed,
      mean,
      stddev,
      bits_less_than_one
    );

    cutlass::reference::device::TensorFillRandomGaussian(
      B.device_view(),
      seed,
      mean,
      stddev,
      bits_less_than_one
    );

    cutlass::reference::device::TensorFillRandomGaussian(
      C_cutlass.device_view(),
      seed,
      mean,
      stddev,
      bits_less_than_one
    );

    // Copying C_cutlass into C_reference
    cutlass::device_memory::copy_device_to_device(
      C_reference.device_data(), // GPU view of tensor
      C_cutlass.device_data(),
      C_cutlass.capacity() // Number of elements to copy
    );

    // Copying the device side view into host memory (because our reference calculation is performed on the cpu)
    C_reference.sync_host();

    // Launching the CUTLASS GEMM Kernel
    result = cutlass_gemm_fp16(
        M, N, K,
        alpha,
        A.device_data(), A.stride(0),
        B.device_data(), B.stride(0),
        beta,
        C_cutlass.device_data(), C_cutlass.stride(0)
    );

    if (result != cudaSuccess){
        return result;
    }


    // Verifying using GEMM perfomed on cpu
    A.sync_host();
    B.sync_host();

    // Copying the CUTLASS GEMM's results into host memory
    C_cutlass.sync_host();

    // Host-side reference GEMM implementation using host::Gemm
    cutlass::reference::host::Gemm<
      cutlass::half_t, // Element A
      cutlass::layout::ColumnMajor, //Layout A
      cutlass::half_t, //Element B
      cutlass::layout::ColumnMajor, //Layout B
      cutlass::half_t, // Element Output
      cutlass::layout::ColumnMajor, // Layout Output
      cutlass::half_t, // Scalar Alpha
      cutlass::half_t // Scalar Beta
    > gemm_ref;

    gemm_ref(
      {M, N, K},
      alpha,
      A.host_ref(),
      B.host_ref(),
      beta,
      C_reference.host_ref()
    );
  

  // Comparing CUTLASS result to Host side GEMM result
  if(!cutlass::reference::host::TensorEquals(
      C_reference.host_view(), // Checking if the two tensors are bitwise identical, another utility function
      C_cutlass.host_view()
  )){
      // In case the values differm the output is written to  a log file
      char const* filename = "cutlass_output_mismatch.csv";
      std::ofstream file(filename);
      file << "\nCUTLASS =\n" << C_cutlass.host_view();
      file << "\nReference =\n" << C_reference.host_view();

      std::cerr << "Error - CUTLASS GEMM kernel differs from reference. Wrote computed and reference results to '" << filename << "'" << std::endl;

      return cudaErrorUnknown;
  }
    return cudaSuccess;
}

int main(int argc, const char *arg[]){
    cudaDeviceProp prop;
    cudaError_t result = cudaGetDeviceProperties(&prop, 0);

    if (result != cudaSuccess) {
    std::cerr << "Failed to query device properties with error " << cudaGetErrorString(result) << std::endl;
    return -1;
  }

    // Checking if current device is half precision compatible
    if (!(prop.major > 5 || (prop.major == 5 && prop.minor >= 3))) {
      std::cerr << "This example uses half precision and is only suitable for devices with compute capability 5.3 or greater.\n";
      std::cerr << "You are using a CUDA device with compute capability " << prop.major << "." << prop.minor << std::endl;
      return -1;
    }

    int M = 256;
    int N = 256;
    int K = 128;

    cutlass::half_t alpha = 1.0_hf;
    cutlass::half_t beta = 0.0_hf;

    result = TestCutlassGemm(
      M,     // GEMM M dimension
      N,     // GEMM N dimension
      K,     // GEMM K dimension
      alpha,     // alpha
      beta     // beta
    );

    if (result == cudaSuccess) {
      std::cout << "Passed." << std::endl;
    }
    
    return result == cudaSuccess ? 0 : -1;
}
