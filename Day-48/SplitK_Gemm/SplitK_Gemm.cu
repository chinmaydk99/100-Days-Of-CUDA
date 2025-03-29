#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

// Datatye for the input, output, the accumulator and the final output
using ElementAccumulator = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;

// Layout of the input and output matrices
using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// Use tensor cores
using MMAOp = cutlass::arch::OpClassTensorOp;

// CUDA SM architecture number
using SmArch = cutlass::arch::Sm75; // For tesla T4 gpu of mine

// Thread block tile size
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;

// Warp tile size
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;

// MMA op tile size
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;

// Using simple alpha * (AB) + beta * C epilogue
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, // Data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value, // Number of elements per vectorized access. For half precision, its 8
    ElementAccumulator,
    ElementComputeEpilogue>;
    
using Gemm = cutlass::gemm::device::GemmSplitKParallel<ElementInputA,
                                                       LayoutInputA,
                                                       ElementInputB,
                                                       LayoutInputB,
                                                       ElementOutput,
                                                       LayoutOutput,
                                                       ElementAccumulator,
                                                       MMAOp,
                                                       SmArch,
                                                       ShapeMMAThreadBlock,
                                                       ShapeMMAWarp,
                                                       ShapeMMAOp,
                                                       EpilogueOp>;

int run() {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major != 7) {
    std::cerr << "Volta Tensor Ops must be run on a machine with compute capability of 70, 72, or 75."
              << std::endl;

  return 0;
  }

  const int length_m = 2560;
  const int length_n = 2048;
  const int length_k = 2048;

  // Defining problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
  
  // Initialising the tensors
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.mk());
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(problem_size.kn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(problem_size.mn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(problem_size.mn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(problem_size.mn());

  // Filling the outpuct matrices on host
  cutlass::reference::host::TensorFillRandomUniform(
    tensor_a.host_view(), 
    1, 
    ElementInputA(4), 
    ElementInputA(-4), 
    0);

  cutlass::reference::host::TensorFillRandomUniform(
    tensor_b.host_view(), 
    1, 
    ElementInputB(4), 
    ElementInputB(-4), 
    0);

  cutlass::reference::host::TensorFillRandomUniform(
    tensor_c.host_view(), 
    1, 
    ElementOutput(4), 
    ElementOutput(-4), 
    0);

  // Filling the output matrices with zeros
  cutlass::reference::host::TensorFill(
    tensor_d.host_view());

  cutlass::reference::host::TensorFill(
    tensor_ref_d.host_view());

  
  // Copy Data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha and beta
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1); // Basically in this case its float(1)
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  // Split K dimension into 16 partitions
  int split_k_slices = 16;
  
  // Gemm Kernel arguments
  typename Gemm::Arguments arguments{problem_size,
                                     tensor_a.device_ref(),
                                     tensor_b.device_ref(),
                                     tensor_c.device_ref(),
                                     tensor_d.device_ref(),
                                     {alpha, beta},
                                     split_k_slices};
  
  // Amount of memory required for the gemm kernel
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocating the memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
 
  Gemm gemm_op;

  // Intialising gemm witb the arguments
  cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // Launching the initialized gemm
  status = gemm_op();
  CUTLASS_CHECK(status);

  // Reference GEMM Kernel (unlike earlier even this is supposed to be run on GPU)
  cutlass::reference::device::Gemm<ElementInputA,
                                   LayoutInputA,
                                   ElementInputB,
                                   LayoutInputB,
                                   ElementOutput,
                                   LayoutOutput,
                                   ElementComputeEpilogue,
                                   ElementComputeEpilogue> gemm_device;

  gemm_device(problem_size,
              alpha,
              tensor_a.device_ref(),
              tensor_b.device_ref(),
              beta,
              tensor_c.device_ref(),
              tensor_ref_d.device_ref());

  cudaDeviceSynchronize();

  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  bool passed = cutlass::reference::host::TensorEquals(
    tensor_d.host_view(),
    tensor_ref_d.host_view());

  std::cout << (passed ? "Passed" : "Failed") << std::endl;

  return (passed ? 0 : -1);
}

int main(){
    run();
}


