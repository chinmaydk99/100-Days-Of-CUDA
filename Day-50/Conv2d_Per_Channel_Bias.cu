#include <iostream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/host_reorder.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/convolution.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

using ElementAccumulator = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = float;

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int NumStages = 4;
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;

using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementComputeEpilogue,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  cutlass::arch::OpMultiplyAdd,
  IteratorAlgorithm
>::Kernel;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

int run() {

  cutlass::conv::Conv2dProblemSize problem_size(
    {1, 7, 7, 512},
    {512, 3, 3, 512},
    {1, 1, 1, 1},
    {1, 1},
    {1, 1},
    cutlass::conv::Mode::kCrossCorrelation,
    1
  );

  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.activation_extent());
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(problem_size.filter_extent());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c_bias({1, 1, 1, problem_size.K});
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(problem_size.output_extent());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(problem_size.output_extent());

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
      tensor_c_bias.host_view(),
      1,
      ElementOutput(4),
      ElementOutput(-4),
      0);
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());

  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c_bias.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);

  typename ImplicitGemm::Arguments arguments{
    problem_size,
    tensor_a.device_ref(),
    tensor_b.device_ref(),
    {tensor_c_bias.device_data(), LayoutOutput::Stride(0)},
    tensor_d.device_ref(),
    {alpha} };

  ImplicitGemm implicit_gemm_op;

  size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = implicit_gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  status = implicit_gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  status = implicit_gemm_op();
  CUTLASS_CHECK(status);

  cutlass::reference::device::Conv2d<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementComputeEpilogue,
      ElementAccumulator,
      cutlass::NumericConverter<ElementOutput, ElementComputeEpilogue>>
    (
      cutlass::conv::Operator::kFprop,
      problem_size,
      tensor_a.device_ref(),
      tensor_b.device_ref(),
      tensor_c_bias.device_ref(),
      tensor_ref_d.device_ref(),
      alpha, ElementComputeEpilogue(0)
    );

  cudaDeviceSynchronize();

  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  for (int n = 0; n < problem_size.N; ++n) {
    for (int p = 0; p < problem_size.P; ++p) {
      for (int q = 0; q < problem_size.Q; ++q) {
        for (int k = 0; k < problem_size.K; ++k) {
          tensor_ref_d.at({n, p, q, k}) =
              std::max(ElementOutput(0),
                       ElementOutput(tensor_ref_d.at({n, p, q, k}) +
                                     tensor_c_bias.at({0, 0, 0, k})));
        }
      }
    }
  }

  std::cout << (cutlass::reference::host::TensorEquals(tensor_d.host_view(),
                                                       tensor_ref_d.host_view())
                    ? "Passed"
                    : "Failed")
            << std::endl;

  CUTLASS_CHECK(status);
  return 0;
}

int main(int argc, char const **args) {

  bool notSupported = false;

  if (!(__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

  if (!(props.major >= 8)) {
    std::cerr << "Ampere Tensor Ops must be run on a machine with compute capability at least 80."
              << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    return 0;
  }

  return run();
}