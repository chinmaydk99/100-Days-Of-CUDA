#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

int main(int argc, char const **args) {

  // A matrix configuration
  using ElementA = cutlass::half_t;                                
  using  LayoutA = cutlass::layout::RowMajor;       
  constexpr int AlignmentA  = 128 /cutlass::sizeof_bits<ElementA>::value; // Defining how many elements can be read in a single memory transaction


  // B matrix configuration
  using         ElementB    = cutlass::half_t;
  using         LayoutB     = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    

  // C/D matrix configuration
  using         ElementC    = cutlass::half_t;
  using         LayoutC     = cutlass::layout::ColumnMajor;
   // We don't need alignment since this is only used in epilogue and alignment is handled separately

  // Core kernel configurations
  using ElementAccumulator  = float; // Element type for internal accumulation
  using ArchTag  = cutlass::arch::Sm80;       // For A100
  using OperatorClass  = cutlass::arch::OpClassTensorOp; // Using tensor cores
  using TilesShape = Shape<_128,_128,_64>; // Threadblock-level tile size
  using ClusterShape  = Shape<_1,_2,_1>; // Shape of the threadblocks in a cluster
  using StageCountType = cutlass::gemm::collective::StageCountAuto;// Software Pipelining Stage count maximized based on the tile size
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;  // Kernel to launch based on the default setting in the Collective Builder

  // Building the main GEMM compute loop
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TilesShape, ClusterShape,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  // Epilogue: Final Part of GEMM, it is this part that needs to be tuned for kernel fusion
  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>>; // This currently does alpha*AB + beta*C. We can use another Epilogue like LinearCombinationBias, LinearCombinationRelu or LinearCombinationClamp.

  // Glue that transforms the composition into a launchable reusable kernel
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int>, // (M, N, K)
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  Gemm gemm_op;
  cutlass::Status status;


  // Define input sizes

  int M = 512;
  int N = 256;
  int K = 128;

  float alpha = 1.25f;
  float beta = -1.25f;

  // Allocate device memory

  cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
  cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
  cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
  cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;

  // Inferred from Collective Builder
  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  // Inferred from collective epilogue
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  block_A.reset(M * K);
  block_B.reset(K * N);
  block_C.reset(M * N);
  block_D.reset(M * N);

  // Launch GEMM on the device

  status = gemm_op({
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    block_A.get(),
    stride_A,
    block_B.get(),
    stride_B,
    {block_C.get(), stride_C, block_D.get(), stride_D, {alpha, beta}}
  });

  if (status != cutlass::Status::kSuccess) {
    return -1;
  }

  return 0;
}
