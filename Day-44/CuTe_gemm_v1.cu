#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"


template <class ProblemShape, // {M, N, K}
        class CtaTiler, // Shape or layout to split data into tiles (blocks)
        class TA, class AStride, class ASmemLayout, class AThreadLayout, // Type, Stride, Shared Memory and Thread block layout
        class TB, class BStride, class BSmemLayout, class BThreadLayout,
        class TC, class CStride, class CSmemLayout, class CThreadLayout,
        class Alpha, class Beta> // Scalars used in epilogue

__global__ static 
__launch_bounds__(decltype(size(CThreadLayout{}))::value) // Informs the compiler: kernel always launches with exactly size(CThreadLayout) threads
void
gemm_device(ProblemShape shape_MNK,
            CtaTiler cta_tiler,
            TA const *A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const *B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC * C, CStride dC, CSmemLayout sC_layout, CThreadLayout tC,
            Alpha alpha, Beta beta)
{
    using namespace cute;
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == INT<3>{}); // {M,N,K}

    CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA)); // {M, K}
    CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB)); //{N,K} and not {K,N} as per BLAS convention
    CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC)); // {M, N}

    // Representing the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // {M,K} with stride dA = {1, lda} for nt {lda,1} for tn
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // {N,K} with stride dB = {1, ldb} for nt {ldb,1} for tn
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // {M,N} with stride = {1,ldc} for nt and {ldc,1} for tn

    // Getting the blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _); // (m, n, k)
    // k not specified since it will be looped over, we'll be needing all tiles
    
    // Global tensor reference, tile shape, which row and column specifically within the block, which modes to move along
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // {BLK_M, BLK_K, k}, In step we are specifying that we will be moving along M and K modes
    // This means local_tile(mA, select<0,2>(cta_tiler), select<0,2>(cta_coord))
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // {BLK_K, BLK_N, k}
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // {BLK_M, BLK_N}

    // Shared memory buffers
    __shared__ TA smemA[cosize_v<ASmemLayout>]; // cosize will give total elements in ASmemLayout. 
    __shared__ TB smemB[cosize_v<BSmemLayout>];

    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // BLOCK_SIZE_M , BLOCK_SIZE_K
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // BLOCK_SIZE_N , BLOCK_SIZE_K


    // Copying tiles from global to shared memory 
    // Naive way would be to use a single thread per block for this which while correct would be extremely slow
    // Efficient way is to use threads to partition the copy
    // We define a thread layout and partition the data so that each thread has its own view of a small chunk in both gA and sA
    // In this case we already have access to the thread layouts as part of the inputs
    Tensor tAgA = local_partition(gA, tA, threadIdx.x); // [THR_M, THR_K, k]
    Tensor tAsA = local_partition(sA, tA, threadIdx.x); // [THR_M, THR_K]

    Tensor tBgB = local_partition(gB, tB, threadIdx.x); // [THR_N, THR_K, k]
    Tensor tBsB = local_partition(sB, tB, threadIdx.x); // [THR_N, THR_K]

    CUTE_STATIC_ASSERT_V(size<0>(tAgA) == size<0>(tAsA)); // THR_M
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // THR_K
    CUTE_STATIC_ASSERT_V(size<0>(tBgB) == size<0>(tBsB)); // THR_N
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)) ; // THR_K

    // Local tile vs local partition: local tile does CTA level tiling while local partition does thread level partitioning
    // For calculating the product again we define a thread layout to partition the workload

    // Using thread layout tC to project threads onto rows of sA , columns of sB
    // Partition sA by rows of tC. Only applying thread row indexing
    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{}); // (THR_M, BLOCK_SIZE_K), A rows for this thread
    // Partition sB by cols of tC. Only applying thread column indexing
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{}); // (THR_N, BLOCK_SIZE_K)
    // Partition gC by tile of tC
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{}); //{THR_M, THR_N}

    // Accumulator
    Tensor tCrC = make_tensor_like(tCgC); // (THR_M, THR_N)

    CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCsA)); // THR_M
    CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCgC)); // THR_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<1>(tCgC)); // THR_N                // THR_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<0>(tCsB)); // THR_N               // THR_N
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCsB)); // BLOCK_K

    // Main loop : Load-Compute Accumulate
    auto K_TILE_MAX = size<2>(tAgA); // size<2>(tAgA) = k = Number of K tiles CTA will loop over

    for(int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile){
        // Per thread copy from global to shared memory
        copy(tAgA(_, _, k_tile), tAsA); // A (THR_M, THR_K) -> (THR_M, THR_K) 
        copy(tBgB(_, _, k_tile), tBsB); // B (THR_N, THR_K) -> (THR_N, THR_K)

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        gemm(tCsA, tCsB, tCrC); // (THR_M, THR_N) += (THR_M, BLK_K) * (THR_N, BLK_K)
        __syncthreads();
    }

    axpby(alpha, tCrC, beta, tCgC); // C = alpha * Accumulator + beta * C
}


// M-major smem sA, n-major smem sB, mn-major threads tA | tB
// M and N are column major
// As per convention in the docs, a matrix is X-major if stride in X-mode is 1
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemma_nt(int m, int n, int k,
            Alpha alpha,
            TA const* A, int ldA,
            TB const* B, int ldB,
            Beta beta,
            TC *C, int ldC,
            cudaStream_t stream = 0)
{
    using namespace cute;

    auto M = int(m);
    auto N = int(n);
    auto K = int(k);

    auto prob_shape = make_shape(M,N,K);

    // Defining strides (column major)
    auto dA = make_stride(Int<1>{}, ldA); // {dM, dK} {1,ldA}
    auto dB = make_stride(Int<1>{}, ldB); // {dN, dK} {1, ldB}
    auto dC = make_stride(Int<1>{}, ldC); // {dM, dN} {1, ldC}

    // Defining Tile Sizes
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};

    auto cta_tiler = make_shape(bM, bN, bK); // {BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K}
    // This will be fed to a composition function to obtain tiles

    // Defining the shared memory layouts
    // Static layouts are more efficient and allow CuTe to have optimized implementations

    auto sA = make_layout(make_shape(bM, bK)); // {BLOCK_SIZE_M, BLOCK_SIZE_K}, M-major
    auto sB = make_layout(make_shape(bN, bK)); // {BLOCK_SIZE_N, BLOCK_SIZE_K}, N-major
    auto sC = make_layout(make_shape(bM, bN)); // {BLOCK_SIZE_M, BLOCK_SIZE_N}, M-major

    // Defining thread layouts
    auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
    auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

    dim3 dimBlock(size(tC)); // This matches the __launch__bounds before the kernel 

    dim3 dimGrid(size(ceil_div(M, bM)), 
                size(ceil_div(N, bN)));
    
    gemm_device<<<dimGrid, dimBlock, 0, stream>>>(
        prob_shape, cta_tiler,
        A, dA, sA, tA,
        B, dB, sB, tB,
        C, dC, sC, tC,
        alpha, beta);
}

int main(int argc, char** argv)
{
  int m = 5120;
  if (argc >= 2)
    sscanf(argv[1], "%d", &m);

  int n = 5120;
  if (argc >= 3)
    sscanf(argv[2], "%d", &n);

  int k = 4096;
  if (argc >= 4)
    sscanf(argv[3], "%d", &k);

  char transA = 'N';
  if (argc >= 5)
    sscanf(argv[4], "%c", &transA);

  char transB = 'T';
  if (argc >= 6)
    sscanf(argv[5], "%c", &transB);

  using TA = float;
  using TB = float;
  using TC = float;
  using TI = float;

  TI alpha = 1.0;
  TI beta  = 0.0;

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
  std::cout << "C = A^" << transA << " B^" << transB << std::endl;

  cute::device_init(0);

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TB> h_B(n*k);
  thrust::host_vector<TC> h_C(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  double gflops = (2.0*m*n*k) * 1e-9;

  const int timing_iterations = 100;
  GPU_Clock timer;

  int ldA = 0, ldB = 0, ldC = m;

  if (transA == 'N') {
    ldA = m;
  } else if (transA == 'T') {
    ldA = k;
  } else {
    assert(false);
  }

  if (transB == 'N') {
    ldB = k;
  } else if (transB == 'T') {
    ldB = n;
  } else {
    assert(false);
  }
  // Run once
  d_C = h_C;
  gemm(transA, transB, m, n, k,
       alpha,
       d_A.data().get(), ldA,
       d_B.data().get(), ldB,
       beta,
       d_C.data().get(), ldC);
  CUTE_CHECK_LAST();
  thrust::host_vector<TC> cute_result = d_C;

  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm(transA, transB, m, n, k,
         alpha,
         d_A.data().get(), ldA,
         d_B.data().get(), ldB,
         beta,
         d_C.data().get(), ldC);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);
  return 0;
}