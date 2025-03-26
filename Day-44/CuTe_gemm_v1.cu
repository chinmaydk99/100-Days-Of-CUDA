#include <iostream>
#include <cuda_runtime.h>


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
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == INT<3>{}); / {M,N,K}

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
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // {BLK_M, BLK_K}, In step we are specifying that we will be moving along M and K modes
    // This means local_tile(mA, select<0,2>(cta_tiler), select<0,2>(cta_coord))
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // {BLK_K, BLK_N}
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // {BLK_M, BLK_N}

    // Shared memory buffers
    __shared__ TA smemA[cosize_v<ASmemLayout>]; // cosize will give total elements in ASmemLayout. 
    __shared__ TB smemB[cosize_v<BSmemLayout>];

    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // BLOCK_SIZE_M , BLOCK_SIZE_K
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // BLOCK_SIZE_N , BLOCK_SIZE_K

    
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
    // Static layouts arse more efficient and allow CuTe to have optimized implementations

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
    
    gemm_device<<<dimGrid, dimBlock 0, stream>>>(
        prob_shape, cta_tiler,
        A, dA, sA, tA,
        B, dB, sB, tB,
        C, dC, sC, tC,
        alpha, beta);
}
