// Full GEMM implementation using CUTLASS PredicatedTileIterator

#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/layout/pitch_linear.h>  
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>
#include <cutlass/transform/pitch_linear_thread_map.h> 
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/device_memory.h>
#include <iostream>

__global__ void gemm_with_tile_iterator(
    cutlass::half_t *A,
    cutlass::half_t *B,
    cutlass::half_t *C,
    int M, int N, int K,
    cutlass::half_t alpha, 
    cutlass::half_t beta,
    int lda, 
    int ldb,
    int ldc
){
    using Element = cutlass::half_t;

    using ThreadBlockShape = cutlass::gemm::GemmShape<64, 64, 16>;

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_id = threadIdx.x;

    // Getting this tile's starting row and col
    int row = block_row * ThreadBlockShape::kM;
    int col = block_col * ThreadBlockShape::kN;

    // Defining the shared memory tiles
    __shared__ Element Asub[ThreadBlockShape::kM][ThreadBlockShape::kK];
    __shared__ Element Bsub[ThreadBlockShape::kK][ThreadBlockShape::kN];

    // Define PitchLinear layout for matrices
    using LayoutA = cutlass::layout::PitchLinear;
    using LayoutB = cutlass::layout::PitchLinear;
    
    // Setup layouts with the appropriate strides
    LayoutA layoutA(lda);
    LayoutB layoutB(ldb);

    // Predicated iterators to load A and B from global memory
    using ShapeA = cutlass::layout::PitchLinearShape<ThreadBlockShape::kM, ThreadBlockShape::kK>;
    using ThreadMapA = cutlass::transform::PitchLinearStripminedThreadMap<ShapeA, 128>;
    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
        ShapeA, Element, LayoutA, 1, ThreadMapA>;

    using ShapeB = cutlass::layout::PitchLinearShape<ThreadBlockShape::kK, ThreadBlockShape::kN>;
    using ThreadMapB = cutlass::transform::PitchLinearStripminedThreadMap<ShapeB, 128>;
    using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
        ShapeB, Element, LayoutB, 1, ThreadMapB>;

    // Create the parameters for the iterators using the layouts
    typename IteratorA::Params params_A(layoutA);
    typename IteratorB::Params params_B(layoutB);

    // Create the extent for the matrix
    cutlass::Coord<2> extent_A = cutlass::make_Coord(M, K);
    cutlass::Coord<2> extent_B = cutlass::make_Coord(K, N);

    // Create offsets for the iterators
    cutlass::Coord<2> threadblock_offset_A = cutlass::make_Coord(row, 0);
    cutlass::Coord<2> threadblock_offset_B = cutlass::make_Coord(0, col);

    // Initialize the iterators with appropriate offsets
    IteratorA iterator_A(params_A, A, extent_A, thread_id, threadblock_offset_A);
    IteratorB iterator_B(params_B, B, extent_B, thread_id, threadblock_offset_B);

    // Create fragments to hold loaded data
    typename IteratorA::Fragment fragment_A;
    typename IteratorB::Fragment fragment_B;

    // Define the result fragment type
    using FragmentC = cutlass::Array<Element, IteratorA::Fragment::kElements>;
    FragmentC fragment_C;

    // Initialize the output fragment
    for(int i = 0; i < FragmentC::kElements; ++i){
        fragment_C[i] = Element(0);
    }

    // Loop over the K dimension in tiles
    for (int tile_k = 0; tile_k < K; tile_k += ThreadBlockShape::kK) {
        // Load data from global memory
        iterator_A.load(fragment_A);
        iterator_B.load(fragment_B);

        // Manually load data from fragments to shared memory
        // Get the current threadblock offset
        int threadblock_k = tile_k;

        // Copy from fragments to shared memory
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < IteratorA::Fragment::kElements; ++i) {
            // Calculate the local indices within the shared memory tile
            int thread_m = i % ThreadBlockShape::kM;
            int thread_k = i / ThreadBlockShape::kM;

            if (row + thread_m < M && threadblock_k + thread_k < K) {
                Asub[thread_m][thread_k] = fragment_A[i];
            }
        }

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < IteratorB::Fragment::kElements; ++i) {
            // Calculate the local indices within the shared memory tile
            int thread_k = i % ThreadBlockShape::kK;
            int thread_n = i / ThreadBlockShape::kK;

            if (threadblock_k + thread_k < K && col + thread_n < N) {
                Bsub[thread_k][thread_n] = fragment_B[i];
            }
        }

        __syncthreads();

        // Compute the matrix multiplication for this tile
        CUTLASS_PRAGMA_UNROLL
        for (int kk = 0; kk < ThreadBlockShape::kK; ++kk) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < FragmentC::kElements; ++i) {
                int local_row = i % ThreadBlockShape::kM;
                int local_col = i / ThreadBlockShape::kM;
                if ((row + local_row) < M && (col + local_col) < N) {
                    // Manually accumulate using direct values rather than += operator
                    Element temp = Asub[local_row][kk] * Bsub[kk][local_col];
                    fragment_C[i] = Element(float(fragment_C[i]) + float(temp));
                }
            }
        }

        __syncthreads();
        
        // Move to the next tile
        ++iterator_A;
        ++iterator_B;
    }

    // Write the computed results back to global memory
    int thread_start_row = row;
    int thread_start_col = col;

    // Write results back to global memory
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FragmentC::kElements; ++i) {
        int local_row = i % ThreadBlockShape::kM;
        int local_col = i / ThreadBlockShape::kM;
        int global_row = thread_start_row + local_row;
        int global_col = thread_start_col + local_col;
        
        if (global_row < M && global_col < N) {
            // Convert from logical coordinates to linear index
            int index = global_row + global_col * ldc;
            C[index] = Element(float(alpha) * float(fragment_C[i]) + float(beta) * float(C[index]));
        }
    }
}

// --- Host-side driver code ---
cudaError_t AllocateAndInitializeTensors(
    int M, int N, int K,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& A,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& B,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& C,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& C_ref
) {
    A.reset(cutlass::MatrixCoord(M, K));
    B.reset(cutlass::MatrixCoord(K, N));
    C.reset(cutlass::MatrixCoord(M, N));
    C_ref.reset(cutlass::MatrixCoord(M, N));

    uint64_t seed = 2025;
    cutlass::half_t mean = cutlass::half_t(0.0f);
    cutlass::half_t stddev = cutlass::half_t(4.0f);

    int bits_less_than_one = 0;

    cutlass::reference::device::TensorFillRandomGaussian(
        A.device_view(), seed, mean, stddev, bits_less_than_one);
    cutlass::reference::device::TensorFillRandomGaussian(
        B.device_view(), seed * 101, mean, stddev, bits_less_than_one);
    cutlass::reference::device::TensorFillRandomGaussian(
        C.device_view(), seed * 303, mean, stddev, bits_less_than_one);

    cutlass::device_memory::copy_device_to_device(
        C_ref.device_data(), C.device_data(), C.capacity());

    return cudaSuccess;
}

cudaError_t RunGEMMTileIterator(int M, int N, int K, cutlass::half_t alpha, cutlass::half_t beta) {
    using Element = cutlass::half_t;
    using Layout = cutlass::layout::ColumnMajor;

    cutlass::HostTensor<Element, Layout> A, B, C, C_ref;
    AllocateAndInitializeTensors(M, N, K, A, B, C, C_ref);

    using ThreadBlockShape = cutlass::gemm::GemmShape<64, 64, 16>;
    int grid_x = (N + ThreadBlockShape::kN - 1) / ThreadBlockShape::kN;
    int grid_y = (M + ThreadBlockShape::kM - 1) / ThreadBlockShape::kM;

    dim3 grid(grid_x, grid_y);
    dim3 block(128);

    gemm_with_tile_iterator<<<grid, block>>>(
        A.device_data(), B.device_data(), C.device_data(),
        M, N, K, alpha, beta,
        A.stride(0), B.stride(0), C.stride(0)
    );

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(result) << std::endl;
        return result;
    }

    C.sync_host();

    cutlass::reference::host::Gemm<Element, Layout, Element, Layout, Element, Layout, Element, Element> gemm_ref;
    gemm_ref({M, N, K}, alpha, A.host_ref(), B.host_ref(), beta, C_ref.host_ref());

    if (!cutlass::reference::host::TensorEquals(C_ref.host_view(), C.host_view())) {
        std::cerr << "Mismatch between CUTLASS and reference result!\n";
        return cudaErrorUnknown;
    }

    std::cout << "\u2705 GEMM passed verification!\n";
    return cudaSuccess;
}

int main() {
    int M = 256, N = 256, K = 256;
    cutlass::half_t alpha = cutlass::half_t(1.0f);
    cutlass::half_t beta  = cutlass::half_t(0.0f);

    cudaError_t status = RunGEMMTileIterator(M, N, K, alpha, beta);
    return status == cudaSuccess ? 0 : -1;
}
