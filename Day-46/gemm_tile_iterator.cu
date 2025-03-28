
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
    using Layout = cutlass::layout::ColumnMajor;

    using ThreadBlockShape = cutlass::gemm::GemmShape<64, 64, 16>;
    // using WarpShape = cutlass::gemm:GemmShape<32, 64, 16>; Warp Level MMA
    // using InstructionShape = cutlass::gemm:GemmShape<16, 8, 8>; Tensor core ops(Wmma or MMA)

    using Element = cutlass::half_t;
    using Layout = cutlass::layout::ColumnMajor;

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_id = threadIdx.x;

    // Getting this tile's starting row and col
    int row = block_row * ThreadBlockShape::kM;
    int col = block_col * ThreadBlockShape::kN;

    // Defining the shared memory tiles
    __shared__ cutlass::half_t Asub[ThreadBlockShape::kM][ThreadBlockShape::kK];
    __shared__ cutlass::half_t Bsub[ThreadBlockShape::kK][ThreadBlockShape::kN];

    //--------------------------------------------------
    // Defining Predicated iterators to load A and B from global memory onto threads
    //---------------------------------------------------

    // ThreadMap for A
    using ShapeA = cutlass::layout::PitchLinearShape<
        ThreadBlockShape::kM,
        ThreadBlockShape::kK
    >;

    using ThreadMapA = cutlass::transform::PitchLinearStripminedThreadMap<
        ShapeA, // tile shape
        128 // Number of threads in block
    >;

    // Tile Iterator for A
    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
        ShapeA,
        cutlass::half_t, // type A
        cutlass::layout::PitchLinear,  // Layout for loading A
        1, // elements to be loaded per access
        ThreadMapA
    >;

    // Creating Layout for global memory
    cutlass::layout::ColumnMajor layoutA(lda);

    // Telling iterator how to handle striding in global memory access
    typename IteratorA::Params params_A(layoutA);

    // Defining bounds 
    cutlass::Coord<2> extent_A = cutlass::make_Coord(M, K);

    // Instantiating the iterator for A
    IteratorA iterator_A(params_A, A, extent_A, thread_id);

    // Tile Iterator for B
    using ShapeB = cutlass::layout::PitchLinearShape<
        ThreadBlockShape::kK,
        ThreadBlockShape::kN
    >;

    using ThreadMapB = cutlass::transform::PitchLinearStripminedThreadMap<
        ShapeB, // tile shape
        128 // Number of threads in block
    >;

    // Tile Iterator for A
    using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
        ShapeB,
        cutlass::half_t, // type B
        cutlass::layout::PitchLinear,  // Layout for loading B
        1, // elements to be loaded per access
        ThreadMapB
    >;

    // Creating Layout for global memory
    cutlass::layout::ColumnMajor layoutB(ldb);

    // Telling iterator how to handle striding in global memory access
    typename IteratorB::Params params_B(layoutB);

    // Defining bounds 
    cutlass::Coord<2> extent_B = cutlass::make_Coord(K, N);

    // Instantiating the iterator for A
    IteratorB iterator_B(params_B, B, extent_B, thread_id);

    //---------------------
    // Fragment Declaration
    //---------------------
    typename IteratorA::Fragment fragment_A;
    
    typename IteratorB::Fragment fragment_B;

    // Defining fragment for C where the accumulation takes place
    using Fragment_C = cutlass::Array<cutlass::half_t, IteratorA::Fragment::kElements>;
    // Type, number of elements

    Fragment_C fragment_C;

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < Fragment_C::kElements; ++i){
        fragment_C[i] = cutlass::half_t(0);
    }

    //-----------------------------------
    // Fragment Wise Accumulation
    //--------------------------------

    for(int tile_k = 0; tile_k < K; tile_k += ThreadBlockShape::kK){// Looping over each k Tile
         // Load tile from global into register fragment_A and fragment_B
        iterator_A.load(fragment_A);
        iterator_B.load(fragment_B);

        // Convert iterator offsets to shared memory positions
        cutlass::MatrixCoord offset_A = iterator_A.thread_offset();
        cutlass::MatrixCoord offset_B = iterator_B.thread_offset();

        // Store fragments into shared memory tiles
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < IteratorA::Fragment::kElements; ++i) {
            int row = offset_A.row() + i % ThreadBlockShape::kM;
            int col = offset_A.column() + i / ThreadBlockShape::kM;
            Asub[row][col] = fragment_A[i];
        }

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < IteratorB::Fragment::kElements; ++i) {
            int row = offset_B.row() + i % ThreadBlockShape::kK;
            int col = offset_B.column() + i / ThreadBlockShape::kK;
            Bsub[row][col] = fragment_B[i];
        }

        __syncthreads();

        // Now we need to loop over the ThreadBlockShape:kK k values within this tile
        CUTLASS_PRAGMA_UNROLL
        for(int kk =0; kk < ThreadBlockShape::kK; ++kk){
            
            CUTLASS_PRAGMA_UNROLL
            // Iterating over all elements of fragment_C

            for(int i = 0;i < Fragment_C::kElements; ++i){
                int a_row = row 
            }
        }
    }

}

cudaError_t AllocateAndInitializeTensors(
    int M, int N, int K,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& A,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& B,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& C,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& C_ref,
){
    // Resetting the shapes for A, B
    A.reset(cutlass::MatrixCoord(M, K));
    B.reset(cutlass::MatrixCoord(K, N));
    C.reset(cutlass::MatrixCoord(M, N));
    C_ref.reset(cutlass::MatrixCoord(M, N));

    uint64_t seed = 2025;
    cutlass::half_t mean = 0.0_hf;
    cutlass::half_t stdddev = 4.0_hf;

    int bits_less_than_one = 0;

    cutlass::reference::device::TensorFillRandomGaussian(
        A.device_view(), seed, mean, stddev, bits_less_than_one);

    cutlass::reference::device::TensorFillRandomGaussian(
        B.device_view(), seed*101, mean, stddev, bits_less_than_one);

    cutlass::reference::device::TensorFillRandomGaussian(
        C.device_view(), seed*303, mean, stddev, bits_less_than_one);

    // Copy C to C_ref on device for correct GEMM output
    cutlass::device_memory::copy_device_to_device(C_ref.device_data(), C.device_data(), C.capacity());

    return cudaSuccess;
}