// We are performing alpha * AB + beta * C

#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/host_tensor.h>
#include <iostream>

int main(){
    // Defining the GEMM operation
    using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, // Half precision for A
      cutlass::layout::ColumnMajor, // Layout of A
      cutlass::half_t, // Half precision for B
      cutlass::layout::ColumnMajor, //Layout for B
      cutlass::half_t, // Output half precision
      cutlass::layout::ColumnMajor, // Layout for output
      float, // Accumulator datatype
      cutlass::arch::OpClassTensorOp, // Using tensor core instructions instead of regular FP16 arithmetic. Can also use OpClassSimt - regular CUDA cores
      cutlass::arch::Sm75 // Target GPU architecture 
    >;

    Gemm gemm_op;
    cutlass::Status status; // To check whether GEMM ran correctly ot not

    // Defining the inout shapes
    int M = 512;
    int N = 256;
    int K = 128;

    float alpha = 0.0f;
    float beta = 0.0f;


    // Allocating Device Memory
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A({M,K});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B({K,N});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C({M,N});
    // This allocates memory on host as well as device

    // Getting raw device pointers using.device_data()
    cutlass::half_t const *ptrA = A.device_data();
    cutlass::half_t const *ptrB = B.device_data();
    cutlass::half_t const *ptrC = C.device_data(); // Used in epilogue alphaAB + beta*C
    cutlass::half_t *ptrD = C.device_data(); // This is or the final output that we write to C's memory

    // Strides(leading dimensions)
    // In column major layout leading dimension is th number of rows
    // For row major we would need number of columns so stride(1)
    int lda = A.device_ref().stride(0);
    int ldb = B.device_ref().stride(0);
    int ldc = C.device_ref().stride(0);
    int ldd = C.device_ref().stride(0);

    // Test values for the matrices
    cutlass::reference::host::TensorFill(A.host_view(), cutlass::half_t(1.0));
    cutlass::reference::host::TensorFill(B.host_view(), cutlass::half_t(2.0));
    cutlass::reference::host::TensorFill(C.host_view(), cutlass::half_t(3.0));

    A.sync_device();
    B.sync_device();
    C.sync_device();

    //Timing the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launching GEMM on device
    status = gemm_op({
      {M,N,K},
      {ptrA, lda},
      {ptrB, ldb},
      {ptrC, ldc},
      {ptrD, ldd},
      {alpha, beta}
    });

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GEMM took: " << milliseconds << " ms\n";


    if (status != cutlass::Status::kSuccess){
      return -1;
    }
    C.sync_host(); 
    
    // Print top-left 4x4 of output
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        std::cout << float(C.at({i, j})) << " ";
      }
      std::cout << "\n";
    }

    return 0;
}