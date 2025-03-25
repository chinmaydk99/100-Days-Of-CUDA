#include <iostream>
#include <cute/tensor.hpp>
#include <cuda_runtime.h>

using namespace cute;

#define CUDA_CHECK(err) do { if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
    exit(EXIT_FAILURE); \
}} while (0)

__global__ void naive_transpose_kernel(float *d_src, float *d_dst, int M, int N){
  constexpr int TILE_DIM = 32;

  // Defining tile dim
  using b = Int<TILE_DIM>;

  // Defining shape for the input and the output
  // auto tensor_shape = make_shape(Int<M>{}, Int<N>{});
  // auto tensor_shape_trans = make_shape(Int<N>{}, Int<M>{});
  auto tensor_shape = make_shape(M,N);
  auto tensor_shape_trans = make_shape(N, M);

  // Defining layouts
  auto layout_src = make_layout(tensor_shape, GenRowMajor{});
  auto layout_dst = make_layout(tensor_shape_trans, GenRowMajor{});

  // Wrapping pointers into multidimensional tensor views with layouts
  Tensor tensor_S = make_tensor(make_gmem_ptr(d_src), layout_src);
  Tensor tensor_D = make_tensor(make_gmem_ptr(d_dst), layout_dst);

  // Creating column major layout and using (M,N) for shape
  auto layout_dst_T = make_layout(tensor_shape, GenColMajor{});

  // Creating column major view of dst tensor
  Tensor tensor_DT = make_tensor(make_gmem_ptr(d_dst), layout_dst_T);

  auto block_shape = make_shape(b{}, b{});

  // Tile division
  Tensor tiled_S = tiled_divide(tensor_S, block_shape);
  Tensor tiled_DT = tiled_divide(tensor_DT, block_shape);

  // Get the tile assigned to block
  auto tile_S = tiled_S(make_coord(_, _), blockIdx.x, blockIdx.y);
  auto tile_DT = tiled_DT(make_coord(_,_), blockIdx.x, blockIdx.y);

  // Thread layout
  auto thread_layout = make_layout(make_shape(Int<8>{}, Int<32>{}), GenRowMajor{});

  // Local partition per thread
  Tensor thread_tile_S =  local_partition(tile_S, thread_layout, threadIdx.x);
  Tensor thread_tile_DT = local_partition(tile_DT, thread_layout, threadIdx.x);

  Tensor rmem = make_tensor_like(thread_tile_S);
  copy(thread_tile_S, rmem);
  copy(rmem, thread_tile_DT);

}

void host_transpose(float *dst, float *src, int M, int N){
  for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            dst[j * M + i] = src[i * N + j];
}

int main() {
    // const int M = 2048;
    // const int N = 2048;
    const int M = 4;
    const int N = 4;
    size_t bytes = M * N * sizeof(float);

    float *h_input = new float[M * N];
    float *h_output = new float[M * N];
    float *h_ref = new float[M * N];

    // Initialize host input
    for (int i = 0; i < M * N; ++i)
        h_input[i] = static_cast<float>(i);

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(256);
    // dim3 gridDim(N / 32, M / 32);
    dim3 gridDim((N + 31) / 32, (M + 31) / 32);

    naive_transpose_kernel<<<gridDim, blockDim>>>(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    std::cout << "Input Matrix (4x4):\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_input[i * N + j] << "\t";
        }
        std::cout << "\n";
    }

    std::cout << "\nTransposed Output Matrix (4x4):\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            std::cout << h_output[i * M + j] << "\t";
        }
        std::cout << "\n";
        }

    // Reference check
    host_transpose(h_ref, h_input, M, N);

    bool correct = true;
    for (int i = 0; i < M * N; ++i) {
        if (fabs(h_ref[i] - h_output[i]) > 1e-4) {
            std::cout << "Mismatch at " << i << ": " << h_ref[i] << " vs " << h_output[i] << std::endl;
            correct = false;
            break;
        }
    }

    if (correct) std::cout << "✅ Transpose PASSED\n";
    else         std::cout << "❌ Transpose FAILED\n";

    delete[] h_input;
    delete[] h_output;
    delete[] h_ref;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
