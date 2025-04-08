#include <cuda_runtime.h>
#include <cmath>

__global__ void average_pooling_3d(const float* __restrict__ input, int kernel_size,
int stride, int padding, float* __restrict__ output, size_t H, size_t W, size_t D,
int Hout, int Wout, int Dout){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int d = blockDim.z * blockIdx.z + threadIdx.z;

    if(col < Wout && row < Hout && d < Dout){
        int input_row_idx = row * stride - padding;
        int input_col_idx = col * stride - padding;
        int input_d_idx = d * stride - padding;

        float sum = 0.0f;

        for(int kx =0; kx < kernel_size; ++kx){
            for(int ky = 0; ky < kernel_size; ++ky){
                for(int kz = 0; kz < kernel_size; ++ kz){
                    int curr_row = input_row_idx + ky;
                    int curr_col = input_col_idx + kx;
                    int curr_d = input_d_idx + kz;

                    if(curr_row >= 0 && curr_row < H && curr_col >= 0 && curr_col < W && curr_d >= 0 && curr_d < D){
                        int offset = curr_d * H * W + curr_row * W + curr_col;
                        sum += input[offset];
                    }
                }
            }
        }
        int output_idx = d * Hout * Wout + row * Wout + col;
        output[output_idx] = sum / (kernel_size * kernel_size * kernel_size);
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H, size_t W, size_t D) {   
    int Hout = static_cast<int>(floor(static_cast<float>((H + 2 * padding - kernel_size) / stride + 1)));
    int Wout = static_cast<int>(floor(static_cast<float>((W + 2 * padding - kernel_size) / stride + 1)));
    int Dout = static_cast<int>(floor(static_cast<float>((D + 2 * padding - kernel_size) / stride + 1)));

    dim3 blockDim(8,8,4);
    dim3 gridDim((Wout + blockDim.x - 1)/ blockDim.x, (Hout + blockDim.y - 1)/ blockDim.y,
    (Dout + blockDim.z -1)/ blockDim.z);

    average_pooling_3d<<<gridDim, blockDim>>>(input, kernel_size, stride, padding, output, H, W, D, Hout, Wout, Dout);

    cudaDeviceSynchronize();
}