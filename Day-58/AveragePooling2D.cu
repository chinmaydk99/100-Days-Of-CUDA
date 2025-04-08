#include <cuda_runtime.h>
#include <cmath>

__global__ void average_pooling_2d(const float* __restrict__ input, int kernel_size, int stride,
int padding, float* __restrict__ output,size_t H, size_t W, int Hout, int Wout){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(col < Wout && row < Hout){
        int input_col_idx = col * stride - padding;
        int input_row_idx = row * stride - padding;

        float sum = 0.0f;

        for(int kx = 0; kx < kernel_size; ++ kx){
            for(int ky = 0; ky < kernel_size; ++ ky){
                int curr_row = input_row_idx + ky;
                int curr_col = input_col_idx + kx;

                if(curr_row >= 0 && curr_row < H && curr_col >= 0 & curr_col < W){
                    int offset = curr_row * W + curr_col;
                    sum += input[offset];
                }
            }
        }

        int output_index = row * Wout + col;
        float average = sum / (kernel_size * kernel_size);
        output[output_index] = average;
    }
}

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H, size_t W) {
    int Wout = static_cast<int>(floor(static_cast<float>((W + 2 * padding - kernel_size)/ stride + 1)));
    int Hout = static_cast<int>(floor(static_cast<float>((H + 2 * padding - kernel_size)/ stride + 1)));

    dim3 blockDim(16,16);
    dim3 gridDim((W+blockDim.x -1)/ blockDim.x, (H + blockDim.y -1)/ blockDim.y);

    average_pooling_2d<<<gridDim, blockDim>>>(input, kernel_size, stride,
    padding, output, H, W, Hout, Wout);
    cudaDeviceSynchronize();
}   