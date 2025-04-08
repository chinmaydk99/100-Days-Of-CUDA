#include <cuda_runtime.h>
#include <cmath>

__global__ void average_pooling_1D(const float* __restrict__ input, int kernel_size,
    int stride, int padding, float* __restrict__ output, size_t H, int Hout){
    int out_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(out_idx < Hout){
        int start_idx_window = out_idx * stride - padding;

        float current_sum = 0.0f;
        int count = kernel_size; // even the padding elements are included

        #pragma unroll // Loop Unrolling
        for(int k = 0; k < kernel_size; ++k){

            int curr_idx =  start_idx_window + k;

            if(curr_idx >= 0 && curr_idx < H){
                current_sum += input[curr_idx];
            }
        }

        if(count > 0){
            output[out_idx] = current_sum / count;
        }else{
            output[out_idx] = 0.0f;
        }

    }
}

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H) {    
    int Hout =  static_cast<int>(floor(static_cast<float>((H + 2*padding - kernel_size)/ stride + 1)));

    dim3 blockDim(256);
    dim3 gridDim((Hout + 255)/ 256);

    average_pooling_1D<<<gridDim, blockDim>>>(input, kernel_size, stride, padding, output, H, Hout);
    cudaDeviceSynchronize(); 

}