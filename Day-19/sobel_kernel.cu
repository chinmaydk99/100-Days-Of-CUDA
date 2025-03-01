#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <torch/extension.h>

#define FILTER_RADIUS 1
#define TILE_DIM 32
#define FILTER_SIZE (2*FILTER_RADIUS+1)

__constant__ float Sobel_X_C[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];
__constant__ float Sobel_Y_C[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];

__global__ void SobelFilterKernel(float *input, float *output, int width, int height){
    unsigned int row = blockDim.y*blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x*blockIdx.x + threadIdx.x;

    __shared__ float input_ds[TILE_DIM][TILE_DIM];

    if(row < height && col < width){
        input_ds[threadIdx.y][threadIdx.x] = input[row*width + col];
    }else{
        input_ds[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    if(row < height && col < width){
        float g_x = 0.0f;
        float g_y = 0.0f;
        for(int fRow = 0; fRow < 2*FILTER_RADIUS+1; ++ fRow){
            for(int fCol = 0; fCol < 2*FILTER_RADIUS+1; ++fCol){
                int sRow = threadIdx.y + fRow - FILTER_RADIUS;
                int sCol = threadIdx.x + fCol - FILTER_RADIUS;

                if(sRow >= 0 && sRow < TILE_DIM && sCol >=0 && sCol < TILE_DIM){
                    g_x += Sobel_X_C[fRow*(2*FILTER_RADIUS+1) + fCol] * input_ds[sRow][sCol];
                    g_y += Sobel_Y_C[fRow*(2*FILTER_RADIUS+1) + fCol] * input_ds[sRow][sCol];
                }else{
                    int gRow = row + fRow - FILTER_RADIUS;
                    int gCol = col + fCol - FILTER_RADIUS;
                    if(gRow >= 0 && gRow < height && gCol >= 0 && gCol < width){
                        g_x += Sobel_X_C[fRow*(2*FILTER_RADIUS+1) + fCol] * input[gRow*width + gCol];
                        g_y += Sobel_Y_C[fRow*(2*FILTER_RADIUS+1) + fCol] * input[gRow*width+ gCol];
                    }
                }
            }
        }
        output[row*width + col] = sqrtf(g_x * g_x + g_y * g_y);
    }
}

void initialize_sobel_filters() {
    float Sobel_X_H[FILTER_SIZE * FILTER_SIZE] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };

    float Sobel_Y_H[FILTER_SIZE * FILTER_SIZE] = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };

    cudaMemcpyToSymbol(Sobel_X_C, Sobel_X_H, sizeof(Sobel_X_H));
    cudaMemcpyToSymbol(Sobel_Y_C, Sobel_Y_H, sizeof(Sobel_Y_H));
}

torch::Tensor sobel_cuda_forward(torch::Tensor input){
    input = input.contiguous();

    const int height = input.size(0);
    const int width = input.size(1);

    auto output = torch::zeros_like(input);

    const dim3 blockSize(16,16);
    const dim3 gridSize((width+ 15)/ 16, (height+15)/16);

    initialize_sobel_filters();


    SobelFilterKernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), width, height);
    // We are getting raw pointers to GPU memory, without these we can't use tensors inside GPU Kernels
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
    }

    return output;
}