#include <iostream>
#include <cuda_runtime.h>

__global__ void conv2Dkernel(float *N, float *F, float *P, int r, int width, int height)
{
    // Calculating the element in the output array for which the current thread will be responsible
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    float Pvalue = 0.0f
    for(int f_row = 0; f_row < 2*r+1; f_row++){
        for(int f_col = 0; f_column < 2*r+1; f_col++){
            int inRow = outRow + f_row -r;
            int inCol = outCol + f_col -r;

            if(inRow >= 0 and inRow < height and inCol >= 0 and inCol < width){
                Pvalue += F[f_Row][f_Col]*N[inRow*width + inCol];
            }
        }
    }
    P[outRow][outCol] = Pvalue;
}