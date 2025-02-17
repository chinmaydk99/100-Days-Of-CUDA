#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>

#define IN_TILE_DIM 32
#define FILTER_RADIUS 1
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

// Declaring the filter in constant memory
__constant__ float F[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];

__global__ void conv2D(float *N, float *P, int width, int height){
    // Compute global row and column indices
    int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    // Shared memory tile for the current block
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];

    // Loading input tile into shared memory handling boundary cases
    if(row >= 0 && row < height && col >= 0 && col < width){
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    }else{
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Computing outputs only for the valid inner region of the tile and not the padding tokens
    // This will be negative or go outside bound for padding tokens
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    // Ensuring the thread is within the valid output region
    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
            float Pvalue = 0.0f;

            // Apply convolution filter
            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                    Pvalue += F[fRow * (2 * FILTER_RADIUS + 1) + fCol] * 
                              N_s[tileRow + fRow][tileCol + fCol];
                }
            }

            // Store the result in the output array
            P[row * width + col] = Pvalue;
        }
    }

}


void verify_result(float *N, float *F, float *P, int width, int height) {
    float *P_cpu = (float*)malloc(width * height * sizeof(float));

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float Pvalue = 0.0f;

            for (int fRow = -FILTER_RADIUS; fRow <= FILTER_RADIUS; fRow++) {
                for (int fCol = -FILTER_RADIUS; fCol <= FILTER_RADIUS; fCol++) {
                    int nRow = row + fRow;
                    int nCol = col + fCol;

                    if (nRow >= 0 && nRow < height && nCol >= 0 && nCol < width) {
                        Pvalue += F[(fRow + FILTER_RADIUS) * (2 * FILTER_RADIUS + 1) + (fCol + FILTER_RADIUS)] *
                                  N[nRow * width + nCol];
                    }
                }
            }

            P_cpu[row * width + col] = Pvalue;
        }
    }

    // Compare CPU vs. GPU results
    for (int i = 0; i < width * height; i++) {
        if (fabs(P[i] - P_cpu[i]) > 1e-4) {
            std::cerr << "Mismatch at index " << i << ": GPU " << P[i] << " vs CPU " << P_cpu[i] << std::endl;
            free(P_cpu);
            return;
        }
    }

    std::cout << "Verification PASSED!" << std::endl;
    free(P_cpu);
}


void verify_result(float *N, float *F, float *P, int width, int height);

int main() {
    int width = 64, height = 64;
    int size = width * height * sizeof(float);
    int filterSize = (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float);

    float *h_N = (float*)malloc(size);
    float *h_P = (float*)malloc(size);
    float *h_F = (float*)malloc(filterSize);

    // Initialize input matrix and filter
    for (int i = 0; i < width * height; i++) {
        h_N[i] = rand() % 256; // Random values
    }
    for (int i = 0; i < (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1); i++) {
        h_F[i] = 1.0 / 9.0; // Example: Simple averaging filter
    }

    // Allocate device memory
    float *d_N, *d_P;
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);

    // Copy input data and filter to device
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, h_F, filterSize); // Copy filter to constant memory

    // Define grid and block dimensions
    dim3 blockDim(IN_TILE_DIM, IN_TILE_DIM);
    dim3 gridDim((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, 
                 (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    // Launch the kernel
    conv2D<<<gridDim, blockDim>>>(d_N, d_P, width, height);

    // Copy result back to host
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    // Verify the result with CPU computation
    verify_result(h_N, h_F, h_P, width, height);

    // Free memory
    cudaFree(d_N);
    cudaFree(d_P);
    free(h_N);
    free(h_P);
    free(h_F);

    return 0;
}

