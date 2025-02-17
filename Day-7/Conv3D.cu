#include <iostream>
#include <cuda_runtime.h>

#define TILE_DIM 8
#define FILTER_RADIUS 1
#define WIDTH 32
#define HEIGHT 32
#define DEPTH 32

// Storing the filter in constant memory
__constant__ float F[(2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1)];

__global__ void conv3DCached(float *N, float *P, int width, int height, int depth) {
    // Compute global indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.z * blockDim.z + threadIdx.z;

    // Shared memory tile to load inputs from global memory
    __shared__ float N_s[TILE_DIM][TILE_DIM][TILE_DIM];

    // Load data into shared memory
    if (d < depth && row < height && col < width) {
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] = N[d * width * height + row * width + col]; // Load from global memory
    } else {
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f; // Zero padding for boundaries
    }

    __syncthreads(); // Synchronize before accessing shared memory

    // Process only valid pixels
    if (d < depth && col < width && row < height) {
        float Pvalue = 0.0f;

        // Loop over filter dimensions
        for (int fDepth = 0; fDepth < 2 * FILTER_RADIUS + 1; fDepth++) {
            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                    // Compute shared memory indices (adjust for halo offset)
                    int s_d = threadIdx.z + fDepth - FILTER_RADIUS;
                    int s_row = threadIdx.y + fRow - FILTER_RADIUS;
                    int s_col = threadIdx.x + fCol - FILTER_RADIUS;
                
                    // Check if the element is within the shared memory tile
                    if (s_d >= 0 && s_d < TILE_DIM &&
                        s_row >= 0 && s_row < TILE_DIM && 
                        s_col >= 0 && s_col < TILE_DIM) {
                        Pvalue += F[fDepth * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) +
                                    fRow * (2 * FILTER_RADIUS + 1) + fCol] * 
                                  N_s[s_d][s_row][s_col];
                    } else {
                        // Retrieve from global memory if outside the shared memory tile
                        int g_d = d + fDepth - FILTER_RADIUS;
                        int g_row = row + fRow - FILTER_RADIUS;
                        int g_col = col + fCol - FILTER_RADIUS;
                        if (g_d >= 0 && g_d < depth &&
                            g_row >= 0 && g_row < height &&
                            g_col >= 0 && g_col < width) {
                            Pvalue += F[fDepth * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) +
                                        fRow * (2 * FILTER_RADIUS + 1) + fCol] * 
                                      N[g_d * width * height + g_row * width + g_col];
                        }
                    }
                }
            }
        }
        // Store output in global memory
        P[d * width * height + row * width + col] = Pvalue;
    }
}



// CPU Implementation for Verification
void convolution_cpu(const float *N, float *P, const float *F, int width, int height, int depth) {
    for (int d = 0; d < depth; d++) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                float Pvalue = 0.0f;
                for (int fDepth = 0; fDepth < 2 * FILTER_RADIUS + 1; fDepth++) {
                    for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
                        for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                            int imgD = d + fDepth - FILTER_RADIUS;
                            int imgRow = row + fRow - FILTER_RADIUS;
                            int imgCol = col + fCol - FILTER_RADIUS;
                            if (imgD >= 0 && imgD < depth &&
                                imgRow >= 0 && imgRow < height &&
                                imgCol >= 0 && imgCol < width) {
                                Pvalue += F[fDepth * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) +
                                            fRow * (2 * FILTER_RADIUS + 1) + fCol] *
                                          N[imgD * width * height + imgRow * width + imgCol];
                            }
                        }
                    }
                }
                P[d * width * height + row * width + col] = Pvalue;
            }
        }
    }
}

int main() {
    int size = WIDTH * HEIGHT * DEPTH * sizeof(float);
    int filterSize = (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float);

    float *h_N = (float *)malloc(size);
    float *h_P = (float *)malloc(size);
    float *h_P_cpu = (float *)malloc(size);
    float h_F[(2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1)];

    // Initialize input matrix and filter
    for (int i = 0; i < WIDTH * HEIGHT * DEPTH; i++) {
        h_N[i] = static_cast<float>(rand() % 10);
    }
    for (int i = 0; i < (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1); i++) {
        h_F[i] = static_cast<float>(rand() % 3 - 1);
    }

    // Allocate device memory
    float *d_N, *d_P;
    cudaMalloc((void **)&d_N, size);
    cudaMalloc((void **)&d_P, size);

    // Copy input data to device
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, h_F, filterSize);

    // Define block and grid sizes
    dim3 blockDim(TILE_DIM, TILE_DIM, TILE_DIM);
    dim3 gridDim((WIDTH + TILE_DIM - 1) / TILE_DIM, (HEIGHT + TILE_DIM - 1) / TILE_DIM, (DEPTH + TILE_DIM - 1) / TILE_DIM);

    // Launch CUDA Kernel
    conv3DCached<<<gridDim, blockDim>>>(d_N, d_P, WIDTH, HEIGHT, DEPTH);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    // Free memory
    free(h_N); free(h_P); free(h_P_cpu);
    cudaFree(d_N); cudaFree(d_P);

    return 0;
}