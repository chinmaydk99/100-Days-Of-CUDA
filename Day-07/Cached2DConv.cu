#include <iostream>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define FILTER_RADIUS 1
#define WIDTH 32
#define HEIGHT 32

//Storing the filter in constant memory
__constant__ float F[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];


__global__ void conv2DCached(float *N, float *P, int width, int height) {
    // Global column and row index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory tile to load inputs from global memory
    __shared__ float N_s[TILE_DIM][TILE_DIM];

    // Loading data into shared memory (no halo elements loaded)
    if (row < height && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col]; // Load from global memory
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f; // Zero padding for boundaries
    }

    __syncthreads();

    // Process only valid pixels
    if (col < width && row < height) {
        float Pvalue = 0.0f;

        // Loop over filter dimensions
        for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
            for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                // Compute shared memory indices (adjust for halo offset)
                int s_row = threadIdx.y + fRow - FILTER_RADIUS;
                int s_col = threadIdx.x + fCol - FILTER_RADIUS;
                
                // Check if the element is within the shared memory tile
                if (s_row >= 0 && s_row < TILE_DIM && s_col >= 0 && s_col < TILE_DIM) {
                    Pvalue += F[fRow*(2*FILTER_RADIUS+1) + fCol] * N_s[s_row][s_col];
                } else {
                    // Retrieve from global memory if outside the shared memory tile
                    int g_row = row + fRow - FILTER_RADIUS;
                    int g_col = col + fCol - FILTER_RADIUS;
                    if (g_row >= 0 && g_row < height && g_col >= 0 && g_col < width) {
                        Pvalue += F[fRow*(2*FILTER_RADIUS+1) + fCol] * N[g_row * width + g_col];
                    }
                }
            }
        }
        P[row * width + col] = Pvalue;
    }
}


// CPU Implementation for Verification
void convolution_cpu(const float *N, float *P, const float *F, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                    int imgRow = row + fRow - FILTER_RADIUS;
                    int imgCol = col + fCol - FILTER_RADIUS;
                    if (imgRow >= 0 && imgRow < height && imgCol >= 0 && imgCol < width) {
                        Pvalue += F[fRow * (2 * FILTER_RADIUS + 1) + fCol] * N[imgRow * width + imgCol];
                    }
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}

int main(){
    int size = WIDTH * HEIGHT * sizeof(float);
    int filterSize = (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float);

    float *h_N = (float *)malloc(size);
    float *h_P = (float *)malloc(size);
    float *h_P_cpu = (float *)malloc(size);
    float h_F[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];


    // Initialize input matrix and filter
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_N[i] = static_cast<float>(rand() % 10);
    }
    for (int i = 0; i < (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1); i++) {
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
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((WIDTH + TILE_DIM - 1) / TILE_DIM, (HEIGHT + TILE_DIM - 1) / TILE_DIM);

    // Launch CUDA Kernel
    conv2DCached<<<gridDim, blockDim>>>(d_N, d_P, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    // CPU Computation for verification
    convolution_cpu(h_N, h_P_cpu, h_F, WIDTH, HEIGHT);

    // Compare results
    bool match = true;
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        if (fabs(h_P[i] - h_P_cpu[i]) > 1e-5) {
            match = false;
            printf("Mismatch at %d: GPU = %f, CPU = %f\n", i, h_P[i], h_P_cpu[i]);
            break;
        }
    }

    if (match) {
        printf("Results match!\n");
    } else {
        printf("Results do NOT match!\n");
    }

    // Free memory
    free(h_N);
    free(h_P);
    free(h_P_cpu);
    cudaFree(d_N);
    cudaFree(d_P);


    return 0;
}