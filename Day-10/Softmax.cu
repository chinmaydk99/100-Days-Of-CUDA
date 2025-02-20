#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define NUM_ROWS 4
#define NUM_COLS 8

__global__ void softmaxKernel(const float *input, float *output, int numCols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= NUM_ROWS) return;

    float maxVal = -INFINITY;
    for (int j = 0; j < numCols; j++) {
        float val = input[row * numCols + j];
        if (val > maxVal)
            maxVal = val;
    }

    float sumExp = 0.0f;
    for (int j = 0; j < numCols; j++) {
        sumExp += expf(input[row * numCols + j] - maxVal);
    }

    for (int j = 0; j < numCols; j++) {
        output[row * numCols + j] = expf(input[row * numCols + j] - maxVal) / sumExp;
    }
}

int main() {
    size_t size = NUM_ROWS * NUM_COLS * sizeof(float);

    float *h_input  = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    if (!h_input || !h_output) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    srand(time(NULL));
    for (int i = 0; i < NUM_ROWS * NUM_COLS; i++) {
        h_input[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
    }
    
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 32;
    int blocks = (NUM_ROWS + threadsPerBlock - 1) / threadsPerBlock;
    softmaxKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, NUM_COLS);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    for (int row = 0; row < NUM_ROWS; row++) {
        float rowSum = 0.0f;
        printf("Row %d: ", row);
        for (int col = 0; col < NUM_COLS; col++) {
            float val = h_output[row * NUM_COLS + col];
            rowSum += val;
            printf("%6.3f ", val);
        }
        printf(" | Sum: %6.3f\n", rowSum);
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}
