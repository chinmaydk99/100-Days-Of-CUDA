#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>

#define MASK_LENGTH 7

// Allocating space for the mask in constant memory
__constant__ int mask[MASK_LENGTH];

__global__ void conv1D(int *array, int *result, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {  // Add bounds check
        int r = MASK_LENGTH / 2;

        // For index i, we go from i-r to i+r
        int start = tid - r;
        int temp = 0;
        
        // Iterate over every element in the mask
        for(int j = 0; j < MASK_LENGTH; j++) {
            if((start + j >= 0) && (start + j < n)) {
                temp += array[start + j] * mask[j];
            }
        }
        result[tid] = temp;
    }
}

void verify_result(int *array, int *mask, int *result, int n, int m) {
    int radius = m / 2;
    int temp;
    int start;
    for (int i = 0; i < n; i++) {
        start = i - radius;
        temp = 0;
        for (int j = 0; j < m; j++) {
            if ((start + j >= 0) && (start + j < n)) {
                temp += array[start + j] * mask[j];
            }
        }
        assert(temp == result[i]);
    }
}

int main() {
    // Number of elements in the result array
    int n = 1 << 20;
    int bytes_n = n * sizeof(int);
    
    // Number of elements in the convolution mask
    int m = 7;
    int bytes_m = m * sizeof(int);
    
    // Allocate host arrays
    int *h_array = new int[n];
    int *h_mask = new int[m];
    int *h_result = new int[n];
    
    // Initialize arrays
    for(int i = 0; i < n; i++) {
        h_array[i] = rand() % 10;
    }
    
    for(int i = 0; i < m; i++) {
        h_mask[i] = rand() % 10;
    }
    
    // Allocate device arrays
    int *d_array, *d_mask, *d_result;
    cudaMalloc((void**)&d_array, bytes_n);
    cudaMalloc((void**)&d_result, bytes_n);
    
    // Copy data to device
    cudaMemcpy(d_array, h_array, bytes_n, cudaMemcpyHostToDevice);
    
    // We use a different call to load the data directly into the symbol
    cudaMemcpyToSymbol(mask, h_mask, bytes_m);
    
    // Launch kernel
    int THREADS = 256;
    int GRID = (n + THREADS - 1) / THREADS;  // Ceiling division
    
    conv1D<<<GRID, THREADS>>>(d_array, d_result, n);  // Fixed: now passing d_mask instead of d_array twice
    
    // Copy result back to host
    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);
    
    // Verify results
    verify_result(h_array, h_mask, h_result, n, m);
    
    // Cleanup
    delete[] h_array;
    delete[] h_mask;
    delete[] h_result;
    cudaFree(d_array);
    cudaFree(d_result);
    
    return 0;
}