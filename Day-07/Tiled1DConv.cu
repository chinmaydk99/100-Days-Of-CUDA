#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>

#define MASK_LENGTH 7

// Allocating space for the mask in constant memory
__constant__ int mask[MASK_LENGTH];

__global__ void conv1D(int *array, int *result, int n) {
    // Global thread ID calculation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
    // Store all elements needed to compute output in shared memory
    extern __shared__ int s_array[];
  
    // r: The number of padded elements on either side
    int r = MASK_LENGTH / 2;
  
    // d: The total number of padded elements
    int d = 2 * r;
  
    // Size of the padded shared memory array
    int n_padded = blockDim.x + d;
  
    // Offset for the second set of loads in shared memory
    int offset = threadIdx.x + blockDim.x;
  
    // Global offset for the array in DRAM
    int g_offset = blockDim.x * blockIdx.x + offset;
  
    // Load the lower elements first starting at the halo
    // This ensure divergence only once
    s_array[threadIdx.x] = array[tid];
  
    // Load in the remaining upper elements
    if (offset < n_padded) {
      s_array[offset] = array[g_offset];
    }
    __syncthreads();
  
    // Temp value for calculation
    int temp = 0;
  
    // Go over each element of the mask
    for (int j = 0; j < MASK_LENGTH; j++) {
      temp += s_array[threadIdx.x + j] * mask[j];
    }
  
    // Write-back the results
    result[tid] = temp;
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

    // Calculate padded length
    int r = MASK_LENGTH / 2;
    int n_p = n + r*2;

    int bytes_p = n_p * sizeof(int);
    int bytes_m = MASK_LENGTH * sizeof(int);
    int bytes_n = n * sizeof(int);
    
    // Allocate host arrays
    int *h_array = new int[n_p];
    int *h_mask = new int[MASK_LENGTH];
    int *h_result = new int[n];
    
    // Initialize arrays
    for(int i = 0; i < n_p; i++){
        // Initialising the padding values with zeroes followed by normal initialisation
        if((i < r) || (i >= (n+r))){
            h_array[i] = 0.0f;
        }else{
            h_array[i] = rand() % 100;
        }
    }
    
    for(int i = 0; i < m; i++) {
        h_mask[i] = rand() % 10;
    }
    
    // Allocate device arrays
    int *d_array, *d_mask, *d_result;
    cudaMalloc((void**)&d_array, bytes_p);
    cudaMalloc((void**)&d_result, bytes_n);
    
    // Copy data to device
    cudaMemcpy(d_array, h_array, bytes_n, cudaMemcpyHostToDevice);
    
    // We use a different call to load the data directly into the symbol
    cudaMemcpyToSymbol(mask, h_mask, bytes_m);
    
    // Launch kernel
    int THREADS = 256;
    int GRID = (n + THREADS - 1) / THREADS;  // Ceiling division
    
    // Amount of sharing memory needed
    // Each thread calculates one output element plus we need to account for padding
    size_t SHMEM = (THREADS + r*2)*sizeof(int);

    conv1D<<<GRID, THREADS, SHEMM>>>(d_array, d_result, n);  // Fixed: now passing d_mask instead of d_array twice
    
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