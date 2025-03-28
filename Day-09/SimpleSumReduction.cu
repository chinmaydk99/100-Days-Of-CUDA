#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  
#define THREADS_PER_BLOCK (N)

__global__ void SimpleSumReductionKernel(float *input, float *output){
    unsigned int i = threadIdx.x;

    for(int stride = 1; stride < blockDim.x; stride *= 2){
        if(threadIdx.x + stride < blockDim.x){
            input[i] += input[i+stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        *output = input[0];
    }
}

int main() {
    float h_input[N], h_output;  // Host input array and result
    float *d_input, *d_output;   // Device input array and result
    
    // Initialize host input with values (for example, 1.0 for all elements)
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;  // Example: sum of N ones should be N
    }

    // Allocate memory on GPU
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with one block of THREADS_PER_BLOCK threads
    SimpleSumReductionKernel<<<1, THREADS_PER_BLOCK>>>(d_input, d_output);

    // Copy result from device to host
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Sum of array elements: %f\n", h_output);

    // Free memory on GPU
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}