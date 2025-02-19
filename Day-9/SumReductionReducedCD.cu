#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  
#define THREADS_PER_BLOCK (N)

__global__ void SimpleSumReductionKernel(float *input, float *output){
    unsigned int i = threadIdx.x;

    for(int stride = blockDim.x / 2; stride >= 1; stride /= 2){
        if(threadIdx.x < stride){
            input[i] += input[i+stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        *output = input[0];
    }
}

int main() {
    float h_input[N], h_output; 
    float *d_input, *d_output;  
    
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f; 
    }

    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    SimpleSumReductionKernel<<<1, THREADS_PER_BLOCK>>>(d_input, d_output);

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);


    printf("Sum of array elements: %f\n", h_output);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}