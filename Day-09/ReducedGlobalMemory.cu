#include <iostream>
#include <cuda_runtime.h>

#define N 1024  
#define THREADS_PER_BLOCK (N/2)

__global__ void SumKernelReducedGM(float *input, float *output){
    __shared__ float input_ds[THREADS_PER_BLOCK];
    unsigned int i = threadIdx.x;
    unsigned int globalIdx = i + blockDim.x;

    input_ds[i] = input[i] + ((globalIdx < N)?input[globalIdx]:0.0f); // Coalescing 

    for(unsigned int stride = blockDim.x/2; stride >=1 ; stride /= 2){
        __syncthreads();
        if(i < stride){
            input_ds[i] += input_ds[i + stride];
        }     
    }

    if(i == 0){
        *output = input_ds[0];
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

    SumKernelReducedGM<<<1, THREADS_PER_BLOCK>>>(d_input, d_output);

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);


    printf("Sum of array elements: %f\n", h_output);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}