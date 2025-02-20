#include <iostream>
#include <cuda_runtime.h>

#define N 8192
#define THREADS_PER_BLOCK 1024
#define SEGMENT_SIZE (4*THREADS_PER_BLOCK)

__global__ void MultiBlockReduction(float *input, float *output){
    __shared__ float input_ds[THREADS_PER_BLOCK];

    int segment = 4*blockDim.x*blockIdx.x;
    
    // Index in shared memory
    int t = threadIdx.x;

    // Index in global memory
    int i = segment + t;

    float sum = 0.0f;
    for(int j = 0; j <4; j++){
        int index = i + j * blockDim.x; // For two elements for threads we were doing i + blockDim.x
        if(index < N){
            sum += input[index];
        }
    }

    input_ds[t] = sum;
    __syncthreads();

    for(unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2){
        __syncthreads();
        input_ds[t] += input_ds[t+stride];
    }

    if(t == 0){
        atomicAdd(output, input_ds[0]);
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
    int numBlocks = (N + SEGMENT_SIZE - 1)/ SEGMENT_SIZE;

    MultiBlockReduction<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_output);

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);


    printf("Sum of array elements: %f\n", h_output);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}