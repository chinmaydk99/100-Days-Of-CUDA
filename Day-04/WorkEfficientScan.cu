#include <iostream>
#include <cuda_runtime.h>

#define N 8

__global__ void efficientPrefixSum(float *in, float *out, int n)
{
    extern __shared__ float temp[];

    int thread_id = threadIdx.x;
    
    if(thread_id >= n) return;

    temp[thread_id] = in[thread_id];
    __syncthreads();

    // Up-Sweep (Reduction Phase)
    for (int d = 1; d < n; d *= 2)
    {
        int index = (thread_id + 1) * 2 * d - 1;
        if (index < n)
        {
            temp[index] += temp[index - d];
        }
        __syncthreads();
    }

    // Set the last element to 0 for exclusive scan
    if (thread_id == 0)
    {
        temp[n - 1] = 0;
    }
    __syncthreads();

    // Down-Sweep (Distribution Phase)
    for (int d = n / 2; d >= 1; d /= 2)
    {
        int index = (thread_id + 1) * 2 * d - 1;
        if (index < n)
        {
            float t = temp[index - d];
            temp[index - d] = temp[index];
            temp[index] += t;
        }
        __syncthreads();
    }
    out[thread_id] = temp[thread_id];
}

int main()
{
    float in_h[N] = {1,2,3,4,5,6,7,8};
    float out_h[N];

    float *in_d, *out_d;
    cudaMalloc((void**)&in_d, N*sizeof(float));
    cudaMalloc((void**)&out_d, N*sizeof(float));

    cudaMemcpy(in_d, in_h, N*sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = N;

    int sharedMemory = N*sizeof(float);

    efficientPrefixSum<<<1, blockSize, sharedMemory>>>(in_d, out_d, N);

    cudaMemcpy(out_h, out_d, N*sizeof(float), cudaMemcpyDeviceToHost);

     std::cout << "Input  : ";
     for (int i = 0; i < N; i++) {
         std::cout << in_h[i] << " ";
     }
     std::cout << "\nOutput : ";
     for (int i = 0; i < N; i++) {
         std::cout << out_h[i] << " ";
     }
     std::cout << std::endl;

    cudaFree(in_d);
    cudaFree(out_d);

    return 0;
}