#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define N 8

__global__ void naive_scan(float *in, float *out, int n)
{
    extern __shared__ float temp[];

    int thread_id = threadIdx.x;

    if(thread_id >= n) return;

    temp[thread_id] = in[thread_id];
    __syncthreads();


    for(int d =1; d<n; d *= 2)
    {
        float t = 0; 
        if (thread_id >= d)
        {
            t = temp[thread_id] + temp[thread_id - d]; 
        }
        __syncthreads(); // Making sure all read operations have been completed before we proceed

        if (thread_id >= d)
        {
            temp[thread_id] = t;
        }
        __syncthreads();
    }

    out[thread_id] = temp[thread_id];
}

int main()
{
    float ip_h[N] = {1,2,3,4,5,6,7,8};
    float op_h[N];

    float *ip_d, *op_d;
    cudaMalloc((void**)&ip_d, N*sizeof(float));
    cudaMalloc((void**)&op_d, N*sizeof(float));

    cudaMemcpy(ip_d, ip_h, N*sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = N;

    int sharedMemory = N*sizeof(float);

    naive_scan<<<1, blockSize, sharedMemory>>>(ip_d, op_d, N);

    cudaMemcpy(op_h, op_d, N*sizeof(float), cudaMemcpyDeviceToHost);

     std::cout << "Input  : ";
     for (int i = 0; i < N; i++) {
         std::cout << ip_h[i] << " ";
     }
     std::cout << "\nOutput : ";
     for (int i = 0; i < N; i++) {
         std::cout << op_h[i] << " ";
     }
     std::cout << std::endl;

    cudaFree(ip_d);
    cudaFree(op_d);

    return 0;
} 