#include <iostream>
#include <cuda_runtime.h>


__global__ void matMultiplyKernel(float *A, float *B, float *C, int A_row, int width, int B_col)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if(row < A_row && col < B_col)
    {   
        float val = 0.0f;
        for(int k=0; k < width; k++)
        {
            val += A[row*width + k] * B[k*B_col + col]; 
        }
        C[row*B_col + col] = val;
    }
}

int main()
{   
    int M = 2;
    int N = 2;
    int P = 3; 
    float *A, *B, *C;

    A = (float *)malloc(M*N*sizeof(float));
    B = (float *)malloc(N*P*sizeof(float));
    C = (float *)malloc(M*P*sizeof(float));

    for(int i = 0; i < M; i++)
    {
        for(int j =0; j< N; j++)
        {
            A[i*N + j] = 1.0f;
        }
    }

    for(int i = 0; i < N; i++)
    {
        for(int j =0; j< P; j++)
        {
            B[i*P + j] = 2.0f;
        }
    }

    

    float *A_d, *B_d, *C_d;
    cudaMalloc((void**) &A_d, M*N*sizeof(float));
    cudaMalloc((void**) &B_d, N*P*sizeof(float));
    cudaMalloc((void**) &C_d, M*P*sizeof(float));

    cudaMemcpy(A_d, A, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N*P*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16,16);
    dim3 gridDim(ceil(P/16.0f), ceil(M/16.0f));

    matMultiplyKernel<<<gridDim,blockDim>>>(A_d,B_d,C_d,M,N,P);
    cudaDeviceSynchronize();

    cudaMemcpy(C, C_d, M*P*sizeof(float), cudaMemcpyDeviceToHost);

    printf("C:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {

            printf("%.2f ",C[i * P + j]);
        }
        printf("\n");
    }
     printf("A:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {

            printf("%.2f ", A[i * N + j]);
        }
        printf("\n");
    }
     printf("B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {

            printf("%.2f ", B[i * P + j]);
        }
        printf("\n");
    }

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}