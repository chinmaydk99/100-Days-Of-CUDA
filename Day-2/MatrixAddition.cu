#include <iostream>

__global__ void matAddKernel(float *A, float *B, float *C, int N)
{
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;

    if(col < N  && row < N)
    {
        C[row*N + col] = A[row*N + col] + B[row*N + col];
    }
}


int main()
{
    const int N = 20;
    float *A, *B, *C;

    A = (float *)malloc(N*N*sizeof(float));
    B = (float *)malloc(N*N*sizeof(float));
    C = (float *)malloc(N*N*sizeof(float));

    for(int i = 0; i < N; i++)
    {
        for(int j =0; j< N; j++)
        {
            A[i*N + j] = 3.0f;
            B[i*N + j] = 4.0f;
            C[i*N + j] = 0.0f;
        }
    }
    
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, N*N*sizeof(float));
    cudaMalloc((void**)&B_d, N*N*sizeof(float));
    cudaMalloc((void**)&C_d, N*N*sizeof(float));

    cudaMemcpy(A_d, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16,16);
    dim3 gridDim(ceil(N/16.0f), ceil(N/16.0f)); // Be careful about integer division here
    
    matAddKernel<<<gridDim,blockDim>>>(A_d,B_d,C_d,N);

    cudaMemcpy(C, C_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    printf("C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            printf("%.2f ",C[i * N + j]);
        }
        printf("\n");
    }
     printf("A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            printf("%.2f ", A[i * N + j]);
        }
        printf("\n");
    }
     printf("B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            printf("%.2f ", B[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}