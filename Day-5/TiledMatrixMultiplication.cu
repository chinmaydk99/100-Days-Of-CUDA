#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void tiledMatmulKernel(float *M, float *N, float *P, int j, int k, int l)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = ty + by*TILE_WIDTH;
    int Col = tx + bx*TILE_WIDTH;

    float Pvalue = 0;

    for(int ph = 0; ph < ceil(k/(float)TILE_WIDTH); ++ph)
    {
        if(Row < j && (ph*TILE_WIDTH + tx) < k)
        {
            Mds[ty][tx] = M[Row*j + ph*TILE_WIDTH + tx];
        }
        else
        {
            Mds[ty][tx] = 0.0;
        }

        if(Col < l && (ph*TILE_WIDTH+ty) < k)
        {
            Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*l + Col];
        }
        else
        {
            Nds[ty][tx] = 0.0;
        }

        __syncthreads();

        for(int i=0; i < k; ++i)
        {
            Pvalue += Mds[ty][i] * Nds[i][tx];
        }
        __syncthreads();
    }

    if(Row < j && Col < l)
    {
        P[Row*l + Col] = Pvalue;
    }
}

int main()
{
    int J = 4, K = 4, L = 4;
    float M[J*K], N[K*L], P[J*L];

    for (int i = 0; i < J * K; ++i) M[i] = rand() % 10;
    for (int i = 0; i < K * L; ++i) N[i] = rand() % 10;

    std::cout << "Matrix M:\n";
    for (int i = 0; i < J; ++i) {
        for (int j = 0; j < K; ++j) std::cout << M[i * K + j] << " ";
        std::cout << std::endl;
    }

    std::cout << "\nMatrix N:\n";
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < L; ++j) std::cout << N[i * L + j] << " ";
        std::cout << std::endl;
    }

    float *M_d, *N_d, *P_d;
    cudaMalloc((void**)&M_d, J*K*sizeof(float));
    cudaMalloc((void**)&N_d, K*L*sizeof(float));
    cudaMalloc((void**)&P_d, J*L*sizeof(float));

    cudaMemcpy(M_d, M, J * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N, K * L * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(P_d, 0, J * L * sizeof(float));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize(ceil(L/(float)TILE_WIDTH), ceil(J/(float)TILE_WIDTH));

    tiledMatmulKernel<<<gridSize, dimBlock>>>(M_d, N_d, P_d, J, K, L);
    cudaMemcpy(P, P_d, J * L * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\nResult Matrix P:\n";
    for (int i = 0; i < J; ++i) {
        for (int j = 0; j < L; ++j) std::cout << P[i * L + j] << " ";
        std::cout << std::endl;
    }

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

    return 0;
}