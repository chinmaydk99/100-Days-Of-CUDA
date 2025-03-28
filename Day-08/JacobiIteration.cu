#include <iostream>
#include <cuda_runtime.h>

#define N 32  
#define MAX_ITER 1000
#define TOL 1e-6

#define TILE_DIM 8 
#define SHARED_DIM (TILE_DIM + 2)

__global__ void jacobi_iteration_3D_optimized(double *U_new, double *U_old, int N) {

    int i = blockIdx.x * TILE_DIM + threadIdx.x;
    int j = blockIdx.y * TILE_DIM + threadIdx.y;
    int k = blockIdx.z * TILE_DIM + threadIdx.z;


    __shared__ double inCur_s[SHARED_DIM][SHARED_DIM];


    double inPrev = (k > 0) ? U_old[i*N*N + j*N + (k-1)] : 0.0;
    double inCur  = U_old[i*N*N + j*N + k];
    double inNext = (k < N-1) ? U_old[i*N*N + j*N + (k+1)] : 0.0;


    if (threadIdx.x < SHARED_DIM && threadIdx.y < SHARED_DIM) {
        inCur_s[threadIdx.y][threadIdx.x] = inCur;
    }

    __syncthreads(); 

    if (i > 0 && i < N-1 && j > 0 && j < N-1 && k > 0 && k < N-1) {
        U_new[i*N*N + j*N + k] = (1.0 / 6.0) * (
            inCur_s[threadIdx.y][threadIdx.x-1] + inCur_s[threadIdx.y][threadIdx.x+1] +
            inCur_s[threadIdx.y-1][threadIdx.x] + inCur_s[threadIdx.y+1][threadIdx.x] +
            inPrev + inNext
        );
    }

    __syncthreads(); 

    inPrev = inCur;
    inCur = inNext;
    inCur_s[threadIdx.y][threadIdx.x] = inCur;
}

int main() {
    int size = N * N * N * sizeof(double);
    
    double *U_old, *U_new;
    cudaMallocManaged(&U_old, size);
    cudaMallocManaged(&U_new, size);


    for (int i = 0; i < N * N * N; i++) {
        U_old[i] = 1.0; 
        U_new[i] = 0.0;
    }


    dim3 threadsPerBlock(TILE_DIM, TILE_DIM, TILE_DIM);
    dim3 numBlocks(N / TILE_DIM, N / TILE_DIM, N / TILE_DIM);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        jacobi_iteration_3D_optimized<<<numBlocks, threadsPerBlock>>>(U_new, U_old, N);
        cudaDeviceSynchronize();

        double *temp = U_old;
        U_old = U_new;
        U_new = temp;
    }

    cudaFree(U_old);
    cudaFree(U_new);

    return 0;
}
