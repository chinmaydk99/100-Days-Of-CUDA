#include <iostream>
#include <cuda_runtime.h>

#define N 128
#define OUT_TILE_DIM 8
#define IN_TILE_DIM (OUT_TILE_DIM+2)

__global__ void stencil3D(float *in, float *out, unsigned int N){
    int iStart = blockIdx.z*OUT_TILE_DIM;
    int j = max(0,blockIdx.y*OUT_TILE_DIM + threadIdx.y - 1);
    int k = max(0,blockIdx.x*OUT_TILE_DIM + threadIdx.x - 1);

    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCur_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];

    if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCur_s[threadIdx.y][threadIdx.x] = in[(iStart * N * N) + j * N + k];
    }
    if (iStart-1 >= 0 && iStart-1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1) * N * N + j * N + k];
    }


    for(int i = iStart; i < iStart + OUT_TILE_DIM; ++i){
        if (i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1) * N * N + j * N + k];
        }

        __syncthreads();

        if(i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >=1 && k < N-1){
            if (threadIdx.x > 0 && threadIdx.x < IN_TILE_DIM - 1 &&
                threadIdx.y > 0 && threadIdx.y < IN_TILE_DIM - 1){
                    out[i * N * N + j * N + k] = 0.1f * inCur_s[threadIdx.y][threadIdx.x - 1] +
                                                0.2f * inCur_s[threadIdx.y][threadIdx.x + 1] +
                                                0.3f * inCur_s[threadIdx.y - 1][threadIdx.x] +
                                                0.4f * inCur_s[threadIdx.y + 1][threadIdx.x] +
                                                0.5f * inPrev_s[threadIdx.y][threadIdx.x] +
                                                0.6f * inNext_s[threadIdx.y][threadIdx.x] +
                                                0.7f * inCur_s[threadIdx.y][threadIdx.x];
                }        
        }
        __syncthreads();
        inPrev_s[threadIdx.y][threadIdx.x] = inCur_s[threadIdx.y][threadIdx.x];
        inCur_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
    }
}


int main() {
    size_t size = N * N * N * sizeof(float);
    float* d_in, * d_out, * h_in, * h_out;

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    h_in = (float*)malloc(size);
    h_out = (float*)malloc(size);

    // Initialize input data
    for (int i = 0; i < N * N * N; i++) h_in[i] = static_cast<float>(rand()) / RAND_MAX;

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 blockDim(IN_TILE_DIM, IN_TILE_DIM);
    dim3 gridDim(N / OUT_TILE_DIM, N / OUT_TILE_DIM, N / OUT_TILE_DIM);

    stencil3D<<<gridDim, blockDim>>>(d_in, d_out, N);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}