#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define NUM_REPS 100

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        assert(result == cudaSuccess);
    }
    return result;
}

void postprocess(const float* reference, const float* data, int size, float ms) {
    double maxError = 0;
    for (int i = 0; i < size; ++i)
        maxError = std::max(maxError, static_cast<double>(std::abs(reference[i] - data[i])));
    std::cout << "    Max error: " << maxError << ", Time: " << ms << " ms" << std::endl;
}

__global__ void copySharedMem(float *odata, const float *idata){
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    int width = gridDim.x * TILE_DIM;

    __shared__ float tile[TILE_DIM][TILE_DIM];

    for (int k = 0; k < TILE_DIM; k += BLOCK_ROWS) {
        tile[threadIdx.y + k][threadIdx.x] = idata[(y + k) * width + x];
    }
    __syncthreads();

    for (int k = 0; k < TILE_DIM; k += BLOCK_ROWS) {
        odata[(y + k) * width + x] = tile[threadIdx.y + k][threadIdx.x];
    }
}

// Global memory copy
__global__ void copy(float *odata, const float *idata){
    int x = blockIdx.x * TILE_DIM + threadIdx.x; // Columns
    int y = blockIdx.y * TILE_DIM + threadIdx.y; // Rows
    
    int width = gridDim.x * TILE_DIM; // Total cols = number of tiles multiplied by size of each tile

    for(int k = 0; k < TILE_DIM; k += BLOCK_ROWS){
        odata[(y+k) * width + x] = idata[(y+k) * width + x];
    }
}

// Naive approach : Coalesced access but strided write to global memory
__global__ void transposeNaive(float *odata, const float *idata){
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    int width = gridDim.x * TILE_DIM;

    for(int k = 0; k < TILE_DIM; k += BLOCK_ROWS){
        odata[(x)*width + y + k] = idata[(y+k)*width + x];
    }
}

// Coalesced approach with tiling
__global__ void transposeCoalesced(float *odata, const float *idata){
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    int width = gridDim.x * TILE_DIM;

    __shared__ float tile[TILE_DIM][TILE_DIM];
    
    // Loading tile of data into shared memory
    for(int k = 0;k < TILE_DIM; k += BLOCK_ROWS){
        tile[threadIdx.y + k][threadIdx.x] = idata[(y + k) * width + x];
    }
    __syncthreads();

    // Recomputing the indices, we will swap blockIndices and not the threadIndices
    // Swapping block indices is sufficient to swap the data, thread indices just gives us relative position within the tile
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for(int k = 0; k < TILE_DIM; k += BLOCK_ROWS){
        // Tile indices are flipped since we need to flip both tile's position and its content
        odata[(y + k) * width + x] = tile[threadIdx.x][threadIdx.y + k];
        // This access in shared memory leads to bank conflict
    }
}

// Shared memory Bank conflicts resolve . Bank conflicts: multiple threads trying to access same memory bank
// Adding an extra offset 1 to the rows
__global__ void transposeNoBankConflicts(float *odata, const float *idata){
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int width = gridDim.x * TILE_DIM;

    for(int k = 0; k < TILE_DIM; k += BLOCK_ROWS){
        tile[threadIdx.y + k][threadIdx.x] = idata[(y + k) * width + x];
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for(int k = 0; k < TILE_DIM; k += BLOCK_ROWS){
        odata[(y + k) * width + x] = tile[threadIdx.x][threadIdx.y + k];
    }
}

int main(int argc, char **argv)
{
    const int nx = 128;
    const int ny = 128;
    const int mem_size = nx * ny * sizeof(float);

    dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    int devId = 0;
    if (argc > 1) devId = atoi(argv[1]);

    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, devId));
    printf("\nDevice : %s\n", prop.name);
    printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", 
           nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
    printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
           dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
  
    checkCuda(cudaSetDevice(devId));

    float *h_idata = (float*)malloc(mem_size);
    float *h_cdata = (float*)malloc(mem_size);
    float *h_tdata = (float*)malloc(mem_size);
    float *gold    = (float*)malloc(mem_size);
  
    float *d_idata, *d_cdata, *d_tdata;
    checkCuda(cudaMalloc(&d_idata, mem_size));
    checkCuda(cudaMalloc(&d_cdata, mem_size));
    checkCuda(cudaMalloc(&d_tdata, mem_size));

    if (nx % TILE_DIM | ny % TILE_DIM) {
        printf("nx and ny must be a multiple of TILE_DIM\n");
        goto error_exit;
    }

    if (TILE_DIM % BLOCK_ROWS) {
        printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
        goto error_exit;
    }

    // host
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
            h_idata[j * nx + i] = j * nx + i;

    // correct result for error checking
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
            gold[j * nx + i] = h_idata[i * nx + j];

    // device
    checkCuda(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

    cudaEvent_t startEvent, stopEvent;
    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));
    float ms;

    printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");

    // ---- copy ----
    printf("%25s", "copy");
    checkCuda(cudaMemset(d_cdata, 0, mem_size));
    copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; i++)
        copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
    postprocess(h_idata, h_cdata, nx * ny, ms);

    // ---- shared memory copy ----
    printf("%25s", "shared memory copy");
    checkCuda(cudaMemset(d_cdata, 0, mem_size));
    copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; i++)
        copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
    postprocess(h_idata, h_cdata, nx * ny, ms);

    // ---- naive transpose ----
    printf("%25s", "naive transpose");
    checkCuda(cudaMemset(d_tdata, 0, mem_size));
    transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; i++)
        transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
    postprocess(gold, h_tdata, nx * ny, ms);

    // ---- coalesced transpose ----
    printf("%25s", "coalesced transpose");
    checkCuda(cudaMemset(d_tdata, 0, mem_size));
    transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; i++)
        transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
    postprocess(gold, h_tdata, nx * ny, ms);

    // ---- bank-conflict-free transpose ----
    printf("%25s", "conflict-free transpose");
    checkCuda(cudaMemset(d_tdata, 0, mem_size));
    transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; i++)
        transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
    postprocess(gold, h_tdata, nx * ny, ms);

error_exit:
    checkCuda(cudaEventDestroy(startEvent));
    checkCuda(cudaEventDestroy(stopEvent));
    checkCuda(cudaFree(d_tdata));
    checkCuda(cudaFree(d_cdata));
    checkCuda(cudaFree(d_idata));
    free(h_idata);
    free(h_tdata);
    free(h_cdata);
    free(gold);
}
