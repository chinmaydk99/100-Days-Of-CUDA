#include <iostream>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS (((((((value to be added)))))))

__global__ void copy(float *odata, const float *idata){
    int x = blockIdx.x * TILED_DIM + threadIdx.x; // Columns
    int y = blockIdx.y * TILED_DIM + threadIdx.y; // Rows
    
    int width = gridDim.x * TILE_DIM; // Total cols = number of tiles multiplied by size of each tile

    for(int k = 0; k < TILE_DIM; k += BLOCK_ROWS){
        odata[(y+k) * width + x] = idata[(y+k) * width + x];
    }
}

// Naive approach : Coalesced access but strided write to global memory
__global__ void transpose_naive(float *odata, const float idata){
    int x = blockIdx.x * TILED_DIM + threadIdx.x;
    int y = blockIdx.y * TILED_DIM + threadIdx.y;

    int width = gridDim.x * TILE_DIM;

    for(int k = 0; k < TILE_DIM; k += BLOCK_ROWS){
        odata[(x)*width + y + k] = idata[(y+k)*width + x];
    }
}

// Coalesced approach with tiling
__global__ void transpose_coalesced(float *odata, const float idata){
    int x = blockDim.x * TILE_DIM + threadIdx.x;
    int y = blockDim.y * TILE_DIM + threadIdx.y;

    int width = gridDim.x * TILE_DIM;

    __shared__ float tile[TILE_DIM][TILE_DIM];
    
    // Loading tile of data into shared memory
    for(int k = 0;k < TILE_DIM; k += BLOCK_ROWS){
        tile[threadIdx.y + k][threadIdx.x] = idata[(y+k)*width + x];
    }
    __syncthreads();

    // Recomputing the indices, we will swap blockIndices and not the threadIndices
    // Swapping block indices is sufficient to swap the data, thread indices just gives us relative position within the tile
    int x = blockDim.y * TILE_DIM + threadIdx.x;
    int y = blockDim.x * TILE_DIM + threadIdx.y;

    for(int k = 0; k < TILE_DIM; k ++ BLOCK_ROWS){
        // Tile indices are flipped since we need to flip both tile's position and its content
        odata[(y+k)*width + x] = tile[threadIdx.x][threadIdx.y + j];
        // This access in shared memory leads to bank conflict
    }
}

// Shared memory Bank conflicts resolve . Bank conflicts: multiple threads trying to access same memory bank
// Adding an extra offset 1 to the rows
__global__ void transpose_no_bank_conflict(float *odata, const float *idata){
    int x = blockDim.x * TILE_DIM + threadIdx.x;
    int y = blockDim.y * TILE_DIM + threadIdx.y;

    __shared__ float tile[TILE_DIM][TILE_DIM+1];
    int width = gridDim.x * TILE_DIM;

    for(int k = 0; k < TILE_DIM; k += BLOCK_ROWS){
        tile[threadIdx.y + k][threadIdx.x] = idata[(y+j)*width + x];
    }

    __syncthreads();

    int x = blockDim.y * TILE_DIM + threadIdx.x;
    int y = blockDim.x * TILE_DIM + threadIdx.y;

    for(int k = 0; k < TILE_DIM; k += BLOCK_ROWS){
        odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y];
    }

}
