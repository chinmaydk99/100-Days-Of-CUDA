#include <iostream>
#include <cuda_runtime.h>
#include <climits>


#define LOCAL_FRONTIER_CAPACITY 1024 

// CSR Graph structure
struct CSRGraph {
    int numVertices;
    int numEdges;
    int *srcPtrs; 
    int *dst; 
};


__global__ void bfs_kernel(CSRGraph csrGraph, unsigned int* level, unsigned int* prevFrontier,
                           unsigned int* currFrontier, unsigned int numPrevFrontier, 
                           unsigned int* numCurrFrontier, unsigned int currLevel) {

    // Shared memory for block-private frontier
    __shared__ unsigned int currFrontier_s[LOCAL_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    
    // Initialize shared variables
    if (threadIdx.x == 0) {
        numCurrFrontier_s = 0;
    }
    __syncthreads();

    // Each thread processes a vertex from the previous frontier
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];

            // Atomically mark neighbor as visited if unvisited
            if (atomicCAS(&level[neighbor], UINT_MAX, currLevel) == UINT_MAX) {
                // Add the neighbor to the block's private frontier. This is now within shared memory
                unsigned int currFrontierIdx = atomicAdd(&numCurrFrontier_s, 1);
                if (currFrontierIdx < LOCAL_FRONTIER_CAPACITY) {
                    currFrontier_s[currFrontierIdx] = neighbor;
                }
            }
        }
    }

    __syncthreads();

    // Update in global memory once all threads in the block have been processed
    __shared__ unsigned int currFrontierStartIdx;
    if (threadIdx.x == 0) {
        currFrontierStartIdx = atomicAdd(numCurrFrontier, numCurrFrontier_s);
    }
    __syncthreads();

    // Threads write the local frontier to the global frontier
    for (unsigned int k = threadIdx.x; k < numCurrFrontier_s; k += blockDim.x) {
        unsigned int currFrontierIdx_g = currFrontierStartIdx + k;
        currFrontier[currFrontierIdx_g] = currFrontier_s[k];
    }
}

int main() {
    const int numVertices = 4;
    const int numEdges = 4;

    int h_srcPtrs[] = {0, 2, 3, 4, 4};
    int h_dst[] = {1, 2, 3, 0};

    unsigned int h_level[numVertices];
    std::fill_n(h_level, numVertices, UINT_MAX);
    h_level[0] = 0; 

    CSRGraph h_csrGraph;
    h_csrGraph.numVertices = numVertices;
    h_csrGraph.numEdges = numEdges;

    cudaMalloc(&h_csrGraph.srcPtrs, (numVertices + 1) * sizeof(int));
    cudaMalloc(&h_csrGraph.dst, numEdges * sizeof(int));

    cudaMemcpy(h_csrGraph.srcPtrs, h_srcPtrs, (numVertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(h_csrGraph.dst, h_dst, numEdges * sizeof(int), cudaMemcpyHostToDevice);

    CSRGraph* d_csrGraph;
    cudaMalloc(&d_csrGraph, sizeof(CSRGraph));
    cudaMemcpy(d_csrGraph, &h_csrGraph, sizeof(CSRGraph), cudaMemcpyHostToDevice);

    unsigned int* d_level, *d_prevFrontier, *d_currFrontier, *d_numCurrFrontier;
    cudaMalloc(&d_level, numVertices * sizeof(unsigned int));
    cudaMemcpy(d_level, h_level, numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_prevFrontier, numVertices * sizeof(unsigned int));
    cudaMalloc(&d_currFrontier, numVertices * sizeof(unsigned int));
    cudaMalloc(&d_numCurrFrontier, sizeof(unsigned int));

    unsigned int h_prevFrontier[numVertices] = {0}; 
    unsigned int h_numCurrFrontier = 1; 
    cudaMemcpy(d_prevFrontier, h_prevFrontier, sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int currLevel = 1;
    unsigned int numPrevFrontier = 1;

    while (numPrevFrontier > 0) {
        h_numCurrFrontier = 0;
        cudaMemcpy(d_numCurrFrontier, &h_numCurrFrontier, sizeof(unsigned int), cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (numPrevFrontier + threadsPerBlock - 1) / threadsPerBlock;

        bfs_kernel<<<blocksPerGrid, threadsPerBlock>>>(*d_csrGraph, d_level, d_prevFrontier, d_currFrontier, 
                                                        numPrevFrontier, d_numCurrFrontier, currLevel);

        cudaMemcpy(&h_numCurrFrontier, d_numCurrFrontier, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        std::swap(d_prevFrontier, d_currFrontier);
        numPrevFrontier = h_numCurrFrontier;
        currLevel++;
    }

    cudaMemcpy(h_level, d_level, numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    std::cout << "Vertex Levels (BFS):\n";
    for (int i = 0; i < numVertices; i++) {
        std::cout << "Vertex " << i << ": Level " << h_level[i] << std::endl;
    }

    cudaFree(h_csrGraph.srcPtrs);
    cudaFree(h_csrGraph.dst);
    cudaFree(d_level);
    cudaFree(d_prevFrontier);
    cudaFree(d_currFrontier);
    cudaFree(d_numCurrFrontier);
    cudaFree(d_csrGraph);

    return 0;
}
