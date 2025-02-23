#include <iostream>
#include <cuda_runtime.h>
#include <climits>

struct CSRGraph {
    int numVertices;
    int numEdges;
    int *srcPtrs;
    int *dst;
};

__global__ void csr_bfs(CSRGraph csrgraph, unsigned int *level, unsigned int *newVertexVisited, unsigned int currLevel) {
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;  // Each thread handles a vertex

    if (vertex < csrgraph.numVertices) {
        if (level[vertex] == currLevel - 1) {  // Process only vertices from the previous level
            for (unsigned int edge = csrgraph.srcPtrs[vertex]; edge < csrgraph.srcPtrs[vertex + 1]; edge++) {
                unsigned int neighbor = csrgraph.dst[edge];
                if (level[neighbor] == UINT_MAX) {  // Mark unvisited neighbor
                    level[neighbor] = currLevel;
                    *newVertexVisited = 1;
                }
            }
        }
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

    CSRGraph *d_csrGraph;
    cudaMalloc(&d_csrGraph, sizeof(CSRGraph));
    cudaMemcpy(d_csrGraph, &h_csrGraph, sizeof(CSRGraph), cudaMemcpyHostToDevice);

    unsigned int *d_level, *d_newVertexVisited;
    cudaMalloc(&d_level, numVertices * sizeof(unsigned int));
    cudaMemcpy(d_level, h_level, numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_newVertexVisited, sizeof(unsigned int));

    unsigned int currLevel = 1;
    unsigned int newVertexVisited = 0;

    do {
        newVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &newVertexVisited, sizeof(unsigned int), cudaMemcpyHostToDevice);

        csr_bfs<<<(numVertices + 255) / 256, 256>>>(*d_csrGraph, d_level, d_newVertexVisited, currLevel);

        cudaMemcpy(&newVertexVisited, d_newVertexVisited, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        currLevel++;

    } while (newVertexVisited != 0);  

    cudaMemcpy(h_level, d_level, numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    std::cout << "Vertex Levels (CSR):\n";
    for (int i = 0; i < numVertices; i++) {
        std::cout << "Vertex " << i << ": Level " << h_level[i] << std::endl;
    }

    cudaFree(d_csrGraph->srcPtrs);
    cudaFree(d_csrGraph->dst);
    cudaFree(d_level);
    cudaFree(d_newVertexVisited);

    return 0;
}
