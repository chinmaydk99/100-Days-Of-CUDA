#include <iostream>
#include <cuda_runtime.h>
#include <climits>

struct CSRGraph{
    int numVertices;
    int numEdges;
    int *srcPtrs;
    int *dst;
};



__global__ void csr_bfs(CSRGraph csrgraph, unsigned int *level, unsigned int *newVertexVisited, unsigned int currLevel){
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x; // Get Vertex. Each thread is assigned to an index

    if(vertex < csrgraph.numVertices){ // Check if vertex is valid
        if(level[vertex] == currLevel-1){ // Only vertices which are a level below are activated
            for(unsigned int edge = csrgraph.srcPtrs[vertex]; edge < csrgraph.srcPtrs[vertex+1]; edge++){
                unsigned int neighbor = csrgraph.dst[edge];
                if(level[neighbor] == UINT_MAX){ // Neighbor is unvisited
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

    CSRGraph d_csrGraph;
    cudaMalloc(&d_csrGraph.srcPtrs, (numVertices + 1) * sizeof(int));
    cudaMalloc(&d_csrGraph.dst, numEdges * sizeof(int));
    cudaMemcpy(d_csrGraph.srcPtrs, h_srcPtrs, (numVertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrGraph.dst, h_dst, numEdges * sizeof(int), cudaMemcpyHostToDevice);

    unsigned int *d_level, *d_newVertexVisited;
    cudaMalloc(&d_level, numVertices * sizeof(unsigned int));
    cudaMemcpy(d_level, h_level, numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_newVertexVisited, sizeof(unsigned int));

    unsigned int currLevel = 1;
    bool finished = false;
    while (!finished) {
        finished = true;
        cudaMemcpy(d_newVertexVisited, &finished, sizeof(unsigned int), cudaMemcpyHostToDevice);

        csr_bfs<<<(numVertices + 255) / 256, 256>>>(d_csrGraph, d_level, d_newVertexVisited, currLevel);

        cudaMemcpy(&finished, d_newVertexVisited, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        currLevel++;
    }

    cudaMemcpy(h_level, d_level, numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost);


    std::cout << "Vertex Levels:\n";
    for (int i = 0; i < numVertices; i++) {
        std::cout << "Vertex " << i << ": Level " << h_level[i] << std::endl;
    }

    cudaFree(d_csrGraph.srcPtrs);
    cudaFree(d_csrGraph.dst);
    cudaFree(d_level);
    cudaFree(d_newVertexVisited);

    return 0;
}