#include <iostream>
#include <cuda_runtime.h>
#include <climits>

struct COOGraph{
    int numVertices;
    int numEdges;
    int *dst;
    int *src;
};

__global__ void edge_bfs(COOGraph coograph, unsigned int *level, unsigned int currLevel, unsigned int *newVertexVisited){
    unsigned int edge = blockDim.x*blockIdx.x + threadIdx.x;

    if(edge < cograph.numEdges){
        unsigned int vertex = coograph.src[edge]; // Source index of the edge
        if(level[vertex] == currLevel-1){
            unsigned int neighbor = coograph.src[vertex];
            if(level[neighbor] == UINT_MAX){
                level[neighbor] = level;
                *newVertexVisited = 1;
            }
        }
    }
}

int main() {
    const int numVertices = 4;
    const int numEdges = 4;

    int h_src[] = {0, 0, 1, 2};
    int h_dst[] = {1, 2, 3, 0};

    unsigned int h_level[numVertices];
    std::fill_n(h_level, numVertices, UINT_MAX);
    h_level[0] = 0;

    COOGraph d_cooGraph;
    cudaMalloc(&d_cooGraph.src, numEdges * sizeof(int));
    cudaMalloc(&d_cooGraph.dst, numEdges * sizeof(int));
    cudaMemcpy(d_cooGraph.src, h_src, numEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cooGraph.dst, h_dst, numEdges * sizeof(int), cudaMemcpyHostToDevice);

    unsigned int *d_level, *d_newVertexVisited;
    cudaMalloc(&d_level, numVertices * sizeof(unsigned int));
    cudaMemcpy(d_level, h_level, numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_newVertexVisited, sizeof(unsigned int));

    unsigned int currLevel = 1;
    bool finished = false;
    while (!finished) {
        finished = true;
        cudaMemcpy(d_newVertexVisited, &finished, sizeof(unsigned int), cudaMemcpyHostToDevice);

        edge_bfs<<<(numEdges + 255) / 256, 256>>>(d_cooGraph, d_level, currLevel, d_newVertexVisited);

        cudaMemcpy(&finished, d_newVertexVisited, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        currLevel++;
    }

    cudaMemcpy(h_level, d_level, numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    std::cout << "Vertex Levels (COO):\n";
    for (int i = 0; i < numVertices; i++) {
        std::cout << "Vertex " << i << ": Level " << h_level[i] << std::endl;
    }

    cudaFree(d_cooGraph.src);
    cudaFree(d_cooGraph.dst);
    cudaFree(d_level);
    cudaFree(d_newVertexVisited);

    return 0;
}