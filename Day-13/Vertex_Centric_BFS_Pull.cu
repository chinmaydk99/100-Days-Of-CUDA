#include <iostream>
#include <cuda_runtime.h>
#include <climits>

struct CSCGraph{
    int numVertices;
    int numEdges;
    int *dstPtrs;
    int *src;
};

__global__ void bfs_pull(CSCGraph cscGraph, unsigned int *level, unsigned int *newVertexVisited, unsigned int currLevel){
    unsigned int vertex = blockDim.x*blockIdx.x + threadIdx.x;

    if(vertex < cscGraph.numVertices){
        if(level[vertex] == UINT_MAX){// If current node is univisited
            for(unsigned edge = cscGraph.dstPtrs[vertex]; edge < cscGraph.dstPtrs[vertex+1]; edge ++){
                unsigned int neighbor = cscGraph.src[edge];
                if(level[neighbor] == currLevel-1){
                    level[vertex] = currLevel;
                    *newVertexVisited = 1;
                    break;
                }
            }
        }
    }
}


int main() {
    const int numVertices = 4;
    const int numEdges = 4;

    int h_dstPtrs[] = {0, 1, 2, 3, 4};
    int h_src[] = {0, 0, 1, 2};

    unsigned int h_level[numVertices];
    std::fill_n(h_level, numVertices, UINT_MAX);
    h_level[0] = 0;

    CSCGraph h_cscGraph;
    h_cscGraph.numEdges = numEdges;
    h_cscGraph.numVertices = numVertices;

    cudaMalloc(&h_cscGraph.src, numEdges*sizeof(int));
    cudaMalloc(&h_cscGraph.dstPtrs, (numVertices+1)*sizeof(int));

    cudaMemcpy(h_cscGraph.dstPtrs, h_dstPtrs, (numVertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(h_cscGraph.src, h_src, numEdges * sizeof(int), cudaMemcpyHostToDevice);

    CSCGraph *d_cscGraph;
    cudaMalloc(&d_cscGraph, sizeof(CSCGraph));
    cudaMemcpy(d_cscGraph, &h_cscGraph, sizeof(CSCGraph), cudaMemcpyHostToDevice);

    unsigned int *d_level, *d_newVertexVisited;
    cudaMalloc(&d_level, numVertices * sizeof(unsigned int));
    cudaMemcpy(d_level, h_level, numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_newVertexVisited, sizeof(unsigned int));

    unsigned int currLevel = 1;
    unsigned int newVertexVisited = 0;

    do {
        newVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &newVertexVisited, sizeof(unsigned int), cudaMemcpyHostToDevice);

        bfs_pull<<<(numVertices + 255) / 256, 256>>>(*d_cscGraph, d_level, d_newVertexVisited, currLevel);
        cudaMemcpy(&newVertexVisited, d_newVertexVisited, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        currLevel++;

    } while (newVertexVisited != 0);

    cudaMemcpy(h_level, d_level, numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    std::cout << "Vertex Levels (CSC):\n";
    for (int i = 0; i < numVertices; i++) {
        std::cout << "Vertex " << i << ": Level " << h_level[i] << std::endl;
    }

    cudaFree(d_cscGraph->dstPtrs);
    cudaFree(d_cscGraph->src);
    cudaFree(d_level);
    cudaFree(d_newVertexVisited);

    return 0;
}

