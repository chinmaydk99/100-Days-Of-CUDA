#include <cuda_runtime.h>
#include <iostream>

struct ELLMatrix {
    int numRows;
    int *colIdx;
    int maxNnzPerRow;
    float *values;
};

__global__ void spmv_ell_kernel(ELLMatrix ellmatrix, float *x, float *y){
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < ellmatrix.numRows){
        float sum = 0.0f;

        for(int i = 0 ; i < ellmatrix.maxNnzPerRow; i ++){
            int idx = i*ellmatrix.numRows + row; // Column major indexing
            int col = ellmatrix.colIdx[idx];
            float value = ellmatrix.values[idx];

            if(col >= 0){// Skipping padding values
                sum += value * x[col];
            }
        }
        y[row] = sum;
    }
}


void spmv_ell_host(ELLMatrix h_ellMatrix, float* h_x, float* h_y) {
    int *d_colIdx;
    float *d_values, *d_x, *d_y;

    cudaMalloc((void**)&d_colIdx, h_ellMatrix.numRows * h_ellMatrix.maxNnzPerRow * sizeof(int));
    cudaMalloc((void**)&d_values, h_ellMatrix.numRows * h_ellMatrix.maxNnzPerRow * sizeof(float));
    cudaMalloc((void**)&d_x, h_ellMatrix.numRows * sizeof(float));
    cudaMalloc((void**)&d_y, h_ellMatrix.numRows * sizeof(float));

    cudaMemcpy(d_colIdx, h_ellMatrix.colIdx, h_ellMatrix.numRows * h_ellMatrix.maxNnzPerRow * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_ellMatrix.values, h_ellMatrix.numRows * h_ellMatrix.maxNnzPerRow * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, h_ellMatrix.numRows * sizeof(float), cudaMemcpyHostToDevice);

    ELLMatrix d_ellMatrix = h_ellMatrix;
    d_ellMatrix.colIdx = d_colIdx;
    d_ellMatrix.values = d_values;

    int blockSize = 256;
    int gridSize = (h_ellMatrix.numRows + blockSize - 1) / blockSize;
    spmv_ell_kernel<<<gridSize, blockSize>>>(d_ellMatrix, d_x, d_y);

    cudaMemcpy(h_y, d_y, h_ellMatrix.numRows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}


int main() {
    int numRows = 3;
    int maxNnzPerRow = 2; 

    // -1 are the values used for padding
    int h_colIdx[] = {
        1, -1,  
        1, 2,  
        2, -1   
    };

    float h_values[] = {
        5.0, 0.0,  // Row 0 (with padding)
        2.0, 4.0,  // Row 1
        3.0, 0.0   // Row 2 (with padding)
    };

    float h_x[] = {1.0, 2.0, 3.0};  // Input vector
    float h_y[3] = {0.0f, 0.0f, 0.0f};  // Output vector initialized to zero

    ELLMatrix h_ellMatrix;
    h_ellMatrix.numRows = numRows;
    h_ellMatrix.maxNnzPerRow = maxNnzPerRow;
    h_ellMatrix.colIdx = h_colIdx;
    h_ellMatrix.values = h_values;

    spmv_ell_host(h_ellMatrix, h_x, h_y);

    std::cout << "Result y = A * x:\n";
    for (int i = 0; i < numRows; i++) {
        std::cout << h_y[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
