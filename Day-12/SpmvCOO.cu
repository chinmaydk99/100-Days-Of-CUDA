#include <iostream>
#include <cuda_runtime.h>

__global__ void spmv_coo(int nnz, float *row, float *col, float *values, float *x, float *y){
    unsigned int index = blockDim.x*blockIdx.x + threadIdx.x;

    if(index < nnz){
        int r = row[index];
        int c = col[index];
        float value = values[idx];

        atomicAdd(&y[r], x[c]*value);
    }
}

void spmv_coo(int nnz, int rows, const int* h_row, const int* h_col, const float* h_values, const float* h_x, float* h_y) {
    int *d_row, *d_col;
    float *d_values, *d_x, *d_y;

    cudaMalloc((void**)&d_row, nnz * sizeof(int));
    cudaMalloc((void**)&d_col, nnz * sizeof(int));
    cudaMalloc((void**)&d_values, nnz * sizeof(float));
    cudaMalloc((void**)&d_x, rows * sizeof(float));
    cudaMalloc((void**)&d_y, rows * sizeof(float));

    cudaMemcpy(d_row, h_row, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, h_col, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, rows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, rows * sizeof(float));  // Initialize y to 0

    int blockSize = 256;
    int gridSize = (nnz + blockSize - 1) / blockSize;

    spmv_coo_kernel<<<gridSize, blockSize>>>(nnz, d_row, d_col, d_values, d_x, d_y);

    cudaMemcpy(h_y, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    int nnz = 3;  // Number of non-zero elements
    int rows = 3;

    int h_row[] = {0, 1, 2}; 
    int h_col[] = {1, 2, 2}; 
    float h_values[] = {5.0, 8.0, 3.0}; 
    float h_x[] = {1.0, 2.0, 3.0};
    float h_y[3] = {0}; 

    spmv_coo(nnz, rows, h_row, h_col, h_values, h_x, h_y);

    std::cout << "Result y = A * x:" << std::endl;
    for (int i = 0; i < rows; i++) {
        std::cout << h_y[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
