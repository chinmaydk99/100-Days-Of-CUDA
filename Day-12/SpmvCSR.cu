#include <iostream>
#include <cuda_runtime.h>

__global__ void spmv_csr(float *rowPtr, float *col, float * values, float *x, float *y, int numrows){
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < numrows){
        float sum = 0.0f;

        for(int i = rowPtr[row]; i < rowPtr[row+1]; i++){
            sum += values[i] * x[col[i]]
        }

        y[row] = sum;
    }
}

void spmv_csr(int* h_row_ptr, int* h_col_idx, float* h_values, float* h_x, float* h_y, int num_rows, int nnz) {
    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;

    cudaMalloc((void**)&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_col_idx, nnz * sizeof(int));
    cudaMalloc((void**)&d_values, nnz * sizeof(float));
    cudaMalloc((void**)&d_x, num_rows * sizeof(float));
    cudaMalloc((void**)&d_y, num_rows * sizeof(float));

    cudaMemcpy(d_row_ptr, h_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, num_rows * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (num_rows + blockSize - 1) / blockSize;
    spmv_csr_kernel<<<gridSize, blockSize>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);

    cudaMemcpy(h_y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    int num_rows = 3;
    int nnz = 4;

    int h_row_ptr[] = {0, 1, 3, 4};
    int h_col_idx[] = {1, 1, 2, 2};
    float h_values[] = {5.0, 2.0, 4.0, 3.0};

 
    float h_x[] = {1.0, 2.0, 3.0};

    float h_y[3] = {0.0f, 0.0f, 0.0f}; 

    spmv_csr(h_row_ptr, h_col_idx, h_values, h_x, h_y, num_rows, nnz);

    std::cout << "Result y = A * x:\n";
    for (int i = 0; i < num_rows; i++) {
        std::cout << h_y[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
