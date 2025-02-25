#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void matrix_multiplication(){
    cublasHandle_t handle;
    cublasCreate(&handle);

    const int m = 3;
    const int n = 3;

    double h_A[] = {1.0, 2.0, 3.0,
                   4.0, 5.0, 6.0,
                   7.0, 8.0, 9.0};
    
    double h_x[] = {1.0, 2.0, 3.0};
    double h_y[] = {0.0, 0.0, 0.0};

    float *d_A, *d_x, *d_y;
    double alpha = 1.0;
    double beta = 0.0;

    cudaMalloc(&d_A, m*n*sizeof(double));
    cudaMalloc(&d_x, n*sizeof(double));
    cudaMalloc(&d_y, m*sizeof(double));

    cudaMemcpy(d_A, h_A, m*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, n*sizeof(double), cudaMemcpyHostToDevice);

    cublasDgemv(handle, CUBLAS_OP_T, m , n, &alpha , d_A, m, d_x, 1, &beta, d_y, 1); // The input array has been transposed since cuBLAS follows column majpr order and I have written the matrix in row major order

    cudaMemcpy(h_y, d_y, m*sizeof(double), cudaMemcpyDeviceToHost);
    
    std::cout << "Result (y = Ax): ";
    for (int i = 0; i < m; i++) {
        std::cout << h_y[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(handle);
}