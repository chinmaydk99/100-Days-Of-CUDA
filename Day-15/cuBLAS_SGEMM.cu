#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void matrix_multiplication(){
    cublasHandle_t handle;
    cublasCreate(&handle);

    const int m = 2;
    const int k = 3;
    const int n = 2;

    float h_A[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    float h_B[] = {
        5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f
    };

    float h_C[] = {
        0.0f, 0.0f,0.0f,
        0.0f, 0.0f, 0.0f
    };

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m*k*sizeof(float));
    cudaMalloc(&d_B, k*n*sizeof(float));
    cudaMalloc(&d_C, m*n*sizeof(float));

    cudaMemcpy(d_A, h_A, m*k*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k*n*sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0.0f;


    cublasSgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_T,  // Transpose both A and B so that the formatting matches
        m, n, k,  // m x k (A) * k x n (B) → m x n (C)
        &alpha, d_A, k,  // lda = k (columns in row-major), would if number of rows if not transposed
        d_B, n,  // ldb = n (columns in row-major), same as above
        &beta, d_C, m   // ldc = m (columns in row-major)
    );
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Output result
    std::cout << "Result matrix C (A * B):" << std::endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << h_C[i + j * m] << " "; 
        }
        std::cout << std::endl;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}

int main() {
    matrix_matrix_multiplication_example();
    return 0;
}

