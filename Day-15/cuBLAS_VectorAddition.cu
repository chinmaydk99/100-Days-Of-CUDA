#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(){
     // Creating the cublas handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    int n =  4;
    double h_x[] = {1.0, 2.0, 3.0, 4.0};
    double h_y[] = {5.0, 6.0, 7.0, 8.0};

    printf("Inputs: ");
    printf("x:");
    for (int i = 0; i < n; i++) printf("%.1f ", h_x[i]); 
    printf("\n");

    printf("y:");
    for (int i = 0; i < n; i++) printf("%.1f ", h_y[i]); 
    printf("\n");

    double alpha = 1.0;

    double *d_x, *d_y;
    cudaMalloc(&d_x, n*sizeof(double));
    cudaMalloc(&d_y, n*sizeof(double));
    
    cudaMemcpy(d_x, h_x, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n*sizeof(double), cudaMemcpyHostToDevice);

    // D here is for double precision
    // This is doing alpha *x + y and result is stored in y
    cublasDaxpy(handle, n, &alpha, d_x, 1, d_y, 1); // Here the 1 is stride between elements

    cudaMemcpy(h_y, d_y, n*sizeof(double), cudaMemcpyDeviceToHost);

    printf("y: ");
    for (int i = 0; i < n; i++) printf("%.1f ", h_y[i]); // [6.0, 8.0, 10.0, 12.0]
    printf("\n");

    cudaFree(d_x); 
    cudaFree(d_y);
    cublasDestroy(handle);

    return 0;
}
   
