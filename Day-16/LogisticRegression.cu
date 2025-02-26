#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;

__global__ void sigmoid_kernel(float *z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = 1.0f / (1.0f + expf(-z[i]));
    }
}

void logistic_regression_cublas(float *d_X, float *d_y, float *d_w, 
                                int n, int d, float learning_rate, int epochs) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;
    float neg_learning_rate = -learning_rate;

    float *d_pred, *d_grad;
    cudaMalloc(&d_pred, n * sizeof(float));  // Stores predictions
    cudaMalloc(&d_grad, d * sizeof(float));  // Stores gradient

    for (int epoch = 0; epoch < epochs; epoch++) {
        cublasSgemv(handle, CUBLAS_OP_T, n, d, &alpha, d_X, d, d_w, 1, &beta, d_pred, 1);

        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        sigmoid_kernel<<<numBlocks, blockSize>>>(d_pred, n);

        float minus_one = -1.0f;
        cublasSaxpy(handle, n, &minus_one, d_y, 1, d_pred, 1);

        cublasSgemv(handle, CUBLAS_OP_N, d, n, &alpha, d_X, n, d_pred, 1, &beta, d_grad, 1);

        cublasSaxpy(handle, d, &neg_learning_rate, d_grad, 1, d_w, 1);

        if (epoch % 100 == 0) {
            float loss = 0.0f;
            cublasSnrm2(handle, n, d_pred, 1, &loss);
            cout << "Epoch: " << epoch << " Loss: " << (loss / n) << endl;
        }
    }

    cudaFree(d_pred);
    cudaFree(d_grad);
    cublasDestroy(handle);
}


int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const int n = 3;
    const int d = 2;
    const float learning_rate = 0.1f;
    const int epochs = 1000;

    float h_X[] = {1.0f, 2.0f,
                   3.0f, 4.0f,
                   5.0f, 6.0f};

    float h_y[] = {0.0f, 1.0f, 1.0f};


    float h_w[d] = {0.0f, 0.0f};

    float *d_X, *d_y, *d_w;
    cudaMalloc(&d_X, n * d * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_w, d * sizeof(float));

    cudaMemcpy(d_X, h_X, n * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_w, 0, d * sizeof(float));  // Initialize weights to zero

    logistic_regression_cublas(d_X, d_y, d_w, n, d, learning_rate, epochs);

    cudaMemcpy(h_w, d_w, d * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Final trained weights: ";
    for (int i = 0; i < d; i++) {
        cout << h_w[i] << " ";
    }
    cout << endl;


    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_w);
    cublasDestroy(handle);

    return 0;
}
