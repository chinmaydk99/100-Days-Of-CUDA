#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

float compute_loss(cublasHandle_t handle, float* d_X, float* d_y, float* d_w, int n, int d, float lambda) {
    float alpha = 1.0f;
    float beta = 0.0f;
    float loss = 0.0f;

    float* d_pred;
    cudaMalloc((void**)&d_pred, n * sizeof(float));


    cublasSgemv(handle, CUBLAS_OP_N, n, d, &alpha, d_X, n, d_w, 1, &beta, d_pred, 1);

    cublasSaxpy(handle, n, &alpha, d_y, 1, d_pred, 1);  // d_pred = d_pred - y

    cublasSnrm2(handle, n, d_pred, 1, &loss);
    loss = (loss * loss) / n;

    float w_norm = 0.0f;
    cublasSnrm2(handle, d, d_w, 1, &w_norm);
    loss += lambda * (w_norm * w_norm);

    cudaFree(d_pred);
    return loss;
}


void ridge_regression_gradient_descent(
    cublasHandle_t handle, float* d_X, float* d_y, float* d_w,
    int n, int d, float lambda, float learning_rate, int epochs
) {
    float alpha = 1.0f;
    float beta = 0.0f;
    float neg_learning_rate = -learning_rate;

    float* d_pred;
    float* d_grad;
    cudaMalloc((void**)&d_pred, n * sizeof(float));
    cudaMalloc((void**)&d_grad, d * sizeof(float));

    for (int epoch = 0; epoch < epochs; epoch++) {
        cublasSgemv(handle, CUBLAS_OP_N, n, d, &alpha, d_X, n, d_w, 1, &beta, d_pred, 1);


        float minus_one = -1.0f;
        cublasSaxpy(handle, n, &minus_one, d_y, 1, d_pred, 1);  // d_pred = Xw - y

        cublasSgemv(handle, CUBLAS_OP_T, n, d, &alpha, d_X, n, d_pred, 1, &beta, d_grad, 1);

        float reg_term = 2.0f * lambda;
        cublasSaxpy(handle, d, &reg_term, d_w, 1, d_grad, 1);  // d_grad += 2 * lambda * w

        cublasSaxpy(handle, d, &neg_learning_rate, d_grad, 1, d_w, 1);

        if (epoch % 100 == 0) {
            float loss = compute_loss(handle, d_X, d_y, d_w, n, d, lambda);
            std::cout << "Epoch " << epoch << " - Loss: " << loss << std::endl;
        }
    }

    cudaFree(d_pred);
    cudaFree(d_grad);
}

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const int n = 1000; 
    const int d = 3;
    const float lambda = 0.1f;
    const float learning_rate = 0.01f;
    const int epochs = 1000;

    float* h_X = new float[n * d];
    float* h_y = new float[n]; 
    float* h_w = new float[d];  


    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            h_X[i * d + j] = static_cast<float>(rand()) / RAND_MAX;
        }
        h_y[i] = static_cast<float>(rand()) / RAND_MAX;
    }


    float *d_X, *d_y, *d_w;
    cudaMalloc((void**)&d_X, n * d * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_w, d * sizeof(float));


    cudaMemcpy(d_X, h_X, n * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_w, 0, d * sizeof(float));  // Initialize weights to 0

   
    ridge_regression_gradient_descent(handle, d_X, d_y, d_w, n, d, lambda, learning_rate, epochs);

    cudaMemcpy(h_w, d_w, d * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Final trained weights (w): ";
    for (int i = 0; i < d; i++) {
        std::cout << h_w[i] << " ";
    }
    std::cout << std::endl;

    delete[] h_X;
    delete[] h_y;
    delete[] h_w;
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_w);
    cublasDestroy(handle);

    return 0;
}
