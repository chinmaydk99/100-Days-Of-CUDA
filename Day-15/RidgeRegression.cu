#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>


void ridge_regression_gradient_descent(
    cublasHandle_t handle, float *d_X, float *d_y, float* d_w,
    int n, int d , float lambda, float learning_rate, int epochs
){
    float alpha = 1.0f;
    float beta = 0.0f;
    float neg_learning_rate = -learning_rate;
    
    float *d_pred;
    float *d_grad;

    cudaMalloc(&d_pred, n*sizeof(float));
    cudaMalloc(&d_grad, d*sizeof(float));

    for(int epoch = 0; epoch < epochs; epoch++){
        // Forward pass: Y_pred = XW
        cublasSgemv(handle, CUBLAS_OP_T,
            n, d , &alpha,
            d_X, d,
            d_w, 1,
            &beta,
            d_pred, 1
        );

        // Calculating the error (XW -Y)
        float minus_one = -1.0f;
        cublasSaxpy(handle, n, &minus_one, d_y, 1, d_pred, 1);

        // Calculating the gradient X^T(XW - Y)
        cublasSgemv(handle, CUBLAS_OP_N,
            d, n , &alpha,
            d_X, n,
            d_pred, 1,
            &beta,
            d_grad, 1
        );

        // Scaling the gradient by 2/n
        float scale = 2.0f/static_cast<float>(n);
        cublasSscal(handle, d, &scale, d_grad, 1);

        // Adding the regularization term
        float reg_scale = 2.0f*lambda;
        cublasSaxpy(handle, d, &reg_scale, d_w, 1, d_grad, 1);

        // Updating the weights: w = w - learning_rate*grad
        cublasSaxpy(handle, d, &neg_learning_rate, d_grad,1, d_w, 1);
    
        if (epoch % 100 == 0) {
            // Residual norm: ||XW-Y|| ^ 2
            float residual_loss = 0.0f;
            cublasSdot(handle, n, d_pred, 1, d_pred, 1, &residual_loss);
            residual_loss /= static_cast<float>(n);

            // Regularization term: lambda * ||w||^2
            float reg_loss = 0.0f;
            cublasSdot(handle, d, d_w, 1, d_w, 1, &reg_loss);
            reg_loss *= lambda;

            float total_loss = residual_loss + reg_loss;
            std::cout << "Epoch: " << epoch << " Loss: " << total_loss << std::endl;
        }
    }
}

int main(){
    cublasHandle_t handle;
    cublasCreate(&handle);

    const int n = 3; //Number of training samples
    const int d = 2; // Number of features
    const float lambda = 0.1f;
    const float learning_rate = 0.0001f;
    const int epochs = 1000;

    // Feature matrix X
    float h_X[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };

    // Target vector y
    float h_y[] = {
        1.0f,
        2.0f,
        3.0f
    };

    // Initialize weights
    float h_w[d] = {0.0f, 0.0f};

    float *d_X, *d_y, *d_w;
    cudaMalloc(&d_X, n*d*sizeof(float));
    cudaMalloc(&d_y, n*sizeof(float));
    cudaMalloc(&d_w, d*sizeof(float));

    cudaMemcpy(d_X, h_X, n*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n*sizeof(float), cudaMemcpyHostToDevice);  

    cudaMemset(d_w, 0, d*sizeof(float)); // We do this instead of copying so that we save some bandwidth

    ridge_regression_gradient_descent(handle, d_X, d_y, d_w, n, d, lambda, learning_rate, epochs);

    cudaMemcpy(h_w, d_w, d*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Final trained weights (w): ";
    for (int i = 0; i < d; i++) {
        std::cout << h_w[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_w);
    cublasDestroy(handle);

    return 0;
}