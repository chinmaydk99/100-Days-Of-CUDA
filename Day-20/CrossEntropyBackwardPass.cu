#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define THREADS_PER_BLOCK 32

__global__ void BackwardPass_Kernel(
    const float* __restrict__ logits, //nrows, vocab_size
    const int* __restrict__ labels, //nrows,
    const float* __restrict__ logsumexp, // nrows,
    float* __restrict__ dlogits, //nrows, vocab_size
    int nrows, 
    int vocab_size
){
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* row_logits = logits + row*vocab_size;
    float* row_dlogits = dlogits + row*vocab_size;

    float logsumexpval = logsumexp[row];

    // Computing the softmax
    for(int i = tid; i < vocab_size; i += blockDim.x){
        row_dlogits[i] = expf(row_logits[i] - logsumexpval); // p_i = exp(x_i - logsumexp)
    }

    if(tid == 0){
        int label = labels[row];
        if(label >= 0){
            row_dlogits[label] -= 1.0f;
        }
    }
}

void compute_cross_entropy_backward(float *h_logits, int *h_labels, float *h_logsumexp, float *h_dlogits, int n_rows, int vocab_size){
    float *d_logits, *d_dlogits, *d_logsumexp;
    int *d_labels;

    cudaMalloc(&d_logits, n_rows * vocab_size * sizeof(float));
    cudaMalloc(&d_dlogits, n_rows * vocab_size * sizeof(float));
    cudaMalloc(&d_labels, n_rows * sizeof(int));
    cudaMalloc(&d_logsumexp, n_rows * sizeof(float));

    cudaMemcpy(d_logits, h_logits, n_rows * vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, n_rows * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_logsumexp, h_logsumexp, n_rows * sizeof(float), cudaMemcpyHostToDevice);
    
    BackwardPass_Kernel<<<n_rows, THREADS_PER_BLOCK>>>(
        d_logits, d_labels, d_dlogits, d_logsumexp, n_rows, vocab_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_dlogits, d_dlogits, n_rows * vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_logits);
    cudaFree(d_labels);
    cudaFree(d_logsumexp);
    cudaFree(d_dlogits);
}

int main() {
    int n_rows = 2, vocab_size = 10; 
    float h_logits[] = {
        2.1, 1.0, 0.5, -0.5, 1.2, 0.8, 0.2, 0.3, -1.0, 0.7,
        1.8, 2.4, -0.2, 1.1, 0.9, 0.5, -0.3, 0.6, 0.0, 1.0
    };
    int h_labels[] = {2, 5}; // Example labels (correct class index)
    float h_logsumexp[] = {3.3026, 3.7883}; // Precomputed logsumexp
    float h_dlogits[20];

    compute_cross_entropy_backward(h_logits, h_labels, h_logsumexp, h_dlogits, n_rows, vocab_size);

    for (int i = 0; i < n_rows; i++) {
        printf("Gradients for row %d:\n", i);
        for (int j = 0; j < vocab_size; j++) {
            printf("%f ", h_dlogits[i * vocab_size + j]);
        }
        printf("\n");
    }

    return 0;
}