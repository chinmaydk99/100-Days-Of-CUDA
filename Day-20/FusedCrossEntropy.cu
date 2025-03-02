#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define THREADS_PER_BLOCK 32

__global__ void FusedCrossEntropy_Kernel(
    const float* __restrict__ logits,
    const int* __restrict__ labels,
    float* __restrict__ loss,
    float* __restrict__ dlogits,
    int nrows,
    int vocab_size
){
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* row_logits = logits + row * vocab_size;
    float* row_dlogits = dlogits + row * vocab_size;

    extern __shared__ float shared_data[];

    // Step 1: Compute max(logits)
    float local_max = -INFINITY;
    for(int i = tid; i < vocab_size; i += blockDim.x){
        local_max = fmaxf(local_max, row_logits[i]);
    }

    shared_data[tid] = local_max;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tid < stride){
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }

    float max_val = shared_data[0];

    // Step 2: Compute sum(exp(logits - max_val))
    float sum_exp = 0.0f;
    for(int i = tid; i < vocab_size; i += blockDim.x){
        sum_exp += expf(row_logits[i] - max_val);
    }

    shared_data[tid] = sum_exp;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tid < stride){
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    float logsumexp = max_val + logf(shared_data[0]);

    // Step 3: Compute cross-entropy loss for this row
    __shared__ float correct_logit;
    if(tid == 0){
        int label = labels[row];
        correct_logit = (label >= 0) ? row_logits[label] : 0.0f;
        loss[row] = (label >= 0) ? (logsumexp - correct_logit) : 0.0f;
    }
    __syncthreads();

    // Step 4: Compute softmax & gradients
    for(int i = tid; i < vocab_size; i += blockDim.x){
        row_dlogits[i] = expf(row_logits[i] - logsumexp);
    }

    __syncthreads();

    // Step 5: Correct gradient for correct class
    if (tid == 0) {
        int label = labels[row];
        if (label >= 0) {
            row_dlogits[label] -= 1.0f; 
        }
    }
}

void compute_fused_cross_entropy(float *h_logits, float *h_loss, int *h_labels, float *h_dlogits, int n_rows, int vocab_size){
    float *d_logits, *d_loss, *d_dlogits;
    int *d_labels;

    cudaMalloc(&d_logits, n_rows * vocab_size * sizeof(float));
    cudaMalloc(&d_loss, n_rows * sizeof(float));
    cudaMalloc(&d_dlogits, n_rows * vocab_size * sizeof(float));
    cudaMalloc(&d_labels, n_rows * sizeof(int));

    cudaMemset(d_dlogits, 0, n_rows * vocab_size * sizeof(float));

    cudaMemcpy(d_logits, h_logits, n_rows * vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, n_rows * sizeof(int), cudaMemcpyHostToDevice);

    int shared_memory_size = THREADS_PER_BLOCK * sizeof(float);
    FusedCrossEntropy_Kernel<<<n_rows, THREADS_PER_BLOCK, shared_memory_size>>>(
        d_logits, d_labels, d_loss, d_dlogits, n_rows, vocab_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_loss, d_loss, n_rows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dlogits, d_dlogits, n_rows * vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_logits);
    cudaFree(d_labels);
    cudaFree(d_loss);
    cudaFree(d_dlogits);
}

int main() {
    int n_rows = 2, vocab_size = 10; 
    float h_logits[] = {
        2.1, 1.0, 0.5, -0.5, 1.2, 0.8, 0.2, 0.3, -1.0, 0.7,
        1.8, 2.4, -0.2, 1.1, 0.9, 0.5, -0.3, 0.6, 0.0, 1.0
    };
    int h_labels[] = {2, 5}; 
    float h_loss[2];
    float h_dlogits[20];

    compute_fused_cross_entropy(h_logits, h_loss, h_labels, h_dlogits, n_rows, vocab_size);

    // Print loss
    for (int i = 0; i < n_rows; i++) {
        printf("Loss for row %d: %f\n", i, h_loss[i]);
    }

    for (int i = 0; i < n_rows; i++) {
        printf("Gradients for row %d:\n", i);
        for (int j = 0; j < vocab_size; j++) {
            printf("%f ", h_dlogits[i * vocab_size + j]);
        }
        printf("\n");
    }

    return 0;
}
