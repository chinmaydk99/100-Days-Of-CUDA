#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>

#include <iostream>
#include <math.h>

#define BLOCK_DIM 128  // Must be <= embedding_dim

__global__ void LayerNorm_3D_Optimized(float* input, float* output, float* gamma, float* beta,
                                       int batch_size, int seq_length, int embedding_dim, float epsilon) {
    int batch_idx = blockIdx.x;  
    int seq_idx   = blockIdx.y;  
    int emb_idx   = threadIdx.x; 

    int index = (batch_idx * seq_length * embedding_dim) + (seq_idx * embedding_dim) + emb_idx;

    __shared__ float shared_mem[BLOCK_DIM];

    float val = input[index];

    shared_mem[emb_idx] = val;
    __syncthreads();

    for (int stride = BLOCK_DIM / 2; stride > 0; stride /= 2) {
        if (emb_idx < stride) {
            shared_mem[emb_idx] += shared_mem[emb_idx + stride];
        }
        __syncthreads();
    }

    float mean = 0.0f;
    if (emb_idx == 0) {
        mean = shared_mem[0] / embedding_dim;
    }
    __syncthreads();

    float diff = val - mean;
    shared_mem[emb_idx] = diff * diff;
    __syncthreads();

    for (int stride = BLOCK_DIM / 2; stride > 0; stride /= 2) {
        if (emb_idx < stride) {
            shared_mem[emb_idx] += shared_mem[emb_idx + stride];
        }
        __syncthreads();
    }

    float variance = 0.0f;
    if (emb_idx == 0) {
        variance = shared_mem[0] / embedding_dim;
    }
    __syncthreads();

    float stddev = sqrtf(variance + epsilon);
    output[index] = gamma[emb_idx] * ((val - mean) / stddev) + beta[emb_idx];
}

int main() {
    const int batch_size = 4, seq_length = 4, embedding_dim = 4;
    const int total_size = batch_size * seq_length * embedding_dim;

    float *Input_h, *Output_h, *Gamma_h, *Beta_h;

    Input_h = (float*)malloc(total_size * sizeof(float));
    Output_h = (float*)malloc(total_size * sizeof(float));
    Gamma_h = (float*)malloc(embedding_dim * sizeof(float));
    Beta_h = (float*)malloc(embedding_dim * sizeof(float));

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_length; j++) {
            for(int k = 0; k < embedding_dim; k++){
                Input_h[i * seq_length * embedding_dim + j * embedding_dim + k] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
    }


    for(int i = 0; i < embedding_dim; i++){
        Gamma_h[i] = 1.0f;
        Beta_h[i] = 0.0f;
    }


    float *Input_d, *Output_d, *Gamma_d, *Beta_d;
    cudaMalloc(&Input_d, total_size * sizeof(float));
    cudaMalloc(&Output_d, total_size * sizeof(float));
    cudaMalloc(&Gamma_d, embedding_dim * sizeof(float));
    cudaMalloc(&Beta_d, embedding_dim * sizeof(float));


    cudaMemcpy(Input_d, Input_h, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Gamma_d, Gamma_h, embedding_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Beta_d, Beta_h, embedding_dim * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(Output_d, 0, total_size * sizeof(float));

    dim3 blocksize(embedding_dim);   
    dim3 gridsize(batch_size, seq_length);
    size_t shared_mem_size = emb_dim * sizeof(float); 

    LayerNorm_3D<<<gridsize, blocksize, shared_mem_size>>>(Input_d, Output_d, Gamma_d, Beta_d, batch_size, seq_length, embedding_dim, 1e-5);
    
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(Output_h, Output_d, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Layer Normalized Output:\n");
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_length; j++) {
            for(int k = 0; k < embedding_dim; k++){
                printf("%.2f ", Output_h[i * seq_length * embedding_dim + j * embedding_dim + k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Free memory
    cudaFree(Input_d);
    cudaFree(Output_d);
    cudaFree(Gamma_d);
    cudaFree(Beta_d);
    free(Input_h);
    free(Output_h);
    free(Gamma_h);
    free(Beta_h);

    return 0;
}