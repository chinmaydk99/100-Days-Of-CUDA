#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

#define CUDA_CHECK(call)                                                          \
{                                                                                \
    cudaError_t err = call;                                                      \
    if (err != cudaSuccess)                                                      \
    {                                                                            \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,       \
                cudaGetErrorString(err));                                        \
        exit(EXIT_FAILURE);                                                      \
    }                                                                            \
}

__global__ void blockScan(const float* input, float* output, float* blockSums, int N) {
    extern __shared__ float temp[];

    int tid = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalId < N) {
        temp[tid] = input[globalId];
    } else {
        temp[tid] = 0.0f;
    }
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    if (tid == blockDim.x - 1) {
        if (blockSums != nullptr) {
            blockSums[blockIdx.x] = temp[tid];
        }
    }

    if (tid == 0) {
        temp[blockDim.x - 1] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            float val_at_index = temp[index];
            float val_at_stride = temp[index - stride];
            temp[index] += val_at_stride;
            temp[index - stride] = val_at_index;
        }
        __syncthreads();
    }

    if (globalId < N) {
        output[globalId] = temp[tid];
    }
}

__global__ void addBlockSums(float* output, const float* blockSumsPrefixSum, int N) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalId < N && blockIdx.x > 0) {
        output[globalId] += blockSumsPrefixSum[blockIdx.x - 1];
    }
}

void cumulativeSumGPU(const std::vector<float>& h_input, std::vector<float>& h_output) {
    int N = h_input.size();
    if (N == 0) return;

    const int BLOCK_SIZE = 256;
    const int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *d_input, *d_intermediate_output, *d_blockSums, *d_blockSumsPrefixSum;
    size_t inputSize = N * sizeof(float);
    size_t blockSumsSize = numBlocks * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK(cudaMalloc(&d_intermediate_output, inputSize));
    CUDA_CHECK(cudaMalloc(&d_blockSums, blockSumsSize));
    CUDA_CHECK(cudaMalloc(&d_blockSumsPrefixSum, blockSumsSize));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), inputSize, cudaMemcpyHostToDevice));

    blockScan<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_input, d_intermediate_output, d_blockSums, N
    );
    CUDA_CHECK(cudaGetLastError());

    if (numBlocks > 1) {
        std::vector<float> h_blockSums(numBlocks);
        std::vector<float> h_blockSumsPrefixSum(numBlocks);

        CUDA_CHECK(cudaMemcpy(h_blockSums.data(), d_blockSums, blockSumsSize, cudaMemcpyDeviceToHost));

        h_blockSumsPrefixSum[0] = 0;
        for (int i = 1; i < numBlocks; ++i) {
            h_blockSumsPrefixSum[i] = h_blockSumsPrefixSum[i - 1] + h_blockSums[i - 1];
        }

        CUDA_CHECK(cudaMemcpy(d_blockSumsPrefixSum, h_blockSumsPrefixSum.data(), blockSumsSize, cudaMemcpyHostToDevice));

        addBlockSums<<<numBlocks, BLOCK_SIZE>>>(
            d_intermediate_output, d_blockSumsPrefixSum, N
        );
        CUDA_CHECK(cudaGetLastError());
    }

    auto elementwiseAddKernel = [] __device__ (float* output, const float* input, int N) {
        int globalId = blockIdx.x * blockDim.x + threadIdx.x;
        if (globalId < N) {
             output[globalId] += input[globalId];
        }
    };
    elementwiseAddKernel<<<numBlocks, BLOCK_SIZE>>>(d_intermediate_output, d_input, N);
    CUDA_CHECK(cudaGetLastError());


    CUDA_CHECK(cudaMemcpy(h_output.data(), d_intermediate_output, inputSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_intermediate_output));
    CUDA_CHECK(cudaFree(d_blockSums));
    CUDA_CHECK(cudaFree(d_blockSumsPrefixSum));
}


int main() {
    int N = 1 << 10;
    std::vector<float> h_input(N);
    std::vector<float> h_output_gpu(N);
    std::vector<float> h_output_cpu(N);

    std::fill(h_input.begin(), h_input.end(), 1.0f);

    std::cout << "Input size: " << N << " elements." << std::endl;

    std::cout << "Calculating cumulative sum on GPU..." << std::endl;
    cumulativeSumGPU(h_input, h_output_gpu);
    std::cout << "GPU calculation complete." << std::endl;

    std::cout << "Calculating cumulative sum on CPU for verification..." << std::endl;
    std::partial_sum(h_input.begin(), h_input.end(), h_output_cpu.begin());
    std::cout << "CPU calculation complete." << std::endl;

    std::cout << "Verifying results..." << std::endl;
    bool passed = true;
    float tolerance = 1e-5;
    for (int i = 0; i < N; ++i) {
        if (std::abs(h_output_gpu[i] - h_output_cpu[i]) > tolerance) {
            std::cerr << "Verification FAILED at index " << i << ": "
                      << "GPU=" << h_output_gpu[i] << ", CPU=" << h_output_cpu[i]
                      << ", Input=" << h_input[i] << std::endl;
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "Verification PASSED!" << std::endl;
    } else {
        std::cout << "Verification FAILED!" << std::endl;
    }

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}