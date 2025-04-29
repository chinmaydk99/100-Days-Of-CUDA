#include <cuda_runtime.h>
#include <stdio.h>

__global__ void cumulative_product_kernel(const float* input, float* output, int N) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < N) {
        shared_mem[tid] = input[gid];
    }
    __syncthreads();
    
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx < blockDim.x) {
            shared_mem[idx] *= shared_mem[idx - stride];
        }
        __syncthreads();
    }
    
    for (int stride = blockDim.x / 4; stride > 0; stride /= 2) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx + stride < blockDim.x) {
            shared_mem[idx + stride] *= shared_mem[idx];
        }
        __syncthreads();
    }
    
    if (gid < N) {
        output[gid] = shared_mem[tid];
    }
}

__global__ void block_scan_kernel(const float* input, float* output, float* block_products, int N) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < N) {
        shared_mem[tid] = input[gid];
    } else {
        shared_mem[tid] = 1.0f;
    }
    __syncthreads();
    
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx < blockDim.x) {
            shared_mem[idx] *= shared_mem[idx - stride];
        }
        __syncthreads();
    }
    
    if (tid == blockDim.x - 1 && blockIdx.x < gridDim.x) {
        block_products[blockIdx.x] = shared_mem[tid];
    }
    
    for (int stride = blockDim.x / 4; stride > 0; stride /= 2) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx + stride < blockDim.x) {
            shared_mem[idx + stride] *= shared_mem[idx];
        }
        __syncthreads();
    }
    
    if (gid < N) {
        output[gid] = shared_mem[tid];
    }
}

__global__ void apply_block_products(float* output, const float* block_products, int N, int block_size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < N && blockIdx.x > 0) {
        float product = 1.0f;
        for (int i = 0; i < blockIdx.x; i++) {
            product *= block_products[i];
        }
        output[gid] *= product;
    }
}

void cumulative_product(const float* h_input, float* h_output, int N) {
    float *d_input, *d_output, *d_block_products;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    
    if (grid_size == 1) {
        // Single block case
        cumulative_product_kernel<<<1, block_size, block_size * sizeof(float)>>>(d_input, d_output, N);
    } else {
        // Multi-block case
        cudaMalloc(&d_block_products, grid_size * sizeof(float));
        
        // Phase 1: Scan within each block and save last element
        block_scan_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
            d_input, d_output, d_block_products, N);
        
        // Phase 2: Scan the block products
        float *d_block_products_scanned;
        cudaMalloc(&d_block_products_scanned, grid_size * sizeof(float));
        
        if (grid_size <= block_size) {
            cumulative_product_kernel<<<1, grid_size, grid_size * sizeof(float)>>>(
                d_block_products, d_block_products_scanned, grid_size);
        } else {
            // Recursive call for large number of blocks
            float *h_block_products = new float[grid_size];
            float *h_block_products_scanned = new float[grid_size];
            
            cudaMemcpy(h_block_products, d_block_products, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
            
            cumulative_product(h_block_products, h_block_products_scanned, grid_size);
            
            cudaMemcpy(d_block_products_scanned, h_block_products_scanned, 
                       grid_size * sizeof(float), cudaMemcpyHostToDevice);
            
            delete[] h_block_products;
            delete[] h_block_products_scanned;
        }
        
        // Phase 3: Apply block scan results to each element
        apply_block_products<<<grid_size, block_size>>>(d_output, d_block_products_scanned, N, block_size);
        
        cudaFree(d_block_products);
        cudaFree(d_block_products_scanned);
    }
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}