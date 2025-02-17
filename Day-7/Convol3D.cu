#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 8  
#define KERNEL_SIZE 3 


__constant__ float d_filter[KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE];

__global__ void conv3D_tiled(
    const float* __restrict__ input,
    float* __restrict__ output,
    int depth, int height, int width
) {
    __shared__ float tile[TILE_SIZE + KERNEL_SIZE - 1][TILE_SIZE + KERNEL_SIZE - 1][TILE_SIZE + KERNEL_SIZE - 1];

    // Calculate global and shared memory indices
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int z = blockIdx.z * TILE_SIZE + threadIdx.z;

    int shared_x = threadIdx.x + (KERNEL_SIZE / 2);
    int shared_y = threadIdx.y + (KERNEL_SIZE / 2);
    int shared_z = threadIdx.z + (KERNEL_SIZE / 2);

    // Load input tile into shared memory
    if (x < width && y < height && z < depth) {
        tile[shared_z][shared_y][shared_x] = input[(z * height * width) + (y * width) + x];
    } else {
        tile[shared_z][shared_y][shared_x] = 0.0f;  // Padding for out-of-bounds threads
    }

    __syncthreads();

    if (threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE && threadIdx.z < TILE_SIZE) {
        if (x >= (KERNEL_SIZE / 2) && x < (width - KERNEL_SIZE / 2) &&
            y >= (KERNEL_SIZE / 2) && y < (height - KERNEL_SIZE / 2) &&
            z >= (KERNEL_SIZE / 2) && z < (depth - KERNEL_SIZE / 2)) {

            float result = 0.0f;
            for (int fz = 0; fz < KERNEL_SIZE; fz++) {
                for (int fy = 0; fy < KERNEL_SIZE; fy++) {
                    for (int fx = 0; fx < KERNEL_SIZE; fx++) {
                        result += d_filter[fz * KERNEL_SIZE * KERNEL_SIZE + fy * KERNEL_SIZE + fx] *
                                  tile[shared_z + fz - (KERNEL_SIZE / 2)][shared_y + fy - (KERNEL_SIZE / 2)][shared_x + fx - (KERNEL_SIZE / 2)];
                    }
                }
            }

            output[(z * height * width) + (y * width) + x] = result;
        }
    }
}

int main() {
    int depth = 16, height = 16, width = 16; 
    int input_size = depth * height * width * sizeof(float);
    int output_size = depth * height * width * sizeof(float);
    int filter_size = KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE * sizeof(float);

    // Allocate host memory
    float *h_input = new float[depth * height * width];
    float *h_output = new float[depth * height * width];
    float h_filter[KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE] = {
        0, 0, 0,  0, 1, 0,  0, 0, 0,
        0, 1, 0,  1, -6, 1,  0, 1, 0,
        0, 0, 0,  0, 1, 0,  0, 0, 0
    };

    // Initialize input with some values
    for (int i = 0; i < depth * height * width; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);

    // Copy data to device
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_filter, h_filter, filter_size);  // Copy filter to constant memory

    // Launch Kernel
    dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE, (depth + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    conv3D_tiled<<<grid, block>>>(d_input, d_output, depth, height, width);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    // Print a small portion of output
    std::cout << "Output (small section):\n";
    for (int i = 0; i < 5; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << "\n";

    // Free memory
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
