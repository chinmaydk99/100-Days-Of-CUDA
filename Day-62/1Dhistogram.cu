#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h> // For floorf

// Helper macro for checking CUDA calls
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while (0)


/*
 * Kernel: Computes a histogram using global atomic operations.
 * Each thread processes multiple input elements using a grid-stride loop.
 */
__global__ void histogram_atomic_kernel(
    const float* __restrict__ input,    // Input data array
    unsigned int* __restrict__ histogram, // Output histogram bins (initialized to 0)
    int n,                              // Number of elements in input array
    int num_bins,                       // Number of bins in the histogram
    float min_val,                      // Minimum value for the histogram range
    float max_val                       // Maximum value for the histogram range (exclusive for the last bin)
) {
    // Global thread index using grid-stride loop
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float range = max_val - min_val;
    // Avoid division by zero if range is zero
    float bin_width_inv = (range > 1e-9f) ? (float)num_bins / range : 0.0f;

    while (idx < n) {
        float value = input[idx];

        // Calculate bin index
        // Ensure value is within [min_val, max_val) before calculating bin
        int bin_index = -1;
        if (value >= min_val && value < max_val) {
             // Scale value to [0, num_bins), then floor to get index
            bin_index = static_cast<int>(floorf((value - min_val) * bin_width_inv));
            // Clamp index just in case of floating point inaccuracies at the boundary
             bin_index = max(0, min(num_bins - 1, bin_index));
        }
        // Optional: Handle values exactly equal to max_val if needed,
        // typically they go in the last bin or are excluded depending on definition.
        // This implementation excludes values >= max_val.

        // If the value falls into a valid bin, atomically increment the counter
        if (bin_index >= 0) {
            // atomicAdd increments the value at the memory address atomically
            atomicAdd(&histogram[bin_index], 1);
        }

        // Move to the next element for this thread
        idx += stride;
    }
}


// ============================================================
//  Entry Point C Function (Host Function)
// ============================================================
extern "C" void solution(
    const float* d_input,     // Device pointer to input data
    unsigned int* d_histogram, // Device pointer to output histogram bins
    int n,                     // Number of input elements
    int num_bins,              // Number of histogram bins
    float min_val,             // Minimum value of histogram range
    float max_val              // Maximum value of histogram range
) {

    // --- Ensure Histogram is Zeroed ---
    // The caller should typically ensure the histogram buffer is zeroed before calling.
    // We can add an explicit zeroing step here for robustness.
    CUDA_CHECK(cudaMemset(d_histogram, 0, num_bins * sizeof(unsigned int)));

    // --- Kernel Launch Configuration ---
    int threads_per_block = 256; // Common choice
    // Launch enough blocks to have a high degree of parallelism,
    // often more blocks than SMs. Let's aim for e.g., 1024 blocks minimum
    // or enough to cover 'n' elements initially.
    int min_blocks = 1024;
    int blocks_per_grid = min(min_blocks, (n + threads_per_block - 1) / threads_per_block);
    // Ensure at least one block is launched
     blocks_per_grid = max(1, blocks_per_grid);


    // --- Launch the Kernel ---
    histogram_atomic_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_input,
        d_histogram,
        n,
        num_bins,
        min_val,
        max_val
    );

    // --- Check for Kernel Launch Errors ---
    CUDA_CHECK(cudaGetLastError());

    // --- Synchronize ---
    // Wait for the kernel to complete before the host function returns.
    CUDA_CHECK(cudaDeviceSynchronize());
}