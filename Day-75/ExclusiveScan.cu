#include <iostream>
#include <cuda_runtime.h>

__global__ void exclusiveScan(int* d_out, const int* d_in, int n) {
    extern __shared__ int temp[]; // allocated on kernel launch
    int tid = threadIdx.x;

    if (2 * tid < n) temp[2 * tid]     = d_in[2 * tid];
    if (2 * tid + 1 < n) temp[2 * tid + 1] = d_in[2 * tid + 1];

    __syncthreads();

    // Up-sweep (reduce) phase
    int stride = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = stride * (2 * tid + 1) - 1;
            int bi = stride * (2 * tid + 2) - 1;
            if (bi < n) temp[bi] += temp[ai];
        }
        stride *= 2;
    }

    // Clear the last element
    if (tid == 0) temp[n - 1] = 0;

    // Down-sweep phase
    for (int d = 1; d < n; d *= 2) {
        stride >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = stride * (2 * tid + 1) - 1;
            int bi = stride * (2 * tid + 2) - 1;
            if (bi < n) {
                int t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
    }
    __syncthreads();

    // Write results to global memory
    if (2 * tid < n) d_out[2 * tid] = temp[2 * tid];
    if (2 * tid + 1 < n) d_out[2 * tid + 1] = temp[2 * tid + 1];
}

void runScanExample() {
    const int N = 8;
    int h_in[N] = {3, 1, 7, 0, 4, 1, 6, 3};
    int h_out[N];

    int *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    exclusiveScan<<<1, N/2, N * sizeof(int)>>>(d_out, d_in, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Input: ";
    for (int i = 0; i < N; ++i) std::cout << h_in[i] << " ";
    std::cout << "\nExclusive Scan Output: ";
    for (int i = 0; i < N; ++i) std::cout << h_out[i] << " ";
    std::cout << "\n";

    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    runScanExample();
    return 0;
}
