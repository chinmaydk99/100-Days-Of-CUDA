#include <iostream>
#include <cuda_runtime.h>

#define N 8  // Array size

__device__ void merge(float *arr, int left, int mid, int right) {
    int leftSize = mid - left + 1;
    int rightSize = right - mid;

    float *L = new float[leftSize];
    float *R = new float[rightSize];

    for (int i = 0; i < leftSize; i++)
        L[i] = arr[left + i];
    for (int i = 0; i < rightSize; i++)
        R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < leftSize && j < rightSize) {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    // Copy remaining elements
    while (i < leftSize)
        arr[k++] = L[i++];
    while (j < rightSize)
        arr[k++] = R[j++];

    delete[] L;
    delete[] R;
}

// CUDA Kernel for parallel merge sort
__global__ void mergeSortParallel(float *arr, int left, int right) {
    int thread_id = threadIdx.x;

    if (left < right) {
        int mid = left + (right - left) / 2;

        if (thread_id == 0) mergeSortParallel<<<1, 1>>>(arr, left, mid);
        if (thread_id == 1) mergeSortParallel<<<1, 1>>>(arr, mid + 1, right);
        __syncthreads();

        merge(arr, left, mid, right);
    }
}


int main() {
    float h_arr[N] = {8.5, 3.1, 7.3, 4.2, 6.9, 2.0, 1.4, 5.8};
    float *d_arr;

    cudaMalloc(&d_arr, N * sizeof(float));
    cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice);

    mergeSortParallel<<<1, N / 2>>>(d_arr, 0, N - 1);
    cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Sorted Array: ";
    for (int i = 0; i < N; i++) std::cout << h_arr[i] << " ";
    std::cout << std::endl;

    cudaFree(d_arr);
    return 0;
}
