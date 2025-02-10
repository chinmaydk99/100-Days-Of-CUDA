#include <iostream>

__global__ void vecAddKernel(float* A, float* B, float* C, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n)
    {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n)
{
    //Define variables in GPU
    float *A_d, *B_d, *C_d;
    int size = (n)*sizeof(float);

    //Allocate memory in GPU
    cudaMalloc((void**)(&A_d), size);
    cudaMalloc((void**)(&B_d), size);
    cudaMalloc((void**)(&C_d),size);

    // Copy variables from Host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    

    //Call the kernel
    int blockSize = 256;
    int numBlocks = ceil(n/blockSize);
    vecAddKernel<<<numBlocks,blockSize>>>(A_d,B_d,C_d,n);

    //Move the result from Device to Host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


// Test Code
int main() {
    const int N = 1000;
    float *A = new float[N];
    float *B = new float[N];
    float *C = new float[N];

    // Initialize input vectors
    for(int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    // Perform vector addition
    vecAdd(A, B, C, N);

    // Verify results
    for(int i = 0; i < N; i++) {
        if(C[i] != A[i] + B[i]) {
            std::cout << "Error: mismatch at position " << i << std::endl;
            break;
        }
    }

    std::cout << "Vector addition completed successfully!" << std::endl;

    // Free host memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}