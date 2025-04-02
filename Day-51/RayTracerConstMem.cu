#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "book.h"

#define DIM 1024
#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f
#define SPHERES 20

static void HandleError( cudaError_t err,
    const char *file,
    int line ) {
if (err != cudaSuccess) {
printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
file, line );
exit( EXIT_FAILURE );
}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

struct Sphere {
    float r, g, b;
    float radius;
    float x, y, z;

    __device__ float hit(float ox, float oy, float *n) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / radius;
            return dz + z;
        }
        return -INF;
    }
};

__constant__ Sphere s[SPHERES];

__global__ void kernel(unsigned char *ptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float ox = x - DIM / 2;
    float oy = y - DIM / 2;

    float r = 0, g = 0, b = 0;
    float maxz = -INF;

    for (int i = 0; i < SPHERES; i++) {
        float n;
        float t = s[i].hit(ox, oy, &n);
        if (t > maxz) {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
    }

    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 2] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;
}

void save_ppm(const char *filename, unsigned char *data, int width, int height) {
    std::ofstream f(filename, std::ios::binary);
    f << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        f.put(data[i * 4 + 0]);  // R
        f.put(data[i * 4 + 1]);  // G
        f.put(data[i * 4 + 2]);  // B
    }
    f.close();
}

int main() {
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    unsigned char *dev_bitmap;
    unsigned char *host_bitmap = new unsigned char[DIM * DIM * 4];

    // Allocate device memory for pixel buffer
    HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap, DIM * DIM * 4));

    // Generate sphere data
    Sphere *temp_s = (Sphere *)malloc(sizeof(Sphere) * SPHERES);
    for (int i = 0; i < SPHERES; i++) {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500;
        temp_s[i].y = rnd(1000.0f) - 500;
        temp_s[i].z = rnd(1000.0f) - 500;
        temp_s[i].radius = rnd(100.0f) + 20;
    }

    // Copy to constant memory
    HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES));
    free(temp_s);

    // Launch kernel
    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel<<<grids, threads>>>(dev_bitmap);

    // Copy result back to host
    HANDLE_ERROR(cudaMemcpy(host_bitmap, dev_bitmap, DIM * DIM * 4, cudaMemcpyDeviceToHost));

    // Save as PPM file
    save_ppm("output.ppm", host_bitmap, DIM, DIM);
    std::cout << "Saved output.ppm\n";

    // Print timing info
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsed;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));
    std::cout << "Render time: " << elapsed << " ms\n";

    // Cleanup
    HANDLE_ERROR(cudaFree(dev_bitmap));
    delete[] host_bitmap;
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return 0;
}
