#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#define INF 2e10f
#define DIM 1024
#define SPHERES 20
#define rnd(x) (x * rand() / RAND_MAX)


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

    // Method to check if the ray does intersect the sphere
    // ox, oy - Ray's origin coordinates
    __device__ float hit(float ox, float oy, float* n) {
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

// Global GPU pointer to sphere array
__device__ Sphere *d_spheres;

// Each thread computes one pixel
__global__ void ray_tracing_kernel(unsigned char* ptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset = x + y * blockDim.x * gridDim.x; // x + y * width

    // Centring the coordinates
    float ox = x - DIM / 2;
    float oy = y - DIM / 2;

    float r = 0, g = 0, b = 0;
    float maxz = -INF;

    for (int i = 0; i < SPHERES; i++) {// If ray hits multiple spheres we need to consider the closest
        // In orthographic camera largest z value corresponds to the closest
        float n;
        float t = d_spheres[i].hit(ox, oy, &n);
        if (t > maxz) {
            float fscale = n;
            r = d_spheres[i].r * fscale;
            g = d_spheres[i].g * fscale;
            b = d_spheres[i].b * fscale;
            maxz = t;
        }
    }

    // Since Each pixel takes 4 bytes: red , green , blue, alpha
    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 2] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;
}

// Save RGBA buffer as a .ppm file
void save_ppm(const char* filename, unsigned char* data, int width, int height) {
    std::ofstream f(filename, std::ios::binary);
    f << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; i++) {
        f.put(data[i * 4 + 0]);  // R
        f.put(data[i * 4 + 1]);  // G
        f.put(data[i * 4 + 2]);  // B
    }
    f.close();
}

int main() {
    // CUDA events for timing
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    // Allocate memory for bitmap
    unsigned char* d_bitmap;
    HANDLE_ERROR(cudaMalloc((void**)&d_bitmap, DIM * DIM * 4));
    unsigned char* host_bitmap = new unsigned char[DIM * DIM * 4];

    // Allocate sphere data
    Sphere* temp = new Sphere[SPHERES];
    for (int i = 0; i < SPHERES; ++i) {
        temp[i].r = rnd(1.0f);
        temp[i].g = rnd(1.0f);
        temp[i].b = rnd(1.0f);
        temp[i].x = rnd(1000.0f) - 500;
        temp[i].y = rnd(1000.0f) - 500;
        temp[i].z = rnd(1000.0f) - 500;
        temp[i].radius = rnd(100.0f) + 20;
    }

    Sphere* d_spheres_ptr;
    HANDLE_ERROR(cudaMalloc((void**)&d_spheres_ptr, sizeof(Sphere) * SPHERES));
    HANDLE_ERROR(cudaMemcpy(d_spheres_ptr, temp, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice));

    // Make pointer visible in device code
    cudaMemcpyToSymbol(d_spheres, &d_spheres_ptr, sizeof(Sphere*));

    // Launch kernel
    dim3 threads(16, 16);
    dim3 blocks(DIM / 16, DIM / 16);
    ray_tracing_kernel<<<blocks, threads>>>(d_bitmap);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    // Copy result back and save image
    HANDLE_ERROR(cudaMemcpy(host_bitmap, d_bitmap, DIM * DIM * 4, cudaMemcpyDeviceToHost));
    save_ppm("output.ppm", host_bitmap, DIM, DIM);
    std::cout << "Saved image as output.ppm\n";

    // Timing
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsed;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));
    std::cout << "Render time: " << elapsed << " ms\n";

    // Cleanup
    HANDLE_ERROR(cudaFree(d_bitmap));
    HANDLE_ERROR(cudaFree(d_spheres_ptr));
    delete[] temp;
    delete[] host_bitmap;
}
