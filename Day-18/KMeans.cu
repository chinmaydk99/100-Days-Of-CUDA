#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32
#define NUM_CLUSTERS 3    
#define NUM_DIMENSIONS 2
#define NUM_POINTS 10
#define MAX_ITER 100

// Kernel to assign points to clusters.
__global__ void AssignClusters(float *data, float *centroids, int *assignments, int num_points) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_points) {
        float minDist = 1e10;
        int bestCluster = 0;
        for (int c = 0; c < NUM_CLUSTERS; c++) {
            float dist = 0.0f;
            for (int d = 0; d < NUM_DIMENSIONS; d++) {
                float diff = data[idx * NUM_DIMENSIONS + d] - centroids[c * NUM_DIMENSIONS + d];
                dist += diff * diff;
            }
            if (dist < minDist) {
                minDist = dist;
                bestCluster = c;
            }
        }
        assignments[idx] = bestCluster;
    }
}

// Kernel to recompute centroids.
__global__ void RecomputeCentroids(float *data, float *centroids, int *assignments, int num_points) {
    unsigned int c = threadIdx.x;

    if (c < NUM_CLUSTERS) {
        float sum[NUM_DIMENSIONS] = {0};
        int count = 0;
        for (int i = 0; i < num_points; i++) {
            if (assignments[i] == c) {
                for (int d = 0; d < NUM_DIMENSIONS; d++) {
                    sum[d] += data[i * NUM_DIMENSIONS + d];
                }
                count++;
            }
        }

        if (count > 0) {
            for (int d = 0; d < NUM_DIMENSIONS; d++) {
                centroids[c * NUM_DIMENSIONS + d] = sum[d] / count;
            }
        }
    }
}

// Kernel to check convergence.
__global__ void CheckConvergence(float *centroids, float *old_centroids, int *converged) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c < NUM_CLUSTERS) {
        float diff = 0.0f;
        for (int d = 0; d < NUM_DIMENSIONS; d++) {
            float delta = centroids[c * NUM_DIMENSIONS + d] - old_centroids[c * NUM_DIMENSIONS + d];
            diff += fabs(delta);
        }
        if (diff > 1e-4) {
            *converged = 0;
        }
    }
}

void KMeans(float *h_data, float *h_centroids, int num_points) {
    float *d_data, *d_centroids, *d_old_centroids;
    int *d_cluster_assignments, *d_converged;

    cudaMalloc(&d_data, num_points * NUM_DIMENSIONS * sizeof(float));
    cudaMalloc(&d_centroids, NUM_CLUSTERS * NUM_DIMENSIONS * sizeof(float));
    cudaMalloc(&d_old_centroids, NUM_CLUSTERS * NUM_DIMENSIONS * sizeof(float));
    cudaMalloc(&d_cluster_assignments, num_points * sizeof(int));
    cudaMalloc(&d_converged, sizeof(int));

    cudaMemcpy(d_data, h_data, num_points * NUM_DIMENSIONS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, NUM_CLUSTERS * NUM_DIMENSIONS * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 128;
    int gridSize = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        cudaMemcpy(d_old_centroids, d_centroids, NUM_CLUSTERS * NUM_DIMENSIONS * sizeof(float), cudaMemcpyDeviceToDevice);

        AssignClusters<<<gridSize, threadsPerBlock>>>(d_data, d_centroids, d_cluster_assignments, num_points);
        cudaDeviceSynchronize();

        RecomputeCentroids<<<1, NUM_CLUSTERS>>>(d_data, d_centroids, d_cluster_assignments, num_points);
        cudaDeviceSynchronize();

        int converged_flag = 1;
        cudaMemcpy(d_converged, &converged_flag, sizeof(int), cudaMemcpyHostToDevice);

        CheckConvergence<<<1, NUM_CLUSTERS>>>(d_centroids, d_old_centroids, d_converged);
        cudaDeviceSynchronize();

        cudaMemcpy(&converged_flag, d_converged, sizeof(int), cudaMemcpyDeviceToHost);
        if (converged_flag) {
            std::cout << "Converged at iteration " << iter << "\n";
            break;
        }
    }

    cudaMemcpy(h_centroids, d_centroids, NUM_CLUSTERS * NUM_DIMENSIONS * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_old_centroids);
    cudaFree(d_cluster_assignments);
    cudaFree(d_converged);
}

int main() {
    float h_data[NUM_POINTS * NUM_DIMENSIONS] = {
        1.0, 2.0, 2.0, 3.0, 8.0, 8.0, 9.0, 10.0, 4.0, 5.0,
        6.0, 7.0, 3.0, 3.0, 7.0, 6.0, 8.0, 9.0, 1.0, 1.5
    };
    float h_centroids[NUM_CLUSTERS * NUM_DIMENSIONS] = {2.0, 2.0, 8.0, 8.0, 5.0, 5.0};

    std::cout << "Initial Centroids:\n";
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        std::cout << "Cluster " << i << ": (" 
                  << h_centroids[i * NUM_DIMENSIONS] << ", " 
                  << h_centroids[i * NUM_DIMENSIONS + 1] << ")\n";
    }

    KMeans(h_data, h_centroids, NUM_POINTS);

    std::cout << "\nFinal Centroids:\n";
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        std::cout << "Cluster " << i << ": (" 
                  << h_centroids[i * NUM_DIMENSIONS] << ", " 
                  << h_centroids[i * NUM_DIMENSIONS + 1] << ")\n";
    }

    return 0;
}
