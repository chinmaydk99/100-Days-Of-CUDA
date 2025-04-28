#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h> // Added for malloc/free

__global__ void triplet_margin_loss_kernel(const float* anchor,
                                          const float* positive,
                                          const float* negative,
                                          float* batch_losses,
                                          float margin,
                                          int B,
                                          int E) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < B) {
        float dist_ap_sq = 0.0f;
        float dist_an_sq = 0.0f;

        for (int k = 0; k < E; ++k) {
            int element_idx = idx * E + k;
            float diff_ap = anchor[element_idx] - positive[element_idx];
            float diff_an = anchor[element_idx] - negative[element_idx];
            dist_ap_sq += diff_ap * diff_ap;
            dist_an_sq += diff_an * diff_an;
        }

        float dist_ap = sqrtf(dist_ap_sq);
        float dist_an = sqrtf(dist_an_sq);

        batch_losses[idx] = fmaxf(0.0f, dist_ap - dist_an + margin);
    }
}