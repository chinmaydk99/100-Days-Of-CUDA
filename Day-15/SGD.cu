#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void sgd_update(cublasHandle_t handle, float *d_params, float *d_grads, )