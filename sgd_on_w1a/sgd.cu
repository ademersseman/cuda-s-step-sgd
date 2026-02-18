#include <cuda_runtime.h>
#include "sgd.h"

#define BLOCK_SIZE 256

// ---------------- CUDA Kernels ----------------
__global__ void compute_gradient_kernel(
    float* X, float* y, float* w, float* grad,
    int batch_size, int n_features)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (j < n_features) {
        float g = 0.0f;

        for (int i = 0; i < batch_size; i++) {

            // compute dot product a_i^T w
            float dot = 0.0f;
            for (int k = 0; k < n_features; k++)
                dot += X[i*n_features + k] * w[k];

            float yi = y[i];

            float s = 1.0f / (1.0f + expf(yi * dot));  // sigmoid(-yi*dot)

            g += -yi * X[i*n_features + j] * s;
        }

        grad[j] = g;
    }
}

__global__ void update_weights_kernel(float* w, float* grad, float lr, int n_features, int batch_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_features) {
        w[idx] -= lr * grad[idx];
    }
}

// ---------------- Wrapper Functions ----------------
extern "C" void compute_gradient(float* X, float* y, float* w, float* grad, int batch_size, int n_features) {
    int blocks_features = (n_features + BLOCK_SIZE - 1) / BLOCK_SIZE;  // Fixed: use n_features for grid size
    compute_gradient_kernel<<<blocks_features, BLOCK_SIZE>>>(X, y, w, grad, batch_size, n_features);
    cudaDeviceSynchronize();
}

extern "C" void update_weights(float* w, float* grad, float lr, int n_features, int batch_size) {
    int blocks_weights = (n_features + BLOCK_SIZE - 1) / BLOCK_SIZE;
    update_weights_kernel<<<blocks_weights, BLOCK_SIZE>>>(w, grad, lr, n_features, batch_size);
    cudaDeviceSynchronize();
}
