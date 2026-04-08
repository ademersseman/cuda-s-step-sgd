#include <cuda_runtime.h>
#include "s_step_sgd.h"
#include <stdio.h>
#include <chrono>

#define BLOCK_SIZE 256

// ---------------- CUDA Kernels ----------------

// Kernel to apply sigmoid to correction blocks
__global__ void apply_sigmoid_kernel(float *correction, int total_samples, int batch_size, int block_idx)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = block_idx * batch_size;
    int idx = start + tid;
    if (idx < start + batch_size && idx < total_samples)
    {
        correction[idx] = 1.0f / (1.0f + __expf(correction[idx]));
    }
}

// ================== CUDA Helper Functions (callable from host) ==================
void cuda_apply_sigmoid_block(float *correction, int total_samples, int batch_size, int block_idx)
{
    int blocks_sigmoid = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_sigmoid_kernel<<<blocks_sigmoid, BLOCK_SIZE>>>(correction, total_samples, batch_size, block_idx);
    cudaDeviceSynchronize();
}