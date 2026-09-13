#include <cuda_runtime.h>
#include <stdio.h>

#include "s_step_sgd.h"

#define BLOCK_SIZE 256

// ---------------- CUDA Kernels ----------------
// Kernel to apply sigmoid to correction blocks
__global__ void apply_sigmoid_kernel(float *correction, size_t total_samples, size_t batch_size, size_t block_idx)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t start = block_idx * batch_size;
    size_t idx = start + tid;
    if (idx < start + batch_size && idx < total_samples)
    {
        correction[idx] = 1.0f / (1.0f + __expf(correction[idx]));
    }
}
// ================== CUDA Helper Functions (callable from host) ==================
void cuda_apply_sigmoid_block(cudaStream_t stream, float *correction, size_t total_samples, size_t batch_size, size_t block_idx)
{
    size_t blocks_sigmoid = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_sigmoid_kernel<<<blocks_sigmoid, BLOCK_SIZE, 0, stream>>>(correction, total_samples, batch_size, block_idx);
    cudaDeviceSynchronize();
}
