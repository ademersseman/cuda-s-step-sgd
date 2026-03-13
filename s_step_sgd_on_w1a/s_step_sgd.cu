#include <cuda_runtime.h>
#include "s_step_sgd.h"
#include <stdio.h>

#define BLOCK_SIZE 256

// ---------------- CUDA Kernels ----------------
__global__ void compute_gradient_kernel(
    float *A, float *y, float *x, float *grad,
    int batch_size, int n_features)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (j < n_features)
    {
        float g = 0.0f;

        for (int i = 0; i < batch_size; i++)
        {

            // compute dot product A_i^T x
            float dot = 0.0f;
            for (int k = 0; k < n_features; k++)
                dot += A[i * n_features + k] * x[k];

            float yi = y[i];

            float s = 1.0f / (1.0f + expf(yi * dot)); // sigmoid(-yi*dot)

            g += -yi * A[i * n_features + j] * s;
        }

        grad[j] = g;
    }
}

// Kernel to scale A by y (element-wise multiply each row by y[i])
__global__ void scale_A_by_y_kernel(float *A, float *y, float *A_scaled, int total_samples, int n_features)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < total_samples)
    {
        float yi = y[i];
        for (int j = 0; j < n_features; j++)
        {
            A_scaled[i * n_features + j] = A[i * n_features + j] * yi;
        }
    }
}

// Kernel to compute A*x for initial correction
__global__ void compute_Ax_kernel(float *A, float *x, float *correction, int total_samples, int n_features)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < total_samples)
    {
        float dot = 0.0f;
        for (int k = 0; k < n_features; k++)
        {
            dot += A[i * n_features + k] * x[k];
        }
        correction[i] = dot;
    }
}

// Kernel to compute block Gram matrix G = A*A'
__global__ void compute_Gram_kernel(float *A, float *G, int total_samples, int n_features, int batch_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < total_samples && j < total_samples)
    {
        float dot = 0.0f;
        for (int k = 0; k < n_features; k++)
        {
            dot += A[i * n_features + k] * A[j * n_features + k];
        }
        G[i * total_samples + j] = dot;
    }
}

// Kernel to apply all corrections for a given block i
__global__ void apply_all_corrections_kernel(float *G, float *correction, int total_samples, int batch_size, int i_block, float eta)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= batch_size)
        return;

    int i_start = i_block * batch_size;
    int i_idx = i_start + tid;

    float sum = 0.0f;
    for (int j = 0; j < i_block; j++)
    {
        int j_start = j * batch_size;
        for (int j_tid = 0; j_tid < batch_size; j_tid++)
        {
            int j_idx = j_start + j_tid;
            sum += G[i_idx * total_samples + j_idx] * correction[j_idx];
        }
    }
    correction[i_idx] += eta * sum;
}

// Kernel to apply sigmoid to correction blocks
__global__ void apply_sigmoid_kernel(float *correction, int total_samples, int batch_size, int block_idx)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = block_idx * batch_size;
    int idx = start + tid;
    if (idx < start + batch_size && idx < total_samples)
    {
        correction[idx] = 1.0f / (1.0f + expf(correction[idx]));
    }
}

// Kernel to compute final gradient: grad = -A' * correction
__global__ void compute_final_gradient_kernel(float *A, float *correction, float *grad, int total_samples, int n_features)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < n_features)
    {
        float g = 0.0f;
        for (int i = 0; i < total_samples; i++)
        {
            g += A[i * n_features + j] * correction[i];
        }
        grad[j] = -g;
    }
}

__global__ void update_weights_kernel(float *x, float *grad, float lr, int n_features, int batch_size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_features)
    {
        x[idx] -= lr * grad[idx];
    }
}

// ---------------- Wrapper Functions ----------------
extern "C" void compute_gradient(float *A, float *y, float *x, float *grad, int batch_size, int n_features)
{
    int blocks_features = (n_features + BLOCK_SIZE - 1) / BLOCK_SIZE; // Fixed: use n_features for grid size
    compute_gradient_kernel<<<blocks_features, BLOCK_SIZE>>>(A, y, x, grad, batch_size, n_features);
    cudaDeviceSynchronize();
}

extern "C" void compute_sstep_gradient(float *A, float *y, float *x, float *grad, int batch_size, int s, float eta, int n_features, int iter, ProfileStats *stats)
{
    int total_samples = s * batch_size;

    // Allocate device memory for intermediate computations
    float *d_X_scaled;
    float *d_correction;
    float *d_G;
    cudaMalloc(&d_X_scaled, total_samples * n_features * sizeof(float));
    cudaMalloc(&d_correction, total_samples * sizeof(float));
    cudaMalloc(&d_G, total_samples * total_samples * sizeof(float));

    // Scale A by y: X_scaled[i,j] = A[i,j] * y[i]
    int blocks_samples = (total_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scale_A_by_y_kernel<<<blocks_samples, BLOCK_SIZE>>>(A, y, d_X_scaled, total_samples, n_features);
    cudaDeviceSynchronize();

    // Compute initial correction = A_scaled*x
    compute_Ax_kernel<<<blocks_samples, BLOCK_SIZE>>>(d_X_scaled, x, d_correction, total_samples, n_features);
    cudaDeviceSynchronize();

    cudaEvent_t start_gram, stop_gram;
    cudaEventCreate(&start_gram);
    cudaEventCreate(&stop_gram);

    cudaEventRecord(start_gram);
    // Compute Gram matrix G = A_scaled*A_scaled'
    dim3 blocks_G((total_samples + 15) / 16, (total_samples + 15) / 16);
    dim3 threads_G(16, 16);
    compute_Gram_kernel<<<blocks_G, threads_G>>>(d_X_scaled, d_G, total_samples, n_features, batch_size);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop_gram);
    float elapsed_gram;
    cudaEventElapsedTime(&elapsed_gram, start_gram, stop_gram);
    stats->gram_time += elapsed_gram;
    cudaEventDestroy(start_gram);
    cudaEventDestroy(stop_gram);
    
    //printf("Recurrence computation time for block %d according to device: %f ms\n", iter, elapsed_gram);
    // Apply sigmoid to first block (i=0) BEFORE using it in corrections, we don't want to call apply_all_corrections_kernel with i=0
    int blocks_sigmoid = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_sigmoid_kernel<<<blocks_sigmoid, BLOCK_SIZE>>>(d_correction, total_samples, batch_size, 0);
    
    cudaDeviceSynchronize();
    
    cudaEvent_t start_recurrence, stop_recurrence;
    cudaEventCreate(&start_recurrence);
    cudaEventCreate(&stop_recurrence);

    cudaEventRecord(start_recurrence);
    // Apply corrections for each block i from 1 to s-1
    for (int i = 1; i < s; i++)
    {
        int blocks_corr = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        apply_all_corrections_kernel<<<blocks_corr, BLOCK_SIZE>>>(d_G, d_correction, total_samples, batch_size, i, eta);
        cudaDeviceSynchronize();

        // Apply sigmoid to current block
        blocks_sigmoid = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        apply_sigmoid_kernel<<<blocks_sigmoid, BLOCK_SIZE>>>(d_correction, total_samples, batch_size, i);
        
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop_recurrence);
    
    
    float elapsed_recurrence;
    cudaEventElapsedTime(&elapsed_recurrence, start_recurrence, stop_recurrence);
    stats->recurrence_time[iter] = elapsed_recurrence;   
  
    cudaEventDestroy(start_recurrence);
    cudaEventDestroy(stop_recurrence);
    
    // Compute final gradient = -A_scaled' * correction
    int blocks_grad = (n_features + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_final_gradient_kernel<<<blocks_grad, BLOCK_SIZE>>>(d_X_scaled, d_correction, grad, total_samples, n_features);
    cudaDeviceSynchronize();

    // Free temporary memory
    cudaFree(d_X_scaled);
    cudaFree(d_correction);
    cudaFree(d_G);
}

extern "C" void update_weights(float *x, float *grad, float lr, int n_features, int batch_size)
{
    int blocks_weights = (n_features + BLOCK_SIZE - 1) / BLOCK_SIZE;
    update_weights_kernel<<<blocks_weights, BLOCK_SIZE>>>(x, grad, lr, n_features, batch_size);
    cudaDeviceSynchronize();
}
