#include <vector>
#include <memory>
#include <cublas_v2.h>
#pragma once

struct ProfileStats {
    double init_corr_time = 0.0f;
    double gram_time = 0.0f;
    double recurrence_time = 0.0f;
    double grad_proj_time = 0.0f;
    double weight_update_time = 0.0f;
    double scaling_time = 0.0f;
};

// CUDA wrapper functions callable from host
void compute_gradient(float* A, float* y, float* x, float* grad, int batch_size, int n_features);
void compute_sstep_gradient(float* A, float* y, float* x, float* grad, int batch_size, int s, float eta, int n_features, int iter, ProfileStats* stats, int num_blocks);
void update_weights(float* x, float* grad, float lr, int n_features, int batch_size);
void cuda_scale_A_by_y(cublasHandle_t handle, float *A, float *y, float *d_A_scaled, int total_samples, int n_features);
void cuda_compute_Ax(float *A_scaled, float *x, float *correction, int total_samples, int n_features);
void cuda_compute_Gram(float *A_scaled, float *G, int total_samples, int n_features, int batch_size, int num_blocks, double& gram_time);
void cuda_apply_sigmoid_block(float *correction, int total_samples, int batch_size, int block_idx);
void cuda_apply_all_corrections(float *G, float *correction, int total_samples, int batch_size, int i_block, float eta);
void cuda_compute_final_gradient(float *A_scaled, float *correction, float *grad, int total_samples, int n_features);

// Optional: train function declaration if needed elsewhere
void train(
    float* d_A,
    float* d_y,
    float* d_x,
    int n_samples,
    int n_features,
    int batch_size,
    int s,
    int maxiters,
    float lr,
    const std::vector<float>& hA,
    const std::vector<float>& hy,
    int print_interval,
    ProfileStats* stats
);
