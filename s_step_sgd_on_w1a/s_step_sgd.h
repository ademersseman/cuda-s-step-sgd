#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <string>
#pragma once

struct ProfileStats {
    double init_corr_time = 0.0f;
    double gram_time = 0.0f;
    double recurrence_time = 0.0f;
    double grad_proj_time = 0.0f;
    double weight_update_time = 0.0f;
    double scaling_time = 0.0f;
};

struct RunParams {
    int batch_size = 16;
    int s = 16;
    int maxiters = 15360;
    float eta = 0.5f;
    int printerval = 512;
    bool approx_gram = false;
};

struct DataParams {
    std::string file_name;
    int n_features;
    int total_samples_unpadded;
    int total_samples;
};

// CUDA wrapper functions callable from host
void compute_sstep_gradient(DataParams* data_params, RunParams* s_step_params, float* A, float* y, float* x, float* grad, ProfileStats* stats);
void cuda_apply_sigmoid_block(float *correction, int total_samples, int batch_size, int block_idx);

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
