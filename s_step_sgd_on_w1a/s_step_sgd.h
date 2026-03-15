#include <vector>
#include <memory>
#pragma once

struct ProfileStats {
    float init_corr_time;
    std::unique_ptr<float[]> gram_time;
    std::unique_ptr<float[]> recurrence_time;
};

// CUDA wrapper functions callable from host
void compute_gradient(float* A, float* y, float* x, float* grad, int batch_size, int n_features);
void compute_sstep_gradient(float* A, float* y, float* x, float* grad, int batch_size, int s, float eta, int n_features, int iter, ProfileStats* stats, int num_blocks);
void update_weights(float* x, float* grad, float lr, int n_features, int batch_size);

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
