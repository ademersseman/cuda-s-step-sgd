#include <vector>
#pragma once

struct ProfileStats {
    float gram_time;
    float init_corr_time;
    float* recurrence_time;
};

// CUDA wrapper functions callable from host
extern "C" void compute_gradient(float* A, float* y, float* x, float* grad, int batch_size, int n_features);
extern "C" void compute_sstep_gradient(float* A, float* y, float* x, float* grad, int batch_size, int s, float eta, int n_features, int iter, ProfileStats* stats);
extern "C" void update_weights(float* x, float* grad, float lr, int n_features, int batch_size);

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
    int print_interval);
