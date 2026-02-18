#include <vector>
#pragma once

// CUDA wrapper functions callable from host
extern "C" void compute_gradient(float* X, float* y, float* w, float* grad, int batch_size, int n_features);
extern "C" void update_weights(float* w, float* grad, float lr, int n_features, int batch_size);

// Optional: train function declaration if needed elsewhere
void train(
    float* d_X,
    float* d_y,
    float* d_w,
    int n_samples,
    int n_features,
    int batch_size,
    int maxiters,
    float lr,
    const std::vector<float>& hX,
    const std::vector<float>& hy,
    int print_interval);
