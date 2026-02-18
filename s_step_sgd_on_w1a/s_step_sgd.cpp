#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include "s_step_sgd.h"
#include <cmath>

// ---------------- Load LIBSVM file ----------------
void load_libsvm(
    const std::string &filename,
    int n_features,
    std::vector<float> &X,
    std::vector<float> &y)
{
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        int label;
        ss >> label;
        y.push_back(label > 0 ? 1.0f : -1.0f);

        std::vector<float> row(n_features, 0.0f);
        std::string token;
        while (ss >> token) {
            auto pos = token.find(':');
            if (pos != std::string::npos) {
                int idx = std::stoi(token.substr(0, pos)) - 1;
                float val = std::stof(token.substr(pos + 1));
                if (idx >= 0 && idx < n_features)
                    row[idx] = val;
            }
        }
        for (int i = 0; i < n_features; i++)
            X.push_back(row[i]);
    }
}

// ---------------- Train Function ----------------
void train(
    float* d_X,
    float* d_y,
    float* d_w,
    int n_samples,
    int n_features,
    int batch_size,
    int s,
    int maxiters,
    float lr,
    const std::vector<float>& hX,
    const std::vector<float>& hy,
    int print_interval)
{
    float* d_grad;
    cudaMalloc(&d_grad, n_features * sizeof(float));

    int total_batch_size = s * batch_size;
    int num_batches = n_samples / total_batch_size;

    // helper to compute objective and accuracy on host
    auto compute_metrics = [&](const std::vector<float>& X, const std::vector<float>& y, const std::vector<float>& w,
                               int n_samples_local, int n_features_local, int &m_unpadded,
                               double &obj_out, double &accuracy_out) {
        // find unpadded length (first zero label)
        m_unpadded = n_samples_local;
        for (int i = 0; i < n_samples_local; ++i) {
            if (y[i] == 0.0f) { m_unpadded = i; break; }
        }
        if (m_unpadded == 0) m_unpadded = n_samples_local;

        double obj = 0.0;
        int correct = 0;
        for (int i = 0; i < m_unpadded; ++i) {
            double dot = 0.0;
            const float yi = y[i];
            for (int k = 0; k < n_features_local; ++k)
                dot += (double)w[k] * (double)X[i*n_features_local + k];

            obj += log(1.0 + exp(-yi * dot));

            double prob = 1.0 / (1.0 + exp(-dot));
            int pred = (prob > 0.5) ? 1 : -1;
            if (pred == (int)yi) ++correct;
        }
        obj_out = obj / (double)m_unpadded;
        accuracy_out = 100.0 * (double)correct / (double)m_unpadded;
    };

    // initial metrics (weights may be zero-initialized)
    std::vector<float> hW(n_features);
    cudaMemcpy(hW.data(), d_w, n_features * sizeof(float), cudaMemcpyDeviceToHost);
    int m_unpadded = n_samples;
    double prev_obj = 0.0;
    double cur_obj = 0.0;
    double cur_acc = 0.0;
    compute_metrics(hX, hy, hW, n_samples, n_features, m_unpadded, prev_obj, cur_acc);

    int iters = 0;
    for (int iter = 0; iter < maxiters; iter++) {

        int b = iter % num_batches;

        float* batch_X = d_X + b * total_batch_size * n_features;
        float* batch_y = d_y + b * total_batch_size;

        cudaMemset(d_grad, 0, n_features*sizeof(float));

        compute_sstep_gradient(batch_X, batch_y, d_w, d_grad, batch_size, s, lr, n_features);
        update_weights(d_w, d_grad, lr, n_features, total_batch_size);

        iters += s;
        if (print_interval > 0 && (iters % print_interval) == 0) {
            // copy weights and compute metrics
            cudaMemcpy(hW.data(), d_w, n_features * sizeof(float), cudaMemcpyDeviceToHost);
            compute_metrics(hX, hy, hW, n_samples, n_features, m_unpadded, cur_obj, cur_acc);
            printf("Iters: %d\t Objective: %.8f\t Training Accuracy: %.4f%%\t Obj val diff %1.15e.\n",
                   iters, cur_obj, cur_acc, fabs(prev_obj - cur_obj));
            prev_obj = cur_obj;
        }
    }
    cudaFree(d_grad);
}

// ---------------- Main ----------------
int main() {
    std::vector<float> hX, hy;
    const std::string fname = "w1a.txt";
    const int n_features = 300;

    load_libsvm(fname, n_features, hX, hy);

    int n_samples = hy.size();

    // Hyperparameters
    int batch_size = 16;
    int s = 16;  // s-step parameter
    int maxiters = 15360 / s;  // Adjust maxiters since each iteration processes s steps
    float lr = 0.5f;
    int print_interval = 512;

    // Pad data to be multiple of s*batch_size
    int total_batch_size = s * batch_size;
    int extra_samples = (total_batch_size - (n_samples % total_batch_size)) % total_batch_size;
    if (extra_samples > 0) {
        for (int i = 0; i < extra_samples; i++) {
            hy.push_back(0.0f);
            for (int j = 0; j < n_features; j++) {
                hX.push_back(0.0f);
            }
        }
        n_samples += extra_samples;
    }

    // Allocate and copy to GPU
    float *dX, *dy, *dw;
    cudaMalloc(&dX, n_samples * n_features * sizeof(float));
    cudaMalloc(&dy, n_samples * sizeof(float));
    cudaMalloc(&dw, n_features * sizeof(float));

    cudaMemcpy(dX, hX.data(), n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy.data(), n_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dw, 0, n_features * sizeof(float));

    train(dX, dy, dw, n_samples, n_features, batch_size, s, maxiters, lr, hX, hy, print_interval);

    // Copy back weights
    std::vector<float> hW(n_features);
    cudaMemcpy(hW.data(), dw, n_features * sizeof(float), cudaMemcpyDeviceToHost);

    // Add this to print weights
    std::cout << "Final weights:" << std::endl;
    for (int i = 0; i < n_features; ++i) {
        std::cout << hW[i] << "\n";
    }
    std::cout << n_features;
    std::cout << std::endl;

    cudaFree(dX);
    cudaFree(dy);
    cudaFree(dw);

    system("pause");
    return 0;
}
