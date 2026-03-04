#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include "s_step_sgd.h"
#include <cmath>
#include <limits>

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
                int idA = std::stoi(token.substr(0, pos)) - 1;
                float val = std::stof(token.substr(pos + 1));
                if (idA >= 0 && idA < n_features)
                    row[idA] = val;
            }
        }
        for (int i = 0; i < n_features; i++)
            X.push_back(row[i]);
    }
}

// ---------------- Train Function ----------------
void train(
    float* d_A,
    float* d_y,
    float* d_x,
    int total_samples,
    int n_features,
    int batch_size,
    int s,
    int maxiters,
    float lr,
    const std::vector<float>& hA,
    const std::vector<float>& hy,
    int print_interval)
{
    // gradient buffer
    float* d_grad;
    cudaMalloc(&d_grad, n_features * sizeof(float));

    int samples_per_step = s * batch_size;
    int num_batches = total_samples / samples_per_step;

    // helper to compute objective and accuracy on host
    auto compute_metrics = [&](const std::vector<float>& X, const std::vector<float>& y, const std::vector<float>& w,
                               int total_samples_local, int n_features_local, int &m_unpadded,
                               double &obj_out, double &accuracy_out) {
        // find unpadded length (first zero label)
        m_unpadded = total_samples_local;
        for (int i = 0; i < total_samples_local; ++i) {
            if (y[i] == 0.0f) { m_unpadded = i; break; }
        }
        if (m_unpadded == 0) m_unpadded = total_samples_local;

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
    cudaMemcpy(hW.data(), d_x, n_features * sizeof(float), cudaMemcpyDeviceToHost);
    int m_unpadded = total_samples;
    double prev_obj = 0.0;
    double cur_obj = 0.0;
    double cur_acc = 0.0;
    compute_metrics(hA, hy, hW, total_samples, n_features, m_unpadded, prev_obj, cur_acc);

    int iters = 0;
    for (int iter = 0; iter < maxiters; iter++) {

        int b = iter % num_batches;

        float* batch_X = d_A + b * samples_per_step * n_features;
        float* batch_y = d_y + b * samples_per_step;

        cudaMemset(d_grad, 0, n_features*sizeof(float));

        compute_sstep_gradient(batch_X, batch_y, d_x, d_grad, batch_size, s, lr, n_features);
        update_weights(d_x, d_grad, lr, n_features, samples_per_step);

        iters += s;
        if (print_interval > 0 && (iters % print_interval) == 0) {
            // copy weights and compute metrics
            cudaMemcpy(hW.data(), d_x, n_features * sizeof(float), cudaMemcpyDeviceToHost);
            compute_metrics(hA, hy, hW, total_samples, n_features, m_unpadded, cur_obj, cur_acc);
            printf("Iters: %d\t Objective: %.8f\t Training Accuracy: %.4f%%\t Obj val diff %1.15e.\n",
                   iters, cur_obj, cur_acc, fabs(prev_obj - cur_obj));
            prev_obj = cur_obj;
        }
    }
    cudaFree(d_grad);
}

// ---------------- Main ----------------
int main() {
    // raw data
    // hA each row is a sample(stored contiguously))
    // hy[i] is label for sample i
    std::vector<float> hA, hy;
    const std::string fname = "w1a.txt";
    // fixed for dataset
    const int n_features = 300;

    load_libsvm(fname, n_features, hA, hy);
    
    // total samples
    int total_samples = hy.size();

    // Hyperparameters

    // samples per minibatch(we process s minibatches per iteration)
    int batch_size = 16;
    // s = how many SGD calculations we perform ahead of time(before updating weights)
    int s = 16;
    // how many iterations run(=gradient computations) Adjust maxiters since each iteration processes s steps vs normal SGD
    int maxiters = 15360 / s;
    float lr = 0.5f;
    int print_interval = 512;
    
    // Pad data to be multiple of s * batch_size
    // samples per iteration
    int samples_per_iter = s * batch_size;
    // calculate extra samples(needed to pad end of dataset to make it divisible by samples_per_iter)
    int extra_samples = total_samples % samples_per_iter;
    for (int i = 0; i < extra_samples; i++) {
        hy.push_back(0.0f);
        for (int j = 0; j < n_features; j++) {
            hA.push_back(0.0f);
        }
    }
    total_samples += extra_samples;

    // Allocate and copy to GPU
    float *dA, *dy, *dx;
    cudaMalloc(&dA, total_samples * n_features * sizeof(float));
    cudaMalloc(&dy, total_samples * sizeof(float));
    cudaMalloc(&dx, n_features * sizeof(float));

    cudaMemcpy(dA, hA.data(), total_samples * n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy.data(), total_samples * sizeof(float), cudaMemcpyHostToDevice);
    // set initial weights to zero
    cudaMemset(dx, 0, n_features * sizeof(float));

    train(dA, dy, dx, total_samples, n_features, batch_size, s, maxiters, lr, hA, hy, print_interval);

    // Copy back weights into hW for printing
    std::vector<float> hW(n_features);
    cudaMemcpy(hW.data(), dx, n_features * sizeof(float), cudaMemcpyDeviceToHost);

    // Add this to print weights
    std::cout << "Final weights:" << std::endl;
    for (int i = 0; i < n_features; ++i) {
        std::cout << hW[i] << "\n";
    }
    std::cout << n_features;
    std::cout << std::endl;

    cudaFree(dA);
    cudaFree(dy);
    cudaFree(dx);

    std::cout << "Press Enter to exit..." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    return 0;
}
