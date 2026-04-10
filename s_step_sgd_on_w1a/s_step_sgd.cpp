#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "s_step_sgd.h"
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <chrono>
#include <cstdlib>
#include <ctime>

// ---------------- Load LIBSVM file ----------------
void load_libsvm(
    DataParams* data_params,
    std::vector<float> &h_A,
    std::vector<float> &h_y)
{
    std::ifstream file(data_params->file_name);
    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + data_params->file_name);

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);

        float label;
        ss >> label;
        h_y.push_back(label);

        // Reserve space for this row and fill with zeros
        h_A.resize(h_A.size() + data_params->n_features, 0.0f);
        float* row = h_A.data() + h_A.size() - data_params->n_features;

        std::string token;
        while (ss >> token) {
            const auto pos = token.find(':');
            if (pos == std::string::npos) continue;

            const int idx = std::stoi(token.substr(0, pos)) - 1;
            const float val = std::stof(token.substr(pos + 1));
            row[idx] = val;
        }
    }
}
// helper to compute objective and accuracy on host
void compute_metrics(
    const DataParams* data_params,
    const std::vector<float>& h_A,
    const std::vector<float>& h_y,
    const std::vector<float>& h_x,
    double &obj_out,
    double &accuracy_out)
{
    double obj = 0.0;
    double correct = 0;
    for (int i = 0; i < data_params->total_samples_unpadded; i++) {
        // Compute dot product A * x for sample i
        double dot = 0.0;
        for (int k = 0; k < data_params->n_features; k++)
            dot += (double)h_x[k] * (double)h_A[i * data_params->n_features + k];

        obj += log(1.0 + exp(-h_y[i] * dot));

        double prob = 1.0 / (1.0 + exp(-dot));
        float pred = (prob > 0.5) ? 1.0f : -1.0f;
        if (pred == h_y[i])
            ++correct;
    }
    obj_out = obj / (double)data_params->total_samples_unpadded;
    accuracy_out = 100.0 * correct / (double)data_params->total_samples_unpadded;
}


bool compare_with_matlab_weights(
    const std::vector<float>& h_x,
    const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open matlab weights file '" << filename << "'\n";
        return false;
    }
    std::vector<float> matlab;
    float value;
    while (file >> value) {
        matlab.push_back(value);
    }
    file.close();

    if (matlab.empty()) {
        std::cerr << "Error: matlab weights file '" << filename << "' is empty or not formatted as floats.\n";
        return false;
    }

    int n = std::min(h_x.size(), matlab.size());
    float sum_abs_diff = 0.0;
    float max_diff = 0.0;
    float min_diff = std::numeric_limits<float>::infinity();
    float max_matlab = -std::numeric_limits<float>::infinity();
    float min_matlab = std::numeric_limits<float>::infinity();
    float max_ours = -std::numeric_limits<float>::infinity();
    float min_ours = std::numeric_limits<float>::infinity();

    for (int i = 0; i < n; ++i) {
        float diff = std::abs(h_x[i] - matlab[i]);
        sum_abs_diff += diff;
        max_diff = std::max(max_diff, diff);
        min_diff = std::min(min_diff, diff);
        max_matlab = std::max(max_matlab, matlab[i]);
        min_matlab = std::min(min_matlab, matlab[i]);
        max_ours = std::max(max_ours, h_x[i]);
        min_ours = std::min(min_ours, h_x[i]);
    }

    float avg_diff = sum_abs_diff / n;
    float range_matlab = max_matlab - min_matlab;
    float range_ours = max_ours - min_ours;

    std::cout << "=== Weight comparison with " << filename << " ===\n";
    std::cout << "Matched entries: " << n << " (matlab file length: " << matlab.size() << ", ours length: " << h_x.size() << ")\n";
    std::cout << "Matlab range: [" << min_matlab << ", " << max_matlab << "] (" << range_matlab << ")\n";
    std::cout << "Our range:    [" << min_ours << ", " << max_ours << "] (" << range_ours << ")\n";
    std::cout << "Average absolute difference: " << avg_diff << "\n";
    std::cout << "Max absolute difference: " << max_diff << "\n";
    std::cout << "Min absolute difference: " << min_diff << "\n";

    if (matlab.size() != h_x.size()) {
        std::cout << "Note: size mismatch; only compared first " << n << " elements.";
    }

    return true;
}

void compute_sstep_gradient(
    const DataParams* data_params,
    const RunParams* s_step_params, 
    float *d_A, 
    float *d_y, 
    float *d_x, 
    float *d_grad, 
    ProfileStats *run_stats) 
{
    int run_samples = s_step_params->s * s_step_params->batch_size;

    float *d_A_scaled;
    float *d_correction;
    float *d_G;
    cudaMalloc(&d_A_scaled, run_samples * data_params->n_features * sizeof(float));
    cudaMalloc(&d_correction, run_samples * sizeof(float));
    cudaMalloc(&d_G, run_samples * run_samples * sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1;
    float beta = 0;
    
    auto scaling_start = std::chrono::high_resolution_clock::now();

    // Scale A by y
    cublasSdgmm(handle, CUBLAS_SIDE_RIGHT, data_params->n_features, run_samples, d_A, data_params->n_features, d_y, 1, d_A_scaled, data_params->n_features);

    auto scaling_end = std::chrono::high_resolution_clock::now();
    
    run_stats->scaling_time += std::chrono::duration<double>(scaling_end - scaling_start).count();

    auto corr_start = std::chrono::high_resolution_clock::now();
    
    // Compute initial correction = A_scaled * x
    cublasSgemv(handle, CUBLAS_OP_T, data_params->n_features, run_samples, &alpha, d_A_scaled, data_params->n_features, d_x, 1, &beta, d_correction, 1);
    
    auto corr_end = std::chrono::high_resolution_clock::now();
    
    run_stats->init_corr_time += std::chrono::duration<double>(corr_end - corr_start).count();
    
    float *d_A_scaled_sub;
    auto gram_start = std::chrono::high_resolution_clock::now();
    if (s_step_params->approx_gram) {
        // Approximate Gram matrix using sampling (from Drineas et al. paper)
        int l = 50; // number of sampled features
        cudaMalloc(&d_A_scaled_sub, run_samples * l * sizeof(float));
        
        for(int i = 0; i < l; i++) {
            int src_col = rand() % data_params->n_features;
            cublasScopy(handle,
                run_samples,
                d_A_scaled + src_col, data_params->n_features,
                d_A_scaled_sub + i, l);
        }
            
        // Compute approximate Gram matrix G_hat = (n_features / l) * A_sub * A_sub^T
        // Both matrices are row-major (total_samples x l).
        float alpha_approx = data_params->n_features / (float)l;
        cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            run_samples,
            run_samples,
            l,
            &alpha_approx,
            d_A_scaled_sub, l,
            d_A_scaled_sub, l,
            &beta,
            d_G, run_samples
            );
    } else
    {
        // Compute full Gram matrix G = A_scaled * A_scaled'
        cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            run_samples,
            run_samples,
            data_params->n_features,
            &alpha,
            d_A_scaled, data_params->n_features,
            d_A_scaled, data_params->n_features,
            &beta,
            d_G, run_samples
            );
    }
    auto gram_end = std::chrono::high_resolution_clock::now();
    run_stats->gram_time += std::chrono::duration<double>(gram_end - gram_start).count();
        
    auto recurrence_start = std::chrono::high_resolution_clock::now();
           
    // Apply corrections for each block i from 1 to s-1
    for (int i = 0; i < s_step_params->s; i++) {
        int i_start = i * s_step_params->batch_size;
        float beta = 1.0f;
        for (int j = 0; j < i; j++)
        {
            int j_start = j * s_step_params->batch_size;
            float* subG = d_G + i_start * run_samples + j_start;
            float* corr_j = d_correction + j_start;
            float* corr_curr = d_correction + i_start;
            cublasSgemv(handle, CUBLAS_OP_T, s_step_params->batch_size, s_step_params->batch_size, &s_step_params->eta, subG, run_samples, corr_j, 1, &beta, corr_curr, 1);
        }
        cuda_apply_sigmoid_block(d_correction, run_samples, s_step_params->batch_size, i);
    }
  
    auto recurrence_end = std::chrono::high_resolution_clock::now();

    run_stats->recurrence_time += std::chrono::duration<double>(recurrence_end - recurrence_start).count();

    auto grad_proj_start = std::chrono::high_resolution_clock::now();

    float negalpha = -1.0f;
    
    // Compute final gradient = -A_scaled' * correction
    cublasSgemv(
        handle,
        CUBLAS_OP_N,
        data_params->n_features,
        run_samples,
        &negalpha,
        d_A_scaled,
        data_params->n_features,
        d_correction,
        1,
        &beta,
        d_grad,
        1
    );

    auto grad_proj_end = std::chrono::high_resolution_clock::now();

    run_stats->grad_proj_time += std::chrono::duration<double>(grad_proj_end - grad_proj_start).count();

    cublasDestroy(handle);
    cudaFree(d_A_scaled);
    cudaFree(d_correction);
    cudaFree(d_G);
    cudaFree(d_A_scaled_sub);
}

// ---------------- Train Function ----------------
void train(
    const DataParams* data_params,
    const RunParams* s_step_params,
    float* d_A,
    float* d_y,
    float* d_x,
    const std::vector<float>& h_A,
    const std::vector<float>& h_y,
    ProfileStats* run_stats)
{
    // gradient buffer
    float* d_grad;
    cudaMalloc(&d_grad, data_params->n_features * sizeof(float));

    // initial metrics
    std::vector<float> h_x(data_params->n_features, 0.0f);
    double prev_obj = 0.0;
    double cur_obj = 0.0;
    double cur_acc = 0.0;
    compute_metrics(data_params, h_A, h_y, h_x, prev_obj, cur_acc);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float negEta = -s_step_params->eta;

    for (int iters = 0; iters <= s_step_params->maxiters; iters += s_step_params->s) {
        float* batch_A = d_A + ((iters * s_step_params->batch_size) % data_params->total_samples) * data_params->n_features;
        float* batch_y = d_y + ((iters * s_step_params->batch_size) % data_params->total_samples);

        cudaMemset(d_grad, 0, data_params->n_features * sizeof(float));
        compute_sstep_gradient(data_params, s_step_params, batch_A, batch_y, d_x, d_grad, run_stats);

        auto weight_update_start = std::chrono::high_resolution_clock::now();

        // Update weights: x = x - lr * grad
        cublasSaxpy(handle, data_params->n_features, &negEta, d_grad, 1, d_x, 1);
        
        auto weight_update_end = std::chrono::high_resolution_clock::now();
        
        run_stats->weight_update_time += std::chrono::duration<double>(weight_update_end - weight_update_start).count();

        if (s_step_params->printerval > 0 && (iters % s_step_params->printerval) == 0) {
            // copy weights and compute metrics
            cudaMemcpy(h_x.data(), d_x, data_params->n_features * sizeof(float), cudaMemcpyDeviceToHost);
            compute_metrics(data_params, h_A, h_y, h_x, cur_obj, cur_acc);
            printf("Iters: %d\t Objective: %.8f\t Training Accuracy: %.4f%%\t Obj val diff %1.15e.\n",
                   iters, cur_obj, cur_acc, fabs(prev_obj - cur_obj));
            prev_obj = cur_obj;
        }
    }
    cublasDestroy(handle);
    cudaFree(d_grad);

    // Copy back weights into h_x for comparison
    cudaMemcpy(h_x.data(), d_x, data_params->n_features * sizeof(float), cudaMemcpyDeviceToHost);
    // Compare final weights to MATLAB reference file
    compare_with_matlab_weights(h_x, "matlab_w1a.txt");

    std::cout << "\n=== Timing Breakdown ===\n";
    std::cout << "Initialization + Correction Time: " << run_stats->init_corr_time << " seconds\n";
    std::cout << "Gram Matrix Time: " << run_stats->gram_time << " seconds\n";
    std::cout << "Recurrence Time: " << run_stats->recurrence_time << " seconds\n";
    std::cout << "Gradient Projection Time: " << run_stats->grad_proj_time << " seconds\n";
    std::cout << "Weight Update Time: " << run_stats->weight_update_time << " seconds\n";
    std::cout << "Scaling Time: " << run_stats->scaling_time << " seconds\n\n";
}

// ---------------- Main ----------------
int main(
    int argc, 
    char** argv) 
{
    std::unique_ptr<RunParams> s_step_params = std::make_unique<RunParams>();
    // samples per minibatch(we process s minibatches per iteration)
    // s = how many SGD calculations we perform ahead of time(before updating weights)
    s_step_params->maxiters = 15360;
    s_step_params->printerval = 15360;
    s_step_params->batch_size = 4*16;
    s_step_params->s = 4*16;
    s_step_params->eta = 0.5f;
    // full gram vs approximate gram at 1x is 30x faster
    // full gram vs approximate gram at 4x is 14x faster
    // full gram vs approximate gram at 8x is 13x faster
    // full gram vs approximate gram at 12x is 18x faster
    
    // Command-line arguments:
    // [batch_size] [s]
    if (argc > 1) {
        if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
            std::cout << "Usage: " << argv[0] << " [batch_size] [s]\n";
            std::cout << "  batch_size  : number of samples per minibatch (default 16)\n";
            std::cout << "  s           : number of minibatches to process before updating weights (default 16)\n";
            return 0;
        }
        s_step_params->batch_size = std::max(1, std::atoi(argv[1]));
    }
    if (argc > 2) {
        s_step_params->s = std::max(1, std::atoi(argv[2]));
    }
    srand(time(NULL));
    
    // load raw data
    std::unique_ptr<DataParams> data_params = std::make_unique<DataParams>();
    data_params->n_features = 300;
    data_params->file_name = "w8a.txt";
    std::vector<float> h_A, h_y;
    load_libsvm(data_params.get(), h_A, h_y);
    data_params->total_samples_unpadded = h_y.size();
    
    // Pad data to be multiple of s * batch_size
    // samples per iteration
    int samples_per_iter = s_step_params->s * s_step_params->batch_size;
    // calculate extra samples(needed to pad end of dataset to make it divisible by samples_per_iter)
    int extra_samples = samples_per_iter - (data_params->total_samples_unpadded % samples_per_iter);
    for (int i = 0; i < extra_samples; i++) {
        h_y.push_back(0.0f);
        for (int j = 0; j < data_params->n_features; j++)
        h_A.push_back(0.0f);
    }
    data_params->total_samples = h_y.size();

    if (s_step_params->batch_size * s_step_params->s > data_params->total_samples) {
        std::cerr << "Error: batch_size * s must be <= total_samples\n";
        return 1;
    }
    
    // Allocate and copy data to GPU
    float *d_A, *d_y, *d_x;
    cudaMalloc(&d_A, data_params->total_samples * data_params->n_features * sizeof(float));
    cudaMalloc(&d_y, data_params->total_samples * sizeof(float));
    cudaMalloc(&d_x, data_params->n_features * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), data_params->total_samples * data_params->n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), data_params->total_samples * sizeof(float), cudaMemcpyHostToDevice);
    // set initial weights to zero
    cudaMemset(d_x, 0, data_params->n_features * sizeof(float));

    std::unique_ptr<ProfileStats> run_stats = std::make_unique<ProfileStats>();

    train(data_params.get(), s_step_params.get(), d_A, d_y, d_x, h_A, h_y, run_stats.get());

    // reset and run again with approximate gram
    cudaMemset(d_x, 0, data_params->n_features * sizeof(float));
    std::unique_ptr<ProfileStats> run_stats_approx = std::make_unique<ProfileStats>();
    s_step_params->approx_gram = true;
    train(data_params.get(), s_step_params.get(), d_A, d_y, d_x, h_A, h_y, run_stats_approx.get());

    cudaFree(d_A);
    cudaFree(d_y);
    cudaFree(d_x);

    std::cout << "Press Enter to exit..." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    return 0;
}
