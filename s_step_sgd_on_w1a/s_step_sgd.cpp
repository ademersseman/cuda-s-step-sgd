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

#define BLOCK_SIZE 256

// ---------------- Load LIBSVM file ----------------
void load_libsvm(
    const std::string &filename,
    int n_features,
    std::vector<float> &A,
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
            A.push_back(row[i]);
    }
}

// helper to compute objective and accuracy on host
void compute_metrics(
    const std::vector<float>& A,
    const std::vector<float>& y,
    const std::vector<float>& w,
    int total_samples_local,
    int n_features_local,
    int &m_unpadded,
    double &obj_out,
    double &accuracy_out)
{
    // find unpadded length (first zero label)
    m_unpadded = total_samples_local;
    for (int i = 0; i < total_samples_local; ++i) {
        if (y[i] == 0.0f) {
            m_unpadded = i;
            break; 
        }
    }
    if (m_unpadded == 0) m_unpadded = total_samples_local;

    double obj = 0.0;
    int correct = 0;
    for (int i = 0; i < m_unpadded; ++i) {
        double dot = 0.0;
        const float yi = y[i];
        for (int k = 0; k < n_features_local; ++k)
            dot += (double)w[k] * (double)A[i*n_features_local + k];

        obj += log(1.0 + exp(-yi * dot));

        double prob = 1.0 / (1.0 + exp(-dot));
        int pred = (prob > 0.5) ? 1 : -1;
        if (pred == (int)yi) ++correct;
    }
    obj_out = obj / (double)m_unpadded;
    accuracy_out = 100.0 * (double)correct / (double)m_unpadded;
}


bool compare_with_matlab_weights(const std::vector<float>& weights, const std::string& filename) {
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

    size_t n = std::min(weights.size(), matlab.size());
    double sum_abs_diff = 0.0;
    double max_diff = 0.0;
    double min_diff = std::numeric_limits<double>::infinity();
    double max_matlab = -std::numeric_limits<double>::infinity();
    double min_matlab = std::numeric_limits<double>::infinity();
    double max_ours = -std::numeric_limits<double>::infinity();
    double min_ours = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < n; ++i) {
        double diff = std::abs((double)weights[i] - (double)matlab[i]);
        sum_abs_diff += diff;
        max_diff = std::max(max_diff, diff);
        min_diff = std::min(min_diff, diff);
        max_matlab = std::max(max_matlab, (double)matlab[i]);
        min_matlab = std::min(min_matlab, (double)matlab[i]);
        max_ours = std::max(max_ours, (double)weights[i]);
        min_ours = std::min(min_ours, (double)weights[i]);
    }

    double avg_diff = sum_abs_diff / (double)n;
    double range_matlab = max_matlab - min_matlab;
    double range_ours = max_ours - min_ours;

    std::cout << "=== Weight comparison with " << filename << " ===\n";
    std::cout << "Matched entries: " << n << " (matlab file length: " << matlab.size() << ", ours length: " << weights.size() << ")\n";
    std::cout << "Matlab range: [" << min_matlab << ", " << max_matlab << "] (" << range_matlab << ")\n";
    std::cout << "Our range:    [" << min_ours << ", " << max_ours << "] (" << range_ours << ")\n";
    std::cout << "Average absolute difference: " << avg_diff << "\n";
    std::cout << "Max absolute difference: " << max_diff << "\n";
    std::cout << "Min absolute difference: " << min_diff << "\n";

    if (matlab.size() != weights.size()) {
        std::cout << "Note: size mismatch; only compared first " << n << " elements.\n";
    }

    return true;
}

void compute_sstep_gradient(float *A, float *y, float *x, float *grad, int batch_size, int s, float eta, int n_features, ProfileStats *stats)
{
    int total_samples = s * batch_size;

    float *d_A_scaled;
    float *d_correction;
    float *d_G;
    cudaMalloc(&d_A_scaled, total_samples * n_features * sizeof(float));
    cudaMalloc(&d_correction, total_samples * sizeof(float));
    cudaMalloc(&d_G, total_samples * total_samples * sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1;
    float beta = 0;
    
    double scaling_start = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1000.0;

    // Scale A by y
    cublasSdgmm(handle, CUBLAS_SIDE_RIGHT, n_features, total_samples, A, n_features, y, 1, d_A_scaled, n_features);

    double scaling_end = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1000.0;
    
    stats->scaling_time += scaling_end - scaling_start;

    double corr_start = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1000.0;

    // Compute initial correction = A_scaled * x
    cublasSgemv(handle, CUBLAS_OP_T, n_features, total_samples, &alpha, d_A_scaled, n_features, x, 1, &beta, d_correction, 1);
    
    double corr_end = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1000.0;

    stats->init_corr_time += corr_end - corr_start;
    
    double gram_start = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1000.0;
    // Compute full Gram matrix G = A_scaled * A_scaled'
    /*
    */
    cublasSgemm(
        handle,
        CUBLAS_OP_T,             // op(A) = A^T in cuBLAS = your row-major A
        CUBLAS_OP_N,             // op(B) = B   in cuBLAS = your row-major A^T
        total_samples,           // m: rows of output
        total_samples,           // n: cols of output
        n_features,              // k: contracting dimension
        &alpha,
        d_A_scaled, n_features,           // lda = n_features
        d_A_scaled, n_features,           // ldb = n_features (same matrix)
        &beta,
        d_G, total_samples       // ldc = total_samples
        );
    /*
    // Approximate Gram matrix using sampling (from Drineas et al. paper)
    int l = 30; // number of sampled features
    std::vector<int> selected_features(l);
    for(int i = 0; i < l; i++) {
        selected_features[i] = rand() % n_features;
    }
    
    float *d_A_scaled_sub;
    cudaMalloc(&d_A_scaled_sub, total_samples * l * sizeof(float));
    
    // Use cublasScopy to copy selected columns of d_A_scaled into d_A_scaled_sub
    // d_A_scaled layout: (total_samples x n_features) row-major
    // → column src_col starts at offset src_col, with stride n_features
    // d_A_scaled_sub layout: (total_samples x l) row-major
    // → column i starts at offset i, with stride l
    for(int i = 0; i < l; i++) {
        int src_col = selected_features[i];
        cublasScopy(handle,
        total_samples,
        d_A_scaled + src_col, n_features,  // src: col src_col, stride = n_features
        d_A_scaled_sub + i,   l);          // dst: col i,       stride = l
    }
    
    // Compute approximate Gram matrix G_hat = (n_features / l) * A_sub * A_sub^T
    // Both matrices are row-major (total_samples x l).
    // cuBLAS is col-major, so we compute G = A_sub^T * A_sub in cuBLAS terms,
    // which gives the col-major result equal to the row-major A_sub * A_sub^T.
    float alpha_approx = n_features / (float)l;
    cublasSgemm(
        handle,
        CUBLAS_OP_T,                    // op(A): transpose A_sub (col-major view)
        CUBLAS_OP_N,                    // op(B): no-op on A_sub (col-major view)
        total_samples,                  // m: rows of output
        total_samples,                  // n: cols of output
        l,                              // k: inner dimension
        &alpha_approx,
        d_A_scaled_sub, l,              // A: lda = l (col-major: l rows)
        d_A_scaled_sub, l,              // B: ldb = l
        &beta,
        d_G, total_samples              // C: ldc = total_samples, row-major output
    );
    */
    double gram_end = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1000.0;
        stats->gram_time += gram_end - gram_start;
        
    double recurrence_start = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1000.0;
           
        // Apply corrections for each block i from 1 to s-1
    for (int i = 0; i < s; i++) {
        int i_start = i * batch_size;
        float beta = 1.0f;
        for (int j = 0; j < i; j++)
        {
            int j_start = j * batch_size;
            float* subG = d_G + i_start * total_samples + j_start;
            float* corr_j = d_correction + j_start;
            float* corr_curr = d_correction + i_start;
            cublasSgemv(handle, CUBLAS_OP_T, batch_size, batch_size, &eta, subG, total_samples, corr_j, 1, &beta, corr_curr, 1);
        }
        cuda_apply_sigmoid_block(d_correction, total_samples, batch_size, i);
    }
  
    double recurrence_end = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1000.0;
    stats->recurrence_time += recurrence_end - recurrence_start;

    double grad_proj_start = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1000.0;
    
    float negalpha = -1.0f;
    
    // Compute final gradient = -A_scaled' * correction
    cublasSgemv(
        handle,
        CUBLAS_OP_N,
        n_features,         // rows of A_col
        total_samples,      // cols of A_col
        &negalpha,
        d_A_scaled,
        n_features,
        d_correction,
        1,
        &beta,
        grad,
        1
    );

    double grad_proj_end = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1000.0;
    stats->grad_proj_time += grad_proj_end - grad_proj_start;

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A_scaled);
    cudaFree(d_correction);
    cudaFree(d_G);
    //cudaFree(d_A_scaled_sub);
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
    const std::vector<float>& h_A,
    const std::vector<float>& h_y,
    int printerval,
    ProfileStats* stats)
{
    // gradient buffer
    float* d_grad;
    cudaMalloc(&d_grad, n_features * sizeof(float));

    // initial metrics (weights may be zero-initialized)
    std::vector<float> h_w(n_features);
    cudaMemcpy(h_w.data(), d_x, n_features * sizeof(float), cudaMemcpyDeviceToHost);
    int m_unpadded = total_samples;
    double prev_obj = 0.0;
    double cur_obj = 0.0;
    double cur_acc = 0.0;
    compute_metrics(h_A, h_y, h_w, total_samples, n_features, m_unpadded, prev_obj, cur_acc);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float neglr = -lr;

    for (int iters = 0; iters <= maxiters; iters += s) {
        float* batch_A = d_A + ((iters * batch_size) % total_samples) * n_features;
        float* batch_y = d_y + ((iters * batch_size) % total_samples);

        cudaMemset(d_grad, 0, n_features*sizeof(float));
        compute_sstep_gradient(batch_A, batch_y, d_x, d_grad, batch_size, s, lr, n_features, stats);

        double weight_update_start = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1000.0;
        // Update weights: x = x - lr * grad
        cublasSaxpy(handle, n_features, &neglr, d_grad, 1, d_x, 1);
        
        double weight_update_end = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1000.0;
        
        stats->weight_update_time += weight_update_end - weight_update_start;

        if (printerval > 0 && (iters % printerval) == 0) {
            // copy weights and compute metrics
            cudaMemcpy(h_w.data(), d_x, n_features * sizeof(float), cudaMemcpyDeviceToHost);
            compute_metrics(h_A, h_y, h_w, total_samples, n_features, m_unpadded, cur_obj, cur_acc);
            printf("Iters: %d\t Objective: %.8f\t Training Accuracy: %.4f%%\t Obj val diff %1.15e.\n",
                   iters, cur_obj, cur_acc, fabs(prev_obj - cur_obj));
            prev_obj = cur_obj;
        }
    }
    cublasDestroy(handle);
    cudaFree(d_grad);
}

// ---------------- Main ----------------
int main(int argc, char** argv) {
    // Command-line arguments:
    //   [batch_size] [s]
    // If any argument is omitted, the default value is used.
    int batch_size = 16;
    int s = 16;

    if (argc > 1) {
        if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
            std::cout << "Usage: " << argv[0] << " [batch_size] [s]\n";
            std::cout << "  batch_size  : number of samples per minibatch (default 16)\n";
            std::cout << "  s           : number of minibatches to process before updating weights (default 16)\n";
            return 0;
        }
        batch_size = std::max(1, std::atoi(argv[1]));
    }
    if (argc > 2) {
        s = std::max(1, std::atoi(argv[2]));
    }
    srand(time(NULL));

    // raw data
    // hA each row is a sample(stored contiguously)
    // hy[i] is label for sample i
    std::vector<float> h_A, h_y;
    const std::string fname = "w1a.txt";
    // fixed for dataset
    const int n_features = 300;

    load_libsvm(fname, n_features, h_A, h_y);
    
    // total samples
    int total_samples = h_y.size();

    // samples per minibatch(we process s minibatches per iteration)
    // int batch_size = 256;
    // s = how many SGD calculations we perform ahead of time(before updating weights)
    // how many iterations run(=gradient computations) Adjust maxiters since each iteration processes s steps vs normal SGD
    int maxiters = 15360;
    float lr = 0.5f;
    int printerval = 512;

    // Pad data to be multiple of s * batch_size
    // samples per iteration
    int samples_per_iter = s * batch_size;
    // calculate extra samples(needed to pad end of dataset to make it divisible by samples_per_iter)
    int extra_samples = samples_per_iter - (total_samples % samples_per_iter);
    for (int i = 0; i < extra_samples; i++) {
        h_y.push_back(0.0f);
        for (int j = 0; j < n_features; j++) {
            h_A.push_back(0.0f);
        }
    }
    total_samples += extra_samples;

    // Allocate and copy to GPU
    float *d_A, *d_y, *d_x;
    cudaMalloc(&d_A, total_samples * n_features * sizeof(float));
    cudaMalloc(&d_y, total_samples * sizeof(float));
    cudaMalloc(&d_x, n_features * sizeof(float));
    // copy data to GPU
    cudaMemcpy(d_A, h_A.data(), total_samples * n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), total_samples * sizeof(float), cudaMemcpyHostToDevice);
    // set initial weights to zero
    cudaMemset(d_x, 0, n_features * sizeof(float));

    std::unique_ptr<ProfileStats> hstats = std::make_unique<ProfileStats>();

    train(d_A, d_y, d_x, total_samples, n_features, batch_size, s, maxiters, lr, h_A, h_y, printerval, hstats.get());
    
    // Copy back weights into h_x for printing
    std::vector<float> h_x(n_features);
    cudaMemcpy(h_x.data(), d_x, n_features * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compare final weights to MATLAB reference file
    compare_with_matlab_weights(h_x, "matlab_w1a.txt");
  
    cudaFree(d_A);
    cudaFree(d_y);
    cudaFree(d_x);

    std::cout << "Press Enter to exit..." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    return 0;
}
