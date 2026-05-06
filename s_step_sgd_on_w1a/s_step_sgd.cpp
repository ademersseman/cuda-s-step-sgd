#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include "s_step_sgd.h"
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
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


// Compute leverage scores (column norms squared) for importance sampling
void compute_column_leverage_scores(
    const DataParams* data_params,
    const float* d_A_scaled,
    int n_samples,
    std::vector<float>& scores)
{
    scores.assign(data_params->n_features, 0.0f);
    std::vector<float> h_A_scaled(n_samples * data_params->n_features);
    cudaMemcpy(h_A_scaled.data(), d_A_scaled, n_samples * data_params->n_features * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute norm squared for each column
    for (int j = 0; j < data_params->n_features; j++) {
        float norm_sq = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            float val = h_A_scaled[i * data_params->n_features + j];
            norm_sq += val * val;
        }
        scores[j] = norm_sq;
    }
    
    // Normalize scores to probabilities
    float sum = std::accumulate(scores.begin(), scores.end(), 0.0f);
    if (sum > 0.0f) {
        for (auto& score : scores) {
            score /= sum;
        }
    }
}

// Sample column indices according to weights
std::vector<int> sample_columns_with_replacement(
    const std::vector<float>& weights,
    int num_samples)
{
    std::vector<int> sampled;
    std::vector<float> cumsum(weights.size());
    
    // Compute cumulative sum
    cumsum[0] = weights[0];
    for (size_t i = 1; i < weights.size(); i++) {
        cumsum[i] = cumsum[i - 1] + weights[i];
    }
    
    // Sample according to weights
    for (int i = 0; i < num_samples; i++) {
        float rand_val = (float)rand() / RAND_MAX;
        for (size_t j = 0; j < cumsum.size(); j++) {
            if (rand_val <= cumsum[j]) {
                sampled.push_back(j);
                break;
            }
        }
    }
    
    return sampled;
}

void compute_sstep_gradient(
    const DataParams *data_params,
    const RunParams *s_step_params, 
    float *d_A, 
    float *d_y, 
    float *d_x, 
    float *d_grad, 
    ProfileStats *run_stats) 
{
    size_t samples_per_iter = s_step_params->s * s_step_params->batch_size;

    float *d_A_scaled;
    float *d_A_scaled_sub;
    float *d_correction;
    float *d_G;
    cudaMalloc(&d_A_scaled, samples_per_iter * data_params->n_features * sizeof(float));
    cudaMalloc(&d_A_scaled_sub, samples_per_iter * s_step_params->approx_gram_l * sizeof(float));
    cudaMalloc(&d_correction, samples_per_iter * sizeof(float));
    cudaMalloc(&d_G, samples_per_iter * samples_per_iter * sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);
    /*
    std::vector<cudaStream_t> streams(s_step_params->approx_gram_l);
    for (int i = 0; i < s_step_params->approx_gram_l; i++)
    cudaStreamCreate(&streams[i]);
    
    // Create one cuBLAS handle per stream
    std::vector<cublasHandle_t> handles(s_step_params->approx_gram_l);
    for (int i = 0; i < s_step_params->approx_gram_l; i++) {
        cublasCreate(&handles[i]);
        cublasSetStream(handles[i], streams[i]);
    }
    */


    float alpha = 1;
    float beta = 0;
    
    CudaRegionTimer scaling_timer;
    scaling_timer.begin();

    // Scale A by y
    cublasSdgmm(handle, CUBLAS_SIDE_RIGHT, data_params->n_features, samples_per_iter, d_A, data_params->n_features, d_y, 1, d_A_scaled, data_params->n_features);

    run_stats->scaling_time += scaling_timer.end();

    /*
    // Launch all copies concurrently  
    CudaRegionTimer gram_overhead_timer;
    gram_overhead_timer.begin();  
    for (int i = 0; i < s_step_params->approx_gram_l; i++) {
        int src_col = rand() % data_params->n_features;
        cublasScopy(handles[i],
        samples_per_iter,
        d_A_scaled + src_col, data_params->n_features,
        d_A_scaled_sub + i,   s_step_params->approx_gram_l);
    }
    */

    CudaRegionTimer corr_timer;
    corr_timer.begin();
    
    // Compute initial correction = A_scaled * x
    cublasSgemv(handle, CUBLAS_OP_T, data_params->n_features, samples_per_iter, &alpha, d_A_scaled, data_params->n_features, d_x, 1, &beta, d_correction, 1);
    
    run_stats->init_corr_time += corr_timer.end();
    
    CudaRegionTimer gram_compute_timer;
    if (s_step_params->approx_gram && s_step_params->approx_gram_type == "uniform") {
        // Approximate Gram matrix using uniform random sampling of columns

        // Compute approximate Gram matrix G_hat = (n_features / l) * A_sub * A_sub^T
        // Both matrices are row-major (total_samples x l).

        CudaRegionTimer gram_overhead_timer;
        gram_overhead_timer.begin();
        
        for(int i = 0; i < s_step_params->approx_gram_l; i++) {
            int src_col = rand() % data_params->n_features;
            cublasScopy(handle,
            samples_per_iter,
            d_A_scaled + src_col, data_params->n_features,
            d_A_scaled_sub + i, s_step_params->approx_gram_l);
        }
        
        
        
        /*
        // Sync all streams before sgemm uses d_A_scaled_sub
        cudaDeviceSynchronize();
        std::vector<float> overhead_times(s_step_params->approx_gram_l);
        for (int i = 0; i < s_step_params->approx_gram_l; i++) {
            overhead_times[i] = overhead_timers[i].end(streams[i]);  // End on default stream
        }
        
        run_stats->gram_overhead_time += *std::max_element(overhead_times.begin(), overhead_times.end()); // gram_overhead_timer.end();
        */
        
        cudaDeviceSynchronize(); // Ensure all copies are done before we proceed to sgemm
        run_stats->gram_overhead_time += gram_overhead_timer.end();

        gram_compute_timer.begin();
        float alpha_approx = s_step_params->approx_gram_l / (float)data_params->n_features;
        cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            samples_per_iter,
            samples_per_iter,
            s_step_params->approx_gram_l,
            &alpha_approx,
            d_A_scaled_sub, s_step_params->approx_gram_l,
            d_A_scaled_sub, s_step_params->approx_gram_l,
            &beta,
            d_G, samples_per_iter
            );
    }
    else if (s_step_params->approx_gram && s_step_params->approx_gram_type == "scoring") { 
        // Approximate Gram matrix using leverage score-based sampling of columns
        
        // Compute leverage scores (column norms squared)
        std::vector<float> leverage_scores;
        compute_column_leverage_scores(data_params, d_A_scaled, samples_per_iter, leverage_scores);
        
        // Sample columns according to leverage scores
        std::vector<int> sampled_cols = sample_columns_with_replacement(leverage_scores, s_step_params->approx_gram_l);
        
        for(int i = 0; i < s_step_params->approx_gram_l; i++) {
            int src_col = sampled_cols[i];
            cublasScopy(handle,
                samples_per_iter,
                d_A_scaled + src_col, data_params->n_features,
                d_A_scaled_sub + i, s_step_params->approx_gram_l);
        }
        
        // Compute approximate Gram matrix G_hat = (n_features / l) * A_sub * A_sub^T
        // Both matrices are row-major (total_samples x l).
        float alpha_approx = s_step_params->approx_gram_l / (float)data_params->n_features;
        cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            samples_per_iter,
            samples_per_iter,
            s_step_params->approx_gram_l,
            &alpha_approx,
            d_A_scaled_sub, s_step_params->approx_gram_l,
            d_A_scaled_sub, s_step_params->approx_gram_l,
            &beta,
            d_G, samples_per_iter
            );
    }
    else
    {
        // Compute full Gram matrix G = A_scaled * A_scaled'
        gram_compute_timer.begin();
        cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            samples_per_iter,
            samples_per_iter,
            data_params->n_features,
            &alpha,
            d_A_scaled, data_params->n_features,
            d_A_scaled, data_params->n_features,
            &beta,
            d_G, samples_per_iter
            );
    }

    run_stats->gram_compute_time += gram_compute_timer.end();
        
    CudaRegionTimer recurrence_timer;
    recurrence_timer.begin();
           
    // Apply corrections for each block i from 1 to s-1
    for (int i = 0; i < s_step_params->s; i++) {
        int i_start = i * s_step_params->batch_size;
        float beta = 1.0f;
        for (int j = 0; j < i; j++)
        {
            int j_start = j * s_step_params->batch_size;
            float* subG = d_G + i_start * samples_per_iter + j_start;
            float* corr_j = d_correction + j_start;
            float* corr_curr = d_correction + i_start;
            cublasSgemv(handle, CUBLAS_OP_T, s_step_params->batch_size, s_step_params->batch_size, &s_step_params->eta, subG, samples_per_iter, corr_j, 1, &beta, corr_curr, 1);
        }
        cuda_apply_sigmoid_block(d_correction, samples_per_iter, s_step_params->batch_size, i);
    }
  
    run_stats->recurrence_time += recurrence_timer.end();

    CudaRegionTimer grad_proj_timer;
    grad_proj_timer.begin();

    float negalpha = -1.0f;
    
    // Compute final gradient = -A_scaled' * correction
    cublasSgemv(
        handle,
        CUBLAS_OP_N,
        data_params->n_features,
        samples_per_iter,
        &negalpha,
        d_A_scaled,
        data_params->n_features,
        d_correction,
        1,
        &beta,
        d_grad,
        1
    );

    run_stats->grad_proj_time += grad_proj_timer.end();

    cublasDestroy(handle);
    for (int i = 0; i < s_step_params->approx_gram_l; i++) {
        cublasDestroy(handles[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_A_scaled);
    cudaFree(d_A_scaled_sub);
    cudaFree(d_correction);
    cudaFree(d_G);
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

    for (int iters = 0; iters <= s_step_params->maxiters; iters++) {
        // determine current batch start offset (wrap around if we exceed total samples)
        int curr_batch_start_offset = ((iters * s_step_params->batch_size * s_step_params->s) % data_params->total_samples);
        float* batch_A = d_A + curr_batch_start_offset * data_params->n_features;
        float* batch_y = d_y + curr_batch_start_offset;

        cudaMemset(d_grad, 0, data_params->n_features * sizeof(float));
        
        CudaRegionTimer iter_timer;
        iter_timer.begin();
        
        compute_sstep_gradient(data_params, s_step_params, batch_A, batch_y, d_x, d_grad, run_stats);

        CudaRegionTimer weight_update_timer;
        weight_update_timer.begin();

        // update weights: x = x - lr * grad
        cublasSaxpy(handle, data_params->n_features, &negEta, d_grad, 1, d_x, 1);
        
        run_stats->weight_update_time += weight_update_timer.end();

        if (iters % s_step_params->printerval == 0) {
            // copy weights and compute metrics
            cudaMemcpy(h_x.data(), d_x, data_params->n_features * sizeof(float), cudaMemcpyDeviceToHost);
            compute_metrics(data_params, h_A, h_y, h_x, cur_obj, cur_acc);
            printf("Iters: %d\t Objective: %.4f\t Training Accuracy: %.4f%%\t Obj val diff: %1.10e\t Time: %.4f\n",
                   iters, cur_obj, cur_acc, fabs(prev_obj - cur_obj), iter_timer.end());
            prev_obj = cur_obj;
        }
    }
    cublasDestroy(handle);
    cudaFree(d_grad);

    // Copy back weights into h_x for comparison
    cudaMemcpy(h_x.data(), d_x, data_params->n_features * sizeof(float), cudaMemcpyDeviceToHost);
}

// ---------------- Main ----------------
int main(
    int argc, 
    char** argv) 
{
    std::unique_ptr<RunParams> s_step_params = std::make_unique<RunParams>();
    std::unique_ptr<DataParams> data_params = std::make_unique<DataParams>();
    
    // Command-line arguments:
    // [batch_size] [s] [training set file name]
    if (argc > 1) {
        if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
            std::cout << "Usage: " << argv[0] << " [batch_size] [s] [training set file name]\n";
            std::cout << "  batch_size   : number of samples per minibatch (default 16)\n";
            std::cout << "  s            : number of minibatches to process before updating weights (default 4)\n";
            std::cout << "  epochs       : number of passes over dataset to process (default 1)\n";
            std::cout << "  training set : file name of training dataset (default 'w1a.txt')\n";
            return 0;
        }
        s_step_params->batch_size = std::max(1, std::atoi(argv[1]));
    }
    if (argc > 2) {
        s_step_params->s = std::max(1, std::atoi(argv[2]));
    }
    if (argc > 3) {
	s_step_params->epochs = std::max(1, std::atoi(argv[3]));
    }
    if (argc > 4) {
        data_params->file_name = argv[4];
    }
    if (argc > 5) {
	s_step_params->approx_gram = true;
        s_step_params->approx_gram_type = argv[5];
    }
    if (argc > 6) {
        s_step_params->approx_gram_l = std::max(1, std::atoi(argv[6]));
    }
    srand(time(NULL));

    std::cout << "batch_size: " << s_step_params->batch_size << ", s: " << s_step_params->s << ", epochs: " << s_step_params->epochs << ", training_set: " << data_params->file_name << ", approx_gram_type: " << s_step_params->approx_gram_type << ", approx_gram_l: " << s_step_params->approx_gram_l << "\n";

    // load raw data
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

    // calculate number of iterations based on epochs and padded total samples
    s_step_params->maxiters = s_step_params->epochs * data_params->total_samples / samples_per_iter;
    s_step_params->printerval = data_params->total_samples / samples_per_iter;

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
    cudaMemset(d_x, 0, data_params->n_features * sizeof(float));

    std::unique_ptr<ProfileStats> run_stats = std::make_unique<ProfileStats>();

    CudaRegionTimer training_timer;
    training_timer.begin();

    train(data_params.get(), s_step_params.get(), d_A, d_y, d_x, h_A, h_y, run_stats.get());

    run_stats->training_time = training_timer.end();

    std::cout << "\n=== Timing Breakdown ===\n";
    std::cout << "Initialization + Correction Time: " << run_stats->init_corr_time << " ms\n";
    std::cout << "Gram Overhead Time: " << run_stats->gram_overhead_time << " ms\n";
    std::cout << "Gram Compute Time: " << run_stats->gram_compute_time << " ms\n";
    std::cout << "Recurrence Time: " << run_stats->recurrence_time << " ms\n";
    std::cout << "Gradient Projection Time: " << run_stats->grad_proj_time << " ms\n";
    std::cout << "Weight Update Time: " << run_stats->weight_update_time << " ms\n";
    std::cout << "Scaling Time: " << run_stats->scaling_time << " ms\n";
    std::cout << "Training Time: " << run_stats->training_time << " ms\n\n";

    cudaFree(d_A);
    cudaFree(d_y);
    cudaFree(d_x);

    return 0;
}
