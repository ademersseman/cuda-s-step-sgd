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
#include <iomanip>

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

void enter_recurrence(
    const DataParams *data_params,
    const RunParams *s_step_params,
    Workspace *workspace,
    float *d_batch_A, 
    ProfileStats *run_stats) 
{    
    float alpha = 1;
    float beta = 0;
        
    CudaRegionTimer corr_timer;
    corr_timer.begin();
    
    // Compute initial correction = A_scaled * x
    cublasSgemv(workspace->handle, CUBLAS_OP_T, data_params->n_features, s_step_params->samples_per_iter, &alpha, d_batch_A, data_params->n_features, workspace->d_x, 1, &beta, workspace->d_correction, 1);
    
    run_stats->init_corr_time += corr_timer.end();
            
    CudaRegionTimer recurrence_timer;
    recurrence_timer.begin();
           
    // Apply corrections for each block i from 1 to s-1
    for (int i = 0; i < s_step_params->s; i++) {
        int i_start = i * s_step_params->batch_size;
        float beta = 1.0f;
        for (int j = 0; j < i; j++)
        {
            int j_start = j * s_step_params->batch_size;
            float* subG = workspace->d_G[workspace->compute_buf] + i_start * s_step_params->samples_per_iter + j_start;
            float* corr_j = workspace->d_correction + j_start;
            float* corr_curr = workspace->d_correction + i_start;
            cublasSgemv(workspace->handle, CUBLAS_OP_T, s_step_params->batch_size, s_step_params->batch_size, &s_step_params->eta, subG, s_step_params->samples_per_iter, corr_j, 1, &beta, corr_curr, 1);
        }
        cuda_apply_sigmoid_block(workspace->compute_stream, workspace->d_correction, s_step_params->samples_per_iter, s_step_params->batch_size, i);
    }
  
    run_stats->recurrence_time += recurrence_timer.end();

    CudaRegionTimer grad_proj_timer;
    grad_proj_timer.begin();

    float negalpha = -1.0f;
    
    // Compute final gradient = -A_scaled' * correction
    cublasSgemv(
        workspace->handle,
        CUBLAS_OP_N,
        data_params->n_features,
        s_step_params->samples_per_iter,
        &negalpha,
        d_batch_A,
        data_params->n_features,
        workspace->d_correction,
        1,
        &beta,
        workspace->d_grad,
        1
    );

    run_stats->grad_proj_time += grad_proj_timer.end();
}

void launch_full_prefetch(
    const DataParams* data_params,
    const RunParams* s_step_params,
    Workspace* workspace,
    float* d_batch_A,
    int target_buf)
{
    cudaEventRecord(workspace->gram_overhead_prefetch_done, workspace->prefetch_stream);

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(
        workspace->prefetch_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        s_step_params->samples_per_iter,
        s_step_params->samples_per_iter,
        data_params->n_features,
        &alpha,
        d_batch_A, data_params->n_features,
        d_batch_A, data_params->n_features,
        &beta,
        workspace->d_G[target_buf], s_step_params->samples_per_iter);

    cudaEventRecord(workspace->gram_prefetch_done, workspace->prefetch_stream);
}

void launch_uniform_prefetch(
    const DataParams* data_params,
    const RunParams* s_step_params,
    Workspace* workspace,
    float* d_batch_A,
    int target_buf)
{
    for (int i = 0; i < s_step_params->approx_gram_l; i++) {
        int src_col = rand() % data_params->n_features;
        cublasScopy(workspace->prefetch_handle,
            s_step_params->samples_per_iter,
            d_batch_A + src_col,                       data_params->n_features,
            workspace->d_batch_A_approx[target_buf] + i, s_step_params->approx_gram_l);
    }
    // signal that this buffer is ready
    cudaEventRecord(workspace->gram_overhead_prefetch_done, workspace->prefetch_stream);

    // sgemm on same stream — won't start until all Scopy calls above finish
    float alpha_approx = s_step_params->approx_gram_l / (float)data_params->n_features;
    float beta = 0.0f;
    cublasSgemm(
        workspace->prefetch_handle,       // same handle = same stream
        CUBLAS_OP_T, CUBLAS_OP_N,
        s_step_params->samples_per_iter,
        s_step_params->samples_per_iter,
        s_step_params->approx_gram_l,
        &alpha_approx,
        workspace->d_batch_A_approx[target_buf], s_step_params->approx_gram_l,
        workspace->d_batch_A_approx[target_buf], s_step_params->approx_gram_l,
        &beta,
        workspace->d_G[target_buf], s_step_params->samples_per_iter);

    // single event after gram is done
    cudaEventRecord(workspace->gram_prefetch_done, workspace->prefetch_stream);
}

void launch_scoring_prefetch(
    const DataParams* data_params,
    const RunParams* s_step_params,
    Workspace* workspace,
    float* d_batch_A,
    int target_buf,
    int iters)
{
    iters = iters % (data_params->total_samples / s_step_params->samples_per_iter);
    if (!workspace->cache_valid[iters]) {
        for (int i = 0; i < data_params->n_features; i++) {
            cublasSdot(workspace->prefetch_handle,
                s_step_params->samples_per_iter,
                d_batch_A + i, data_params->n_features,
                d_batch_A + i, data_params->n_features,
                workspace->d_scores + i);   // device pointer — no sync per call
        }
   
        cudaStreamSynchronize(workspace->prefetch_stream);
        cudaMemcpy(workspace->score_cache[iters].data(), workspace->d_scores, data_params->n_features * sizeof(float), cudaMemcpyDeviceToHost);
    
        // normalize on CPU
        float sum = std::accumulate(workspace->score_cache[iters].begin(), workspace->score_cache[iters].end(), 0.0f);
        for (auto& s : workspace->score_cache[iters])
            s /= sum;
        
        workspace->cache_valid[iters] = true;
    }
    std::vector<int> sampled_cols = sample_columns_with_replacement(workspace->score_cache[iters], s_step_params->approx_gram_l);

    
    // now launch async on prefetch stream
    for (int i = 0; i < s_step_params->approx_gram_l; i++) {
        cublasScopy(workspace->prefetch_handle,
            s_step_params->samples_per_iter,
            d_batch_A + sampled_cols[i],          data_params->n_features,
            workspace->d_batch_A_approx[target_buf] + i,  s_step_params->approx_gram_l);
    }

    cudaEventRecord(workspace->gram_overhead_prefetch_done, workspace->prefetch_stream);
        
    float alpha_approx = s_step_params->approx_gram_l / (float)data_params->n_features;
    float beta = 0.0f;
    cublasSgemm(
        workspace->prefetch_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        s_step_params->samples_per_iter,
        s_step_params->samples_per_iter,
        s_step_params->approx_gram_l,
        &alpha_approx,
        workspace->d_batch_A_approx[target_buf], s_step_params->approx_gram_l,
        workspace->d_batch_A_approx[target_buf], s_step_params->approx_gram_l,
        &beta,
        workspace->d_G[target_buf], s_step_params->samples_per_iter);

    cudaEventRecord(workspace->gram_prefetch_done, workspace->prefetch_stream);
}

// ---------------- Train Function ----------------
void train(
    const DataParams* data_params,
    const RunParams* s_step_params,
    Workspace* workspace,
    const std::vector<float>& h_A,
    const std::vector<float>& h_y,
    ProfileStats* run_stats)
{
    // initial metrics
    std::vector<float> h_x(data_params->n_features, 0.0f);
    double prev_obj = 0.0;
    double cur_obj = 0.0;
    double cur_acc = 0.0;
    compute_metrics(data_params, h_A, h_y, h_x, prev_obj, cur_acc);

    float negEta = -s_step_params->eta;

    CudaRegionTimer scaling_timer;
    scaling_timer.begin();

    // Scale A by y
    cublasSdgmm(workspace->handle, CUBLAS_SIDE_RIGHT, data_params->n_features, data_params->total_samples_unpadded, workspace->d_A, data_params->n_features, workspace->d_y, 1, workspace->d_A_scaled, data_params->n_features);

    run_stats->scaling_time += scaling_timer.end();

    if (s_step_params->approx_gram && s_step_params->approx_gram_type == "uniform")
        launch_uniform_prefetch(data_params, s_step_params, workspace, workspace->d_A_scaled, workspace->prefetch_buf);
    else if (s_step_params->approx_gram && s_step_params->approx_gram_type == "scoring")
        launch_scoring_prefetch(data_params, s_step_params, workspace, workspace->d_A_scaled, workspace->prefetch_buf, 0);
    else
        launch_full_prefetch(data_params, s_step_params, workspace, workspace->d_A_scaled, workspace->prefetch_buf);

    for (int iters = 0; iters <= s_step_params->maxiters; iters++) {
        // determine current batch start offset (wrap around if we exceed total samples)
        int curr_batch_start_offset = ((iters * s_step_params->batch_size * s_step_params->s) % data_params->total_samples);
        float* d_batch_A = workspace->d_A_scaled + curr_batch_start_offset * data_params->n_features;

        CudaRegionTimer iter_timer;
        iter_timer.begin();

        // wait for current iter prefetch to finish 
        CudaRegionTimer gram_overhead_timer;
        gram_overhead_timer.begin();
        
        cudaStreamWaitEvent(workspace->compute_stream, workspace->gram_overhead_prefetch_done, 0);
        //cudaEventSynchronize(workspace->gram_overhead_prefetch_done);
        
        run_stats->gram_overhead_time += gram_overhead_timer.end();
        
        CudaRegionTimer gram_compute_timer;
        gram_compute_timer.begin();

        cudaStreamWaitEvent(workspace->compute_stream, workspace->gram_prefetch_done, 0);
        // cudaEventSynchronize(workspace->gram_prefetch_done);

        run_stats->gram_compute_time += gram_compute_timer.end();

        // launch next iteration's prefetch
        // compute reads from prefetch_buf, prefetch writes into the other one
        workspace->compute_buf  =  workspace->prefetch_buf;
        int next_buf            = (workspace->prefetch_buf + 1) % 2;

        // next batch pointer for prefetch
        int next_offset = ((iters + 1) * s_step_params->samples_per_iter) % data_params->total_samples;
        float* d_next_batch_A = workspace->d_A_scaled + next_offset * data_params->n_features;

        if (s_step_params->approx_gram && s_step_params->approx_gram_type == "uniform")
            launch_uniform_prefetch(data_params, s_step_params, workspace, d_next_batch_A, next_buf);
        else if (s_step_params->approx_gram && s_step_params->approx_gram_type == "scoring")
            launch_scoring_prefetch(data_params, s_step_params, workspace, d_next_batch_A, next_buf, iters + 1);
        else
            launch_full_prefetch(data_params, s_step_params, workspace, d_next_batch_A, next_buf);
        
        enter_recurrence(data_params, s_step_params, workspace, d_batch_A, run_stats);

        CudaRegionTimer weight_update_timer;
        weight_update_timer.begin();
        
        // update weights: x = x - lr * grad
        cublasSaxpy(workspace->handle, data_params->n_features, &negEta, workspace->d_grad, 1, workspace->d_x, 1);
        
        run_stats->weight_update_time += weight_update_timer.end();

        // alternate buffers for next iteration
        workspace->prefetch_buf = next_buf;

        if (iters != 0 && iters % s_step_params->printerval == 0) {
            // copy weights and compute metrics
            cudaMemcpy(h_x.data(), workspace->d_x, data_params->n_features * sizeof(float), cudaMemcpyDeviceToHost);
            compute_metrics(data_params, h_A, h_y, h_x, cur_obj, cur_acc);
            printf("Iters: %d\t Objective: %.4f\t Training Accuracy: %.4f%%\t Obj val diff: %1.10e\t Time: %.4f\n",
                   iters, cur_obj, cur_acc, fabs(prev_obj - cur_obj), iter_timer.end());
            prev_obj = cur_obj;
        }
    }
}

// ---------------- Main ----------------
int main(
    int argc, 
    char** argv) 
{
    std::unique_ptr<RunParams> s_step_params = std::make_unique<RunParams>();
    std::unique_ptr<DataParams> data_params = std::make_unique<DataParams>();
    
    // Command-line arguments:
    // [batch_size] [s] [epochs] [training set file name] [approx type] [l]
    if (argc > 1) {
        if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
            std::cout << "Usage: " << argv[0] << " [batch_size] [s] [epochs] [training set file name] [approx type] [l]\n";
            std::cout << "  batch_size   : number of samples per minibatch (default 16)\n";
            std::cout << "  s            : number of minibatches to process before updating weights (default 4)\n";
            std::cout << "  epochs       : number of passes over dataset to process (default 1)\n";
            std::cout << "  training set : file name of training dataset (default 'w1a.txt')\n";
            std::cout << "  approx type  : type of approximate Gram matrix ('uniform' or 'scoring')\n";
            std::cout << "  l            : number of columns to sample for approximate Gram matrix (default 64)\n";
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

    // load raw data into host
    std::vector<float> h_A, h_y;
    load_libsvm(data_params.get(), h_A, h_y);
    data_params->total_samples_unpadded = h_y.size();
    
    // Pad data to be multiple of s * batch_size
    s_step_params->samples_per_iter = s_step_params->s * s_step_params->batch_size;
    // calculate extra samples(needed to pad end of dataset to make it divisible by samples_per_iter)
    int extra_samples = s_step_params->samples_per_iter - (data_params->total_samples_unpadded % s_step_params->samples_per_iter);
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

    // calculate number of iterations(total_samples / s * batch_size) based on epochs and padded total samples
    s_step_params->maxiters = s_step_params->epochs * data_params->total_samples / s_step_params->samples_per_iter;
    s_step_params->printerval = data_params->total_samples / s_step_params->samples_per_iter;
    
    std::unique_ptr<Workspace> workspace = std::make_unique<Workspace>(data_params.get(), s_step_params.get());
    std::unique_ptr<ProfileStats> run_stats = std::make_unique<ProfileStats>();
    
    // copy data to GPU
    cudaMemcpy(workspace->d_A, h_A.data(), data_params->total_samples * data_params->n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(workspace->d_y, h_y.data(), data_params->total_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(workspace->d_x, 0, data_params->n_features * sizeof(float));

    // train
    CudaRegionTimer training_timer;
    training_timer.begin();

    train(data_params.get(), s_step_params.get(), workspace.get(), h_A, h_y, run_stats.get());

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

    return 0;
}
