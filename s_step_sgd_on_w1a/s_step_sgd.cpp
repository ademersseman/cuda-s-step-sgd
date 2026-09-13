#include "s_step_sgd.h"
#include "libsvm_loader.h"
#include "prefetch.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// helper to compute objective and accuracy on host
void compute_metrics(
    const DataParams* data_params,
    const std::vector<float>& h_A,
    const std::vector<float>& h_y,
    const std::vector<float>& h_x,
    double &obj_out,
    double &accuracy_out)
{
    double obj{0.0};
    double correct{0.0};
    const size_t total_samples{data_params->total_samples_unpadded};
    const size_t n_features{data_params->n_features};

    for (size_t i{0}; i < total_samples; ++i) {
        double dot{0.0};
        for (size_t k{0}; k < n_features; ++k) {
            dot += static_cast<double>(h_x[k]) * static_cast<double>(h_A[i * n_features + k]);
        }

        obj += std::log(1.0 + exp(-h_y[i] * dot));

        const double prob{1.0 / (1.0 + std::exp(-dot))};
        const float pred{(prob > 0.5) ? 1.0f : -1.0f};
        if (pred == h_y[i]) {
            ++correct;
        }
    }

    obj_out = obj / static_cast<double>(total_samples);
    accuracy_out = 100.0 * correct / static_cast<double>(total_samples);
}


void enter_recurrence(
    const DataParams *data_params,
    const RunParams *s_step_params,
    Workspace *workspace,
    float *d_batch_A,
    ProfileStats *run_stats)
{
    float alpha{1};
    float beta{0};

    CudaRegionTimer corr_timer{};
    corr_timer.begin();

    cublasSgemv(
        workspace->handle,
        CUBLAS_OP_T,
        data_params->n_features,
        s_step_params->samples_per_iter,
        &alpha,
        d_batch_A,
        data_params->n_features,
        workspace->d_x,
        1,
        &beta,
        workspace->d_correction,
        1);

    run_stats->init_corr_time += corr_timer.end();

    CudaRegionTimer recurrence_timer{};
    recurrence_timer.begin();

    for (size_t i{0}; i < s_step_params->s; ++i) {
        const size_t i_start{i * s_step_params->batch_size};
        float beta_recurrence{1.0f};
        for (size_t j{0}; j < i; ++j) {
            const size_t j_start{j * s_step_params->batch_size};
            float* subG = workspace->d_G[workspace->compute_buf] + i_start * s_step_params->samples_per_iter + j_start;
            float* corr_j = workspace->d_correction + j_start;
            float* corr_curr = workspace->d_correction + i_start;
            cublasSgemv(
                workspace->handle,
                CUBLAS_OP_T,
                s_step_params->batch_size,
                s_step_params->batch_size,
                &s_step_params->eta,
                subG,
                s_step_params->samples_per_iter,
                corr_j,
                1,
                &beta_recurrence,
                corr_curr,
                1);
        }
        cuda_apply_sigmoid_block(workspace->compute_stream, workspace->d_correction, s_step_params->samples_per_iter, s_step_params->batch_size, i);
    }

    run_stats->recurrence_time += recurrence_timer.end();

    CudaRegionTimer grad_proj_timer{};
    grad_proj_timer.begin();

    const float negalpha{-1.0f};
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
        1);

    run_stats->grad_proj_time += grad_proj_timer.end();
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
    std::vector<float> h_x(data_params->n_features, 0.0f);
    double prev_obj{0.0};
    double cur_obj{0.0};
    double cur_acc{0.0};
    compute_metrics(data_params, h_A, h_y, h_x, prev_obj, cur_acc);

    const float negEta{-s_step_params->eta};

    CudaRegionTimer scaling_timer{};
    scaling_timer.begin();
    cublasSdgmm(
        workspace->handle,
        CUBLAS_SIDE_RIGHT,
        data_params->n_features,
        data_params->total_samples_unpadded,
        workspace->d_A,
        data_params->n_features,
        workspace->d_y,
        1,
        workspace->d_A_scaled,
        data_params->n_features);
    run_stats->scaling_time += scaling_timer.end();

    prefetch_gram(data_params, s_step_params, workspace, workspace->d_A_scaled, workspace->prefetch_buf, 0);

    for (size_t iters{0}; iters < s_step_params->maxiters; ++iters) {
        // determine current batch start offset (wrap around if we exceed total samples)
        const size_t curr_batch_start_offset{((iters * s_step_params->batch_size * s_step_params->s) % data_params->total_samples)};
        float* d_batch_A = workspace->d_A_scaled + curr_batch_start_offset * data_params->n_features;

        CudaRegionTimer iter_timer{};
        iter_timer.begin();

        // wait for current iter prefetch to finish 
        CudaRegionTimer gram_overhead_timer{};
        gram_overhead_timer.begin();
        
        cudaStreamWaitEvent(workspace->compute_stream, workspace->gram_overhead_prefetch_done, 0);
        //cudaEventSynchronize(workspace->gram_overhead_prefetch_done);
        
        run_stats->gram_overhead_time += gram_overhead_timer.end();
        
        CudaRegionTimer gram_compute_timer{};
        gram_compute_timer.begin();

        cudaStreamWaitEvent(workspace->compute_stream, workspace->gram_prefetch_done, 0);
        // cudaEventSynchronize(workspace->gram_prefetch_done);

        run_stats->gram_compute_time += gram_compute_timer.end();

        // launch next iteration's prefetch
        // compute reads from prefetch_buf, prefetch writes into the other one
        workspace->compute_buf = workspace->prefetch_buf;
        const size_t next_buf{(workspace->prefetch_buf + 1) % 2};

        // next batch pointer for prefetch
        const size_t next_offset{((iters + 1) * s_step_params->samples_per_iter) % data_params->total_samples};
        float* d_next_batch_A = workspace->d_A_scaled + next_offset * data_params->n_features;

        prefetch_gram(data_params, s_step_params, workspace, d_next_batch_A, next_buf, iters + 1);

        enter_recurrence(data_params, s_step_params, workspace, d_batch_A, run_stats);

        CudaRegionTimer weight_update_timer{};
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

            std::printf(
                "Iters: %zu\t Objective: %.4f\t Training Accuracy: %.4f%%\t Obj val diff: %1.10e\t Time: %.4f\n",
                iters + 1,
                cur_obj,
                cur_acc,
                std::fabs(prev_obj - cur_obj),
                iter_timer.end());

            prev_obj = cur_obj;
        }
    }
}

// ---------------- Main ----------------
int main(int argc, char** argv) {
    std::unique_ptr<RunParams> s_step_params{std::make_unique<RunParams>()};
    std::unique_ptr<DataParams> data_params{std::make_unique<DataParams>()};
    
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
        s_step_params->batch_size = std::max<size_t>(1, std::atoi(argv[1]));
    }
    if (argc > 2) {
        s_step_params->s = std::max<size_t>(1, std::atoi(argv[2]));
    }
    if (argc > 3) {
        s_step_params->epochs = std::max<size_t>(1, std::atoi(argv[3]));
    }
    if (argc > 4) {
        data_params->file_name = argv[4];
    }
    if (argc > 5) {
        s_step_params->approx_gram = true;
        const std::string approx_type{argv[5]};
        if (approx_type == "uniform") {
            s_step_params->approx_gram_type = ApproxGramType::Uniform;
        } else if (approx_type == "scoring") {
            s_step_params->approx_gram_type = ApproxGramType::Scoring;
        } else {
            std::cerr << "Error: approx type must be 'uniform' or 'scoring'\n";
            return 1;
        }
    }
    if (argc > 6) {
        s_step_params->approx_gram_l = std::max<size_t>(1, std::atoi(argv[6]));
    }

    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    std::cout << "batch_size: " << s_step_params->batch_size
              << ", s: " << s_step_params->s
              << ", epochs: " << s_step_params->epochs
              << ", training_set: " << data_params->file_name
              << ", approx_gram_type: " << to_string(s_step_params->approx_gram_type)
              << ", approx_gram_l: " << s_step_params->approx_gram_l << '\n';

    // load raw data into host
    std::vector<float> h_A{};
    std::vector<float> h_y{};
    load_libsvm(data_params.get(), h_A, h_y);
    data_params->total_samples_unpadded = h_y.size();
    
    // Pad data to be multiple of s * batch_size
    s_step_params->samples_per_iter = s_step_params->s * s_step_params->batch_size;
    // calculate extra samples(needed to pad end of dataset to make it divisible by samples_per_iter)
    const size_t extra_samples{s_step_params->samples_per_iter - (data_params->total_samples_unpadded % s_step_params->samples_per_iter)};
    for (size_t i{0}; i < extra_samples; ++i) {
        h_y.push_back(0.0f);
        for (size_t j{0}; j < data_params->n_features; ++j) {
            h_A.push_back(0.0f);
        }
    }
    data_params->total_samples = h_y.size();

    if (s_step_params->batch_size * s_step_params->s > data_params->total_samples) {
        std::cerr << "Error: batch_size * s must be <= total_samples\n";
        return 1;
    }

    // calculate number of iterations(one iteration samples total_samples / s * batch_size) based on epochs and total samples
    s_step_params->maxiters = s_step_params->epochs * data_params->total_samples / s_step_params->samples_per_iter;
    s_step_params->printerval = data_params->total_samples / s_step_params->samples_per_iter;
    
    std::unique_ptr<Workspace> workspace{std::make_unique<Workspace>(data_params.get(), s_step_params.get())};
    std::unique_ptr<ProfileStats> run_stats{std::make_unique<ProfileStats>()};
    
    // copy data to GPU
    cudaMemcpy(workspace->d_A, h_A.data(), data_params->total_samples * data_params->n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(workspace->d_y, h_y.data(), data_params->total_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(workspace->d_x, 0, data_params->n_features * sizeof(float));

    CudaRegionTimer training_timer{};
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
