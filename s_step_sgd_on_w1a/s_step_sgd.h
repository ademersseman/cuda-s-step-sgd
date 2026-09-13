#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <string>
#include <vector>

enum class ApproxGramType {
    Uniform,
    Scoring
};

inline const char* to_string(ApproxGramType type) {
    switch (type) {
        case ApproxGramType::Uniform:
            return "uniform";
        case ApproxGramType::Scoring:
            return "scoring";
        default:
            return "unknown";
    }
}

struct ProfileStats {
    float init_corr_time{0.0f};
    float gram_overhead_time{0.0f};
    float gram_compute_time{0.0f};
    float recurrence_time{0.0f};
    float grad_proj_time{0.0f};
    float weight_update_time{0.0f};
    float scaling_time{0.0f};
    float training_time{0.0f};
};

struct RunParams {
    size_t batch_size{16};
    size_t s{4};
    size_t samples_per_iter{batch_size * s};
    size_t epochs{10};
    size_t maxiters{65536};
    size_t printerval{0};
    float eta{0.5f};
    bool approx_gram{false};
    size_t approx_gram_l{64};
    ApproxGramType approx_gram_type{ApproxGramType::Uniform};
};

struct DataParams {
    std::string file_name{"w1a.txt"};
    size_t n_features{300};
    size_t total_samples_unpadded{0};
    size_t total_samples{0};
};

struct CudaRegionTimer {
    cudaEvent_t start{nullptr};
    cudaEvent_t stop{nullptr};

    CudaRegionTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    void begin(cudaStream_t stream = 0) {
        cudaEventRecord(start, stream);
    }

    float end(cudaStream_t stream = 0) {
        cudaEventRecord(stop, stream);
        float ms{0.0f};
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }

    ~CudaRegionTimer() {
        if (start) {
            cudaEventDestroy(start);
        }
        if (stop) {
            cudaEventDestroy(stop);
        }
    }
};

struct Workspace {
    float* d_A{};
    float* d_y{};
    float* d_x{};

    float* d_A_scaled{};
    float* d_batch_A_approx[2]{};
    float* d_correction{};
    float* d_G[2]{};
    float* d_grad{};

    float* d_scores{};
    std::vector<std::vector<float>> score_cache{};
    std::vector<bool> cache_valid{};

    cublasHandle_t handle{};
    cublasHandle_t prefetch_handle{};
    cudaStream_t compute_stream{};
    cudaStream_t prefetch_stream{};
    cudaEvent_t gram_overhead_prefetch_done{};
    cudaEvent_t gram_prefetch_done{};
    size_t prefetch_buf{0};
    size_t compute_buf{0};

    Workspace(const DataParams* data_params, const RunParams* s_step_params) {
        cudaMalloc(&d_A, data_params->total_samples * data_params->n_features * sizeof(float));
        cudaMalloc(&d_y, data_params->total_samples * sizeof(float));
        cudaMalloc(&d_x, data_params->n_features * sizeof(float));

        cudaMalloc(&d_A_scaled, data_params->total_samples * data_params->n_features * sizeof(float));
        cudaMalloc(&d_batch_A_approx[0], s_step_params->samples_per_iter * s_step_params->approx_gram_l * sizeof(float));
        cudaMalloc(&d_batch_A_approx[1], s_step_params->samples_per_iter * s_step_params->approx_gram_l * sizeof(float));
        cudaMalloc(&d_correction, s_step_params->samples_per_iter * sizeof(float));
        cudaMalloc(&d_G[0], s_step_params->samples_per_iter * s_step_params->samples_per_iter * sizeof(float));
        cudaMalloc(&d_G[1], s_step_params->samples_per_iter * s_step_params->samples_per_iter * sizeof(float));
        cudaMalloc(&d_grad, data_params->n_features * sizeof(float));

        cudaMalloc(&d_scores, data_params->n_features * sizeof(float));
        score_cache = std::vector<std::vector<float>>(
            data_params->total_samples / s_step_params->samples_per_iter,
            std::vector<float>(data_params->n_features));
        cache_valid = std::vector<bool>(data_params->total_samples / s_step_params->samples_per_iter, false);

        cublasCreate(&handle);
        cublasCreate(&prefetch_handle);
        cudaStreamCreateWithFlags(&prefetch_stream, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
        cublasSetStream(prefetch_handle, prefetch_stream);
        cublasSetStream(handle, compute_stream);
        cudaEventCreateWithFlags(&gram_overhead_prefetch_done, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&gram_prefetch_done, cudaEventDisableTiming);
    }

    ~Workspace() {
        cudaFree(d_A);
        cudaFree(d_y);
        cudaFree(d_x);

        cudaFree(d_A_scaled);
        cudaFree(d_batch_A_approx[0]);
        cudaFree(d_batch_A_approx[1]);
        cudaFree(d_correction);
        cudaFree(d_G[0]);
        cudaFree(d_G[1]);
        cudaFree(d_grad);

        cudaFree(d_scores);

        cublasDestroy(handle);
        cublasDestroy(prefetch_handle);
        cudaStreamDestroy(prefetch_stream);
        cudaStreamDestroy(compute_stream);
        cudaEventDestroy(gram_prefetch_done);
        cudaEventDestroy(gram_overhead_prefetch_done);
    }
};

void cuda_apply_sigmoid_block(
    cudaStream_t stream,
    float* correction,
    size_t total_samples,
    size_t batch_size,
    size_t block_idx);

void enter_recurrence(
    const DataParams* data_params,
    const RunParams* s_step_params,
    Workspace* workspace,
    float* d_batch_A,
    ProfileStats* run_stats);

void train(
    const DataParams* data_params,
    const RunParams* s_step_params,
    Workspace* workspace,
    const std::vector<float>& h_A,
    const std::vector<float>& h_y,
    ProfileStats* stats);
