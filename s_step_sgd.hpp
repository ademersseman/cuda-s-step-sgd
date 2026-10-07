#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <source_location>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

inline std::string call_site(const std::source_location& location) {
    return std::string(location.file_name()) + ":" + std::to_string(location.line()) + " in " +
           location.function_name();
}

inline void check_cuda(cudaError_t status,
                       const std::source_location& location = std::source_location::current()) {
    if (status != cudaSuccess) {
        throw std::runtime_error("CUDA error at " + call_site(location) + ": " +
                                 cudaGetErrorString(status));
    }
}

inline void check_cublas(cublasStatus_t status,
                         const std::source_location& location = std::source_location::current()) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS error at " + call_site(location) + ": status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

struct ProfileStats {
    float training_time_ms{0.0f};
};

struct RunParams {
    int batch_size{16};
    int s{4};
    int samples_per_iteration{batch_size * s};
    int iteration_count{100};
    int print_interval{0};
    float learning_rate{0.5f};
    size_t requested_sampled_feature_count{64};
    size_t leverage_rank{32};
    size_t leverage_row_count{512};
    float uniform_probability_mix{0.1f};
    unsigned int sketch_seed{0};
};

inline size_t stochastic_sampled_feature_count(float correction_ratio, float random_draw,
                                               size_t maximum_sampled_feature_count) {
    correction_ratio = correction_ratio > 0.0f ? correction_ratio : 0.0f;
    random_draw = std::clamp(random_draw, 0.0f, 1.0f);
    size_t lower_count = 8;
    size_t upper_count = 16;
    float upper_probability = correction_ratio / 0.01f;
    if (correction_ratio >= 0.05f) {
        lower_count = 32;
        upper_count = 64;
        upper_probability = (correction_ratio - 0.05f) / 0.15f;
    } else if (correction_ratio >= 0.01f) {
        lower_count = 16;
        upper_count = 32;
        upper_probability = (correction_ratio - 0.01f) / 0.04f;
    }
    const size_t selected_count =
        random_draw < std::clamp(upper_probability, 0.0f, 1.0f) ? upper_count : lower_count;
    return std::min(selected_count, maximum_sampled_feature_count);
}

struct DataParams {
    std::string dataset_path{};
    size_t feature_count{300};
    size_t sample_count{0};
    size_t padded_sample_count{0};
    std::uint64_t dataset_seed{0};
    bool is_sstep_binary{false};
};

struct MetricSums {
    double objective_sum{0.0};
    unsigned long long correct_count{0};
};

struct RecurrenceStats {
    float corrected_residual_difference_squared_sum{0.0f};
    float base_residual_squared_sum{0.0f};
};

struct LeverageSketch {
    std::vector<size_t> sampled_column_indices{};
    std::vector<float> sampled_column_scales{};
    std::vector<size_t> sampled_feature_counts{};
    std::vector<size_t> sketch_offsets{};
    size_t* device_sampled_column_indices{nullptr};
    float* device_sampled_column_scales{nullptr};
    size_t sampled_feature_count{0};
    size_t sketch_count{0};

    LeverageSketch() = default;
    ~LeverageSketch() {
        cudaFree(device_sampled_column_indices);
        cudaFree(device_sampled_column_scales);
    }
    LeverageSketch(const LeverageSketch&) = delete;
    LeverageSketch& operator=(const LeverageSketch&) = delete;
    LeverageSketch(LeverageSketch&& other) noexcept
        : sampled_column_indices(std::move(other.sampled_column_indices)),
          sampled_column_scales(std::move(other.sampled_column_scales)),
          sampled_feature_counts(std::move(other.sampled_feature_counts)),
          sketch_offsets(std::move(other.sketch_offsets)),
          device_sampled_column_indices(
              std::exchange(other.device_sampled_column_indices, nullptr)),
          device_sampled_column_scales(std::exchange(other.device_sampled_column_scales, nullptr)),
          sampled_feature_count(other.sampled_feature_count), sketch_count(other.sketch_count) {}
    LeverageSketch& operator=(LeverageSketch&& other) noexcept {
        if (this != &other) {
            cudaFree(device_sampled_column_indices);
            cudaFree(device_sampled_column_scales);
            sampled_column_indices = std::move(other.sampled_column_indices);
            sampled_column_scales = std::move(other.sampled_column_scales);
            sampled_feature_counts = std::move(other.sampled_feature_counts);
            sketch_offsets = std::move(other.sketch_offsets);
            device_sampled_column_indices =
                std::exchange(other.device_sampled_column_indices, nullptr);
            device_sampled_column_scales =
                std::exchange(other.device_sampled_column_scales, nullptr);
            sampled_feature_count = other.sampled_feature_count;
            sketch_count = other.sketch_count;
        }
        return *this;
    }

    size_t size() const {
        return sampled_feature_count;
    }

    size_t size(size_t sketch_index) const {
        return sampled_feature_counts.at(sketch_index);
    }

    size_t offset(size_t sketch_index) const {
        return sketch_offsets.at(sketch_index);
    }

    void upload() {
        check_cuda(cudaMalloc(&device_sampled_column_indices,
                              sampled_column_indices.size() * sizeof(size_t)));
        check_cuda(cudaMalloc(&device_sampled_column_scales,
                              sampled_column_scales.size() * sizeof(float)));
        check_cuda(cudaMemcpy(device_sampled_column_indices, sampled_column_indices.data(),
                              sampled_column_indices.size() * sizeof(size_t),
                              cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(device_sampled_column_scales, sampled_column_scales.data(),
                              sampled_column_scales.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
};

struct SampledBatchSlot {
    float* device_sampled_features{nullptr};
    cudaEvent_t prefetch_done{nullptr};
    cudaEvent_t compute_done{nullptr};
};

struct RecurrenceGraph {
    size_t slot_index{0};
    size_t sampled_feature_count{0};
    cudaGraph_t graph{nullptr};
    cudaGraphExec_t executable{nullptr};
};

struct Workspace {
    float* device_features{nullptr};
    float* device_labels{nullptr};
    float* device_weights{nullptr};
    float* device_signed_features{nullptr};
    float* device_block_margins_or_residuals{nullptr};
    float* device_projected_gradient_prefix{nullptr};
    float* device_base_residuals{nullptr};
    float* device_metric_margins{nullptr};
    MetricSums* device_metric_sums{nullptr};
    RecurrenceStats* device_recurrence_stats{nullptr};
    RecurrenceStats* host_recurrence_stats{nullptr};
    LeverageSketch leverage_sketch{};

    cudaStream_t prefetch_stream{nullptr};
    cudaStream_t compute_stream{nullptr};
    cudaEvent_t recurrence_stats_copy_done{nullptr};
    cublasHandle_t cublas_handle{nullptr};
    std::array<SampledBatchSlot, 2> sampled_batch_slots{};
    std::vector<RecurrenceGraph> recurrence_graphs{};
    size_t current_slot_index{0};

    Workspace(const DataParams* data_params, const RunParams* run_params,
              LeverageSketch initial_leverage_sketch)
        : device_features(nullptr), device_labels(nullptr), device_weights(nullptr),
          device_signed_features(nullptr), device_block_margins_or_residuals(nullptr),
          device_projected_gradient_prefix(nullptr), device_base_residuals(nullptr),
          device_metric_margins(nullptr), device_metric_sums(nullptr),
          device_recurrence_stats(nullptr), host_recurrence_stats(nullptr),
          leverage_sketch(std::move(initial_leverage_sketch)), prefetch_stream(nullptr),
          compute_stream(nullptr), recurrence_stats_copy_done(nullptr), cublas_handle(nullptr),
          sampled_batch_slots{}, recurrence_graphs{}, current_slot_index(0) {
        const size_t dataset_element_count =
            data_params->padded_sample_count * data_params->feature_count;
        const size_t sampled_element_count =
            run_params->samples_per_iteration * leverage_sketch.size();
        try {
            check_cuda(cudaStreamCreateWithFlags(&prefetch_stream, cudaStreamNonBlocking));
            check_cuda(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
            check_cuda(
                cudaEventCreateWithFlags(&recurrence_stats_copy_done, cudaEventDisableTiming));
            check_cublas(cublasCreate(&cublas_handle));
            for (auto& batch_slot : sampled_batch_slots) {
                check_cuda(
                    cudaEventCreateWithFlags(&batch_slot.prefetch_done, cudaEventDisableTiming));
                check_cuda(
                    cudaEventCreateWithFlags(&batch_slot.compute_done, cudaEventDisableTiming));
            }
            check_cuda(cudaMalloc(&device_features, dataset_element_count * sizeof(float)));
            check_cuda(
                cudaMalloc(&device_labels, data_params->padded_sample_count * sizeof(float)));
            check_cuda(cudaMalloc(&device_weights, data_params->feature_count * sizeof(float)));
            check_cuda(cudaMalloc(&device_signed_features, dataset_element_count * sizeof(float)));
            check_cuda(cudaMalloc(&device_block_margins_or_residuals,
                                  run_params->samples_per_iteration * sizeof(float)));
            check_cuda(cudaMalloc(&device_projected_gradient_prefix,
                                  run_params->s * leverage_sketch.size() * sizeof(float)));
            check_cuda(cudaMalloc(&device_base_residuals,
                                  run_params->samples_per_iteration * sizeof(float)));
            check_cuda(
                cudaMalloc(&device_metric_margins, data_params->sample_count * sizeof(float)));
            check_cuda(cudaMalloc(&device_metric_sums, sizeof(MetricSums)));
            check_cuda(cudaMalloc(&device_recurrence_stats, sizeof(RecurrenceStats)));
            check_cuda(cudaMallocHost(&host_recurrence_stats, sizeof(RecurrenceStats)));
            for (auto& batch_slot : sampled_batch_slots) {
                check_cuda(cudaMalloc(&batch_slot.device_sampled_features,
                                      sampled_element_count * sizeof(float)));
            }

            check_cublas(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
            check_cublas(cublasSetStream(cublas_handle, compute_stream));
        } catch (...) {
            for (auto& recurrence_graph : recurrence_graphs) {
                cudaGraphExecDestroy(recurrence_graph.executable);
                cudaGraphDestroy(recurrence_graph.graph);
            }
            cudaFree(device_features);
            cudaFree(device_labels);
            cudaFree(device_weights);
            cudaFree(device_signed_features);
            cudaFree(device_block_margins_or_residuals);
            cudaFree(device_projected_gradient_prefix);
            cudaFree(device_base_residuals);
            cudaFree(device_metric_margins);
            cudaFree(device_metric_sums);
            cudaFree(device_recurrence_stats);
            cudaFreeHost(host_recurrence_stats);
            for (auto& batch_slot : sampled_batch_slots) {
                cudaFree(batch_slot.device_sampled_features);
                cudaEventDestroy(batch_slot.prefetch_done);
                cudaEventDestroy(batch_slot.compute_done);
            }
            cudaEventDestroy(recurrence_stats_copy_done);
            cublasDestroy(cublas_handle);
            cudaStreamDestroy(compute_stream);
            cudaStreamDestroy(prefetch_stream);
            throw;
        }
    }

    ~Workspace() {
        for (auto& recurrence_graph : recurrence_graphs) {
            cudaGraphExecDestroy(recurrence_graph.executable);
            cudaGraphDestroy(recurrence_graph.graph);
        }
        cudaFree(device_features);
        cudaFree(device_labels);
        cudaFree(device_weights);
        cudaFree(device_signed_features);
        cudaFree(device_block_margins_or_residuals);
        cudaFree(device_projected_gradient_prefix);
        cudaFree(device_base_residuals);
        cudaFree(device_metric_margins);
        cudaFree(device_metric_sums);
        cudaFree(device_recurrence_stats);
        cudaFreeHost(host_recurrence_stats);
        for (auto& batch_slot : sampled_batch_slots) {
            cudaFree(batch_slot.device_sampled_features);
            cudaEventDestroy(batch_slot.prefetch_done);
            cudaEventDestroy(batch_slot.compute_done);
        }
        cudaEventDestroy(recurrence_stats_copy_done);
        cublasDestroy(cublas_handle);
        cudaStreamDestroy(compute_stream);
        cudaStreamDestroy(prefetch_stream);
    }
    Workspace(const Workspace&) = delete;
    Workspace(Workspace&&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    Workspace& operator=(Workspace&&) = delete;
};

void enter_recurrence(const DataParams* data_params, const RunParams* run_params,
                      Workspace* workspace, float* device_signed_batch,
                      size_t sampled_feature_count);

void cuda_sigmoid(cudaStream_t stream, const float* signed_margins, float* residuals,
                  size_t sample_count);
void cuda_corrected_sigmoid_and_accumulate(cudaStream_t stream, const float* corrected_margins,
                                           const float* base_residuals, float* corrected_residuals,
                                           size_t sample_count, RecurrenceStats* recurrence_stats);
void cuda_exclusive_prefix_columns(cudaStream_t stream, float* column_values, size_t column_count,
                                   size_t row_count);
void cuda_gather_weighted_columns(cudaStream_t stream, const float* source_features,
                                  float* sampled_features, size_t row_count,
                                  size_t source_feature_count, const size_t* sampled_column_indices,
                                  const float* sampled_column_scales, size_t sampled_feature_count,
                                  float scale_multiplier);
void cuda_accumulate_metrics(cudaStream_t stream, const float* signed_margins, const float* labels,
                             size_t sample_count, MetricSums* metric_sums);

void train(const DataParams* data_params, const RunParams* run_params, Workspace* workspace,
           ProfileStats* profile_stats);
