#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>

#include "s_step_sgd.hpp"

constexpr std::size_t kBlockSize{256};

struct AddRecurrenceStats {
    __device__ RecurrenceStats operator()(const RecurrenceStats& left,
                                          const RecurrenceStats& right) const {
        return {left.corrected_residual_difference_squared_sum +
                    right.corrected_residual_difference_squared_sum,
                left.base_residual_squared_sum + right.base_residual_squared_sum};
    }
};

struct AddMetricSums {
    __host__ __device__ MetricSums operator()(const MetricSums& left,
                                              const MetricSums& right) const {
        return {left.objective_sum + right.objective_sum,
                left.correct_count + right.correct_count};
    }
};

struct CalculateMetricSums {
    const float* signed_margins;
    const float* labels;

    __device__ MetricSums operator()(std::size_t sample_index) const {
        const double signed_margin = signed_margins[sample_index];
        const double objective =
            fmax(0.0, -signed_margin) + log1p(exp(-fabs(signed_margin)));
        const unsigned long long correct =
            signed_margin > 0.0 || (signed_margin == 0.0 && labels[sample_index] < 0.0);
        return {objective, correct};
    }
};

// ---------------- CUDA Kernels ----------------
// Convert each signed margin into its logistic residual.
__global__ void sigmoid_kernel(const float* signed_margins, float* residuals, std::size_t sample_count) {
    const std::size_t sample_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (sample_index < sample_count) {
        residuals[sample_index] = 1.0f / (1.0f + __expf(signed_margins[sample_index]));
    }
}

// Form corrected residuals and reduce their relative change for the controller.
__global__ void corrected_sigmoid_and_accumulate_kernel(const float* corrected_margins,
                                                        const float* base_residuals,
                                                        float* corrected_residuals,
                                                        std::size_t sample_count,
                                                        RecurrenceStats* recurrence_stats) {
    using BlockReduce = cub::BlockReduce<RecurrenceStats, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage reduction_storage;
    const std::size_t sample_index = threadIdx.x + blockIdx.x * blockDim.x;
    float difference_squared = 0.0f;
    float base_squared = 0.0f;
    if (sample_index < sample_count) {
        const float base_residual = base_residuals[sample_index];
        const float corrected_residual = 1.0f / (1.0f + __expf(corrected_margins[sample_index]));
        corrected_residuals[sample_index] = corrected_residual;
        const float difference = corrected_residual - base_residual;
        difference_squared = difference * difference;
        base_squared = base_residual * base_residual;
    }
    const RecurrenceStats block_sum = BlockReduce(reduction_storage).Reduce(
        {difference_squared, base_squared}, AddRecurrenceStats{});

    if (threadIdx.x == 0) {
        atomicAdd(&recurrence_stats->corrected_residual_difference_squared_sum,
                  block_sum.corrected_residual_difference_squared_sum);
        atomicAdd(&recurrence_stats->base_residual_squared_sum,
                  block_sum.base_residual_squared_sum);
    }
}
// Gather selected columns and apply leverage-probability scaling to each output entry.
__global__ void gather_weighted_columns_kernel(
    const float* source_features, float* sampled_features, std::size_t row_count,
    std::size_t source_feature_count, const std::size_t* sampled_column_indices,
    const float* sampled_column_scales, std::size_t sampled_feature_count, float scale_multiplier) {
    const std::size_t entry_index = threadIdx.x + blockIdx.x * blockDim.x;
    const std::size_t entry_count = row_count * sampled_feature_count;
    if (entry_index >= entry_count)
        return;
    const std::size_t row_index = entry_index / sampled_feature_count;
    const std::size_t sampled_column_index = entry_index % sampled_feature_count;
    sampled_features[entry_index] = source_features[row_index * source_feature_count +
                                                    sampled_column_indices[sampled_column_index]] *
                                    sampled_column_scales[sampled_column_index] * scale_multiplier;
}

// ================== CUDA Helper Functions (callable from host) ==================
// Launch asynchronous sigmoid evaluation on the supplied stream.
void cuda_sigmoid(cudaStream_t stream, const float* signed_margins, float* residuals,
                  std::size_t sample_count) {
    const std::size_t block_count = (sample_count + kBlockSize - 1) / kBlockSize;
    // Evaluate one logistic residual per signed margin.
    sigmoid_kernel<<<block_count, kBlockSize, 0, stream>>>(signed_margins, residuals, sample_count);
    check_cuda(cudaGetLastError());
}

// Clear recurrence statistics and launch fused corrected-residual accumulation.
void cuda_corrected_sigmoid_and_accumulate(cudaStream_t stream, const float* corrected_margins,
                                           const float* base_residuals, float* corrected_residuals,
                                           std::size_t sample_count, RecurrenceStats* recurrence_stats) {
    check_cuda(cudaMemsetAsync(recurrence_stats, 0, sizeof(RecurrenceStats), stream));
    const std::size_t block_count = (sample_count + kBlockSize - 1) / kBlockSize;
    // Produce corrected residuals while reducing the base-to-corrected change.
    corrected_sigmoid_and_accumulate_kernel<<<block_count, kBlockSize, 0, stream>>>(
        corrected_margins, base_residuals, corrected_residuals, sample_count, recurrence_stats);
    check_cuda(cudaGetLastError());
}

// Launch asynchronous leverage-weighted feature gathering.
void cuda_gather_weighted_columns(cudaStream_t stream, const float* source_features,
                                  float* sampled_features, std::size_t row_count,
                                  std::size_t source_feature_count, const std::size_t* sampled_column_indices,
                                  const float* sampled_column_scales, std::size_t sampled_feature_count,
                                  float scale_multiplier) {
    const std::size_t entry_count = row_count * sampled_feature_count;
    const std::size_t block_count = (entry_count + kBlockSize - 1) / kBlockSize;
    // Gather leverage-sampled columns and apply their probability correction.
    gather_weighted_columns_kernel<<<block_count, kBlockSize, 0, stream>>>(
        source_features, sampled_features, row_count, source_feature_count, sampled_column_indices,
        sampled_column_scales, sampled_feature_count, scale_multiplier);
    check_cuda(cudaGetLastError());
}

// Alternative Thrust permutation-iterator implementation (disabled).
// Required headers: thrust/execution_policy.h, thrust/iterator/counting_iterator.h,
// thrust/iterator/permutation_iterator.h, thrust/iterator/transform_iterator.h,
// thrust/iterator/zip_iterator.h, thrust/transform.h, and thrust/tuple.h.
//
// void cuda_gather_weighted_columns_thrust(
//     cudaStream_t stream, const float* source_features, float* sampled_features,
//     std::size_t row_count, std::size_t source_feature_count,
//     const std::size_t* sampled_column_indices, const float* sampled_column_scales,
//     std::size_t sampled_feature_count, float scale_multiplier) {
//     const std::size_t entry_count = row_count * sampled_feature_count;
//     auto output_index = thrust::make_counting_iterator<std::size_t>(0);
//
//     // Map B[row, k] to A[row, sampled_column_indices[k]] without storing an index array.
//     auto source_index = thrust::make_transform_iterator(
//         output_index, [=] __device__(std::size_t index) {
//             const std::size_t row = index / sampled_feature_count;
//             const std::size_t sampled_index = index % sampled_feature_count;
//             return row * source_feature_count + sampled_column_indices[sampled_index];
//         });
//     auto gathered = thrust::make_permutation_iterator(source_features, source_index);
//     auto input =
//         thrust::make_zip_iterator(thrust::make_tuple(gathered, output_index));
//
//     // Fuse the indexed load and leverage-probability scaling into one generated kernel.
//     thrust::transform(
//         thrust::cuda::par.on(stream), input, input + entry_count, sampled_features,
//         [=] __device__(const auto& value_and_index) {
//             const float value = thrust::get<0>(value_and_index);
//             const std::size_t index = thrust::get<1>(value_and_index);
//             const std::size_t sampled_index = index % sampled_feature_count;
//             return value * sampled_column_scales[sampled_index] * scale_multiplier;
//         });
// }

// Query CUB once for the persistent temporary storage used by metric reductions.
std::size_t cuda_metric_reduction_workspace_size(std::size_t sample_count) {
    std::size_t workspace_size = 0;
    auto sample_indices = thrust::make_counting_iterator<std::size_t>(0);
    check_cuda(cub::DeviceReduce::TransformReduce(
        nullptr, workspace_size, sample_indices, static_cast<MetricSums*>(nullptr), sample_count,
        AddMetricSums{}, CalculateMetricSums{nullptr, nullptr}, MetricSums{}));
    return workspace_size;
}

// Transform logistic metrics and reduce them device-wide using CUB.
void cuda_accumulate_metrics(cudaStream_t stream, const float* signed_margins, const float* labels,
                             std::size_t sample_count, void* reduction_workspace,
                             std::size_t reduction_workspace_size, MetricSums* metric_sums) {
    auto sample_indices = thrust::make_counting_iterator<std::size_t>(0);
    check_cuda(cub::DeviceReduce::TransformReduce(
        reduction_workspace, reduction_workspace_size, sample_indices, metric_sums, sample_count,
        AddMetricSums{}, CalculateMetricSums{signed_margins, labels}, MetricSums{}, stream));
}
