#include <cuda_runtime.h>

#include "s_step_sgd.hpp"

constexpr size_t kBlockSize{256};

// ---------------- CUDA Kernels ----------------
__global__ void sigmoid_kernel(const float* signed_margins, float* residuals, size_t sample_count) {
    const size_t sample_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (sample_index < sample_count) {
        residuals[sample_index] = 1.0f / (1.0f + __expf(signed_margins[sample_index]));
    }
}

__global__ void corrected_sigmoid_and_accumulate_kernel(const float* corrected_margins,
                                                        const float* base_residuals,
                                                        float* corrected_residuals,
                                                        size_t sample_count,
                                                        RecurrenceStats* recurrence_stats) {
    __shared__ float difference_squared_sums[kBlockSize];
    __shared__ float base_squared_sums[kBlockSize];
    const size_t sample_index = threadIdx.x + blockIdx.x * blockDim.x;
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
    difference_squared_sums[threadIdx.x] = difference_squared;
    base_squared_sums[threadIdx.x] = base_squared;
    __syncthreads();

    for (size_t reduction_offset = kBlockSize / 2; reduction_offset > 0; reduction_offset /= 2) {
        if (threadIdx.x < reduction_offset) {
            difference_squared_sums[threadIdx.x] +=
                difference_squared_sums[threadIdx.x + reduction_offset];
            base_squared_sums[threadIdx.x] += base_squared_sums[threadIdx.x + reduction_offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&recurrence_stats->corrected_residual_difference_squared_sum,
                  difference_squared_sums[0]);
        atomicAdd(&recurrence_stats->base_residual_squared_sum, base_squared_sums[0]);
    }
}
__global__ void exclusive_prefix_columns_kernel(float* column_values, size_t column_count,
                                                size_t row_count) {
    const size_t column_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (column_index >= column_count)
        return;
    float prefix_sum = 0.0f;
    for (size_t row_index = 0; row_index < row_count; ++row_index) {
        const size_t entry_index = row_index * column_count + column_index;
        const float current_value = column_values[entry_index];
        column_values[entry_index] = prefix_sum;
        prefix_sum += current_value;
    }
}

__global__ void gather_weighted_columns_kernel(
    const float* source_features, float* sampled_features, size_t row_count,
    size_t source_feature_count, const size_t* sampled_column_indices,
    const float* sampled_column_scales, size_t sampled_feature_count, float scale_multiplier) {
    const size_t entry_index = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t entry_count = row_count * sampled_feature_count;
    if (entry_index >= entry_count)
        return;
    const size_t row_index = entry_index / sampled_feature_count;
    const size_t sampled_column_index = entry_index % sampled_feature_count;
    sampled_features[entry_index] = source_features[row_index * source_feature_count +
                                                    sampled_column_indices[sampled_column_index]] *
                                    sampled_column_scales[sampled_column_index] * scale_multiplier;
}

__global__ void accumulate_metrics_kernel(const float* signed_margins, const float* labels,
                                          size_t sample_count, MetricSums* metric_sums) {
    __shared__ double block_objective_sums[kBlockSize];
    __shared__ unsigned long long block_correct_counts[kBlockSize];
    const size_t sample_index = threadIdx.x + blockIdx.x * blockDim.x;

    double sample_objective = 0.0;
    unsigned long long sample_correct_count = 0;
    if (sample_index < sample_count) {
        const double signed_margin = signed_margins[sample_index];
        sample_objective = fmax(0.0, -signed_margin) + log1p(exp(-fabs(signed_margin)));
        sample_correct_count =
            signed_margin > 0.0 || (signed_margin == 0.0 && labels[sample_index] < 0.0);
    }
    block_objective_sums[threadIdx.x] = sample_objective;
    block_correct_counts[threadIdx.x] = sample_correct_count;
    __syncthreads();

    for (size_t reduction_offset = kBlockSize / 2; reduction_offset > 0; reduction_offset /= 2) {
        if (threadIdx.x < reduction_offset) {
            block_objective_sums[threadIdx.x] +=
                block_objective_sums[threadIdx.x + reduction_offset];
            block_correct_counts[threadIdx.x] +=
                block_correct_counts[threadIdx.x + reduction_offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&metric_sums->objective_sum, block_objective_sums[0]);
        atomicAdd(&metric_sums->correct_count, block_correct_counts[0]);
    }
}

// ================== CUDA Helper Functions (callable from host) ==================
void cuda_sigmoid(cudaStream_t stream, const float* signed_margins, float* residuals,
                  size_t sample_count) {
    const size_t block_count = (sample_count + kBlockSize - 1) / kBlockSize;
    sigmoid_kernel<<<block_count, kBlockSize, 0, stream>>>(signed_margins, residuals, sample_count);
    check_cuda(cudaGetLastError());
}

void cuda_corrected_sigmoid_and_accumulate(cudaStream_t stream, const float* corrected_margins,
                                           const float* base_residuals, float* corrected_residuals,
                                           size_t sample_count, RecurrenceStats* recurrence_stats) {
    check_cuda(cudaMemsetAsync(recurrence_stats, 0, sizeof(RecurrenceStats), stream));
    const size_t block_count = (sample_count + kBlockSize - 1) / kBlockSize;
    corrected_sigmoid_and_accumulate_kernel<<<block_count, kBlockSize, 0, stream>>>(
        corrected_margins, base_residuals, corrected_residuals, sample_count, recurrence_stats);
    check_cuda(cudaGetLastError());
}

void cuda_exclusive_prefix_columns(cudaStream_t stream, float* column_values, size_t column_count,
                                   size_t row_count) {
    const size_t block_count = (column_count + kBlockSize - 1) / kBlockSize;
    exclusive_prefix_columns_kernel<<<block_count, kBlockSize, 0, stream>>>(
        column_values, column_count, row_count);
    check_cuda(cudaGetLastError());
}

void cuda_gather_weighted_columns(cudaStream_t stream, const float* source_features,
                                  float* sampled_features, size_t row_count,
                                  size_t source_feature_count, const size_t* sampled_column_indices,
                                  const float* sampled_column_scales, size_t sampled_feature_count,
                                  float scale_multiplier) {
    const size_t entry_count = row_count * sampled_feature_count;
    const size_t block_count = (entry_count + kBlockSize - 1) / kBlockSize;
    gather_weighted_columns_kernel<<<block_count, kBlockSize, 0, stream>>>(
        source_features, sampled_features, row_count, source_feature_count, sampled_column_indices,
        sampled_column_scales, sampled_feature_count, scale_multiplier);
    check_cuda(cudaGetLastError());
}

void cuda_accumulate_metrics(cudaStream_t stream, const float* signed_margins, const float* labels,
                             size_t sample_count, MetricSums* metric_sums) {
    const size_t block_count = (sample_count + kBlockSize - 1) / kBlockSize;
    accumulate_metrics_kernel<<<block_count, kBlockSize, 0, stream>>>(signed_margins, labels,
                                                                      sample_count, metric_sums);
    check_cuda(cudaGetLastError());
}
