#include "prefetch.hpp"
#include "feature_sampling.hpp"

#include <algorithm>
#include <cmath>

// Estimate persistent GPU leverage probabilities and initialize each block's active width.
LeverageSketch prepare_leverage_sketch(const DataParams& data_params, const RunParams& run_params,
                                       const float* device_features) {
    LeverageSketch leverage_sketch{};
    leverage_sketch.sampled_feature_count = run_params.requested_sampled_feature_count;
    leverage_sketch.device_probabilities.allocate(data_params.feature_count);
    leverage_sketch.device_alias_probabilities.allocate(data_params.feature_count);
    leverage_sketch.device_alias_indices.allocate(data_params.feature_count);
    estimate_column_leverage_probabilities(
        data_params, run_params, device_features, leverage_sketch.device_probabilities,
        leverage_sketch.device_alias_probabilities, leverage_sketch.device_alias_indices);
    leverage_sketch.sketch_count = static_cast<std::size_t>(std::max(1, run_params.iteration_count));
    const std::size_t initial_sampled_feature_count = run_params.fixed_sampled_feature_count == 0
                                                          ? std::min<std::size_t>(
                                                                64, leverage_sketch.sampled_feature_count)
                                                          : run_params.fixed_sampled_feature_count;
    leverage_sketch.sampled_feature_counts.assign(leverage_sketch.sketch_count,
                                                  initial_sampled_feature_count);
    return leverage_sketch;
}

// Queue sampling, gathering, and full-Gram construction for one double-buffer slot.
void prefetch_leverage_features(const DataParams& data_params, const RunParams& run_params,
                                Workspace* workspace, float* device_next_signed_batch,
                                std::size_t slot_index, std::size_t sampled_feature_count) {
    auto& target_slot = workspace->sampled_batch_slots[slot_index];
    const auto& leverage_sketch = workspace->leverage_sketch;
    const float scale_multiplier = std::sqrt(static_cast<float>(leverage_sketch.size()) /
                                             static_cast<float>(sampled_feature_count));
    // Generate a fresh maximum-width random plan so every block uses an independent draw.
    workspace->random_generator.generate_uniform(target_slot.device_random_values,
                                                 2 * leverage_sketch.size());
    // Use the persistent alias table to select each leverage-weighted feature in O(1).
    cuda_sample_leverage_columns(
        workspace->prefetch_stream, leverage_sketch.device_probabilities,
        leverage_sketch.device_alias_probabilities, leverage_sketch.device_alias_indices,
        data_params.feature_count, target_slot.device_random_values, leverage_sketch.size(),
        sampled_feature_count, target_slot.device_sampled_column_indices,
        target_slot.device_sampled_column_scales);
    // Gather and scale the leverage-sampled columns of the next signed data block.
    cuda_gather_weighted_columns(workspace->prefetch_stream, device_next_signed_batch,
                                 target_slot.device_sampled_features,
                                 run_params.samples_per_iteration, data_params.feature_count,
                                 target_slot.device_sampled_column_indices,
                                 target_slot.device_sampled_column_scales,
                                 sampled_feature_count, scale_multiplier);
    constexpr float one{1.0f};
    constexpr float zero{0.0f};
    const int sample_count = run_params.samples_per_iteration;
    const int active_feature_count = static_cast<int>(sampled_feature_count);
    // Materialize only the lower Gram triangle consumed by the recurrence.
    check_cublas(cublasSsyrk(
        workspace->prefetch_cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, sample_count,
        active_feature_count, &one, target_slot.device_sampled_features, active_feature_count,
        &zero, target_slot.device_approximate_gram, sample_count));
    check_cuda(cudaEventRecord(target_slot.prefetch_done, workspace->prefetch_stream));
}
