#include "prefetch.hpp"
#include "feature_sampling.hpp"

#include <algorithm>
#include <cmath>

LeverageSketch prepare_leverage_sketch(const DataParams* data_params, const RunParams* run_params,
                                       const std::vector<float>& host_features) {
    LeverageSketch leverage_sketch{};
    leverage_sketch.sampled_feature_count = run_params->requested_sampled_feature_count;
    const auto probabilities =
        estimate_column_leverage_probabilities(data_params, run_params, host_features);
    leverage_sketch.sketch_count = static_cast<size_t>(std::max(1, run_params->iteration_count));
    leverage_sketch.sampled_feature_counts.reserve(leverage_sketch.sketch_count);
    leverage_sketch.sketch_offsets.reserve(leverage_sketch.sketch_count);
    leverage_sketch.sampled_column_indices.reserve(leverage_sketch.sketch_count *
                                                   leverage_sketch.sampled_feature_count);
    leverage_sketch.sampled_column_scales.reserve(leverage_sketch.sketch_count *
                                                  leverage_sketch.sampled_feature_count);
    for (size_t sketch_index = 0; sketch_index < leverage_sketch.sketch_count; ++sketch_index) {
        leverage_sketch.sampled_feature_counts.push_back(
            std::min<size_t>(8, leverage_sketch.sampled_feature_count));
        leverage_sketch.sketch_offsets.push_back(leverage_sketch.sampled_column_indices.size());
        std::vector<size_t> sampled_column_indices;
        std::vector<float> sampled_column_scales;
        sample_leverage_columns(probabilities, leverage_sketch.sampled_feature_count,
                                run_params->sketch_seed + static_cast<unsigned int>(sketch_index),
                                sampled_column_indices, sampled_column_scales);
        leverage_sketch.sampled_column_indices.insert(leverage_sketch.sampled_column_indices.end(),
                                                      sampled_column_indices.begin(),
                                                      sampled_column_indices.end());
        leverage_sketch.sampled_column_scales.insert(leverage_sketch.sampled_column_scales.end(),
                                                     sampled_column_scales.begin(),
                                                     sampled_column_scales.end());
    }
    leverage_sketch.upload();
    return leverage_sketch;
}

void prefetch_leverage_features(const DataParams* data_params, const RunParams* run_params,
                                Workspace* workspace, float* device_next_signed_batch,
                                size_t slot_index, size_t iteration_index,
                                size_t sampled_feature_count) {
    auto& target_slot = workspace->sampled_batch_slots[slot_index];
    const auto& leverage_sketch = workspace->leverage_sketch;
    const size_t sketch_offset = leverage_sketch.offset(iteration_index);
    const float scale_multiplier = std::sqrt(static_cast<float>(leverage_sketch.size()) /
                                             static_cast<float>(sampled_feature_count));
    // Gather and scale the leverage-sampled columns of the next signed data block.
    cuda_gather_weighted_columns(workspace->prefetch_stream, device_next_signed_batch,
                                 target_slot.device_sampled_features,
                                 run_params->samples_per_iteration, data_params->feature_count,
                                 leverage_sketch.device_sampled_column_indices + sketch_offset,
                                 leverage_sketch.device_sampled_column_scales + sketch_offset,
                                 sampled_feature_count, scale_multiplier);
    check_cuda(cudaEventRecord(target_slot.prefetch_done, workspace->prefetch_stream));
}
