#pragma once

#include "s_step_sgd.hpp"
#include <vector>

// Prepare persistent GPU leverage probabilities and initialize each block's active width.
LeverageSketch prepare_leverage_sketch(const DataParams& data_params, const RunParams& run_params,
                                       const float* device_features);
// Asynchronously gather sampled features and construct their full Gram in one buffer slot.
void prefetch_leverage_features(const DataParams& data_params, const RunParams& run_params,
                                Workspace* workspace, float* device_next_signed_batch,
                                std::size_t slot_index, std::size_t sampled_feature_count);
