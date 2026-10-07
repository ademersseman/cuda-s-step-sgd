#pragma once

#include "s_step_sgd.hpp"
#include <vector>

LeverageSketch prepare_leverage_sketch(const DataParams* data_params, const RunParams* run_params,
                                       const std::vector<float>& host_features);
void prefetch_leverage_features(const DataParams* data_params, const RunParams* run_params,
                                Workspace* workspace, float* device_next_signed_batch,
                                size_t slot_index, size_t iteration_index,
                                size_t sampled_feature_count);
