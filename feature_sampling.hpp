#pragma once
#include "s_step_sgd.hpp"
#include <vector>
std::vector<float> estimate_column_leverage_probabilities(const DataParams* data_params,
                                                          const RunParams* run_params,
                                                          const std::vector<float>& features);
void sample_leverage_columns(const std::vector<float>& probabilities, size_t requested_sample_count,
                             unsigned int seed, std::vector<size_t>& sampled_column_indices,
                             std::vector<float>& sampled_column_scales);
