#pragma once
#include "s_step_sgd.hpp"
#include <vector>
// Estimate feature probabilities and construct their constant-time alias sampler.
void estimate_column_leverage_probabilities(const DataParams& data_params,
                                            const RunParams& run_params,
                                            const float* device_features,
                                            float* device_probabilities,
                                            float* device_alias_probabilities,
                                            int* device_alias_indices);
// Convert uniform device random values into leverage-sampled columns and scales.
void cuda_sample_leverage_columns(cudaStream_t stream, const float* device_probabilities,
                                  const float* device_alias_probabilities,
                                  const int* device_alias_indices,
                                  std::size_t feature_count, const float* device_random_values,
                                  std::size_t maximum_sample_count,
                                  std::size_t requested_sample_count,
                                  std::size_t* device_sampled_column_indices,
                                  float* device_sampled_column_scales);
