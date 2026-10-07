#pragma once

#include "s_step_sgd.hpp"

void load_dataset(DataParams* data_params, std::vector<float>& host_features,
                  std::vector<float>& host_labels);
