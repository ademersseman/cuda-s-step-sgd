#pragma once

#include "s_step_sgd.hpp"

// Load either an SSTEP binary dataset or legacy LIBSVM text into dense host arrays.
void load_dataset(DataParams& data_params, std::vector<float>& host_features,
                  std::vector<float>& host_labels);
