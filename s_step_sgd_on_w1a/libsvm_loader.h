#pragma once

#include "s_step_sgd.h"

void load_libsvm(
    DataParams* data_params,
    std::vector<float>& h_A,
    std::vector<float>& h_y);
