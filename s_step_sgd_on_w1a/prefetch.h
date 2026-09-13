#pragma once

#include "s_step_sgd.h"

void prefetch_gram(
    const DataParams* data_params,
    const RunParams* s_step_params,
    Workspace* workspace,
    float* d_next_batch_A,
    size_t buffer,
    size_t iters);
