#include "prefetch.h"

#include <cstdlib>
#include <numeric>
#include <vector>

std::vector<size_t> sample_columns_with_replacement(const std::vector<float>& weights, size_t l) {
    if (weights.empty() || l == 0) {
        return {};
    }

    std::vector<float> cumsum(weights.size(), 0.0f);
    cumsum[0] = weights[0];
    for (size_t i{1}; i < weights.size(); ++i) {
        cumsum[i] = cumsum[i - 1] + weights[i];
    }

    std::vector<size_t> sampled{};
    sampled.reserve(l);

    for (size_t i{0}; i < l; ++i) {
        const float rand_val{static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)};
        for (size_t j{0}; j < cumsum.size(); ++j) {
            if (rand_val <= cumsum[j]) {
                sampled.push_back(j);
                break;
            }
        }
    }

    return sampled;
}

void launch_full_prefetch(
    const DataParams* data_params,
    const RunParams* s_step_params,
    Workspace* workspace,
    float* d_batch_A,
    size_t target_buf)
{
    cudaEventRecord(workspace->gram_overhead_prefetch_done, workspace->prefetch_stream);

    const float alpha{1.0f};
    const float beta{0.0f};
    cublasSgemm(
        workspace->prefetch_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        s_step_params->samples_per_iter,
        s_step_params->samples_per_iter,
        data_params->n_features,
        &alpha,
        d_batch_A,
        data_params->n_features,
        d_batch_A,
        data_params->n_features,
        &beta,
        workspace->d_G[target_buf],
        s_step_params->samples_per_iter);

    cudaEventRecord(workspace->gram_prefetch_done, workspace->prefetch_stream);
}

void launch_uniform_prefetch(
    const DataParams* data_params,
    const RunParams* s_step_params,
    Workspace* workspace,
    float* d_batch_A,
    size_t target_buf)
{
    for (size_t i{0}; i < s_step_params->approx_gram_l; ++i) {
        const size_t src_col{static_cast<size_t>(std::rand()) % data_params->n_features};
        cublasScopy(
            workspace->prefetch_handle,
            s_step_params->samples_per_iter,
            d_batch_A + src_col,
            data_params->n_features,
            workspace->d_batch_A_approx[target_buf] + i,
            s_step_params->approx_gram_l);
    }

    cudaEventRecord(workspace->gram_overhead_prefetch_done, workspace->prefetch_stream);

    const float alpha_approx{s_step_params->approx_gram_l / static_cast<float>(data_params->n_features)};
    const float beta{0.0f};
    cublasSgemm(
        workspace->prefetch_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        s_step_params->samples_per_iter,
        s_step_params->samples_per_iter,
        s_step_params->approx_gram_l,
        &alpha_approx,
        workspace->d_batch_A_approx[target_buf],
        s_step_params->approx_gram_l,
        workspace->d_batch_A_approx[target_buf],
        s_step_params->approx_gram_l,
        &beta,
        workspace->d_G[target_buf],
        s_step_params->samples_per_iter);

    cudaEventRecord(workspace->gram_prefetch_done, workspace->prefetch_stream);
}

void launch_scoring_prefetch(
    const DataParams* data_params,
    const RunParams* s_step_params,
    Workspace* workspace,
    float* d_batch_A,
    size_t target_buf,
    size_t iters)
{
    const size_t num_iters{data_params->total_samples / s_step_params->samples_per_iter};
    if (num_iters == 0) {
        return;
    }

    iters = (iters + 1) % num_iters;
    if (!workspace->cache_valid[iters]) {
        for (size_t i{0}; i < data_params->n_features; ++i) {
            cublasSdot(
                workspace->prefetch_handle,
                s_step_params->samples_per_iter,
                d_batch_A + i,
                data_params->n_features,
                d_batch_A + i,
                data_params->n_features,
                workspace->d_scores + i);
        }

        cudaStreamSynchronize(workspace->prefetch_stream);
        cudaMemcpy(
            workspace->score_cache[iters].data(),
            workspace->d_scores,
            data_params->n_features * sizeof(float),
            cudaMemcpyDeviceToHost);

        const float sum{std::accumulate(workspace->score_cache[iters].begin(), workspace->score_cache[iters].end(), 0.0f)};
        for (auto& s : workspace->score_cache[iters]) {
            s /= sum;
        }

        workspace->cache_valid[iters] = true;
    }

    std::vector<size_t> sampled_cols = sample_columns_with_replacement(workspace->score_cache[iters], s_step_params->approx_gram_l);
    for (size_t i{0}; i < s_step_params->approx_gram_l; ++i) {
        cublasScopy(
            workspace->prefetch_handle,
            s_step_params->samples_per_iter,
            d_batch_A + sampled_cols[i],
            data_params->n_features,
            workspace->d_batch_A_approx[target_buf] + i,
            s_step_params->approx_gram_l);
    }

    cudaEventRecord(workspace->gram_overhead_prefetch_done, workspace->prefetch_stream);

    const float alpha_approx{s_step_params->approx_gram_l / static_cast<float>(data_params->n_features)};
    const float beta{0.0f};
    cublasSgemm(
        workspace->prefetch_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        s_step_params->samples_per_iter,
        s_step_params->samples_per_iter,
        s_step_params->approx_gram_l,
        &alpha_approx,
        workspace->d_batch_A_approx[target_buf],
        s_step_params->approx_gram_l,
        workspace->d_batch_A_approx[target_buf],
        s_step_params->approx_gram_l,
        &beta,
        workspace->d_G[target_buf],
        s_step_params->samples_per_iter);

    cudaEventRecord(workspace->gram_prefetch_done, workspace->prefetch_stream);
}

void prefetch_gram(
    const DataParams* data_params,
    const RunParams* s_step_params,
    Workspace* workspace,
    float* d_next_batch_A,
    size_t buffer,
    size_t iters)
{
    if (s_step_params->approx_gram && s_step_params->approx_gram_type == ApproxGramType::Uniform) {
        launch_uniform_prefetch(data_params, s_step_params, workspace, d_next_batch_A, buffer);
    } else if (s_step_params->approx_gram && s_step_params->approx_gram_type == ApproxGramType::Scoring) {
        launch_scoring_prefetch(data_params, s_step_params, workspace, d_next_batch_A, buffer, iters + 1);
    } else {
        launch_full_prefetch(data_params, s_step_params, workspace, d_next_batch_A, buffer);
    }
}
