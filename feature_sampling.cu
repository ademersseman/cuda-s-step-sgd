#include "feature_sampling.hpp"
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cusolverDn.h>
#include <numeric>
#include <random>
#include <source_location>
#include <stdexcept>

namespace {
void check_cusolver(cusolverStatus_t status,
                    const std::source_location& location = std::source_location::current()) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error("cuSOLVER error at " + call_site(location) + ": status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

void check_factorization_info(int* device_info) {
    int info = 0;
    check_cuda(cudaMemcpy(&info, device_info, sizeof(info), cudaMemcpyDeviceToHost));
    if (info != 0)
        throw std::runtime_error("GPU factorization failed with info=" + std::to_string(info));
}

void orthonormalize_columns(cusolverDnHandle_t solver_handle, float* matrix, int row_count,
                            int column_count, float* tau, float* solver_workspace,
                            int solver_workspace_size, int* device_info) {
    check_cusolver(cusolverDnSgeqrf(solver_handle, row_count, column_count, matrix, row_count, tau,
                                    solver_workspace, solver_workspace_size, device_info));
    check_factorization_info(device_info);
    check_cusolver(cusolverDnSorgqr(solver_handle, row_count, column_count, column_count, matrix,
                                    row_count, tau, solver_workspace, solver_workspace_size,
                                    device_info));
    check_factorization_info(device_info);
}

__global__ void calculate_leverage_scores(const float* right_basis, float* scores,
                                          int feature_count, int rank) {
    const int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= feature_count)
        return;
    float score = 0.0f;
    for (int component = 0; component < rank; ++component) {
        const float value = right_basis[feature + component * feature_count];
        score += value * value;
    }
    scores[feature] = score;
}

__global__ void normalize_leverage_scores(float* probabilities, int feature_count, float score_sum,
                                          float uniform_mix) {
    const int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= feature_count)
        return;
    const float uniform_probability = 1.0f / static_cast<float>(feature_count);
    const float normalized_score =
        score_sum > FLT_MIN ? probabilities[feature] / score_sum : uniform_probability;
    probabilities[feature] =
        (1.0f - uniform_mix) * normalized_score + uniform_mix * uniform_probability;
}
} // namespace

std::vector<float> estimate_column_leverage_probabilities(const DataParams* data_params,
                                                          const RunParams* run_params,
                                                          const std::vector<float>& features) {
    const size_t sampled_row_count =
        std::min(data_params->sample_count, run_params->leverage_row_count);
    const size_t rank =
        std::min({run_params->leverage_rank, sampled_row_count, data_params->feature_count});
    const size_t basis_size = rank;
    if (!sampled_row_count || !rank || !data_params->feature_count ||
        data_params->sample_count > static_cast<size_t>(INT_MAX) ||
        data_params->feature_count > static_cast<size_t>(INT_MAX) ||
        sampled_row_count > static_cast<size_t>(INT_MAX) ||
        basis_size > static_cast<size_t>(INT_MAX))
        throw std::invalid_argument("invalid leverage dimensions");

    const int row_count = static_cast<int>(sampled_row_count);
    const int feature_count = static_cast<int>(data_params->feature_count);
    const int subspace_size = static_cast<int>(basis_size);
    std::mt19937 random_generator(run_params->sketch_seed);
    std::vector<size_t> shuffled_rows(data_params->sample_count);
    std::iota(shuffled_rows.begin(), shuffled_rows.end(), 0);
    std::shuffle(shuffled_rows.begin(), shuffled_rows.end(), random_generator);

    // Row-major sampled features are column-major B^T to cuBLAS without a transpose copy.
    std::vector<float> sampled_features(sampled_row_count * data_params->feature_count);
    for (size_t sampled_row = 0; sampled_row < sampled_row_count; ++sampled_row)
        std::copy_n(features.data() + shuffled_rows[sampled_row] * data_params->feature_count,
                    data_params->feature_count,
                    sampled_features.data() + sampled_row * data_params->feature_count);

    std::normal_distribution<float> normal;
    std::vector<float> random_basis(data_params->feature_count * basis_size);
    for (float& value : random_basis)
        value = normal(random_generator);

    cublasHandle_t cublas_handle = nullptr;
    cusolverDnHandle_t solver_handle = nullptr;
    float* device_sampled_features = nullptr;
    float* device_random_basis = nullptr;
    float* device_left_basis = nullptr;
    float* device_right_basis = nullptr;
    float* device_tau = nullptr;
    float* device_solver_workspace = nullptr;
    float* device_probabilities = nullptr;
    int* device_info = nullptr;

    try {
        check_cublas(cublasCreate(&cublas_handle));
        check_cusolver(cusolverDnCreate(&solver_handle));
        check_cuda(cudaMalloc(&device_sampled_features, sampled_features.size() * sizeof(float)));
        check_cuda(cudaMalloc(&device_random_basis, random_basis.size() * sizeof(float)));
        check_cuda(cudaMalloc(&device_left_basis, sampled_row_count * basis_size * sizeof(float)));
        check_cuda(cudaMalloc(&device_right_basis,
                              data_params->feature_count * basis_size * sizeof(float)));
        check_cuda(cudaMalloc(&device_tau, basis_size * sizeof(float)));
        check_cuda(cudaMalloc(&device_probabilities, data_params->feature_count * sizeof(float)));
        check_cuda(cudaMalloc(&device_info, sizeof(int)));
        check_cuda(cudaMemcpy(device_sampled_features, sampled_features.data(),
                              sampled_features.size() * sizeof(float), cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(device_random_basis, random_basis.data(),
                              random_basis.size() * sizeof(float), cudaMemcpyHostToDevice));

        int left_geqrf_workspace_size = 0;
        int left_orgqr_workspace_size = 0;
        int right_geqrf_workspace_size = 0;
        int right_orgqr_workspace_size = 0;
        check_cusolver(cusolverDnSgeqrf_bufferSize(solver_handle, row_count, subspace_size,
                                                   device_left_basis, row_count,
                                                   &left_geqrf_workspace_size));
        check_cusolver(cusolverDnSorgqr_bufferSize(solver_handle, row_count, subspace_size,
                                                   subspace_size, device_left_basis, row_count,
                                                   device_tau, &left_orgqr_workspace_size));
        check_cusolver(cusolverDnSgeqrf_bufferSize(solver_handle, feature_count, subspace_size,
                                                   device_right_basis, feature_count,
                                                   &right_geqrf_workspace_size));
        check_cusolver(cusolverDnSorgqr_bufferSize(solver_handle, feature_count, subspace_size,
                                                   subspace_size, device_right_basis, feature_count,
                                                   device_tau, &right_orgqr_workspace_size));
        const int solver_workspace_size =
            std::max({left_geqrf_workspace_size, left_orgqr_workspace_size,
                      right_geqrf_workspace_size, right_orgqr_workspace_size});
        check_cuda(cudaMalloc(&device_solver_workspace,
                              static_cast<size_t>(solver_workspace_size) * sizeof(float)));

        constexpr float one = 1.0f;
        constexpr float zero = 0.0f;
        // Form and orthonormalize B * Omega.
        check_cublas(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, row_count, subspace_size,
                                 feature_count, &one, device_sampled_features, feature_count,
                                 device_random_basis, feature_count, &zero, device_left_basis,
                                 row_count));
        orthonormalize_columns(solver_handle, device_left_basis, row_count, subspace_size,
                               device_tau, device_solver_workspace, solver_workspace_size,
                               device_info);

        // Map the sampled column space back to an orthonormal rank-k feature basis.
        check_cublas(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, feature_count,
                                 subspace_size, row_count, &one, device_sampled_features,
                                 feature_count, device_left_basis, row_count, &zero,
                                 device_right_basis, feature_count));
        orthonormalize_columns(solver_handle, device_right_basis, feature_count, subspace_size,
                               device_tau, device_solver_workspace, solver_workspace_size,
                               device_info);
        constexpr int threads_per_block = 256;
        const int block_count = (feature_count + threads_per_block - 1) / threads_per_block;
        calculate_leverage_scores<<<block_count, threads_per_block>>>(
            device_right_basis, device_probabilities, feature_count, subspace_size);
        check_cuda(cudaGetLastError());

        float leverage_sum = 0.0f;
        check_cublas(
            cublasSasum(cublas_handle, feature_count, device_probabilities, 1, &leverage_sum));
        const float uniform_mix = std::clamp(run_params->uniform_probability_mix, 0.0f, 1.0f);
        normalize_leverage_scores<<<block_count, threads_per_block>>>(
            device_probabilities, feature_count, leverage_sum, uniform_mix);
        check_cuda(cudaGetLastError());

        std::vector<float> probabilities(data_params->feature_count);
        check_cuda(cudaMemcpy(probabilities.data(), device_probabilities,
                              probabilities.size() * sizeof(float), cudaMemcpyDeviceToHost));

        cudaFree(device_sampled_features);
        cudaFree(device_random_basis);
        cudaFree(device_left_basis);
        cudaFree(device_right_basis);
        cudaFree(device_tau);
        cudaFree(device_solver_workspace);
        cudaFree(device_probabilities);
        cudaFree(device_info);
        cusolverDnDestroy(solver_handle);
        cublasDestroy(cublas_handle);
        return probabilities;
    } catch (...) {
        cudaFree(device_sampled_features);
        cudaFree(device_random_basis);
        cudaFree(device_left_basis);
        cudaFree(device_right_basis);
        cudaFree(device_tau);
        cudaFree(device_solver_workspace);
        cudaFree(device_probabilities);
        cudaFree(device_info);
        cusolverDnDestroy(solver_handle);
        cublasDestroy(cublas_handle);
        throw;
    }
}

void sample_leverage_columns(const std::vector<float>& probabilities, size_t requested_sample_count,
                             unsigned int seed, std::vector<size_t>& sampled_column_indices,
                             std::vector<float>& sampled_column_scales) {
    if (probabilities.empty() || !requested_sample_count)
        throw std::invalid_argument("empty feature sample");
    std::mt19937 random_generator(seed + 2);
    std::discrete_distribution<size_t> draw_column(probabilities.begin(), probabilities.end());
    sampled_column_indices.resize(requested_sample_count);
    sampled_column_scales.resize(requested_sample_count);
    for (size_t sample_index = 0; sample_index < requested_sample_count; ++sample_index) {
        sampled_column_indices[sample_index] = draw_column(random_generator);
        sampled_column_scales[sample_index] =
            1.0f / std::sqrt(float(requested_sample_count) *
                             probabilities[sampled_column_indices[sample_index]]);
    }
}
