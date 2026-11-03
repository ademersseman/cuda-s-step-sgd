#include "feature_sampling.hpp"
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cusolverDn.h>
#include <source_location>
#include <stdexcept>
#include <vector>

namespace {
// Throw a source-aware exception when a cuSOLVER call fails.
void check_cusolver(cusolverStatus_t status,
                    const std::source_location& location = std::source_location::current()) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error("cuSOLVER error at " + call_site(location) + ": status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

// Copy a cuSOLVER status value to the host and reject failed factorizations.
void check_factorization_info(int* device_info) {
    int info = 0;
    check_cuda(cudaMemcpy(&info, device_info, sizeof(info), cudaMemcpyDeviceToHost));
    if (info != 0)
        throw std::runtime_error("GPU factorization failed with info=" + std::to_string(info));
}

// Convert a column-major matrix into an explicit orthonormal QR basis in place.
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

// Convert each right-basis row norm directly into a normalized sampling probability.
__global__ void calculate_leverage_probabilities(const float* right_basis, float* probabilities,
                                                 int feature_count, int rank,
                                                 float uniform_mix) {
    const int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= feature_count)
        return;
    float score = 0.0f;
    for (int component = 0; component < rank; ++component) {
        const float value = right_basis[feature + component * feature_count];
        score = fmaf(value, value, score);
    }
    const float uniform_probability = 1.0f / static_cast<float>(feature_count);
    probabilities[feature] =
        (1.0f - uniform_mix) * score / static_cast<float>(rank) +
        uniform_mix * uniform_probability;
}

// Gather randomly selected dataset rows into a compact row-major device matrix.
__global__ void gather_sampled_rows(const float* features, float* sampled_features,
                                    const double* random_values, std::size_t sample_count,
                                    std::size_t feature_count,
                                    std::size_t sampled_row_count) {
    for (std::size_t sampled_row = blockIdx.x; sampled_row < sampled_row_count;
         sampled_row += gridDim.x) {
        const std::size_t source_row =
            sampled_row_count == sample_count
                ? sampled_row
                : min(static_cast<std::size_t>(random_values[sampled_row] * sample_count),
                      sample_count - 1);
        for (std::size_t feature = threadIdx.x; feature < feature_count;
             feature += blockDim.x)
            sampled_features[sampled_row * feature_count + feature] =
                features[source_row * feature_count + feature];
    }
}

// Draw columns from a Walker-Vose alias table and form importance scales.
__global__ void sample_leverage_columns(const float* probabilities,
                                        const float* alias_probabilities,
                                        const int* alias_indices, int feature_count,
                                        const float* random_values,
                                        std::size_t maximum_sample_count,
                                        std::size_t requested_sample_count,
                                        std::size_t* sampled_column_indices,
                                        float* sampled_column_scales) {
    const std::size_t sample_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sample_index >= requested_sample_count)
        return;

    const int bucket = min(static_cast<int>(random_values[sample_index] * feature_count),
                           feature_count - 1);
    const float threshold = random_values[maximum_sample_count + sample_index];
    const int selected_column = threshold <= alias_probabilities[bucket]
                                    ? bucket
                                    : alias_indices[bucket];
    sampled_column_indices[sample_index] = static_cast<std::size_t>(selected_column);
    sampled_column_scales[sample_index] =
        rsqrtf(static_cast<float>(maximum_sample_count) *
               fmaxf(probabilities[selected_column], FLT_MIN));
}

// Build a Walker-Vose table once from normalized leverage probabilities.
void build_alias_table(std::vector<float>& probabilities,
                       std::vector<float>& alias_probabilities,
                       std::vector<int>& alias_indices) {
    const int feature_count = static_cast<int>(probabilities.size());
    double probability_sum = 0.0;
    for (const float probability : probabilities)
        probability_sum += probability;
    if (probability_sum <= 0.0)
        throw std::runtime_error("invalid leverage probability sum");

    std::vector<double> scaled_probabilities(feature_count);
    std::vector<int> small;
    std::vector<int> large;
    small.reserve(feature_count);
    large.reserve(feature_count);
    for (int feature = 0; feature < feature_count; ++feature) {
        probabilities[feature] =
            static_cast<float>(probabilities[feature] / probability_sum);
        scaled_probabilities[feature] =
            static_cast<double>(probabilities[feature]) * feature_count;
        (scaled_probabilities[feature] < 1.0 ? small : large).push_back(feature);
    }

    alias_probabilities.assign(feature_count, 1.0f);
    alias_indices.resize(feature_count);
    while (!small.empty() && !large.empty()) {
        const int small_feature = small.back();
        small.pop_back();
        const int large_feature = large.back();
        large.pop_back();
        alias_probabilities[small_feature] =
            static_cast<float>(scaled_probabilities[small_feature]);
        alias_indices[small_feature] = large_feature;
        scaled_probabilities[large_feature] += scaled_probabilities[small_feature] - 1.0;
        (scaled_probabilities[large_feature] < 1.0 ? small : large).push_back(large_feature);
    }
    for (const int feature : small)
        alias_indices[feature] = feature;
    for (const int feature : large)
        alias_indices[feature] = feature;
}
} // namespace

// Build a randomized basis and retain normalized probabilities plus their alias table.
void estimate_column_leverage_probabilities(const DataParams& data_params,
                                            const RunParams& run_params,
                                            const float* device_features,
                                            float* device_probabilities,
                                            float* device_alias_probabilities,
                                            int* device_alias_indices) {
    const std::size_t sampled_row_count =
        std::min(data_params.sample_count, run_params.leverage_row_count);
    const std::size_t rank =
        std::min({run_params.leverage_rank, sampled_row_count, data_params.feature_count});
    const std::size_t basis_size = rank;
    if (!sampled_row_count || !rank || !data_params.feature_count ||
        data_params.sample_count > static_cast<std::size_t>(INT_MAX) ||
        data_params.feature_count > static_cast<std::size_t>(INT_MAX) ||
        sampled_row_count > static_cast<std::size_t>(INT_MAX) ||
        basis_size > static_cast<std::size_t>(INT_MAX))
        throw std::invalid_argument("invalid leverage dimensions");

    const int row_count = static_cast<int>(sampled_row_count);
    const int feature_count = static_cast<int>(data_params.feature_count);
    const int subspace_size = static_cast<int>(basis_size);
    const std::size_t sampled_element_count = sampled_row_count * data_params.feature_count;
    const std::size_t random_basis_element_count = data_params.feature_count * basis_size;
    const std::size_t generated_normal_count =
        random_basis_element_count + random_basis_element_count % 2;
    CurandGenerator random_generator(run_params.sketch_seed);

    cublasHandle_t cublas_handle = nullptr;
    cusolverDnHandle_t solver_handle = nullptr;
    double* device_random_rows = nullptr;
    float* device_sampled_features = nullptr;
    float* device_random_basis = nullptr;
    float* device_left_basis = nullptr;
    float* device_right_basis = nullptr;
    float* device_tau = nullptr;
    float* device_solver_workspace = nullptr;
    int* device_info = nullptr;

    try {
        check_cublas(cublasCreate(&cublas_handle));
        check_cusolver(cusolverDnCreate(&solver_handle));
        check_cuda(cudaMalloc(&device_random_rows, sampled_row_count * sizeof(double)));
        check_cuda(cudaMalloc(&device_sampled_features, sampled_element_count * sizeof(float)));
        check_cuda(cudaMalloc(&device_random_basis, generated_normal_count * sizeof(float)));
        check_cuda(cudaMalloc(&device_left_basis, sampled_row_count * basis_size * sizeof(float)));
        check_cuda(cudaMalloc(&device_right_basis,
                              data_params.feature_count * basis_size * sizeof(float)));
        check_cuda(cudaMalloc(&device_tau, basis_size * sizeof(float)));
        check_cuda(cudaMalloc(&device_info, sizeof(int)));
        if (sampled_row_count != data_params.sample_count)
            random_generator.generate_uniform(device_random_rows, sampled_row_count);
        random_generator.generate_normal(device_random_basis, generated_normal_count);
        constexpr int threads_per_block = 256;
        const unsigned int gather_block_count =
            static_cast<unsigned int>(std::min<std::size_t>(sampled_row_count, 65535));
        gather_sampled_rows<<<gather_block_count, threads_per_block>>>(
            device_features, device_sampled_features, device_random_rows,
            data_params.sample_count, data_params.feature_count, sampled_row_count);
        check_cuda(cudaGetLastError());

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
                              static_cast<std::size_t>(solver_workspace_size) * sizeof(float)));

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
        const int block_count = (feature_count + threads_per_block - 1) / threads_per_block;
        const float uniform_mix = std::clamp(run_params.uniform_probability_mix, 0.0f, 1.0f);
        calculate_leverage_probabilities<<<block_count, threads_per_block>>>(
            device_right_basis, device_probabilities, feature_count, subspace_size, uniform_mix);
        check_cuda(cudaGetLastError());

        std::vector<float> host_probabilities(feature_count);
        std::vector<float> host_alias_probabilities;
        std::vector<int> host_alias_indices;
        check_cuda(cudaMemcpy(host_probabilities.data(), device_probabilities,
                              host_probabilities.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        build_alias_table(host_probabilities, host_alias_probabilities, host_alias_indices);
        check_cuda(cudaMemcpy(device_probabilities, host_probabilities.data(),
                              host_probabilities.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(device_alias_probabilities, host_alias_probabilities.data(),
                              host_alias_probabilities.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(device_alias_indices, host_alias_indices.data(),
                              host_alias_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

        cudaFree(device_random_rows);
        cudaFree(device_sampled_features);
        cudaFree(device_random_basis);
        cudaFree(device_left_basis);
        cudaFree(device_right_basis);
        cudaFree(device_tau);
        cudaFree(device_solver_workspace);
        cudaFree(device_info);
        cusolverDnDestroy(solver_handle);
        cublasDestroy(cublas_handle);
    } catch (...) {
        cudaFree(device_random_rows);
        cudaFree(device_sampled_features);
        cudaFree(device_random_basis);
        cudaFree(device_left_basis);
        cudaFree(device_right_basis);
        cudaFree(device_tau);
        cudaFree(device_solver_workspace);
        cudaFree(device_info);
        cusolverDnDestroy(solver_handle);
        cublasDestroy(cublas_handle);
        throw;
    }
}

// Convert device uniform draws into leverage-sampled feature indices and importance scales.
void cuda_sample_leverage_columns(cudaStream_t stream, const float* device_probabilities,
                                  const float* device_alias_probabilities,
                                  const int* device_alias_indices,
                                  std::size_t feature_count, const float* device_random_values,
                                  std::size_t maximum_sample_count,
                                  std::size_t requested_sample_count,
                                  std::size_t* device_sampled_column_indices,
                                  float* device_sampled_column_scales) {
    if (!feature_count || !requested_sample_count || requested_sample_count > maximum_sample_count ||
        feature_count > static_cast<std::size_t>(INT_MAX))
        throw std::invalid_argument("empty feature sample");
    constexpr int threads_per_block = 256;
    const std::size_t block_count =
        (requested_sample_count + threads_per_block - 1) / threads_per_block;
    sample_leverage_columns<<<static_cast<unsigned int>(block_count), threads_per_block, 0, stream>>>(
        device_probabilities, device_alias_probabilities, device_alias_indices,
        static_cast<int>(feature_count), device_random_values, maximum_sample_count,
        requested_sample_count, device_sampled_column_indices, device_sampled_column_scales);
    check_cuda(cudaGetLastError());
}
