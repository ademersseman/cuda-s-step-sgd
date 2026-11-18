#include "s_step_sgd.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace {
void check_close(const char* field, double actual, double expected, double tolerance) {
    if (!(std::abs(actual - expected) <= tolerance))
        throw std::runtime_error(std::string(field) + " expected " + std::to_string(expected) +
                                 ", got " + std::to_string(actual));
}

void check_shape(int b, int s, int factor_width = 0) {
    const int n = b * s;
    std::vector<float> gram(n * n, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> margins(n), base(n), result(n), factor(n * factor_width);
    for (std::size_t i = 0; i < factor.size(); ++i)
        factor[i] = std::sin(0.193 * i) / std::sqrt(double(factor_width));
    for (int i = 0; i < n; ++i) {
        margins[i] = std::sin(0.43 * i);
        base[i] = 1.0 / (1.0 + std::exp(static_cast<double>(margins[i])));
        for (int j = 0; j < i / b * b; ++j) {
            if (factor_width == 0) {
                gram[j * n + i] = std::cos(0.31 * i + 0.07 * j);
            } else {
                double dot = 0.0;
                for (int k = 0; k < factor_width; ++k)
                    dot += double(factor[i * factor_width + k]) * factor[j * factor_width + k];
                gram[j * n + i] = dot;
            }
        }
    }

    CudaStream stream;
    DeviceBuffer<float> device_gram(gram.size()), device_margins(n), device_base(n);
    DeviceBuffer<RecurrenceStats> device_stats(1);
    check_cuda(cudaMemcpyAsync(device_gram, gram.data(), gram.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(device_base, base.data(), n * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemsetAsync(device_stats, 0xff, sizeof(RecurrenceStats), stream));
    DeviceBuffer<float> device_factor, device_prefix;
    if (factor_width > 0) {
        device_factor.allocate(factor.size());
        device_prefix.allocate(s * factor_width);
        check_cuda(cudaMemcpyAsync(device_factor, factor.data(), factor.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
    }
    // Reuse the statistics buffer across a nonzero correction and a zero correction.
    for (const float alpha : {0.125f, 0.0f}) {
        check_cuda(cudaMemcpyAsync(device_margins, margins.data(), n * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        if (factor_width > 0)
            cuda_factor_recurrence(stream, device_factor, device_base, device_margins, device_prefix,
                                   b, s, factor_width, alpha, device_stats);
        else
            cuda_fused_gram_recurrence(stream, device_gram, device_base, device_margins, n, b,
                                       alpha, device_stats);
        RecurrenceStats stats{};
        check_cuda(cudaMemcpyAsync(result.data(), device_margins, n * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        check_cuda(cudaMemcpyAsync(&stats, device_stats, sizeof(stats), cudaMemcpyDeviceToHost,
                                   stream));
        check_cuda(cudaStreamSynchronize(stream));
        double difference_squared = 0.0, base_squared = 0.0;
        for (int i = 0; i < n; ++i) {
            double margin = margins[i];
            for (int j = 0; j < i / b * b; ++j)
                margin += double(alpha) * gram[j * n + i] * base[j];
            const double residual = 1.0 / (1.0 + std::exp(margin));
            check_close("residual", result[i], residual, 3e-6);
            difference_squared += (residual - base[i]) * (residual - base[i]);
            base_squared += double(base[i]) * base[i];
        }
        check_close("correction sum of squares", stats.corrected_residual_difference_squared_sum,
                    difference_squared, 1e-7 + 1e-5 * difference_squared);
        check_close("base sum of squares", stats.base_residual_squared_sum, base_squared,
                    1e-5 * base_squared);
    }
}
}  // namespace

int main() {
    try {
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || (status == cudaSuccess && count == 0)) {
            std::cout << "No CUDA device available\n";
            return 77;
        }
        check_cuda(status);
        check_shape(5, 1);
        check_shape(1, 67);
        check_shape(7, 13);
        check_shape(33, 9);
        check_shape(32, 128);
        check_shape(7, 1, 2);
        check_shape(5, 13, 16);
        check_shape(3, 271, 64);
        check_shape(1, 17, 512);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
