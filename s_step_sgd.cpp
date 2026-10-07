#include "s_step_sgd.hpp"
#include "libsvm_loader.hpp"
#include "prefetch.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

void compute_metrics(const DataParams* data_params, Workspace* workspace, double& objective,
                     double& accuracy) {
    const float one{1.0f};
    const float zero{0.0f};
    check_cublas(cublasSgemv(workspace->cublas_handle, CUBLAS_OP_T, data_params->feature_count,
                             data_params->sample_count, &one, workspace->device_signed_features,
                             data_params->feature_count, workspace->device_weights, 1, &zero,
                             workspace->device_metric_margins, 1));
    check_cuda(cudaMemsetAsync(workspace->device_metric_sums, 0, sizeof(MetricSums),
                               workspace->compute_stream));
    cuda_accumulate_metrics(workspace->compute_stream, workspace->device_metric_margins,
                            workspace->device_labels, data_params->sample_count,
                            workspace->device_metric_sums);

    MetricSums metric_sums{};
    check_cuda(cudaMemcpyAsync(&metric_sums, workspace->device_metric_sums, sizeof(metric_sums),
                               cudaMemcpyDeviceToHost, workspace->compute_stream));
    check_cuda(cudaStreamSynchronize(workspace->compute_stream));
    objective = metric_sums.objective_sum / static_cast<double>(data_params->sample_count);
    accuracy = 100.0 * static_cast<double>(metric_sums.correct_count) /
               static_cast<double>(data_params->sample_count);
}

void launch_cublas_recurrence(const RunParams* run_params, Workspace* workspace,
                              size_t active_sampled_feature_count) {
    const int batch_size = run_params->batch_size;
    const int sampled_feature_count = static_cast<int>(active_sampled_feature_count);
    const long long sampled_batch_stride =
        static_cast<long long>(batch_size) * sampled_feature_count;
    const float one{1.0f};
    const float zero{0.0f};

    cuda_sigmoid(workspace->compute_stream, workspace->device_block_margins_or_residuals,
                 workspace->device_base_residuals, run_params->samples_per_iteration);
    check_cublas(cublasSgemmStridedBatched(
        workspace->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, sampled_feature_count, 1, batch_size,
        &one, workspace->sampled_batch_slots[workspace->current_slot_index].device_sampled_features,
        sampled_feature_count, sampled_batch_stride, workspace->device_base_residuals, batch_size,
        batch_size, &zero, workspace->device_projected_gradient_prefix, sampled_feature_count,
        sampled_feature_count, run_params->s));
    cuda_exclusive_prefix_columns(workspace->compute_stream,
                                  workspace->device_projected_gradient_prefix,
                                  active_sampled_feature_count, run_params->s);
    const float learning_rate_per_sample =
        run_params->learning_rate / static_cast<float>(batch_size);
    check_cublas(cublasSgemmStridedBatched(
        workspace->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, batch_size, 1, sampled_feature_count,
        &learning_rate_per_sample,
        workspace->sampled_batch_slots[workspace->current_slot_index].device_sampled_features,
        sampled_feature_count, sampled_batch_stride, workspace->device_projected_gradient_prefix,
        sampled_feature_count, sampled_feature_count, &one,
        workspace->device_block_margins_or_residuals, batch_size, batch_size, run_params->s));
    cuda_corrected_sigmoid_and_accumulate(
        workspace->compute_stream, workspace->device_block_margins_or_residuals,
        workspace->device_base_residuals, workspace->device_block_margins_or_residuals,
        run_params->samples_per_iteration, workspace->device_recurrence_stats);
}

cudaGraphExec_t recurrence_graph_for(const Workspace* workspace, size_t slot_index,
                                     size_t sampled_feature_count) {
    const auto graph =
        std::find_if(workspace->recurrence_graphs.begin(), workspace->recurrence_graphs.end(),
                     [slot_index, sampled_feature_count](const RecurrenceGraph& candidate) {
                         return candidate.slot_index == slot_index &&
                                candidate.sampled_feature_count == sampled_feature_count;
                     });
    if (graph == workspace->recurrence_graphs.end())
        throw std::runtime_error("missing recurrence CUDA graph");
    return graph->executable;
}

void prepare_recurrence_graphs(const RunParams* run_params, Workspace* workspace) {
    std::vector<size_t> sampled_feature_counts;
    for (const size_t candidate : {size_t{8}, size_t{16}, size_t{32}, size_t{64}}) {
        const size_t sampled_feature_count = std::min(candidate, workspace->leverage_sketch.size());
        if (std::find(sampled_feature_counts.begin(), sampled_feature_counts.end(),
                      sampled_feature_count) == sampled_feature_counts.end())
            sampled_feature_counts.push_back(sampled_feature_count);
    }

    const size_t original_slot_index = workspace->current_slot_index;
    for (const size_t sampled_feature_count : sampled_feature_counts) {
        // Initialize cuBLAS's kernels for this shape before stream capture.
        workspace->current_slot_index = 0;
        launch_cublas_recurrence(run_params, workspace, sampled_feature_count);
        check_cuda(cudaStreamSynchronize(workspace->compute_stream));

        for (size_t slot_index = 0; slot_index < workspace->sampled_batch_slots.size();
             ++slot_index) {
            workspace->current_slot_index = slot_index;
            RecurrenceGraph recurrence_graph{slot_index, sampled_feature_count};
            check_cuda(cudaStreamBeginCapture(workspace->compute_stream,
                                              cudaStreamCaptureModeThreadLocal));
            launch_cublas_recurrence(run_params, workspace, sampled_feature_count);
            check_cuda(cudaStreamEndCapture(workspace->compute_stream, &recurrence_graph.graph));
            check_cuda(
                cudaGraphInstantiate(&recurrence_graph.executable, recurrence_graph.graph, 0));
            workspace->recurrence_graphs.push_back(recurrence_graph);
        }
    }
    workspace->current_slot_index = original_slot_index;
}

void enter_recurrence(const DataParams* data_params, const RunParams* run_params,
                      Workspace* workspace, float* device_signed_batch,
                      size_t active_sampled_feature_count) {
    const float alpha{1};
    const float beta{0};

    // Compute each sample's signed margin at the base model for this s-step block.
    check_cublas(cublasSgemv(workspace->cublas_handle, CUBLAS_OP_T, data_params->feature_count,
                             run_params->samples_per_iteration, &alpha, device_signed_batch,
                             data_params->feature_count, workspace->device_weights, 1, &beta,
                             workspace->device_block_margins_or_residuals, 1));

    check_cuda(cudaGraphLaunch(recurrence_graph_for(workspace, workspace->current_slot_index,
                                                    active_sampled_feature_count),
                               workspace->compute_stream));
}

// ---------------- Train Function ----------------
void train(const DataParams* data_params, const RunParams* run_params, Workspace* workspace,
           ProfileStats* profile_stats) {
    double previous_objective{std::log(2.0)};
    double current_objective{0.0};
    double current_accuracy{0.0};

    // Form the signed design matrix by multiplying every sample row by its label.
    check_cublas(cublasSdgmm(workspace->cublas_handle, CUBLAS_SIDE_RIGHT,
                             data_params->feature_count, data_params->sample_count,
                             workspace->device_features, data_params->feature_count,
                             workspace->device_labels, 1, workspace->device_signed_features,
                             data_params->feature_count));
    // The compute and prefetch streams are nonblocking, so neither implicitly
    // waits for the other. Finish one-time label scaling before the sampled
    // feature prefetch reads device_signed_features.
    check_cuda(cudaStreamSynchronize(workspace->compute_stream));

    // Capture one cuBLAS recurrence graph for each buffer slot and adaptive sampled width.
    prepare_recurrence_graphs(run_params, workspace);

    // Gather the first block using its independently sampled leverage columns.
    if (run_params->iteration_count > 0)
        prefetch_leverage_features(data_params, run_params, workspace,
                                   workspace->device_signed_features, workspace->current_slot_index,
                                   0, workspace->leverage_sketch.size(0));

    const auto training_start = std::chrono::steady_clock::now();
    std::mt19937 controller_random_generator(run_params->sketch_seed ^ 0x9e3779b9U);
    std::uniform_real_distribution<float> uniform_distribution(0.0f, 1.0f);
    float smoothed_correction_ratio = 0.0f;
    bool has_smoothed_correction_ratio = false;
    bool has_pending_recurrence_stats = false;
    size_t pending_stats_block_index = 0;
    size_t sampled_feature_count_sum = 0;

    for (int iteration_index{0}; iteration_index < run_params->iteration_count; ++iteration_index) {
        const size_t block_index = static_cast<size_t>(iteration_index);
        // determine current batch start offset (wrap around if we exceed total samples)
        const size_t current_batch_start_offset{
            ((block_index * run_params->samples_per_iteration) % data_params->padded_sample_count)};
        float* device_signed_batch{workspace->device_signed_features +
                                   current_batch_start_offset * data_params->feature_count};

        const auto iteration_start = std::chrono::steady_clock::now();

        // wait for current iter prefetch to finish
        auto& current_batch_slot = workspace->sampled_batch_slots[workspace->current_slot_index];
        check_cuda(
            cudaStreamWaitEvent(workspace->compute_stream, current_batch_slot.prefetch_done, 0));

        // Compute reads from the current slot while prefetch writes into the other slot.
        const size_t next_slot_index{(workspace->current_slot_index + 1) % 2};
        auto& next_batch_slot = workspace->sampled_batch_slots[next_slot_index];

        // Wait until compute finishes reading a reused buffer before prefetch overwrites it.
        check_cuda(
            cudaStreamWaitEvent(workspace->prefetch_stream, next_batch_slot.compute_done, 0));

        // next batch pointer for prefetch
        const size_t next_batch_start_offset{
            ((block_index + 1) * run_params->samples_per_iteration) %
            data_params->padded_sample_count};
        float* device_next_signed_batch = workspace->device_signed_features +
                                          next_batch_start_offset * data_params->feature_count;

        // Gather the next block using its own leverage sample while this block is computed.
        if (iteration_index + 1 < run_params->iteration_count)
            prefetch_leverage_features(data_params, run_params, workspace, device_next_signed_batch,
                                       next_slot_index, block_index + 1,
                                       workspace->leverage_sketch.size(block_index + 1));

        const size_t active_sampled_feature_count = workspace->leverage_sketch.size(block_index);
        sampled_feature_count_sum += active_sampled_feature_count;
        enter_recurrence(data_params, run_params, workspace, device_signed_batch,
                         active_sampled_feature_count);

        // Record when recurrence finishes reading this buffer so prefetch can safely reuse it.
        check_cuda(cudaEventRecord(current_batch_slot.compute_done, workspace->compute_stream));

        const float learning_rate_per_sample =
            run_params->learning_rate / static_cast<float>(run_params->batch_size);
        const float one{1.0f};
        // Apply all corrected block residuals as x += (learning_rate / batch_size) A^T r.
        check_cublas(cublasSgemv(workspace->cublas_handle, CUBLAS_OP_N, data_params->feature_count,
                                 run_params->samples_per_iteration, &learning_rate_per_sample,
                                 device_signed_batch, data_params->feature_count,
                                 workspace->device_block_margins_or_residuals, 1, &one,
                                 workspace->device_weights, 1));

        // Consume the previous block's statistics while this block remains queued on the GPU.
        if (has_pending_recurrence_stats) {
            check_cuda(cudaEventSynchronize(workspace->recurrence_stats_copy_done));
            const auto& recurrence_stats = *workspace->host_recurrence_stats;
            const float correction_ratio =
                std::sqrt(recurrence_stats.corrected_residual_difference_squared_sum /
                          std::max(recurrence_stats.base_residual_squared_sum, 1e-20f));
            smoothed_correction_ratio =
                has_smoothed_correction_ratio
                    ? 0.9f * smoothed_correction_ratio + 0.1f * correction_ratio
                    : correction_ratio;
            has_smoothed_correction_ratio = true;
            workspace->leverage_sketch.sampled_feature_counts[pending_stats_block_index + 3] =
                stochastic_sampled_feature_count(smoothed_correction_ratio,
                                                 uniform_distribution(controller_random_generator),
                                                 workspace->leverage_sketch.size());
            has_pending_recurrence_stats = false;
        }

        // Copy this block's statistics only when they can control a future prefetched block.
        if (block_index + 3 < workspace->leverage_sketch.sketch_count) {
            check_cuda(cudaMemcpyAsync(workspace->host_recurrence_stats,
                                       workspace->device_recurrence_stats, sizeof(RecurrenceStats),
                                       cudaMemcpyDeviceToHost, workspace->compute_stream));
            check_cuda(
                cudaEventRecord(workspace->recurrence_stats_copy_done, workspace->compute_stream));
            pending_stats_block_index = block_index;
            has_pending_recurrence_stats = true;
        }

        // alternate buffers for next iteration
        workspace->current_slot_index = next_slot_index;

        if (run_params->print_interval != 0 && iteration_index != 0 &&
            iteration_index % run_params->print_interval == 0) {
            compute_metrics(data_params, workspace, current_objective, current_accuracy);

            const auto output_flags = std::cout.flags();
            const auto output_precision = std::cout.precision();
            std::cout << "Iters: " << iteration_index + 1 << "\t Objective: " << std::fixed
                      << std::setprecision(4) << current_objective
                      << "\t Training Accuracy: " << current_accuracy
                      << "%\t Obj val diff: " << std::scientific << std::setprecision(10)
                      << std::fabs(previous_objective - current_objective)
                      << "\t Time: " << std::fixed << std::setprecision(4)
                      << std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() -
                                                                  iteration_start)
                             .count()
                      << '\n';
            std::cout.flags(output_flags);
            std::cout.precision(output_precision);
            previous_objective = current_objective;
        }
    }
    check_cuda(cudaDeviceSynchronize());

    profile_stats->training_time_ms =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - training_start)
            .count();

    const double average_sampled_feature_count =
        static_cast<double>(sampled_feature_count_sum) /
        static_cast<double>(std::max(1, run_params->iteration_count));
    std::cout << "average_sampled_features: " << average_sampled_feature_count
              << "\nfinal_smoothed_correction_ratio: " << smoothed_correction_ratio << '\n';

    compute_metrics(data_params, workspace, current_objective, current_accuracy);
    const auto output_flags = std::cout.flags();
    const auto output_precision = std::cout.precision();
    std::cout << "Final Objective: " << std::fixed << std::setprecision(8) << current_objective
              << "\nFinal Training Accuracy: " << std::setprecision(6) << current_accuracy << "%\n";
    std::cout.flags(output_flags);
    std::cout.precision(output_precision);
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "  --batch-size N       samples per minibatch (default 16)\n"
              << "  --s N                minibatches per block iteration (default 4)\n"
              << "  --n-iters N          block iterations (default 100)\n"
              << "  --dataset FILE       LIBSVM or shared-binary dataset (default w1a.txt)\n"
              << "  --sampled-features N maximum leverage samples (default 64)\n"
              << "  --eta VALUE          learning rate (default 0.5)\n"
              << "  --weights-out FILE   write the final float32 weights\n\n"
              << "Legacy positional arguments remain supported:\n"
              << "  batch_size s n_iters dataset leverage sampled_features eta "
                 "[weights_out]\n";
}

const char* consume_option_value(int argc, char** argv, int& index, const std::string& option) {
    if (++index >= argc) {
        throw std::invalid_argument("Missing value for " + option);
    }
    return argv[index];
}

// ---------------- Main ----------------
int run_main(int argc, char** argv) {
    std::unique_ptr<RunParams> run_params{std::make_unique<RunParams>()};
    std::unique_ptr<DataParams> data_params{std::make_unique<DataParams>()};
    std::string weights_output_path{};

    for (int argument_index = 1; argument_index < argc; ++argument_index) {
        const std::string option{argv[argument_index]};
        if (option == "-h" || option == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (option == "--batch-size") {
            run_params->batch_size =
                std::max(1, std::stoi(consume_option_value(argc, argv, argument_index, option)));
        } else if (option == "--s") {
            run_params->s =
                std::max(1, std::stoi(consume_option_value(argc, argv, argument_index, option)));
        } else if (option == "--n-iters") {
            run_params->iteration_count =
                std::max(1, std::stoi(consume_option_value(argc, argv, argument_index, option)));
        } else if (option == "--dataset") {
            data_params->dataset_path = consume_option_value(argc, argv, argument_index, option);
        } else if (option == "--sampled-features") {
            run_params->requested_sampled_feature_count =
                std::stoull(consume_option_value(argc, argv, argument_index, option));
            if (run_params->requested_sampled_feature_count < 8)
                throw std::invalid_argument("--sampled-features must be at least 8");
        } else if (option == "--eta") {
            run_params->learning_rate =
                std::stof(consume_option_value(argc, argv, argument_index, option));
        } else if (option == "--weights-out") {
            weights_output_path = consume_option_value(argc, argv, argument_index, option);
        } else {
            throw std::invalid_argument("Unknown option: " + option);
        }
    }

    std::cout << "batch_size: " << run_params->batch_size << ", s: " << run_params->s
              << ", iteration_count: " << run_params->iteration_count
              << ", training_set: " << data_params->dataset_path << ", gram_mode: leverage"
              << ", requested_sampled_feature_count: "
              << run_params->requested_sampled_feature_count
              << ", learning_rate: " << run_params->learning_rate << '\n';
    // load raw data into host
    std::vector<float> host_features{};
    std::vector<float> host_labels{};
    load_dataset(data_params.get(), host_features, host_labels);
    data_params->sample_count = host_labels.size();
    std::cout << "dataset_samples: " << data_params->sample_count
              << ", dataset_features: " << data_params->feature_count;
    if (data_params->is_sstep_binary) {
        std::cout << ", dataset_format: sstep_binary";
    }
    std::cout << '\n';

    // Pad data to be multiple of s * batch_size
    run_params->samples_per_iteration = run_params->s * run_params->batch_size;

    // calculate extra samples(needed to pad end of dataset to make it divisible by
    // samples_per_iteration)
    const size_t remainder{data_params->sample_count % run_params->samples_per_iteration};
    const size_t extra_samples{remainder == 0 ? 0 : run_params->samples_per_iteration - remainder};
    for (size_t padding_sample = 0; padding_sample < extra_samples; ++padding_sample) {
        host_labels.push_back(0.0f);
        for (size_t feature_index = 0; feature_index < data_params->feature_count;
             ++feature_index) {
            host_features.push_back(0.0f);
        }
    }
    data_params->padded_sample_count = host_labels.size();

    if (static_cast<size_t>(run_params->samples_per_iteration) > data_params->padded_sample_count) {
        std::cerr << "Error: batch_size * s must be <= padded_sample_count\n";
        return 1;
    }

    std::cout << "block_iterations: " << run_params->iteration_count << '\n';
    run_params->print_interval = 0;

    LeverageSketch leverage_sketch =
        prepare_leverage_sketch(data_params.get(), run_params.get(), host_features);
    std::cout << "effective_sampled_features: " << leverage_sketch.size() << '\n';
    std::cout << "initial_sampled_features: " << leverage_sketch.size(0) << '\n';
    std::cout << "sampled_feature_controller: stochastic_residual_correction\n";
    std::cout << "feature_sketch_count: " << leverage_sketch.sketch_count << '\n';
    std::unique_ptr<Workspace> workspace{std::make_unique<Workspace>(
        data_params.get(), run_params.get(), std::move(leverage_sketch))};
    ProfileStats profile_stats{};

    // copy data to GPU
    check_cuda(
        cudaMemcpy(workspace->device_features, host_features.data(),
                   data_params->padded_sample_count * data_params->feature_count * sizeof(float),
                   cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(workspace->device_labels, host_labels.data(),
                          data_params->padded_sample_count * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cuda(
        cudaMemset(workspace->device_weights, 0, data_params->feature_count * sizeof(float)));
    // Workspace streams use cudaStreamNonBlocking and therefore do not inherit
    // ordering from these default-stream initialization operations.
    check_cuda(cudaDeviceSynchronize());

    train(data_params.get(), run_params.get(), workspace.get(), &profile_stats);

    if (!weights_output_path.empty()) {
        std::vector<float> weights(data_params->feature_count);
        check_cuda(cudaMemcpy(weights.data(), workspace->device_weights,
                              weights.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::ofstream output(weights_output_path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(weights.data()),
                     static_cast<std::streamsize>(weights.size() * sizeof(float)));
        if (!output)
            throw std::runtime_error("Could not write weights: " + weights_output_path);
    }

    std::cout << "\n=== Timing ===\n";
    std::cout << "Training Time: " << profile_stats.training_time_ms << " ms\n\n";

    return 0;
}

int main(int argc, char** argv) {
    try {
        return run_main(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
