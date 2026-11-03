#include "s_step_sgd.hpp"
#include "dataset_loader.hpp"
#include "prefetch.hpp"

#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Compute the full-dataset logistic objective and classification accuracy on the GPU.
void compute_metrics(const DataParams& data_params, Workspace* workspace, double& objective,
                     double& accuracy) {
    nvtx3::scoped_range metrics_range{"metrics"};
    constexpr float one{1.0f};
    constexpr float zero{0.0f};
    // Compute signed margins for every sample using the current model.
    check_cublas(cublasSgemv(workspace->cublas_handle, CUBLAS_OP_T, data_params.feature_count,
                             data_params.sample_count, &one, workspace->device_signed_features,
                             data_params.feature_count, workspace->device_weights, 1, &zero,
                             workspace->device_metric_margins, 1));
    // Transform and reduce per-sample losses and correct predictions with CUB.
    cuda_accumulate_metrics(workspace->compute_stream, workspace->device_metric_margins,
                            workspace->device_labels, data_params.sample_count,
                            workspace->device_metric_reduction_workspace,
                            workspace->metric_reduction_workspace_size,
                            workspace->device_metric_sums);

    MetricSums metric_sums{};
    check_cuda(cudaMemcpyAsync(&metric_sums, workspace->device_metric_sums, sizeof(metric_sums),
                               cudaMemcpyDeviceToHost, workspace->compute_stream));
    check_cuda(cudaStreamSynchronize(workspace->compute_stream));
    objective = metric_sums.objective_sum / static_cast<double>(data_params.sample_count);
    accuracy = 100.0 * static_cast<double>(metric_sums.correct_count) /
               static_cast<double>(data_params.sample_count);
}

// Copy the current model to the host and write it as raw float32 weights.
void write_weights(const DataParams& data_params, Workspace* workspace,
                   const std::string& output_path) {
    std::vector<float> weights(data_params.feature_count);
    check_cuda(cudaMemcpy(weights.data(), workspace->device_weights,
                          weights.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::ofstream output(output_path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(weights.data()),
                 static_cast<std::streamsize>(weights.size() * sizeof(float)));
    if (!output)
        throw std::runtime_error("Could not write weights: " + output_path);
}

// Save an iteration checkpoint and report its cumulative training time.
void save_training_checkpoint(const DataParams& data_params, const RunParams& run_params,
                              Workspace* workspace, int completed_iterations,
                              float cumulative_training_time_ms) {
    const std::string checkpoint_path = run_params.checkpoint_prefix + "_iter" +
                                        std::to_string(completed_iterations) + ".bin";
    write_weights(data_params, workspace, checkpoint_path);
    std::cout << "checkpoint: iteration=" << completed_iterations
              << " training_ms=" << cumulative_training_time_ms
              << " weights_path=" << checkpoint_path << '\n';
}

// Submit the cuBLAS and CUDA operations that apply one sampled s-step recurrence.
void launch_cublas_recurrence(const RunParams& run_params, Workspace* workspace) {
    const int batch_size = run_params.batch_size;
    const int sample_count = run_params.samples_per_iteration;
    constexpr float one{1.0f};

    // Convert every base-model margin in the s-step block into a logistic residual.
    cuda_sigmoid(workspace->compute_stream, workspace->device_block_margins_or_residuals,
                 workspace->device_base_residuals, run_params.samples_per_iteration);
    const float learning_rate_per_sample =
        run_params.learning_rate / static_cast<float>(batch_size);
    const float* gram =
        workspace->sampled_batch_slots[workspace->current_slot_index].device_approximate_gram;
    // Reference only Gram blocks G_ij with j < i; diagonal and upper blocks are ignored.
    for (int minibatch_index = 1; minibatch_index < run_params.s; ++minibatch_index) {
        const int preceding_sample_count = minibatch_index * batch_size;
        check_cublas(cublasSgemv(
            workspace->cublas_handle, CUBLAS_OP_N, batch_size, preceding_sample_count,
            &learning_rate_per_sample, gram + preceding_sample_count, sample_count,
            workspace->device_base_residuals, 1, &one,
            workspace->device_block_margins_or_residuals + preceding_sample_count, 1));
    }
    // Convert corrected margins to residuals and accumulate controller statistics.
    cuda_corrected_sigmoid_and_accumulate(
        workspace->compute_stream, workspace->device_block_margins_or_residuals,
        workspace->device_base_residuals, workspace->device_block_margins_or_residuals,
        run_params.samples_per_iteration, workspace->device_recurrence_stats);
}

// Find the captured recurrence executable for a double-buffer slot.
cudaGraphExec_t recurrence_graph_for(const Workspace* workspace, std::size_t slot_index) {
    const auto graph =
        std::find_if(workspace->recurrence_graphs.begin(), workspace->recurrence_graphs.end(),
                     [slot_index](const RecurrenceGraph& candidate) {
                         return candidate.slot_index == slot_index;
                     });
    if (graph == workspace->recurrence_graphs.end())
        throw std::runtime_error("missing recurrence CUDA graph");
    return graph->executable;
}

// Warm and capture one recurrence CUDA Graph for each double-buffer slot.
void prepare_recurrence_graphs(const RunParams& run_params, Workspace* workspace) {
    const std::size_t original_slot_index = workspace->current_slot_index;
    // Initialize cuBLAS's fixed-shape GEMV before stream capture.
    workspace->current_slot_index = 0;
    launch_cublas_recurrence(run_params, workspace);
    check_cuda(cudaStreamSynchronize(workspace->compute_stream));

    for (std::size_t slot_index = 0; slot_index < workspace->sampled_batch_slots.size();
         ++slot_index) {
        workspace->current_slot_index = slot_index;
        RecurrenceGraph recurrence_graph{slot_index};
        check_cuda(cudaStreamBeginCapture(workspace->compute_stream,
                                          cudaStreamCaptureModeThreadLocal));
        launch_cublas_recurrence(run_params, workspace);
        check_cuda(cudaStreamEndCapture(workspace->compute_stream, recurrence_graph.graph.put()));
        check_cuda(
            cudaGraphInstantiate(recurrence_graph.executable.put(), recurrence_graph.graph, 0));
        workspace->recurrence_graphs.push_back(std::move(recurrence_graph));
    }
    workspace->current_slot_index = original_slot_index;
}

// Compute base margins and replay the captured recurrence for the current block.
void enter_recurrence(const DataParams& data_params, const RunParams& run_params,
                      Workspace* workspace, float* device_signed_batch) {
    constexpr float alpha{1};
    constexpr float beta{0};

    {
        nvtx3::scoped_range base_margin_range{"base margins"};
        // Compute each sample's signed margin at the base model for this s-step block.
        check_cublas(cublasSgemv(workspace->cublas_handle, CUBLAS_OP_T, data_params.feature_count,
                                 run_params.samples_per_iteration, &alpha, device_signed_batch,
                                 data_params.feature_count, workspace->device_weights, 1, &beta,
                                 workspace->device_block_margins_or_residuals, 1));
    }

    {
        nvtx3::scoped_range recurrence_range{"recurrence graph"};
        // Replay the captured recurrence using the Gram stored in the current slot.
        check_cuda(cudaGraphLaunch(recurrence_graph_for(workspace, workspace->current_slot_index),
                                   workspace->compute_stream));
    }
}

// Run exact sequential minibatch SGD without leverage sampling or recurrence work.
void train_sequential(const DataParams& data_params, const RunParams& run_params,
                      Workspace* workspace, ProfileStats* profile_stats) {
    constexpr float one{1.0f};
    constexpr float zero{0.0f};
    const float learning_rate_per_sample =
        run_params.learning_rate / static_cast<float>(run_params.batch_size);
    auto timed_segment_start = std::chrono::steady_clock::now();
    float cumulative_training_time_ms = 0.0f;

    nvtx3::scoped_range training_loop_range{"sequential training loop"};
    for (int iteration_index = 0; iteration_index < run_params.iteration_count;
         ++iteration_index) {
        nvtx3::scoped_range iteration_range{"sequential SGD iteration"};
        const std::size_t batch_start_offset =
            (static_cast<std::size_t>(iteration_index) * run_params.batch_size) %
            data_params.padded_sample_count;
        float* device_signed_batch =
            workspace->device_signed_features + batch_start_offset * data_params.feature_count;

        {
            nvtx3::scoped_range margin_range{"sequential margins"};
            check_cublas(cublasSgemv(
                workspace->cublas_handle, CUBLAS_OP_T, data_params.feature_count,
                run_params.batch_size, &one, device_signed_batch, data_params.feature_count,
                workspace->device_weights, 1, &zero,
                workspace->device_block_margins_or_residuals, 1));
        }
        {
            nvtx3::scoped_range residual_range{"sequential residuals"};
            cuda_sigmoid(workspace->compute_stream,
                         workspace->device_block_margins_or_residuals,
                         workspace->device_block_margins_or_residuals,
                         run_params.batch_size);
        }
        {
            nvtx3::scoped_range update_range{"sequential model update"};
            check_cublas(cublasSgemv(
                workspace->cublas_handle, CUBLAS_OP_N, data_params.feature_count,
                run_params.batch_size, &learning_rate_per_sample, device_signed_batch,
                data_params.feature_count, workspace->device_block_margins_or_residuals, 1,
                &one, workspace->device_weights, 1));
        }

        const int completed_iterations = iteration_index + 1;
        if (run_params.checkpoint_interval != 0 &&
            completed_iterations % run_params.checkpoint_interval == 0) {
            check_cuda(cudaDeviceSynchronize());
            const auto checkpoint_time = std::chrono::steady_clock::now();
            cumulative_training_time_ms +=
                std::chrono::duration<float, std::milli>(checkpoint_time - timed_segment_start)
                    .count();
            save_training_checkpoint(data_params, run_params, workspace, completed_iterations,
                                     cumulative_training_time_ms);
            timed_segment_start = std::chrono::steady_clock::now();
        }
    }
    check_cuda(cudaDeviceSynchronize());
    cumulative_training_time_ms +=
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() -
                                                 timed_segment_start)
            .count();
    profile_stats->training_time_ms = cumulative_training_time_ms;

    std::cout << "execution_mode: sequential_full_features"
              << "\naverage_sampled_features: 0"
              << "\nfinal_smoothed_correction_ratio: 0"
              << "\ncontroller_updates: 0"
              << "\nskipped_controller_stats: 0"
              << "\nsampled_feature_histogram:\n";

    double objective = 0.0;
    double accuracy = 0.0;
    compute_metrics(data_params, workspace, objective, accuracy);
    const auto output_flags = std::cout.flags();
    const auto output_precision = std::cout.precision();
    std::cout << "Final Objective: " << std::fixed << std::setprecision(8) << objective
              << "\nFinal Training Accuracy: " << std::setprecision(6) << accuracy << "%\n";
    std::cout.flags(output_flags);
    std::cout.precision(output_precision);
}

// ---------------- Train Function ----------------
// Run pipelined s-step SGD while adapting sampled width from delayed GPU statistics.
void train(const DataParams& data_params, const RunParams& run_params, Workspace* workspace,
           ProfileStats* profile_stats) {
    double previous_objective{std::log(2.0)};
    double current_objective{0.0};
    double current_accuracy{0.0};

    // Form the signed design matrix in place because leverage preparation no longer needs A.
    {
        nvtx3::scoped_range signed_features_range{"form signed features"};
        // cuBLAS views row-major samples as columns, so right-side DGMM applies one label per sample.
        check_cublas(cublasSdgmm(
            workspace->cublas_handle, CUBLAS_SIDE_RIGHT,
            static_cast<int>(data_params.feature_count),
            static_cast<int>(data_params.padded_sample_count), workspace->device_signed_features,
            static_cast<int>(data_params.feature_count), workspace->device_labels, 1,
            workspace->device_signed_features, static_cast<int>(data_params.feature_count)));
    }
    // The compute and prefetch streams are nonblocking, so neither implicitly
    // waits for the other. Finish one-time label scaling before the sampled
    // feature prefetch reads device_signed_features.
    check_cuda(cudaStreamSynchronize(workspace->compute_stream));

    if (run_params.s == 1) {
        train_sequential(data_params, run_params, workspace, profile_stats);
        return;
    }

    // Capture one cuBLAS recurrence graph for each buffer slot and adaptive sampled width.
    {
        nvtx3::scoped_range graph_capture_range{"capture recurrence graphs"};
        prepare_recurrence_graphs(run_params, workspace);
    }

    // Gather the first block using its independently sampled leverage columns.
    if (run_params.iteration_count > 0)
        prefetch_leverage_features(data_params, run_params, workspace,
                                   workspace->device_signed_features, workspace->current_slot_index,
                                   workspace->leverage_sketch.size(0));

    auto timed_segment_start = std::chrono::steady_clock::now();
    float cumulative_training_time_ms = 0.0f;
    constexpr float increase_width_threshold{0.20f};
    constexpr float decrease_width_threshold{0.05f};
    constexpr std::size_t low_ratio_updates_before_decrease{3};
    float smoothed_correction_ratio = 0.0f;
    bool has_smoothed_correction_ratio = false;
    std::size_t controller_sampled_feature_count = workspace->leverage_sketch.size(0);
    const std::size_t maximum_controller_sampled_feature_count =
        std::min(workspace->leverage_sketch.size(), adaptive_sampled_feature_counts.back());
    std::size_t consecutive_low_ratio_updates = 0;
    std::vector<float> correction_ratio_history;
    if (run_params.trace_controller)
        correction_ratio_history.reserve(run_params.iteration_count);
    std::size_t oldest_stats_slot_index = 0;
    std::size_t next_stats_slot_index = 0;
    std::size_t pending_stats_count = 0;
    std::size_t controller_update_count = 0;
    std::size_t skipped_stats_copy_count = 0;
    std::size_t sampled_feature_count_sum = 0;
    std::map<std::size_t, std::size_t> sampled_feature_count_histogram;
    const bool adaptive_controller = run_params.fixed_sampled_feature_count == 0;

    nvtx3::scoped_range training_loop_range{"training loop"};
    for (int iteration_index{0}; iteration_index < run_params.iteration_count; ++iteration_index) {
        nvtx3::scoped_range iteration_range{"s-step iteration"};
        const std::size_t block_index = static_cast<std::size_t>(iteration_index);
        // determine current batch start offset (wrap around if we exceed total samples)
        const std::size_t current_batch_start_offset{
            ((block_index * run_params.samples_per_iteration) % data_params.padded_sample_count)};
        float* device_signed_batch{workspace->device_signed_features +
                                   current_batch_start_offset * data_params.feature_count};

        const auto iteration_start = std::chrono::steady_clock::now();

        // wait for current iter prefetch to finish
        auto& current_batch_slot = workspace->sampled_batch_slots[workspace->current_slot_index];
        check_cuda(
            cudaStreamWaitEvent(workspace->compute_stream, current_batch_slot.prefetch_done, 0));

        // Compute reads from the current slot while prefetch writes into the other slot.
        const std::size_t next_slot_index{(workspace->current_slot_index + 1) % 2};
        auto& next_batch_slot = workspace->sampled_batch_slots[next_slot_index];

        // Wait until compute finishes reading a reused buffer before prefetch overwrites it.
        check_cuda(
            cudaStreamWaitEvent(workspace->prefetch_stream, next_batch_slot.compute_done, 0));

        // Consume completed statistics, bounding the host to two blocks ahead of the controller.
        if (adaptive_controller) {
            nvtx3::scoped_range controller_range{"controller poll and update"};
            while (pending_stats_count > 0) {
                auto& stats_slot =
                    workspace->controller_stats_slots[oldest_stats_slot_index];
                const cudaError_t query_status = cudaEventQuery(stats_slot.copy_done);
                if (query_status == cudaErrorNotReady) {
                    if (pending_stats_count < workspace->controller_stats_slots.size())
                        break;
                    check_cuda(cudaEventSynchronize(stats_slot.copy_done));
                } else {
                    check_cuda(query_status);
                }

                const auto& recurrence_stats = *stats_slot.host_stats;
                const float correction_ratio =
                    std::sqrt(recurrence_stats.corrected_residual_difference_squared_sum /
                              std::max(recurrence_stats.base_residual_squared_sum, 1e-20f));
                if (stats_slot.sampled_feature_count == controller_sampled_feature_count) {
                    if (run_params.trace_controller)
                        correction_ratio_history.push_back(correction_ratio);
                    smoothed_correction_ratio =
                        has_smoothed_correction_ratio
                            ? 0.5f * smoothed_correction_ratio + 0.5f * correction_ratio
                            : correction_ratio;
                    has_smoothed_correction_ratio = true;
                    ++controller_update_count;

                    bool width_changed = false;
                    if (smoothed_correction_ratio > increase_width_threshold) {
                        const std::size_t increased_sampled_feature_count =
                            std::min(2 * controller_sampled_feature_count,
                                     maximum_controller_sampled_feature_count);
                        width_changed =
                            increased_sampled_feature_count != controller_sampled_feature_count;
                        controller_sampled_feature_count = increased_sampled_feature_count;
                        consecutive_low_ratio_updates = 0;
                    } else if (smoothed_correction_ratio < decrease_width_threshold) {
                        ++consecutive_low_ratio_updates;
                        if (consecutive_low_ratio_updates >= low_ratio_updates_before_decrease) {
                            const std::size_t decreased_sampled_feature_count =
                                std::max(controller_sampled_feature_count / 2,
                                         adaptive_sampled_feature_counts.front());
                            width_changed =
                                decreased_sampled_feature_count != controller_sampled_feature_count;
                            controller_sampled_feature_count = decreased_sampled_feature_count;
                            consecutive_low_ratio_updates = 0;
                        }
                    } else {
                        consecutive_low_ratio_updates = 0;
                    }

                    // Discard the old moving average until feedback from the new width arrives.
                    if (width_changed)
                        has_smoothed_correction_ratio = false;
                }
                oldest_stats_slot_index =
                    (oldest_stats_slot_index + 1) % workspace->controller_stats_slots.size();
                --pending_stats_count;
            }
        }

        // Apply the most recent completed controller state before gathering the next block.
        if (adaptive_controller && block_index + 1 < workspace->leverage_sketch.sketch_count)
            workspace->leverage_sketch.sampled_feature_counts[block_index + 1] =
                controller_sampled_feature_count;

        // next batch pointer for prefetch
        const std::size_t next_batch_start_offset{
            ((block_index + 1) * run_params.samples_per_iteration) %
            data_params.padded_sample_count};
        float* device_next_signed_batch =
            workspace->device_signed_features + next_batch_start_offset * data_params.feature_count;

        // Gather the next block using its own leverage sample while this block is computed.
        if (iteration_index + 1 < run_params.iteration_count) {
            nvtx3::scoped_range prefetch_range{"prefetch next block"};
            prefetch_leverage_features(data_params, run_params, workspace, device_next_signed_batch,
                                       next_slot_index,
                                       workspace->leverage_sketch.size(block_index + 1));
        }

        const std::size_t active_sampled_feature_count = workspace->leverage_sketch.size(block_index);
        sampled_feature_count_sum += active_sampled_feature_count;
        ++sampled_feature_count_histogram[active_sampled_feature_count];
        enter_recurrence(data_params, run_params, workspace, device_signed_batch);

        // Record when recurrence finishes reading this buffer so prefetch can safely reuse it.
        check_cuda(cudaEventRecord(current_batch_slot.compute_done, workspace->compute_stream));

        const float learning_rate_per_sample =
            run_params.learning_rate / static_cast<float>(run_params.batch_size);
        constexpr float one{1.0f};
        {
            nvtx3::scoped_range model_update_range{"model update"};
            // Apply all corrected block residuals as x += (learning_rate / batch_size) A^T r.
            check_cublas(cublasSgemv(
                workspace->cublas_handle, CUBLAS_OP_N, data_params.feature_count,
                run_params.samples_per_iteration, &learning_rate_per_sample, device_signed_batch,
                data_params.feature_count, workspace->device_block_margins_or_residuals, 1, &one,
                workspace->device_weights, 1));
        }

        // Queue this block's statistics when a ring slot is available; otherwise keep computing.
        if (adaptive_controller && block_index + 3 < workspace->leverage_sketch.sketch_count) {
            if (pending_stats_count < workspace->controller_stats_slots.size()) {
                nvtx3::scoped_range statistics_copy_range{"controller statistics copy"};
                auto& stats_slot = workspace->controller_stats_slots[next_stats_slot_index];
                stats_slot.sampled_feature_count = active_sampled_feature_count;
                check_cuda(cudaMemcpyAsync(stats_slot.host_stats, workspace->device_recurrence_stats,
                                           sizeof(RecurrenceStats), cudaMemcpyDeviceToHost,
                                           workspace->compute_stream));
                check_cuda(cudaEventRecord(stats_slot.copy_done, workspace->compute_stream));
                next_stats_slot_index =
                    (next_stats_slot_index + 1) % workspace->controller_stats_slots.size();
                ++pending_stats_count;
            } else {
                ++skipped_stats_copy_count;
            }
        }

        // alternate buffers for next iteration
        workspace->current_slot_index = next_slot_index;

        if (run_params.print_interval != 0 && iteration_index != 0 &&
            iteration_index % run_params.print_interval == 0) {
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

        const int completed_iterations = iteration_index + 1;
        if (run_params.checkpoint_interval != 0 &&
            completed_iterations % run_params.checkpoint_interval == 0) {
            check_cuda(cudaDeviceSynchronize());
            const auto checkpoint_time = std::chrono::steady_clock::now();
            cumulative_training_time_ms +=
                std::chrono::duration<float, std::milli>(checkpoint_time - timed_segment_start)
                    .count();
            save_training_checkpoint(data_params, run_params, workspace, completed_iterations,
                                     cumulative_training_time_ms);
            timed_segment_start = std::chrono::steady_clock::now();
        }
    }
    check_cuda(cudaDeviceSynchronize());
    cumulative_training_time_ms +=
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() -
                                                 timed_segment_start)
            .count();
    profile_stats->training_time_ms = cumulative_training_time_ms;

    const double average_sampled_feature_count =
        static_cast<double>(sampled_feature_count_sum) /
        static_cast<double>(std::max(1, run_params.iteration_count));
    std::cout << "average_sampled_features: " << average_sampled_feature_count
              << "\nfinal_smoothed_correction_ratio: " << smoothed_correction_ratio
              << "\ncontroller_updates: " << controller_update_count
              << "\nskipped_controller_stats: " << skipped_stats_copy_count
              << "\nsampled_feature_histogram:";
    for (const auto& [sampled_feature_count, block_count] : sampled_feature_count_histogram)
        std::cout << ' ' << sampled_feature_count << ':' << block_count;
    if (run_params.trace_controller) {
        std::cout << "\nsampled_feature_schedule:";
        for (int iteration_index = 0; iteration_index < run_params.iteration_count;
             ++iteration_index)
            std::cout << ' ' << workspace->leverage_sketch.size(iteration_index);
        std::cout << "\ncorrection_ratio_history:";
        for (const float correction_ratio : correction_ratio_history)
            std::cout << ' ' << correction_ratio;
    }
    std::cout << '\n';

    compute_metrics(data_params, workspace, current_objective, current_accuracy);
    const auto output_flags = std::cout.flags();
    const auto output_precision = std::cout.precision();
    std::cout << "Final Objective: " << std::fixed << std::setprecision(8) << current_objective
              << "\nFinal Training Accuracy: " << std::setprecision(6) << current_accuracy << "%\n";
    std::cout.flags(output_flags);
    std::cout.precision(output_precision);
}

// Print the command-line interface and legacy positional syntax.
void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "  --batch-size N       samples per minibatch (default 16)\n"
              << "  --s N                minibatches per block iteration (default 4)\n"
              << "  --n-iters N          block iterations (default 100)\n"
              << "  --dataset FILE       LIBSVM or shared-binary dataset (default w1a.txt)\n"
              << "  --sampled-features N maximum leverage samples (default 512)\n"
              << "  --fixed-l N          hold leverage sample width constant (power of two, 2-512)\n"
              << "  --leverage-rank N    randomized leverage basis rank (default 32)\n"
              << "  --leverage-rows N    dataset rows used to prepare leverage scores (default 512)\n"
              << "  --eta VALUE          learning rate (default 0.5)\n"
              << "  --sketch-seed N      feature-sampling RNG seed (default 0)\n"
              << "  --trace-controller   print per-block widths and correction ratios\n"
              << "  --checkpoint-every N save weights every N block iterations\n"
              << "  --checkpoint-prefix FILE prefix for iteration checkpoint files\n"
              << "  --weights-out FILE   write the final float32 weights\n\n"
              << "Legacy positional arguments remain supported:\n"
              << "  batch_size s n_iters dataset leverage sampled_features eta "
                 "[weights_out]\n";
}

// Advance to and return the value following a named command-line option.
const char* consume_option_value(int argc, char** argv, int& index, const std::string& option) {
    if (++index >= argc) {
        throw std::invalid_argument("Missing value for " + option);
    }
    return argv[index];
}

// ---------------- Main ----------------
// Parse configuration, prepare data and GPU resources, train, and optionally save weights.
int run_main(int argc, char** argv) {
    RunParams run_params{};
    DataParams data_params{};
    std::string weights_output_path{};

    for (int argument_index = 1; argument_index < argc; ++argument_index) {
        const std::string option{argv[argument_index]};
        if (option == "-h" || option == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (option == "--batch-size") {
            run_params.batch_size =
                std::max(1, std::stoi(consume_option_value(argc, argv, argument_index, option)));
        } else if (option == "--s") {
            run_params.s =
                std::max(1, std::stoi(consume_option_value(argc, argv, argument_index, option)));
        } else if (option == "--n-iters") {
            run_params.iteration_count =
                std::max(1, std::stoi(consume_option_value(argc, argv, argument_index, option)));
        } else if (option == "--dataset") {
            data_params.dataset_path = consume_option_value(argc, argv, argument_index, option);
        } else if (option == "--sampled-features") {
            run_params.requested_sampled_feature_count =
                std::stoull(consume_option_value(argc, argv, argument_index, option));
        } else if (option == "--fixed-l") {
            run_params.fixed_sampled_feature_count =
                std::stoull(consume_option_value(argc, argv, argument_index, option));
        } else if (option == "--leverage-rank") {
            run_params.leverage_rank =
                std::stoull(consume_option_value(argc, argv, argument_index, option));
        } else if (option == "--leverage-rows") {
            run_params.leverage_row_count =
                std::stoull(consume_option_value(argc, argv, argument_index, option));
        } else if (option == "--eta") {
            run_params.learning_rate =
                std::stof(consume_option_value(argc, argv, argument_index, option));
        } else if (option == "--sketch-seed") {
            run_params.sketch_seed = static_cast<unsigned int>(
                std::stoul(consume_option_value(argc, argv, argument_index, option)));
        } else if (option == "--trace-controller") {
            run_params.trace_controller = true;
        } else if (option == "--checkpoint-every") {
            run_params.checkpoint_interval =
                std::max(1, std::stoi(consume_option_value(argc, argv, argument_index, option)));
        } else if (option == "--checkpoint-prefix") {
            run_params.checkpoint_prefix =
                consume_option_value(argc, argv, argument_index, option);
        } else if (option == "--weights-out") {
            weights_output_path = consume_option_value(argc, argv, argument_index, option);
        } else {
            throw std::invalid_argument("Unknown option: " + option);
        }
    }

    const std::size_t maximum_l = run_params.requested_sampled_feature_count;
    if (maximum_l < 2 || maximum_l > 512 || (maximum_l & (maximum_l - 1)) != 0)
        throw std::invalid_argument(
            "--sampled-features must be a power of two from 2 through 512");
    const std::size_t fixed_l = run_params.fixed_sampled_feature_count;
    if (fixed_l != 0 &&
        (fixed_l < 2 || fixed_l > 512 || (fixed_l & (fixed_l - 1)) != 0))
        throw std::invalid_argument("--fixed-l must be a power of two from 2 through 512");
    if (fixed_l > run_params.requested_sampled_feature_count)
        throw std::invalid_argument("--fixed-l cannot exceed --sampled-features");
    if (run_params.leverage_rank == 0 || run_params.leverage_rank > 512)
        throw std::invalid_argument("--leverage-rank must be between 1 and 512");
    if (run_params.leverage_row_count < run_params.leverage_rank)
        throw std::invalid_argument("--leverage-rows must be at least --leverage-rank");
    if ((run_params.checkpoint_interval == 0) != run_params.checkpoint_prefix.empty())
        throw std::invalid_argument(
            "--checkpoint-every and --checkpoint-prefix must be provided together");

    std::cout << "batch_size: " << run_params.batch_size << ", s: " << run_params.s
              << ", iteration_count: " << run_params.iteration_count
              << ", training_set: " << data_params.dataset_path << ", gram_mode: leverage"
              << ", requested_sampled_feature_count: " << run_params.requested_sampled_feature_count
              << ", fixed_l: " << run_params.fixed_sampled_feature_count
              << ", leverage_rank: " << run_params.leverage_rank
              << ", leverage_rows: " << run_params.leverage_row_count
              << ", learning_rate: " << run_params.learning_rate
              << ", sketch_seed: " << run_params.sketch_seed
              << ", checkpoint_interval: " << run_params.checkpoint_interval << '\n';
    // load raw data into host
    std::vector<float> host_features{};
    std::vector<float> host_labels{};
    load_dataset(data_params, host_features, host_labels);
    data_params.sample_count = host_labels.size();
    std::cout << "dataset_samples: " << data_params.sample_count
              << ", dataset_features: " << data_params.feature_count << '\n';

    // Pad data to be multiple of s * batch_size
    run_params.samples_per_iteration = run_params.s * run_params.batch_size;

    // Reserve padding only on the GPU so a host resize cannot duplicate the full dataset.
    const std::size_t samples_per_iteration =
        static_cast<std::size_t>(run_params.samples_per_iteration);
    const std::size_t remainder = data_params.sample_count % samples_per_iteration;
    const std::size_t extra_samples = (samples_per_iteration - remainder) % samples_per_iteration;
    data_params.padded_sample_count = data_params.sample_count + extra_samples;

    if (static_cast<std::size_t>(run_params.samples_per_iteration) > data_params.padded_sample_count) {
        std::cerr << "Error: batch_size * s must be <= padded_sample_count\n";
        return 1;
    }

    std::cout << "block_iterations: " << run_params.iteration_count << '\n';
    run_params.print_interval = 0;

    // Upload the dataset once so leverage preparation and training share the same allocation.
    DeviceBuffer<float> device_features(data_params.padded_sample_count * data_params.feature_count);
    check_cuda(cudaMemcpy(device_features, host_features.data(),
                          data_params.sample_count * data_params.feature_count * sizeof(float),
                          cudaMemcpyHostToDevice));
    if (extra_samples != 0) {
        check_cuda(cudaMemset(
            device_features + data_params.sample_count * data_params.feature_count, 0,
            extra_samples * data_params.feature_count * sizeof(float)));
        check_cuda(cudaDeviceSynchronize());
    }

    LeverageSketch leverage_sketch{};
    double leverage_preparation_ms = 0.0;
    if (run_params.s > 1) {
        const auto leverage_preparation_start = std::chrono::steady_clock::now();
        leverage_sketch = prepare_leverage_sketch(data_params, run_params, device_features);
        leverage_preparation_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                      leverage_preparation_start)
                .count();
    }
    std::cout << "Leverage Preparation Time: " << leverage_preparation_ms << " ms\n";
    std::cout << "effective_sampled_features: "
              << (run_params.s == 1 ? 0 : leverage_sketch.size()) << '\n';
    std::cout << "initial_sampled_features: "
              << (run_params.s == 1 ? 0 : leverage_sketch.size(0)) << '\n';
    std::cout << "sampled_feature_controller: ";
    if (run_params.s == 1)
        std::cout << "disabled_sequential_bypass\n";
    else
        std::cout << (run_params.fixed_sampled_feature_count == 0
                          ? "deterministic_residual_correction"
                          : "fixed")
                  << '\n';
    std::cout << "feature_sketch_count: " << leverage_sketch.sketch_count << '\n';
    Workspace workspace{data_params, run_params, std::move(leverage_sketch),
                        std::move(device_features)};
    ProfileStats profile_stats{};

    // Copy labels and zero the labels belonging to the GPU-only feature padding.
    check_cuda(cudaMemcpy(workspace.device_labels, host_labels.data(),
                          data_params.sample_count * sizeof(float), cudaMemcpyHostToDevice));
    if (extra_samples != 0)
        check_cuda(cudaMemset(workspace.device_labels + data_params.sample_count, 0,
                              extra_samples * sizeof(float)));
    check_cuda(cudaMemset(workspace.device_weights, 0, data_params.feature_count * sizeof(float)));
    // Workspace streams use cudaStreamNonBlocking and therefore do not inherit
    // ordering from these default-stream initialization operations.
    check_cuda(cudaDeviceSynchronize());

    train(data_params, run_params, &workspace, &profile_stats);

    if (!weights_output_path.empty())
        write_weights(data_params, &workspace, weights_output_path);

    std::cout << "\n=== Timing ===\n";
    std::cout << "Training Time: " << profile_stats.training_time_ms << " ms\n\n";

    return 0;
}

// Translate uncaught application exceptions into a nonzero process exit status.
int main(int argc, char** argv) {
    try {
        return run_main(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
