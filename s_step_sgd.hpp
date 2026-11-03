#pragma once

#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Format a source location for CUDA and cuBLAS error messages.
inline std::string call_site(const std::source_location& location) {
    return std::string(location.file_name()) + ":" + std::to_string(location.line()) + " in " +
           location.function_name();
}

// Throw a source-aware exception when a CUDA Runtime call fails.
inline void check_cuda(cudaError_t status,
                       const std::source_location& location = std::source_location::current()) {
    if (status != cudaSuccess) {
        throw std::runtime_error("CUDA error at " + call_site(location) + ": " +
                                 cudaGetErrorString(status));
    }
}

// Throw a source-aware exception when a cuBLAS call fails.
inline void check_cublas(cublasStatus_t status,
                         const std::source_location& location = std::source_location::current()) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS error at " + call_site(location) + ": status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

// Throw a source-aware exception when a cuRAND call fails.
inline void check_curand(curandStatus_t status,
                         const std::source_location& location = std::source_location::current()) {
    if (status != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error("cuRAND error at " + call_site(location) + ": status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

template <typename T> class DeviceBuffer {
  public:
    // Create an empty device allocation owner.
    DeviceBuffer() = default;
    // Allocate storage for the requested number of device elements.
    explicit DeviceBuffer(std::size_t count) {
        allocate(count);
    }
    // Release the owned device allocation.
    ~DeviceBuffer() {
        cudaFree(pointer_);
    }

    // Prevent two objects from owning the same device allocation.
    DeviceBuffer(const DeviceBuffer&) = delete;
    // Prevent copy assignment of device allocation ownership.
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Transfer device allocation ownership from another buffer.
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : pointer_(std::exchange(other.pointer_, nullptr)) {}
    // Release the current allocation and take ownership from another buffer.
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            cudaFree(pointer_);
            pointer_ = std::exchange(other.pointer_, nullptr);
        }
        return *this;
    }

    // Replace the current allocation with storage for count device elements.
    void allocate(std::size_t count) {
        T* allocation = nullptr;
        check_cuda(cudaMalloc(&allocation, count * sizeof(T)));
        cudaFree(pointer_);
        pointer_ = allocation;
    }

    // Expose the owned allocation to CUDA APIs expecting a raw pointer.
    operator T*() const {
        return pointer_;
    }

  private:
    T* pointer_{nullptr};
};

template <typename T> class PinnedBuffer {
  public:
    // Create an empty pinned-host allocation owner.
    PinnedBuffer() = default;
    // Allocate pinned host storage for the requested number of elements.
    explicit PinnedBuffer(std::size_t count) {
        allocate(count);
    }
    // Release the owned pinned-host allocation.
    ~PinnedBuffer() {
        cudaFreeHost(pointer_);
    }

    // Prevent two objects from owning the same pinned-host allocation.
    PinnedBuffer(const PinnedBuffer&) = delete;
    // Prevent copy assignment of pinned-host allocation ownership.
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    // Transfer pinned-host allocation ownership from another buffer.
    PinnedBuffer(PinnedBuffer&& other) noexcept
        : pointer_(std::exchange(other.pointer_, nullptr)) {}
    // Release the current allocation and take ownership from another buffer.
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            cudaFreeHost(pointer_);
            pointer_ = std::exchange(other.pointer_, nullptr);
        }
        return *this;
    }

    // Replace the current allocation with pinned storage for count elements.
    void allocate(std::size_t count) {
        T* allocation = nullptr;
        check_cuda(cudaMallocHost(&allocation, count * sizeof(T)));
        cudaFreeHost(pointer_);
        pointer_ = allocation;
    }

    // Expose the owned allocation to CUDA APIs expecting a raw pointer.
    operator T*() const {
        return pointer_;
    }
    // Access the first object stored in the pinned allocation.
    T& operator*() const {
        return *pointer_;
    }

  private:
    T* pointer_{nullptr};
};

class CudaStream {
  public:
    // Create a CUDA stream with the requested behavior flags.
    explicit CudaStream(unsigned int flags = cudaStreamNonBlocking) {
        check_cuda(cudaStreamCreateWithFlags(&stream_, flags));
    }
    // Destroy the owned CUDA stream.
    ~CudaStream() {
        if (stream_)
            cudaStreamDestroy(stream_);
    }

    // Prevent two objects from owning the same CUDA stream.
    CudaStream(const CudaStream&) = delete;
    // Prevent copy assignment of CUDA stream ownership.
    CudaStream& operator=(const CudaStream&) = delete;
    // Transfer CUDA stream ownership from another object.
    CudaStream(CudaStream&& other) noexcept : stream_(std::exchange(other.stream_, nullptr)) {}
    // Destroy the current stream and take ownership of another stream.
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_)
                cudaStreamDestroy(stream_);
            stream_ = std::exchange(other.stream_, nullptr);
        }
        return *this;
    }

    // Expose the owned stream handle to CUDA APIs.
    operator cudaStream_t() const {
        return stream_;
    }

  private:
    cudaStream_t stream_{nullptr};
};

class CudaEvent {
  public:
    // Create a CUDA event with the requested behavior flags.
    explicit CudaEvent(unsigned int flags = cudaEventDisableTiming) {
        check_cuda(cudaEventCreateWithFlags(&event_, flags));
    }
    // Destroy the owned CUDA event.
    ~CudaEvent() {
        if (event_)
            cudaEventDestroy(event_);
    }

    // Prevent two objects from owning the same CUDA event.
    CudaEvent(const CudaEvent&) = delete;
    // Prevent copy assignment of CUDA event ownership.
    CudaEvent& operator=(const CudaEvent&) = delete;
    // Transfer CUDA event ownership from another object.
    CudaEvent(CudaEvent&& other) noexcept : event_(std::exchange(other.event_, nullptr)) {}
    // Destroy the current event and take ownership of another event.
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_)
                cudaEventDestroy(event_);
            event_ = std::exchange(other.event_, nullptr);
        }
        return *this;
    }

    // Expose the owned event handle to CUDA APIs.
    operator cudaEvent_t() const {
        return event_;
    }

  private:
    cudaEvent_t event_{nullptr};
};

class CublasHandle {
  public:
    // Create a cuBLAS library handle.
    CublasHandle() {
        check_cublas(cublasCreate(&handle_));
    }
    // Destroy the owned cuBLAS handle.
    ~CublasHandle() {
        if (handle_)
            cublasDestroy(handle_);
    }

    // Prevent two objects from owning the same cuBLAS handle.
    CublasHandle(const CublasHandle&) = delete;
    // Prevent copy assignment of cuBLAS handle ownership.
    CublasHandle& operator=(const CublasHandle&) = delete;
    // Transfer cuBLAS handle ownership from another object.
    CublasHandle(CublasHandle&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    // Destroy the current handle and take ownership of another handle.
    CublasHandle& operator=(CublasHandle&& other) noexcept {
        if (this != &other) {
            if (handle_)
                cublasDestroy(handle_);
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    // Expose the owned handle to cuBLAS APIs.
    operator cublasHandle_t() const {
        return handle_;
    }

  private:
    cublasHandle_t handle_{nullptr};
};

class CurandGenerator {
  public:
    // Create a seeded Philox generator for asynchronous device sampling.
    explicit CurandGenerator(unsigned long long seed) {
        check_curand(curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_PHILOX4_32_10));
        try {
            check_curand(curandSetPseudoRandomGeneratorSeed(generator_, seed));
        } catch (...) {
            curandDestroyGenerator(generator_);
            generator_ = nullptr;
            throw;
        }
    }
    // Destroy the owned cuRAND generator.
    ~CurandGenerator() {
        if (generator_)
            curandDestroyGenerator(generator_);
    }

    // Prevent two objects from owning the same cuRAND generator.
    CurandGenerator(const CurandGenerator&) = delete;
    // Prevent copy assignment of cuRAND generator ownership.
    CurandGenerator& operator=(const CurandGenerator&) = delete;
    // Transfer cuRAND generator ownership from another object.
    CurandGenerator(CurandGenerator&& other) noexcept
        : generator_(std::exchange(other.generator_, nullptr)) {}
    // Destroy the current generator and take ownership of another generator.
    CurandGenerator& operator=(CurandGenerator&& other) noexcept {
        if (this != &other) {
            if (generator_)
                curandDestroyGenerator(generator_);
            generator_ = std::exchange(other.generator_, nullptr);
        }
        return *this;
    }

    // Queue future random-number generation on the selected CUDA stream.
    void set_stream(cudaStream_t stream) {
        check_curand(curandSetStream(generator_, stream));
    }
    // Fill a device buffer with uniform random values in the interval (0, 1].
    void generate_uniform(float* values, std::size_t count) {
        check_curand(curandGenerateUniform(generator_, values, count));
    }
    // Fill a device buffer with double-precision uniform random values in (0, 1].
    void generate_uniform(double* values, std::size_t count) {
        check_curand(curandGenerateUniformDouble(generator_, values, count));
    }
    // Fill a device buffer with normally distributed random values.
    void generate_normal(float* values, std::size_t count, float mean = 0.0f,
                         float standard_deviation = 1.0f) {
        check_curand(curandGenerateNormal(generator_, values, count, mean, standard_deviation));
    }

  private:
    curandGenerator_t generator_{nullptr};
};

class CudaGraph {
  public:
    // Create an empty CUDA graph owner for later capture.
    CudaGraph() = default;
    // Destroy the owned CUDA graph.
    ~CudaGraph() {
        if (graph_)
            cudaGraphDestroy(graph_);
    }

    // Prevent two objects from owning the same CUDA graph.
    CudaGraph(const CudaGraph&) = delete;
    // Prevent copy assignment of CUDA graph ownership.
    CudaGraph& operator=(const CudaGraph&) = delete;
    // Transfer CUDA graph ownership from another object.
    CudaGraph(CudaGraph&& other) noexcept : graph_(std::exchange(other.graph_, nullptr)) {}
    // Destroy the current graph and take ownership of another graph.
    CudaGraph& operator=(CudaGraph&& other) noexcept {
        if (this != &other) {
            if (graph_)
                cudaGraphDestroy(graph_);
            graph_ = std::exchange(other.graph_, nullptr);
        }
        return *this;
    }

    // Return the graph-handle address for CUDA output parameters.
    cudaGraph_t* put() {
        return &graph_;
    }
    // Expose the owned graph handle to CUDA APIs.
    operator cudaGraph_t() const {
        return graph_;
    }

  private:
    cudaGraph_t graph_{nullptr};
};

class CudaGraphExec {
  public:
    // Create an empty executable-graph owner for later instantiation.
    CudaGraphExec() = default;
    // Destroy the owned executable CUDA graph.
    ~CudaGraphExec() {
        if (executable_)
            cudaGraphExecDestroy(executable_);
    }

    // Prevent two objects from owning the same executable graph.
    CudaGraphExec(const CudaGraphExec&) = delete;
    // Prevent copy assignment of executable-graph ownership.
    CudaGraphExec& operator=(const CudaGraphExec&) = delete;
    // Transfer executable-graph ownership from another object.
    CudaGraphExec(CudaGraphExec&& other) noexcept
        : executable_(std::exchange(other.executable_, nullptr)) {}
    // Destroy the current executable and take ownership of another one.
    CudaGraphExec& operator=(CudaGraphExec&& other) noexcept {
        if (this != &other) {
            if (executable_)
                cudaGraphExecDestroy(executable_);
            executable_ = std::exchange(other.executable_, nullptr);
        }
        return *this;
    }

    // Return the executable-handle address for CUDA output parameters.
    cudaGraphExec_t* put() {
        return &executable_;
    }
    // Expose the owned executable graph handle to CUDA APIs.
    operator cudaGraphExec_t() const {
        return executable_;
    }

  private:
    cudaGraphExec_t executable_{nullptr};
};

struct ProfileStats {
    float training_time_ms{0.0f};
};

struct RunParams {
    int batch_size{16};
    int s{4};
    int samples_per_iteration{batch_size * s};
    int iteration_count{100};
    int print_interval{0};
    float learning_rate{0.5f};
    std::size_t requested_sampled_feature_count{512};
    std::size_t fixed_sampled_feature_count{0};
    std::size_t leverage_rank{32};
    std::size_t leverage_row_count{512};
    float uniform_probability_mix{0.1f};
    unsigned int sketch_seed{0};
    bool trace_controller{false};
    int checkpoint_interval{0};
    std::string checkpoint_prefix{};
};

inline constexpr std::array<std::size_t, 8> adaptive_sampled_feature_counts{
    2, 4, 8, 16, 32, 64, 128, 256};

struct DataParams {
    std::string dataset_path{};
    std::size_t feature_count{300};
    std::size_t sample_count{0};
    std::size_t padded_sample_count{0};
    std::uint64_t dataset_seed{0};
};

struct MetricSums {
    double objective_sum{0.0};
    unsigned long long correct_count{0};
};

// Return the temporary storage required by the device-wide metric reduction.
std::size_t cuda_metric_reduction_workspace_size(std::size_t sample_count);

struct RecurrenceStats {
    float corrected_residual_difference_squared_sum{0.0f};
    float base_residual_squared_sum{0.0f};
};

struct LeverageSketch {
    std::vector<std::size_t> sampled_feature_counts{};
    DeviceBuffer<float> device_probabilities{};
    DeviceBuffer<float> device_alias_probabilities{};
    DeviceBuffer<int> device_alias_indices{};
    std::size_t sampled_feature_count{0};
    std::size_t sketch_count{0};

    // Create an empty leverage-sampling plan.
    LeverageSketch() = default;
    // Prevent copies of the sketch's device allocations.
    LeverageSketch(const LeverageSketch&) = delete;
    // Prevent copy assignment of the sketch's device allocations.
    LeverageSketch& operator=(const LeverageSketch&) = delete;
    // Transfer the leverage-sampling plan and its device allocations.
    LeverageSketch(LeverageSketch&&) noexcept = default;
    // Replace this plan by moving another leverage-sampling plan into it.
    LeverageSketch& operator=(LeverageSketch&&) noexcept = default;

    // Return the maximum number of feature samples generated for each block.
    std::size_t size() const {
        return sampled_feature_count;
    }

    // Return the active sampled width selected for one block.
    std::size_t size(std::size_t sketch_index) const {
        return sampled_feature_counts.at(sketch_index);
    }
};

struct SampledBatchSlot {
    DeviceBuffer<float> device_sampled_features{};
    DeviceBuffer<float> device_approximate_gram{};
    DeviceBuffer<float> device_random_values{};
    DeviceBuffer<std::size_t> device_sampled_column_indices{};
    DeviceBuffer<float> device_sampled_column_scales{};
    CudaEvent prefetch_done{};
    CudaEvent compute_done{};
};

struct ControllerStatsSlot {
    PinnedBuffer<RecurrenceStats> host_stats{};
    CudaEvent copy_done{};
    std::size_t sampled_feature_count{0};
};

struct RecurrenceGraph {
    std::size_t slot_index{0};
    CudaGraph graph{};
    CudaGraphExec executable{};
};

struct Workspace {
    DeviceBuffer<float> device_labels{};
    DeviceBuffer<float> device_weights{};
    DeviceBuffer<float> device_signed_features{};
    DeviceBuffer<float> device_block_margins_or_residuals{};
    DeviceBuffer<float> device_base_residuals{};
    DeviceBuffer<float> device_metric_margins{};
    DeviceBuffer<MetricSums> device_metric_sums{};
    DeviceBuffer<std::byte> device_metric_reduction_workspace{};
    std::size_t metric_reduction_workspace_size{0};
    DeviceBuffer<RecurrenceStats> device_recurrence_stats{};
    LeverageSketch leverage_sketch{};

    CudaStream prefetch_stream{};
    CudaStream compute_stream{};
    CublasHandle cublas_handle{};
    CublasHandle prefetch_cublas_handle{};
    CurandGenerator random_generator;
    std::array<SampledBatchSlot, 2> sampled_batch_slots{};
    // Limit host submission to two blocks ahead so adaptive decisions remain timely.
    std::array<ControllerStatsSlot, 2> controller_stats_slots{};
    std::vector<RecurrenceGraph> recurrence_graphs{};
    std::size_t current_slot_index{0};

    // Allocate persistent GPU resources and configure cuBLAS for training.
    Workspace(const DataParams& data_params, const RunParams& run_params,
              LeverageSketch initial_leverage_sketch,
              DeviceBuffer<float> initial_device_features)
        : device_signed_features(std::move(initial_device_features)),
          leverage_sketch(std::move(initial_leverage_sketch)),
          random_generator(static_cast<unsigned long long>(run_params.sketch_seed) + 2ULL) {
        const std::size_t sampled_element_count =
            run_params.samples_per_iteration * leverage_sketch.size();
        device_labels.allocate(data_params.padded_sample_count);
        device_weights.allocate(data_params.feature_count);
        device_block_margins_or_residuals.allocate(run_params.samples_per_iteration);
        device_metric_margins.allocate(data_params.sample_count);
        device_metric_sums.allocate(1);
        metric_reduction_workspace_size =
            cuda_metric_reduction_workspace_size(data_params.sample_count);
        device_metric_reduction_workspace.allocate(metric_reduction_workspace_size);
        if (run_params.s > 1) {
            const std::size_t sample_count =
                static_cast<std::size_t>(run_params.samples_per_iteration);
            device_base_residuals.allocate(run_params.samples_per_iteration);
            device_recurrence_stats.allocate(1);
            for (auto& batch_slot : sampled_batch_slots) {
                batch_slot.device_sampled_features.allocate(sampled_element_count);
                batch_slot.device_approximate_gram.allocate(sample_count * sample_count);
                batch_slot.device_random_values.allocate(2 * leverage_sketch.size());
                batch_slot.device_sampled_column_indices.allocate(leverage_sketch.size());
                batch_slot.device_sampled_column_scales.allocate(leverage_sketch.size());
            }
            for (auto& stats_slot : controller_stats_slots)
                stats_slot.host_stats.allocate(1);
        }

        check_cublas(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
        check_cublas(cublasSetStream(cublas_handle, compute_stream));
        check_cublas(cublasSetMathMode(prefetch_cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
        check_cublas(cublasSetStream(prefetch_cublas_handle, prefetch_stream));
        random_generator.set_stream(prefetch_stream);
    }
    // Prevent copies of the workspace's GPU resources.
    Workspace(const Workspace&) = delete;
    // Prevent moving a workspace whose internal operations reference its members.
    Workspace(Workspace&&) = delete;
    // Prevent copy assignment of the workspace's GPU resources.
    Workspace& operator=(const Workspace&) = delete;
    // Prevent move assignment of the workspace's GPU resources.
    Workspace& operator=(Workspace&&) = delete;
};

// Compute one block's base margins and launch its sampled recurrence graph.
void enter_recurrence(const DataParams& data_params, const RunParams& run_params,
                      Workspace* workspace, float* device_signed_batch);

// Convert signed margins into logistic residuals on a CUDA stream.
void cuda_sigmoid(cudaStream_t stream, const float* signed_margins, float* residuals,
                  std::size_t sample_count);
// Form corrected residuals and reduce the statistics used by the width controller.
void cuda_corrected_sigmoid_and_accumulate(cudaStream_t stream, const float* corrected_margins,
                                           const float* base_residuals, float* corrected_residuals,
                                           std::size_t sample_count, RecurrenceStats* recurrence_stats);
// Gather and probability-scale sampled feature columns into a dense GPU buffer.
void cuda_gather_weighted_columns(cudaStream_t stream, const float* source_features,
                                  float* sampled_features, std::size_t row_count,
                                  std::size_t source_feature_count, const std::size_t* sampled_column_indices,
                                  const float* sampled_column_scales, std::size_t sampled_feature_count,
                                  float scale_multiplier);
// Reduce logistic objective and correct-prediction counts across all samples.
void cuda_accumulate_metrics(cudaStream_t stream, const float* signed_margins, const float* labels,
                             std::size_t sample_count, void* reduction_workspace,
                             std::size_t reduction_workspace_size, MetricSums* metric_sums);
// Train the model with the pipelined adaptive s-step algorithm.
void train(const DataParams& data_params, const RunParams& run_params, Workspace* workspace,
           ProfileStats* profile_stats);
