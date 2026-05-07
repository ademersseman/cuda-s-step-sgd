#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>

struct ProfileStats {
    float init_corr_time = 0.0f;
    float gram_overhead_time = 0.0f;
    float gram_compute_time = 0.0f;
    float recurrence_time = 0.0f;
    float grad_proj_time = 0.0f;
    float weight_update_time = 0.0f;
    float scaling_time = 0.0f;
    float training_time = 0.0f;
};

struct RunParams {
    int batch_size = 16;
    int s = 4;
    int samples_per_iter = batch_size * s;
    int epochs = 10;
    int maxiters = 65536;
    float eta = 0.5f;
    int printerval = 512;
    bool approx_gram = false;
    int approx_gram_l = 64; // number of columns to sample for approximate Gram matrix (if enabled)
    std::string approx_gram_type = "uniform"; // method for sampling features for approximate Gram matrix ("uniform" or "leverage")
};

struct DataParams {
    std::string file_name = "synthetic_data_e8.txt";
    int n_features = 300;
    int total_samples_unpadded;
    int total_samples;
};

struct CudaRegionTimer {
    cudaEvent_t start, stop;

    CudaRegionTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    void begin(cudaStream_t stream = 0) {
        cudaEventRecord(start, stream);
    }

    float end(cudaStream_t stream = 0) {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }

    ~CudaRegionTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

struct Workspace {
    float *d_A;
    float *d_y;
    float *d_x;

    float *d_A_scaled;
    float *d_batch_A_approx;
    float *d_correction;
    float *d_G;
    float *d_grad;
    cublasHandle_t handle;

    Workspace(const DataParams* data_params, const RunParams* s_step_params)  {
        cudaMalloc(&d_A, data_params->total_samples * data_params->n_features * sizeof(float));
        cudaMalloc(&d_y, data_params->total_samples * sizeof(float));
        cudaMalloc(&d_x, data_params->n_features * sizeof(float));

        cudaMalloc(&d_A_scaled, data_params->total_samples * data_params->n_features * sizeof(float));
        cudaMalloc(&d_batch_A_approx, s_step_params->samples_per_iter * s_step_params->approx_gram_l * sizeof(float));
        cudaMalloc(&d_correction, s_step_params->samples_per_iter * sizeof(float));
        cudaMalloc(&d_G, s_step_params->samples_per_iter * s_step_params->samples_per_iter * sizeof(float));
        cudaMalloc(&d_grad, data_params->n_features * sizeof(float));
        cublasCreate(&handle);
    }

    ~Workspace() {
        cudaFree(d_A);
        cudaFree(d_y);
        cudaFree(d_x);

        cudaFree(d_A_scaled);
        cudaFree(d_batch_A_approx);
        cudaFree(d_correction);
        cudaFree(d_G);
        cudaFree(d_grad);
        cublasDestroy(handle);
    }
};

// CUDA wrapper functions callable from host
void compute_sstep_gradient(const DataParams *data_params, const RunParams *s_step_params, Workspace *workspace, float* A, ProfileStats* stats);
void cuda_apply_sigmoid_block(float *correction, int total_samples, int batch_size, int block_idx);
void cuda_compute_gram_uniform_approx(float *d_A_scaled, int *d_sampled_cols, float *d_G, int samples_per_iter, int n_features, int l);

void train(
    const DataParams* data_params,
    const RunParams* s_step_params,
    Workspace* workspace,
    const std::vector<float>& h_A,
    const std::vector<float>& h_y,
    ProfileStats* stats
);
