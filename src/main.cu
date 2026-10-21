#include <cuda_runtime.h>
#include <curand.h>

#include "gemm.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

bool check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
    return false;
}

bool check_curand(curandStatus_t status, const char* operation) {
    if (status == CURAND_STATUS_SUCCESS) {
        return true;
    }
    std::cerr << operation << " failed with cuRAND status "
              << static_cast<int>(status) << '\n';
    return false;
}

class DeviceBuffer {
public:
    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    cudaError_t allocate(std::size_t count) {
        return cudaMalloc(&data_, count * sizeof(float));
    }

    float* get() const {
        return data_;
    }

private:
    float* data_ = nullptr;
};

class CurandGenerator {
public:
    ~CurandGenerator() {
        if (generator_ != nullptr) {
            curandDestroyGenerator(generator_);
        }
    }

    curandStatus_t create() {
        return curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_DEFAULT);
    }

    curandGenerator_t get() const {
        return generator_;
    }

private:
    curandGenerator_t generator_ = nullptr;
};

}  // namespace

int main() {
    constexpr int M = 257;
    constexpr int N = 193;
    constexpr int K = 129;
    constexpr int BENCH_ITERATIONS = 100;
    constexpr float ABS_TOLERANCE = 1e-3f;
    constexpr float REL_TOLERANCE = 1e-3f;

    int device_count = 0;
    if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) {
        return 1;
    }
    if (device_count == 0) {
        std::cerr << "No CUDA devices found.\n";
        return 1;
    }
    if (!check_cuda(cudaSetDevice(0), "cudaSetDevice")) {
        return 1;
    }

    cudaDeviceProp properties{};
    if (!check_cuda(cudaGetDeviceProperties(&properties, 0),
                    "cudaGetDeviceProperties")) {
        return 1;
    }
    std::cout << "Running gemm_naive on " << properties.name << " ("
              << properties.major << '.' << properties.minor << ")\n";

    const std::size_t a_count = static_cast<std::size_t>(M) * K;
    const std::size_t b_count = static_cast<std::size_t>(K) * N;
    const std::size_t c_count = static_cast<std::size_t>(M) * N;

    DeviceBuffer device_a;
    DeviceBuffer device_b;
    DeviceBuffer device_c;
    if (!check_cuda(device_a.allocate(a_count), "cudaMalloc(A)") ||
        !check_cuda(device_b.allocate(b_count), "cudaMalloc(B)") ||
        !check_cuda(device_c.allocate(c_count), "cudaMalloc(C)")) {
        return 1;
    }

    CurandGenerator generator;
    if (!check_curand(generator.create(), "curandCreateGenerator") ||
        !check_curand(curandSetPseudoRandomGeneratorSeed(generator.get(), 1234),
                      "curandSetPseudoRandomGeneratorSeed") ||
        !check_curand(curandGenerateUniform(generator.get(), device_a.get(),
                                            a_count),
                      "curandGenerateUniform(A)") ||
        !check_curand(curandGenerateUniform(generator.get(), device_b.get(),
                                            b_count),
                      "curandGenerateUniform(B)")) {
        return 1;
    }

    std::vector<float> host_a(a_count);
    std::vector<float> host_b(b_count);
    std::vector<float> host_c(c_count);
    if (!check_cuda(cudaMemcpy(host_a.data(), device_a.get(),
                               a_count * sizeof(float), cudaMemcpyDeviceToHost),
                    "cudaMemcpy(A to host)") ||
        !check_cuda(cudaMemcpy(host_b.data(), device_b.get(),
                               b_count * sizeof(float), cudaMemcpyDeviceToHost),
                    "cudaMemcpy(B to host)")) {
        return 1;
    }

    gemm_naive(device_a.get(), device_b.get(), device_c.get(), M, N, K);
    if (!check_cuda(cudaGetLastError(), "gemm_naive launch") ||
        !check_cuda(cudaDeviceSynchronize(), "gemm_naive execution")) {
        return 1;
    }

    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};
    if (!check_cuda(cudaEventCreate(&start_event), "cudaEventCreate(start)")) {
        return 1;
    }
    if (!check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate(stop)")) {
        cudaEventDestroy(start_event);
        return 1;
    }

    bool timing_succeeded =
        check_cuda(cudaEventRecord(start_event), "cudaEventRecord(start)");
    if (timing_succeeded) {
        for (int iteration = 0; iteration < BENCH_ITERATIONS; ++iteration) {
            gemm_naive(device_a.get(), device_b.get(), device_c.get(), M, N, K);
        }
        timing_succeeded =
            check_cuda(cudaGetLastError(), "gemm_naive benchmark launch") &&
            check_cuda(cudaEventRecord(stop_event), "cudaEventRecord(stop)") &&
            check_cuda(cudaEventSynchronize(stop_event),
                       "cudaEventSynchronize(stop)");
    }

    float elapsed_ms = 0.0f;
    if (timing_succeeded) {
        timing_succeeded = check_cuda(
            cudaEventElapsedTime(&elapsed_ms, start_event, stop_event),
            "cudaEventElapsedTime");
    }
    cudaEventDestroy(stop_event);
    cudaEventDestroy(start_event);
    if (!timing_succeeded) {
        return 1;
    }

    if (!check_cuda(cudaMemcpy(host_c.data(), device_c.get(),
                               c_count * sizeof(float), cudaMemcpyDeviceToHost),
                    "cudaMemcpy(C to host)")) {
        return 1;
    }

    std::size_t mismatch_count = 0;
    float max_absolute_error = 0.0f;
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float expected = 0.0f;
            for (int k = 0; k < K; ++k) {
                expected += host_a[static_cast<std::size_t>(row) * K + k] *
                            host_b[static_cast<std::size_t>(k) * N + col];
            }

            const std::size_t index = static_cast<std::size_t>(row) * N + col;
            const float actual = host_c[index];
            const float absolute_error = std::fabs(actual - expected);
            const float tolerance =
                ABS_TOLERANCE + REL_TOLERANCE * std::fabs(expected);
            if (absolute_error > tolerance) {
                ++mismatch_count;
            }
            max_absolute_error =
                std::max(max_absolute_error, absolute_error);
        }
    }

    std::cout << "GEMM: A(" << M << "x" << K << ") * B(" << K << "x" << N
              << ") -> C(" << M << "x" << N << ")\n"
              << "Max absolute error: " << max_absolute_error << '\n'
              << "Tolerance: abs " << ABS_TOLERANCE << " + rel "
              << REL_TOLERANCE << " * |reference|\n"
              << "Average kernel time (" << BENCH_ITERATIONS
              << " iterations): "
              << elapsed_ms / BENCH_ITERATIONS << " ms\n"
              << "Estimated throughput: "
              << (2.0 * M * N * K * BENCH_ITERATIONS) /
                     (elapsed_ms * 1.0e6)
              << " GFLOP/s\n";

    if (mismatch_count != 0) {
        std::cerr << "FAILED: " << mismatch_count
                  << " output elements exceeded tolerance.\n";
        return 1;
    }

    std::cout << "PASSED: all output elements match the CPU reference.\n";
    return 0;
}
