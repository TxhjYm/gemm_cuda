#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "gemm.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

constexpr int kM = 512;
constexpr int kN = 624;
constexpr int kK = 129;
constexpr float kAbsoluteTolerance = 1e-3f;
constexpr float kRelativeTolerance = 1e-3f;
constexpr float kInitialOutputValue = 1.25f;
constexpr unsigned int kRandomSeed = 1234;
constexpr int kWarmupIterations = 10;
constexpr int kTimedIterations = 100;

using GemmFunction =
    void (*)(const float*, const float*, float*, int, int, int);

class GemmTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices found.";
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

        generate_random_data(host_a_, host_b_, kRandomSeed);
        compute_golden(host_a_, host_b_, golden_, kM, kN, kK);

        ASSERT_EQ(cudaMalloc(&device_a_, a_count_ * sizeof(float)),
                  cudaSuccess);
        ASSERT_EQ(cudaMalloc(&device_b_, b_count_ * sizeof(float)),
                  cudaSuccess);
        ASSERT_EQ(cudaMalloc(&device_c_, c_count_ * sizeof(float)),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(device_a_, host_a_.data(),
                             a_count_ * sizeof(float),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(device_b_, host_b_.data(),
                             b_count_ * sizeof(float),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaEventCreate(&start_event_), cudaSuccess);
        ASSERT_EQ(cudaEventCreate(&stop_event_), cudaSuccess);
    }

    void TearDown() override {
        if (stop_event_ != nullptr) {
            EXPECT_EQ(cudaEventDestroy(stop_event_), cudaSuccess);
        }
        if (start_event_ != nullptr) {
            EXPECT_EQ(cudaEventDestroy(start_event_), cudaSuccess);
        }
        if (device_c_ != nullptr) {
            EXPECT_EQ(cudaFree(device_c_), cudaSuccess);
        }
        if (device_b_ != nullptr) {
            EXPECT_EQ(cudaFree(device_b_), cudaSuccess);
        }
        if (device_a_ != nullptr) {
            EXPECT_EQ(cudaFree(device_a_), cudaSuccess);
        }
    }

    void ExpectKernelMatchesReference(GemmFunction gemm,
                                      const char* kernel_name) {
        std::fill(host_c_.begin(), host_c_.end(), kInitialOutputValue);
        ASSERT_EQ(cudaMemcpy(device_c_, host_c_.data(),
                             c_count_ * sizeof(float),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);

        for (int i = 0; i < kWarmupIterations; ++i) {
            gemm(device_a_, device_b_, device_c_, kM, kN, kK);
            ASSERT_EQ(cudaGetLastError(), cudaSuccess)
                << "warmup launch " << i;
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaEventRecord(start_event_), cudaSuccess);
        for (int i = 0; i < kTimedIterations; ++i) {
            gemm(device_a_, device_b_, device_c_, kM, kN, kK);
            ASSERT_EQ(cudaGetLastError(), cudaSuccess)
                << "timed launch " << i;
        }
        ASSERT_EQ(cudaEventRecord(stop_event_), cudaSuccess);
        ASSERT_EQ(cudaEventSynchronize(stop_event_), cudaSuccess);

        float elapsed_milliseconds = 0.0f;
        ASSERT_EQ(cudaEventElapsedTime(&elapsed_milliseconds, start_event_,
                                       stop_event_),
                  cudaSuccess);
        std::cout << "[ PERF ] " << kernel_name << ": "
                  << elapsed_milliseconds / kTimedIterations
                  << " ms/iteration (average of " << kTimedIterations
                  << " launches after " << kWarmupIterations
                  << " warmup launches)\n";

        ASSERT_EQ(cudaMemcpy(host_c_.data(), device_c_,
                             c_count_ * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);

        for (std::size_t i = 0; i < c_count_; ++i) {
            const float absolute_error = std::fabs(host_c_[i] - golden_[i]);
            const float tolerance =
                kAbsoluteTolerance +
                kRelativeTolerance * std::fabs(golden_[i]);
            EXPECT_LE(absolute_error, tolerance) << "output index " << i;
        }
    }

private:
    static constexpr std::size_t a_count_ =
        static_cast<std::size_t>(kM) * kK;
    static constexpr std::size_t b_count_ =
        static_cast<std::size_t>(kK) * kN;
    static constexpr std::size_t c_count_ =
        static_cast<std::size_t>(kM) * kN;

    std::vector<float> host_a_ = std::vector<float>(a_count_);
    std::vector<float> host_b_ = std::vector<float>(b_count_);
    std::vector<float> host_c_ = std::vector<float>(c_count_);
    std::vector<float> golden_ = std::vector<float>(c_count_);
    float* device_a_ = nullptr;
    float* device_b_ = nullptr;
    float* device_c_ = nullptr;
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_ = nullptr;
};

TEST_F(GemmTest, NaiveKernelMatchesReference) {
    ExpectKernelMatchesReference(gemm_naive, "gemm_naive");
}

TEST_F(GemmTest, SharedTilingKernelMatchesReference) {
    ExpectKernelMatchesReference(gemm_shared_tiling, "gemm_shared_tiling");
}

}  // namespace
