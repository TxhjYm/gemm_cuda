#include "utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

bool check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
    return false;
}

void generate_random_data(std::vector<float>& A, std::vector<float>& B,
                          unsigned int seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    for (float& value : A) {
        value = distribution(generator);
    }
    for (float& value : B) {
        value = distribution(generator);
    }
}

void compute_golden(const std::vector<float>& A, const std::vector<float>& B,
                    std::vector<float>& golden, int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[static_cast<std::size_t>(row) * K + k] *
                       B[static_cast<std::size_t>(k) * N + col];
            }
            golden[static_cast<std::size_t>(row) * N + col] = sum;
        }
    }
}

bool check_result(const std::vector<float>& actual,
                  const std::vector<float>& golden, int M, int N, int K,
                  float absolute_tolerance, float relative_tolerance) {
    std::size_t mismatch_count = 0;
    float max_absolute_error = 0.0f;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float absolute_error = std::fabs(actual[index] - golden[index]);
        const float tolerance =
            absolute_tolerance + relative_tolerance * std::fabs(golden[index]);
        if (absolute_error > tolerance) {
            ++mismatch_count;
        }
        max_absolute_error = std::max(max_absolute_error, absolute_error);
    }

    std::cout << "GEMM: A(" << M << "x" << K << ") * B(" << K << "x" << N
              << ") -> C(" << M << "x" << N << ")\n"
              << "Max absolute error: " << max_absolute_error << '\n'
              << "Tolerance: abs " << absolute_tolerance << " + rel "
              << relative_tolerance << " * |reference|\n";

    if (mismatch_count != 0) {
        std::cerr << "FAILED: " << mismatch_count
                  << " output elements exceeded tolerance.\n";
        return false;
    }

    std::cout << "PASSED: all output elements match the CPU reference.\n";
    return true;
}
