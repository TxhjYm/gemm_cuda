#include <cuda_runtime.h>

#include "gemm.h"
#include "utils.h"

#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    constexpr int M = 257;
    constexpr int N = 193;
    constexpr int K = 129;
    constexpr float ABS_TOLERANCE = 1e-3f;
    constexpr float REL_TOLERANCE = 1e-3f;
    constexpr unsigned int RANDOM_SEED = 1234;

    int device_count = 0;
    int exit_code = 1;
    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    std::size_t a_count = static_cast<std::size_t>(M) * K;
    std::size_t b_count = static_cast<std::size_t>(K) * N;
    std::size_t c_count = static_cast<std::size_t>(M) * N;
    std::vector<float> host_a(a_count);
    std::vector<float> host_b(b_count);
    std::vector<float> host_c(c_count);
    std::vector<float> golden(c_count);

    do {
        if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) {
            break;
        }
        if (device_count == 0) {
            std::cerr << "No CUDA devices found.\n";
            break;
        }
        if (!check_cuda(cudaSetDevice(0), "cudaSetDevice")) {
            break;
        }

        generate_random_data(host_a, host_b, RANDOM_SEED);
        compute_golden(host_a, host_b, golden, M, N, K);

        if (!check_cuda(cudaMalloc(&device_a, a_count * sizeof(float)),
                        "cudaMalloc(A)") ||
            !check_cuda(cudaMalloc(&device_b, b_count * sizeof(float)),
                        "cudaMalloc(B)") ||
            !check_cuda(cudaMalloc(&device_c, c_count * sizeof(float)),
                        "cudaMalloc(C)")) {
            break;
        }
        if (!check_cuda(cudaMemcpy(device_a, host_a.data(),
                                   a_count * sizeof(float),
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy(A to device)") ||
            !check_cuda(cudaMemcpy(device_b, host_b.data(),
                                   b_count * sizeof(float),
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy(B to device)")) {
            break;
        }

        gemm_naive(device_a, device_b, device_c, M, N, K);
        if (!check_cuda(cudaGetLastError(), "gemm_naive launch") ||
            !check_cuda(cudaDeviceSynchronize(), "gemm_naive execution")) {
            break;
        }

        if (!check_cuda(cudaMemcpy(host_c.data(), device_c,
                                   c_count * sizeof(float),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy(C to host)")) {
            break;
        }

        exit_code = check_result(host_c, golden, M, N, K, ABS_TOLERANCE,
                                 REL_TOLERANCE)
                        ? 0
                        : 1;
    } while (false);

    if (device_c != nullptr) {
        if (!check_cuda(cudaFree(device_c), "cudaFree(C)")) {
            exit_code = 1;
        }
    }
    if (device_b != nullptr) {
        if (!check_cuda(cudaFree(device_b), "cudaFree(B)")) {
            exit_code = 1;
        }
    }
    if (device_a != nullptr) {
        if (!check_cuda(cudaFree(device_a), "cudaFree(A)")) {
            exit_code = 1;
        }
    }
    return exit_code;
}
