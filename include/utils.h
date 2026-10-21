#pragma once

#include <cuda_runtime.h>

#include <vector>

bool check_cuda(cudaError_t status, const char* operation);

void generate_random_data(std::vector<float>& A, std::vector<float>& B,
                          unsigned int seed);

void compute_golden(const std::vector<float>& A, const std::vector<float>& B,
                    std::vector<float>& golden, int M, int N, int K);

bool check_result(const std::vector<float>& actual,
                  const std::vector<float>& golden, int M, int N, int K,
                  float absolute_tolerance, float relative_tolerance);
