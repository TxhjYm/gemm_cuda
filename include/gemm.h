#pragma once

void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_shared_tiling(const float* A, const float* B, float* C, int M, int N, int K);