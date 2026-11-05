#include "gemm.h"

__global__ void gemm_register_tiling_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int BM = 64;
    constexpr int BK = 8;
    constexpr int BN = 64;
    constexpr int TM = 8;
    constexpr int TN = 8;

    const int tileRow = blockIdx.y;
    const int tileCol = blockIdx.x;

    // 逻辑上的 8×8 线程网格
    const int numThreadsN = BN / TN;  // 8
    const int threadRow = threadIdx.x / numThreadsN;  // 0~7
    const int threadCol = threadIdx.x % numThreadsN;  // 0~7

    // 加载 As[BM][BK] 的索引：64个线程，BK=8列 → 每轮8行，stride=8，共8轮
    const int innerRowA = threadIdx.x / BK;   // 0~7
    const int innerColA = threadIdx.x % BK;   // 0~7
    const int strideA = blockDim.x / BK;      // 8

    // 加载 Bs[BK][BN] 的索引：64个线程，BN=64列 → 每轮1行，stride=1，共8轮
    const int innerRowB = threadIdx.x / BN;   // 0
    const int innerColB = threadIdx.x % BN;   // 0~63
    const int strideB = blockDim.x / BN;      // 1

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int t = 0; t < (K + BK - 1) / BK; t++) {
        // 协作加载 As
        for (int offset = 0; offset < BM; offset += strideA) {
            int localRow = innerRowA + offset;
            int globalRow = tileRow * BM + localRow;
            int globalCol = t * BK + innerColA;
            if (globalRow < M && globalCol < K)
                As[localRow][innerColA] = A[globalRow * K + globalCol];
            else
                As[localRow][innerColA] = 0.0f;
        }

        // 协作加载 Bs
        for (int offset = 0; offset < BK; offset += strideB) {
            int localRow = innerRowB + offset;
            int globalRow = t * BK + localRow;
            int globalCol = tileCol * BN + innerColB;
            if (globalRow < K && globalCol < N)
                Bs[localRow][innerColB] = B[globalRow * N + globalCol];
            else
                Bs[localRow][innerColB] = 0.0f;
        }

        __syncthreads();

        // 计算：外积累加
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            for (int i = 0; i < TM; i++)
                regM[i] = As[threadRow * TM + i][dotIdx];
            for (int i = 0; i < TN; i++)
                regN[i] = Bs[dotIdx][threadCol * TN + i];
            for (int resIdxM = 0; resIdxM < TM; resIdxM++)
                for (int resIdxN = 0; resIdxN < TN; resIdxN++)
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
        }

        __syncthreads();
    }

    // 写回结果
    for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
        for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
            int globalRow = tileRow * BM + threadRow * TM + resIdxM;
            int globalCol = tileCol * BN + threadCol * TN + resIdxN;
            if (globalRow < M && globalCol < N)
                C[globalRow * N + globalCol] = threadResults[resIdxM * TN + resIdxN];
        }
    }
}

void gemm_register_tiling(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(64, 1);
    dim3 grid((N + 63) / 64, (M + 63) / 64);
    gemm_register_tiling_kernel<<<grid, block>>>(A, B, C, M, N, K);
}