#include "gemm.h"

// =============================================================================
// Register tiling 性能分析与优化方法
// =============================================================================
//
// 基准形状 M=512, N=624, K=129（见 tests/test_gemm.cu），H100 PCIe 实测：
//
//   原始版本  64x64 tile, 64 线程, TM=TN=8     54.6 us   （每 block 2 个 warp）
//   当前版本  32x64 tile, 128 线程, TM=TN=4    33.4 us   （已应用 [A]）
//   cuBLAS                                     14.4 us
//
// -----------------------------------------------------------------------------
// 一、原始版本为什么慢（ncu 实测，64x64 / 64 线程）
// -----------------------------------------------------------------------------
//
// 1) 并行度极低（主因）
//    grid = (ceil(624/64), ceil(512/64)) = 80 个 block x 64 线程 = 160 个 warp。
//    H100 有 114 个 SM：80 个 block 只覆盖 80 个 SM（34 个空闲），每个活跃
//    SM 只有 2 个 warp（占用率 3.1%，硬件上限 64 warp/SM）。
//    ncu 实测：每 scheduler 平均 1 个 active warp、0.21 个 eligible warp，
//    78.6% 的周期没有指令可发射，IPC 仅 0.43（上限 4）。
//
// 2) global load 延迟完全暴露
//    每轮 k-tile 都是 load -> __syncthreads -> 计算 -> __syncthreads，共 17 轮；
//    只有 2 个 warp，无法掩盖 global 访存延迟：ncu 显示 46% 的 stall 是
//    long scoreboard（等 global load 数据），barrier 只占 0.05。
//    且 load 循环的步长取自运行时值 blockDim.x / BK，ptxas 无法完全展开，
//    SASS 里只有 4 条 LDG，8 次 load 实际是串行发出的。
//
// 3) As 的 bank conflict
//    As[64][8] 行距为 8 个 float。读取 As[threadRow*8 + i][dotIdx] 时，一个
//    warp 内 threadRow = 0..3，四个地址相差 64 个 float，而 64 % 32 == 0，
//    四组线程落在同一个 bank -> 每次 LDS 产生 4-way conflict。
//    ncu 实测 696,320 次 bank conflict / 1,044,480 个 shared-load wavefront。
//    （Bs 被编译器合并成 LDS.128，冲突不严重。）
//
// 4) tile 尾部浪费（次要）
//    K=129、BK=8 -> 17 轮，最后一轮 7/8 是 0；N=624 -> 最后一列 tile 只有
//    48/64 列有效。合计约 8% 的无效计算。
//
// 对比：shared tiling 有 1248 个 block x 256 线程（约 1 万个 warp），
// occupancy 接近 100%，所以虽然 shared load 多 8 倍，反而只要 29 us。
// register tiling 省掉了 shared load，却把并行度降到 1/60，得不偿失。
//
// -----------------------------------------------------------------------------
// 二、优化方法（按收益排序）
// -----------------------------------------------------------------------------
//
// [A] 已应用：减少每线程结果数，增加 block 数和每 block 线程数
//     64x64/64 线程 -> 32x64/128 线程：grid 80 -> 160 个 block，总 warp 数
//     160 -> 640，寄存器 94 -> 56（理论允许 9 个 block/SM）。实测 54.6 -> 33.4 us。
//     注意：160 个 block 在 114 个 SM 上仍只有 1.4 waves，每 SM 约 4~8 个
//     warp；继续减小 tile（如 32x32/64 线程 -> 320 个 block）还能改善负载均衡。
//
// [B] 已应用：编译期确定加载步长，并展开加载和外积计算循环
//     kThreads = (BM / TM) * (BN / TN)，strideA = kThreads / BK，
//     strideB = kThreads / BN；当前配置下分别为 128、16、2。A 加载循环
//     每线程执行 2 次，B 加载循环每线程执行 4 次；两个循环及 dotIdx
//     外积计算循环均加了 #pragma unroll。加载循环保留 offset 写法即可：
//
//         #pragma unroll
//         for (int offset = 0; offset < BM; offset += strideA) {
//             int localRow = innerRowA + offset;
//             ...
//         }
//
//     修改 tile 尺寸或线程布局时，要确认 kThreads 与实际 block 大小一致，
//     且每线程步长可覆盖 tile。完全展开提供更多指令调度空间，但不保证
//     所有 LDG 都背靠背发出；性能收益需在目标 GPU/配置上实测。
//
// [C] 可选：As padding，消除 bank conflict
//     As[BM][BK + 1]（行距 9 个 float）可以打散 bank。但收益取决于每个 warp
//     里有几个 threadRow：64x64 时是 4-way（约 5%）；当前 32x64 只有 2-way，
//     实测基本持平甚至略慢（20.2 -> 21.6 us）；32x32 时约 4%（18.9 -> 18.1 us）。
//     建议改完 [B] 后实测再决定。注意 Bs 不要 padding（Bs[BK][BN + 1]），
//     否则会破坏 LDS.128 的 16 字节对齐，反而更慢。
//
// [D] BK 从 8 加到 16（配合 [B] 后 17.8 us）
//     每个 k-tile 的计算量翻倍，load/barrier 轮数 17 -> 9；shared 只增加约
//     2 KB（As[32][17] + Bs[16][64] 约 6.3 KB）。代价是 K=129 时最后一轮
//     15/16 为 0，无效计算从约 5% 增加到约 12%，但总体上仍是净收益。
//
// [E] 不要过早做 double buffering / 寄存器预取
//     64 线程版本把下一块 tile 预取到寄存器后，寄存器涨到 160，实测反而
//     36 us（不预取 30.7 us）；当前几何下预取也只有约 1 us 的波动。
//     double buffering 是缓冲/流水线策略，并非只能在 sm_80+ 实现；较老架构
//     也可用普通 global load、shared memory 和同步来实现，但异步搬运与计算
//     重叠的能力和开销不同。sm_80+ 才支持 cp.async，可用于异步拷贝 global
//     memory 到 shared memory；需要流水线时可优先评估它，并实测寄存器、占用率
//     和性能收益。
//
// [F] 编译目标架构
//     CMakeLists.txt 目前只生成 sm_75，H100 上靠驱动 JIT compute_75 PTX 运行
//     （首次 launch 时编译，不计入 benchmark，但用不了 Hopper 特性）。
//     建议 -DCMAKE_CUDA_ARCHITECTURES=90，ncu 里也能看到原生 SASS。
//
// [G] 进一步优化：split-K
//     M x N 只有 32 万个输出，本身无法喂满 114 个 SM。可以把 K=129 拆成
//     2~4 段，每个 block 只算一段，最后用 atomicAdd 或第二个 kernel 归约
//     （cuBLAS 在小形状上通常也会选择 split-K）。
//
// -----------------------------------------------------------------------------
// 三、实测汇总（M=512, N=624, K=129, H100 PCIe, 500 次平均）
// -----------------------------------------------------------------------------
//   原始 64x64 / 64 线程                      54.0 us
//   当前 32x64 / 128 线程                      33.4 us（repo 内 build，JIT）
//   32x64 / 128 线程 + [B] 展开 load 循环      20.2 us
//   32x64 / 128 线程 + [B] + [D] BK=16         17.8 us
//   32x32 /  64 线程 + [B]                     18.9 us（再加 [C] 为 18.1 us）
//   cuBLAS                                     14.0~14.4 us
//
// 除注明外均为 standalone 基准（nvcc -arch=sm_90）的结果；原实现在 standalone
// 和 repo（compute_75 PTX + JIT）下分别为 54.0 / 54.6 us，基本一致。
// =============================================================================

namespace {

constexpr int BM = 32;
constexpr int BK = 8;
constexpr int BN = 64;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int kThreads = (BM / TM) * (BN / TN);

}  // namespace

__global__ void gemm_register_tiling_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int tileRow = blockIdx.y;
    const int tileCol = blockIdx.x;

    // 逻辑线程网格：BM/TM = 8 行 × BN/TN = 16 列 = 128 个线程
    const int numThreadsN = BN / TN;  // 16
    const int threadRow = threadIdx.x / numThreadsN;  // 0~7
    const int threadCol = threadIdx.x % numThreadsN;  // 0~15

    // 加载 As[BM][BK] 的索引：128个线程，BK=8列 → 每轮16行，stride=16，共2轮
    const int innerRowA = threadIdx.x / BK;   // 0~15
    const int innerColA = threadIdx.x % BK;   // 0~7
    constexpr int strideA = kThreads / BK;      // 16

    // 加载 Bs[BK][BN] 的索引：128个线程，BN=64列 → 每轮2行，stride=2，共4轮
    const int innerRowB = threadIdx.x / BN;   // 0~1
    const int innerColB = threadIdx.x % BN;   // 0~63
    constexpr int strideB = kThreads / BN;      // 2

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int t = 0; t < (K + BK - 1) / BK; t++) {
        // 协作加载 As
        #pragma unroll
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
        #pragma unroll
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
        #pragma unroll
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
    dim3 block((BM/TM) * (BN/TN), 1);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_register_tiling_kernel<<<grid, block>>>(A, B, C, M, N, K);
}