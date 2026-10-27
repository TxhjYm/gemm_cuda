# GEMM CUDA

CUDA GEMM（通用矩阵乘法）优化项目示例。

## 构建和运行

依赖：CMake 3.18 或更高版本、包含 `nvcc` 的 CUDA Toolkit，以及驱动兼容的
CUDA GPU。GoogleTest 作为 Git 子模块包含在项目中。

首次克隆项目及其子模块：

```bash
git clone --recurse-submodules git@github.com:TxhjYm/gemm_cuda.git
```

已有项目副本需要初始化或更新 GoogleTest 子模块时：

```bash
./setup.sh
```

构建并运行测试：

```bash
./build.sh make
./run.sh                         # 运行全部测试
./run.sh list                    # 列出全部测试
./run.sh GemmTest.NaiveKernelMatchesReference
./run.sh GemmTest.Naive* GemmTest.Shared*
```

`./run.sh` 会运行 GoogleTest 测试集。测试会在预热后使用 CUDA Event 测量
naive、shared-memory tiling 和 cuBLAS GEMM，并将结果与 CPU 参考实现进行比较。
测试名可以是完整名称或 GoogleTest filter；提供多个名称或 filter 时会作为
备选项组合。所有实现使用的矩阵形状都不是 tile 大小的整数倍。没有可用
CUDA 设备时会跳过测试。

## 使用 Nsight Compute 分析性能

确保已安装 Nsight Compute，且命令行工具 `ncu` 在 `PATH` 中。选择一个实现运行：

```bash
./scripts/profile.sh naive
./scripts/profile.sh shared
./scripts/profile.sh cublas
```

脚本会在独立的 `build-ncu/` 目录中创建优化构建，启用 CUDA 源码行号信息，
并自动针对第一块可见 GPU 的计算能力编译，然后运行所选实现对应的测试。
默认使用 `basic` 指标集，跳过 10 次预热调用后采集一个 kernel launch。

成功采集后，报告保存在项目的 `results/ncu/` 目录中。例如：

```text
results/ncu/naive.ncu-rep
```

可以用 Nsight Compute 图形界面打开：

```bash
ncu-ui results/ncu/naive.ncu-rep
```

也可以用 `--set full` 收集更全面但耗时更长的指标，或用 `--output FILE`
指定报告路径。若目标报告已存在，脚本会覆盖它：

```bash
./scripts/profile.sh shared --set full
./scripts/profile.sh naive --output /tmp/naive-profile
```

### 如果没有生成报告

运行脚本时请留意终端输出。只有 ncu 成功采集后，才会生成 `.ncu-rep` 文件。
如果看到类似下面的错误：

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

说明当前用户或容器没有读取 GPU 性能计数器的权限；测试本身可能仍通过，但 ncu
无法采集性能数据，因此不会生成有效报告。请联系机器或容器管理员开放 NVIDIA
性能计数器访问权限，然后重新运行 `./scripts/profile.sh naive`。检查默认报告位置：

```bash
ls -lh results/ncu/
```

脚本会在 ncu 失败时返回非零退出码，并提示预期报告路径；不会把未采集成功
误报为成功。共享服务器或容器环境通常需要管理员调整权限，普通用户无法仅靠
修改项目中的 CMake 设置解决这个问题。

### 建议的分析流程

1. 先运行正常测试并记录 `[ PERF ]` 时间：
   `./run.sh GemmTest.NaiveKernelMatchesReference`
2. 采集对应实现的 ncu 报告：`./scripts/profile.sh naive`
3. 在 Nsight Compute 中打开报告。先看 **GPU Speed Of Light**，再检查
   **Memory Workload Analysis**、**Compute Workload Analysis** 和 **Occupancy**，
   了解可能的性能瓶颈。
4. 每次只改动一个因素，重新运行正确性测试，然后用相同参数再次采集报告。
   ncu 报告用于分析硬件性能计数器；正常测试的 CUDA Event 时间用于性能比较。

Nsight Compute 会重放 kernel 来采集硬件计数器，因此分析时测试会比平时运行更慢。
不要用 ncu 运行期间打印的测试耗时作为最终性能数据。naive 和 shared 会按名称
筛选各自的 kernel。cuBLAS 内部选择 kernel，脚本不筛选 kernel 名称，而是在预热后
采集遇到的第一个 kernel。
