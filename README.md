# GEMM CUDA

CUDA GEMM optimization project scaffold.

## Build and run

Requirements: CMake 3.18 or newer, a CUDA Toolkit with `nvcc`, and a CUDA-capable
GPU with a compatible driver.

```bash
./build.sh make
./run.sh
```

The executable tests the naive CUDA GEMM kernel with reproducible cuRAND input,
compares its output against a CPU reference, and reports average kernel latency
and estimated throughput. Other GEMM implementation files are still
placeholders.
