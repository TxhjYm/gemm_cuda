# GEMM CUDA

CUDA GEMM optimization project scaffold.

## Build and run

Requirements: CMake 3.18 or newer, a CUDA Toolkit with `nvcc`, and a CUDA-capable
GPU with a compatible driver.

```bash
./build.sh make
./run.sh
```

The executable generates reproducible input on the host, copies it to the GPU,
tests the naive CUDA GEMM kernel once against a CPU reference, and reports
whether the results match. Other GEMM implementation files are still
placeholders.
