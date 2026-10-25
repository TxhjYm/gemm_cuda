# GEMM CUDA

CUDA GEMM optimization project scaffold.

## Build and run

Requirements: CMake 3.18 or newer, a CUDA Toolkit with `nvcc`, and a CUDA-capable
GPU with a compatible driver. GoogleTest is included as a Git submodule.

Clone the repository and its submodule over SSH:

```bash
git clone --recurse-submodules git@github.com:TxhjYm/gemm_cuda.git
```

For an existing checkout, initialize or update the GoogleTest submodule with:

```bash
./setup.sh
```

```bash
./build.sh make
./run.sh                         # run all tests
./run.sh list                    # list all tests
./run.sh GemmTest.NaiveKernelMatchesReference
./run.sh GemmTest.Naive* GemmTest.Shared*
```

`./run.sh` runs the GoogleTest suite. It benchmarks the naive, shared-memory
tiled, and cuBLAS GEMM implementations using CUDA events after warmup, and
checks each result against a CPU reference. Test names can be exact names or
GoogleTest filter patterns; when multiple names or patterns are provided, they
are combined as alternatives. All implementations use a non-tile-aligned
matrix shape. Tests are skipped when no CUDA device is available.
