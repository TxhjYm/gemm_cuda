#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${project_dir}/build-ncu"
output_dir="${project_dir}/results/ncu"
ncu_set="basic"

usage() {
    cat <<'EOF'
Usage: ./scripts/profile.sh <naive|shared|cublas> [--set basic|full] [--output FILE]

Profiles one GEMM test with Nsight Compute and saves a .ncu-rep report.
The default metric set is "basic". The output defaults to
results/ncu/<kernel>.ncu-rep.
EOF
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi

kernel="$1"
shift
case "$kernel" in
    naive)
        test_name="GemmTest.NaiveKernelMatchesReference"
        kernel_filter=(--kernel-name 'regex:gemm_naive_kernel')
        ;;
    shared)
        test_name="GemmTest.SharedTilingKernelMatchesReference"
        kernel_filter=(--kernel-name 'regex:gemm_shared_tiling_kernel')
        ;;
    cublas)
        test_name="GemmTest.CublasMatchesReference"
        kernel_filter=()
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown kernel: ${kernel}" >&2
        usage >&2
        exit 2
        ;;
esac

report="${output_dir}/${kernel}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --set)
            if [[ $# -lt 2 ]]; then
                echo "--set requires a value" >&2
                exit 2
            fi
            ncu_set="$2"
            shift 2
            ;;
        --output)
            if [[ $# -lt 2 ]]; then
                echo "--output requires a path" >&2
                exit 2
            fi
            report="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v ncu >/dev/null 2>&1; then
    echo "Nsight Compute CLI (ncu) was not found in PATH." >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi was not found; cannot select the profiling GPU architecture." >&2
    exit 1
fi

compute_capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
if [[ ! "$compute_capability" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
    echo "Could not read the compute capability of the first visible GPU." >&2
    exit 1
fi
cuda_architecture="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"

mkdir -p "$(dirname "$report")"

cmake -S "$project_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGEMM_CUDA_LINEINFO=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$cuda_architecture"
cmake --build "$build_dir" --target gemm_tests --parallel

export_path="$report"
report_file="${report}.ncu-rep"
if [[ "$report" == *.ncu-rep ]]; then
    report_file="$report"
    export_path="${report%.ncu-rep}"
fi

echo "正在分析 ${kernel}；报告预期位置：${report_file}"
ncu_log="$(mktemp)"
trap 'rm -f "$ncu_log"' EXIT

set +e
ncu --set "$ncu_set" \
    "${kernel_filter[@]}" \
    --launch-skip 10 \
    --launch-count 1 \
    --force-overwrite \
    --export "$export_path" \
    "$build_dir/gemm_tests" \
    "--gtest_filter=${test_name}" 2>&1 | tee "$ncu_log"
ncu_status=${PIPESTATUS[0]}
set -e

if [[ $ncu_status -ne 0 ]]; then
    echo "Nsight Compute 分析失败，没有生成有效报告：${report_file}" >&2
    if grep -q "ERR_NVGPUCTRPERM" "$ncu_log"; then
        cat >&2 <<'EOF'
原因：当前用户或容器没有访问 NVIDIA GPU 性能计数器的权限。
请联系机器/容器管理员开放 Nsight Compute 性能计数器访问权限，
然后重新运行本命令。仅修改项目的 CMake 配置无法解决此权限问题。
EOF
    fi
    exit "$ncu_status"
fi

if [[ ! -s "$report_file" ]]; then
    echo "Nsight Compute 返回成功，但没有找到预期报告文件：${report_file}" >&2
    exit 1
fi

echo "分析完成，报告已保存：${report_file}"
