#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x build/gemm_tests ]; then
    echo "Executable not found; build the project first with ./build.sh make." >&2
    exit 1
fi

if [ "${1:-}" == "list" ]; then
    if [ "$#" -ne 1 ]; then
        echo "Usage: ./run.sh [list|<test-name-or-filter> ...]" >&2
        exit 1
    fi
    exec ./build/gemm_tests --gtest_list_tests
fi

if [ "$#" -eq 0 ]; then
    exec ./build/gemm_tests
fi

filter="$1"
shift
for test_name in "$@"; do
    filter+=":$test_name"
done

exec ./build/gemm_tests "--gtest_filter=$filter"
