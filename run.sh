#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x build/gemmtestgpu ]; then
    echo "Executable not found; build the project first with ./build.sh make." >&2
    exit 1
fi

exec ./build/gemmtestgpu