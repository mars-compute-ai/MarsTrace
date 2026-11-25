#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
nvcc -O3 -arch=sm_90a -lineinfo -o build/matmul driver.cu
