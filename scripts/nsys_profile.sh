#!/usr/bin/env bash
# Nsight Systems capture of one attention call per method (run inside WSL/Linux).
# Usage: bash scripts/nsys_profile.sh [N] [methods...]   then: bash scripts/nsys_stats.sh <methods...>
# Needs Nsight Systems >= 2025 for Blackwell GPUs (the 2024.6 build bundled with CUDA 12.8 records no kernels).
N="${1:-4096}"; shift || true
[ $# -eq 0 ] && set -- naive sdpa_flash
export PATH=/opt/nvidia/nsight-systems/2026.1.3/target-linux-x64:/usr/local/cuda-12.8/bin:$PATH
source ~/fa-venv/bin/activate
mkdir -p /tmp/prof
for m in "$@"; do
  echo "##### nsys: $m N=$N"
  nsys profile -f true --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx \
    -o "/tmp/prof/$m" python -m scripts.profile_target --method "$m" --n "$N" 2>&1 | grep -v -i warn | tail -2
done
