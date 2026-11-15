#!/usr/bin/env bash
# Build the official flash-attn package from source in the WSL venv, for the local GPU only.
# Usage (inside WSL):  bash scripts/build_flash_attn.sh > ~/flash_attn_build.log 2>&1
# Progress: grep -o '\[[0-9]*/[0-9]*\]' ~/flash_attn_build.log | tail -1
#
# Memory: one backward kernel file needs ~6.3 GB RAM to compile (measured, sm_120). WSL had
# ~15 GB, so MAX_JOBS above 2 can run WSL out of memory. Default here is 1 (about 80 minutes).
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
source ~/fa-venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-120}"
export MAX_JOBS="${MAX_JOBS:-1}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
echo "start $(date -Is) MAX_JOBS=$MAX_JOBS ARCHS=$FLASH_ATTN_CUDA_ARCHS"
uv pip install setuptools wheel packaging ninja psutil einops pip 2>&1 | tail -1
python -m pip install -v --no-build-isolation --no-cache-dir flash-attn
status=$?
echo "pip exit $status at $(date -Is)"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
