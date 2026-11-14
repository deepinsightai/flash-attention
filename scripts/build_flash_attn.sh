#!/usr/bin/env bash
# Build the official flash-attn package from source in the WSL venv, for the local GPU only.
# Usage: bash scripts/build_flash_attn.sh   (run inside WSL; log with: ... > /tmp/flash_attn_build.log 2>&1)
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
source ~/fa-venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-120}"
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
uv pip install setuptools wheel packaging ninja psutil einops 2>&1 | tail -1
uv pip install flash-attn --no-build-isolation --no-cache 2>&1 | tail -15
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
