#!/usr/bin/env bash
# Time the compilation of ONE flash-attn kernel file for sm_120, to estimate the full build.
# Run inside WSL:  bash scripts/fa_compile_one.sh [kernel-file-name]
set -u
export PATH=$HOME/.local/bin:/usr/local/cuda-12.8/bin:$PATH
source ~/fa-venv/bin/activate
KERNEL="${1:-flash_fwd_hdim128_bf16_sm80.cu}"
W=$HOME/fa_src
if [ ! -d "$W" ]; then
  mkdir -p "$W"; cd "$W"
  URL=$(curl -s https://pypi.org/pypi/flash-attn/json | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(next(u['url'] for u in d['urls'] if u['packagetype']=='sdist'))")
  curl -sL -o src.tar.gz "$URL" && tar xzf src.tar.gz
fi
cd "$W"/flash_attn-*/
ls csrc/flash_attn/src/*.cu | wc -l | xargs echo "kernel files in sdist:"
ls csrc/cutlass/include >/dev/null 2>&1 && echo "cutlass headers: present" || echo "cutlass headers: MISSING"

TORCH=$(python -c "import torch,os;print(os.path.dirname(torch.__file__))")
PYINC=$(python -c "import sysconfig;print(sysconfig.get_paths()['include'])")
OUT=/tmp/fa_one.o
/usr/bin/time -v nvcc -c "csrc/flash_attn/src/$KERNEL" -o "$OUT" \
  -O3 -std=c++17 -gencode arch=compute_120,code=sm_120 \
  -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ \
  -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -Icsrc/flash_attn -Icsrc/flash_attn/src -Icsrc/cutlass/include \
  -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" -I"$PYINC" \
  -DTORCH_EXTENSION_NAME=flash_attn_2_cuda -D_GLIBCXX_USE_CXX11_ABI=1 \
  2>&1 | grep -E "error|Elapsed|Maximum resident|Exit status" | head -12
ls -la "$OUT" 2>/dev/null | awk '{print "object file bytes:", $5}'
