#!/usr/bin/env bash
# One-command environment setup for a fresh Linux GPU machine (Lambda, AWS, WSL).
#
#   git clone -b jason-harness https://github.com/deepinsightai/flash-attention.git bench
#   cd bench && bash scripts/lambda_setup.sh            # add --no-tests to skip validation
#
# What it does:
#   1. installs uv if missing and creates ~/bench-venv with Python 3.12 (independent of system Python)
#   2. installs pinned PyTorch (CUDA 12.8 build) and the project requirements
#   3. installs official flash-attn from its prebuilt GitHub wheel; builds from source only if no
#      wheel matches (build uses MAX_JOBS sized to RAM: ~6.5 GB per backward kernel file)
#   4. records the environment to results/<gpu>/environment.txt
#   5. runs the test suites and the FP64 cross-check, so failures show before any timed run
#
# Pins: PyTorch 2.8.0 is the newest version with prebuilt flash-attn 2.8.3.post2 wheels
# (torch 2.4-2.8, CUDA 12, Python 3.12; checked against the v2.8.3.post2 release assets).
set -euo pipefail

TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
FA_VERSION="${FA_VERSION:-2.8.3.post2}"
VENV="${VENV:-$HOME/bench-venv}"
RUN_TESTS=1
[ "${1:-}" = "--no-tests" ] && RUN_TESTS=0

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
t0=$(date +%s)
step() { echo; echo "=== [$(( $(date +%s) - t0 ))s] $*"; }

step "GPU"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader

step "uv and Python 3.12 venv at $VENV"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

step "PyTorch $TORCH_VERSION"
uv pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
uv pip install -r requirements.txt pip setuptools wheel packaging ninja psutil einops

step "flash-attn $FA_VERSION"
if python -c "import flash_attn" 2>/dev/null; then
  echo "already installed: $(python -c 'import flash_attn; print(flash_attn.__version__)')"
else
  WHEEL_URL=$(python - "$FA_VERSION" <<'PY'
import sys, torch
fa = sys.argv[1]
tv = ".".join(torch.__version__.split("+")[0].split(".")[:2])
cuda_major = torch.version.cuda.split(".")[0]
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
name = f"flash_attn-{fa}+cu{cuda_major}torch{tv}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl"
print(f"https://github.com/Dao-AILab/flash-attention/releases/download/v{fa}/{name}")
PY
)
  echo "wheel: $WHEEL_URL"
  if curl -sfIL "$WHEEL_URL" >/dev/null; then
    uv pip install "$WHEEL_URL"
  else
    echo "no prebuilt wheel; building from source for this GPU only"
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
    MEM_GB=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    JOBS=$(( MEM_GB / 7 )); [ "$JOBS" -lt 1 ] && JOBS=1
    [ "$JOBS" -gt "$(nproc)" ] && JOBS=$(nproc)
    echo "arch sm_$CC, MAX_JOBS=$JOBS (RAM available ${MEM_GB} GB)"
    FLASH_ATTN_CUDA_ARCHS="$CC" MAX_JOBS="$JOBS" NVCC_THREADS=1 \
      python -m pip install --no-build-isolation --no-cache-dir "flash-attn==$FA_VERSION"
  fi
fi

step "environment record"
ENV_FILE=$(python - <<'PY'
import torch, pathlib
from bench import metadata
meta = metadata.collect(0)
try:
    import flash_attn
    meta["flash_attn_version"] = flash_attn.__version__
except Exception as e:
    meta["flash_attn_version"] = f"unavailable ({type(e).__name__})"
out = pathlib.Path("results") / metadata.gpu_slug(meta["gpu"]) / "environment.txt"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("".join(f"{k}: {v}\n" for k, v in meta.items()))
print(out)
PY
)
cat "$ENV_FILE"

if [ "$RUN_TESTS" = 1 ]; then
  step "tests (harness, validation, analysis)"
  python -m pytest tests -q -p no:cacheprovider
  step "tests (baselines)"
  (cd "attention-baselines (1)/attention-baselines" && python -m pytest -q -p no:cacheprovider)
  step "cross-check vs FP64 reference"
  python -m scripts.cross_check 2>/dev/null
fi

step "done. Activate with: source $VENV/bin/activate"
