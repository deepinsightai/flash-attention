#!/usr/bin/env bash
# Colab / Linux setup. Run in a notebook cell with:  !bash scripts/colab_setup.sh
set -euo pipefail

nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

pip install -q -r requirements.txt pytest

# Triton ships with PyTorch on Linux; print the version for the record.
python -c "import triton; print('triton', triton.__version__)" || true

# Official FlashAttention (FA2) reference arm. Building from source can take 10+ minutes;
# the harness skips 'flash_attn' automatically if it is missing.
# pip install flash-attn --no-build-isolation

python -m pytest tests -q
