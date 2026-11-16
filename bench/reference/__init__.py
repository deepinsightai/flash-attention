"""Authoritative reference implementations used to calibrate our own kernels.

# The tutorial file next to this wrapper module is third-party code.

triton_tutorial_fused_attention_v3_2_0.py
    The official Triton fused-attention tutorial (FlashAttention-2 algorithm), copied unmodified from
    https://github.com/triton-lang/triton/blob/v3.2.0/python/tutorials/06-fused-attention.py
    (MIT licence; authors: the Triton team, after Tri Dao's FlashAttention-2). v3.2.0 is used rather
    than the release matching our Triton (3.4.0) because 3.3+ switched to TMA tensor descriptors,
    which the tutorial notes only work on post-Ampere GPUs; v3.2.0 uses block pointers and runs on
    the A100. Cite it in the paper.

Limits of the tutorial, enforced by the wrapper rather than by editing the file:
    - FP16 only (probabilities and gradients are cast to float16 inside the kernels).
    - The backward kernel implements causal attention only; its own test suite runs causal only.
    - N must be a multiple of 128 (backward preprocess block).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_TUTORIAL = Path(__file__).with_name("triton_tutorial_fused_attention_v3_2_0.py")
_mod = None


def _load():
    global _mod
    if _mod is None:
        spec = importlib.util.spec_from_file_location("triton_tutorial_fused_attention", _TUTORIAL)
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)
    return _mod


def tutorial_fa2(q, k, v, causal=False, scale=None):
    """Harness interface around the tutorial's `attention(q, k, v, causal, sm_scale)`."""
    if q.dtype != torch.float16:
        raise NotImplementedError("Triton tutorial kernel is FP16-only")
    if q.shape[2] % 128:
        raise NotImplementedError("Triton tutorial kernel needs N to be a multiple of 128")
    needs_grad = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
    if needs_grad and not causal:
        raise NotImplementedError("Triton tutorial backward is causal-only")
    sm_scale = q.shape[-1] ** -0.5 if scale is None else scale
    return _load().attention(q.contiguous(), k.contiguous(), v.contiguous(), causal, sm_scale)
