"""Triton implementations of the baselines and the FlashAttention family.

All functions share the harness interface fn(q, k, v, causal=False, scale=None) -> out on
[B, H, N, d] tensors, support autograd, and accumulate in FP32.

    triton_naive         standard attention, N x N matrices in HBM
    triton_rabe_staats   chunked exact attention, not IO-aware
    triton_fa1 .. fa4    FlashAttention generations (see flash_fwd.py for what each one models)
"""

from __future__ import annotations

import torch

from ._util import check_head_dim, check_qkv, default_scale
from .flash_bwd import flash_bwd
from .flash_fwd import flash_fwd
from .naive import naive_attention_triton
from .rabe_staats import rabe_staats_triton


class _FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, scale, version):
        o, lse = flash_fwd(q, k, v, causal, scale, version)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.cfg = (causal, scale, version)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        causal, scale, version = ctx.cfg
        dq, dk, dv = flash_bwd(q, k, v, o, do, lse, causal, scale, version)
        return dq, dk, dv, None, None, None


def flash_attention_triton(q, k, v, causal=False, scale=None, version=2):
    q, k, v = check_qkv(q, k, v, causal, need_equal_dims=True)
    check_head_dim(q.shape[-1])
    return _FlashAttention.apply(q, k, v, causal, default_scale(q.shape[-1], scale), version)


def _make(version):
    def fn(q, k, v, causal=False, scale=None):
        return flash_attention_triton(q, k, v, causal=causal, scale=scale, version=version)

    fn.__name__ = f"flash_attention_triton_v{version}"
    return fn


triton_fa1, triton_fa2, triton_fa3, triton_fa4 = (_make(i) for i in (1, 2, 3, 4))

METHODS = {
    "triton_naive": naive_attention_triton,
    "triton_rabe_staats": rabe_staats_triton,
    "triton_fa1": triton_fa1,
    "triton_fa2": triton_fa2,
    "triton_fa3": triton_fa3,
    "triton_fa4": triton_fa4,
}

__all__ = ["METHODS", "flash_attention_triton", "naive_attention_triton", "rabe_staats_triton",
           "triton_fa1", "triton_fa2", "triton_fa3", "triton_fa4"]

# Apply tuned launch configurations for this GPU, if a tuning file exists.
from . import tuning as _tuning  # noqa: E402

TUNING_FILE = _tuning.load()
