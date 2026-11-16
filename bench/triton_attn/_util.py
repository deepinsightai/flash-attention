"""Shared helpers for the Triton attention kernels."""

from __future__ import annotations

import math

import torch
import triton

LOG2E = 1.4426950408889634


def check_qkv(q, k, v, causal: bool, need_equal_dims: bool = False):
    """Validate the interface shapes and return contiguous [B, H, N, d] tensors."""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("expected 4-D tensors [B, H, N, d]")
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("q, k, v must share batch, head and head dimension")
    if k.shape[2] != v.shape[2]:
        raise ValueError("k and v must have the same sequence length")
    if q.shape[2] != k.shape[2]:
        raise ValueError("the Triton kernels support self-attention only (Nq == Nk)")
    if need_equal_dims and v.shape[-1] != q.shape[-1]:
        raise ValueError("the fused Triton kernels need dv == d")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("Triton kernels need CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported dtype {q.dtype}")
    return q.contiguous(), k.contiguous(), v.contiguous()


def check_head_dim(d: int):
    if d < 16 or d > 256 or d & (d - 1):
        raise ValueError(f"head_dim must be a power of two in [16, 256]; got {d}")


def default_scale(d: int, scale):
    return 1.0 / math.sqrt(d) if scale is None else float(scale)


def dot_precision(dtype: torch.dtype) -> str:
    """FP32 inputs use full-precision FMA dots so errors compare fairly with the FP64 reference."""
    return "ieee" if dtype == torch.float32 else "tf32"


# Tuning hooks. Tuned launch configurations, filled by bench.triton_attn.tuning
# (from configs/tuning/<gpu>.json) or by scripts/tune_triton.py during a search. Keys:
#   flash:   (version, "fwd" | "bwd", d, dtype_name) -> (BLOCK_M, BLOCK_N, num_warps, num_stages)
#   bmm:     tag ("naive" | "rabe_staats")           -> (BM, BN, BK, num_warps, num_stages)
#   softmax: "naive"                                 -> (BLOCK, num_warps)
#   rs_elem: "rabe_staats"                           -> num_warps for the summary/merge/elementwise kernels
TUNED: dict = {"flash": {}, "bmm": {}, "softmax": {}, "rs_elem": {}}


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def flash_config(d: int, dtype: torch.dtype, bwd: bool = False, version: int | None = None):
    """(BLOCK_M, BLOCK_N, num_warps, num_stages) chosen to fit ~100 KB of shared memory.

    One fixed table, no autotuning, so every FlashAttention variant gets the same tile shapes
    and the comparison between them reflects the algorithm, not the tuner. A tuned entry in
    TUNED["flash"] for this (version, pass, d, dtype) takes precedence when present.
    """
    tuned = TUNED["flash"].get((version, "bwd" if bwd else "fwd", d, dtype_name(dtype)))
    if tuned is not None:
        return tuple(tuned)
    wide = dtype == torch.float32
    if bwd:
        if d <= 64:
            return (32, 64, 4, 2) if wide else (64, 64, 4, 2)
        return (16, 32, 4, 1) if wide else (32, 64, 4, 1)
    if d <= 64:
        return (64, 32, 4, 2) if wide else (128, 64, 4, 2)
    return (32, 32, 4, 1) if wide else (128, 32, 4, 2)


def cdiv(a, b):
    return triton.cdiv(a, b)
