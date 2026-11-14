"""Shared helpers: input validation, default scale, accumulation dtype."""

from __future__ import annotations

import math

import torch


def check_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> None:
    """Validate the common interface: q [B, H, Nq, d], k [B, H, Nk, d], v [B, H, Nk, dv]."""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"expected 4-D tensors [B, H, N, d]; got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if q.shape[:2] != k.shape[:2] or k.shape[:2] != v.shape[:2]:
        raise ValueError("q, k, v must share batch and head dimensions")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension")
    if k.shape[2] != v.shape[2]:
        raise ValueError("k and v must have the same sequence length")
    if causal and q.shape[2] != k.shape[2]:
        raise ValueError("causal attention requires Nq == Nk (self-attention)")


def default_scale(head_dim: int, scale: float | None) -> float:
    return 1.0 / math.sqrt(head_dim) if scale is None else scale


def accum_dtype(dtype: torch.dtype) -> torch.dtype:
    """Accumulate in at least FP32 (FP64 inputs stay FP64), as GPU attention kernels do."""
    return torch.promote_types(dtype, torch.float32)


def causal_mask(nq: int, nk: int, device, q_offset: int = 0, k_offset: int = 0) -> torch.Tensor:
    """Boolean mask, True where key position > query position (i.e. masked out)."""
    qi = torch.arange(q_offset, q_offset + nq, device=device).unsqueeze(1)
    kj = torch.arange(k_offset, k_offset + nk, device=device).unsqueeze(0)
    return kj > qi
