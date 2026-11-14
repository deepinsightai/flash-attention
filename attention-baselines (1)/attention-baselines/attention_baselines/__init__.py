"""Baseline attention implementations for the FlashAttention reproducibility benchmark.

All methods share the interface fn(q, k, v, causal=False, scale=None) -> out with
q [B, H, Nq, d], k [B, H, Nk, d], v [B, H, Nk, dv] and out [B, H, Nq, dv].
"""

from .linformer import linformer_attention, linformer_projections
from .naive import naive_attention
from .performer import orthogonal_gaussian_features, performer_attention, softmax_kernel_features
from .rabe_staats import rabe_staats_attention
from .reference import attention_error, reference_attention
from .registry import METHODS, SUPPORTS_CAUSAL, get_attention

__all__ = [
    "naive_attention",
    "rabe_staats_attention",
    "performer_attention",
    "linformer_attention",
    "orthogonal_gaussian_features",
    "softmax_kernel_features",
    "linformer_projections",
    "reference_attention",
    "attention_error",
    "get_attention",
    "METHODS",
    "SUPPORTS_CAUSAL",
]
