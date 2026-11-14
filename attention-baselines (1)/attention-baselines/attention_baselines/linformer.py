"""Linformer attention (Wang et al., 2020).

Projects keys and values along the sequence axis with E, F in R^{k x N}:
softmax(Q (E K)^T / sqrt(d)) (F V). Theta(N k d) time, Theta(N k) memory.

In the paper E and F are learned. In an untrained micro-benchmark they must be fixed,
and the choice decides the approximation error (runtime and memory do not depend on it):

* "gaussian": E = F with i.i.d. N(0, 1/k) entries (Johnson-Lindenstrauss style). Faithful
  to the paper's analysis, but on random inputs the output error exceeds 100% for all k:
  random projections without training do not approximate attention.
* "pooling": E = F averages k contiguous segments of the sequence. Deterministic, exact
  when k == N, and its error falls as k grows; error depends on how similar neighbouring
  tokens are.

Causal masking is not supported: every projected key mixes all N positions, including
future ones.
"""

from __future__ import annotations

import torch

from .common import check_inputs
from .naive import naive_attention


def linformer_projections(
    seq_len: int,
    proj_dim: int,
    *,
    kind: str = "gaussian",
    generator: torch.Generator | None = None,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a [proj_dim, seq_len] projection matrix (shared for keys and values)."""
    if kind == "gaussian":
        e = torch.randn(proj_dim, seq_len, generator=generator, dtype=torch.float64) / proj_dim ** 0.5
    elif kind == "pooling":
        if proj_dim > seq_len:
            raise ValueError("pooling projection needs proj_dim <= seq_len")
        e = torch.zeros(proj_dim, seq_len, dtype=torch.float64)
        bounds = [(r * seq_len) // proj_dim for r in range(proj_dim + 1)]  # strictly increasing
        for r in range(proj_dim):
            a, b = bounds[r], bounds[r + 1]
            e[r, a:b] = 1.0 / (b - a)
    else:
        raise ValueError(f"unknown projection kind {kind!r}; use 'gaussian' or 'pooling'")
    return e.to(device=device, dtype=dtype)


def linformer_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
    *,
    proj_k: torch.Tensor | None = None,
    proj_v: torch.Tensor | None = None,
    proj_dim: int = 256,
    kind: str = "gaussian",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Linformer attention. Pass fixed `proj_k` (and optionally `proj_v`) for reproducible benchmarks."""
    if causal:
        raise NotImplementedError(
            "Linformer does not support causal masking: projecting along the sequence axis "
            "mixes future positions into every projected key and value."
        )
    check_inputs(q, k, v, causal)
    n = k.shape[2]
    if proj_k is None:
        proj_k = linformer_projections(n, min(proj_dim, n), kind=kind, generator=generator)
    if proj_v is None:
        proj_v = proj_k
    if proj_k.shape[1] != n or proj_v.shape[1] != n:
        raise ValueError(f"projection matrices must have {n} columns (the key sequence length)")
    # [r, N] @ [B, H, N, d] broadcasts to [B, H, r, d]
    k_proj = torch.matmul(proj_k.to(device=k.device, dtype=k.dtype), k)
    v_proj = torch.matmul(proj_v.to(device=v.device, dtype=v.dtype), v)
    return naive_attention(q, k_proj, v_proj, causal=False, scale=scale)
