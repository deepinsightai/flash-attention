"""One entry point for the benchmark harness: build any baseline with its random parameters fixed.

Example
-------
    attn = get_attention("performer", seq_len=4096, head_dim=128, device="cuda", seed=0, num_features=256)
    out = attn(q, k, v, causal=True)

Random parameters (Performer features, Linformer projections) are drawn once here, so they
are not regenerated -- or timed -- on every call.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable

import torch

from .linformer import linformer_attention, linformer_projections
from .naive import naive_attention
from .performer import orthogonal_gaussian_features, performer_attention
from .rabe_staats import rabe_staats_attention

METHODS = ("naive", "rabe_staats", "performer", "linformer")
SUPPORTS_CAUSAL = {"naive": True, "rabe_staats": True, "performer": True, "linformer": False}


def get_attention(
    name: str,
    *,
    seq_len: int,
    head_dim: int,
    device="cpu",
    seed: int = 0,
    **kwargs,
) -> Callable[..., torch.Tensor]:
    """Return fn(q, k, v, causal=False) -> out for the named method.

    kwargs:
      rabe_staats: query_chunk, key_chunk, checkpoint
      performer:   num_features (default round(d ln d)), causal_chunk
      linformer:   proj_dim (default 256), kind ("gaussian" | "pooling")
    """
    gen = torch.Generator().manual_seed(seed)
    if name == "naive":
        return naive_attention
    if name == "rabe_staats":
        return partial(rabe_staats_attention, **kwargs)
    if name == "performer":
        m = kwargs.pop("num_features", None) or max(1, round(head_dim * math.log(head_dim)))
        w = orthogonal_gaussian_features(m, head_dim, generator=gen, device=device)
        return partial(performer_attention, features=w, **kwargs)
    if name == "linformer":
        proj_dim = min(kwargs.pop("proj_dim", 256), seq_len)
        kind = kwargs.pop("kind", "gaussian")
        e = linformer_projections(seq_len, proj_dim, kind=kind, generator=gen, device=device)
        return partial(linformer_attention, proj_k=e, **kwargs)
    raise ValueError(f"unknown method {name!r}; choose from {METHODS}")
