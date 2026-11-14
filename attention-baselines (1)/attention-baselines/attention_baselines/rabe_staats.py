"""Memory-efficient exact attention (Rabe & Staats, 2021).

Queries and keys are processed in chunks. Each (query chunk, key chunk) pair is
summarized by a triple (m, l, o) -- row max, sum of exponentials, and
exponential-weighted sum of values -- and summaries are merged with the
running-max rescaling of online softmax (Milakov & Gimelshein, 2018).

Peak memory is O(query_chunk * key_chunk) per head instead of O(N^2); time is
still Theta(N^2 d). Gradient checkpointing (as in the paper) recomputes chunk
summaries in the backward pass instead of storing them.

This implementation is deliberately *not* IO-aware: every chunk's scores are
produced by separate PyTorch ops and round-trip through GPU memory. It is the
control for isolating the effect of FlashAttention's fused, IO-aware tiling.
"""

from __future__ import annotations

from functools import partial

import torch
from torch.utils.checkpoint import checkpoint as _checkpoint

from .common import accum_dtype, causal_mask, check_inputs, default_scale


def _summarize_chunk(qc, kc, vc, *, scale, causal, q_start, k_start):
    """Return (m, l, o) for one query chunk against one key chunk, in the accumulation dtype."""
    acc = accum_dtype(qc.dtype)
    s = torch.matmul(qc, kc.transpose(-2, -1)).to(acc) * scale
    if causal:
        mask = causal_mask(qc.shape[2], kc.shape[2], qc.device, q_offset=q_start, k_offset=k_start)
        s = s.masked_fill(mask, float("-inf"))
    # The max only keeps exp() in range; the output does not depend on it, so it is
    # detached. Rows whose keys are all masked get m = -inf and contribute l = o = 0.
    m = s.amax(dim=-1, keepdim=True).detach()
    m_safe = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    p = torch.exp(s - m_safe)
    l = p.sum(dim=-1, keepdim=True)
    o = torch.matmul(p, vc.to(acc))
    return m, l, o


def _query_chunk_attention(qc, k, v, *, scale, key_chunk, causal, q_start, use_ckpt):
    acc = accum_dtype(qc.dtype)
    B, H, nq, _ = qc.shape
    dv = v.shape[-1]
    m = torch.full((B, H, nq, 1), float("-inf"), dtype=acc, device=qc.device)
    l = torch.zeros((B, H, nq, 1), dtype=acc, device=qc.device)
    o = torch.zeros((B, H, nq, dv), dtype=acc, device=qc.device)
    q_end = q_start + nq

    for k_start in range(0, k.shape[2], key_chunk):
        if causal and k_start > q_end - 1:
            break  # every remaining key is in the future for every query in this chunk
        k_end = min(k_start + key_chunk, k.shape[2])
        fn = partial(_summarize_chunk, scale=scale, causal=causal, q_start=q_start, k_start=k_start)
        args = (qc, k[:, :, k_start:k_end], v[:, :, k_start:k_end])
        m_c, l_c, o_c = _checkpoint(fn, *args, use_reentrant=False) if use_ckpt else fn(*args)

        # Merge (m, l, o) with (m_c, l_c, o_c): the associative operator from the preliminaries.
        m_new = torch.maximum(m, m_c)
        m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
        alpha = torch.exp(m - m_safe)
        beta = torch.exp(m_c - m_safe)
        l = alpha * l + beta * l_c
        o = alpha * o + beta * o_c
        m = m_new

    return (o / l).to(qc.dtype)


def rabe_staats_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
    query_chunk: int = 1024,
    key_chunk: int = 4096,
    checkpoint: bool = True,
) -> torch.Tensor:
    """Exact attention with O(query_chunk * key_chunk) working memory per head.

    The defaults (1024, 4096) are the chunk sizes used in the paper. Setting both to
    about sqrt(N) gives the O(sqrt(N)) memory regime discussed in the paper.
    """
    check_inputs(q, k, v, causal)
    scale = default_scale(q.shape[-1], scale)
    needs_grad = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
    use_ckpt = checkpoint and needs_grad

    outs = []
    for q_start in range(0, q.shape[2], query_chunk):
        q_end = min(q_start + query_chunk, q.shape[2])
        fn = partial(
            _query_chunk_attention,
            scale=scale, key_chunk=key_chunk, causal=causal, q_start=q_start, use_ckpt=use_ckpt,
        )
        qc = q[:, :, q_start:q_end]
        # Outer checkpoint: store only this chunk's output, not the per-key-chunk merge state.
        outs.append(_checkpoint(fn, qc, k, v, use_reentrant=False) if use_ckpt else fn(qc, k, v))
    return torch.cat(outs, dim=2)
