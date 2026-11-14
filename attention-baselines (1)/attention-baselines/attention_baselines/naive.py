"""Standard scaled dot-product attention (Vaswani et al., 2017).

Materializes the full N x N score and probability matrices: Theta(N^2 d) time,
Theta(N^2) extra memory, Theta(Nd + N^2) HBM traffic. This is the canonical
O(N^2)-memory reference.
"""

from __future__ import annotations

import torch

from .common import causal_mask, check_inputs, default_scale


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    check_inputs(q, k, v, causal)
    s = torch.matmul(q, k.transpose(-2, -1)) * default_scale(q.shape[-1], scale)
    if causal:
        s = s.masked_fill(causal_mask(q.shape[2], k.shape[2], q.device), float("-inf"))
    # torch.softmax subtracts the row max internally, so this is numerically stable.
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)
