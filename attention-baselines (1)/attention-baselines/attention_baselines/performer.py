"""Performer / FAVOR+ attention (Choromanski et al., 2021).

Approximates the softmax kernel exp(q.k / sqrt(d)) by phi(q).phi(k) with m positive
orthogonal random features, then reorders the product so no N x N matrix is formed:
Theta(N m d) time, Theta(N m + m d) extra memory.

Accuracy depends strongly on the norms of q and k: the estimator's variance grows
roughly like exp(||q + k||^2 / sqrt(d)). On unit-variance random inputs with d = 64 the
error is large for any practical m; on small-norm inputs it falls like 1/sqrt(m).
Report the input scale alongside any Performer error number.
"""

from __future__ import annotations

import math

import torch

from .common import accum_dtype, check_inputs, default_scale


def orthogonal_gaussian_features(
    num_features: int,
    head_dim: int,
    *,
    generator: torch.Generator | None = None,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw an [m, d] matrix whose rows are marginally N(0, I_d) and orthogonal within blocks of d.

    Each block is the Q factor of a Gaussian matrix. Q is multiplied by sign(diag(R)) so its
    columns are uniformly distributed (Haar); without this correction LAPACK's sign convention
    biases the directions and makes the kernel estimate biased. Row norms are then redrawn from
    the chi distribution so each row has the norm of a Gaussian vector.
    """
    blocks = []
    for _ in range(math.ceil(num_features / head_dim)):
        g = torch.randn(head_dim, head_dim, generator=generator, dtype=torch.float64)
        qmat, r = torch.linalg.qr(g)
        qmat = qmat * torch.sign(torch.diagonal(r)).unsqueeze(0)
        blocks.append(qmat.T)
    w = torch.cat(blocks, dim=0)[:num_features]
    norms = torch.randn(num_features, head_dim, generator=generator, dtype=torch.float64).norm(dim=1, keepdim=True)
    return (w * norms).to(device=device, dtype=dtype)


def softmax_kernel_features(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    is_query: bool,
    scale: float,
    stabilize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Positive random features: phi(x) = exp(w.x' - ||x'||^2 / 2) / sqrt(m), with x' = x * sqrt(scale).

    With stabilize=True a constant is subtracted inside exp() -- per row for queries, one global
    value for keys -- which cancels between numerator and denominator of the attention output.
    Use stabilize=False, eps=0 to get the raw unbiased estimator.
    """
    x = x * math.sqrt(scale)
    logits = torch.matmul(x, w.T) - (x * x).sum(dim=-1, keepdim=True) / 2
    if stabilize:
        if is_query:
            logits = logits - logits.amax(dim=-1, keepdim=True).detach()
        else:
            logits = logits - logits.amax(dim=(-2, -1), keepdim=True).detach()
    return (torch.exp(logits) + eps) / math.sqrt(w.shape[0])


def _causal_linear_attention(qp, kp, v, chunk_size):
    """Causal linear attention computed in chunks: exact within a chunk, prefix state across chunks.

    Memory is O(chunk^2 + m d) instead of the O(N m d) of a naive cumulative-sum implementation.
    """
    B, H, N, m = qp.shape
    state = torch.zeros(B, H, m, v.shape[-1], dtype=qp.dtype, device=qp.device)
    z = torch.zeros(B, H, m, dtype=qp.dtype, device=qp.device)
    nums, dens = [], []
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        qc, kc, vc = qp[:, :, s:e], kp[:, :, s:e], v[:, :, s:e]
        a = torch.tril(torch.matmul(qc, kc.transpose(-2, -1)))
        nums.append(torch.matmul(a, vc) + torch.matmul(qc, state))
        dens.append(a.sum(dim=-1) + torch.einsum("bhcm,bhm->bhc", qc, z))
        state = state + torch.matmul(kc.transpose(-2, -1), vc)
        z = z + kc.sum(dim=2)
    return torch.cat(nums, dim=2), torch.cat(dens, dim=2)


def performer_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: float | None = None,
    *,
    features: torch.Tensor | None = None,
    num_features: int | None = None,
    generator: torch.Generator | None = None,
    causal_chunk: int = 128,
    eps: float = 1e-6,
) -> torch.Tensor:
    """FAVOR+ attention. Pass a fixed `features` matrix [m, d] for reproducible benchmarks.

    If `features` is None, m = num_features (default round(d * ln d)) features are drawn.
    """
    check_inputs(q, k, v, causal)
    d = q.shape[-1]
    scale = default_scale(d, scale)
    acc = accum_dtype(q.dtype)
    if features is None:
        m = num_features or max(1, round(d * math.log(d)))
        features = orthogonal_gaussian_features(m, d, generator=generator)
    w = features.to(device=q.device, dtype=acc)

    qp = softmax_kernel_features(q.to(acc), w, is_query=True, scale=scale, eps=eps)
    kp = softmax_kernel_features(k.to(acc), w, is_query=False, scale=scale, eps=eps)
    vv = v.to(acc)

    if causal:
        num, den = _causal_linear_attention(qp, kp, vv, causal_chunk)
    else:
        kv = torch.matmul(kp.transpose(-2, -1), vv)                  # [B, H, m, dv]
        num = torch.matmul(qp, kv)                                   # [B, H, N, dv]
        den = torch.einsum("bhnm,bhm->bhn", qp, kp.sum(dim=2))      # [B, H, N]
    return (num / den.unsqueeze(-1)).to(q.dtype)
