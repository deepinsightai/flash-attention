"""Independent attention reference and numerical validation.

# AI-assisted (ChatGPT): initial implementation; review and test before submission.
"""
import math
import torch


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """FP64 scaled dot-product attention for small validation inputs.

    Input tensors have shape (..., N, d) for q/k and (..., N, dv) for v.
    Intended for self-attention, with matching sequence lengths.
    """
    if q.ndim < 2 or q.shape[:-1] != k.shape[:-1] or q.shape[:-1] != v.shape[:-1]:
        raise ValueError('Expected matching batch/head and sequence dimensions')
    if q.shape[-1] != k.shape[-1]:
        raise ValueError('Query/key head dimensions differ')
    q64, k64, v64 = q.to(torch.float64), k.to(torch.float64), v.to(torch.float64)
    scores = (q64 @ k64.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        n = q.shape[-2]
        future = torch.ones((n, n), device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(future, -torch.inf)
    return torch.softmax(scores, dim=-1) @ v64


def compare_outputs(actual: torch.Tensor, reference: torch.Tensor, epsilon: float = 1e-8) -> dict:
    """Return descriptive numerical errors; no universal pass threshold is assumed."""
    if actual.shape != reference.shape:
        raise ValueError(f'Shape mismatch: {tuple(actual.shape)} vs {tuple(reference.shape)}')
    a = actual.to(torch.float64)
    r = reference.to(torch.float64)
    if not torch.isfinite(a).all() or not torch.isfinite(r).all():
        raise ValueError('Non-finite values found in outputs')
    error = (a-r).abs()
    relative = error / r.abs().clamp_min(epsilon)
    return {
        'max_absolute_error': error.max().item(),
        'mean_absolute_error': error.mean().item(),
        'root_mean_square_error': error.square().mean().sqrt().item(),
        'max_relative_error': relative.max().item(),
    }
