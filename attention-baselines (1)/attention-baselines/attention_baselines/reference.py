"""FP64 reference attention and error metrics, used by the tests and the benchmark."""

from __future__ import annotations

import torch

from .naive import naive_attention


def reference_attention(q, k, v, causal: bool = False, scale: float | None = None) -> torch.Tensor:
    """Exact attention computed in FP64 on the inputs' device.

    Note: this materializes an N x N FP64 matrix (8 bytes per entry), so it limits N
    for error measurements (N = 16K needs 2 GB per head). Measure error at moderate N.
    """
    return naive_attention(q.double(), k.double(), v.double(), causal=causal, scale=scale)


def attention_error(out: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """Error metrics of `out` against `ref`, computed in FP64.

    rel_fro  = ||out - ref||_F / ||ref||_F   (main metric to report)
    max_abs  = max |out - ref|
    mean_abs = mean |out - ref|
    """
    diff = out.double() - ref.double()
    return {
        "rel_fro": (diff.norm() / ref.double().norm()).item(),
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
    }
