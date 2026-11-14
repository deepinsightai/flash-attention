"""GPU timing and peak-memory measurement for one attention call.

Each repeat is timed with CUDA events after a synchronize, so host launch overhead and
other streams do not leak in. Peak memory is the extra memory beyond the live inputs
(q, k, v and the output gradient), measured per repeat with reset_peak_memory_stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class Measurement:
    runtime_ms: float
    peak_memory_mb: float


def _call(fn: Callable, q, k, v, causal: bool, grad_out, backward: bool):
    if backward:
        out = fn(q, k, v, causal=causal)
        out.backward(grad_out)
        q.grad = k.grad = v.grad = None
        return out
    with torch.no_grad():
        return fn(q, k, v, causal=causal)


def measure(
    fn: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    backward: bool,
    warmup: int,
    repeats: int,
) -> list[Measurement]:
    """Run warm-up iterations, then `repeats` timed iterations. Raises torch.OutOfMemoryError on OOM."""
    grad_out = torch.randn_like(q) if backward else None
    for _ in range(warmup):
        _call(fn, q, k, v, causal, grad_out, backward)
    results: list[Measurement] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = _call(fn, q, k, v, causal, grad_out, backward)
        end.record()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        results.append(Measurement(start.elapsed_time(end), peak / 2**20))
        del out
    return results
