"""Method registry for the harness.

Every method is built by `build(name, seq_len, head_dim, device, seed)` and returns
fn(q, k, v, causal=False) -> out on [B, H, N, d] tensors (the shared interface).

Shaaz's baselines come from the attention_baselines package. Reference arms (PyTorch SDPA,
official flash-attn) are wrapped here. New methods, such as the Triton kernels, plug in with
`register(name, builder, supports_causal)` and need no change to the sweep driver.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

_BASELINES = Path(__file__).resolve().parents[1] / "attention-baselines (1)" / "attention-baselines"
if _BASELINES.exists() and str(_BASELINES) not in sys.path:
    sys.path.insert(0, str(_BASELINES))

from attention_baselines import SUPPORTS_CAUSAL as _BASE_CAUSAL  # noqa: E402
from attention_baselines import get_attention as _get_baseline  # noqa: E402

Builder = Callable[..., Callable]
_REGISTRY: dict[str, tuple[Builder, bool]] = {}


def register(name: str, builder: Builder, supports_causal: bool = True) -> None:
    _REGISTRY[name] = (builder, supports_causal)


def _baseline_builder(name: str) -> Builder:
    def build(seq_len: int, head_dim: int, device, seed: int):
        return _get_baseline(name, seq_len=seq_len, head_dim=head_dim, device=device, seed=seed)

    return build


for _name, _causal in _BASE_CAUSAL.items():
    register(_name, _baseline_builder(_name), _causal)


def _sdpa_builder(backend_name: str) -> Builder:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backend = getattr(SDPBackend, backend_name)

    def build(seq_len: int, head_dim: int, device, seed: int):
        def fn(q, k, v, causal=False, scale=None):
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)

        return fn

    return build


register("sdpa_flash", _sdpa_builder("FLASH_ATTENTION"))
register("sdpa_efficient", _sdpa_builder("EFFICIENT_ATTENTION"))
register("sdpa_math", _sdpa_builder("MATH"))


def _flash_attn_builder(seq_len: int, head_dim: int, device, seed: int):
    """Official flash-attn package (FA2 on Ampere/Ada/Hopper). Expects [B, N, H, d]."""
    from flash_attn import flash_attn_func  # raises ImportError if not installed

    def fn(q, k, v, causal=False, scale=None):
        out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal,
                              softmax_scale=scale)
        return out.transpose(1, 2)

    return fn


register("flash_attn", _flash_attn_builder)

METHODS = tuple(_REGISTRY)


def supports_causal(name: str) -> bool:
    return _REGISTRY[name][1]


def build(name: str, seq_len: int, head_dim: int, device, seed: int = 0) -> Callable:
    if name not in _REGISTRY:
        raise ValueError(f"unknown method {name!r}; choose from {tuple(_REGISTRY)}")
    return _REGISTRY[name][0](seq_len, head_dim, device, seed)


def reference_error(out: torch.Tensor, q, k, v, causal: bool) -> dict:
    """Error against the FP64 reference (Shaaz's reference_attention). Keys: rel_fro, max_abs, mean_abs."""
    from attention_baselines import attention_error, reference_attention

    return attention_error(out, reference_attention(q, k, v, causal=causal))
