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
# Preferred storage layout per method. "bhnd" = contiguous [B, H, N, d] (default);
# "bnhd" = contiguous [B, N, H, d], passed to the method as a [B, H, N, d] view. Inputs are created
# in the preferred layout before timing, so no method pays for a layout copy inside the timed call.
_LAYOUT: dict[str, str] = {}


def register(name: str, builder: Builder, supports_causal: bool = True, layout: str = "bhnd") -> None:
    _REGISTRY[name] = (builder, supports_causal)
    _LAYOUT[name] = layout


def make_qkv(name: str, shape, dtype, device, generator=None, requires_grad: bool = False):
    """Return (q, k, v, leaves): [B, H, N, d] tensors in `name`'s preferred layout, and the leaf
    tensors whose .grad must be cleared between repeats (the same tensors for "bhnd")."""
    B, H, N, d = shape
    if _LAYOUT.get(name, "bhnd") == "bnhd":
        leaves = [torch.randn((B, N, H, d), device=device, dtype=dtype, generator=generator) for _ in range(3)]
        for t in leaves:
            t.requires_grad_(requires_grad)
        q, k, v = (t.transpose(1, 2) for t in leaves)
        return q, k, v, leaves
    leaves = [torch.randn(shape, device=device, dtype=dtype, generator=generator) for _ in range(3)]
    for t in leaves:
        t.requires_grad_(requires_grad)
    return (*leaves, leaves)


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
    import flash_attn

    # FA1 and FA2 share the import name; refuse to time the wrong one.
    if not flash_attn.__version__.startswith("2."):
        raise ImportError(f"flash_attn {flash_attn.__version__} installed; 'flash_attn' needs FA2")
    from flash_attn import flash_attn_func  # raises ImportError if not installed

    def fn(q, k, v, causal=False, scale=None):
        out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal,
                              softmax_scale=scale)
        return out.transpose(1, 2)

    return fn


register("flash_attn", _flash_attn_builder)


# Official FlashAttention-1 (flash-attn 1.0.9, CUDA). Lives in its own venv because
# it shares the `flash_attn` import name with FA2. FA1 only has the packed "unpadded" API on
# [total_tokens, H, d]; for batch 1 our [1, H, N, d] tensor is passed as a strided view (FA1 takes
# arbitrary row and head strides), so no copy is timed. Batch > 1 is refused rather than copied.
def _flash_attn_v1_builder(seq_len: int, head_dim: int, device, seed: int):
    import flash_attn

    if not flash_attn.__version__.startswith("1."):
        raise ImportError(f"flash_attn {flash_attn.__version__} installed; 'flash_attn_v1' needs FA1")
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func

    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    def fn(q, k, v, causal=False, scale=None):
        if q.shape[0] != 1:
            raise NotImplementedError("flash_attn_v1 wrapper supports batch 1 only (avoids a timed copy)")
        n = q.shape[2]
        # inputs arrive as views of contiguous [1, N, H, d] storage (layout "bnhd"), so these are
        # contiguous [N, H, d] tensors; FA1's backward requires contiguous q, k, v and out
        q3, k3, v3 = (t.transpose(1, 2)[0] for t in (q, k, v))
        if not (q3.is_contiguous() and k3.is_contiguous() and v3.is_contiguous()):
            raise NotImplementedError("flash_attn_v1 needs [B, N, H, d]-contiguous inputs (layout bnhd)")
        out = flash_attn_unpadded_func(q3, k3, v3, cu, cu, n, n, 0.0, softmax_scale=scale, causal=causal)
        return out.unsqueeze(0).transpose(1, 2)

    return fn


register("flash_attn_v1", _flash_attn_v1_builder, layout="bnhd")


def _tutorial_builder(seq_len: int, head_dim: int, device, seed: int):
    from bench.reference import tutorial_fa2

    return tutorial_fa2


# Official Triton fused-attention tutorial (FA2 algorithm); FP16 only, causal-only backward.
register("tutorial_fa2", _tutorial_builder)


def _register_triton() -> None:
    """Triton kernels (bench/triton_attn). Skipped silently where Triton is not importable."""
    try:
        from bench import triton_attn
    except ImportError:
        return
    for name, fn in triton_attn.METHODS.items():
        register(name, lambda seq_len, head_dim, device, seed, _fn=fn: _fn, True)


_register_triton()

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
