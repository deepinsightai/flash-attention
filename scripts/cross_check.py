"""Cross-check every registered method against Aneet's independent FP64 reference.

Shaaz's tests compare against Shaaz's own reference; this uses a second, independent one,
so a shared bug in a single reference would show up as a disagreement here.

    python -m scripts.cross_check
"""

from __future__ import annotations

import torch

from bench import methods
from validation.reference import compare_outputs, reference_attention

EXACT = {"naive", "rabe_staats", "sdpa_math", "sdpa_efficient", "sdpa_flash", "flash_attn"}


def main():
    dev = "cuda"
    N, B, H, D = 1024, 1, 8, 128
    gen = torch.Generator(device=dev).manual_seed(0)
    q, k, v = (torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16, generator=gen) for _ in range(3))
    print(f"{'method':16s} {'mask':9s} {'max_abs':>10s} {'rmse':>10s}  verdict")
    for name in methods.METHODS:
        for causal in (False, True):
            if causal and not methods.supports_causal(name):
                continue
            try:
                fn = methods.build(name, N, D, dev, 0)
                with torch.no_grad():
                    out = fn(q, k, v, causal=causal)
                err = compare_outputs(out, reference_attention(q, k, v, causal=causal))
            except Exception as e:  # noqa: BLE001
                print(f"{name:16s} {'causal' if causal else 'noncausal':9s} {'-':>10s} {'-':>10s}  UNAVAILABLE ({type(e).__name__})")
                continue
            # Exact methods should sit at BF16 rounding level (~1e-2 abs for unit-normal inputs).
            verdict = ("ok" if err["max_absolute_error"] < 0.05 else "CHECK") if name in EXACT else "approximate"
            print(f"{name:16s} {'causal' if causal else 'noncausal':9s} {err['max_absolute_error']:10.4f} "
                  f"{err['root_mean_square_error']:10.5f}  {verdict}")


if __name__ == "__main__":
    main()
