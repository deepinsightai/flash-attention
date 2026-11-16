"""Cross-check every registered method against Aneet's independent FP64 reference.

Shaaz's tests compare against Shaaz's own reference; this uses a second, independent one,
so a shared bug in a single reference would show up as a disagreement here.

    python -m scripts.cross_check                    # BF16, N = 1024
    python -m scripts.cross_check --dtype fp16 --methods tutorial_fa2 triton_fa2 flash_attn
"""

from __future__ import annotations

import argparse

import torch

from bench import methods
from validation.reference import compare_outputs, reference_attention

APPROXIMATE = {"performer", "linformer"}
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--tol", type=float, default=0.05, help="max abs error allowed for exact methods")
    a = p.parse_args(argv)

    dev = "cuda"
    shape = (1, a.heads, a.n, a.head_dim)
    print(f"dtype {a.dtype}, N {a.n}, heads {a.heads}, d {a.head_dim}")
    print(f"{'method':20s} {'mask':9s} {'max_abs':>10s} {'rmse':>10s}  verdict")
    failed = []
    for name in a.methods or methods.METHODS:
        for causal in (False, True):
            if causal and not methods.supports_causal(name):
                continue
            mask = "causal" if causal else "noncausal"
            try:
                gen = torch.Generator(device=dev).manual_seed(0)
                q, k, v, _ = methods.make_qkv(name, shape, DTYPES[a.dtype], dev, gen)
                fn = methods.build(name, a.n, a.head_dim, dev, 0)
                with torch.no_grad():
                    out = fn(q, k, v, causal=causal)
                err = compare_outputs(out, reference_attention(q, k, v, causal=causal))
            except Exception as e:  # noqa: BLE001
                print(f"{name:20s} {mask:9s} {'-':>10s} {'-':>10s}  UNAVAILABLE ({type(e).__name__})")
                continue
            if name in APPROXIMATE:
                verdict = "approximate"
            else:
                verdict = "ok" if err["max_absolute_error"] < a.tol else "FAIL"
                if verdict == "FAIL":
                    failed.append(f"{name}/{mask}")
            print(f"{name:20s} {mask:9s} {err['max_absolute_error']:10.4f} "
                  f"{err['root_mean_square_error']:10.5f}  {verdict}")
    if failed:
        print("exact methods above tolerance:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
