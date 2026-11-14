"""Peak GPU memory vs sequence length for each baseline -- a sanity check, not the final benchmark.

Run from the repository root on a GPU (a free Colab/Kaggle T4 is enough):
    python -m scripts.memory_sanity
    python -m scripts.memory_sanity --dtype bf16 --max-log2n 16 --backward

Expected shapes on a log-log plot: naive grows with slope ~2 (N^2), the others much
more slowly. If not, something is materializing an N x N tensor it should not.
Writes results/memory_sanity.csv and results/memory_sanity.png.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from attention_baselines import METHODS, get_attention

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--min-log2n", type=int, default=9)
    p.add_argument("--max-log2n", type=int, default=15)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--dtype", choices=DTYPES, default="fp16")
    p.add_argument("--causal", action="store_true")
    p.add_argument("--backward", action="store_true", help="measure forward+backward instead of forward only")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def peak_memory_mib(fn, q, k, v, causal, backward):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    if backward:
        out = fn(q, k, v, causal=causal)
        out.sum().backward()
    else:
        with torch.no_grad():
            out = fn(q, k, v, causal=causal)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del out
    for t in (q, k, v):
        t.grad = None
    return peak / 2**20


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU (use Colab or Kaggle)")
    dtype = DTYPES[args.dtype]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.methods:
        if args.causal and name == "linformer":
            print("skipping linformer: no causal support")
            continue
        for log2n in range(args.min_log2n, args.max_log2n + 1):
            n = 2 ** log2n
            q, k, v = (torch.randn(1, args.heads, n, args.d, device="cuda", dtype=dtype,
                                   requires_grad=args.backward) for _ in range(3))
            fn = get_attention(name, seq_len=n, head_dim=args.d, device="cuda", seed=0)
            try:
                mib, status = peak_memory_mib(fn, q, k, v, args.causal, args.backward), "ok"
            except torch.cuda.OutOfMemoryError:
                mib, status = float("nan"), "oom"
            del q, k, v
            torch.cuda.empty_cache()
            rows.append(dict(method=name, n=n, peak_mib=mib, status=status, dtype=args.dtype,
                             causal=args.causal, backward=args.backward, gpu=torch.cuda.get_device_name()))
            print(f"{name:12s} N={n:6d}  {status:3s}  {mib:10.1f} MiB")
            if status == "oom":
                break  # larger N will also OOM

    csv_path = out_dir / "memory_sanity.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name in args.methods:
        pts = [(r["n"], r["peak_mib"]) for r in rows if r["method"] == name and r["status"] == "ok"]
        if pts:
            ax.plot(*zip(*pts), marker="o", label=name)
    ax.set(xscale="log", yscale="log", xlabel="sequence length N", ylabel="peak extra memory (MiB)",
           title=f"Peak memory, {'fwd+bwd' if args.backward else 'forward'}, {args.dtype}, "
                 f"{'causal' if args.causal else 'non-causal'}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "memory_sanity.png", dpi=150)
    print(f"wrote {out_dir / 'memory_sanity.png'}")


if __name__ == "__main__":
    main()
