"""Approximation error of Performer (vs m) and Linformer (vs k) against FP64 exact attention.

Run from the repository root:
    python -m scripts.sweep_approx_error                      # CPU is fine
    python -m scripts.sweep_approx_error --device cuda --n 2048

Writes results/approx_error.csv and results/approx_error.png.

`--qk-scales` sets the standard deviation of the entries of q and k. Random-feature
methods are very sensitive to it (Performer's variance grows like exp(||q+k||^2/sqrt(d))),
so always report it with the numbers.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import torch

from attention_baselines import (
    attention_error,
    linformer_attention,
    linformer_projections,
    orthogonal_gaussian_features,
    performer_attention,
    reference_attention,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=1024, help="sequence length")
    p.add_argument("--d", type=int, default=64, help="head dimension")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--qk-scales", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--features", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024])
    p.add_argument("--proj-dims", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev, dtype = args.device, torch.float32
    rows = []

    for qk_scale in args.qk_scales:
        for seed in range(args.seeds):
            g = torch.Generator().manual_seed(seed)
            shape = (1, args.heads, args.n, args.d)
            q = (torch.randn(shape, generator=g) * qk_scale).to(dev, dtype)
            k = (torch.randn(shape, generator=g) * qk_scale).to(dev, dtype)
            v = torch.randn(shape, generator=g).to(dev, dtype)
            ref = reference_attention(q, k, v)
            fg = torch.Generator().manual_seed(10_000 + seed)

            for m in args.features:
                w = orthogonal_gaussian_features(m, args.d, generator=fg, device=dev)
                err = attention_error(performer_attention(q, k, v, features=w), ref)
                rows.append(dict(method="performer", variant="favor+", param=m, qk_scale=qk_scale, seed=seed, **err))

            for kind in ("gaussian", "pooling"):
                for r in (p for p in args.proj_dims if p <= args.n):
                    e = linformer_projections(args.n, r, kind=kind, generator=fg, device=dev)
                    err = attention_error(linformer_attention(q, k, v, proj_k=e), ref)
                    rows.append(dict(method="linformer", variant=kind, param=r, qk_scale=qk_scale, seed=seed, **err))
        print(f"qk_scale={qk_scale}: done")

    csv_path = out_dir / "approx_error.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")
    plot(rows, args, out_dir / "approx_error.png")


def plot(rows, args, path):
    import matplotlib.pyplot as plt

    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["method"], r["variant"], r["qk_scale"])][r["param"]].append(r["rel_fro"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    panels = [("performer", "favor+", "Performer (FAVOR+)", "random features m"),
              ("linformer", "gaussian", "Linformer, Gaussian projection", "projected length k"),
              ("linformer", "pooling", "Linformer, mean-pooling projection", "projected length k")]
    for ax, (method, variant, title, xlabel) in zip(axes, panels):
        for qk_scale in args.qk_scales:
            by_param = groups[(method, variant, qk_scale)]
            xs = sorted(by_param)
            mean = [sum(by_param[x]) / len(by_param[x]) for x in xs]
            lo = [min(by_param[x]) for x in xs]
            hi = [max(by_param[x]) for x in xs]
            line, = ax.plot(xs, mean, marker="o", label=f"q,k std = {qk_scale}")
            ax.fill_between(xs, lo, hi, color=line.get_color(), alpha=0.15)
        if method == "performer":
            xs = sorted(groups[(method, variant, args.qk_scales[0])])
            y0 = sum(groups[(method, variant, args.qk_scales[0])][xs[0]]) / args.seeds
            ax.plot(xs, [y0 * math.sqrt(xs[0] / x) for x in xs], "k--", lw=1, label=r"$\propto 1/\sqrt{m}$")
        ax.set(xscale="log", yscale="log", title=title, xlabel=xlabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"relative error $\|O - O_{ref}\|_F / \|O_{ref}\|_F$")
    fig.suptitle(f"Approximation error vs FP64 exact attention (N={args.n}, d={args.d}, "
                 f"mean and min-max over {args.seeds} seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
