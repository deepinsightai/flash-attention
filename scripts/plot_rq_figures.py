"""One figure per research question (RQ1-RQ4, plus RQ6 partial) from the A100 session of 2026-11-14.

    python -m scripts.plot_rq_figures [DATA_ROOT] [OUT_DIR]

DATA_ROOT defaults to the unpacked full mirror of the instance (a100-session-2026-11-14/final-mirror/bench).
Which data each figure uses (only current, non-superseded results):
    primary                    results_primary              (8 heads, d=128, BF16, batch 1)
      Performer causal fwd+bwd results_primary_performer_split   (replaces the pre-fix rows)
    RQ2 timing                 results_rq2_v2 (tuned, new backward), results_rq2_newbwd (FA1/FA2 fwd+bwd),
                               results_rq2_fa1venv (official FA1 forward)          (128 heads, d=128, BF16)
    RQ2 traffic (Nsight)       results_ncu/ncu_summary.csv  (128 heads), closeout/ncu/ncu_summary_N*.txt (8 heads, causal)
Superseded data (results_rq2, results_ncu_v1_gradaccum) is not used.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_S = Path(r"C:\Users\jfaki\Desktop\bcit-comp-9060\a100-session-2026-11-14")
# final/ has the last runs (performer_split, rq2_newbwd, closeout); final-mirror/ has results_rq2_fa1venv,
# which the final bundle's --exclude='*venv*' skipped. Each results folder is read from the first root that has it.
_REPO_DATA = Path(__file__).resolve().parents[1] / "data" / "a100-2026-11-14"
# The repo holds the primary-study CSVs (data/a100-2026-11-14); the RQ2 and Nsight data exist only in the
# offline session archive. The RQ2 figure is skipped when its data is not found.
ROOTS = [Path(sys.argv[1])] if len(sys.argv) > 1 else [_REPO_DATA, _S / "final" / "bench", _S / "final-mirror" / "bench"]


def root_for(d):
    for r in ROOTS:
        if (r / d).exists():
            return r
    raise FileNotFoundError(d)
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "figures" / "a100-2026-11-14"
OUT.mkdir(parents=True, exist_ok=True)
GPU = "NVIDIA A100-SXM4-40GB, PyTorch 2.8, flash-attn 2.8.3, Triton 3.4"
PRIMARY_CFG = f"Primary study: batch 1, 8 heads, d=128, BF16, 10 repeats (median, IQR band). {GPU}."
RQ2_CFG = f"RQ2: batch 1, 128 heads, d=128, BF16, 10 repeats. {GPU}."

NAME = {"naive": "PyTorch naive", "sdpa_math": "SDPA math", "rabe_staats": "Rabe-Staats (PyTorch)",
        "sdpa_efficient": "SDPA mem-efficient", "sdpa_flash": "SDPA flash", "flash_attn": "FlashAttention-2 (official)",
        "performer": "Performer", "linformer": "Linformer", "triton_naive": "Triton naive",
        "triton_rabe_staats": "Triton Rabe-Staats", "triton_fa1": "Triton FA1", "triton_fa2": "Triton FA2",
        "flash_attn_v1": "FlashAttention-1 (official)"}
COLOR = {"naive": "#1f77b4", "sdpa_math": "#7f7f7f", "rabe_staats": "#9467bd", "sdpa_efficient": "#17becf",
         "sdpa_flash": "#2ca02c", "flash_attn": "black", "performer": "#ff7f0e", "linformer": "#8c564b",
         "triton_naive": "#1f77b4", "triton_rabe_staats": "#9467bd", "triton_fa1": "#d62728",
         "triton_fa2": "#2ca02c", "flash_attn_v1": "#e377c2"}
STYLE = {"triton_naive": "--", "triton_rabe_staats": "--", "triton_fa1": "--", "triton_fa2": "--"}


# ------------------------------------------------------------------------------------------------ data
def runs(*dirs):
    frames = []
    for d in dirs:
        for f in (root_for(d) / d).glob("*/runs.csv"):
            frames.append(pd.read_csv(f))
    return pd.concat(frames, ignore_index=True)


def ooms(*dirs):
    frames = [pd.read_csv(f) for d in dirs for f in (root_for(d) / d).glob("*/oom.csv")]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["method", "mask", "pass_type", "N"])


def med(r):
    g = r.groupby(["method", "precision", "mask", "pass_type", "N"])
    m = g.agg(ms=("runtime_ms", "median"), q1=("runtime_ms", lambda x: x.quantile(.25)),
              q3=("runtime_ms", lambda x: x.quantile(.75)), mem=("peak_memory_mb", "median"),
              err=("numerical_error", "first"), heads=("heads", "first")).reset_index()
    return m


prim_raw = runs("results_primary")
perf = runs("results_primary_performer_split")
key = (prim_raw.method == "performer") & (prim_raw["mask"] == "causal") & (prim_raw.pass_type == "forward_backward")
prim = med(pd.concat([prim_raw[~key], perf], ignore_index=True))
prim_oom = ooms("results_primary")

RQ2_DIRS = ("results_rq2_v2", "results_rq2_newbwd", "results_rq2_fa1venv", "results_ncu", "closeout")
HAVE_RQ2 = all(any((r / d).exists() for r in ROOTS) for d in RQ2_DIRS)
if HAVE_RQ2:
    rq2_v2 = runs("results_rq2_v2")
    newbwd = runs("results_rq2_newbwd")
    fa1 = runs("results_rq2_fa1venv")
    rq2_raw = pd.concat([
        rq2_v2[(rq2_v2.precision == "BF16") & ~((rq2_v2.pass_type == "forward_backward")
                                                 & rq2_v2.method.isin(["triton_fa2", "triton_fa1", "flash_attn"]))],
        newbwd, fa1[fa1.pass_type == "forward"]], ignore_index=True)
    rq2 = med(rq2_raw)
    ncu = pd.read_csv(root_for("results_ncu") / "results_ncu" / "ncu_summary.csv")

    close = []
    for f in sorted((root_for("closeout") / "closeout" / "ncu").glob("ncu_summary_N*.txt")):
        for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"TOTAL (\S+) (fwd|bwd) N=(\d+)\s+read_MB ([\d.]+)\s+write_MB ([\d.]+)\s+gpu_time_us ([\d.]+)", ln)
            if m:
                close.append({"method": m[1], "pass": m[2], "N": int(m[3]), "MB": float(m[4]) + float(m[5]),
                              "us": float(m[6])})
    close = pd.DataFrame(close)


def sel(m, method, mask, pas, prec="BF16"):
    s = m[(m.method == method) & (m["mask"] == mask) & (m.pass_type == pas) & (m.precision == prec)]
    return s.sort_values("N")


def xaxis(ax, ns):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n // 1024}K" if n >= 1024 else str(n) for n in ns], fontsize=8)
    ax.set_xlabel("Sequence length N")
    ax.grid(True, which="both", alpha=0.25)


def line(ax, m, method, mask, pas, value="ms", band=True, label=None, **kw):
    s = sel(m, method, mask, pas)
    if s.empty:
        return s
    ax.plot(s.N, s[value], STYLE.get(method, "-"), color=COLOR[method], marker="o", ms=3.5,
            lw=2.4 if method == "flash_attn" else 1.6, label=label or NAME[method], **kw)
    if band and value == "ms":
        ax.fill_between(s.N, s.q1, s.q3, color=COLOR[method], alpha=0.18, lw=0)
    return s


_OOM_SLOT: dict = {}


def mark_oom(ax, oom, method, mask, pas, y):
    """Mark the first out-of-memory N with an x; stack markers of different methods vertically."""
    o = oom[(oom.method == method) & (oom["mask"] == mask) & (oom.pass_type == pas)]
    if len(o):
        n = int(o.N.min())
        k = _OOM_SLOT.setdefault((id(ax), n), [0])
        y2 = y / (2.2 ** k[0])
        k[0] += 1
        ax.scatter([n], [y2], marker="x", s=70, color=COLOR[method], zorder=5)
        ax.annotate(f"{NAME[method]}: OOM", (n, y2), textcoords="offset points", xytext=(6, -3), fontsize=7,
                    color=COLOR[method])


def save(fig, name, cfg):
    fig.text(0.5, 0.005, cfg, ha="center", fontsize=7.5, color="#444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT / name, dpi=170)
    plt.close(fig)
    print("wrote", OUT / name)


NS_P = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
NS_R = [1024, 2048, 4096, 8192, 16384]

# ------------------------------------------------------------------------------------------------ RQ1
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
fig.suptitle("RQ1: Is FlashAttention faster, with higher throughput and O(N) rather than O(N\u00b2) memory, than the baselines?",
             fontsize=12, weight="bold")
exact = ["naive", "sdpa_math", "rabe_staats", "sdpa_efficient", "sdpa_flash", "flash_attn"]
ax = axes[0]
for mth in exact:
    line(ax, prim, mth, "noncausal", "forward_backward")
    mark_oom(ax, prim_oom, mth, "noncausal", "forward_backward", 3e4)
xaxis(ax, NS_P); ax.set_yscale("log"); ax.set_ylabel("Runtime, forward + backward (ms)")
ax.set_title("(a) Runtime: the FA2 kernels (official, SDPA flash) are fastest from N \u2265 4K;\n"
             "naive and SDPA-math run out of memory at 32K", fontsize=9.5)
ax.legend(fontsize=7.5, loc="upper left")

ax = axes[1]
for mth in exact:
    line(ax, prim, mth, "noncausal", "forward_backward", value="mem")
    mark_oom(ax, prim_oom, mth, "noncausal", "forward_backward", 3e4)
n = np.array(NS_P, float)
fa = sel(prim, "flash_attn", "noncausal", "forward_backward").set_index("N").mem
ax.plot(n, fa[512] * n / 512, ":", color="gray", lw=1)
ax.plot(n[:5], sel(prim, "naive", "noncausal", "forward_backward").set_index("N").mem[512] * (n[:5] / 512) ** 2,
        ":", color="gray", lw=1)
ax.text(700, fa[512] * 700 / 512 * 0.45, "slope \u221d N", fontsize=8.5, color="gray")
nv512 = sel(prim, "naive", "noncausal", "forward_backward").set_index("N").mem[512]
ax.text(1500, nv512 * (1500 / 512) ** 2 * 2.2, "slope \u221d N\u00b2", fontsize=8.5, color="gray")
xaxis(ax, NS_P); ax.set_yscale("log"); ax.set_ylabel("Peak extra GPU memory (MB)")
r16 = sel(prim, "naive", "noncausal", "forward_backward").set_index("N").mem[16384] / fa[16384]
ax.set_title(f"(b) Memory: FA2 linear, naive quadratic ({r16:.0f}\u00d7 less at N=16K)", fontsize=9.5)

ax = axes[2]
tf = []
for mth in exact:
    s = sel(prim, mth, "noncausal", "forward_backward")
    tf.append((mth, s.N.max()))
for mask, off in (("noncausal", -0.2), ("causal", 0.2)):
    vals = [np.log2(sel(prim, mth, mask, "forward_backward").N.max()) for mth in exact]
    ax.barh(np.arange(len(exact)) + off, vals, height=0.38, color=[COLOR[m] for m in exact],
            alpha=1.0 if mask == "noncausal" else 0.55, label=mask)
ax.set_yticks(range(len(exact))); ax.set_yticklabels([NAME[m] for m in exact], fontsize=8)
ax.set_xlim(8, 16.8)
ax.set_xticks(range(9, 17)); ax.set_xticklabels([f"{2 ** k // 1024}K" if 2 ** k >= 1024 else str(2 ** k) for k in range(9, 17)])
ax.set_xlabel("Largest N completed, forward + backward (sweep stops at 64K)")
ax.set_title("(c) Largest N that fits on a 40 GB A100\n(dark: non-causal, light: causal)", fontsize=9.5)
ax.grid(True, axis="x", alpha=0.25)
save(fig, "rq1_speed_memory.png", PRIMARY_CFG + " Non-causal unless stated.")

# ------------------------------------------------------------------------------------------------ RQ2
if not HAVE_RQ2:
    print("skipping RQ2 figure: RQ2/Nsight data not found in", [str(r) for r in ROOTS])
else:
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    fig.suptitle("RQ2: How much of FlashAttention's advantage is IO-awareness, isolated from engineering? "
                 "(same Triton toolchain, same tuning search)", fontsize=12, weight="bold")
    tri = ["triton_naive", "triton_rabe_staats", "triton_fa2", "flash_attn"]
    ax = axes[0, 0]
    for mth in tri:
        line(ax, rq2, mth, "noncausal", "forward_backward")
    o2 = ooms("results_rq2_v2")
    mark_oom(ax, o2, "triton_naive", "noncausal", "forward_backward", 2e3)
    xaxis(ax, NS_R); ax.set_yscale("log"); ax.set_ylabel("Runtime, forward + backward (ms)")
    ax.set_title("(a) Same toolchain: Triton FA2 beats Triton naive and Rabe-Staats", fontsize=9.5)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for mth in ["triton_naive", "triton_rabe_staats", "triton_fa1", "triton_fa2", "flash_attn"]:
        s = close[(close.method == mth) & (close["pass"] == "bwd")].sort_values("N")
        ax.plot(s.N, s.MB / 1e3, STYLE.get(mth, "-"), color=COLOR[mth], marker="o", ms=3.5, label=NAME[mth])
    xaxis(ax, NS_R); ax.set_yscale("log"); ax.set_ylabel("Measured HBM traffic per call (GB, read + write)")
    ax.set_title("(b) Nsight: naive / Rabe-Staats traffic grows \u221d N\u00b2; FA2 grows ~linearly here because\n"
                 "K/V re-reads hit the 40 MB L2 (8 heads, causal, forward + backward)", fontsize=9.5)
    ax.legend(fontsize=7.5)

    ax = axes[0, 2]
    n8 = ncu[(ncu.N == 8192) & (ncu["mask"] == "noncausal") & (ncu["pass"] == "fwd")].set_index("method")
    order = ["triton_naive", "triton_rabe_staats", "flash_attn_v1", "triton_fa1", "triton_fa2", "flash_attn"]
    order = [m for m in order if m in n8.index]
    gb = [n8.loc[m, "dram_total_MB"] / 1e3 for m in order]
    bars = ax.bar(range(len(order)), gb, color=[COLOR[m] for m in order])
    for b, m in zip(bars, order):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.15, f"{n8.loc[m, 'dram_total_MB'] / 1e3:.1f} GB\n"
                f"{n8.loc[m, 'kernel_time_ms']:.0f} ms", ha="center", fontsize=7.5)
    ax.set_yscale("log"); ax.set_ylim(0.3, 800)
    SHORT = {"triton_naive": "Triton\nnaive", "triton_rabe_staats": "Triton\nRabe-Staats", "flash_attn_v1": "FA1\n(official)",
             "triton_fa1": "Triton\nFA1", "triton_fa2": "Triton\nFA2", "flash_attn": "FA2\n(official)"}
    ax.set_xticks(range(len(order))); ax.set_xticklabels([SHORT[m] for m in order], fontsize=8)
    ax.set_ylabel("HBM traffic (GB)")
    ax.set_title("(c) N=8K, forward, 128 heads: FA2 moves 65\u00d7 less than naive", fontsize=9.5)
    ax.grid(True, axis="y", which="both", alpha=0.25)

    ax = axes[1, 0]
    for pas, mk in (("fwd", "o"), ("fwd_bwd", "s")):
        s = ncu[(ncu.N == 8192) & (ncu["mask"] == "noncausal") & (ncu["pass"] == pas)]
        for _, r in s.iterrows():
            ax.scatter(r.dram_bw_pct_of_peak, r.sm_throughput_pct, s=80, marker=mk, color=COLOR[r.method],
                       edgecolor="k", lw=0.5)
            if pas == "fwd":  # one label per method; the square of the same colour is its fwd+bwd point
                ax.annotate(SHORT[r.method].replace("\n", " "), (r.dram_bw_pct_of_peak, r.sm_throughput_pct),
                            fontsize=7, xytext={"flash_attn_v1": (-14, -17), "triton_rabe_staats": (7, 4)}.get(r.method, (6, 6)),
                            textcoords="offset points")
    ax.axvspan(50, 100, color="#d62728", alpha=0.06); ax.text(52, 5, "memory-bound region", fontsize=8, color="#a33")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("HBM bandwidth used (% of A100 peak)"); ax.set_ylabel("SM compute throughput (% of peak)")
    ax.set_title("(d) Why: baselines are bandwidth-bound; FA2 uses ~3% of bandwidth and is compute-bound\n"
                 "(Nsight, N=8K, non-causal; circle fwd, square fwd+bwd)", fontsize=9.5)
    ax.grid(True, alpha=0.25)

    # (e) two side-by-side panels (forward | forward+backward) inside the slot of axes[1, 1]
    # Redrawn so each line says who is faster than whom.
    slot = axes[1, 1].get_subplotspec()
    axes[1, 1].remove()
    sub = slot.subgridspec(1, 2, wspace=0.08)
    ax_f = fig.add_subplot(sub[0, 0])
    ax_b = fig.add_subplot(sub[0, 1], sharey=ax_f)
    for ax, pas, lab in ((ax_f, "forward", "forward"), (ax_b, "forward_backward", "forward + backward")):
        nv = sel(rq2, "triton_naive", "noncausal", pas).set_index("N").ms
        rs = sel(rq2, "triton_rabe_staats", "noncausal", pas).set_index("N").ms
        f2 = sel(rq2, "triton_fa2", "noncausal", pas).set_index("N").ms
        of = sel(rq2, "flash_attn", "noncausal", pas).set_index("N").ms
        for ratio, color, mk, txt in (
                (nv / f2, "#1f77b4", "o", "Triton FA2 vs Triton naive"),
                (rs / f2, "#9467bd", "o", "Triton FA2 vs Triton Rabe-Staats"),
                (f2 / of, "#555555", "s", "official FA2 vs Triton FA2")):
            r = ratio.dropna()
            ax.plot(r.index, r, "-", color=color, marker=mk, ms=4, lw=1.8, label=txt)
            if len(r):
                ax.annotate(f"{r.iloc[-1]:.1f}\u00d7", (r.index[-1], r.iloc[-1]), xytext=(4, 0),
                            textcoords="offset points", fontsize=7.5, color=color, va="center")
        ax.axhline(1, color="k", lw=0.8)
        xaxis(ax, NS_R)
        ax.set_xlim(850, 24000)
        ax.set_title(lab, fontsize=9)
    ax_f.set_ylim(0.8, 4.2)
    ax_f.set_ylabel("times faster  (slower method's time / faster method's time)", fontsize=8.5)
    plt.setp(ax_b.get_yticklabels(), visible=False)
    ax_b.legend(fontsize=7, loc="upper right", title="blue, purple: IO-aware tiling\ngrey: engineering", title_fontsize=6.8)
    ax_f.annotate("(e) How much faster is the first-named kernel? (non-causal; above 1 = faster)\n"
                  "Triton Rabe-Staats is slower than Triton naive because it moves more HBM bytes (FP32 chunk\n"
                  "scores), so FA2's lead over it is larger. Naive fwd+bwd stops at 4K (out of memory).",
                  xy=(0, 1), xycoords="axes fraction", xytext=(0, 34), textcoords="offset points",
                  fontsize=8.2, ha="left", va="bottom")

    ax = axes[1, 2]
    for mth in ["flash_attn_v1", "triton_fa1", "triton_fa2", "flash_attn"]:
        line(ax, rq2, mth, "noncausal", "forward")
    xaxis(ax, NS_R); ax.set_yscale("log"); ax.set_ylabel("Runtime, forward (ms)")
    r1 = n8.loc["flash_attn_v1", "dram_total_MB"] / n8.loc["flash_attn", "dram_total_MB"]
    ax.set_title(f"(f) Same idea, FA1 \u2192 FA2: FA1 keeps its running output in HBM\n"
                 f"({r1:.0f}\u00d7 more traffic at N=8K) and is ~3\u00d7 slower", fontsize=9.5)
    ax.legend(fontsize=7.5)
    save(fig, "rq2_io_vs_engineering.png", RQ2_CFG + " Panel (b): 8 heads, causal. Nsight Compute 2025.1, DRAM bytes.")

# ------------------------------------------------------------------------------------------------ RQ3
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
fig.suptitle("RQ3: Does exact FlashAttention match the efficiency of approximate attention, without approximation error?",
             fontsize=12, weight="bold")
ax = axes[0]
for mth in ["naive", "performer", "linformer", "flash_attn"]:
    line(ax, prim, mth, "noncausal", "forward_backward")
mark_oom(ax, prim_oom, "naive", "noncausal", "forward_backward", 3e3)
xaxis(ax, NS_P); ax.set_yscale("log"); ax.set_ylabel("Runtime, forward + backward (ms)")
ax.set_title("(a) Non-causal: FA2 is faster than Linformer up to ~2K\n"
             "and than Performer up to 16K (Performer wins from 32K)", fontsize=9.5)
ax.legend(fontsize=8)
ax = axes[1]
for mth in ["naive", "performer", "flash_attn"]:
    line(ax, prim, mth, "causal", "forward_backward")
mark_oom(ax, prim_oom, "naive", "causal", "forward_backward", 3e3)
xaxis(ax, NS_P); ax.set_yscale("log"); ax.set_ylabel("Runtime, forward + backward (ms)")
ax.set_title("(b) Causal (Linformer has no causal mode):\nFA2 beats Performer at every N up to 64K", fontsize=9.5)
ax.legend(fontsize=8)
ax = axes[2]
for mth in ["sdpa_flash", "rabe_staats", "naive", "performer", "linformer", "flash_attn"]:
    s = sel(prim, mth, "noncausal", "forward").dropna(subset=["err"])
    ax.plot(s.N, s.err, "--" if mth == "sdpa_flash" else "-", color=COLOR[mth], marker="o", ms=4,
            lw=2.4 if mth == "flash_attn" else 1.4, label=NAME[mth], zorder=4 if mth == "flash_attn" else 3)
ax.axhspan(1e-3, 1e-2, color="#2ca02c", alpha=0.08)
ax.text(560, 1.25e-3, "BF16 rounding level", fontsize=8, color="#2a2")
ax.axhline(1, color="#d62728", lw=0.8, ls=":")
ax.text(560, 1.15, "error = 1 (no better than zero output)", fontsize=7, color="#d62728")
xaxis(ax, [512, 1024, 2048, 4096]); ax.set_yscale("log")
ax.set_ylabel("Relative error vs FP64 reference (Frobenius)")
ax.set_title("(c) Accuracy: exact methods at BF16 rounding (~2e-3);\n"
             "Performer / Linformer far off on these inputs", fontsize=9.5)
ax.legend(fontsize=7.5)
save(fig, "rq3_exact_vs_approximate.png",
     PRIMARY_CFG + " Inputs i.i.d. N(0,1); approximations untrained (Performer m=621 features, Linformer k=256). "
     "Error measured for N \u2264 4096 (FP64 reference limit).")

# ------------------------------------------------------------------------------------------------ RQ4
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
fig.suptitle("RQ4: Does the advantage grow with N, and does it differ between causal and non-causal masking?",
             fontsize=12, weight="bold")
for ax, base in zip(axes[:2], ["naive", "rabe_staats"]):
    for mask, color in (("noncausal", "#1f77b4"), ("causal", "#d62728")):
        for pas, ls in (("forward", "-"), ("forward_backward", "--")):
            b = sel(prim, base, mask, pas).set_index("N").ms
            f = sel(prim, "flash_attn", mask, pas).set_index("N").ms
            r = (b / f).dropna()
            ax.plot(r.index, r, ls, color=color, marker="o", ms=3.5,
                    label=f"{'non-causal' if mask == 'noncausal' else 'causal'}, "
                          f"{'forward' if pas == 'forward' else 'forward + backward'}")
    ax.axhline(1, color="k", lw=0.8)
    xaxis(ax, NS_P); ax.set_yscale("log")
    if base == "naive":
        ax.set_ylim(0.7, 80)  # head-room so the legend does not cover the curves
    ax.set_ylabel(f"times faster: {NAME[base]} time / FA2 time")
    # Legend title names the comparison; colour = masking, line style = pass
    ax.legend(fontsize=7.8, loc="upper left" if base == "naive" else "lower right",
              title=f"FlashAttention-2 (official) vs {NAME[base]}\n"
                                  "colour = masking, line = pass", title_fontsize=8)
axes[0].set_title("(a) vs PyTorch naive: grows from ~1\u00d7 to 3\u201313\u00d7; larger when causal\n"
                  "(curves end where naive runs out of memory)", fontsize=9.5)
_rs = [(sel(prim, "rabe_staats", mk, ps).set_index("N").ms / sel(prim, "flash_attn", mk, ps).set_index("N").ms)
       for mk in ("noncausal", "causal") for ps in ("forward", "forward_backward")]
_lo = [r[512] for r in _rs]
_hi = [r[65536] for r in _rs]
axes[1].set_title(f"(b) vs Rabe-Staats (PyTorch): {min(_lo):.0f}\u2013{max(_lo):.0f}\u00d7 at N=512, "
                  f"{min(_hi):.0f}\u2013{max(_hi):.0f}\u00d7 at N=64K", fontsize=9.5)
ax = axes[2]
_tf16 = []
for mask, color in (("noncausal", "#1f77b4"), ("causal", "#d62728")):
    for pas, ls in (("forward", "-"), ("forward_backward", "--")):
        s = sel(prim, "flash_attn", mask, pas)
        t = s.ms.values
        ns = s.N.values
        tflops = 4 * 8 * ns.astype(float) ** 2 * 128 * (3.5 if pas == "forward_backward" else 1.0) \
            / (2 if mask == "causal" else 1) / (t * 1e-3) / 1e12
        ax.plot(ns, tflops, ls, color=color, marker="o", ms=3.5,
                label=f"{'non-causal' if mask == 'noncausal' else 'causal'}, "
                      f"{'forward' if pas == 'forward' else 'forward + backward'}")
        _tf16 += [v for nn, v in zip(ns, tflops) if nn >= 16384]
ax.axhline(312, color="gray", ls=":"); ax.text(600, 290, "A100 BF16 dense peak (312)", fontsize=7.5, color="gray")
xaxis(ax, NS_P); ax.set_ylim(0, 340); ax.set_ylabel("FlashAttention-2 (official) achieved TFLOPS")
ax.set_title(f"(c) Why it grows: FA2 is launch-bound at small N (8 heads) and\n"
             f"reaches {min(_tf16) / 312:.0%}\u2013{max(_tf16) / 312:.0%} of peak from N \u2265 16K", fontsize=9.5)
ax.legend(fontsize=7.8, title="FlashAttention-2 (official) throughput\ncolour = masking, line = pass", title_fontsize=8)
save(fig, "rq4_scaling_and_masking.png", PRIMARY_CFG)

# ------------------------------------------------------------------------------------------------ RQ6 (partial)
fig, ax = plt.subplots(figsize=(9, 5.2))
fig.suptitle("RQ6 (partial): how much does the separate flash-attn library still add over PyTorch SDPA?",
             fontsize=11, weight="bold")
for base, color in (("sdpa_flash", "#2ca02c"), ("sdpa_efficient", "#17becf"), ("sdpa_math", "#7f7f7f")):
    for mask, ls in (("noncausal", "-"), ("causal", "--")):
        b = sel(prim, base, mask, "forward_backward").set_index("N").ms
        f = sel(prim, "flash_attn", mask, "forward_backward").set_index("N").ms
        r = (b / f).dropna()
        ax.plot(r.index, r, ls, color=color, marker="o", ms=3.5, label=f"{NAME[base]} / FA2, {mask}")
ax.axhline(1, color="k", lw=0.8)
xaxis(ax, NS_P); ax.set_yscale("log"); ax.set_ylabel("time relative to official FA2 (forward + backward)")
ax.set_title("SDPA flash is within ~5% of official FA2 from N \u2265 4K (faster below 2K: flash-attn call overhead);\n"
             "cuDNN backend not measured", fontsize=9)
ax.legend(fontsize=7.5)
save(fig, "rq6_sdpa_vs_flash_attn.png", PRIMARY_CFG)
