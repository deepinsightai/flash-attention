"""Sweep driver: methods x sequence length x mask x pass type on the current GPU.

    python -m bench.sweep --methods naive sdpa_flash --log2n-min 9 --log2n-max 14
    python -m bench.sweep --methods naive rabe_staats performer linformer sdpa_flash \
        --mask both --pass-type both --repeats 10 --check-error

Output (one directory per GPU model, so GPU types are never mixed):
    results/<gpu>/runs.csv      one row per repeat; the file analysis/benchmark_analysis.py reads
    results/<gpu>/oom.csv       configurations that ran out of memory (no invented runtimes)
    results/<gpu>/failures.csv  configurations that raised another error

Re-running the same command resumes: configurations already in runs.csv are skipped.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from . import export, metadata, methods
from .timing import measure

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=["naive", "sdpa_flash"])
    p.add_argument("--n", type=int, nargs="+", help="explicit sequence lengths (overrides --log2n-*)")
    p.add_argument("--log2n-min", type=int, default=9)
    p.add_argument("--log2n-max", type=int, default=14)
    p.add_argument("--mask", choices=["causal", "noncausal", "both"], default="noncausal")
    p.add_argument("--pass-type", choices=["forward", "forward_backward", "both"], default="forward")
    p.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--check-error", action="store_true", help="compare one output with the FP64 reference")
    p.add_argument("--error-max-n", type=int, default=4096, help="largest N for the FP64 reference check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results")
    return p.parse_args(argv)


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def run_config(args, meta, method, N, mask, pass_type, device):
    """Returns ('ok', rows) | ('oom', None) | ('fail', message)."""
    dtype = DTYPES[args.dtype]
    causal = mask == "causal"
    backward = pass_type == "forward_backward"
    gen = torch.Generator(device=device).manual_seed(args.seed)
    shape = (args.batch, args.heads, N, args.head_dim)
    q, k, v = (torch.randn(shape, device=device, dtype=dtype, generator=gen) for _ in range(3))
    if backward:
        for t in (q, k, v):
            t.requires_grad_(True)
    fn = methods.build(method, N, args.head_dim, device, args.seed)

    err = {}
    if args.check_error and N <= args.error_max_n:
        try:
            with torch.no_grad():
                err = methods.reference_error(fn(q, k, v, causal=causal), q, k, v, causal)
        except torch.OutOfMemoryError:
            err = {}  # the FP64 reference may not fit; leave the error blank

    results = measure(fn, q, k, v, causal=causal, backward=backward, warmup=args.warmup, repeats=args.repeats)
    stamp = metadata.utc_now()
    rows = []
    for i, m in enumerate(results):
        rows.append({
            "method": method, "gpu": meta["gpu"], "precision": args.dtype.upper(), "mask": mask,
            "pass_type": pass_type, "N": N, "batch_size": args.batch, "head_dim": args.head_dim,
            "repeat": i, "runtime_ms": round(m.runtime_ms, 6), "peak_memory_mb": round(m.peak_memory_mb, 3),
            "heads": args.heads,
            "numerical_error": err.get("rel_fro", ""), "max_abs_error": err.get("max_abs", ""),
            "timestamp_utc": stamp, "seed": args.seed, "warmup": args.warmup,
            **{c: meta[c] for c in ("gpu_mem_gb", "compute_capability", "sm_count", "torch_version",
                                    "cuda_version", "triton_version", "driver_version", "git_commit")},
        })
    return rows


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    device = torch.device("cuda", 0)
    meta = metadata.collect(0)
    out = Path(args.out_dir) / metadata.gpu_slug(meta["gpu"])
    runs_csv, oom_csv, fail_csv = out / "runs.csv", out / "oom.csv", out / "failures.csv"
    done = export.completed_configs(runs_csv)

    ns = sorted(args.n) if args.n else [2**i for i in range(args.log2n_min, args.log2n_max + 1)]
    masks = ["noncausal", "causal"] if args.mask == "both" else [args.mask]
    passes = ["forward", "forward_backward"] if args.pass_type == "both" else [args.pass_type]
    print(f"GPU: {meta['gpu']} (cc {meta['compute_capability']}), torch {meta['torch_version']}, writing to {out}")

    for pass_type in passes:
        for mask in masks:
            for method in args.methods:
                if mask == "causal" and not methods.supports_causal(method):
                    print(f"skip {method}: no causal mode")
                    continue
                for N in ns:
                    cid = (method, meta["gpu"], args.dtype.upper(), mask, pass_type, str(N),
                           str(args.batch), str(args.head_dim))
                    if tuple(map(str, cid)) in done:
                        print(f"resume-skip {method} {mask} {pass_type} N={N}")
                        continue
                    base = {"method": method, "gpu": meta["gpu"], "precision": args.dtype.upper(), "mask": mask,
                            "pass_type": pass_type, "N": N, "batch_size": args.batch,
                            "head_dim": args.head_dim, "heads": args.heads, "timestamp_utc": metadata.utc_now()}
                    try:
                        rows = run_config(args, meta, method, N, mask, pass_type, device)
                    except torch.OutOfMemoryError:
                        export.append_rows(oom_csv, export.OOM_COLUMNS, [base])
                        print(f"OOM  {method} {mask} {pass_type} N={N}; skipping larger N")
                        _cleanup()
                        break  # memory grows with N, so larger N would also fail
                    except Exception as e:  # noqa: BLE001 - record and continue with other configs
                        export.append_rows(fail_csv, export.FAIL_COLUMNS, [{**base, "error": repr(e)[:300]}])
                        print(f"FAIL {method} {mask} {pass_type} N={N}: {e!r}")
                        _cleanup()
                        continue
                    export.append_rows(runs_csv, export.RUN_COLUMNS, rows)
                    med = sorted(r["runtime_ms"] for r in rows)[len(rows) // 2]
                    print(f"ok   {method:14s} {mask:9s} {pass_type:16s} N={N:<7d} median {med:9.3f} ms")
                    _cleanup()


if __name__ == "__main__":
    main()
