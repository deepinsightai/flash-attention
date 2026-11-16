"""Search launch configurations for the Triton kernels used in RQ2, on the current GPU.

    python -m scripts.tune_triton                 # full search (A100: roughly 20-30 min, mostly compiling)
    python -m scripts.tune_triton --quick         # small grids and sizes, to test the script

Tuned groups (each one a coordinate-descent step, in this order):
    fa1 fwd, fa1 bwd, fa2 fwd, fa2 bwd     (BLOCK_M, BLOCK_N, num_warps, num_stages)
    bmm[naive], softmax[naive]             matmul tiles, then softmax block/warps
    bmm[rabe_staats], rs_elem[rabe_staats] matmul tiles, then summary/elementwise warps

Rules, identical for every group:
    - candidates = a fixed grid (plus the original default), pruned only by a shared-memory estimate
      and, for FA2 forward, BLOCK_M % BLOCK_N == 0 (required for correct causal block skipping)
    - every candidate must match an FP64 reference (forward, and gradients where a backward is
      tuned) on a small odd-sized problem before it is timed
    - objective = geometric mean of median times over the group's N values and both masks
      (warm-up 2, timed 5 per point); a candidate is abandoned once any point is 3x the best
    - the winner is written to configs/tuning/<gpu>.json, which bench.triton_attn loads on import

Assumption: configurations are tuned in BF16 and reused for FP16 (same element size, same tiles).
The tuning search is never part of the timed benchmark runs.
"""

from __future__ import annotations

import argparse
import itertools
import math
import time

import torch

from bench import metadata
from bench.triton_attn import METHODS, _util, tuning
from bench.triton_attn._util import TUNED, dtype_name

DEV = "cuda"


# ---------------------------------------------------------------------------------------- helpers
def ref_attention(q, k, v, causal, scale):
    s = (q.double() @ k.double().transpose(-2, -1)) * scale
    if causal:
        n = q.shape[2]
        s = s.masked_fill(torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return torch.softmax(s, -1) @ v.double()


def rel(a, b):
    return ((a.double() - b).norm() / b.norm()).item()


def correct(method, dtype, with_grad, d, tol=1.5e-2) -> bool:
    torch.manual_seed(0)
    n = 320  # not a multiple of any block size, exercises the boundary masks
    for causal in (False, True):
        q, k, v = (torch.randn(1, 2, n, d, device=DEV, dtype=dtype, requires_grad=with_grad) for _ in range(3))
        out = METHODS[method](q, k, v, causal=causal)
        qd, kd, vd = (t.detach().double().requires_grad_(with_grad) for t in (q, k, v))
        ref = ref_attention(qd, kd, vd, causal, d ** -0.5)
        if rel(out, ref) > tol:
            return False
        if with_grad:
            g = torch.randn_like(out)
            out.backward(g)
            ref.backward(g.double())
            if any(rel(a.grad, b.grad) > tol for a, b in ((q, qd), (k, kd), (v, vd))):
                return False
    return True


def time_call(fn, warmup=2, reps=5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


def make_call(method, n, causal, args, dtype, backward):
    shape = (args.batch, args.heads, n, args.head_dim)
    q, k, v = (torch.randn(shape, device=DEV, dtype=dtype, requires_grad=backward) for _ in range(3))
    fn = METHODS[method]
    if backward:
        g = torch.randn(shape, device=DEV, dtype=dtype)

        def call():
            fn(q, k, v, causal=causal).backward(g)
            q.grad = k.grad = v.grad = None
    else:
        def call():
            with torch.no_grad():
                fn(q, k, v, causal=causal)
    return call


def evaluate(method, ns, args, dtype, backward, best_points):
    """Geometric-mean median time (ms) over ns x masks, or inf if abandoned/failed.

    Fixed: a candidate is abandoned only when one point is 3x slower than the
    best candidate's time at the SAME (N, mask). The first version compared each point with the best
    candidate's geometric mean over all N, which abandoned nearly every candidate at its largest N.
    """
    logs = []
    for n in ns:
        for causal in (False, True):
            try:
                t = time_call(make_call(method, n, causal, args, dtype, backward), args.warmup, args.reps)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                return math.inf, logs + [(n, causal, "oom")]
            logs.append((n, causal, t))
            ref = best_points.get((n, causal))
            if ref is not None and t > 3 * ref:  # clearly worse than the best at this exact point
                return math.inf, logs
    return math.exp(sum(math.log(t) for _, _, t in logs) / len(logs)), logs


def search(group, method, candidates, setter, ns, args, dtype, backward, log):
    """Try every candidate (default first); keep the fastest correct one."""
    best_t, best_c = math.inf, None
    best_points: dict = {}
    default_t = math.inf
    t0 = time.time()
    for i, cand in enumerate(candidates):
        setter(cand)
        rec = {"group": group, "config": list(cand)}
        try:
            if not correct(method, dtype, backward, args.head_dim):
                rec["result"] = "incorrect"
                log.append(rec)
                continue
            t, points = evaluate(method, ns, args, dtype, backward, best_points)
        except Exception as e:  # noqa: BLE001 - compile errors, out of resources
            rec["result"] = f"error: {type(e).__name__}: {str(e)[:120]}"
            log.append(rec)
            torch.cuda.empty_cache()
            continue
        rec["result"] = t if t < math.inf else "abandoned"
        rec["points"] = [[n, c, p] for n, c, p in points]
        log.append(rec)
        if i == 0:
            default_t = t
        if t < best_t:
            best_t, best_c = t, cand
            best_points = {(n, c): p for n, c, p in points}
    if best_c is None:
        raise RuntimeError(f"{group}: no candidate was both correct and runnable; see the search log")
    setter(best_c)
    dt = time.time() - t0
    speedup = (default_t / best_t) if best_t < math.inf and default_t < math.inf else float("nan")
    print(f"{group:22s} best {best_c} geo-mean {best_t:9.3f} ms | default {default_t:9.3f} ms "
          f"| x{speedup:.2f} | {len(candidates)} candidates, {dt:.0f} s", flush=True)
    return {"group": group, "best": list(best_c), "best_ms": best_t, "default_ms": default_t,
            "candidates": len(candidates), "seconds": round(dt, 1)}


def smem_ok(bytes_needed, limit):
    return bytes_needed <= limit


# ------------------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=128)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--quick", action="store_true", help="tiny grids and sizes, for testing the script")
    p.add_argument("--only", nargs="+", help="subset of groups: fa1 fa2 naive rabe_staats")
    p.add_argument("--fa1-ns", type=int, nargs="+", default=[1024, 4096],
                   help="N values for FA1 (slow at large N; it is only a calibration reference)")
    a = p.parse_args(argv)
    if a.quick:
        a.heads = min(a.heads, 16)

    dtype = torch.bfloat16
    d = a.head_dim
    props = torch.cuda.get_device_properties(0)
    smem = getattr(props, "shared_memory_per_block_optin", 99 * 1024)
    e = 2  # bytes per element for BF16/FP16
    groups = a.only or ["fa1", "fa2", "naive", "rabe_staats"]
    flash_ns = [1024, 2048] if a.quick else [1024, 4096, 16384]
    naive_ns = [512, 1024] if a.quick else [1024, 4096]
    rs_ns = [1024, 2048] if a.quick else [2048, 4096]
    log, summary = [], []
    print(f"GPU {metadata.gpu_label(0)}, smem/block {smem // 1024} KB, batch {a.batch} x heads {a.heads}, "
          f"d {d}, {dtype_name(dtype)}", flush=True)

    def fkey(v, pas):
        return (v, pas, d, dtype_name(dtype))

    # FlashAttention variants: forward first (no grad), then backward with the tuned forward fixed
    bm_f = [64, 128] if a.quick else [32, 64, 128]
    bn_f = [32, 64] if a.quick else [32, 64, 128]
    wr = [4] if a.quick else [4, 8]
    st_f = [1, 2] if a.quick else [1, 2, 3, 4]
    st_b = [1] if a.quick else [1, 2]
    bm_b = [32, 64] if a.quick else [16, 32, 64, 128]
    for v in (1, 2):
        if f"fa{v}" not in groups:
            continue
        method = f"triton_fa{v}"
        default_f = (TUNED["flash"].pop(fkey(v, "fwd"), None)
                     or (_util.flash_config(d, dtype, bwd=True) if v == 1 else _util.flash_config(d, dtype)))
        fwd = [default_f] + [c for c in itertools.product(bm_f, bn_f, wr, st_f)
                             if c != default_f
                             and smem_ok((c[0] * d + 2 * c[1] * d * max(c[3], 1)) * e, smem)
                             and (v != 2 or c[0] % c[1] == 0)]
        ns_v = (a.fa1_ns if not a.quick else flash_ns) if v == 1 else flash_ns
        summary.append(search(f"fa{v} fwd", method, fwd,
                              lambda c, v=v: TUNED["flash"].__setitem__(fkey(v, "fwd"), tuple(c)),
                              ns_v, a, dtype, False, log))
        TUNED["flash"].pop(fkey(v, "bwd"), None)
        default_b = _util.flash_config(d, dtype, bwd=True)
        bwd = [default_b] + [c for c in itertools.product(bm_b, bn_f, wr, st_b)
                             if c != default_b
                             and smem_ok((2 * c[0] * d + 2 * c[1] * d * max(c[3], 1)) * e, smem)]
        summary.append(search(f"fa{v} bwd", method, bwd,
                              lambda c, v=v: TUNED["flash"].__setitem__(fkey(v, "bwd"), tuple(c)),
                              ns_v, a, dtype, True, log))
        if v >= 2:  # The separate dQ kernel of the FA2 backward
            from bench.triton_attn.flash_bwd import dq_config

            TUNED["flash"].pop(fkey(v, "bwd_dq"), None)
            default_q = dq_config(d, dtype, v)
            dqc = [default_q] + [c for c in itertools.product(bm_f, bn_f, wr, st_b)
                                 if c != default_q
                                 and smem_ok((2 * c[0] * d + 2 * c[1] * d * max(c[3], 1)) * e, smem)]
            summary.append(search(f"fa{v} bwd_dq", method, dqc,
                                  lambda c, v=v: TUNED["flash"].__setitem__(fkey(v, "bwd_dq"), tuple(c)),
                                  ns_v, a, dtype, True, log))

    # matmul tiles per caller, then the caller's small kernels
    tiles = [64, 128] if a.quick else [32, 64, 128]
    bks = [32] if a.quick else [32, 64]
    st_m = [2] if a.quick else [2, 3, 4]
    for tag, method, ns in (("naive", "triton_naive", naive_ns), ("rabe_staats", "triton_rabe_staats", rs_ns)):
        if tag not in groups:
            continue
        TUNED["bmm"].pop(tag, None)
        default_m = (64, 64, 32, 4, 2)
        cands = [default_m] + [c for c in itertools.product(tiles, tiles, bks, wr, st_m)
                               if c != default_m and smem_ok((c[0] * c[2] + c[2] * c[1]) * 4 * c[4], smem)]
        summary.append(search(f"bmm[{tag}]", method, cands,
                              lambda c, tag=tag: TUNED["bmm"].__setitem__(tag, tuple(c)),
                              ns, a, dtype, True, log))
        if tag == "naive":
            TUNED["softmax"].pop("naive", None)
            sm = [(1024, 4)] + [c for c in itertools.product([256, 512, 1024, 2048, 4096], [2, 4, 8, 16])
                                if c != (1024, 4)]
            if a.quick:
                sm = sm[:4]
            summary.append(search("softmax[naive]", method, sm,
                                  lambda c: TUNED["softmax"].__setitem__("naive", tuple(c)),
                                  ns, a, dtype, True, log))
        else:
            TUNED["rs_elem"].pop("rabe_staats", None)
            el = [(8, 4)] + [c for c in itertools.product([2, 4, 8, 16], [2, 4, 8]) if c != (8, 4)]
            if a.quick:
                el = el[:4]
            summary.append(search("rs_elem[rabe_staats]", method, el,
                                  lambda c: TUNED["rs_elem"].__setitem__("rabe_staats", tuple(c)),
                                  ns, a, dtype, True, log))

    # reuse the BF16 FlashAttention configs for FP16 (same element size)
    for (v, pas, dd, dt), cfg in list(TUNED["flash"].items()):
        if dt == "bfloat16":
            TUNED["flash"][(v, pas, dd, "float16")] = cfg

    meta = {**metadata.collect(0), "timestamp_utc": metadata.utc_now(), "batch": a.batch, "heads": a.heads,
            "head_dim": d, "dtype": dtype_name(dtype), "quick": a.quick, "flash_ns": flash_ns, "fa1_ns": a.fa1_ns,
            "naive_ns": naive_ns, "rs_ns": rs_ns, "warmup": a.warmup, "reps": a.reps, "summary": summary}
    path = tuning.save(dict(TUNED), meta, log)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
