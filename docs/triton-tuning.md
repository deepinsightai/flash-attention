# Triton kernel tuning for RQ2: process and results

RQ2 asks how much of FlashAttention's speed comes from I/O-aware tiling rather than engineering. To
answer it, naive attention, Rabe-Staats, FlashAttention-1 and FlashAttention-2 are all written in Triton
(`bench/triton_attn/`) and compared with each other. That comparison is only fair if every kernel gets the
same tuning effort, so all of them go through one search with identical rules. This document describes
that search and what it found on the A100.

- Script: `scripts/tune_triton.py`
- Loader: `bench/triton_attn/tuning.py` (applied automatically when `bench.triton_attn` is imported)
- Result file: `configs/tuning/nvidia-a100-sxm4-40gb-40gb.json` (SHA-256 `04cfe17ae5477479...`)
- Used by: `data/a100-2026-11-14/results_rq2_v2` and `results_rq2_newbwd`

## 1. What is tuned

Each kernel family exposes launch parameters through `bench/triton_attn/_util.TUNED`. Without a tuning
file the kernels use their original fixed defaults.

| Group | Kernel | Parameters | Original default |
| --- | --- | --- | --- |
| `fa1 fwd` | FA1 forward | (BLOCK_M, BLOCK_N, num_warps, num_stages) | (32, 64, 4, 1) |
| `fa1 bwd` | FA1 backward | same | (32, 64, 4, 1) |
| `fa2 fwd` | FA2 forward | same | (128, 32, 4, 2) |
| `fa2 bwd` | FA2 backward, dK/dV kernel | same | (32, 64, 4, 1) |
| `fa2 bwd_dq` | FA2 backward, dQ kernel | same | (64, 64, 4, 1) |
| `bmm[naive]` | Matrix multiply used by naive (QK^T, PV and the backward products) | (BM, BN, BK, num_warps, num_stages) | (64, 64, 32, 4, 2) |
| `softmax[naive]` | Naive softmax forward and backward | (BLOCK, num_warps) | (1024, 4) |
| `bmm[rabe_staats]` | Matrix multiply used by Rabe-Staats (separate entry from naive) | (BM, BN, BK, num_warps, num_stages) | (64, 64, 32, 4, 2) |
| `rs_elem[rabe_staats]` | Rabe-Staats chunk-summary kernel warps, backward elementwise kernel warps | (warps, warps) | (8, 4) |

The Rabe-Staats chunk sizes (1024 queries x 4096 keys, as in the paper) are fixed, not tuned.

## 2. Search rules (identical for every group)

1. **Candidates:** a fixed grid plus the current default (always tried first).
   - Flash forward: BLOCK_M in {32, 64, 128}, BLOCK_N in {32, 64, 128}, warps in {4, 8}, stages in {1, 2, 3, 4}.
   - Flash backward and dQ: BLOCK_M in {16, 32, 64, 128} (dQ: {32, 64, 128}), BLOCK_N in {32, 64, 128}, warps in {4, 8}, stages in {1, 2}.
   - Matmul: BM and BN in {32, 64, 128}, BK in {32, 64}, warps in {4, 8}, stages in {2, 3, 4}.
   - Softmax: BLOCK in {256, ..., 4096}, warps in {2, 4, 8, 16}. Rabe-Staats warps: {2, 4, 8, 16} x {2, 4, 8}.
2. **Pruning:** only by an estimate of shared memory against the GPU's per-block limit (163 KB on the A100),
   and for FA2 forward by BLOCK_M % BLOCK_N == 0, which correct causal block-skipping needs.
3. **Correctness gate:** before timing, every candidate is checked against an FP64 reference on a small
   problem with N = 320 (not a multiple of any block size, so the boundary masks are exercised), both masks.
   Groups that tune a backward also check dQ, dK and dV. Tolerance: relative error 1.5e-2.
4. **Objective:** geometric mean of the median time over the group's N values and both masks. Each point is
   2 warm-up calls and 5 timed calls, measured with CUDA events. Forward groups time the forward only;
   backward groups time forward+backward.
5. **Early stop:** a candidate is abandoned once one of its points is more than 3x slower than the best
   candidate so far at the same (N, mask).
6. **Order:** coordinate descent. FA1 forward, then FA1 backward with the forward fixed, then FA2 forward,
   FA2 dK/dV, FA2 dQ; then the naive matmul, then naive softmax; then the Rabe-Staats matmul, then its
   elementwise kernels.
7. **Precision:** tuned in BF16; the same configurations are reused for FP16 (same element size).
8. Tuning runs are never part of the timed benchmark runs.

Problem size: batch 1, 128 heads, d = 128, BF16 (the RQ2 shape). N values per group:

| Group | N values |
| --- | --- |
| FA2 | 1024, 4096, 16384 |
| FA1 | 1024, 4096 (FA1 is slow at large N and is only a calibration reference) |
| Naive | 1024, 4096 |
| Rabe-Staats | 2048, 4096 |

## 3. How to run

```bash
source ~/bench-venv/bin/activate              # torch 2.8.0, Triton 3.4.0 (see scripts/lambda_setup.sh)
python -m scripts.tune_triton                 # full search; about 16 min on the A100, plus compile time
python -m scripts.tune_triton --quick         # tiny grids, to test the script on any GPU
python -m scripts.tune_triton --only fa2      # one family
python -m pytest tests/test_triton_attn.py -q # re-check correctness with the tuned file in place
```

The output is `configs/tuning/<gpu-slug>.json` with three parts: `meta` (GPU, driver, versions, code
label, settings and a per-group summary), `tuned` (the chosen configurations), and `search_log` (every
candidate with its result: time, `abandoned`, `incorrect` or the compile error, and its per-point times).
`bench.triton_attn` loads the file matching the current GPU name. Other GPUs fall back to the defaults, so
tune once per GPU model.

## 4. History on the A100 (2026-11-14 and 15, UTC)

1. **First tuning (03:16 UTC), discarded.** The early-stop rule compared each point with the best
   candidate's *mean over all N*. A candidate's large-N point is always far above that mean, so almost
   every candidate was abandoned at its largest N. For example, 43 of 44 FA2-forward candidates and 99 of
   100 naive-matmul candidates were dropped, and most groups kept their default. It also ran with the old
   FA2 backward, which used atomic adds and took about 3.3x longer than the current one. The first sweep
   (`results_rq2`, not in this repo) used this tuning and is superseded.
2. **Fix.** The rule now compares each point with the best candidate's time at the *same* (N, mask)
   (`evaluate()` in `scripts/tune_triton.py`). The FA2 backward was rewritten into separate dK/dV and dQ
   kernels without atomics, and the dQ kernel became its own tuning group.
3. **Second attempt**, discarded: it overlapped another GPU job, so its timings were contended.
4. **Final tuning (04:19 UTC)**, all groups, GPU otherwise idle. This is the committed file. Total
   search time 975 s.

## 5. Results (final tuning)

"Start" is the first candidate tried in the final run: the original default, except for FA1 forward, where
it was the first tuning's choice (the script starts from whatever configuration is loaded). Times are
the geometric mean over the group's points (ms).

| Group | Start config | Start (ms) | Chosen config | Chosen (ms) | Gain | Candidates (timed / abandoned / failed) |
| --- | --- | --- | --- | --- | --- | --- |
| fa1 fwd | (32, 128, 4, 1) | 6.170 | (32, 128, 4, 2) | 5.298 | 1.16x | 36 / 22 / 2 |
| fa1 bwd | (32, 64, 4, 1) | 18.403 | (16, 64, 4, 1) | 17.199 | 1.07x | 22 / 22 / 2 |
| fa2 fwd | (128, 32, 4, 2) | 6.662 | (128, 32, 4, 3) | 5.720 | 1.16x | 33 / 11 / 0 |
| fa2 bwd (dK/dV) | (32, 64, 4, 1) | 26.490 | (32, 64, 4, 1) | 26.490 | 1.00x | 28 / 16 / 2 |
| fa2 bwd_dq | (64, 64, 4, 1) | 26.491 | (128, 32, 4, 2) | 24.810 | 1.07x | 30 / 4 / 0 |
| bmm[naive] | (64, 64, 32, 4, 2) | 13.239 | (128, 64, 64, 4, 3) | 11.463 | 1.15x | 100 / 0 / 0 |
| softmax[naive] | (1024, 4) | 11.453 | (1024, 8) | 11.142 | 1.03x | 20 / 0 / 0 |
| bmm[rabe_staats] | (64, 64, 32, 4, 2) | 40.080 | (128, 64, 64, 4, 3) | 36.270 | 1.11x | 100 / 0 / 0 |
| rs_elem[rabe_staats] | (8, 4) | 36.294 | (16, 2) | 36.251 | 1.00x | 12 / 0 / 0 |

No candidate failed the correctness gate. The six failures were compile-time `OutOfResources` errors:
the shared-memory estimate let through configurations that needed 184 to 229 KB against the 163 KB limit
(FA1 forward with BLOCK_M 128 and 4 stages; 128 x 128 backward tiles).

Chosen configurations, per point (median ms, batch 1, 128 heads, d = 128, BF16):

| Group | N = 1024 non-causal / causal | N = 4096 non-causal / causal | N = 16384 non-causal / causal |
| --- | --- | --- | --- |
| fa1 fwd | 1.90 / 1.15 | 26.40 / 13.65 | not tuned |
| fa1 fwd+bwd | 6.07 / 3.51 | 89.18 / 46.11 | not tuned |
| fa2 fwd | 0.54 / 0.44 | 6.30 / 4.10 | 100.43 / 57.32 |
| fa2 fwd+bwd | 2.32 / 1.57 | 30.60 / 17.29 | 474.55 / 254.59 |
| naive fwd+bwd | 2.96 / 3.03 | 41.26 / 41.64 | not tuned |

Rabe-Staats forward+backward, N = 2048 and 4096: 18.6 and 70.5 ms (both masks; its causal path does not skip work).

## 6. What the results mean

- **Tuning moved each kernel by 0 to 16%.** No method gained or lost enough to change the RQ2 ranking.
  The gaps that RQ2 measures are 2.5x to 3.6x (Triton FA2 against Triton naive and Rabe-Staats, forward),
  far larger than any tuning gain.
- **Every method got the same search,** so the remaining differences come from the algorithms and their
  memory traffic, not from how much effort went into each kernel.
- **Against the original defaults** (not the first tuning), FA1 forward improved from 9.652 ms to 5.298 ms
  (1.82x). The default FA1 forward used the backward's tile table, which was a poor fit.
- **The FA2 backward speedup came from the rewrite, not tuning.** The old atomic backward measured
  88.7 ms on the same objective; the rewritten one 26.5 ms (3.35x). Tuning added 1.07x on the dQ kernel.

## 7. Caveats

1. **FA1 forward start value.** The final run measured its starting configuration (32, 128, 4, 1) at
   6.170 ms, but the first tuning had measured the same configuration at 5.300 ms. The difference is in
   the non-causal points only (N = 4096: 32.1 against 26.1 ms) and is unexplained. It was the first
   candidate of the run, so GPU clock ramp-up is a possible cause (clocks were not locked). The chosen
   (32, 128, 4, 2) at 5.298 ms equals the earlier measurement, so the real gain from the final FA1-forward
   search is probably close to 1.0x, not 1.16x.
2. Five timed calls per point is a short measurement; differences under about 3% are within noise.
3. Tuning used 128 heads. The same configurations are used for the 8-head Nsight runs, where they were not tuned.
4. FP16 reuses the BF16 choices without its own search.
5. Coordinate descent tunes one kernel at a time with the others fixed; it does not search combinations.
6. Clocks were not locked during tuning or the timed sweeps.
