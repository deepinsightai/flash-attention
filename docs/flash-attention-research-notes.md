# FlashAttention Research Notes

Author: Jason
Branch: `jason-research`
Course: COMP 9060 Group Project (BCIT, Fall 2026)

## Topic

Project option: **Option B — Modern Algorithm Application**.
Modern algorithm under study: **FlashAttention**, an IO-aware exact attention
algorithm for GPUs. The project compares FlashAttention against the standard
(naive) attention implementation, framing the contribution as a
systems/complexity improvement (memory and data-movement), not a model-accuracy
one.

## The core problem

Standard scaled dot-product attention computes, for sequence length `N` and head
dimension `d`:

- `S = Q Kᵀ`  (an `N x N` score matrix)
- `P = softmax(S)`  (row-wise)
- `O = P V`

A naive implementation materializes the full `N x N` matrices `S` and `P` in GPU
high-bandwidth memory (HBM). That costs `O(N^2)` memory and, more importantly,
`O(N^2)` memory traffic. On modern GPUs attention is **memory-bound**, not
compute-bound: the bottleneck is moving data between large-but-slow HBM and
small-but-fast on-chip SRAM.

## The key idea: IO-awareness

FlashAttention computes **exact** attention (bit-for-bit the same math, not an
approximation) while minimizing HBM reads/writes. Two mechanisms:

1. **Tiling.** Split `Q`, `K`, `V` into blocks that fit in SRAM. Load a `Q`
   block and stream `K`/`V` blocks through SRAM, accumulating the output
   block-by-block. The full `N x N` score matrix is never written to HBM.
2. **Online softmax.** Softmax needs a row-wide normalization, but tiling only
   exposes one block at a time. Keep a running max `m` and running sum `l`; when
   a new block has a larger max, rescale the accumulated output by
   `exp(old_max - new_max)` and continue. The final result is the exact softmax,
   computed incrementally.

Backward pass: instead of storing `P`, FlashAttention **recomputes** attention
tiles on the fly — trading a little extra compute for a large memory saving (a
time/space tradeoff).

## Complexity summary

| Quantity        | Naive attention | FlashAttention |
| --------------- | --------------- | -------------- |
| Memory          | `O(N^2)`        | `O(N)`         |
| HBM traffic     | `O(N^2)`        | `O(N^2 d^2 / M)` (M = SRAM size) |
| Compute (FLOPs) | `O(N^2 d)`      | `O(N^2 d)` (unchanged) |

The win is entirely in **data movement**; the arithmetic is unchanged. This is
why the speedup is large even though FLOPs are identical — attention was
memory-bound.

## The paper series (the "evolution of ideas")

All versions keep the same exact-attention math and online-softmax core; each
re-co-designs the kernel to the newest GPU's memory hierarchy and features.

| # | Paper | Year | arXiv | Key contribution |
| - | ----- | ---- | ----- | ---------------- |
| 1 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | 2022 | 2205.14135 | Original IO-aware tiling + online softmax; `O(N)` memory; 2–4x speedup |
| 2 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | 2023 | 2307.08691 | Better GPU work partitioning, fewer non-matmul ops; ~2x over v1 (A100) |
| 3 | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | 2024 | 2407.08608 | Hopper/H100: asynchrony (warp specialization) + FP8; ~1.2 PFLOPS FP8 |
| 4 | FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling | 2026 | 2603.05451 | Blackwell/B200: algorithm-kernel pipelining; implemented in CuTe-DSL |

PDFs are stored in the project `papers/` folder (outside this repo).

## Planned experiments

- **Baseline:** standard PyTorch attention (`softmax(QKᵀ)V`), plus optionally a
  memory-efficient non-IO-aware variant.
- **Modern:** FlashAttention via `flash-attn` or PyTorch
  `scaled_dot_product_attention` (FlashAttention backend).
- **Sweep:** fix `d`, vary sequence length `N` (e.g. 512 → 32k). Measure:
  1. peak GPU memory,
  2. forward + backward wall-clock time,
  3. throughput (tokens/s),
  4. max `N` before out-of-memory (OOM).
- **Correctness:** assert FlashAttention output matches the naive output within
  floating-point tolerance (demonstrates "exact, not approximate").
- **Rigor:** multiple runs; report mean ± std; plot memory and time vs. `N` on
  log scales.
- **Wireless tie-in (course theme):** run on a long-sequence wireless task
  (Transformer over channel-state / signal time-series) where long context is
  the payoff.

## Hardware caveat

- v1/v2 require NVIDIA Ampere or newer; v3 requires Hopper (H100); v4 targets
  Blackwell (B200). Our reproducible benchmarks will realistically run against
  the GPU generation we can actually access. v4 benchmarks are not reproducible
  without a B200 — it belongs in the "state of the art / future directions"
  narrative.

## Open questions / TODO

- [ ] Confirm which GPU the team has access to (determines max FlashAttention version).
- [ ] Pick the wireless long-sequence dataset for the applied experiment.
- [ ] Gather related-work predecessors: online softmax (Milakov & Gimelshein, 2018),
      Rabe & Staats "Self-attention Does Not Need O(n^2) Memory" (2021), and
      approximate-attention baselines (Performer, Linformer, Reformer).
