# Baseline Algorithms for the FlashAttention Comparison

Author: Jason
Branch: `jason-research`
Course: COMP 9060 Group Project (BCIT, Fall 2026) — Option B (Modern Algorithm Application)

## Why baselines need careful framing

FlashAttention computes the **exact same function** as standard attention
(`softmax(Q Kᵀ) V`). It is not a different model with different outputs — it is
the same mathematics computed with a smarter memory/IO schedule. Therefore the
comparison axis for this project is **runtime, memory, and max sequence length**,
not accuracy.

The baselines below are *alternative ways to compute (or approximate) attention*.
They split into two families:

- **Exact** baselines: produce the same output as FlashAttention, differ only in
  compute/memory strategy.
- **Approximate** baselines: sub-quadratic methods that trade accuracy for
  speed/memory — the "traditional alternatives" people reached for before
  FlashAttention made exact attention cheap.

This satisfies the Option B rubric requirement of **2–4 traditional/classical
baselines** (Background & Baselines = 15%; Experimental Comparison = 20%), and it
directly supports the "advantages AND limitations" Critical Analysis criterion:
FlashAttention gets the speed/memory win with **no approximation**, whereas the
sub-quadratic methods pay an accuracy cost.

## Verified baseline citations

All citations verified against the primary arXiv sources (Oct 2026).

### 1. Naive / standard attention (primary exact baseline)
- The textbook implementation that materializes the full `N x N` score matrix in
  GPU HBM.
- Complexity: `O(N^2)` memory, `O(N^2 d)` compute.
- Role: the main "traditional" baseline FlashAttention was designed to beat.
- Reference: Vaswani et al., "Attention Is All You Need," 2017, arXiv:1706.03762.

### 2. Memory-efficient attention — Rabe & Staats (exact, chunked, not IO-aware)
- Title: "Self-attention Does Not Need O(n²) Memory"
- Authors: Markus N. Rabe, Charles Staats
- Year: 2021
- arXiv: **2112.05682**
- Idea: lazy-softmax chunking. `O(1)` memory for single-query attention,
  `O(log n)` for self-attention; **time stays `O(n²)`**.
- Role: isolates "chunking alone" from FlashAttention's IO-aware tiling. It is the
  direct conceptual predecessor to FlashAttention — comparing against it shows how
  much of FlashAttention's speedup comes specifically from being IO-aware (SRAM
  tiling), not just from chunked softmax.

### 3. Linformer (approximate, low-rank, linear)
- Title: "Linformer: Self-Attention with Linear Complexity"
- Authors: Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma
- Year: 2020
- arXiv: **2006.04768**
- Idea: the self-attention matrix is approximately low-rank; project keys/values
  to a smaller dimension `k`, reducing complexity from `O(n²)` to `O(nk)` ≈ `O(n)`
  in time and space.
- Role: represents the low-rank approximation family. Introduces an accuracy gap
  vs. exact attention — the trade-off FlashAttention avoids.

### 4. Performer (approximate, kernel/random-feature, linear)
- Title: "Rethinking Attention with Performers"
- Authors: Krzysztof Choromanski et al. (Google)
- Year: 2020
- arXiv: **2009.14794**
- Idea: FAVOR+ (Fast Attention Via positive Orthogonal Random features) — an
  unbiased kernel approximation of softmax attention that scales linearly in `n`.
- Role: represents the kernel/linear-attention family. Like Linformer, trades
  exactness for sub-quadratic scaling.

## Recommended configuration (3 baselines)

For a semester-scoped project, the cleanest and most gradeable story uses three
baselines spanning both families:

1. **Naive attention** (#1) — main traditional/exact baseline.
2. **Rabe & Staats memory-efficient attention** (#2) — chunked-but-not-IO-aware,
   to isolate the IO contribution.
3. **Linformer OR Performer** (#3 or #4) — one approximate linear-attention
   method representing the sub-quadratic family.

Narrative: exact-and-fast (FlashAttention) vs. exact-and-slow (naive) vs.
chunked-exact (Rabe–Staats) vs. approximate-but-sub-quadratic
(Linformer/Performer). This lands squarely in the rubric's "2–4" range.

Linformer and Performer may both be included if time allows (that uses the full
4-baseline budget).

## Metrics for a fair comparison

Swept over sequence length `N` (fix head dim `d`):

- Peak GPU memory
- Forward + backward latency
- Throughput (tokens/s)
- Max `N` before out-of-memory (OOM)

For the **approximate** baselines (Linformer, Performer), additionally report an
accuracy/quality gap:

- Output MSE vs. exact attention, and/or
- Downstream task accuracy

This is what makes the "exact vs. approximate" contrast meaningful and fills the
Critical Analysis criterion.

## Candidate PDFs to add to `papers/`

(The `papers/` folder is gitignored; store PDFs locally, cite by arXiv ID.)

| Baseline | arXiv | Suggested filename |
| -------- | ----- | ------------------ |
| Attention Is All You Need (naive) | 1706.03762 | `b1_attention_2017_1706.03762.pdf` |
| Rabe & Staats memory-efficient    | 2112.05682 | `b2_rabe-staats_2021_2112.05682.pdf` |
| Linformer                         | 2006.04768 | `b3_linformer_2020_2006.04768.pdf` |
| Performer                         | 2009.14794 | `b4_performer_2020_2009.14794.pdf` |
