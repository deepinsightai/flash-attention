# Attention baselines

Reference implementations of the four baselines for the FlashAttention reproducibility benchmark (COMP 9060, BCIT, Fall 2026), with correctness tests and two sanity-check experiments.

| Method | File | Exact | Causal | Time | Extra memory |
| --- | --- | --- | --- | --- | --- |
| Naive attention | `naive.py` | Yes | Yes | Θ(N²d) | Θ(N²) |
| Rabe–Staats | `rabe_staats.py` | Yes | Yes | Θ(N²d) | O(query_chunk × key_chunk) |
| Performer (FAVOR+) | `performer.py` | No | Yes (chunked) | Θ(Nmd) | Θ(Nm + md) |
| Linformer | `linformer.py` | No | **No** | Θ(Nkd) | Θ(Nk) |

## Interface

Every method has the same signature:

```python
out = attn(q, k, v, causal=False, scale=None)
# q: [B, H, Nq, d]   k: [B, H, Nk, d]   v: [B, H, Nk, dv]   ->   out: [B, H, Nq, dv]
```

`scale` defaults to 1/√d. Output dtype equals input dtype. Rabe–Staats and Performer accumulate internally in FP32 (FP64 inputs stay FP64), as GPU attention kernels do.

For the benchmark harness, use the registry. It draws each method's random parameters once, so they are not regenerated or timed on every call:

```python
from attention_baselines import get_attention, METHODS, SUPPORTS_CAUSAL

attn = get_attention("performer", seq_len=4096, head_dim=128, device="cuda", seed=0, num_features=256)
out = attn(q, k, v, causal=True)
```

Error against exact attention:

```python
from attention_baselines import reference_attention, attention_error

err = attention_error(out, reference_attention(q, k, v, causal=True))  # {'rel_fro', 'max_abs', 'mean_abs'}
```

`reference_attention` computes in FP64 and materializes an N × N FP64 matrix, so measure error at moderate N (N = 16K needs 2 GB per head).

## Running

Correctness tests run on any CPU in seconds:

```bash
pip install -r requirements.txt
python -m pytest -q
```

On a GPU, the tests also check FP16 and BF16. BF16 runs only on GPUs with native support (Ampere or newer). Recent PyTorch emulates BF16 on a T4, but emulated BF16 is not representative, so the test skips it there.

Experiments, from the repository root:

```bash
python -m scripts.sweep_approx_error       # error vs m (Performer) and k (Linformer); CPU is fine
python -m scripts.memory_sanity            # peak memory vs N; needs a GPU (a free T4 is enough)
python -m scripts.memory_sanity --backward --causal
```

Both write a CSV and a PNG to `results/`. On Colab, open `colab_quickstart.ipynb`.

## Sanity-check results (T4, November 2026)

These validate the implementations against the theory. They are **not** the paper's benchmark results, which come from the shared harness on the A100. Measured on a free Colab T4 (15 GB): 1 batch × 8 heads, d = 64, FP16, forward pass, non-causal. Tests: 58 passed, 1 skipped (causal Linformer, by design).

### Peak memory vs sequence length (`results/memory_sanity.png`)

| N | Naive (MiB) | Rabe–Staats (MiB) | Performer (MiB) | Linformer (MiB) |
| --- | --- | --- | --- | --- |
| 512 | 16.6 | 25.1 | 17.6 | 5.0 |
| 1,024 | 33.0 | 98.1 | 35.2 | 9.5 |
| 2,048 | 130.0 | 195.1 | 70.5 | 18.5 |
| 4,096 | 516.0 | 389.1 | 144.0 | 36.5 |
| 8,192 | 2,056.0 | 395.3 | 282.0 | 72.5 |
| 16,384 | 8,208.0 | 403.3 | 568.0 | 144.5 |
| 32,768 | out of memory | 419.3 | 1,128.0 | 288.5 |

- **Naive is quadratic.** From N = 2,048 on, memory grows 4× per doubling of N. The measured 8,208 MiB at N = 16,384 matches S and P in FP16: 2 × N² × 8 heads × 2 bytes = 8,192 MiB. At N = 32,768 the failed allocation was 17,179,869,184 bytes, exactly 32,768² × 8 × 2 bytes for the score matrix alone, which exceeds the T4's memory.
- **Rabe–Staats is flat above N = 4,096**, at about 400 MiB. Once N exceeds the key chunk (4,096), working memory is set by the chunk sizes, not by N. Below that, one key chunk covers the whole sequence, so chunking saves nothing, and FP32 accumulation makes it use *more* memory than naive. It only wins above N ≈ 3–4K. Smaller chunks move the crossover down at the cost of more, smaller kernel launches.
- **Performer and Linformer are linear.** Memory doubles with N. Linformer is lower because its N × 256 score matrix is FP16. Performer stores N × 266 query and key features in FP32.

### Approximation error vs m and k (`results/approx_error.png`)

N = 1,024, d = 64, relative Frobenius error against FP64 exact attention, mean over 5 seeds; q and k entries drawn with standard deviation 0.25, 0.5 or 1.0.

| Method | q, k std 0.25 | q, k std 0.5 | q, k std 1.0 |
| --- | --- | --- | --- |
| Performer, m = 16 → 1,024 | 0.25 → 0.026 | about 1.1 → 0.25 | about 5 → 3.6 |
| Linformer Gaussian, k = 16 → 1,024 | about 70 → 1.3 | about 140 → 1.3 | about 140 → 1.3 |
| Linformer pooling, k = 16 → 512 | about 0.06, flat | about 0.25, flat | about 0.8, flat |
| Linformer pooling, k = N = 1,024 | about 3e-7 (exact) | about 3e-7 (exact) | about 4e-7 (exact) |

- **Performer** follows the theoretical 1/√m decay only for small-norm inputs (std 0.25). At std 1.0 the error stays around 400% for every m: the estimator's variance grows like exp(‖q + k‖²/√d).
- **Gaussian Linformer** does not approximate attention when untrained. Even at k = N the projection is not the identity, so the error stays above 100%.
- **Pooling Linformer** is exact at k = N, as designed. It is flat below that because the test tokens are independent: averaging two unrelated tokens loses as much as averaging fifty. Real sequences have correlated neighbours and should do better at intermediate k.
- **Conclusion for RQ3:** on random inputs the error depends more on the input scale than on the method. Accuracy comparisons need either a stated input scale or real q, k, v from a pretrained model.

## Implementation notes and trade-offs

**Naive.** Materializes S = QKᵀ/√d and P = softmax(S). `torch.softmax` subtracts the row maximum internally, so it is numerically stable. The causal mask is applied after computing the full matrix, as in a standard implementation, so causal runs do not save work.

**Rabe–Staats.** Each (query chunk, key chunk) pair is reduced to a triple (m, ℓ, o): row max, sum of exponentials, and exponential-weighted sum of values. Triples are merged with the running-max rescaling of online softmax (Lemma 1 in the preliminaries).

- The row max is detached from the autograd graph. It only keeps `exp()` in range; the output does not depend on it, so gradients stay exact.
- Fully masked rows (possible inside a causal chunk) get m = −∞ and contribute ℓ = o = 0, handled without NaNs.
- Causal runs skip key chunks entirely in the future of the query chunk.
- Gradient checkpointing is applied at two levels, as in the paper: per key chunk and per query chunk. Only the output is stored; the backward pass recomputes the rest.
- Default chunks are (1024, 4096), the paper's values. Chunks of about √N give the paper's O(√N) memory regime. Memory only drops below naive attention once N exceeds the key chunk (measured crossover N ≈ 3–4K with the defaults).
- **Not IO-aware.** Each chunk's scores pass through GPU memory between separate PyTorch ops. This is intended: it is the control for isolating the effect of FlashAttention's fused, IO-aware tiling (RQ2). A fused version (for example via `torch.compile`, or xFormers' memory-efficient kernel) would be a second, stronger control.

**Performer.** FAVOR+ with positive orthogonal random features.

- *QR sign correction.* Orthogonal blocks come from the QR decomposition of Gaussian matrices. LAPACK does not make R's diagonal positive, so Q's columns are not uniformly distributed (their first coordinates share a sign), and the kernel estimate becomes biased, by about 10% in our checks. Multiplying Q by sign(diag R) fixes this; `test_performer_kernel_estimate_is_unbiased` guards against regressions.
- *Stabilization.* A per-row constant (queries) and one global constant (keys) are subtracted inside `exp()`. They cancel between numerator and denominator, so the result is unchanged but overflow is avoided. A small `eps = 1e-6` keeps features positive.
- *Causal.* Computed in chunks: exact masked products within a chunk, a running m × d state across chunks. Memory is O(chunk² + md) instead of the O(Nmd) of a naive prefix sum.
- *Default m* = round(d ln d), following the paper's m = Θ(d log d).
- *Sensitivity to input norms.* The estimator's variance grows roughly like exp(‖q + k‖²/√d). Measured error falls as 1/√m for small-norm inputs but stays above 100% at unit variance (see the sanity-check results above). **Always report the input distribution with Performer error numbers.**

**Linformer.** Keys and values are projected along the sequence axis, then attended with naive attention over k projected positions.

- In the paper, the projections E and F are learned. In an untrained micro-benchmark they must be fixed, and that choice determines the error. It does not affect runtime or memory. Two options are implemented:
  - `kind="gaussian"`: i.i.d. N(0, 1/k) entries, the Johnson–Lindenstrauss construction from the paper's analysis. Measured error is above 100% for every k, including k = N. Random, untrained projections do not approximate attention.
  - `kind="pooling"`: each projected position averages a contiguous segment. It is deterministic, exact at k = N, and its error falls as k grows. Its quality depends on how similar neighbouring tokens are.
- Raises `NotImplementedError` for `causal=True`: every projected key mixes all positions, including future ones.
- Projection matrices are sized to a fixed N, so each sequence length needs its own; the registry builds them.

## Decisions still open for the team

1. **Linformer projection** for the error comparison (RQ3): Gaussian (faithful to the paper, but its error is not meaningful untrained), pooling (meaningful, but a variant), or runtime and memory only, with no error comparison.
2. **Input distribution** for error measurements: the q and k scale, or real q, k, v captured from a pretrained model (for example a GPT-2 layer). Performer's numbers depend heavily on this.
3. **Precision**: FP16 vs BF16 for the final runs. All code is dtype-agnostic.
4. **Fused Rabe–Staats** as a second RQ2 control.

## Layout

```
attention_baselines/   common.py, naive.py, rabe_staats.py, performer.py, linformer.py,
                       reference.py (FP64 reference + error metrics), registry.py (get_attention)
tests/                 test_correctness.py
scripts/               sweep_approx_error.py, memory_sanity.py
results/               CSVs and plots from the sanity checks
colab_quickstart.ipynb
```

## References

- Vaswani et al., *Attention Is All You Need*, 2017. arXiv:1706.03762
- Rabe and Staats, *Self-attention Does Not Need O(n²) Memory*, 2021. arXiv:2112.05682
- Milakov and Gimelshein, *Online normalizer calculation for softmax*, 2018. arXiv:1805.02867
- Choromanski et al., *Rethinking Attention with Performers*, ICLR 2021. arXiv:2009.14794
- Wang et al., *Linformer: Self-Attention with Linear Complexity*, 2020. arXiv:2006.04768
