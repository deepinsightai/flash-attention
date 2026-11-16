# Interfaces: method signature and benchmark CSV

Agreed contract between the method authors, the harness (`bench/`), validation, and analysis
(`analysis/benchmark_analysis.py`). Change it only by editing this file and telling everyone.

## Method signature

```python
fn(q, k, v, causal=False, scale=None) -> out
# q: [B, H, Nq, d]   k: [B, H, Nk, d]   v: [B, H, Nk, dv]   out: [B, H, Nq, dv]
```

- Self-attention only for the benchmark (Nq == Nk).
- Output dtype equals input dtype. Accumulate in FP32 internally.
- Must support autograd, so that `out.backward(grad)` works for the forward+backward runs. A method
  without a backward is benchmarked forward-only, and the paper must say so.
- Methods with random parameters (Performer features, Linformer projections) draw them once in the
  builder, never inside the timed call.
- Linformer has no causal mode; the harness skips it for causal runs. The paper must say so.

### Adding a method (Triton kernels, etc.)

```python
# bench/methods.py
from bench import methods
methods.register("flash_triton", lambda seq_len, head_dim, device, seed: my_fn, supports_causal=True)
```

For a Triton kernel with a hand-written backward, wrap it in a `torch.autograd.Function` so that
`out.backward(grad)` works. The FA forward must save what the backward needs (the per-row
logsumexp); Bahaa (forward) and Shaaz (backward) agree that interface in the kernel file's docstring.

## CSV output

One directory per GPU model: `results/<gpu-slug>/`. GPU types are never mixed in one file.

| File | Content |
| --- | --- |
| `runs.csv` | one row per repeat of a configuration that finished |
| `oom.csv` | configurations that ran out of memory (no runtime is invented) |
| `failures.csv` | configurations that raised any other error, with the message |

### `runs.csv` columns

Required by `analysis/benchmark_analysis.py` (first 11 columns, in this order):

| Column | Values |
| --- | --- |
| method | registry name, e.g. `naive`, `rabe_staats`, `sdpa_flash`, `flash_attn` |
| gpu | model plus memory, e.g. `NVIDIA A100-SXM4-40GB 40GB` |
| precision | `BF16`, `FP16`, `FP32` |
| mask | `causal` or `noncausal` |
| pass_type | `forward` or `forward_backward` |
| N | sequence length |
| batch_size, head_dim | fixed per study |
| repeat | 0-based repeat index |
| runtime_ms | CUDA-event time of one call (forward, or forward + backward) |
| peak_memory_mb | peak allocated memory above the live inputs during that call |

Extra metadata: `heads`, `numerical_error` (relative Frobenius error vs the FP64 reference),
`max_abs_error`, `gpu_mem_gb`, `compute_capability`, `sm_count`, `torch_version`, `cuda_version`,
`triton_version`, `driver_version`, `git_commit`, `timestamp_utc`, `seed`, `warmup`.

Error columns are filled only for N <= `--error-max-n` (default 4096), because the FP64 reference
materializes an N x N matrix. They are blank above that, never zero.

## Measurement rules

- Warm-up iterations first, then `--repeats` timed repeats (use 10 or more for reported results).
- Each repeat: synchronize, reset peak-memory stats, CUDA-event start/stop, synchronize.
- A sweep stops a method at its first OOM for a given mask and pass type and logs it to `oom.csv`.
- Re-running a command resumes: configurations already in `runs.csv` are skipped.
- Compare results only within one `gpu` value.

## Running

```bash
python -m bench.sweep --methods naive rabe_staats performer linformer sdpa_flash \
    --log2n-min 9 --log2n-max 16 --mask both --pass-type both --repeats 10 --check-error
python -m analysis.benchmark_analysis results/<gpu-slug>/runs.csv --output results/<gpu-slug>/analysis
```

## Triton methods (ench/triton_attn/)

Registered automatically when Triton imports (Linux, WSL, Colab). Triton does not run on the
Windows setup here: Application Control blocks its compiled helper module.

| Registry name | What it is |
| --- | --- |
| 	riton_naive | Three unfused kernels; the N x N scores and probabilities live in HBM |
| 	riton_rabe_staats | Chunked exact attention, one kernel per step, not IO-aware |
| 	riton_fa1 | FlashAttention: grid over batch*heads, O/m/l round-trip through HBM per block pair |
| 	riton_fa2 | FlashAttention-2: Q-block parallel, deferred normalization, causal block skipping |
| 	riton_fa3 | FA2 plus FA3's pipelined softmax/matmul order. No TMA, warp specialization or FP8 |
| 	riton_fa4 | FA2 plus FA4's software exp2 (half of each tile) and conditional rescaling. No TMEM or 2-CTA |

Constraints: self-attention only, dv == d, d a power of two in [16, 256] for the fused kernels.
The forward saves L = m + log2(l) (base-2 logsumexp of scale * log2(e) * q.k); the backward
recomputes P = exp2(s - L). FA3 and FA4 share the FA2 backward. Tests: 	ests/test_triton_attn.py.
