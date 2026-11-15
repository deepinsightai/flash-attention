# Study parameters

Every parameter of the benchmark in one place: what is fixed, what is swept, what is measured, and
where each value comes from. Status: **Fixed** (agreed in the proposal or a team decision),
**Default** (current harness or code default, can change), **Open** (needs a team decision).

Sources: proposal Section 4 (Plan & Methodology), professor's feedback, `bench/sweep.py`,
`attention-baselines/attention_baselines/*`, `docs/decisions.md`.

## 1. Research questions

| ID | Question | Study |
| --- | --- | --- |
| RQ1 | Does FlashAttention give lower runtime, higher throughput, and O(N) rather than O(N^2) memory than the baselines? | Primary |
| RQ2 | How much of the advantage comes from IO-aware tiling, isolated against exact non-IO-aware Rabe-Staats? | Primary |
| RQ3 | Does it match approximate methods' efficiency with no approximation error? | Primary |
| RQ4 | Does the advantage grow with N and differ between causal and non-causal masking? | Primary |
| RQ5 | Does each FlashAttention version reproduce its claimed gain over its predecessor across GPU generations? | Lineage |
| RQ6 | What are the practical limits (hardware dependence, alpha-stage FA4, BF16/FP8/FP4 precision, reproducibility), and how much benefit remains over cuDNN/SDPA? | Both |

## 2. Methods

### Same-stack arm (main comparison, Triton)

| Method | Exact | Causal | Implementation | Status |
| --- | --- | --- | --- | --- |
| Naive attention | Yes | Yes | Triton, unfused (QK^T, softmax, PV as separate kernels), fwd + bwd | Fixed (planned) |
| Rabe-Staats | Yes | Yes | Triton, chunked, not fused, fwd + bwd | Fixed (planned) |
| Tiled FlashAttention (ours) | Yes | Yes | Triton, adapted from the Triton fused-attention tutorial (cite it), fwd + bwd | Fixed (planned) |

### Approximate methods

| Method | Causal | Implementation | Parameters | Status |
| --- | --- | --- | --- | --- |
| Performer (FAVOR+) | Yes (chunked) | PyTorch | `num_features` m = round(d ln d) = 621 for d = 128; `causal_chunk` = 128; orthogonal Gaussian features, drawn once per seed | Default; Triton vs PyTorch is Open |
| Linformer | **No** | PyTorch | `proj_dim` k = min(256, N); Gaussian projection, drawn once per seed | Default |

### Reference arm (secondary)

| Method | Registry name | Notes |
| --- | --- | --- |
| Official FlashAttention 2 (`flash-attn`) | `flash_attn` | Ampere and newer; library version to be pinned |
| PyTorch SDPA, Flash backend | `sdpa_flash` | PyTorch's built-in FA2-style kernel; not available on Windows builds |
| PyTorch SDPA, memory-efficient backend | `sdpa_efficient` | xFormers-style kernel |
| PyTorch SDPA, math backend | `sdpa_math` | Unfused reference path |
| FlashAttention 3 | - | Lineage study, Hopper only |
| FlashAttention 4 | - | Lineage study, Blackwell; support on RTX PRO 6000 (sm_120) untested |

### Existing PyTorch baselines (Shaaz)

`naive`, `rabe_staats` (PyTorch, `query_chunk` = 1024, `key_chunk` = 4096 as in the paper, gradient
checkpointing on) are registered in the harness today. They serve as correctness references and
sanity checks; the Triton versions replace them in the main comparison.

## 3. Swept parameters

| Parameter | Values | Status |
| --- | --- | --- |
| Sequence length N | Powers of 2 from 512 to 65,536 (64K), extended until out-of-memory | Fixed (proposal) |
| Mask | `noncausal`, `causal` | Fixed |
| Pass type | `forward`, `forward_backward` | Fixed |
| Method | All methods in Section 2 that run on the GPU | Fixed |
| GPU | One per study run; see Section 6 | Fixed per study |

Harness: `--log2n-min 9 --log2n-max 16 --mask both --pass-type both`.

## 4. Fixed parameters

| Parameter | Value | Status |
| --- | --- | --- |
| Precision | BF16 inputs; FP32 accumulation inside kernels | Fixed (proposal) |
| Head dimension d | 128 | Fixed (proposal) |
| Batch size | 1 | Default (proposal says fixed, value not stated) |
| Heads | 8 | Default |
| Self-attention | Nq = Nk = N | Fixed |
| Softmax scale | 1 / sqrt(d) | Fixed |
| Inputs | i.i.d. standard normal q, k, v, generated on the GPU from a seeded generator | Default |
| Seed | 0 | Default |
| Dropout, ALiBi, sliding window, GQA/MQA | Not used | Fixed (out of scope) |

Input scale matters for the approximate methods: Performer's error grows roughly like
exp(||q + k||^2 / sqrt(d)), so on unit-variance inputs its error is large for any practical m.
Report the input distribution with every Performer or Linformer error figure (**Open**: whether
to add a scaled-input setting for RQ3).

## 5. Measurement protocol

| Item | Value | Status |
| --- | --- | --- |
| Warm-up iterations | 3 per configuration | Default |
| Timed repeats | 10 per configuration (at least ten) | Fixed (proposal) |
| Timer | CUDA events around one call, with synchronize before and after | Fixed |
| Forward+backward timing | One call = forward plus `out.backward(grad)` with a fixed random gradient | Fixed |
| Peak memory | `max_memory_allocated` above the live inputs, reset before each repeat | Fixed |
| OOM handling | Logged to `oom.csv`, no runtime recorded; larger N for that method, mask and pass type skipped | Fixed |
| Numerical error | Relative Frobenius and max absolute error vs the FP64 reference, for N <= 4096 | Default (`--error-max-n`) |
| Resume | Configurations already in `runs.csv` are skipped on rerun | Fixed |
| Near-OOM timings | Unreliable (seen on the 5080: naive fwd+bwd 28 ms at 8K, 2,592 ms at 16K) | **Open**: flag rows above ~80% of GPU memory |

## 6. Hardware

| Study | GPU | Source | Status |
| --- | --- | --- | --- |
| Primary (RQ1-RQ4) | NVIDIA A100 (record 40 GB or 80 GB) | Colab Pro+, fallback Lambda 1x A100 | Fixed GPU type; provider Open |
| Lineage FA1/FA2 | A100 | Same as primary | Fixed |
| Lineage FA3 | H100 | Not on Colab so far; AWS `p5.4xlarge` spot or Lambda | Open |
| Lineage FA4 | Blackwell data-center (B200, sm_100) or RTX PRO 6000 (sm_120) | Colab G4 / AWS `g7e`; B200 on Lambda | Open |
| Development only | RTX 5080 16 GB (sm_120), WSL2 | Jason's PC | Not reported in the paper |

Rule: results are compared only within one GPU model and memory size. Never pool A100 40 GB with
80 GB, or Ampere with Blackwell.

## 7. Software (record per run; pin per GPU)

Recorded on every CSV row: GPU name and memory, compute capability, SM count, driver, CUDA,
PyTorch, Triton, git commit, timestamp. To pin per machine: Python, PyTorch, Triton, CUDA,
`flash-attn` version (**Open** until the A100 environment is set up).

## 8. Metrics and outputs

Per configuration (median and IQR over the repeats unless noted):

- Latency, forward and forward+backward (ms)
- Throughput (tokens/s = batch x N / latency)
- Peak extra GPU memory (MB)
- Maximum N before out-of-memory, per method, mask, pass type, GPU
- Numerical error vs FP64 reference (relative Frobenius, max absolute)
- Speedup vs naive and vs Rabe-Staats at matched configuration
- Empirical scaling exponent: slope of log(runtime) and log(memory) vs log(N)

Files: `results/<gpu>/runs.csv` (one row per repeat), `oom.csv`, `failures.csv`; analysis by
`analysis/benchmark_analysis.py`. Column definitions: `docs/interfaces.md`.

## 9. Validation criteria

| Check | Threshold | Status |
| --- | --- | --- |
| Exact methods vs FP64 reference, BF16 | max abs error around BF16 rounding (< 0.05 at N = 1024 on unit-normal inputs) | Default (`scripts/cross_check.py`) |
| Gradients (dq, dk, dv) vs FP64 | Tolerance per dtype | **Open** (Aneet) |
| Causal masking | Changing future tokens does not change earlier outputs | Fixed |
| A method enters the timed study | Only after passing the correctness tests | Fixed |

## 10. Scope limits

- One attention layer, single GPU. No full-model training or inference, no multi-GPU.
- Linformer has no causal mode; it appears only in non-causal results.
- The paper's page limit includes appendices.
- Cross-generation comparisons (RQ5) mix hardware and algorithm effects and are framed per generation.

## 11. Open decisions affecting parameters

1. Batch size and head count: confirm 1 and 8, or choose values that keep small-N runs GPU-bound.
2. Performer and Linformer: PyTorch with a stated caveat, or Triton.
3. A scaled-input setting for RQ3 accuracy.
4. Near-OOM flag threshold.
5. Gradient tolerances.
6. Primary-study provider (Colab Pro+ or Lambda) and A100 memory size.
7. FA3/FA4 hardware, or drop them as a stated limitation.
