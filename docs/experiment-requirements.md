# Experiment Requirements: Feasibility, GPU, and Time

Author: Jason
Branch: `jason-research`
Course: COMP 9060 Group Project (BCIT, Fall 2026) — Option B (FlashAttention)

This doc answers two planning questions:
1. Can we actually test all the algorithms (FlashAttention + the baselines)?
2. What GPU type / how much GPU time do we need, and how long will it take?

---

## 1. Can we test everything? Short answer: yes, with one hard constraint

The deciding factor is the **GPU architecture**, not our code. FlashAttention is a
CUDA kernel with strict hardware requirements (verified against Dao-AILab/
flash-attention, Oct 2026):

| Algorithm | Needs | Runs on free tier (T4)? |
| --------- | ----- | ----------------------- |
| Naive attention (baseline) | any CUDA GPU | ✅ yes |
| Rabe–Staats mem-efficient (baseline) | any CUDA GPU (pure PyTorch) | ✅ yes |
| Linformer (baseline) | any CUDA GPU (pure PyTorch) | ✅ yes |
| Performer (baseline) | any CUDA GPU (pure PyTorch) | ✅ yes |
| **FlashAttention-2** (`flash-attn`) | **compute capability ≥ 8.0** (Ampere/Ada/Hopper) | ❌ **no** — T4 is Turing (7.5) |
| FlashAttention-1 (legacy) | Turing/Volta via legacy build | ⚠️ possible but fiddly |
| FlashAttention-3 | Hopper (H100), CUDA ≥ 12.3 | ❌ no |
| FlashAttention-4 | Blackwell (B200) | ❌ no |

**The one hard constraint:** the modern algorithm we are studying
(FlashAttention-2) does **not** run on the T4 GPUs that Google Colab Free and
Kaggle hand out. It requires an **Ampere-or-newer** GPU (compute capability 8.0+).

Practical consequences:
- All 4 baselines run anywhere, including free T4. They're pure-PyTorch or
  materialize-the-matrix implementations.
- To benchmark the actual FlashAttention-2 kernel we need an Ampere+ GPU
  (A100, A10, L4, RTX 3090/4090, or H100).
- `bf16` requires Ampere+; `fp16` works more broadly. Our experiments should use
  `fp16` to keep baselines and FlashAttention on the same dtype where possible.

### Fallback if no Ampere GPU is available
- Use **PyTorch `scaled_dot_product_attention`** with the flash backend
  (`torch.nn.attention.sdpa_kernel`). On supported GPUs this dispatches to a
  FlashAttention kernel without compiling `flash-attn`. On a T4 it falls back to
  the memory-efficient / math backend — still a valid *exact* comparison point,
  but it is **not** the true FlashAttention-2 kernel, which must be disclosed in
  the paper.
- FlashAttention-1 can be built for Turing, but the toolchain is finicky; treat
  as a last resort.

---

## 2. Recommended GPU

A **single modern GPU is enough** — this project does **not** need a cluster. The
experiments are single-GPU micro-benchmarks (one attention layer, sweeping
sequence length). No multi-GPU, no distributed training.

Recommended, in order of preference:

| Option | GPU | VRAM | Why | Rough cost |
| ------ | --- | ---- | --- | ---------- |
| **Best value** | 1x A100 40GB | 40 GB | Big VRAM → push `N` very high before OOM; the "canonical" FlashAttention GPU | ~$1–2 / GPU-hr (cloud spot) |
| Good | 1x A10 / L4 24GB | 24 GB | Ampere/Ada, cheap, plenty for the sweep | ~$0.5–1 / GPU-hr |
| Consumer | RTX 3090 / 4090 | 24 GB | If a teammate owns one — free, Ampere/Ada, fully supported | $0 (owned) |
| Overkill | 1x H100 80GB | 80 GB | Needed only if we also want to show FlashAttention-**3** | ~$2–4 / GPU-hr |

Where to get one:
- A teammate's RTX 3090/4090 desktop (ideal: free, full support).
- Cloud single-GPU instance (Lambda, RunPod, Vast.ai, GCP/AWS spot) — rent an
  A100 or A10 by the hour.
- University/BCIT lab machines, if any have Ampere+ GPUs.
- NOT Colab Free / Kaggle for the FlashAttention-2 runs (T4 won't work); they are
  fine for developing code and running the baselines.

VRAM note: more VRAM is directly useful here, because one of our headline results
is "max sequence length before OOM." A 40–80 GB card lets us show FlashAttention
reaching sequence lengths where naive attention OOMs — the money shot of the paper.

---

## 3. Time estimate

The experiments themselves are **cheap and fast** — this is a micro-benchmark, not
model training. The time cost is mostly setup and writing, not compute.

### Compute time (actual GPU hours)
A single run = one algorithm, one sequence length, forward+backward, repeated for
mean±std. Each measurement is seconds. A full sweep:

- 5 algorithms (FlashAttn + 4 baselines)
- ~8 sequence lengths (512 → 64k, log-spaced)
- ~10 repeats each for statistics
- forward + backward

≈ 5 × 8 × 10 × 2 = 800 short measurements. Even at a generous few seconds each
plus warmup/OOM handling, the **entire sweep runs in weeks on the order of 1–3
GPU-hours**. Add re-runs, debugging, and extra configs (dtype, head dim): budget
**5–10 GPU-hours total**. At ~$1–2/hr that is **under ~$20** of cloud compute, or
free on a teammate's card.

### Wall-clock / human time (the real cost)

| Phase | Effort |
| ----- | ------ |
| Environment setup (CUDA, PyTorch, build/install `flash-attn`) | 0.5–1 day — `flash-attn` compilation is the usual pain point |
| Implement benchmark harness (sweep, timing, memory, OOM capture) | 1–2 days |
| Implement/wire baselines (naive, Rabe–Staats, Linformer, Performer) | 2–3 days |
| Run sweeps + collect data | a few GPU-hours (can run overnight) |
| Plots, tables, statistical analysis | 1–2 days |
| Writing (paper + presentation) | ongoing, per rubric |

**Bottom line:** compute is trivial (hours, not days; single GPU, no cluster). The
schedule is dominated by setup and software engineering, with `flash-attn`
installation being the single most likely time sink.

---

## 4. Concrete recommendation

1. Get access to **one Ampere+ GPU** — ideally a teammate's RTX 3090/4090, else a
   rented A100/A10 for a handful of hours.
2. Develop code + run all 4 baselines anywhere (even free T4).
3. Reserve the Ampere GPU for the FlashAttention-2 runs and the final combined
   sweep.
4. Standardize on `fp16`, fixed head dim, batch size; sweep sequence length `N`.
5. If we also want FlashAttention-3 numbers, we'd need an H100 — treat as optional
   stretch, not required.

### Open items for the team
- [ ] Does any teammate have an Ampere/Ada GPU (RTX 30xx/40xx, A-series)?
- [ ] Confirm cloud budget (even $20 is plenty) if no local GPU.
- [ ] Decide: true `flash-attn` kernel (needs Ampere+) vs. PyTorch SDPA flash
      backend as fallback — and document the choice in the paper.
