# RQ figures from the A100 session of 2026-11-14

Made by `python -m scripts.plot_rq_figures [DATA_ROOT] [OUT_DIR]` (pandas, matplotlib, numpy).
GPU: A100-SXM4-40GB, PyTorch 2.8, flash-attn 2.8.3, Triton 3.4. Written explanations: `remarks.md`.

| Figure | Question | Data |
| --- | --- | --- |
| `rq1_speed_memory.png` | RQ1: runtime, peak memory and largest N, FlashAttention against the baselines | `data/a100-2026-11-14` (primary study) |
| `rq2_io_vs_engineering.png` | RQ2: I/O awareness against engineering, with naive, Rabe-Staats, FA1 and FA2 all in Triton, against official FA1/FA2, plus measured HBM traffic | `data/a100-2026-11-14` (RQ2 sweeps at batch 1, 128 heads, and Nsight Compute summaries) |
| `rq3_exact_vs_approximate.png` | RQ3: FlashAttention against Performer and Linformer: speed and error | `data/a100-2026-11-14` |
| `rq4_scaling_and_masking.png` | RQ4: speedup against N, causal against non-causal, FA2 throughput | `data/a100-2026-11-14` |
| `rq6_sdpa_vs_flash_attn.png` | RQ6 (partial): PyTorch SDPA backends against official FA2 | `data/a100-2026-11-14` |

All five figures rebuild byte-identical from `data/a100-2026-11-14`.
