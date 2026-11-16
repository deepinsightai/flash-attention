# A100 session data, 2026-11-14 (primary study and RQ2)

GPU: NVIDIA A100-SXM4-40GB (Lambda Cloud `gpu_1x_a100_sxm4`). Software: PyTorch 2.8.0+cu128, Triton 3.4.0,
flash-attn 2.8.3.post2, driver 570.148.08, CUDA 12.8. Code label on every row: `0f72783+working-tree-20261114-2051-v2`
(commit `0f72783` plus the harness changes committed alongside this data).

## Primary study

Configuration: batch 1, 8 heads, d = 128, BF16, N = 512 to 65,536, causal and non-causal, forward and
forward+backward, 3 warm-up calls, 10 timed repeats, seed 0. Methods: `naive`, `rabe_staats`, `performer`,
`linformer` (PyTorch baselines), `sdpa_flash`, `sdpa_efficient`, `sdpa_math` (PyTorch SDPA), `flash_attn`
(official FlashAttention-2). Column definitions: `docs/interfaces.md`.

| File | Rows / content | SHA-256 |
| --- | --- | --- |
| `results_primary/nvidia-a100-sxm4-40gb-40gb/runs.csv` | One row per timed repeat (2,250 runs) | 6454e7973926296642f326af111bb9b89113552bbca2f254c449fd28b98af3c1 |
| `results_primary/nvidia-a100-sxm4-40gb-40gb/oom.csv` | Configurations that ran out of memory (no runtime recorded) | 2b49fba05240ad90f6cb60959f09e370643945306611b1f83e124dbf7e9b32b6 |
| `results_primary_performer_split/nvidia-a100-sxm4-40gb-40gb/runs.csv` | Performer, causal, forward+backward, rerun after the `torch.split` fix in `performer.py` | 03cd9d88ebdbf8619ea7f8254a914a8d1969bd7dc2f8cb098c709e22ae8c93e8 |

Use the second file in place of the Performer causal forward+backward rows of the first; those were timed
with the slow slicing version (same outputs, about 21x slower at 64K). `scripts/plot_rq_figures.py` does this.

Command that produced `results_primary`:

```bash
python -m bench.sweep --methods naive rabe_staats performer linformer sdpa_flash sdpa_efficient sdpa_math flash_attn \
  --batch 1 --heads 8 --head-dim 128 --log2n-min 9 --log2n-max 16 --mask both --pass-type both \
  --repeats 10 --warmup 3 --dtype bf16 --check-error --error-max-n 4096 --out-dir results_primary
```

The Performer rerun used the same arguments with `--methods performer --mask causal --pass-type forward_backward`.

## RQ2 data (same GPU and session)

Configuration: batch 1, 128 heads (so every kernel fills the GPU), d = 128, BF16, N = 1K to 16K, both masks,
10 timed repeats. Naive, Rabe-Staats, FA1 and FA2 are written in Triton (`triton_*`) and compared with official
FA1 (`flash_attn_v1`, flash-attn 1.0.9, separate venv) and official FA2 (`flash_attn`).

| File | Content | SHA-256 |
| --- | --- | --- |
| `results_rq2_v2/.../runs.csv`, `oom.csv` | Triton naive, Rabe-Staats, FA1, FA2 and official FA2, with fixed tuning; BF16 and FP16 (FP16 vs the Triton tutorial). About 70% of the planned grid. | 86d003fe24a2d702..., 6d44b948a1126d98... |
| `results_rq2_newbwd/.../runs.csv` | Forward+backward of `triton_fa2` (rewritten dK/dV + dQ backward), `triton_fa1`, `flash_attn`, both masks, N = 1K to 16K. Replaces those methods' forward+backward rows in `results_rq2_v2`. | a8e37431f76964ca... |
| `results_rq2_fa1venv/.../runs.csv`, `failures.csv` | Official FA1. Forward only is valid; its forward+backward failed (see `failures.csv`). | 58b73b7ed9b5603f..., bd5b8ac12ac58956... |
| `results_ncu/ncu_summary.csv` | Nsight Compute HBM bytes (`dram__bytes_read.sum` + `dram__bytes_write.sum`) per method, 128 heads, N = 2K and 8K, both masks, forward and forward+backward | dd9ca48f6962ad54... |
| `closeout/ncu/ncu_summary_N*.txt` | Nsight Compute HBM bytes per kernel, 8 heads, causal, N = 1K to 16K, forward and backward. Lines starting `TOTAL` are parsed by the plot script. | see `sha256sum` |

Not included: superseded runs (first tuning had an early-stop bug; old atomic FA2 backward; first Nsight
run counted gradient accumulation), raw `.ncu-rep` reports (about 345 MB) and logs. They stay in Jason's
offline session archive. The Triton kernels and session scripts that produced the RQ2 data are not in this
commit either.

## Rebuild the figures

All five figures (RQ1, RQ2, RQ3, RQ4, RQ6) rebuild byte-identical from this folder:

```bash
python -m scripts.plot_rq_figures data/a100-2026-11-14 figures/a100-2026-11-14
```
