# Primary-study data, A100 session of 2026-11-14

GPU: NVIDIA A100-SXM4-40GB (Lambda Cloud `gpu_1x_a100_sxm4`). Software: PyTorch 2.8.0+cu128, Triton 3.4.0,
flash-attn 2.8.3.post2, driver 570.148.08, CUDA 12.8. Code label on every row: `0f72783+working-tree-20261114-2051-v2`
(commit `0f72783` plus the harness changes committed alongside this data).

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

Rebuild the figures (RQ1, RQ3, RQ4, RQ6) from this folder:

```bash
python -m scripts.plot_rq_figures data/a100-2026-11-14 figures/a100-2026-11-14
```

Not in the repo: the RQ2 data (Triton kernels, official FA1, Nsight Compute HBM traffic) and the raw Nsight
reports, about 3.8 GB in total. They are kept offline by Jason; the RQ2 figure in `figures/a100-2026-11-14/`
was made from them and is skipped by the script when they are absent.
