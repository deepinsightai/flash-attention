# Project decisions

Record each decision with its date and reason so the paper's methodology and limitations sections can cite it.

## Settled

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-11 | All exact methods (naive, Rabe-Staats, tiled FA) are written in Triton; official CUDA FlashAttention is a separate reference arm. | Professor's feedback: PyTorch Rabe-Staats vs official CUDA FA mixes I/O-awareness with engineering quality. Cite the Triton fused-attention tutorial. |
| 2026-11 | Results are compared only within the same GPU model and memory size; the GPU is logged on every row. | Colab assigns different GPUs on different days. |
| 2026-11 | CSV schema and method signature as in `docs/interfaces.md`. | Matches `analysis/benchmark_analysis.py` and the baselines' registry. |
| 2026-11 | OOM runs are logged separately and never given a runtime. | Analysis script rejects missing runtimes; avoids invented data. |

## Open

| Question | Options | Owner |
| --- | --- | --- |
| Who writes the Triton FA forward and backward? | Bahaa (forward), Shaaz (backward) proposed | team |
| Hardware scope for the lineage study (FA3, FA4) | A100 guaranteed; H100/B200 conditional on Colab, AWS, or Lambda access | Bahaa |
| Performer and Linformer in Triton or PyTorch? | PyTorch with a stated caveat is proposed | team |
| Cut order if time runs short | Triton backward for naive/Rabe-Staats first, then FA3/FA4 | team |
| Primary-study GPU | Colab Pro+ A100 (40 or 80 GB, record which) | Jason |

## Known limitations to state in the paper

- Linformer has no causal mode.
- Page limit includes appendices.
- `sdpa_flash` is unavailable on the Windows PyTorch build; run it on Linux or Colab.
- FP64 reference error is computed only up to N = 4096 by default.
