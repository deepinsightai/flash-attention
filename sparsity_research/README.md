# Preliminary attention sparsity study

COMP 9060 — individual contribution by Aneet Kaur. CPU pilot recorded November 24, 2026. This folder is a standalone handoff for the team, not a modification of its existing kernels or harness.

## Method

Dense scaled dot-product attention, synthetic Q/K/V, batch 1, one head, dimension 64, non-causal, FP32. Inverted dropout is applied after softmax with a fixed Bernoulli mask per seed and probability: retained weights are divided by (1-p). Masks and inputs are generated outside timing; p=1 is excluded. Lengths: 128, 256, 512, 1024. Probabilities: 0, .1, .25, .5, .75, .9. Seeds: 42–46. Three warmups and ten timed repetitions per configuration; randomized probability order within seed. Wall-clock timing includes dispatch and CUDA synchronization when applicable. Recorded environment: Python 3.12.10, PyTorch 2.15.0+cpu, two CPU threads during benchmark.

## Results and interpretation

120 seed configurations, 1,200 timings, 24 aggregated configurations. Latency is the median of seed medians; IQR is Q3 minus Q1 across seed medians. Speedup is the median of paired seed-level ratios (zero-dropout latency / current latency), not a ratio of aggregate medians. Throughput is the median of seed-level input tokens per second.

| Length | p=0 latency (ms) | p=.9 latency (ms) | Paired speedup |
|---:|---:|---:|---:|
| 128 | 0.13055 | 0.16535 | 0.7850 |
| 256 | 0.33530 | 0.39435 | 0.8529 |
| 512 | 1.15335 | 1.30090 | 0.8819 |
| 1024 | 5.02630 | 5.80905 | 0.8653 |

RQ1/RQ4: dropout did not accelerate the measured dense CPU implementation; zeroing entries does not skip dense arithmetic. RQ3: matched FP64 error measures numerical agreement using the same mask; deviation from unmasked FP64 attention measures dropout-induced change, not Linformer/Performer approximation error. These are descriptive pilot results, not a significance test or GPU performance claim.

## Files and reproduction

- `Sparsity_Attention_Research.ipynb`: original executed extended notebook, preserved with outputs. Run a copy in a separate working directory because it writes CSVs to the working directory.
- `results/cpu/`: original recorded CSVs, unchanged except filenames.
- `Attention_Sparsity_Research_Findings.docx`: preliminary report.
- `run_pilot.py`: standalone adaptation of the notebook benchmark with explicit device and separate output directory.
- `verify_results.py`: checks archived CSV aggregation, not tensor-level correctness.

Install dependencies in a suitable environment (`python -m pip install -r requirements.txt`; for CUDA use a compatible CUDA-enabled PyTorch build). From this folder:

```sh
python verify_results.py
python run_pilot.py --device cpu --output results/new_cpu_run
python run_pilot.py --device cuda --output results/new_gpu_run
```

Output directories must not already exist. The CUDA command is a proposed dense pilot rerun, not a FlashAttention kernel benchmark. GPU timing and numerical results may differ. Keep hardware-specific results separate. The standalone script has not been benchmarked on an A100.

## Handoff and limitations

The team still needs to agree on dropout semantics for each algorithm, causal setting, shapes, precision and timing protocol before integrating with its existing harness. Linformer, Performer, Rabe-Staats and FlashAttention 1/2 were not tested here. No A100 results are included. since Jason you requested the existing code be committed and indicated that you would handle the rest. This folder contains only the pilot contribution, not teammates' branch archives or earlier validation PR files.
