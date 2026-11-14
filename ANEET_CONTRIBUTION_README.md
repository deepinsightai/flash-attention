# Aneet — numerical validation and benchmark analysis

## Scope
Starter contribution for the approved COMP 9060 proposal, **A Reproducibility Benchmark of the FlashAttention Family Against Classical and Approximate Attention**. This is **not** a completed experiment or a verified result for the team.

- `validation/reference.py`: independent FP64 attention reference for small self-attention inputs, causal masking, numerical error metrics.
- `analysis/benchmark_analysis.py`: CSV ingestion, median and IQR, matched speedups, log-log runtime slopes, runtime plots.
- `tests/`: basic unit tests.
- `examples/example_runs.csv`: **synthetic demonstration data only**, not experimental results.

## Setup (from this folder)
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m analysis.benchmark_analysis examples/example_runs.csv --output results/demo
```

## Real benchmark data
Ask Jason for a CSV containing **one row per repeat** and the following exact columns:
`method,gpu,precision,mask,pass_type,N,batch_size,head_dim,repeat,runtime_ms,peak_memory_mb`

Optional additional columns can include CUDA/PyTorch/kernel versions, error metrics and throughput. All methods must be compared on the same hardware and compatible settings. The script refuses duplicate run identifiers and calculates speedup only where the baseline has matching configuration keys.

**Important:** The current analysis uses median *reported* peak memory, not a memory-allocation experiment. OOM runs must be separately logged (do not give them invented runtimes). An independent FP64 reference may be too memory intensive for large N; validate at small N. Runtime slope estimates are empirical fits, not proofs of asymptotic complexity.

## Limitations and next steps
- Agree with teammates on exact schema and naming (e.g., `naive`, `flashattention`).
- Integrate the reference validator with actual baseline and FlashAttention outputs.
- Add tests for gradient correctness, head/batch shapes, masking, approximation errors, and selected numerical tolerances.
- Add memory figures, throughput, maximum successful N/OOM reports, and interpretation of RQ1–RQ6 **after real results arrive**.
- Check professor's multiple-datasets criterion and hardware feasibility with the team.
- Avoid treating example data as real research findings.

## AI Use
The initial code and README in this folder were drafted with assistance from ChatGPT. Functions and tests have `AI-assisted` comments. Review, understand, test, and edit them before committing. Record any subsequent AI use as required by the course instructions. The research interpretation and conclusions must be written by the student from real results.

## Upload to the team repository
Upload the contents of this package into the **root** of `aneet-validation-analysis`.
Do not replace the team's existing `README.md` or `.gitignore`.
`ANEET_CONTRIBUTION_README.md` is intentionally separate.
The `examples/` CSV is synthetic, and `results/` should remain ignored.
