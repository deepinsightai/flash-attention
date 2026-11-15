# Benchmark input audit — follow-up contribution

This optional check catches malformed benchmark records before computing medians, speedups, or scaling fits. It complements the existing validation and analysis code; it does not replace those modules or claim to reproduce any GPU measurements.

## Files

- `analysis/benchmark_audit.py`: standard-library CSV validation and per-method row counts.
- `tests/test_benchmark_audit.py`: small regression tests for valid and invalid records.

## Run from the repository root

```bash
python -m unittest discover -s tests -p 'test_benchmark_audit.py' -v
python -m analysis.benchmark_audit path/to/runs.csv
```

The second command exits with status 0 when all required fields are valid, 1 for invalid rows, and 2 for missing required columns or unreadable input.

## Integration note

**The input schema is provisional.** This script currently expects `method`, `N`, and `runtime_ms`. Before using it with Jason's timing harness, confirm its actual column names and units, and adjust the script if necessary. Additional fields such as GPU model, dtype, causal flag, pass direction, peak memory, repetition index, and error should be retained in the original benchmark output for stratified comparisons. Do not combine results from different GPUs, precisions, or forward/backward configurations into one performance claim.

No experimental results or measured speedups are included in this update.
