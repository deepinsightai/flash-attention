"""CSV export. The first block of columns is exactly what analysis/benchmark_analysis.py requires;
the rest are extra metadata. OOM and failed configurations go to separate files because the
analysis script rejects rows with missing runtimes."""

from __future__ import annotations

import csv
from pathlib import Path

REQUIRED_COLUMNS = [
    "method", "gpu", "precision", "mask", "pass_type", "N", "batch_size", "head_dim",
    "repeat", "runtime_ms", "peak_memory_mb",
]
EXTRA_COLUMNS = [
    "heads", "numerical_error", "max_abs_error", "gpu_mem_gb", "compute_capability", "sm_count",
    "torch_version", "cuda_version", "triton_version", "driver_version", "git_commit",
    "timestamp_utc", "seed", "warmup",
]
RUN_COLUMNS = REQUIRED_COLUMNS + EXTRA_COLUMNS
OOM_COLUMNS = ["method", "gpu", "precision", "mask", "pass_type", "N", "batch_size", "head_dim", "heads",
                "timestamp_utc"]
FAIL_COLUMNS = OOM_COLUMNS + ["error"]

CONFIG_KEYS = ["method", "gpu", "precision", "mask", "pass_type", "N", "batch_size", "head_dim"]


def append_rows(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def completed_configs(path: Path) -> set[tuple]:
    """Configurations already in a runs CSV, so an interrupted sweep (e.g. a dropped Colab
    session) can resume without producing duplicate rows."""
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {tuple(str(row[k]) for k in CONFIG_KEYS) for row in csv.DictReader(f)}


def config_id(row: dict) -> tuple:
    return tuple(str(row[k]) for k in CONFIG_KEYS)
