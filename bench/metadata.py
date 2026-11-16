"""Per-run environment metadata. Every CSV row carries the GPU model so results are only
compared within one GPU type (professor's requirement)."""

from __future__ import annotations

import datetime as _dt
import re
import subprocess

import torch


def _run(cmd: list[str], cwd: str | None = None) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, cwd=cwd).strip()
    except Exception:
        return ""


def gpu_label(device_index: int = 0) -> str:
    """Name plus memory size, e.g. 'NVIDIA A100-SXM4-40GB 40GB'. Separates 40 GB from 80 GB A100s.

    Uses nvidia-smi's memory.total (MiB), which matches the marketed size (40960 MiB -> 40GB);
    PyTorch's usable total (39.5 GiB on an A100 40GB) is the fallback.
    """
    props = torch.cuda.get_device_properties(device_index)
    mib = _run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", str(device_index)])
    try:
        gb = round(float(mib.splitlines()[0]) / 1024)
    except (ValueError, IndexError):
        gb = round(props.total_memory / 2**30)
    return f"{props.name} {gb}GB"


def code_version() -> str:
    """git commit of the working tree, else the COMMIT file shipped with an upload package."""
    commit = _run(["git", "rev-parse", "--short", "HEAD"])
    if commit:
        return commit
    from pathlib import Path

    f = Path(__file__).resolve().parents[1] / "COMMIT"
    return f.read_text(encoding="utf-8").strip() if f.exists() else ""


def gpu_slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-").lower()


def collect(device_index: int = 0) -> dict:
    props = torch.cuda.get_device_properties(device_index)
    try:
        import triton

        triton_version = triton.__version__
    except Exception:
        triton_version = ""
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", str(device_index)])
    return {
        "gpu": gpu_label(device_index),
        "gpu_mem_gb": round(props.total_memory / 2**30, 2),
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "triton_version": triton_version,
        "driver_version": driver.splitlines()[0] if driver else "",
        "git_commit": code_version(),
    }


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
