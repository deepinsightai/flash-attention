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
    """Name plus memory size, e.g. 'NVIDIA A100-SXM4-40GB 40GB'. Separates 40 GB from 80 GB A100s."""
    props = torch.cuda.get_device_properties(device_index)
    gb = round(props.total_memory / 2**30)
    return f"{props.name} {gb}GB"


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
        "git_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
    }


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
