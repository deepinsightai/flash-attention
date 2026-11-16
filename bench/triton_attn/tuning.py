"""Load and save tuned launch configurations for the Triton kernels.

Configurations are stored per GPU model in configs/tuning/<gpu-slug>.json (written by
scripts/tune_triton.py) and applied to _util.TUNED. Without a file the kernels use their original
fixed defaults, so behaviour is unchanged until a tuning run exists for the GPU in use.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ._util import TUNED

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "tuning"


def gpu_slug() -> str:
    from bench.metadata import gpu_label, gpu_slug as _slug

    return _slug(gpu_label(0))


def config_path(slug: str | None = None) -> Path:
    return CONFIG_DIR / f"{slug or gpu_slug()}.json"


def to_json(tuned: dict) -> dict:
    return {
        "flash": [[v, p, d, dt, list(cfg)] for (v, p, d, dt), cfg in tuned["flash"].items()],
        "bmm": {k: list(v) for k, v in tuned["bmm"].items()},
        "softmax": {k: list(v) for k, v in tuned["softmax"].items()},
        "rs_elem": {k: list(v) for k, v in tuned["rs_elem"].items()},
    }


def from_json(data: dict) -> dict:
    return {
        "flash": {(v, p, d, dt): tuple(cfg) for v, p, d, dt, cfg in data.get("flash", [])},
        "bmm": {k: tuple(v) for k, v in data.get("bmm", {}).items()},
        "softmax": {k: tuple(v) for k, v in data.get("softmax", {}).items()},
        "rs_elem": {k: tuple(v) for k, v in data.get("rs_elem", {}).items()},
    }


def apply(tuned: dict) -> None:
    for key in TUNED:
        TUNED[key].clear()
        TUNED[key].update(tuned.get(key, {}))


def load(path: Path | None = None) -> Path | None:
    """Apply the tuning file for the current GPU (or `path`). Returns the file used, or None."""
    if not torch.cuda.is_available():
        return None
    path = path or config_path()
    if not path.exists():
        return None
    apply(from_json(json.loads(path.read_text(encoding="utf-8"))["tuned"]))
    return path


def save(tuned: dict, meta: dict, log: list, path: Path | None = None) -> Path:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "tuned": to_json(tuned), "search_log": log}, indent=1),
                    encoding="utf-8")
    return path
