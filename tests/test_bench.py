"""Harness tests. The schema test runs anywhere; the sweep tests need a CUDA GPU."""

import pandas as pd
import pytest
import torch

from analysis.benchmark_analysis import REQUIRED, load_runs
from bench import export, methods

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_run_columns_cover_analysis_requirements():
    assert set(REQUIRED) <= set(export.RUN_COLUMNS)
    assert export.RUN_COLUMNS[: len(export.REQUIRED_COLUMNS)] == export.REQUIRED_COLUMNS


def test_registry_has_baselines_and_reference_arms():
    for name in ("naive", "rabe_staats", "performer", "linformer", "sdpa_flash", "flash_attn"):
        assert name in methods.METHODS
    assert not methods.supports_causal("linformer")


def test_register_custom_method():
    methods.register("dummy", lambda seq_len, head_dim, device, seed: (lambda q, k, v, causal=False: q), True)
    assert "dummy" in methods._REGISTRY
    assert methods.supports_causal("dummy")


@cuda
def test_sweep_writes_analysis_compatible_csv_and_resumes(tmp_path):
    from bench import sweep

    argv = ["--methods", "naive", "--n", "256", "512", "--repeats", "3", "--warmup", "1",
            "--mask", "both", "--pass-type", "both", "--check-error", "--out-dir", str(tmp_path)]
    sweep.main(argv)
    (runs,) = list(tmp_path.glob("*/runs.csv"))
    df = load_runs(runs)  # raises if the analysis script would reject the file
    assert len(df) == 2 * 2 * 2 * 3  # N x mask x pass x repeats
    assert df.gpu.nunique() == 1
    assert (df.runtime_ms > 0).all() and (df.peak_memory_mb >= 0).all()
    assert pd.to_numeric(df.numerical_error).notna().all()
    sweep.main(argv)  # resume: nothing new should be appended
    assert len(load_runs(runs)) == len(df)


@cuda
def test_bnhd_layout_inputs_are_views_of_contiguous_storage(tmp_path):
    # A method registered with layout="bnhd" must receive [B, H, N, d] views of
    # contiguous [B, N, H, d] storage, and forward+backward must run through the sweep.
    from bench import sweep

    seen = []

    def builder(seq_len, head_dim, device, seed):
        def fn(q, k, v, causal=False, scale=None):
            seen.append(q.transpose(1, 2).is_contiguous() and not q.is_contiguous())
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return fn

    methods.register("dummy_bnhd", builder, True, layout="bnhd")
    q, k, v, leaves = methods.make_qkv("dummy_bnhd", (1, 4, 256, 64), torch.float32, "cuda", requires_grad=True)
    assert q.shape == (1, 4, 256, 64) and leaves[0].shape == (1, 256, 4, 64) and leaves[0].is_leaf
    sweep.main(["--methods", "dummy_bnhd", "--n", "256", "--heads", "4", "--head-dim", "64", "--dtype", "fp32",
                "--repeats", "2", "--warmup", "1", "--pass-type", "both", "--out-dir", str(tmp_path)])
    (runs,) = list(tmp_path.glob("*/runs.csv"))
    assert len(load_runs(runs)) == 2 * 2 and seen and all(seen)


@cuda
def test_oom_is_logged_not_timed(tmp_path):
    from bench import sweep

    sweep.main(["--methods", "naive", "--n", "256", "1048576", "--batch", "64", "--heads", "64",
                "--repeats", "2", "--warmup", "0", "--out-dir", str(tmp_path)])
    (oom,) = list(tmp_path.glob("*/oom.csv"))
    assert pd.read_csv(oom).N.tolist() == [1048576]
    (runs,) = list(tmp_path.glob("*/runs.csv"))
    assert 1048576 not in load_runs(runs).N.values
