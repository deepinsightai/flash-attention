"""Triton kernels against the FP64 reference (forward and all three gradients).

Needs a CUDA GPU and Triton. Triton does not run on this project's Windows setup (Application
Control blocks its compiled helper), so run these in WSL or Colab.
"""

import sys
from pathlib import Path

import pytest
import torch

triton = pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

_BASELINES = Path(__file__).resolve().parents[1] / "attention-baselines (1)" / "attention-baselines"
if str(_BASELINES) not in sys.path:
    sys.path.insert(0, str(_BASELINES))

from attention_baselines import reference_attention  # noqa: E402

from bench.triton_attn import METHODS  # noqa: E402

# relative Frobenius error limits: roughly 4x what the FP32-accumulating kernels measure
TOL = {torch.float16: 2e-3, torch.bfloat16: 1.5e-2, torch.float32: 2e-4}
FA4_TOL_FP32 = 2e-3  # degree-3 exp2 polynomial: 1e-4 relative error per element


def _rel(a, b):
    return ((a.double() - b).norm() / b.norm()).item()


def _check(name, B, H, N, d, dtype, causal):
    torch.manual_seed(1)
    q, k, v = (torch.randn(B, H, N, d, device="cuda", dtype=dtype).requires_grad_() for _ in range(3))
    do = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
    out = METHODS[name](q, k, v, causal=causal)
    out.backward(do)
    qd, kd, vd = (t.detach().double().requires_grad_() for t in (q, k, v))
    ref = reference_attention(qd, kd, vd, causal=causal)
    ref.backward(do.double())
    tol = FA4_TOL_FP32 if (name == "triton_fa4" and dtype == torch.float32) else TOL[dtype]
    assert out.dtype == dtype and q.grad.dtype == dtype
    for label, got, want in (("out", out, ref), ("dq", q.grad, qd.grad), ("dk", k.grad, kd.grad), ("dv", v.grad, vd.grad)):
        assert _rel(got, want) < tol, f"{name} {label} rel error {_rel(got, want):.2e} >= {tol}"


@pytest.mark.parametrize("name", list(METHODS))
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_matches_fp64_reference(name, causal, dtype):
    _check(name, 2, 3, 200, 64, dtype, causal)  # N = 200 is not a multiple of any block size


@pytest.mark.parametrize("name", list(METHODS))
@pytest.mark.parametrize("d", [16, 32, 128])
def test_head_dims(name, d):
    _check(name, 1, 2, 160, d, torch.float16, True)


@pytest.mark.parametrize("name", ["triton_rabe_staats", "triton_fa2", "triton_fa4"])
def test_longer_sequence_causal(name):
    _check(name, 1, 2, 2500, 64, torch.bfloat16, True)  # several chunks / blocks per row


def test_rabe_staats_multiple_chunks():
    q, k, v = (torch.randn(1, 2, 300, 32, device="cuda", dtype=torch.float16) for _ in range(3))
    out = METHODS["triton_rabe_staats"](q, k, v, causal=True)
    from bench.triton_attn import rabe_staats_triton

    small = rabe_staats_triton(q, k, v, causal=True, query_chunk=64, key_chunk=96)
    assert _rel(small, out.double()) < 2e-3


def test_flash_variants_agree_with_each_other():
    q, k, v = (torch.randn(1, 2, 512, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    outs = [METHODS[f"triton_fa{i}"](q, k, v, causal=True) for i in (1, 2, 3, 4)]
    for o in outs[1:]:
        assert _rel(o, outs[0].double()) < 3e-3


def test_custom_scale_and_rejects_cross_attention():
    q = torch.randn(1, 1, 64, 32, device="cuda", dtype=torch.float16)
    out = METHODS["triton_fa2"](q, q, q, scale=0.3)
    ref = reference_attention(q.double(), q.double(), q.double(), scale=0.3)
    assert _rel(out, ref) < 2e-3
    with pytest.raises(ValueError):
        METHODS["triton_fa2"](q, q[:, :, :32], q[:, :, :32])
