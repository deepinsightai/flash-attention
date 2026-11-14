"""Correctness tests. They run on CPU in seconds:  pytest -q

Exact methods are compared in FP64, where they must agree to ~1e-12. Lower-precision
checks use tolerances, never exact equality: changing the order of summation changes
floating-point rounding even when the math is identical.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from attention_baselines import (
    METHODS,
    SUPPORTS_CAUSAL,
    attention_error,
    get_attention,
    linformer_attention,
    linformer_projections,
    naive_attention,
    orthogonal_gaussian_features,
    performer_attention,
    rabe_staats_attention,
    reference_attention,
    softmax_kernel_features,
)


def rand_qkv(B=2, H=3, N=37, d=16, dtype=torch.float64, qk_scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, N, d, generator=g, dtype=dtype) * qk_scale
    k = torch.randn(B, H, N, d, generator=g, dtype=dtype) * qk_scale
    v = torch.randn(B, H, N, d, generator=g, dtype=dtype)
    return q, k, v


# --------------------------------------------------------------------------- naive


@pytest.mark.parametrize("causal", [False, True])
def test_naive_matches_pytorch_sdpa(causal):
    q, k, v = rand_qkv(dtype=torch.float32)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(naive_attention(q, k, v, causal=causal), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("causal", [False, True])
def test_naive_fp32_close_to_fp64_reference(causal):
    q, k, v = rand_qkv(dtype=torch.float32)
    err = attention_error(naive_attention(q, k, v, causal=causal), reference_attention(q, k, v, causal=causal))
    assert err["rel_fro"] < 1e-5


def test_causal_first_row_attends_only_to_itself():
    q, k, v = rand_qkv()
    out = naive_attention(q, k, v, causal=True)
    torch.testing.assert_close(out[:, :, 0], v[:, :, 0])


# --------------------------------------------------------------------------- Rabe-Staats


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("N", [1, 37, 128])
@pytest.mark.parametrize("chunks", [(8, 16), (16, 7), (1024, 4096)])
def test_rabe_staats_is_exact(causal, N, chunks):
    q, k, v = rand_qkv(N=N)
    out = rabe_staats_attention(q, k, v, causal=causal, query_chunk=chunks[0], key_chunk=chunks[1])
    torch.testing.assert_close(out, naive_attention(q, k, v, causal=causal), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("causal", [False, True])
def test_rabe_staats_stable_with_large_scores(causal):
    # Scores of order 1e3: exp() overflows unless the running max is subtracted correctly.
    q, k, v = rand_qkv(qk_scale=50.0)
    out = rabe_staats_attention(q, k, v, causal=causal, query_chunk=8, key_chunk=5)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, naive_attention(q, k, v, causal=causal), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_rabe_staats_gradients_match_naive(causal, checkpoint):
    q, k, v = rand_qkv(B=1, H=2, N=29, d=8)
    upstream = torch.randn_like(v)

    def grads(fn):
        qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
        (fn(qq, kk, vv) * upstream).sum().backward()
        return qq.grad, kk.grad, vv.grad

    expected = grads(lambda a, b, c: naive_attention(a, b, c, causal=causal))
    got = grads(lambda a, b, c: rabe_staats_attention(
        a, b, c, causal=causal, query_chunk=8, key_chunk=6, checkpoint=checkpoint))
    for g, e in zip(got, expected):
        torch.testing.assert_close(g, e, atol=1e-10, rtol=1e-10)


def test_rabe_staats_fp32_close_to_reference():
    q, k, v = rand_qkv(dtype=torch.float32, N=200)
    out = rabe_staats_attention(q, k, v, query_chunk=32, key_chunk=48)
    assert attention_error(out, reference_attention(q, k, v))["rel_fro"] < 1e-5


# --------------------------------------------------------------------------- Performer


def test_orthogonal_features_are_orthogonal_within_blocks():
    w = orthogonal_gaussian_features(32, 16, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    block = w[:16] / w[:16].norm(dim=1, keepdim=True)
    torch.testing.assert_close(block @ block.T, torch.eye(16, dtype=torch.float64), atol=1e-10, rtol=0)


def test_performer_kernel_estimate_is_unbiased():
    # With many features, phi(q).phi(k) should approach exp(q.k / sqrt(d)).
    # This test catches the QR sign bug: without the sign(diag(R)) correction some estimates are off by ~10%.
    d = 8
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4, d, generator=g, dtype=torch.float64) * 0.5
    y = torch.randn(4, d, generator=g, dtype=torch.float64) * 0.5
    w = orthogonal_gaussian_features(1 << 16, d, generator=g, dtype=torch.float64)
    scale = 1 / math.sqrt(d)
    est = softmax_kernel_features(x, w, is_query=True, scale=scale, stabilize=False, eps=0.0) @ \
        softmax_kernel_features(y, w, is_query=False, scale=scale, stabilize=False, eps=0.0).T
    exact = torch.exp(x @ y.T * scale)
    assert ((est - exact).abs() / exact).max() < 0.05


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("chunk", [1, 8, 64])
def test_performer_matches_explicit_kernel_formula(causal, chunk):
    # The linear-time computation must equal the explicit N x N form of the *same* approximation.
    q, k, v = rand_qkv()
    d = q.shape[-1]
    w = orthogonal_gaussian_features(24, d, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    scale = 1 / math.sqrt(d)
    a = softmax_kernel_features(q, w, is_query=True, scale=scale) @ \
        softmax_kernel_features(k, w, is_query=False, scale=scale).transpose(-2, -1)
    if causal:
        a = torch.tril(a)
    expected = (a @ v) / a.sum(dim=-1, keepdim=True)
    got = performer_attention(q, k, v, causal=causal, features=w, causal_chunk=chunk)
    torch.testing.assert_close(got, expected, atol=1e-10, rtol=1e-10)


def test_performer_error_falls_with_more_features():
    # Small-norm inputs: the regime where the estimator works (see performer.py).
    def mean_error(m):
        errs = []
        for seed in range(4):
            q, k, v = rand_qkv(B=1, H=2, N=128, d=32, qk_scale=0.25, seed=seed)
            w = orthogonal_gaussian_features(m, 32, generator=torch.Generator().manual_seed(100 + seed),
                                             dtype=torch.float64)
            out = performer_attention(q, k, v, features=w)
            errs.append(attention_error(out, reference_attention(q, k, v))["rel_fro"])
        return sum(errs) / len(errs)

    assert mean_error(1024) < 0.5 * mean_error(32)


# --------------------------------------------------------------------------- Linformer


def test_linformer_rejects_causal():
    q, k, v = rand_qkv()
    with pytest.raises(NotImplementedError):
        linformer_attention(q, k, v, causal=True)


def test_linformer_identity_projection_is_exact():
    q, k, v = rand_qkv()
    n = k.shape[2]
    eye = torch.eye(n, dtype=torch.float64)
    torch.testing.assert_close(linformer_attention(q, k, v, proj_k=eye), naive_attention(q, k, v))


def test_linformer_pooling_with_full_rank_is_exact():
    q, k, v = rand_qkv()
    n = k.shape[2]
    e = linformer_projections(n, n, kind="pooling", dtype=torch.float64)
    torch.testing.assert_close(linformer_attention(q, k, v, proj_k=e), naive_attention(q, k, v))


def test_linformer_pooling_rows_are_averages():
    e = linformer_projections(10, 3, kind="pooling", dtype=torch.float64)
    torch.testing.assert_close(e.sum(dim=1), torch.ones(3, dtype=torch.float64))
    assert ((e > 0).sum(dim=0) == 1).all()  # every position belongs to exactly one segment


# --------------------------------------------------------------------------- interface


@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("causal", [False, True])
def test_registry_interface(name, causal):
    if causal and not SUPPORTS_CAUSAL[name]:
        pytest.skip(f"{name} does not support causal masking")
    q, k, v = rand_qkv(dtype=torch.float32, N=64, d=16)
    attn = get_attention(name, seq_len=64, head_dim=16, seed=0, **({"proj_dim": 16} if name == "linformer" else {}))
    out = attn(q, k, v, causal=causal)
    assert out.shape == q.shape and out.dtype == q.dtype and torch.isfinite(out).all()


def native_bf16():
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # older PyTorch has no including_emulation argument and reports native support only
        return torch.cuda.is_bf16_supported()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gpu_low_precision(name, dtype):
    if dtype == torch.bfloat16 and not native_bf16():
        pytest.skip("no native BF16 on this GPU (e.g. T4); emulated BF16 is not representative")
    q, k, v = rand_qkv(B=1, H=4, N=256, d=64, dtype=dtype, qk_scale=0.5)
    q, k, v = q.cuda(), k.cuda(), v.cuda()
    attn = get_attention(name, seq_len=256, head_dim=64, device="cuda", seed=0)
    out = attn(q, k, v)
    assert out.dtype == dtype and torch.isfinite(out).all()
    if name in ("naive", "rabe_staats"):  # exact methods: error at the level of input rounding
        assert attention_error(out, reference_attention(q, k, v))["rel_fro"] < 2e-2
