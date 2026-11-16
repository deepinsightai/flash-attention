"""Naive (standard) attention in Triton.

Three separate kernels, exactly the standard implementation: S = scale * Q K^T is written to HBM,
softmax reads S and writes P to HBM, and O = P V reads P. The N x N matrices S and P live in global
memory (in the input dtype, as in a PyTorch eager implementation). The backward pass keeps P from
the forward and runs the textbook gradient formulas, again with N x N intermediates in HBM:

    dV = P^T dO         dP = dO V^T
    dS = P * (dP - rowsum(dP * P))
    dQ = scale * dS K   dK = scale * dS^T Q

Memory is Theta(N^2) per head and the kernels are not fused. This is the control for the IO-aware
variants, written with the same toolchain so only the algorithm differs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._util import TUNED, check_qkv, cdiv, default_scale, dot_precision


@triton.jit
def _bmm_kernel(
    A, B, C, M, N, K,
    sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
    alpha,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ACC: tl.constexpr, PREC: tl.constexpr,
):
    """C[bh] (+)= alpha * A[bh] @ B[bh], arbitrary strides so transposes are free."""
    pid = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    grid_n = tl.cdiv(N, BN)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = A + bh * sab + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = B + bh * sbb + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kmask = offs_k < K - k0
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=kmask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision=PREC)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    acc = acc * alpha
    c_ptrs = C + bh * scb + offs_m[:, None] * scm + offs_n[None, :] * scn
    cmask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if ACC:
        acc += tl.load(c_ptrs, mask=cmask, other=0.0).to(tl.float32)
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=cmask)


def bmm(a, b, *, ta=False, tb=False, alpha=1.0, out=None, out_dtype=None, accumulate=False, tag="naive"):
    """Batched matmul over dim 0 of [BH, ., .] tensors. ta/tb treat the stored operand as transposed.

    With ta, `a` is stored [BH, K, M]; with tb, `b` is stored [BH, N, K].
    """
    BH = a.shape[0]
    M, K = (a.shape[2], a.shape[1]) if ta else (a.shape[1], a.shape[2])
    N = b.shape[1] if tb else b.shape[2]
    assert (b.shape[2] if tb else b.shape[1]) == K
    if out is None:
        out = torch.empty((BH, M, N), device=a.device, dtype=out_dtype or a.dtype)
    sam, sak = (a.stride(2), a.stride(1)) if ta else (a.stride(1), a.stride(2))
    sbk, sbn = (b.stride(2), b.stride(1)) if tb else (b.stride(1), b.stride(2))
    # Tuned tiles per caller (naive / rabe_staats), else the original 64x64x32
    BM, BN, BK, warps, stages = TUNED["bmm"].get(tag, (64, 64, 32, 4, 2))
    prec = dot_precision(a.dtype)
    grid = (cdiv(M, BM) * cdiv(N, BN), BH)
    _bmm_kernel[grid](
        a, b, out, M, N, K,
        a.stride(0), sam, sak, b.stride(0), sbk, sbn, out.stride(0), out.stride(1), out.stride(2),
        alpha, BM=BM, BN=BN, BK=BK, ACC=accumulate, PREC=prec, num_warps=warps, num_stages=stages,
    )
    return out


@triton.jit
def _softmax_fwd_kernel(S, P, N, CAUSAL: tl.constexpr, BLOCK: tl.constexpr):
    """Row softmax over [BH*N, N]; three passes over the row (max, sum, normalize)."""
    row = tl.program_id(0).to(tl.int64)
    q_idx = row % N
    cols = tl.arange(0, BLOCK)
    s_row = S + row * N
    p_row = P + row * N
    m = tl.full([BLOCK], float("-inf"), tl.float32)
    for start in range(0, N, BLOCK):
        c = start + cols
        x = tl.load(s_row + c, mask=c < N, other=float("-inf")).to(tl.float32)
        if CAUSAL:
            x = tl.where(c <= q_idx, x, float("-inf"))
        m = tl.maximum(m, x)
    m_row = tl.max(m, axis=0)
    acc = tl.zeros([BLOCK], tl.float32)
    for start in range(0, N, BLOCK):
        c = start + cols
        x = tl.load(s_row + c, mask=c < N, other=float("-inf")).to(tl.float32)
        if CAUSAL:
            x = tl.where(c <= q_idx, x, float("-inf"))
        acc += tl.exp(x - m_row)
    denom = tl.sum(acc, axis=0)
    for start in range(0, N, BLOCK):
        c = start + cols
        x = tl.load(s_row + c, mask=c < N, other=float("-inf")).to(tl.float32)
        if CAUSAL:
            x = tl.where(c <= q_idx, x, float("-inf"))
        tl.store(p_row + c, (tl.exp(x - m_row) / denom).to(P.dtype.element_ty), mask=c < N)


@triton.jit
def _softmax_bwd_kernel(P, DP, DS, N, BLOCK: tl.constexpr):
    """dS = P * (dP - rowsum(P * dP)). Masked entries have P = 0, so dS = 0 there."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.float32)
    for start in range(0, N, BLOCK):
        c = start + cols
        p = tl.load(P + row * N + c, mask=c < N, other=0.0).to(tl.float32)
        dp = tl.load(DP + row * N + c, mask=c < N, other=0.0).to(tl.float32)
        acc += p * dp
    dot = tl.sum(acc, axis=0)
    for start in range(0, N, BLOCK):
        c = start + cols
        p = tl.load(P + row * N + c, mask=c < N, other=0.0).to(tl.float32)
        dp = tl.load(DP + row * N + c, mask=c < N, other=0.0).to(tl.float32)
        tl.store(DS + row * N + c, (p * (dp - dot)).to(DS.dtype.element_ty), mask=c < N)


def softmax_fwd(s, causal: bool):
    BH, N, _ = s.shape
    p = torch.empty_like(s)
    block, warps = TUNED["softmax"].get("naive", (1024, 4))
    _softmax_fwd_kernel[(BH * N,)](s, p, N, CAUSAL=causal, BLOCK=block, num_warps=warps)
    return p


def softmax_bwd(p, dp):
    BH, N, _ = p.shape
    ds = torch.empty_like(p)
    block, warps = TUNED["softmax"].get("naive", (1024, 4))
    _softmax_bwd_kernel[(BH * N,)](p, dp, ds, N, BLOCK=block, num_warps=warps)
    return ds


class _NaiveAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, scale):
        B, H, N, d = q.shape
        dv = v.shape[-1]
        q3, k3, v3 = (t.reshape(B * H, N, -1) for t in (q, k, v))
        s = bmm(q3, k3, tb=True, alpha=scale)            # [BH, N, N] written to HBM
        p = softmax_fwd(s, causal)                       # [BH, N, N] written to HBM
        del s
        o = bmm(p, v3)
        ctx.save_for_backward(q3, k3, v3, p)
        ctx.causal, ctx.scale, ctx.shape = causal, scale, (B, H, N, d, dv)
        return o.reshape(B, H, N, dv)

    @staticmethod
    def backward(ctx, do):
        q3, k3, v3, p = ctx.saved_tensors
        B, H, N, d, dv = ctx.shape
        do3 = do.contiguous().reshape(B * H, N, dv)
        dv_ = bmm(p, do3, ta=True)                        # P^T dO
        dp = bmm(do3, v3, tb=True)                        # dO V^T
        ds = softmax_bwd(p, dp)
        del dp
        dq = bmm(ds, k3, alpha=ctx.scale)
        dk = bmm(ds, q3, ta=True, alpha=ctx.scale)
        return (dq.reshape(B, H, N, d), dk.reshape(B, H, N, d), dv_.reshape(B, H, N, dv), None, None)


def naive_attention_triton(q, k, v, causal=False, scale=None):
    q, k, v = check_qkv(q, k, v, causal)
    return _NaiveAttention.apply(q, k, v, causal, default_scale(q.shape[-1], scale))
