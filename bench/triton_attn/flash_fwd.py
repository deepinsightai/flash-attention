"""Forward kernels for the FlashAttention family, one per generation.

Common conventions (the interface Bahaa's forward and Shaaz's backward agree on)
------------------------------------------------------------------------------
- q, k, v, o are contiguous [B, H, N, d] tensors with dv == d, d a power of two in [16, 256].
- The softmax scale is folded into the exponent base: scores are s = scale * log2(e) * q.k, and
  probabilities are exp2(s - m). This is the FlashAttention-2 trick (exp2 is the hardware
  instruction; the multiply is hoisted out of the inner loop).
- The forward saves L [B*H, N], FP32: L = m + log2(l), the base-2 logsumexp of the scaled scores
  of each row. The backward recomputes P = exp2(s - L). No N x N matrix is ever stored.
- FP32 accumulation; P is cast to the dtype of V for the P V matmul, as in the CUDA kernels.

What each variant models
------------------------
v1  FlashAttention (2022). Grid over batch*heads only. Outer loop over K/V blocks, inner loop over
    Q blocks, so the running output O, row max m and row sum l are read from and written to HBM for
    every (K/V block, Q block) pair. O is kept normalized after every update (Algorithm 1).
v2  FlashAttention-2 (2023). Grid over (Q blocks, batch*heads). Outer loop over Q blocks (parallel
    programs), inner loop over K/V blocks, state in registers. O stays unnormalized until the end
    (one division per row), causal blocks above the diagonal are skipped, and only diagonal blocks
    are masked.
v3  FlashAttention-3 (2024). The FA2 loop with the paper's intra-warpgroup pipelining (Algorithm 2):
    S_next = Q K_{j+1}^T is issued before P_j V_j, and the softmax of block j+1 is overlapped with
    the matmul of block j. Hopper-only features (TMA, warp specialization, WGMMA async, FP8) are
    not expressible portably in Triton and are not modelled. On GPUs without async matmul the
    pipelined order has no hardware to overlap with, so expect v3 ~ v2 there.
v4  FlashAttention-4 (2026). The FA2 loop with the paper's two algorithmic changes for Blackwell:
    (a) software exp2: half of each score tile goes through a degree-3 polynomial evaluated on the
        FMA units instead of the exp2 hardware unit (Cody-Waite range reduction + bit-shift
        exponent), and (b) conditional rescaling: the running max is only updated, and the
        accumulator only rescaled, when the new max exceeds the stale one by more than 2^8, which
        skips most rescale multiplications. Tensor memory, 2-CTA MMA and the CuTe-DSL pipelines are
        hardware features and are not modelled.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._util import LOG2E, TUNED, cdiv, dot_precision, dtype_name, flash_config

# degree-3 fit of 2^f on [0, 1), constrained to p(0) = 1 and p(1) = 2; max relative error 1.0e-4
_C1 = tl.constexpr(0.6954243896402501)
_C2 = tl.constexpr(0.22630766174945782)
_C3 = tl.constexpr(0.07826794861029207)
# FA4 skips a rescale while the max grows by at most this many octaves (2^8 = 256, as in the paper)
RESCALE_THRESHOLD = 8.0


# ------------------------------------------------------------------------------------------ v1
@triton.jit
def _fa1_fwd_kernel(Q, K, V, O, M, Lsum, N, qk_scale,
                    BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
                    CAUSAL: tl.constexpr, PREC: tl.constexpr):
    bh = tl.program_id(0).to(tl.int64)
    base = bh * N * D
    vec = bh * N
    offs_d = tl.arange(0, D)
    for start_n in range(0, N, BN):
        offs_n = start_n + tl.arange(0, BN)
        mn = offs_n < N
        k_t = tl.load(K + base + offs_d[:, None] + offs_n[None, :] * D, mask=mn[None, :], other=0.0)
        v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :], mask=mn[:, None], other=0.0)
        lo = 0
        if CAUSAL:
            lo = (start_n // BM) * BM
        for start_m in range(lo, N, BM):
            offs_m = start_m + tl.arange(0, BM)
            mm = offs_m < N
            q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :], mask=mm[:, None], other=0.0)
            # running state lives in HBM
            o_i = tl.load(O + base + offs_m[:, None] * D + offs_d[None, :], mask=mm[:, None], other=0.0)
            m_i = tl.load(M + vec + offs_m, mask=mm, other=0.0)
            l_i = tl.load(Lsum + vec + offs_m, mask=mm, other=0.0)
            s = tl.dot(q, k_t, input_precision=PREC) * qk_scale
            if CAUSAL:
                s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))
            s = tl.where(mn[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            p = tl.exp2(s - m_new[:, None])
            alpha = tl.exp2(m_i - m_new)
            l_new = alpha * l_i + tl.sum(p, axis=1)
            pv = tl.dot(p.to(v.dtype), v, input_precision=PREC)
            o_new = (o_i * (alpha * l_i)[:, None] + pv) / l_new[:, None]
            tl.store(O + base + offs_m[:, None] * D + offs_d[None, :], o_new, mask=mm[:, None])
            tl.store(M + vec + offs_m, m_new, mask=mm)
            tl.store(Lsum + vec + offs_m, l_new, mask=mm)


# ------------------------------------------------------------------------------------------ v2
@triton.jit
def _fa2_inner(acc, l_i, m_i, q, K, V, base, offs_m, start_lo, start_hi, N, qk_scale,
               BN: tl.constexpr, D: tl.constexpr, MASK: tl.constexpr, PREC: tl.constexpr):
    offs_d = tl.arange(0, D)
    for start_n in range(start_lo, start_hi, BN):
        offs_n = start_n + tl.arange(0, BN)
        mn = offs_n < N
        k_t = tl.load(K + base + offs_d[:, None] + offs_n[None, :] * D, mask=mn[None, :], other=0.0)
        s = tl.dot(q, k_t, input_precision=PREC) * qk_scale
        if MASK:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))
        s = tl.where(mn[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :], mask=mn[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=PREC)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _flash_epilogue(acc, l_i, m_i, O, Lse, base, bh, N, offs_m, mm,
                    D: tl.constexpr):
    offs_d = tl.arange(0, D)
    acc = acc / l_i[:, None]
    tl.store(O + base + offs_m[:, None] * D + offs_d[None, :], acc.to(O.dtype.element_ty), mask=mm[:, None])
    tl.store(Lse + bh * N + offs_m, m_i + tl.log2(l_i), mask=mm)


@triton.jit
def _fa2_fwd_kernel(Q, K, V, O, Lse, N, qk_scale,
                    BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
                    CAUSAL: tl.constexpr, PREC: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    base = bh * N * D
    offs_m = pid_m * BM + tl.arange(0, BM)
    mm = offs_m < N
    offs_d = tl.arange(0, D)
    q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :], mask=mm[:, None], other=0.0)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    if CAUSAL:
        # blocks strictly below the diagonal need no mask; blocks above it are skipped entirely
        acc, l_i, m_i = _fa2_inner(acc, l_i, m_i, q, K, V, base, offs_m, 0, pid_m * BM, N, qk_scale,
                                   BN, D, False, PREC)
        acc, l_i, m_i = _fa2_inner(acc, l_i, m_i, q, K, V, base, offs_m, pid_m * BM,
                                   min((pid_m + 1) * BM, N), N, qk_scale, BN, D, True, PREC)
    else:
        acc, l_i, m_i = _fa2_inner(acc, l_i, m_i, q, K, V, base, offs_m, 0, N, N, qk_scale,
                                   BN, D, False, PREC)
    _flash_epilogue(acc, l_i, m_i, O, Lse, base, bh, N, offs_m, mm, D)


# ------------------------------------------------------------------------------------------ v3
@triton.jit
def _fa3_scores(q, K, base, offs_m, start_n, N, qk_scale,
                BN: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr, PREC: tl.constexpr):
    offs_d = tl.arange(0, D)
    offs_n = start_n + tl.arange(0, BN)
    mn = offs_n < N
    k_t = tl.load(K + base + offs_d[:, None] + offs_n[None, :] * D, mask=mn[None, :], other=0.0)
    s = tl.dot(q, k_t, input_precision=PREC) * qk_scale
    if CAUSAL:
        s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))
    return tl.where(mn[None, :], s, float("-inf"))


@triton.jit
def _fa3_softmax(s, m_i, l_i):
    m_new = tl.maximum(m_i, tl.max(s, axis=1))
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(s - m_new[:, None])
    return p, m_new, alpha, l_i * alpha + tl.sum(p, axis=1)


@triton.jit
def _fa3_fwd_kernel(Q, K, V, O, Lse, N, qk_scale,
                    BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
                    CAUSAL: tl.constexpr, PREC: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    base = bh * N * D
    offs_m = pid_m * BM + tl.arange(0, BM)
    mm = offs_m < N
    offs_d = tl.arange(0, D)
    q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :], mask=mm[:, None], other=0.0)
    hi = N
    if CAUSAL:
        hi = min((pid_m + 1) * BM, N)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)

    # prologue: scores and softmax of block 0
    s0 = _fa3_scores(q, K, base, offs_m, 0, N, qk_scale, BN, D, CAUSAL, PREC)
    p_cur, m_i, alpha_cur, l_i = _fa3_softmax(s0, m_i, l_i)
    # steady state: issue S_{j} = Q K_j^T first, then P_{j-1} V_{j-1}, then softmax of S_j
    for start_n in range(BN, hi, BN):
        s_next = _fa3_scores(q, K, base, offs_m, start_n, N, qk_scale, BN, D, CAUSAL, PREC)
        offs_pv = start_n - BN + tl.arange(0, BN)
        v_prev = tl.load(V + base + offs_pv[:, None] * D + offs_d[None, :], mask=(offs_pv < N)[:, None], other=0.0)
        acc = acc * alpha_cur[:, None] + tl.dot(p_cur.to(v_prev.dtype), v_prev, input_precision=PREC)
        p_cur, m_i, alpha_cur, l_i = _fa3_softmax(s_next, m_i, l_i)
    # epilogue: last P V
    last = ((hi - 1) // BN) * BN
    offs_pv = last + tl.arange(0, BN)
    v_last = tl.load(V + base + offs_pv[:, None] * D + offs_d[None, :], mask=(offs_pv < N)[:, None], other=0.0)
    acc = acc * alpha_cur[:, None] + tl.dot(p_cur.to(v_last.dtype), v_last, input_precision=PREC)
    _flash_epilogue(acc, l_i, m_i, O, Lse, base, bh, N, offs_m, mm, D)


# ------------------------------------------------------------------------------------------ v4
@triton.jit
def _exp2_poly(x):
    """2^x on the FMA units: clamp, split x = j + f, evaluate a cubic in f, add j to the exponent."""
    x = tl.maximum(x, -127.0)                      # also maps -inf to exactly 0.0 (see below)
    j = tl.floor(x)
    f = x - j
    p = 1.0 + f * (_C1 + f * (_C2 + f * _C3))      # Horner form, 3 FMAs
    bits = p.to(tl.int32, bitcast=True) + (j.to(tl.int32) << 23)
    return bits.to(tl.float32, bitcast=True)       # x == -127: exponent field 0, mantissa 0 -> 0.0


@triton.jit
def _fa4_fwd_kernel(Q, K, V, O, Lse, N, qk_scale,
                    BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
                    CAUSAL: tl.constexpr, PREC: tl.constexpr, TAU: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    base = bh * N * D
    HALF: tl.constexpr = BN // 2
    offs_m = pid_m * BM + tl.arange(0, BM)
    mm = offs_m < N
    offs_d = tl.arange(0, D)
    q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :], mask=mm[:, None], other=0.0)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    hi = N
    if CAUSAL:
        hi = min((pid_m + 1) * BM, N)
    for start_n in range(0, hi, BN):
        offs_a = start_n + tl.arange(0, HALF)
        offs_b = start_n + HALF + tl.arange(0, HALF)
        ma = offs_a < N
        mb = offs_b < N
        k_a = tl.load(K + base + offs_d[:, None] + offs_a[None, :] * D, mask=ma[None, :], other=0.0)
        k_b = tl.load(K + base + offs_d[:, None] + offs_b[None, :] * D, mask=mb[None, :], other=0.0)
        s_a = tl.dot(q, k_a, input_precision=PREC) * qk_scale
        s_b = tl.dot(q, k_b, input_precision=PREC) * qk_scale
        if CAUSAL:
            s_a = tl.where(offs_m[:, None] >= offs_a[None, :], s_a, float("-inf"))
            s_b = tl.where(offs_m[:, None] >= offs_b[None, :], s_b, float("-inf"))
        s_a = tl.where(ma[None, :], s_a, float("-inf"))
        s_b = tl.where(mb[None, :], s_b, float("-inf"))

        # conditional rescale: keep the stale max unless it is exceeded by more than TAU octaves
        m_cand = tl.maximum(m_i, tl.maximum(tl.max(s_a, axis=1), tl.max(s_b, axis=1)))
        need = (m_cand - m_i) > TAU                       # also true on the first block (m_i = -inf)
        m_use = tl.where(need, m_cand, m_i)
        if tl.max(need.to(tl.int32), axis=0) > 0:
            alpha = tl.exp2(m_i - m_use)                  # 1.0 for rows that kept their max
            acc = acc * alpha[:, None]
            l_i = l_i * alpha
        m_i = m_use

        p_a = tl.exp2(s_a - m_i[:, None])                 # MUFU exp2
        p_b = _exp2_poly(s_b - m_i[:, None])              # FMA polynomial exp2
        l_i = l_i + tl.sum(p_a, axis=1) + tl.sum(p_b, axis=1)
        v_a = tl.load(V + base + offs_a[:, None] * D + offs_d[None, :], mask=ma[:, None], other=0.0)
        v_b = tl.load(V + base + offs_b[:, None] * D + offs_d[None, :], mask=mb[:, None], other=0.0)
        acc = tl.dot(p_a.to(v_a.dtype), v_a, acc, input_precision=PREC)
        acc = tl.dot(p_b.to(v_b.dtype), v_b, acc, input_precision=PREC)
    _flash_epilogue(acc, l_i, m_i, O, Lse, base, bh, N, offs_m, mm, D)


# ------------------------------------------------------------------------------------- launcher
def flash_fwd(q, k, v, causal: bool, scale: float, version: int):
    """Returns (o, lse). q, k, v: contiguous [B, H, N, d]. lse: FP32 [B*H, N], base-2 convention."""
    B, H, N, d = q.shape
    BH = B * H
    BM, BN, warps, stages = flash_config(d, q.dtype, version=version)
    prec = dot_precision(q.dtype)
    qk_scale = scale * LOG2E
    lse = torch.empty((BH, N), device=q.device, dtype=torch.float32)
    if version == 1:
        # Tuned FA1 forward config if present, else the original default
        BM, BN, warps, stages = (TUNED["flash"].get((1, "fwd", d, dtype_name(q.dtype)))
                                 or flash_config(d, q.dtype, bwd=True))
        o_acc = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        m = torch.full((BH, N), float("-inf"), device=q.device, dtype=torch.float32)
        l = torch.zeros((BH, N), device=q.device, dtype=torch.float32)
        _fa1_fwd_kernel[(BH,)](q, k, v, o_acc, m, l, N, qk_scale, BM=BM, BN=BN, D=d, CAUSAL=causal,
                               PREC=prec, num_warps=warps, num_stages=stages)
        lse = m + torch.log2(l)
        return o_acc.to(q.dtype), lse
    o = torch.empty_like(q)
    grid = (cdiv(N, BM), BH)
    kw = dict(BM=BM, BN=BN, D=d, CAUSAL=causal, PREC=prec, num_warps=warps, num_stages=stages)
    if version == 2:
        _fa2_fwd_kernel[grid](q, k, v, o, lse, N, qk_scale, **kw)
    elif version == 3:
        _fa3_fwd_kernel[grid](q, k, v, o, lse, N, qk_scale, **kw)
    elif version == 4:
        _fa4_fwd_kernel[grid](q, k, v, o, lse, N, qk_scale, TAU=RESCALE_THRESHOLD, **kw)
    else:
        raise ValueError(f"unknown FlashAttention version {version}")
    return o, lse
