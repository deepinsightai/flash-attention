"""Backward kernels for the FlashAttention variants.

Both backward passes recompute P = exp2(scale*log2(e) * Q K^T - L) tile by tile from the forward's
saved logsumexp L, so no N x N matrix is stored. Per tile:

    dV += P^T dO            dP = dO V^T          dS = P * (dP - D)      D = rowsum(dO * O)
    dK += scale * dS^T Q    dQ += scale * dS K

`flash_v2_bwd` (FlashAttention-2 schedule, also used for the FA3 and FA4 variants here):
    two kernels. dK/dV: grid (key blocks, batch*heads); each program owns one K/V block, loops over
    query blocks and accumulates dK and dV in registers. dQ: grid (query blocks, batch*heads); each
    program owns one query block and loops over key blocks. No atomics (the first version used FP32 atomic adds
    for dQ and was ~16x slower than its forward on the A100).

`flash_v1_bwd` (FlashAttention-1, Algorithm 4):
    grid (batch*heads,). One program loops over key blocks (outer) and query blocks (inner) and
    updates dQ with a plain read-modify-write in HBM, because only one program touches a head.

The FA3 and FA4 papers change the backward for Hopper and Blackwell hardware (warpgroup MMA, tensor
memory, 2-CTA mode). Triton does not expose those, so the variants here share the FA2 backward and
differ from FA2 only in the forward.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._util import LOG2E, TUNED, cdiv, dot_precision, dtype_name, flash_config


def dq_config(d: int, dtype: torch.dtype, version: int):
    """(BLOCK_M, BLOCK_N, num_warps, num_stages) for the FA2 dQ kernel; tuned entry wins if present."""
    tuned = TUNED["flash"].get((version, "bwd_dq", d, dtype_name(dtype)))
    if tuned is not None:
        return tuple(tuned)
    if dtype == torch.float32 or d > 128:
        return (32, 32, 4, 1)
    return (64, 64, 4, 1)


@triton.jit
def _delta_kernel(O, DO, Dl, n_rows, D: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    cols = tl.arange(0, D)
    mask = rows[:, None] < n_rows
    o = tl.load(O + rows[:, None] * D + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    do = tl.load(DO + rows[:, None] * D + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    tl.store(Dl + rows, tl.sum(o * do, axis=1), mask=rows < n_rows)


def preprocess_delta(o, do):
    """D = rowsum(dO * O) for [BH, N, d] tensors; returns FP32 [BH, N]."""
    BH, N, d = o.shape
    delta = torch.empty((BH, N), device=o.device, dtype=torch.float32)
    # D must be a power of two for tl.arange; pad via a masked view is not needed for the benchmark
    # head dims, and the non-power-of-two path falls back to torch.
    if d & (d - 1):
        return (o.float() * do.float()).sum(-1)
    _delta_kernel[(cdiv(BH * N, 128),)](o, do, delta, BH * N, D=d, BLOCK=128, num_warps=4)
    return delta


@triton.jit
def _bwd_block(q, do, lse, dlt, k, v, offs_m, offs_n, mm, mn, qk_scale,
               CAUSAL: tl.constexpr, PREC: tl.constexpr):
    """Gradients for one (query block, key block) tile. Returns (dv_tile, dk_tile, dq_tile), unscaled."""
    s = tl.dot(q, tl.trans(k), input_precision=PREC) * qk_scale
    p = tl.exp2(s - lse[:, None])
    if CAUSAL:
        p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)
    p = tl.where(mm[:, None] & mn[None, :], p, 0.0)
    dv_t = tl.dot(tl.trans(p).to(do.dtype), do, input_precision=PREC)
    dp = tl.dot(do, tl.trans(v), input_precision=PREC)
    ds = p * (dp - dlt[:, None])
    dk_t = tl.dot(tl.trans(ds).to(q.dtype), q, input_precision=PREC)
    dq_t = tl.dot(ds.to(k.dtype), k, input_precision=PREC)
    return dv_t, dk_t, dq_t


# The FA2 backward is split into a dK/dV kernel (this one, no dQ) and a separate
# dQ kernel below, as in the FlashAttention-2 paper's schedule. The previous version accumulated dQ
# with FP32 atomic adds from every key-block program (N^2 d / BLOCK_N atomics per head), which made
# the backward ~16x the forward on the A100. Kept algorithmically identical otherwise.
@triton.jit
def _fa2_bwd_kernel(Q, K, V, DO, DQ, DK, DV, Lse, Dl, N, scale, qk_scale,
                    BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
                    CAUSAL: tl.constexpr, PREC: tl.constexpr):
    pid_n = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    base = bh * N * D
    offs_d = tl.arange(0, D)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mn = offs_n < N
    kv_off = base + offs_n[:, None] * D + offs_d[None, :]
    k = tl.load(K + kv_off, mask=mn[:, None], other=0.0)
    v = tl.load(V + kv_off, mask=mn[:, None], other=0.0)
    dk = tl.zeros((BN, D), tl.float32)
    dv = tl.zeros((BN, D), tl.float32)
    lo = 0
    if CAUSAL:
        lo = (pid_n * BN // BM) * BM          # query blocks above the diagonal see none of these keys
    for start_m in range(lo, N, BM):
        offs_m = start_m + tl.arange(0, BM)
        mm = offs_m < N
        q_off = base + offs_m[:, None] * D + offs_d[None, :]
        q = tl.load(Q + q_off, mask=mm[:, None], other=0.0)
        do = tl.load(DO + q_off, mask=mm[:, None], other=0.0)
        lse = tl.load(Lse + bh * N + offs_m, mask=mm, other=0.0)
        dlt = tl.load(Dl + bh * N + offs_m, mask=mm, other=0.0)
        dv_t, dk_t, dq_t = _bwd_block(q, do, lse, dlt, k, v, offs_m, offs_n, mm, mn, qk_scale, CAUSAL, PREC)
        dv += dv_t
        dk += dk_t
    tl.store(DK + kv_off, (dk * scale).to(DK.dtype.element_ty), mask=mn[:, None])
    tl.store(DV + kv_off, dv.to(DV.dtype.element_ty), mask=mn[:, None])


# dQ for the FA2 schedule. Grid (query blocks, batch*heads); each program owns
# one query block, loops over key blocks, and accumulates dQ in registers. No atomics, no N x N storage.
@triton.jit
def _fa2_bwd_dq_kernel(Q, K, V, DO, DQ, Lse, Dl, N, scale, qk_scale,
                       BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
                       CAUSAL: tl.constexpr, PREC: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    base = bh * N * D
    offs_d = tl.arange(0, D)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mm = offs_m < N
    q_off = base + offs_m[:, None] * D + offs_d[None, :]
    q = tl.load(Q + q_off, mask=mm[:, None], other=0.0)
    do = tl.load(DO + q_off, mask=mm[:, None], other=0.0)
    lse = tl.load(Lse + bh * N + offs_m, mask=mm, other=0.0)
    dlt = tl.load(Dl + bh * N + offs_m, mask=mm, other=0.0)
    dq = tl.zeros((BM, D), tl.float32)
    hi = N
    if CAUSAL:
        hi = tl.minimum((pid_m + 1) * BM, N)    # keys after the last query of this block are masked
    for start_n in range(0, hi, BN):
        offs_n = start_n + tl.arange(0, BN)
        mn = offs_n < N
        kv_off = base + offs_n[:, None] * D + offs_d[None, :]
        k = tl.load(K + kv_off, mask=mn[:, None], other=0.0)
        v = tl.load(V + kv_off, mask=mn[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision=PREC) * qk_scale
        p = tl.exp2(s - lse[:, None])
        if CAUSAL:
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)
        p = tl.where(mm[:, None] & mn[None, :], p, 0.0)
        dp = tl.dot(do, tl.trans(v), input_precision=PREC)
        ds = p * (dp - dlt[:, None])
        dq += tl.dot(ds.to(k.dtype), k, input_precision=PREC)
    tl.store(DQ + q_off, (dq * scale).to(DQ.dtype.element_ty), mask=mm[:, None])


@triton.jit
def _fa1_bwd_kernel(Q, K, V, DO, DQ, DK, DV, Lse, Dl, N, scale, qk_scale,
                    BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
                    CAUSAL: tl.constexpr, PREC: tl.constexpr):
    bh = tl.program_id(0).to(tl.int64)
    base = bh * N * D
    offs_d = tl.arange(0, D)
    for start_n in range(0, N, BN):
        offs_n = start_n + tl.arange(0, BN)
        mn = offs_n < N
        kv_off = base + offs_n[:, None] * D + offs_d[None, :]
        k = tl.load(K + kv_off, mask=mn[:, None], other=0.0)
        v = tl.load(V + kv_off, mask=mn[:, None], other=0.0)
        dk = tl.zeros((BN, D), tl.float32)
        dv = tl.zeros((BN, D), tl.float32)
        lo = 0
        if CAUSAL:
            lo = (start_n // BM) * BM
        for start_m in range(lo, N, BM):
            offs_m = start_m + tl.arange(0, BM)
            mm = offs_m < N
            q_off = base + offs_m[:, None] * D + offs_d[None, :]
            q = tl.load(Q + q_off, mask=mm[:, None], other=0.0)
            do = tl.load(DO + q_off, mask=mm[:, None], other=0.0)
            lse = tl.load(Lse + bh * N + offs_m, mask=mm, other=0.0)
            dlt = tl.load(Dl + bh * N + offs_m, mask=mm, other=0.0)
            dv_t, dk_t, dq_t = _bwd_block(q, do, lse, dlt, k, v, offs_m, offs_n, mm, mn, qk_scale, CAUSAL, PREC)
            dv += dv_t
            dk += dk_t
            dq_old = tl.load(DQ + q_off, mask=mm[:, None], other=0.0)     # HBM read-modify-write
            tl.store(DQ + q_off, dq_old + dq_t * scale, mask=mm[:, None])
        tl.store(DK + kv_off, (dk * scale).to(DK.dtype.element_ty), mask=mn[:, None])
        tl.store(DV + kv_off, dv.to(DV.dtype.element_ty), mask=mn[:, None])


def flash_bwd(q, k, v, o, do, lse, causal: bool, scale: float, version: int):
    """q, k, v, o, do: contiguous [B, H, N, d]. lse: FP32 [B*H, N] in the base-2 convention."""
    B, H, N, d = q.shape
    BH = B * H
    do = do.contiguous()
    delta = preprocess_delta(o.reshape(BH, N, d), do.reshape(BH, N, d))
    # FA1 accumulates dQ in an FP32 HBM buffer; FA2+ writes dQ once from its own kernel
    dq = (torch.zeros(q.shape, device=q.device, dtype=torch.float32) if version == 1
          else torch.empty_like(q))
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    BM, BN, warps, stages = flash_config(d, q.dtype, bwd=True, version=version)
    prec = dot_precision(q.dtype)
    common = dict(BM=BM, BN=BN, D=d, CAUSAL=causal, PREC=prec, num_warps=warps, num_stages=stages)
    args = (q, k, v, do, dq, dk, dv, lse, delta, N, scale, scale * LOG2E)
    if version == 1:
        _fa1_bwd_kernel[(BH,)](*args, **common)
        return dq.to(q.dtype), dk, dv
    _fa2_bwd_kernel[(cdiv(N, BN), BH)](*args, **common)
    BMq, BNq, wq, sq = dq_config(d, q.dtype, version)
    _fa2_bwd_dq_kernel[(cdiv(N, BMq), BH)](q, k, v, do, dq, lse, delta, N, scale, scale * LOG2E,
                                           BM=BMq, BN=BNq, D=d, CAUSAL=causal, PREC=prec,
                                           num_warps=wq, num_stages=sq)
    return dq, dk, dv
