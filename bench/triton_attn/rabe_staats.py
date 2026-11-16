"""Memory-efficient attention (Rabe & Staats, 2021) in Triton, deliberately not IO-aware.

# bmm calls tagged "rabe_staats" and kernel warps read from TUNED, so tuning
# is per method; defaults unchanged.

Queries and keys are processed in chunks (defaults 1024 and 4096, as in the paper). For each
(query chunk, key chunk) pair, separate kernels

    1. write the chunk scores S (FP32) to HBM            (matmul kernel)
    2. read S, write P = exp(S - rowmax) and the row max and row sum   (summary kernel)
    3. read P, write the chunk output P V                (matmul kernel)
    4. merge (m, l, o) into the running state            (merge kernel)

so every chunk's scores make a round trip through global memory. Peak memory is
O(query_chunk * key_chunk) per head instead of O(N^2); time is still Theta(N^2 d).

Backward: the forward saves only the output and the per-row logsumexp L (as the paper's
checkpointing does). The backward recomputes P = exp(S - L) chunk by chunk and applies the standard
gradient formulas, accumulating dQ, dK, dV in FP32. Kernels are the same non-fused building blocks.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._util import TUNED, check_qkv, cdiv, default_scale
from .flash_bwd import preprocess_delta
from .naive import bmm


@triton.jit
def _summary_kernel(S, P, Mo, Lo, qc, kc, q_start, k_start, CAUSAL: tl.constexpr, BLOCK: tl.constexpr):
    """One program per score row: m = rowmax(S), P = exp(S - m), l = rowsum(P)."""
    row = tl.program_id(0).to(tl.int64)
    r = row % qc
    cols = tl.arange(0, BLOCK)
    x = tl.load(S + row * kc + cols, mask=cols < kc, other=float("-inf"))
    if CAUSAL:
        x = tl.where(k_start + cols <= q_start + r, x, float("-inf"))
    m = tl.max(x, axis=0)
    m_safe = tl.where(m == float("-inf"), 0.0, m)   # fully masked row: contributes l = 0, o = 0
    p = tl.exp(x - m_safe)
    tl.store(P + row * kc + cols, p.to(P.dtype.element_ty), mask=cols < kc)
    tl.store(Mo + row, m)
    tl.store(Lo + row, tl.sum(p, axis=0))


@triton.jit
def _merge_kernel(M, L, O, Mc, Lc, Oc, dv, FIRST: tl.constexpr, BLOCK_D: tl.constexpr):
    """Merge a chunk summary (Mc, Lc, Oc) into the running state (M, L, O) with max rescaling."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    cmask = cols < dv
    m_c = tl.load(Mc + row)
    l_c = tl.load(Lc + row)
    o_c = tl.load(Oc + row * dv + cols, mask=cmask, other=0.0)
    if FIRST:
        tl.store(M + row, m_c)
        tl.store(L + row, l_c)
        tl.store(O + row * dv + cols, o_c, mask=cmask)
    else:
        m = tl.load(M + row)
        l = tl.load(L + row)
        o = tl.load(O + row * dv + cols, mask=cmask, other=0.0)
        m_new = tl.maximum(m, m_c)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        a = tl.exp(m - m_safe)
        b = tl.exp(m_c - m_safe)
        tl.store(M + row, m_new)
        tl.store(L + row, a * l + b * l_c)
        tl.store(O + row * dv + cols, a * o + b * o_c, mask=cmask)


@triton.jit
def _finalize_kernel(M, L, O, Out, Lse, dv, BLOCK_D: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    l = tl.load(L + row)
    o = tl.load(O + row * dv + cols, mask=cols < dv, other=0.0)
    tl.store(Out + row * dv + cols, (o / l).to(Out.dtype.element_ty), mask=cols < dv)
    tl.store(Lse + row, tl.load(M + row) + tl.log(l))


@triton.jit
def _bwd_elem_kernel(S, DP, Lse, Dl, P, DS, qc, kc, q_start, k_start,
                     CAUSAL: tl.constexpr, BLOCK: tl.constexpr):
    """P = exp(S - L) (masked), dS = P * (dP - D). One program per (row, column block)."""
    row = tl.program_id(0).to(tl.int64)
    r = row % qc
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cmask = cols < kc
    s = tl.load(S + row * kc + cols, mask=cmask, other=float("-inf"))
    dp = tl.load(DP + row * kc + cols, mask=cmask, other=0.0)
    lse = tl.load(Lse + row)
    d = tl.load(Dl + row)
    p = tl.exp(s - lse)
    if CAUSAL:
        p = tl.where(k_start + cols <= q_start + r, p, 0.0)
    ds = p * (dp - d)
    tl.store(P + row * kc + cols, p.to(P.dtype.element_ty), mask=cmask)
    tl.store(DS + row * kc + cols, ds.to(DS.dtype.element_ty), mask=cmask)


def _pow2(n):
    return 1 << (n - 1).bit_length()


class _RabeStaats(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, scale, qchunk, kchunk):
        B, H, N, d = q.shape
        dv = v.shape[-1]
        BH = B * H
        q3, k3, v3 = (t.reshape(BH, N, -1) for t in (q, k, v))
        out = torch.empty((BH, N, dv), device=q.device, dtype=q.dtype)
        lse = torch.empty((BH, N), device=q.device, dtype=torch.float32)
        for qs in range(0, N, qchunk):
            qe = min(qs + qchunk, N)
            nq = qe - qs
            qc = q3[:, qs:qe]
            m = torch.empty((BH, nq), device=q.device, dtype=torch.float32)
            l = torch.empty_like(m)
            o = torch.empty((BH, nq, dv), device=q.device, dtype=torch.float32)
            first = True
            for ks in range(0, N, kchunk):
                if causal and ks > qe - 1:
                    break
                ke = min(ks + kchunk, N)
                nk = ke - ks
                s = bmm(qc, k3[:, ks:ke], tb=True, alpha=scale, out_dtype=torch.float32, tag="rabe_staats")
                p = torch.empty((BH, nq, nk), device=q.device, dtype=v.dtype)
                m_c = torch.empty((BH, nq), device=q.device, dtype=torch.float32)
                l_c = torch.empty_like(m_c)
                _summary_kernel[(BH * nq,)](s, p, m_c, l_c, nq, nk, qs, ks, CAUSAL=causal,
                                            BLOCK=_pow2(nk), num_warps=TUNED["rs_elem"].get("rabe_staats", (8, 4))[0])
                del s
                o_c = bmm(p, v3[:, ks:ke], out_dtype=torch.float32, tag="rabe_staats")
                del p
                _merge_kernel[(BH * nq,)](m, l, o, m_c, l_c, o_c, dv, FIRST=first,
                                          BLOCK_D=_pow2(dv), num_warps=4)
                first = False
            out_c = torch.empty((BH, nq, dv), device=q.device, dtype=q.dtype)
            lse_c = torch.empty((BH, nq), device=q.device, dtype=torch.float32)
            _finalize_kernel[(BH * nq,)](m, l, o, out_c, lse_c, dv, BLOCK_D=_pow2(dv), num_warps=4)
            out[:, qs:qe] = out_c
            lse[:, qs:qe] = lse_c
        ctx.save_for_backward(q3, k3, v3, out, lse)
        ctx.cfg = (causal, scale, qchunk, kchunk, (B, H, N, d, dv))
        return out.reshape(B, H, N, dv)

    @staticmethod
    def backward(ctx, do):
        q3, k3, v3, out, lse = ctx.saved_tensors
        causal, scale, qchunk, kchunk, (B, H, N, d, dv) = ctx.cfg
        BH = B * H
        do3 = do.contiguous().reshape(BH, N, dv)
        delta = preprocess_delta(out, do3)               # D = rowsum(dO * O), [BH, N]
        dq = torch.zeros((BH, N, d), device=q3.device, dtype=torch.float32)
        dk = torch.zeros((BH, N, d), device=q3.device, dtype=torch.float32)
        dvv = torch.zeros((BH, N, dv), device=q3.device, dtype=torch.float32)
        for qs in range(0, N, qchunk):
            qe = min(qs + qchunk, N)
            nq = qe - qs
            lse_q = lse[:, qs:qe].contiguous()
            delta_q = delta[:, qs:qe].contiguous()
            for ks in range(0, N, kchunk):
                if causal and ks > qe - 1:
                    break
                ke = min(ks + kchunk, N)
                nk = ke - ks
                s = bmm(q3[:, qs:qe], k3[:, ks:ke], tb=True, alpha=scale, out_dtype=torch.float32, tag="rabe_staats")
                dp = bmm(do3[:, qs:qe], v3[:, ks:ke], tb=True, out_dtype=torch.float32, tag="rabe_staats")
                p = torch.empty((BH, nq, nk), device=q3.device, dtype=q3.dtype)
                ds = torch.empty_like(p)
                blk = min(_pow2(nk), 1024)
                _bwd_elem_kernel[(BH * nq, cdiv(nk, blk))](
                    s, dp, lse_q, delta_q, p, ds, nq, nk, qs, ks,
                    CAUSAL=causal, BLOCK=blk, num_warps=TUNED["rs_elem"].get("rabe_staats", (8, 4))[1])
                del s, dp
                bmm(p, do3[:, qs:qe], ta=True, out=dvv[:, ks:ke], accumulate=True, tag="rabe_staats")
                bmm(ds, k3[:, ks:ke], alpha=scale, out=dq[:, qs:qe], accumulate=True, tag="rabe_staats")
                bmm(ds, q3[:, qs:qe], ta=True, alpha=scale, out=dk[:, ks:ke], accumulate=True, tag="rabe_staats")
        return (dq.to(q3.dtype).reshape(B, H, N, d), dk.to(q3.dtype).reshape(B, H, N, d),
                dvv.to(q3.dtype).reshape(B, H, N, dv), None, None, None, None)


def rabe_staats_triton(q, k, v, causal=False, scale=None, query_chunk=1024, key_chunk=4096):
    q, k, v = check_qkv(q, k, v, causal)
    if key_chunk > 8192:
        raise ValueError("key_chunk above 8192 does not fit one summary-kernel row")
    return _RabeStaats.apply(q, k, v, causal, default_scale(q.shape[-1], scale), query_chunk, key_chunk)
