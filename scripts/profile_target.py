"""Profiling target for Nsight Systems / Nsight Compute.

Runs warm-up, then a profiled region of one attention call, wrapped in an NVTX range and
cudaProfilerStart/Stop so the profilers capture only the call of interest.

    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -o out \
        python -m scripts.profile_target --method sdpa_flash --n 4096
    ncu --profile-from-start off --set full -o out python -m scripts.profile_target --method naive --n 2048
"""

from __future__ import annotations

import argparse

import torch

from bench import methods


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="sdpa_flash")
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--backward", action="store_true")
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    dev = "cuda"
    shape = (1, args.heads, args.n, args.head_dim)
    q, k, v, leaves = methods.make_qkv(args.method, shape, torch.bfloat16, dev, requires_grad=args.backward)
    fn = methods.build(args.method, args.n, args.head_dim, dev, 0)
    grad = torch.randn_like(q)

    # Clear leaf grads every call so the profiled call has no grad-accumulation kernel
    def call():
        out = fn(q, k, v, causal=args.causal)
        if args.backward:
            out.backward(grad)
            for t in leaves:
                t.grad = None
        return out

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"{args.method}_N{args.n}")
    call()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
