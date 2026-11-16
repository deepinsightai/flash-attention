## RQ1

RQ1 asks whether FlashAttention gives (1) lower runtime, (2) higher throughput, and (3)
  O(N) rather than O(N^2) memory compared with the baselines. Each panel of
  rq1_speed_memory.png covers part of that, using forward+backward, non-causal, on the
  A100.
  
  (a) Runtime against sequence length: the speed half of RQ1
  - Each line is one method's median time per call. Lower is faster, and both axes are
  logarithmic.
  - Below about 2K, there's no meaningful difference. All methods take under 1 ms, because
  the GPU is mostly idle with batch 1 and 8 heads and launch overhead dominates. FA2
  (black) is even slightly slower than SDPA's flash backend there, from library call
  overhead.
  - From 4K up, the FA2 kernels (official FA2 and SDPA flash, which overlap) are the
  fastest exact methods, and the gap widens with N. FA2 is:
    - 2.7x faster than naive at 4K, and 3.0 to 3.4x at 8K to 16K;
    - about 13x faster than SDPA math at 16K;
    - 2.4x faster than SDPA mem-efficient at 64K;
    - 8 to 31x faster than PyTorch Rabe-Staats.
  
  - The naive and SDPA-math lines stop at 16K: they run out of memory at 32K, marked with
  x.
  - Answer: yes, FlashAttention is faster, but only once the sequence is long enough to
  keep the GPU busy.
  
  (b) Peak memory against sequence length: the O(N) against O(N^2) half
  - The y-axis is extra GPU memory used during forward+backward. The grey dotted lines show
  the slopes of linear and quadratic growth.
  - FA2's line runs parallel to the linear guide: 6 MB at 512 up to 772 MB at 64K, doubling
  each time N doubles. That's O(N).
  - Naive runs parallel to the quadratic guide: 18 MB up to 16.4 GB at 16K, quadrupling
  each time N doubles. That's O(N^2). SDPA math does the same at about twice the size.
  - At 16K, FA2 uses 85x less memory than naive.
  - Rabe-Staats is also nearly flat, 0.4 to 0.9 GB. That's linear memory without
  FlashAttention's speed, which is where RQ2 picks up.
  - Answer: yes. Its memory is linear and the baseline's is quadratic, as claimed.
  
  (c) Largest N that fits on a 40 GB A100: the practical consequence of (b)
  - Each bar is the longest sequence that completed forward+backward.
  - Naive and SDPA math stop at 16K, in both causal and non-causal mode.
  - FA2, SDPA flash, SDPA mem-efficient and Rabe-Staats all reach 64K, the top of the
  sweep. Their true limit is higher and wasn't measured, so 64K is a lower bound.
  - Answer: linear memory lets FlashAttention handle at least 4x longer sequences on the
  same GPU.
  
  Throughput isn't drawn in this figure. At a fixed workload, throughput is just the
  inverse of runtime, so (a) shows it implicitly. The explicit throughput curve is RQ4's
  panel (c): FA2 reaches 58 to 70% of the A100's peak TFLOPS from 16K up. You could cite
  that panel for RQ1 as well, or I could add a throughput panel to this figure.
  
  Caveats for the paper
  1. The small-N result comes from our 8-head setup, not from FlashAttention itself.
  2. Memory is measured as peak usage above the input tensors, not total GPU memory.
  3. The 64K bars in (c) are where the sweep ended, not where those methods ran out of
  memory.
  
  In one sentence: from about 4K up, FlashAttention is the fastest exact method (3x over
  naive), its memory grows linearly while naive's grows quadratically (85x less at 16K),
  and so it runs sequences at least 4x longer before running out of memory.


## RQ 2

RQ2 asks how much of FlashAttention's advantage comes from I/O-aware tiling, isolated
  from engineering quality. The figure's approach: write naive, Rabe-Staats and
  FlashAttention with the same Triton tooling and the same tuning search, so differences
  between them come from memory traffic. Then compare our Triton FA2 with official CUDA FA2
  to measure engineering separately. Unless noted, panels use batch 1, 128 heads, d = 128,
  BF16, non-causal.
  
  (a) Runtime with the tooling held equal
  - Forward+backward time against N for Triton naive, Triton Rabe-Staats, Triton FA2 and
  official FA2.
  - Triton FA2 is faster than both Triton baselines at every N. At 4K: naive 41 ms,
  Rabe-Staats 70 ms, Triton FA2 30 ms, official FA2 20 ms.
  - Triton naive runs out of memory at 8K.
  - What it shows: with engineering held equal, tiling still wins. Speed alone can't show
  why, which is what the next three panels do.
  
  (b) Measured HBM traffic against N (Nsight; 8 heads, causal, forward+backward)
  - Naive and Rabe-Staats traffic grows about 4x per N doubling (quadratic), reaching 66
  and 53 GB at 16K.
  - Triton FA2 and official FA2 stay at 0.6 to 0.9 GB at 16K, about 100x less.
  - FA2's traffic grows roughly linearly here because each head's K and V fit in the 40 MB
  L2 cache, so re-reading them doesn't touch HBM. In theory FA's traffic is still
  quadratic, with a much smaller constant.
  - What it shows: the tiled kernels really do avoid HBM. That's the mechanism the claim
  depends on.
  
  (c) Traffic and time at one point (N = 8K, forward)
  - Each bar is the GB moved, labeled with the time:
  
  ┌────────────────────┬──────────┬────────┐
  │ Kernel             │ Traffic  │ Time   │
  ├────────────────────┼──────────┼────────┤
  │ Triton naive       │ 70.5 GB  │ 70 ms  │
  ├────────────────────┼──────────┼────────┤
  │ Triton Rabe-Staats │ 116.5 GB │ 93 ms  │
  ├────────────────────┼──────────┼────────┤
  │ Official FA1       │ 88.1 GB  │ 71 ms  │
  ├────────────────────┼──────────┼────────┤
  │ Triton FA1         │ 88.8 GB  │ 129 ms │
  ├────────────────────┼──────────┼────────┤
  │ Triton FA2         │ 1.1 GB   │ 31 ms  │
  ├────────────────────┼──────────┼────────┤
  │ Official FA2       │ 1.3 GB   │ 24 ms  │
  └────────────────────┴──────────┴────────┘
  
  - FA2 moves 65x less data than naive.
  - Rabe-Staats moves more than naive: it saves memory, but not traffic.
  
  (d) Why the speed follows the traffic: memory-bound against compute-bound
  - Each point is one kernel: bandwidth used (x-axis) against compute throughput (y-axis).
  - The baselines sit in the memory-bound region, using 65 to 81% of the A100's bandwidth:
  naive, Rabe-Staats and official FA1.
  - Both FA2 versions use only 2 to 3% of bandwidth and are limited by compute instead (59
  to 74%).
  - What it shows: the baselines are slow because they wait on memory. FA2 removes that
  wait, which turns the problem from memory-bound into compute-bound. That's I/O-awareness,
  directly measured.
  
  (e) The split: how much is I/O and how much is engineering Read each line as "the
  first-named kernel is X times faster":
  
  ┌────────────────────────────────────────────────┬─────────────┬────────────────────┐
  │                                                │ Forward     │ Forward + backward │
  ├────────────────────────────────────────────────┼─────────────┼────────────────────┤
  │ Triton FA2 vs Triton naive (I/O, same tooling) │ 2.5 to 2.8x │ 1.05 to 1.4x       │
  ├────────────────────────────────────────────────┼─────────────┼────────────────────┤
  │ Triton FA2 vs Triton Rabe-Staats (I/O)         │ 3.4 to 3.6x │ 1.9 to 2.5x        │
  ├────────────────────────────────────────────────┼─────────────┼────────────────────┤
  │ Official FA2 vs Triton FA2 (engineering,       │ 1.2x        │ 1.5 to 1.8x        │
  │ similar traffic)                               │             │                    │
  └────────────────────────────────────────────────┴─────────────┴────────────────────┘
  
  Combining these gives the RQ2 answer:
  - Forward, N = 8K: official FA2 is about 3.1x faster than Triton naive. That's 2.5x from
  I/O times 1.24x from engineering, so about 80% of the speedup is I/O-awareness (measured
  on a log scale).
  - Forward+backward, N = 4K: about 2.1x total. That's 1.35x from I/O times 1.5x from
  engineering, so only about 40% is I/O. Our Triton FA2 backward is weaker than the
  official CUDA one, so engineering carries more of the gap there.
  
  (f) The same mechanism, FA1 to FA2
  - Official FA1 moves 69x more traffic than FA2, because it reads and writes its running
  output in HBM on every step, and it's about 3x slower.
  - Our Triton FA1 reproduces FA1's traffic almost exactly (88.8 against 88.1 GB), which
  validates the Triton versions.
  
  The answer, in one sentence With the tooling and tuning held equal, FlashAttention's
  forward speedup comes mostly (about 80%) from I/O-awareness: it cuts HBM traffic by 65x
  or more and becomes compute-bound. Kernel engineering adds 1.2 to 1.8x at similar
  traffic, which accounts for a larger share of the training pass.
  
  Caveats for the paper
  1. The forward+backward I/O share is a lower bound, because our Triton FA2 backward isn't
  as well engineered as official FA2's.
  2. Our Rabe-Staats writes FP32 chunk scores. That's an implementation choice that adds
  traffic, so naive is the cleaner I/O baseline.
  3. Nsight counts HBM bytes only. The L2 and shared-memory traffic was only measured for
  three backward kernels.
  4. Panel (b) uses 8 heads, the others use 128. The trends match, but the numbers aren't
  comparable across them.
  5. Naive runs out of memory at 8K for forward+backward, so that comparison stops at 4K.
  6. One GPU, one session.

  ## RQ3

  The figure's three panels together answer RQ3: does FlashAttention match approximate
  attention's efficiency without approximation error? I checked the numbers below against
  the A100 tables, and the Performer line uses the fixed (torch.split) version. At 64K
  causal it is about 511 ms, not 10.8 s.
  
  (a) Non-causal runtime, forward + backward: the efficiency half
  - FA2 beats Performer up to N=16K (20 against 31 ms), and Performer wins from 32K (63
  against 77 ms). At 64K Performer is 2.3× faster.
  - Linformer is level with FA2 at about 2K, then pulls away (about 48× faster at 64K).
  - For RQ3: FA2 matches approximate attention only up to roughly 16K to 32K. Past that,
  the approximations win on speed.
  
  (b) Causal runtime: where the approximation advantage doesn't exist
  - Linformer has no causal mode, so it doesn't appear. FA2 beats Performer at every N up
  to 64K.
  - The ratio shrinks as N grows: 26× at 4K, 11.5× at 16K, 6× at 32K, 3.2× at 64K. If that
  trend held, Performer would cross FA2 somewhere around 250K. That is an extrapolation and
  was not measured.
  - For RQ3: for autoregressive models, the usual case, exact FlashAttention matches or
  beats Performer through 64K.
  
  (c) Accuracy against the FP64 reference: the "no approximation error" half
  - Exact methods sit in the bf16 rounding band (about 0.002 to 0.006). Performer's error
  is 4.5 to 9 and Linformer's grows from 3 to about 90.
  - An error of 1 is the same as outputting zeros, so on these inputs the approximations
  are unusable.
  - For RQ3: FA2 pays no accuracy cost, while the approximations pay a very large one.
  
  Verdict: FA2 doesn't universally match the approximations' speed, but it does so across
  the practical range of lengths, and it does so exactly.
  
  Caveats that belong on or near the figure
  1. The error depends on the inputs. The inputs are untrained and i.i.d. N(0,1). I tested
  Performer on small inputs: its error falls from 4.5 at std 1.0 to 0.05 at std 0.25.
  Linformer wasn't tested this way. The footnote says "untrained" but "far off" could
  mislead.
  2. Error stops at N=4K, but the crossovers in (a) happen at 16K to 32K, where no error
  was measured.
  3. Only forward + backward is shown, and runtime only. Memory is the other half of
  "efficiency" and isn't plotted.
  
  Suggested additions
  - A memory panel. At 64K, FA2 uses 772 MB (forward + backward) against about 13 GB for
  Performer and 1.2 GB for Linformer, so FA2 is already more efficient on memory than both.
  - An accuracy-against-input-scale panel for Performer.
  - A marker or label for the approximate methods.

  ## RQ4

   The three panels together answer RQ4: does the advantage grow with N, and does it differ
  between causal and non-causal masking? I checked the numbers against the A100 tables.
  
  (a) FA2 against PyTorch naive
  - The speedup is about 1× up to 1K and grows to 3 to 13×. Causal forward reaches 12.7× at
  8K and 13.3× at 16K. Non-causal forward + backward is 3.0 to 3.4×.
  - Causal gives the larger advantage. Naive computes the full N×N matrix and then masks
  it, while FA2 skips the masked blocks, so FA2 does roughly half the work.
  - For RQ4: the advantage grows with N and it is larger when causal.
  
  (b) FA2 against PyTorch Rabe-Staats
  - The speedup starts at 3 to 9× at N=512 and reaches 25 to 38× at N=64K. Causal forward +
  backward is the highest at 38×.
  - It levels off after about 8K to 16K. Both methods are quadratic in compute at that
  point, so the ratio settles at the constant gap in efficiency.
  - For RQ4: the advantage grows with N and then saturates. It doesn't grow forever.
  
  (c) FA2's throughput
  - FA2 starts at under 10 TFLOP/s at N=512 and reaches 180 to 220 TFLOP/s (58 to 70% of
  the 312 TFLOP/s peak) from N=16K. Causal is lower (58 to 65% against 68 to 70%
  non-causal).
  - At small N, batch 1 with 8 heads gives FA2 about 32 thread blocks at N=512, fewer than
  the A100's 108 SMs, so it is launch- and occupancy-bound.
  - For RQ4: this explains the growth. Part of the speedup curve rising at small and mid N
  is FA2 becoming efficient, not the baselines getting worse. The plateau after 16K is FA2
  reaching its limit.
  
  Answer to RQ4: yes on both counts. The advantage grows from about 1× to 3 to 13× over
  naive and to 25 to 38× over Rabe-Staats, and then saturates. It is larger with a causal
  mask.
  
  Caveats
  1. Causal's larger advantage is partly because the baselines ignore the mask. A
  causal-aware naive implementation would narrow it. SDPA mem-efficient does skip masked
  blocks, so FA2's lead over it is only about 1.8 to 2.5×.
  2. The small-N flatness comes from the setup. It uses batch 1 and 8 heads, so a larger
  batch would push the growth to smaller N.
  3. Naive's non-causal curve has an irregular point. The speedup is 6.2× at 8K and 4.0× at
  16K, because naive's time grows only 2.4× from 8K to 16K (expected 4×). I don't know why.
  It comes from naive, not FA2.
  4. Panel (c) uses the FA2 paper's convention: forward is 4N²dH FLOPs (halved for causal)
  and backward is 2.5× forward.


  ## RQ6

