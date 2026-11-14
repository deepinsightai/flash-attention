# 3 Mathematical and Algorithmic Preliminaries

This section fixes notation, states the computation that every method in this study performs or approximates, and derives the time, memory, and memory-traffic (IO) costs that our experiments measure. We first describe standard attention and its costs (§3.2), then the two approximate baselines (§3.3), the online-softmax identity that underlies all exact memory-efficient methods (§3.4), FlashAttention (§3.5), and its successors (§3.6). Table 1 and Table 2 (§3.7) summarize the comparison.

## 3.1 Notation

We consider a single attention head. Let $N$ be the sequence length and $d$ the head dimension, with queries, keys, and values $Q, K, V \in \mathbb{R}^{N \times d}$ whose rows are $q_i, k_j, v_j$. On a GPU, data resides in large off-chip high-bandwidth memory (HBM) and is computed on in a small on-chip SRAM of $M$ elements, where $d \le M \le Nd$. Costs for a batch of $B$ sequences with $H$ heads are multiplied by $BH$, which we omit. All complexities are stated for the forward pass unless noted.

## 3.2 Standard Attention

Scaled dot-product attention [9] computes

$$
S = \frac{QK^{\top}}{\sqrt{d}} \in \mathbb{R}^{N \times N}, \qquad
P = \operatorname{softmax}(S), \qquad
O = PV \in \mathbb{R}^{N \times d},
$$

where the softmax is applied row-wise. For numerical stability it is evaluated with the row maximum $m_i = \max_j S_{ij}$ subtracted, which leaves $P$ unchanged:

$$
P_{ij} = \frac{e^{S_{ij} - m_i}}{\ell_i}, \qquad \ell_i = \sum_{l=1}^{N} e^{S_{il} - m_i}.
$$

The two matrix products each cost $2N^2 d$ floating-point operations (FLOPs), so the forward pass costs approximately $4N^2 d$ FLOPs. A causal mask sets $S_{ij} = -\infty$ for $j > i$; this halves the useful work, although a standard implementation computes the full matrix and masks it afterwards.

**Backward pass.** Given the upstream gradient $\mathrm{d}O$, and defining $D_i = \sum_k \mathrm{d}O_{ik}\, O_{ik}$, the gradients are

$$
\mathrm{d}V = P^{\top}\mathrm{d}O, \qquad
\mathrm{d}P = \mathrm{d}O\, V^{\top}, \qquad
\mathrm{d}S_{ij} = P_{ij}\left(\mathrm{d}P_{ij} - D_i\right), \qquad
\mathrm{d}Q = \frac{\mathrm{d}S\, K}{\sqrt{d}}, \qquad
\mathrm{d}K = \frac{\mathrm{d}S^{\top} Q}{\sqrt{d}}.
$$

The expression for $\mathrm{d}S$ applies the softmax Jacobian row by row, using the identity $\sum_j P_{ij}\,\mathrm{d}P_{ij} = D_i$. This identity lets $D$ be computed from $O$ and $\mathrm{d}O$ alone, which later methods exploit to avoid storing $P$. The backward pass requires five products of size $N \times N \times d$, about $2.5\times$ the forward FLOPs. A standard implementation stores $P$ from the forward pass, requiring $\Theta(N^2)$ memory.

**Memory traffic.** A standard implementation writes $S$ to HBM, reads it to compute the softmax, writes $P$, and reads $P$ again to form $PV$. This requires $\Theta(Nd + N^2)$ HBM accesses [4]. Because the softmax performs only $O(1)$ arithmetic operations per element read, it is memory-bound: on modern GPUs its runtime is set by HBM bandwidth rather than arithmetic throughput [5]. This observation motivates every exact method below.

## 3.3 Approximate Attention

The first response to the quadratic cost replaced softmax attention with a cheaper function that approximates it. Both approximate baselines in this study reach time linear in $N$ by never forming an $N \times N$ matrix. Their error is controlled by a rank parameter rather than being zero.

**Performer.** Performer [2] views the unnormalized attention weight as a kernel $\kappa(q, k) = \exp(q^{\top}k)$, with $q$ and $k$ pre-scaled by $d^{-1/4}$ so that the $1/\sqrt{d}$ factor is absorbed. Its FAVOR+ mechanism approximates the kernel by an inner product of positive random features:

$$
\varphi(x) = \frac{1}{\sqrt{m}}\, e^{-\lVert x \rVert^2/2} \left[ e^{w_1^{\top}x}, \ldots, e^{w_m^{\top}x} \right]^{\top}, \qquad w_i \sim \mathcal{N}(0, I_d),
$$

with the $w_i$ orthogonalized. This estimator is unbiased, $\mathbb{E}[\varphi(q)^{\top}\varphi(k)] = \exp(q^{\top}k)$, and positivity keeps the estimated weights non-negative. Writing $Q' = \varphi(Q)$ and $K' = \varphi(K) \in \mathbb{R}^{N \times m}$, associativity of matrix multiplication allows

$$
\widehat{O} = \widehat{D}^{-1} \left( Q' \left( K'^{\top} V \right) \right), \qquad
\widehat{D} = \operatorname{diag}\left( Q' \left( K'^{\top} \mathbf{1}_N \right) \right).
$$

Forming the $m \times d$ product $K'^{\top}V$ first avoids the $N \times N$ matrix entirely, giving $\Theta(Nmd)$ time and $\Theta(Nm + md)$ extra memory. The estimator's variance decreases as $1/m$, and $m = \Theta(d \log d)$ features are used in practice. Because $\widehat{O}$ is a ratio of unbiased estimates, it is itself only approximately unbiased. Causal masking requires prefix sums over $K'^{\top}V$, which are either sequential or memory-intensive in practice.

**Linformer.** Linformer [10] assumes $P$ is approximately low rank and projects keys and values along the sequence dimension with matrices $E, F \in \mathbb{R}^{k \times N}$:

$$
\widehat{O} = \operatorname{softmax}\left( \frac{Q (EK)^{\top}}{\sqrt{d}} \right) (FV), \qquad Q(EK)^{\top} \in \mathbb{R}^{N \times k}.
$$

This costs $\Theta(Nkd)$ time and $\Theta(Nk)$ memory for the score matrix. A Johnson–Lindenstrauss argument shows that $k = \Theta(d/\varepsilon^2)$ suffices for $\varepsilon$-approximation with high probability, independent of $N$ [10]. Two properties constrain its use in this benchmark. First, $E$ and $F$ are learned parameters sized to a fixed $N$; in an untrained layer, the approximation error depends on their initialization, which we fix to scaled Gaussian random projections. Second, every row of $EK$ mixes all $N$ positions, including future ones, so Linformer does not support causal masking and is excluded from causal experiments.

## 3.4 Online Softmax and Exact Memory-Efficient Attention

The second response keeps attention exact and changes the order of evaluation. Its foundation is that softmax can be computed in a single streaming pass.

**Online softmax.** Milakov and Gimelshein [6] compute the maximum and normalizer of a sequence $x_1, \ldots, x_N$ in one pass:

$$
m_j = \max(m_{j-1}, x_j), \qquad
\ell_j = \ell_{j-1}\, e^{m_{j-1} - m_j} + e^{x_j - m_j}, \qquad
m_0 = -\infty,\ \ell_0 = 0.
$$

By induction, $\ell_j = \sum_{l \le j} e^{x_l - m_j}$; the factor $e^{m_{j-1} - m_j} \le 1$ rescales the accumulated sum whenever a new maximum appears.

**A merge operator for attention.** We extend this to attention for a fixed query $q$ with scores $s_j = q^{\top}k_j/\sqrt{d}$. We say a triple $(c, \ell, o)$ with $c \in \mathbb{R}$, $\ell \in \mathbb{R}$, $o \in \mathbb{R}^d$ *represents* a set of key indices $A$ if

$$
\ell = \sum_{j \in A} e^{s_j - c}, \qquad o = \sum_{j \in A} e^{s_j - c}\, v_j .
$$

The attention output over $A$ is then $o/\ell$, and this ratio does not depend on the reference value $c$. Choosing $c = \max_{j \in A} s_j$ only keeps the exponentials in floating-point range. For triples representing disjoint sets $A$ and $B$, define

$$
(c_a, \ell_a, o_a) \oplus (c_b, \ell_b, o_b) = \left( c,\ e^{c_a - c}\,\ell_a + e^{c_b - c}\,\ell_b,\ e^{c_a - c}\, o_a + e^{c_b - c}\, o_b \right), \qquad c = \max(c_a, c_b).
$$

**Lemma 1.** If $(c_a, \ell_a, o_a)$ represents $A$ and $(c_b, \ell_b, o_b)$ represents $B$ with $A \cap B = \emptyset$, then their merge represents $A \cup B$. The operator $\oplus$ is associative and commutative.

*Proof.* Multiplying a triple's $\ell$ and $o$ by $e^{c_a - c}$ re-expresses the same sums with reference $c$, since $e^{c_a - c}\, e^{s_j - c_a} = e^{s_j - c}$. The merged sums are therefore the sums over $A$ and $B$, both referenced to $c$, which together are the sums over $A \cup B$. Associativity and commutativity follow from those of addition and $\max$. $\square$

Lemma 1 implies that the keys may be partitioned into blocks of any size and the partial results combined in any order or tree, yielding exact attention. The only discrepancy is floating-point rounding, because the order of summation changes. Lemma 1 is the correctness argument for every method in §3.4–§3.6.

**Rabe–Staats attention.** Rabe and Staats [7] implement this idea in a high-level framework. For a single query, maintaining running sums over the keys requires $O(1)$ memory, and processing queries sequentially gives $O(\log N)$ memory for self-attention. Their practical implementation processes queries and keys in chunks of size about $\sqrt{N}$, giving $O(\sqrt{N})$ memory while retaining parallelism, and uses gradient checkpointing so that the backward pass recomputes each chunk rather than storing it. Time remains $\Theta(N^2 d)$. Crucially, the method is not IO-aware: each chunk's scores are produced by a separate framework operation and written to HBM before the next operation reads them, so total HBM traffic remains $\Theta(N^2)$ even though peak memory is small. This makes it the appropriate control for isolating the contribution of IO-aware tiling (RQ2).

## 3.5 FlashAttention

FlashAttention [4] fuses the computation of §3.4 into a single GPU kernel that keeps every tile in on-chip SRAM, so that no $N \times N$ quantity is ever written to HBM.

**Forward pass.** Choose block sizes $B_c = \lceil M/4d \rceil$ and $B_r = \min(\lceil M/4d \rceil, d)$ so that one block each of $Q$, $K$, $V$, and the score tile fit in SRAM together. Partition $Q$ and $O$ into $T_r = \lceil N/B_r \rceil$ row blocks and $K, V$ into $T_c = \lceil N/B_c \rceil$ column blocks, and keep row statistics $m, \ell \in \mathbb{R}^N$ in HBM. For each key/value block $j$ (outer loop) and query block $i$ (inner loop), the kernel loads the blocks into SRAM, computes the tile $S_{ij} = Q_i K_j^{\top}$ with its row maximum $\tilde{m}_{ij}$, $\tilde{P}_{ij} = \exp(S_{ij} - \tilde{m}_{ij})$, and row sum $\tilde{\ell}_{ij}$, and updates

$$
m_i^{\mathrm{new}} = \max(m_i, \tilde{m}_{ij}), \qquad
\ell_i^{\mathrm{new}} = e^{m_i - m_i^{\mathrm{new}}}\, \ell_i + e^{\tilde{m}_{ij} - m_i^{\mathrm{new}}}\, \tilde{\ell}_{ij},
$$

$$
O_i \leftarrow \operatorname{diag}(\ell_i^{\mathrm{new}})^{-1} \left( \operatorname{diag}(\ell_i)\, e^{m_i - m_i^{\mathrm{new}}}\, O_i + e^{\tilde{m}_{ij} - m_i^{\mathrm{new}}}\, \tilde{P}_{ij} V_j \right).
$$

This is the merge of Lemma 1 applied block-wise, with $O_i$ kept normalized after every step. Causal masking is applied within each tile.

**Backward pass.** FlashAttention stores only $O$ and the statistics $(m, \ell)$. The backward pass reloads tiles of $Q, K, V, O, \mathrm{d}O$, recomputes $S_{ij}$ and $P_{ij} = \operatorname{diag}(\ell_i)^{-1}\exp(S_{ij} - m_i)$ in SRAM, and applies the gradient formulas of §3.2 tile by tile, using $D_i = \sum_k \mathrm{d}O_{ik}\, O_{ik}$. Recomputation increases FLOPs but eliminates the $\Theta(N^2)$ reads of a stored $P$, which reduces wall-clock time because the computation is memory-bound.

**Complexity.** Dao et al. [4] prove the following:

- **Correctness and memory.** FlashAttention returns $O = \operatorname{softmax}(QK^{\top}/\sqrt{d})\,V$ using $O(N^2 d)$ FLOPs and $O(N)$ memory beyond its inputs and outputs.
- **IO.** It requires $\Theta(N^2 d^2 M^{-1})$ HBM accesses, compared with $\Theta(Nd + N^2)$ for standard attention. The backward pass has the same IO complexity.
- **Lower bound.** No exact attention algorithm can use $o(N^2 d^2 M^{-1})$ HBM accesses for all $M \in [d, Nd]$, so FlashAttention is IO-optimal in this sense.

The IO bound follows from a counting argument. Each block of $K$ and $V$ is read once, contributing $\Theta(Nd)$. For each of the $T_c = \Theta(Nd/M)$ key/value blocks, all of $Q$ and $O$ are streamed through SRAM, contributing $\Theta(Nd)$ per block and $\Theta(N^2 d^2/M)$ in total. For typical head dimensions ($d = 64$–$128$) and SRAM sizes of order $10^5$ elements per streaming multiprocessor, $d^2/M < 1$, so FlashAttention performs substantially fewer HBM accesses than standard attention despite executing more FLOPs.

## 3.6 The FlashAttention Lineage

FlashAttention-2, -3, and -4 compute the same exact function with the same asymptotic costs. Each targets the bottleneck that dominates on a newer GPU generation, and the mathematical changes are confined to how the online softmax of Lemma 1 is scheduled, referenced, and rounded.

**FlashAttention-2.** FlashAttention-2 [3] reduces non-matrix-multiply FLOPs, which execute up to $16\times$ more slowly than matrix multiplies on GPU tensor cores (on an A100, 312 TFLOP/s for FP16 matrix multiplies versus 19.5 TFLOP/s for other FP32 operations). It keeps the output unnormalized until the end. For query block $i$ over key blocks $j = 1, \ldots, T_c$:

$$
m^{(j)} = \max\left( m^{(j-1)}, \operatorname{rowmax}(S_{ij}) \right), \qquad
\tilde{P}_{ij} = e^{S_{ij} - m^{(j)}}, \qquad
\ell^{(j)} = e^{m^{(j-1)} - m^{(j)}}\, \ell^{(j-1)} + \operatorname{rowsum}(\tilde{P}_{ij}),
$$

$$
\tilde{O}^{(j)} = \operatorname{diag}\left( e^{m^{(j-1)} - m^{(j)}} \right) \tilde{O}^{(j-1)} + \tilde{P}_{ij} V_j, \qquad
O_i = \operatorname{diag}\left( \ell^{(T_c)} \right)^{-1} \tilde{O}^{(T_c)}.
$$

It stores only the log-sum-exp $L_i = m_i + \log \ell_i$, from which the backward pass reconstructs $P_{ij} = \exp(S_{ij} - L_i)$. It also swaps the loop order so that each thread block owns one query block, parallelizes over the sequence dimension in addition to batch and heads, and skips tiles that lie entirely above the diagonal under causal masking. These changes yield roughly a $2\times$ speedup over FlashAttention, reaching 50–73% of peak A100 throughput [3].

**FlashAttention-3.** On the H100, FP16 matrix multiplies run at 989 TFLOP/s while the exponential unit delivers 3.9 TFLOP/s, so the exponentials of the softmax can stall the matrix multiplies [8]. FlashAttention-3 retains the FlashAttention-2 recurrence and overlaps the two: producer warps load tiles asynchronously while consumer warps compute, two warpgroups alternate between softmax and matrix multiply ("ping-pong" scheduling), and the softmax of one block overlaps the score computation of the next. For FP8, it adds per-block quantization scales and *incoherent processing*: $Q$ and $K$ are multiplied by the same random orthogonal matrix $\mathbf{M} = d^{-1/2} H_d \operatorname{diag}(\pm 1)$, where $H_d$ is a Hadamard matrix, so that

$$
(Q\mathbf{M})(K\mathbf{M})^{\top} = Q \mathbf{M}\mathbf{M}^{\top} K^{\top} = QK^{\top}.
$$

The scores are unchanged in exact arithmetic, but outlier coordinates are spread across all $d$ dimensions, reducing FP8 quantization error. The transform costs $O(d \log d)$ per row. FlashAttention-3 reports a 1.5–2.0$\times$ speedup over FlashAttention-2 in FP16, reaching 740 TFLOP/s, and approximately 1.2 PFLOP/s in FP8 with 2.6$\times$ lower error than a baseline FP8 attention [8].

**FlashAttention-4.** On Blackwell GPUs, tensor-core throughput approximately doubled relative to Hopper, while shared-memory bandwidth and exponential throughput scaled less [11]. FlashAttention-4 makes two changes to the softmax arithmetic. First, *conditional rescaling* keeps a stale reference $c$ and rescales only when the running maximum exceeds it by more than a threshold $\tau$:

$$
\text{if } m^{(j)} - c > \tau: \quad \tilde{O} \leftarrow e^{c - m^{(j)}}\, \tilde{O}, \quad \ell \leftarrow e^{c - m^{(j)}}\, \ell, \quad c \leftarrow m^{(j)}; \qquad \tilde{P}_{ij} = e^{S_{ij} - c}.
$$

By Lemma 1, the output $\tilde{O}/\ell$ is independent of the reference $c$, so skipping a rescale leaves the result exact; the threshold bounds $e^{S_{ij} - c} \le e^{\tau}$, preventing overflow. Second, a fraction of exponentials is computed in software on the fused multiply-add units, relieving the exponential unit. Using range reduction $x = n + f$ with $n = \lfloor x \rfloor$ and $f \in [0, 1)$,

$$
2^{x} = 2^{n} \cdot 2^{f}, \qquad 2^{f} \approx 1 + p_1 f + p_2 f^2 + p_3 f^3,
$$

where the factor $2^n$ is applied by writing $n$ into the floating-point exponent. FlashAttention-4 further stores accumulators in Blackwell's tensor memory and pairs thread blocks in the backward pass to halve shared-memory traffic. It reports up to approximately 1.6 PFLOP/s in BF16 on a B200 (71% utilization), up to $1.3\times$ faster than cuDNN 9.13 and $2.7\times$ faster than Triton [11]. The reported evaluation covers BF16 and FP16 only.

## 3.7 Complexity Summary

Tables 1 and 2 summarize the methods. Only the approximate methods reduce time below $\Theta(N^2 d)$. The exact methods share the same FLOP count and differ in memory and HBM traffic, which is why our evaluation reports achieved throughput ($4N^2 d$ divided by runtime) alongside raw runtime.

**Table 1: Time, memory, and memory traffic per attention head (forward pass unless noted).** $m$: random features; $k$: projected length; $M$: SRAM size in elements.

| Method | Exact | Time (FLOPs) | Extra memory (forward) | Stored for backward | HBM accesses |
| --- | --- | --- | --- | --- | --- |
| Standard [9] | Yes | $\Theta(N^2 d)$ | $\Theta(N^2)$ | $\Theta(N^2)$ ($P$) | $\Theta(Nd + N^2)$ |
| Performer [2] | No | $\Theta(Nmd)$ | $\Theta(Nm + md)$ | $\Theta(Nm + md)$ | $\Theta(Nm + Nd)$$^{\dagger}$ |
| Linformer [10] | No | $\Theta(Nkd)$ | $\Theta(Nk)$ | $\Theta(Nk)$ | $\Theta(Nk + Nd)$$^{\dagger}$ |
| Rabe–Staats [7] | Yes | $\Theta(N^2 d)$ | $O(\sqrt{N})$ chunked | $O(\sqrt{N})$ (checkpointing) | $\Theta(N^2)$ (unfused) |
| FlashAttention [4] | Yes | $\Theta(N^2 d)$ | $\Theta(N)$ ($m, \ell$) | $\Theta(N)$ ($m, \ell$) | $\Theta(N^2 d^2/M)$ |
| FlashAttention-2 [3] | Yes | $\Theta(N^2 d)$ | $\Theta(N)$ | $\Theta(N)$ ($L$) | $\Theta(N^2 d^2/M)$ |
| FlashAttention-3 [8] | Yes$^{\ddagger}$ | $\Theta(N^2 d)$ | $\Theta(N)$ | $\Theta(N)$ | $\Theta(N^2 d^2/M)$ |
| FlashAttention-4 [11] | Yes | $\Theta(N^2 d)$ | $\Theta(N)$ | $\Theta(N)$ | $\Theta(N^2 d^2/M)$ |

$^{\dagger}$ Our estimate; not analyzed in the original papers. $^{\ddagger}$ Exact in FP16/BF16; FP8 mode adds quantization error. Lower bound for any exact method: $\Omega(N^2 d^2/M)$ HBM accesses for $M \in [d, Nd]$ [4].

**Table 2: Key idea, causal-mask support, and target hardware.**

| Method | Year | Key mathematical idea | Causal mask | Target hardware |
| --- | --- | --- | --- | --- |
| Standard [9] | 2017 | Direct evaluation of $\operatorname{softmax}(QK^{\top}/\sqrt{d})V$ | Mask after full computation | Any |
| Linformer [10] | 2020 | Low-rank projection $K \to EK$, $V \to FV$ | Not supported | Any |
| Performer [2] | 2020 | Random features $\exp(q^{\top}k) \approx \varphi(q)^{\top}\varphi(k)$; reordered product | Prefix sums | Any |
| Rabe–Staats [7] | 2021 | Chunked online softmax with checkpointing | Supported | Any |
| FlashAttention [4] | 2022 | IO-aware tiling with block-wise merge; recomputation in backward | Within tiles | Ampere |
| FlashAttention-2 [3] | 2023 | Deferred normalization; log-sum-exp statistic; query-outer loop | Skips masked tiles | Ampere |
| FlashAttention-3 [8] | 2024 | Asynchronous overlap; incoherent processing for FP8 | Supported | Hopper |
| FlashAttention-4 [11] | 2026 | Conditional rescaling; polynomial exponential | Supported | Blackwell |
