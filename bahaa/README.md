Bahaa Al-Juboori – FlashAttention Lineage Study
My Responsibility
My primary responsibility in the COMP 9060 group project is the FlashAttention lineage study, comparing FlashAttention-1, FlashAttention-2, FlashAttention-3, and FlashAttention-4 across Ampere, Hopper, and Blackwell GPU generations.

Project Goal
The project investigates whether FlashAttention provides faster and more memory-efficient exact attention than traditional attention methods, and how the FlashAttention family evolved across newer GPU architectures.

My Current Work
1. Blackwell Environment Verification
Verified the university-provided Google Colab Pro+ environment.

Current tested environment:

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
Compute Capability: 12.0
GPU Memory: approximately 95 GB
CUDA: 13.0
PyTorch: 2.11.0+cu130
2. Flash SDPA Compatibility Test
Verified that PyTorch's Flash Attention SDPA backend executes successfully on the Blackwell GPU.

3. Initial Pilot Benchmark
Compared a naive scaled dot-product attention implementation against PyTorch Flash SDPA.

Initial configuration:

Batch size: 1
Attention heads: 8
Sequence length: 1024
Head dimension: 128
Precision: BF16
Initial result:

Naive attention: approximately 0.0528 ms
Flash SDPA: approximately 0.0432 ms
Speedup: approximately 1.22x
This was only a pilot compatibility test and is not considered a final project result.

Next Steps
Build the sequence-length benchmark
Measure runtime and throughput
Measure peak GPU memory
Determine maximum sequence length before OOM
Study FA1, FA2, FA3, and FA4 implementations
Compare results across Ampere, Hopper, and Blackwell
Document limitations of cross-generation comparisons
