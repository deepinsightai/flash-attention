# University Colab Pro+ GPU Environment

Tested by: Bahaa Al-Juboori  
Date: October 2026

## Available runtime

Google Colab Pro+ provides:

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- VRAM: approximately 96 GB
- PyTorch: 2.11.0+cu130
- CUDA available: True
- PyTorch CUDA version: 13.0
- nvcc: CUDA 13.0, V13.0.88

## Project relevance

This gives the team access to Blackwell-generation NVIDIA hardware through Google Colab Pro+.

The next step is to test whether the FlashAttention-4 CuTe-DSL implementation supports this specific Blackwell GPU.

Because the FlashAttention-4 paper primarily benchmarks B200/GB200 hardware, results from the RTX PRO 6000 Blackwell should be treated as an independent hardware evaluation rather than an exact reproduction of the paper.