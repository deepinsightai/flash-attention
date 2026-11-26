"""Dense attention dropout pilot; standalone version of the executed notebook.
Run from any directory. Archived CPU results are never overwritten.
"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)

import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

import platform
print("Python:", platform.python_version())
print("CPU threads:", torch.get_num_threads())


import torch
import math

# Select the available device
device = args.device
if device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

# Create a small example
torch.manual_seed(42)

Q = torch.randn(1, 4, 8, device=device)
K = torch.randn(1, 4, 8, device=device)
V = torch.randn(1, 4, 8, device=device)

# Step 1: Calculate attention scores
scores = torch.matmul(Q, K.transpose(-2, -1))

# Step 2: Scale the scores
scores = scores / math.sqrt(Q.shape[-1])

# Step 3: Convert scores to probabilities
weights = torch.softmax(scores, dim=-1)

# Step 4: Calculate attention output
output = torch.matmul(weights, V)

print("Attention weights:")
print(weights)

print("\nAttention output shape:", output.shape)
print("Device:", output.device)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

lengths = [128, 256, 512, 1024]
probabilities = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
seeds = [42, 43, 44, 45, 46]
repeats = 10
warmups = 3
head_dim = 64

# Limit CPU thread variability in the pilot
if device == "cpu":
    torch.set_num_threads(2)
print("Device:", device)
print("CPU threads:", torch.get_num_threads())

def attention(q, k, v, mask, p):
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(q.shape[-1])
    w = torch.softmax(scores, dim=-1)
    if p > 0:
        w = w * mask / (1 - p)
    return torch.matmul(w, v)


def sync():
    if device == "cuda":
        torch.cuda.synchronize()

# Wall-clock latency includes Python dispatch and synchronization

def measure(q, k, v, mask, p):
    for i in range(warmups):
        attention(q, k, v, mask, p)
    sync()
    times = []
    for i in range(repeats):
        sync()
        start = time.perf_counter()
        attention(q, k, v, mask, p)
        sync()
        times.append((time.perf_counter() - start) * 1000)
    return times

import random

records = []
timing_records = []

with torch.no_grad():
    for length in lengths:
        for seed in seeds:
            torch.manual_seed(seed)
            q = torch.randn(1, length, head_dim, device=device)
            k = torch.randn(1, length, head_dim, device=device)
            v = torch.randn(1, length, head_dim, device=device)
            random_values = torch.rand(1, length, length, device=device)
            full_mask = random_values >= 0
            dense_ref = attention(q.double(), k.double(), v.double(), full_mask, 0)

            order = probabilities.copy()
            random.Random(seed).shuffle(order)
            for p in order:
                mask = random_values >= p
                result = attention(q, k, v, mask, p)
                ref = attention(q.double(), k.double(), v.double(), mask, p)
                assert torch.isfinite(result).all().item()
                if p == 0:
                    assert torch.allclose(result.double(), dense_ref, atol=1e-5, rtol=1e-4)
                difference = (result.double() - ref).abs()
                deviation = (ref - dense_ref).abs()
                times = measure(q, k, v, mask, p)
                records.append({"length": length, "seed": seed, "dropout": p,
                    "actual_sparsity": 1 - mask.float().mean().item(),
                    "matched_mae": difference.mean().item(),
                    "matched_max_error": difference.max().item(),
                    "dense_deviation_mae": deviation.mean().item(),
                    "median_ms": np.median(times), "device": device})
                for i in range(len(times)):
                    timing_records.append({"length": length, "seed": seed,
                        "dropout": p, "repeat": i + 1, "runtime_ms": times[i],
                        "device": device})

results = pd.DataFrame(records)
raw_times = pd.DataFrame(timing_records)
print("Configurations:", len(results))
print("Timing measurements:", len(raw_times))
print(results.head().to_string(index=False))

baseline = results[results["dropout"] == 0][["length", "seed", "median_ms"]]
baseline = baseline.rename(columns={"median_ms": "baseline_ms"})
results = results.merge(baseline, on=["length", "seed"])
results["speedup"] = results["baseline_ms"] / results["median_ms"]
results["tokens_per_second"] = results["length"] * 1000 / results["median_ms"]

summary = results.groupby(["length", "dropout"]).agg(
    median_ms=("median_ms", "median"),
    q1_ms=("median_ms", lambda x: x.quantile(0.25)),
    q3_ms=("median_ms", lambda x: x.quantile(0.75)),
    speedup=("speedup", "median"),
    tokens_per_second=("tokens_per_second", "median"),
    actual_sparsity=("actual_sparsity", "mean"),
    matched_mae=("matched_mae", "mean"),
    dense_deviation_mae=("dense_deviation_mae", "mean")
).reset_index()
summary["iqr_ms"] = summary["q3_ms"] - summary["q1_ms"]
print(summary.round(6).to_string(index=False))

for column, label in [("median_ms", "Median latency (ms)"),
                      ("tokens_per_second", "Tokens per second"),
                      ("speedup", "Speedup vs zero dropout"),
                      ("dense_deviation_mae", "MAE vs dense reference"),
                      ("matched_mae", "MAE vs matched FP64 reference")]:
    plt.figure(figsize=(7, 4))
    for length in lengths:
        part = summary[summary["length"] == length]
        plt.plot(part["dropout"], part[column], marker="o", label=str(length))
    if column == "speedup":
        plt.axhline(1, linestyle="--")
    plt.xlabel("Dropout probability")
    plt.ylabel(label)
    plt.title(label + " (" + device + ")")
    plt.legend(title="Sequence length")
    plt.tight_layout()
    plt.savefig(args.output / (column + ".png"), dpi=150)
    plt.close()

for length in lengths:
    part = summary[summary["length"] == length]
    first = part[part["dropout"] == 0].iloc[0]
    last = part[part["dropout"] == 0.9].iloc[0]
    print("Length:", length)
    print("Latency at p=0:", round(first["median_ms"], 4), "ms")
    print("Latency at p=0.9:", round(last["median_ms"], 4), "ms")
    print("Paired speedup:", round(last["speedup"], 4))
    print("Dense output deviation:", round(last["dense_deviation_mae"], 6))
    print("Matched FP64 MAE:", format(last["matched_mae"], ".3e"))
    print()

results.to_csv(args.output / "attention_seed_results.csv", index=False)
raw_times.to_csv(args.output / "attention_raw_timings.csv", index=False)
summary.to_csv(args.output / "attention_summary.csv", index=False)
print("CSV files saved.")
