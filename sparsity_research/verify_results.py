"""Check archived CSV aggregation without rerunning benchmarks."""
from pathlib import Path
import numpy as np
import pandas as pd

folder = Path(__file__).resolve().parent / "results" / "cpu"
raw = pd.read_csv(folder / "attention_raw_timings.csv")
seeds = pd.read_csv(folder / "attention_seed_results.csv")
summary = pd.read_csv(folder / "attention_summary.csv")
keys = ["length", "seed", "dropout"]
assert len(raw) == 1200 and len(seeds) == 120 and len(summary) == 24
assert not raw.duplicated(keys + ["repeat"]).any()
assert all(not x.isna().any().any() for x in [raw, seeds, summary])
assert (raw.groupby(keys).size() == 10).all()
assert (raw.runtime_ms > 0).all()
assert set(raw.device) == set(seeds.device) == {"cpu"}
med = raw.groupby(keys).runtime_ms.median().rename("recomputed").reset_index()
s = seeds.merge(med, on=keys, validate="one_to_one")
assert len(s) == len(seeds)
np.testing.assert_allclose(s.median_ms, s.recomputed)
b = s[s.dropout == 0][["length", "seed", "recomputed"]].rename(columns={"recomputed": "base"})
s = s.merge(b, on=["length", "seed"], validate="many_to_one")
np.testing.assert_allclose(s.baseline_ms, s.base)
np.testing.assert_allclose(s.speedup, s.base / s.recomputed)
np.testing.assert_allclose(s.tokens_per_second, s.length * 1000 / s.recomputed)
a = s.groupby(["length", "dropout"]).agg(median_ms=("recomputed", "median"), q1_ms=("recomputed", lambda x: x.quantile(.25)), q3_ms=("recomputed", lambda x: x.quantile(.75)), speedup=("speedup", "median"), tokens_per_second=("tokens_per_second", "median"), actual_sparsity=("actual_sparsity", "mean"), matched_mae=("matched_mae", "mean"), dense_deviation_mae=("dense_deviation_mae", "mean")).reset_index()
a["iqr_ms"] = a.q3_ms - a.q1_ms
m = summary.merge(a, on=["length", "dropout"], suffixes=("_saved", "_check"), validate="one_to_one")
assert len(m) == len(summary)
for col in a.columns[2:]:
    np.testing.assert_allclose(m[col + "_saved"], m[col + "_check"], rtol=1e-9, atol=1e-12)
print("PASS: 1200 timings, 120 seed configurations, 24 summaries; all aggregations agree.")
print("Numerical error aggregates checked, not independently recomputed from tensors.")
