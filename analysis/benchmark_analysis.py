"""Benchmark statistics and reproducible figures.

# AI-assisted (ChatGPT): initial implementation; review and test before submission.

Input CSV: one row per repeated measurement with columns:
method,gpu,precision,mask,pass_type,N,batch_size,head_dim,repeat,runtime_ms,peak_memory_mb
Optional: numerical_error. Only comparable configurations are paired for speedup.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KEYS = ['method','gpu','precision','mask','pass_type','N','batch_size','head_dim']
PAIR_KEYS = [x for x in KEYS if x != 'method']
REQUIRED = KEYS + ['repeat','runtime_ms','peak_memory_mb']


def load_runs(path):
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED)-set(df.columns))
    if missing:
        raise ValueError(f'Missing required CSV columns: {missing}')
    for col in ['N','batch_size','head_dim','repeat','runtime_ms','peak_memory_mb']:
        df[col] = pd.to_numeric(df[col], errors='raise')
    if df[REQUIRED].isna().any().any():
        raise ValueError('Required columns contain missing values')
    if (df[['runtime_ms','peak_memory_mb']] < 0).any().any() or (df.runtime_ms <= 0).any():
        raise ValueError('Runtime must be positive; memory cannot be negative')
    if df.duplicated(KEYS+['repeat']).any():
        raise ValueError('Duplicate configuration/repeat IDs')
    return df


def summarize(df):
    grouped = df.groupby(KEYS, dropna=False)
    summary = grouped.agg(repeats=('runtime_ms','size'),median_runtime_ms=('runtime_ms','median'),
                          q1_runtime_ms=('runtime_ms',lambda x:x.quantile(.25)),
                          q3_runtime_ms=('runtime_ms',lambda x:x.quantile(.75)),
                          median_peak_memory_mb=('peak_memory_mb','median'),
                          max_peak_memory_mb=('peak_memory_mb','max')).reset_index()
    summary['iqr_runtime_ms'] = summary.q3_runtime_ms-summary.q1_runtime_ms
    return summary


def speedup_table(summary, baseline='naive'):
    base = summary[summary.method == baseline][PAIR_KEYS+['median_runtime_ms']].rename(columns={'median_runtime_ms':'baseline_median_runtime_ms'})
    merged = summary.merge(base, on=PAIR_KEYS, how='left', validate='many_to_one')
    merged['speedup_vs_'+baseline] = merged.baseline_median_runtime_ms / merged.median_runtime_ms
    return merged


def fit_slopes(summary):
    """Fit log(runtime) against log(N) within each exact method/hardware/configuration group."""
    group_cols = [c for c in KEYS if c != 'N']
    records = []
    for group, subset in summary.groupby(group_cols, dropna=False):
        subset = subset.sort_values('N')
        if subset.N.nunique() < 4:
            continue
        slope, intercept = np.polyfit(np.log(subset.N.to_numpy(float)), np.log(subset.median_runtime_ms.to_numpy(float)), 1)
        records.append(dict(zip(group_cols,group), empirical_runtime_exponent=slope,
                            log_intercept=intercept, num_sizes=subset.N.nunique(),
                            min_N=int(subset.N.min()),max_N=int(subset.N.max())))
    return pd.DataFrame(records, columns=group_cols+['empirical_runtime_exponent','log_intercept','num_sizes','min_N','max_N'])


def plot_runtime(summary, output_dir):
    """Separate plots for each GPU/precision/mask/pass/batch/head combination."""
    grouping = [c for c in PAIR_KEYS if c != 'N']
    for index, (group, subset) in enumerate(summary.groupby(grouping, dropna=False)):
        fig, ax = plt.subplots(figsize=(7,4.5))
        for method, data in subset.groupby('method'):
            data = data.sort_values('N')
            ax.plot(data.N, data.median_runtime_ms, marker='o', label=method)
        ax.set(xscale='log', yscale='log', xlabel='Sequence length N', ylabel='Median runtime (ms)', title='Runtime scaling — matched configuration')
        ax.legend();ax.grid(True,which='both',alpha=.25)
        fig.tight_layout();fig.savefig(output_dir/f'runtime_group_{index:03d}.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description='Analyze actual benchmark CSV results')
    parser.add_argument('csv',help='CSV with one row per repeated run')
    parser.add_argument('--output',default='results/analysis')
    parser.add_argument('--baseline',default='naive')
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    runs=load_runs(args.csv)
    summary=summarize(runs)
    summary.to_csv(out/'summary.csv',index=False)
    speedup_table(summary,args.baseline).to_csv(out/'speedups.csv',index=False)
    fit_slopes(summary).to_csv(out/'empirical_slopes.csv',index=False)
    plot_runtime(summary,out)
    print(f'Analyzed {len(runs)} measured runs; wrote outputs to {out}')
    if (summary.repeats < 10).any():print('WARNING: some configurations have fewer than 10 repeats promised in the proposal')

if __name__=='__main__':main()
