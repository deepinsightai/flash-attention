"""# AI-assisted (ChatGPT): starter tests; review and extend."""
import pandas as pd
from analysis.benchmark_analysis import summarize,speedup_table,fit_slopes


def test_summary_and_speedup():
    rows=[]
    for method,t in [('naive',20.),('flashattention',10.)]:
        for n in [512,1024,2048,4096]:
            for rep in range(10):
                rows.append(dict(method=method,gpu='A100',precision='BF16',mask='noncausal',pass_type='forward',N=n,batch_size=1,head_dim=128,repeat=rep,runtime_ms=t*(n/512)**2,peak_memory_mb=100.))
    s=summarize(pd.DataFrame(rows))
    x=speedup_table(s)
    flash=x[x.method=='flashattention']
    assert (flash.speedup_vs_naive==2).all()
    slopes=fit_slopes(s)
    assert len(slopes)==2
    assert all(abs(v-2)<1e-9 for v in slopes.empirical_runtime_exponent)


def test_unmatched_hardware_has_no_speedup():
    rows = []
    for method, gpu in [('naive', 'A100'), ('flashattention', 'H100')]:
        rows.append(dict(method=method, gpu=gpu, precision='BF16', mask='noncausal',
                         pass_type='forward', N=512, batch_size=1, head_dim=128,
                         repeat=0, runtime_ms=10., peak_memory_mb=100.))
    summary = summarize(pd.DataFrame(rows))
    result = speedup_table(summary)
    assert pd.isna(result.loc[result.method == 'flashattention', 'speedup_vs_naive']).all()


def test_duplicate_repeats_rejected(tmp_path):
    import pytest
    from analysis.benchmark_analysis import load_runs
    row = dict(method='naive', gpu='A100', precision='BF16', mask='noncausal',
               pass_type='forward', N=512, batch_size=1, head_dim=128,
               repeat=0, runtime_ms=10., peak_memory_mb=100.)
    path = tmp_path / 'duplicate.csv'
    pd.DataFrame([row, row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Duplicate'):
        load_runs(path)
