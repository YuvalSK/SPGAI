import pandas as pd, numpy as np, glob, os

OUT = os.path.dirname(os.path.abspath(__file__))
EXCL = 'estrogen_receptor_alpha'

sc = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{OUT}/scores_rep*.csv'))])
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{OUT}/differences_R4_rep*.csv'))])
sc = sc[sc.score == 'R4']

# per test x truth x train x model x marker: mean, sd over reps (ERα kept here, flagged in the summary)
pm = sc.groupby(['test', 'truth', 'train', 'model', 'marker']).value.agg(['mean', 'std', 'min', 'max']).reset_index()
pm.to_csv(f'{OUT}/r4_per_marker_mean_sd.csv', index=False)

# median over the 10 markers (ERα excluded), per rep, then mean/sd over reps
sc10 = sc[sc.marker != EXCL]
med = sc10.groupby(['test', 'truth', 'train', 'model', 'rep']).value.median().reset_index()
medw = med.pivot_table(index=['test', 'truth', 'train', 'model'], columns='rep', values='value')
medw['mean'] = medw[[0, 1, 2]].mean(1); medw['sd'] = medw[[0, 1, 2]].std(1)
medw.to_csv(f'{OUT}/median_r4_by_rep.csv')

# per-marker calls in each rep, and better/worse counts (ERα excluded)
df10 = df[df.marker != EXCL]
key = ['kind', 'test', 'truth', 'train_a', 'model_a', 'train_b', 'model_b']
calls = df10.pivot_table(index=key + ['marker'], columns='rep', values='call', aggfunc='first')
calls['diff_mean'] = df10.groupby(key + ['marker'])['diff'].mean()
calls['diff_sd'] = df10.groupby(key + ['marker'])['diff'].std()
calls.to_csv(f'{OUT}/calls_by_rep.csv')

cnt = df10.groupby(key + ['rep', 'call']).size().unstack('call', fill_value=0)
for c in ('better', 'n.s.', 'worse'):
    if c not in cnt: cnt[c] = 0
cnt = cnt[['better', 'n.s.', 'worse']].reset_index()
w = cnt.pivot_table(index=key, columns='rep', values=['better', 'worse'])
w.columns = [f'{a}_rep{b}' for a, b in w.columns]

def consistent(g):
    out = []
    for m, gg in g.groupby('marker'):
        c = set(gg.call)
        if len(c) == 1 and c != {'n.s.'}:
            out.append(f'{m}:{c.pop()}')
    return '; '.join(out)

w['median_diff_mean'] = df10.groupby(key + ['rep'])['diff'].median().groupby(key).mean()
w['same_call_all_3'] = df10.groupby(key)[['marker', 'call']].apply(consistent)
w.to_csv(f'{OUT}/call_counts_by_rep.csv')

with pd.option_context('display.width', 250, 'display.max_rows', 500):
    print(medw.round(3).to_string())
