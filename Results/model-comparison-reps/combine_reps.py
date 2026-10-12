import pandas as pd, numpy as np, json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = r'C:/Users/Yuval/OneDrive - Bar Ilan University/Academic/Postdoc/Multiplexed analyses/IMC/Results'
OUT = f'{R}/model-comparison-reps'
EXCL = 'estrogen_receptor_alpha'

sc = [pd.read_csv(f'{R}/model-comparison/scores.csv').assign(rep=0)]
df = [pd.read_csv(f'{R}/model-comparison/differences_R4.csv').assign(rep=0)]
for k in (1, 2):
    sc.append(pd.read_csv(f'{OUT}/scores_rep{k}.csv').assign(rep=k))
    df.append(pd.read_csv(f'{OUT}/differences_R4_rep{k}.csv').assign(rep=k))
sc = pd.concat(sc); df = pd.concat(df)
sc = sc[(sc.score == 'R4') & (sc.marker != EXCL)]
df = df[df.marker != EXCL]

# per marker x model x cohort: mean, sd over reps
pm = sc.groupby(['cohort', 'model', 'marker']).value.agg(['mean', 'std', 'min', 'max']).reset_index()
pm.to_csv(f'{OUT}/r4_per_marker_mean_sd.csv', index=False)

# median over markers, per rep, then mean/sd over reps
med = sc.groupby(['cohort', 'model', 'rep']).value.median().reset_index()
medw = med.pivot_table(index=['cohort', 'model'], columns='rep', values='value')
medw['mean'] = medw[[0, 1, 2]].mean(1); medw['sd'] = medw[[0, 1, 2]].std(1)
medw.to_csv(f'{OUT}/median_r4_by_rep.csv')
print(medw.round(3).to_string())

# rank of each model per rep (median R4), per cohort
for c in med.cohort.unique():
    m = med[(med.cohort == c) & ~med.model.str.startswith('ResNet')]
    print('\n', c)
    for r in (0, 1, 2):
        mm = m[m.rep == r].sort_values('value', ascending=False)
        print(' rep', r, ' > '.join(f'{a} {v:.3f}' for a, v in zip(mm.model, mm.value)))

# best per-cell model per marker per rep
pc = sc[~sc.model.str.startswith('ResNet')]
best = pc.loc[pc.groupby(['cohort', 'marker', 'rep']).value.idxmax().values] if False else \
    pc.sort_values('value').groupby(['cohort', 'marker', 'rep']).tail(1)
bw = best.pivot_table(index=['cohort', 'marker'], columns='rep', values='model', aggfunc='first')
bw.to_csv(f'{OUT}/best_model_per_marker.csv')
print('\n', bw.to_string())

# call consistency
calls = df.pivot_table(index=['cohort', 'model', 'vs', 'marker'], columns='rep', values='call', aggfunc='first')
calls['pattern'] = calls[[0, 1, 2]].apply(lambda r: '/'.join(r), axis=1)
calls['consistent'] = calls[[0, 1, 2]].nunique(1) == 1
calls.to_csv(f'{OUT}/calls_by_rep.csv')
cnt = df.groupby(['cohort', 'model', 'vs', 'rep']).call.value_counts().unstack(fill_value=0)
for k in ('better', 'n.s.', 'worse'):
    if k not in cnt: cnt[k] = 0
cnt = cnt[['better', 'n.s.', 'worse']].reset_index()
cs = cnt.groupby(['cohort', 'model', 'vs']).apply(
    lambda g: pd.Series({k: '/'.join(str(int(x)) for x in g.sort_values('rep')[k]) for k in ('better', 'n.s.', 'worse')}))
agree = calls.groupby(['cohort', 'model', 'vs']).consistent.sum().rename('same call in 3/3 (of 10)')
# robust = same non-n.s. call in all 3 reps
rob = calls[calls.consistent & (calls[0] != 'n.s.')].reset_index()
robs = rob.groupby(['cohort', 'model', 'vs', 0]).marker.apply(lambda x: ', '.join(sorted(x))).unstack(fill_value='')
robs.columns = [f'{c} in 3/3' for c in robs.columns]
summ = cs.join(agree).join(robs)
summ.to_csv(f'{OUT}/call_counts_by_rep.csv')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
print('\n', summ.to_string())

# mean diff over reps per comparison x marker
dm = df.groupby(['cohort', 'model', 'vs', 'marker'])['diff'].agg(['mean', 'std', 'min', 'max']).reset_index()
dm.to_csv(f'{OUT}/diff_r4_mean_sd.csv', index=False)
for (c, a, b) in [('jackson', 'GraphSAGE', 'MLP'), ('jackson', 'Table + ring', 'Table'), ('danenberg', 'MLP', 'Table'),
                  ('danenberg', 'GraphSAGE', 'MLP'), ('danenberg', 'GCN', 'MLP'), ('jackson', 'GCN', 'MLP')]:
    x = dm[(dm.cohort == c) & (dm.model == a) & (dm.vs == b)]
    print(f'\n{c} {a} - {b}:', '; '.join(f'{m} {u:+.3f}±{s:.3f}' for m, u, s in zip(x.marker, x['mean'], x['std'])))

# figure: per-marker mean R4 ± SD (range bars) per model, two cohorts
nice = {'c_erb_b_2_her2': 'HER2', 'cd20': 'CD20', 'cd3': 'CD3', 'cd45': 'CD45', 'cd68': 'CD68',
        'cytokeratin_5': 'CK5', 'cytokeratin_8_18': 'CK8/18', 'ki_67': 'Ki-67', 'pan_cytokeratin': 'panCK', 'sma': 'SMA'}
models = ['Table', 'Table + ring', 'MLP', 'GCN', 'GraphSAGE', 'ResNet-UNet residual', 'ResNet-UNet gamma', 'ResNet-UNet immuvis']
cols = dict(zip(models, ['#4c72b0', '#64b5cd', '#dd8452', '#c44e52', '#55a868', '#8c8c8c', '#b0b0b0', '#5f5f5f']))
markers = sorted(nice, key=lambda m: -pm[(pm.cohort == 'Jackson (held-out patients)') & (pm.model == 'MLP') & (pm.marker == m)]['mean'].iloc[0])
fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
for ax, c, lab in zip(axes, ['Jackson (held-out patients)', 'Danenberg (cross-cohort)'],
                      ['Jackson 2020, held-out patients (3 splits × seeds)', 'Danenberg 2022, cross-cohort (3 seeds)']):
    present = [m for m in models if ((pm.cohort == c) & (pm.model == m)).any()]
    w = 0.8 / len(models)
    for i, mo in enumerate(present):
        pass
        x = pm[(pm.cohort == c) & (pm.model == mo)].set_index('marker').reindex(markers)
        xs = np.arange(len(markers)) + (i - len(present) / 2 + 0.5) * w
        ax.bar(xs, x['mean'], w, color=cols[mo], label=mo)
        ax.errorbar(xs, x['mean'], yerr=[(x['mean'] - x['min']).clip(lower=0), (x['max'] - x['mean']).clip(lower=0)], fmt='none', ecolor='k', lw=0.8, capsize=1.5)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_ylabel('R4 (per-cell r,\nimage mean removed)')
    ax.text(0.01, 0.97, lab, transform=ax.transAxes, va='top', fontsize=10)
    ax.set_ylim(-0.35, 1.0)
    ax.tick_params(labelbottom=True)
axes[1].set_xticks(np.arange(len(markers))); axes[1].set_xticklabels([nice[m] for m in markers])
axes[0].set_xticks(np.arange(len(markers))); axes[0].set_xticklabels([nice[m] for m in markers])
axes[1].set_xlabel('Marker (bars = mean of 3 replicates; whiskers = min–max)')
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()
fig.savefig(f'{OUT}/fig_model_comparison_reps.png', dpi=150, bbox_inches='tight')
print('fig saved')
