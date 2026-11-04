"""iLISI vs cLISI of virtual cells, for every model with saved per-cell predictions; held-out cells and held-out cohort.

A virtual cell is the vector of a model's 10 predicted markers (ERα excluded), in the test cohort's target space.
- iLISI (mixing of virtual with measured cells): measured cells of one half of the test patients and virtual cells of
  the other half, pooled; LISI over the label {measured, virtual}, normalised to (LISI − 1) ∈ [0, 1].
  1 = virtual cells sit among measured cells as if they were measured; 0 = they occupy their own region.
  The two halves have different patients, so the measured-vs-measured reference (ceiling) is below 1.
- cLISI (cell types kept apart): virtual cells only, LISI over the published cell type mapped to 6 coarse classes (as
  `cell-type-graph`), normalised to (C − LISI) / (C − 1) ∈ [0, 1]; 1 = neighbourhoods are type-pure.
LISI as in harmonypy (Korsunsky 2019): 90 nearest neighbours, Gaussian weights at perplexity 30, inverse Simpson.
Features are standardised by the measured cells' SD per marker. 3 random subsamples (5,000 measured + 5,000 virtual
cells per subsample); points are means, bars the range.

Settings:
- held-out cells: the table model trained on the test cohort's own training patients (the only within-cohort model
  with saved per-cell predictions);
- held-out cohort: every model in the main figure, trained on the other cohort.
- '+ residuals': the LightGBM prediction plus a whole residual vector (all 10 markers) drawn from a random cell of
  that model's own source-cohort held-out predictions (a generative version; keeps cross-marker residual structure).
Writes lisi_virtual_cells.csv and fig_lisi_virtual_cells.png/pdf here.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
DATA = os.path.join(R, '..', 'IMC data')
EXCL = 'estrogen_receptor_alpha'
SEED, N_SUB, N_REP, K, PERP = 20261029, 5000, 3, 90, 30
JAC, DAN, NONE, INK, INK2, GREY = '#2a78d6', '#008300', '#4a3aa7', '#0b0b0b', '#52514e', '#b0aea7'
COARSE = ["epithelial", "T cell", "B cell", "myeloid", "endothelial", "stroma"]
D_COARSE = {"CD8^{+} T cells": "T cell", "CD4^{+} T cells": "T cell", "T_{Reg} & T_{Ex}": "T cell", "B cells": "B cell",
            "Macrophages": "myeloid", "Granulocytes": "myeloid", "Macrophages & granulocytes": "myeloid",
            "Endothelial": "endothelial", "Fibroblasts": "stroma", "Fibroblasts FSP1^{+}": "stroma",
            "Myofibroblasts": "stroma", "Myofibroblasts PDPN^{+}": "stroma"}
J_COARSE = {1: "T cell", 2: "B cell", 4: "myeloid", 6: "myeloid", 7: "endothelial",
            **{m: "stroma" for m in range(8, 14)}, **{m: "epithelial" for m in range(14, 28)}}

TESTS = {
    'Danenberg': dict(
        npz=f'{DATA}/Danenberg2022/cross-cohort-metrics/percell_predictions_test.npz',
        cells=f'{DATA}/Danenberg2022/danenberg-cells-prep/cells.parquet',
        ref=f'{R}/eva-zeroshot/source_reference_jackson.csv',
        extra={'ImmuVis': f'{DATA}/Danenberg2022/immuvis-jackson/percell_pred_immuvis.npz',
               'EVA': f'{DATA}/Danenberg2022/eva-zeroshot/percell_pred_eva.npz'},
        own=('Table (Danenberg-trained)', 'LightGBM'),
        other=[('ResNet-UNet immuvis', 'U-Net', JAC), ('ImmuVis', 'ImmuVis', JAC), ('EVA', 'EVA (zero-shot)', NONE),
               ('Table [Jackson-scaled inputs]', 'LightGBM', JAC), ('MLP [Jackson-scaled inputs]', 'MLP', JAC),
               ('GCN [Jackson-scaled inputs]', 'GCN', JAC), ('GraphSAGE [Jackson-scaled inputs]', 'GraphSAGE', JAC)],
        own_col=DAN),
    'Jackson': dict(
        npz=f'{DATA}/Jackson2020/cross-cohort-metrics-jackson/percell_predictions_test.npz',
        cells=f'{DATA}/Jackson2020/jackson-cells-prep/cells.parquet',
        ref=f'{R}/image-models-jackson/source_reference_danenberg.csv',
        extra={'ImmuVis': f'{DATA}/Jackson2020/image-models-jackson/percell_pred_immuvis.npz',
               'EVA': f'{DATA}/Jackson2020/image-models-jackson/percell_pred_eva.npz'},
        own=('Table (Jackson-trained)', 'LightGBM'),
        other=[('ImmuVis', 'ImmuVis', DAN), ('EVA', 'EVA (zero-shot)', NONE),
               ('Table [Danenberg-scaled inputs]', 'LightGBM', DAN), ('MLP [Danenberg-scaled inputs]', 'MLP', DAN),
               ('GCN [Danenberg-scaled inputs]', 'GCN', DAN), ('GraphSAGE [Danenberg-scaled inputs]', 'GraphSAGE', DAN)],
        own_col=JAC)}
OTHER = {'Danenberg': 'Jackson', 'Jackson': 'Danenberg'}


def lisi(X, labels, k=K, perp=PERP):
    """harmonypy-style LISI: Gaussian kernel calibrated to `perp` over k neighbours, inverse Simpson of label mass."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    D, I = nn.kneighbors(X)
    D, I = D[:, 1:] ** 2, I[:, 1:]
    n = len(X)
    beta, lo, hi = np.ones(n), np.full(n, -np.inf), np.full(n, np.inf)
    target = np.log(perp)
    for _ in range(50):
        P = np.exp(-(D - D[:, :1]) * beta[:, None])
        sP = P.sum(1)
        H = np.log(sP) + beta * ((D - D[:, :1]) * P).sum(1) / sP
        P /= sP[:, None]
        up = H > target                       # entropy too high → sharpen
        lo = np.where(up, beta, lo); hi = np.where(up, hi, beta)
        beta = np.where(up, np.where(np.isinf(hi), beta * 2, (beta + hi) / 2), np.where(np.isinf(lo), beta / 2, (beta + lo) / 2))
    lab = labels[I]
    simpson = sum(((lab == u) * P).sum(1) ** 2 for u in np.unique(labels))
    return 1 / simpson


def coarse_labels(T, Z):
    import duckdb                                    # local pyarrow < 21 cannot read these parquets
    cols = duckdb.sql(f"select * from '{T['cells']}' limit 0").columns
    want = [x for x in ('image_name', 'object_number', 'metacluster', 'cell_phenotype', 'is_epithelial') if x in cols]
    c = duckdb.sql(f"select {', '.join(want)} from '{T['cells']}'").df()
    key = pd.MultiIndex.from_arrays([c.image_name.astype(str), c.object_number.astype(np.int64)])
    if 'metacluster' in c:
        lab = c.metacluster.map(J_COARSE)
    else:
        epi = c.groupby(c.cell_phenotype.astype(str)).is_epithelial.transform('mean') > 0.5
        lab = pd.Series(np.where(epi, 'epithelial', c.cell_phenotype.astype(str).map(D_COARSE)), index=c.index)
    s = pd.Series(lab.values, index=key)
    s = s[~s.index.duplicated()]
    q = pd.MultiIndex.from_arrays([Z['image'].astype(str), Z['object_number'].astype(np.int64)])
    return s.reindex(q).values


def load(test):
    T = TESTS[test]
    Z = np.load(T['npz'], allow_pickle=True)          # our own kernel output; 'patient' is an object array in J→D
    mk = sorted(k.split('__')[1] for k in Z.files if k.startswith('truth__') and EXCL not in k)
    truth = np.column_stack([Z[f'truth__{m}'] for m in mk]).astype(np.float64)

    def pred(M):
        if M in T['extra']:
            E = np.load(T['extra'][M], allow_pickle=False)
            assert (E['image'] == Z['image']).all() and (E['object_number'] == Z['object_number']).all()
            return np.column_stack([E[f'predt__{M}__{m}'] for m in mk]).astype(np.float64)
        return np.column_stack([Z[f'pred__{M}__{m}'] for m in mk]).astype(np.float64)
    ref = pd.read_csv(T['ref']).set_index('marker').loc[mk, 'cell_mean'].values
    return dict(truth=truth, pred=pred, patient=Z['patient'].astype(str), lab=coarse_labels(T, Z), ref=ref, mk=mk)


L = {t: load(t) for t in TESTS}
# residual pools: each held-out-cells LightGBM's residual vectors on its own cohort's test cells
RES = {t: (L[t]['truth'] - L[t]['pred'](TESTS[t]['own'][0])) for t in TESTS}
RES = {t: r[np.isfinite(r).all(1)] for t, r in RES.items()}

rows = []
for test, T in TESTS.items():
    d = L[test]
    ok = np.isfinite(d['truth']).all(1) & pd.notna(d['lab'])
    pats = np.unique(d['patient'])
    entries = [('held-out cells', T['own'][1], T['own'][0], T['own_col'], None),
               ('held-out cells', T['own'][1] + ' + residuals', T['own'][0], T['own_col'], test)] + \
              [('held-out cohort', lab, M, col, None) for M, lab, col in T['other']] + \
              [('held-out cohort', 'LightGBM + residuals', T['other'][[o[1] for o in T['other']].index('LightGBM')][0],
                T['other'][0][2] if test == 'Jackson' else JAC, OTHER[test]),
               ('reference', 'Constant (training mean)', None, GREY, None),
               ('reference', 'Measured cells', 'truth', INK, None)]
    for setting, label, M, col, res_from in entries:
        P = (d['truth'] if M == 'truth' else np.tile(d['ref'], (len(d['truth']), 1)) if M is None else d['pred'](M))
        for rep in range(N_REP):
            rng = np.random.default_rng(SEED + rep)
            half = set(rng.choice(pats, len(pats) // 2, replace=False))
            inA = np.isin(d['patient'], list(half))
            a = rng.choice(np.flatnonzero(ok & inA), N_SUB, replace=False)
            b = rng.choice(np.flatnonzero(ok & ~inA & np.isfinite(P).all(1)), N_SUB, replace=False)
            V = P[b].copy()
            if res_from is not None:
                V += RES[res_from][rng.integers(0, len(RES[res_from]), N_SUB)]
            sd = d['truth'][a].std(0) + 1e-9
            Xi = np.vstack([d['truth'][a], V]) / sd
            Xi += rng.normal(0, 1e-6, Xi.shape)             # break exact ties (constant model)
            il = lisi(Xi, np.r_[np.zeros(N_SUB), np.ones(N_SUB)])
            cl = lisi(V / sd + rng.normal(0, 1e-6, V.shape), d['lab'][b])
            C = len(np.unique(d['lab'][b]))
            rows.append(dict(test=test, setting=setting, model=label, colour=col, rep=rep,
                             iLISI=float(np.mean(il) - 1), cLISI=float((C - np.mean(cl)) / (C - 1)),
                             iLISI_virtual_only=float(np.mean(il[N_SUB:]) - 1)))
        print(test, setting, label, np.round(pd.DataFrame(rows[-N_REP:])[['iLISI', 'cLISI']].mean().values, 3), flush=True)

df = pd.DataFrame(rows)
S = df.groupby(['test', 'setting', 'model', 'colour'], sort=False).agg(
    iLISI=('iLISI', 'mean'), iLISI_lo=('iLISI', 'min'), iLISI_hi=('iLISI', 'max'),
    cLISI=('cLISI', 'mean'), cLISI_lo=('cLISI', 'min'), cLISI_hi=('cLISI', 'max')).reset_index()
S.drop(columns='colour').to_csv(f'{HERE}/lisi_virtual_cells.csv', index=False)
print(S.drop(columns='colour').round(3).to_string(index=False))

MARK = {'held-out cells': 'o', 'held-out cohort': '^', 'reference': 's'}
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, test in zip(axes, TESTS):
    s = S[S.test == test]
    for r in s.itertuples():
        gen = 'residuals' in r.model
        ax.errorbar(r.iLISI, r.cLISI, xerr=[[r.iLISI - r.iLISI_lo], [r.iLISI_hi - r.iLISI]],
                    yerr=[[r.cLISI - r.cLISI_lo], [r.cLISI_hi - r.cLISI]], fmt='none', ecolor=GREY, elinewidth=0.8, zorder=1)
        ax.scatter(r.iLISI, r.cLISI, s=70, marker=MARK[r.setting], zorder=3,
                   facecolor='white' if gen else r.colour, edgecolor=r.colour, linewidth=1.8)
    # labels: points near iLISI = 0 are crowded, so list them in a column with leader lines; others label in place
    crowd = s[s.iLISI < 0.15].sort_values('cLISI', ascending=False)
    ys = np.linspace(0.935, 0.935 - 0.022 * (len(crowd) - 1), len(crowd))
    for r, y in zip(crowd.itertuples(), ys):
        ax.annotate(r.model, (r.iLISI, r.cLISI), xytext=(0.2, y), textcoords='data', fontsize=8, color=INK2, va='center',
                    arrowprops=dict(arrowstyle='-', color=GREY, lw=0.6))
    for r in s[s.iLISI >= 0.15].itertuples():
        ax.annotate(r.model, (r.iLISI, r.cLISI), xytext=(6, 5), textcoords='offset points', fontsize=8, color=INK2)
    ax.set_xlabel('iLISI, virtual vs measured cells (1 = indistinguishable)')
    ax.set_ylabel(f'cLISI, cell-type separation of virtual cells (1 = pure)\ntest: {test} held-out patients')
    ax.set_xlim(-0.03, 1.0); ax.set_ylim(0.66, 0.95)
    ax.spines[['top', 'right']].set_visible(False)
h = [plt.Line2D([], [], marker=m, ls='', color=INK2, markerfacecolor=INK2 if k != 'g' else 'white', label=l)
     for m, k, l in (('o', 'p', 'held-out cells (trained on the test cohort)'), ('^', 'p', 'held-out cohort (trained on the other cohort)'),
                     ('o', 'g', 'hollow: prediction + residual vector (generative)'), ('s', 'p', 'reference'))]
h += [plt.Line2D([], [], marker='o', ls='', color=c, label=l) for c, l in ((JAC, 'trained on Jackson'), (DAN, 'trained on Danenberg'), (NONE, 'no IMC training'))]
fig.legend(handles=h, loc='lower center', ncol=4, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(f'{HERE}/fig_lisi_virtual_cells.png', dpi=200)
fig.savefig(f'{HERE}/fig_lisi_virtual_cells.pdf')
