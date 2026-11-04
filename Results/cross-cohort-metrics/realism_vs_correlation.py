"""Correlation vs distribution mismatch, at pixel, 32 µm patch and cell level, both directions.

Each panel: x = the main figure's correlation at that level (pooled pixel r, pooled patch r, cell-level r within
image), y = Wasserstein-2 distance between the predicted and the true distribution of values at that level, in the
target space, mean over the 10 markers (ERα excluded). Lower y = closer marginal distribution.

This is a distribution-match diagnostic, not a realism score: every model here predicts a conditional mean, which is
under-dispersed by construction, so a spread ratio < 1 is expected and > 1 points to a calibration problem (§18.6).

Pixel and patch: only per-image sums are saved (n, Σ, Σ²), so W2 is the moment bound
sqrt((μp − μg)² + (σp − σg)²), which is exact for distributions that differ only in location and scale and a lower
bound otherwise. Cell: exact 1D W2 from the per-cell values (quantile matching), plus the same moment bound as a check.
95% CIs: patient bootstrap, the same draws for every model and marker within a direction (1000 for pixel and patch,
N_BOOT_CELL for the exact cell-level W2, which needs a sort per draw).
Writes realism_vs_correlation.csv and fig_realism_vs_correlation.png/pdf here.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
DATA = os.path.join(R, '..', 'IMC data')
EXCL = 'estrogen_receptor_alpha'
CEIL, CONST = 'Painted true cell means', 'Constant (training mean)'
JAC, DAN, NONE, INK, INK2 = '#2a78d6', '#008300', '#4a3aa7', '#0b0b0b', '#52514e'
COL = {'J': JAC, 'D': DAN, 'fm': NONE}

DIRS = [dict(
    name='Jackson → Danenberg', summary=f'{R}/cross-cohort-metrics/metrics_summary.csv',
    npz=f'{DATA}/Danenberg2022/cross-cohort-metrics/percell_predictions_test.npz',
    extra=[('ImmuVis', f'{DATA}/Danenberg2022/immuvis-jackson/percell_pred_immuvis.npz',
            f'{R}/immuvis-jackson/pixel_sums_test_immuvis.csv', f'{R}/immuvis-jackson-patch/patch_sums_test_immuvis.csv'),
           ('EVA', f'{DATA}/Danenberg2022/eva-zeroshot/percell_pred_eva.npz',
            f'{R}/eva-zeroshot/pixel_sums_test_eva.csv', f'{R}/eva-zeroshot-patch/patch_sums_test_eva.csv')],
    pixel=f'{R}/cross-cohort-metrics/pixel_sums_test.csv',
    patch=[f'{R}/cross-cohort-metrics-patch/patch_sums_test.csv', f'{R}/unet-patch-sums/patch_sums_per_image.csv'],
    ref=f'{R}/eva-zeroshot/source_reference_jackson.csv',
    models=[('ResNet-UNet immuvis', 'U-Net', 'J', False), ('ImmuVis', 'ImmuVis', 'J', False), ('EVA', 'EVA', 'fm', False),
            ('Table [Jackson-scaled inputs]', 'LightGBM', 'J', True), ('MLP [Jackson-scaled inputs]', 'MLP', 'J', True),
            ('GCN [Jackson-scaled inputs]', 'GCN', 'J', True), ('GraphSAGE [Jackson-scaled inputs]', 'GraphSAGE', 'J', True),
            ('Table (Danenberg-trained)', 'LightGBM (target)', 'D', True)]),
    dict(
    name='Danenberg → Jackson', summary=f'{R}/cross-cohort-metrics-jackson/metrics_summary.csv',
    npz=f'{DATA}/Jackson2020/cross-cohort-metrics-jackson/percell_predictions_test.npz',
    extra=[('ImmuVis', f'{DATA}/Jackson2020/image-models-jackson/percell_pred_immuvis.npz',
            f'{R}/image-models-jackson/pixel_sums_test_immuvis.csv', f'{R}/image-models-jackson-patch/patch_sums_test_immuvis.csv'),
           ('EVA', f'{DATA}/Jackson2020/image-models-jackson/percell_pred_eva.npz',
            f'{R}/image-models-jackson/pixel_sums_test_eva.csv', f'{R}/image-models-jackson-patch/patch_sums_test_eva.csv')],
    pixel=f'{R}/cross-cohort-metrics-jackson/pixel_sums_test.csv',
    patch=[f'{R}/cross-cohort-metrics-jackson-patch/patch_sums_test.csv'],
    ref=f'{R}/image-models-jackson/source_reference_danenberg.csv',
    models=[('ImmuVis', 'ImmuVis', 'D', False), ('EVA', 'EVA', 'fm', False),
            ('Table [Danenberg-scaled inputs]', 'LightGBM', 'D', True), ('MLP [Danenberg-scaled inputs]', 'MLP', 'D', True),
            ('GCN [Danenberg-scaled inputs]', 'GCN', 'D', True), ('GraphSAGE [Danenberg-scaled inputs]', 'GraphSAGE', 'D', True),
            ('Table (Jackson-trained)', 'LightGBM (target)', 'J', True)])]
LEVELS = [('pixel', 'pooled pixel r', 'Pixel-level correlation (pooled)'),
          ('patch', 'pooled patch r', 'Patch-level correlation, 32 µm (pooled)'),
          ('cell', 'R4', 'Cell-level correlation (within image)')]


SEED, N_BOOT, N_BOOT_CELL = 20261013, 1000, 100
BOOT = {}          # (direction, model, level, marker) -> bootstrap vector of W2


def moment_w2(mp, sp, mg, sg):
    return np.sqrt((mp - mg) ** 2 + (sp - sg) ** 2)


def moments(n, S, SS):
    m = S / n
    return m, np.sqrt(np.maximum(SS / n - m ** 2, 0))


def level_rows(D, level, SUMS, images, ref, img_pat, n_pat, W):
    rows = []
    SUMS = SUMS[SUMS.image.isin(images)]
    for M in [m for m, *_ in D['models']] + [CEIL]:
        for mk, df in SUMS[SUMS.model == M].groupby('marker'):
            use_t = 'Sp_t' in df and df.Sp_t.notna().all()
            pc = img_pat.reindex(df.image).values.astype(np.int64)
            cols = ['n', 'Sg', 'Sgg'] + (['Sp_t', 'Spp_t'] if use_t else ['Sp', 'Spp'])
            pp = np.stack([np.bincount(pc, df[c].values.astype(np.float64), n_pat) for c in cols])   # per patient
            tot, bt = pp.sum(1), W @ pp.T                                                         # point, draws
            mg, sg = moments(tot[0], tot[1], tot[2])
            mp, sp = moments(tot[0], tot[3], tot[4])
            bmg, bsg = moments(bt[:, 0], bt[:, 1], bt[:, 2])
            bmp, bsp = moments(bt[:, 0], bt[:, 3], bt[:, 4])
            BOOT[(D['name'], M, level, mk)] = moment_w2(bmp, bsp, bmg, bsg)
            w2 = float(moment_w2(mp, sp, mg, sg))
            rows.append(dict(model=M, level=level, marker=mk, W2=w2, W2_bound=w2,
                             mean_pred=mp, sd_pred=sp, mean_true=mg, sd_true=sg, n_images=df.image.nunique()))
            if M == CEIL:
                c = ref.loc[mk, 'pixel_mean']
                BOOT[(D['name'], CONST, level, mk)] = moment_w2(c, 0, bmg, bsg)
                w2 = float(moment_w2(c, 0, mg, sg))
                rows.append(dict(model=CONST, level=level, marker=mk, W2=w2, W2_bound=w2,
                                 mean_pred=c, sd_pred=0, mean_true=mg, sd_true=sg))
    return rows


def run_direction(D):
    Z = np.load(D['npz'], allow_pickle=True)      # our own kernel output; 'patient' is an object array in J→D
    images = set(Z['image'])
    pats, pcode = np.unique(Z['patient'].astype(str), return_inverse=True)
    n_pat = len(pats)
    img_pat = pd.Series(pcode, index=Z['image']).groupby(level=0).first()
    rng = np.random.default_rng(SEED)
    W = rng.multinomial(n_pat, np.full(n_pat, 1 / n_pat), size=N_BOOT).astype(np.float64)
    by_pat = [np.flatnonzero(pcode == i) for i in range(n_pat)]
    CELL_IDX = [np.concatenate([by_pat[i] for i in rng.integers(0, n_pat, n_pat)]) for _ in range(N_BOOT_CELL)]
    ref = pd.read_csv(D['ref']).set_index('marker')
    extra = {name: (npz, pix, pat) for name, npz, pix, pat in D['extra']}
    pix = pd.read_csv(D['pixel'])
    pix = pd.concat([pix[~pix.model.isin(extra)]] + [pd.read_csv(p) for _, p, _ in extra.values()], ignore_index=True)
    pat = pd.concat([pd.read_csv(p) for p in D['patch']] + [pd.read_csv(p) for *_, p in extra.values()], ignore_index=True)
    pat = pat.drop_duplicates(['model', 'marker', 'image'])
    rows = level_rows(D, 'pixel', pix, images, ref, img_pat, n_pat, W) + \
        level_rows(D, 'patch', pat, images, ref, img_pat, n_pat, W)

    def cell_boot(t, p):
        out = np.empty(N_BOOT_CELL)
        for b, ix in enumerate(CELL_IDX):
            tb, pb = t[ix], p[ix]
            ok = np.isfinite(tb) & np.isfinite(pb)
            out[b] = np.sqrt(np.mean((np.sort(tb[ok]) - np.sort(pb[ok])) ** 2))
        return out
    markers = sorted(k.split('__')[1] for k in Z.files if k.startswith('truth__'))
    for M, *_ in D['models']:
        if M in extra:
            E = np.load(extra[M][0], allow_pickle=False)
            assert (E['image'] == Z['image']).all() and (E['object_number'] == Z['object_number']).all()
            get = lambda mk: E[f'predt__{M}__{mk}']
        else:
            get = lambda mk: Z[f'pred__{M}__{mk}']
        for mk in markers:
            t, p = Z[f'truth__{mk}'].astype(np.float64), get(mk).astype(np.float64)
            ok = np.isfinite(t) & np.isfinite(p)
            ts, ps = np.sort(t[ok]), np.sort(p[ok])
            BOOT[(D['name'], M, 'cell', mk)] = cell_boot(t, p)
            rows.append(dict(model=M, level='cell', marker=mk, W2=float(np.sqrt(np.mean((ts - ps) ** 2))),
                             W1=float(np.mean(np.abs(ts - ps))), W2_bound=moment_w2(ps.mean(), ps.std(), ts.mean(), ts.std()),
                             mean_pred=ps.mean(), sd_pred=ps.std(), mean_true=ts.mean(), sd_true=ts.std()))
    for mk in markers:
        t = Z[f'truth__{mk}'].astype(np.float64)
        c = ref.loc[mk, 'cell_mean']
        BOOT[(D['name'], CONST, 'cell', mk)] = np.array([np.sqrt(np.nanmean((t[ix] - c) ** 2)) for ix in CELL_IDX])
        t = t[np.isfinite(t)]
        rows.append(dict(model=CONST, level='cell', marker=mk, W2=float(np.sqrt(np.mean((t - c) ** 2))),
                         W1=float(np.mean(np.abs(t - c))), W2_bound=moment_w2(c, 0, t.mean(), t.std()),
                         mean_pred=c, sd_pred=0, mean_true=t.mean(), sd_true=t.std()))
    PM = pd.DataFrame(rows).assign(direction=D['name'])
    S = pd.read_csv(D['summary'])
    S = S[S.stat == 'mean']
    agg = []
    for (M, lev), df in PM[PM.marker != EXCL].groupby(['model', 'level']):
        assert df.marker.nunique() == 10, (M, lev, df.marker.nunique())
        metric = dict((l, m) for l, m, _ in LEVELS)[lev]
        r = S[(S.model == M) & (S.metric == metric)]
        bw = np.mean([BOOT[(D['name'], M, lev, mk)] for mk in df.marker], 0)
        agg.append(dict(direction=D['name'], model=M, level=lev, r=r.value.iloc[0] if len(r) else np.nan,
                        r_lo=r.lo.iloc[0] if len(r) else np.nan, r_hi=r.hi.iloc[0] if len(r) else np.nan,
                        W2=df.W2.mean(), W2_lo=np.percentile(bw, 2.5), W2_hi=np.percentile(bw, 97.5),
                        W2_bound=df.W2_bound.mean(), W1=df.W1.mean() if 'W1' in df else np.nan,
                        sd_ratio=(df.sd_pred / df.sd_true).mean(), mean_offset=(df.mean_pred - df.mean_true).mean()))
    if 1:   # oracle is exact at the cell level
        agg.append(dict(direction=D['name'], model=CEIL, level='cell', r=1.0, W2=0.0, W2_bound=0.0, W1=0.0, sd_ratio=1.0,
                        mean_offset=0.0))
    return PM, pd.DataFrame(agg)


PMs, AGGs = zip(*[run_direction(D) for D in DIRS])
pd.concat(PMs).to_csv(f'{HERE}/realism_vs_correlation_per_marker.csv', index=False)
AGG = pd.concat(AGGs)
AGG.to_csv(f'{HERE}/realism_vs_correlation.csv', index=False)
with pd.option_context('display.width', 200):
    print(AGG.round(3).to_string(index=False))

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK})
fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))
ytop = {lev: AGG[AGG.level == lev].W2.max() * 1.15 for lev, *_ in LEVELS}
LABELS = []
# candidate label offsets (points), tried in order; leader line drawn when the label sits far from its point
OFFS = [(7, 3), (7, -10), (-7, 3), (-7, -10), (0, 9), (0, -15)] + \
       [(dx, dy) for r_ in (22, 34, 46) for dx, dy in ((r_, 0), (-r_, 0), (r_, r_ * 0.6), (-r_, r_ * 0.6),
                                                        (r_, -r_ * 0.6), (-r_, -r_ * 0.6), (0, r_ * 0.8), (0, -r_ * 0.8))]


def place_labels():
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    taken = {}
    for ax, *_ in LABELS:      # every marker is an obstacle
        taken.setdefault(ax, [])
    for ax, lab, x, y, c in LABELS:
        px, py = ax.transData.transform((x, y))
        taken[ax].append((px - 7, py - 7, px + 7, py + 7))
    for ax, lab, x, y, c in LABELS:
        best = None
        for dx, dy in OFFS:
            ha = 'left' if dx > 0 else 'right' if dx < 0 else 'center'
            t = ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=7.5, color=c, ha=ha,
                            va='bottom' if dy >= 0 else 'top', zorder=5)
            bb = t.get_window_extent(rend).expanded(1.05, 1.15)
            abb = ax.get_window_extent(rend)
            box = (bb.x0, bb.y0, bb.x1, bb.y1)
            clash = any(not (box[2] < o[0] or box[0] > o[2] or box[3] < o[1] or box[1] > o[3]) for o in taken[ax])
            inside = abb.x0 <= box[0] and box[2] <= abb.x1 + 30 and abb.y0 <= box[1] and box[3] <= abb.y1
            if not clash and inside:
                best = (t, box, dx, dy)
                break
            t.remove()
        if best is None:      # fall back to the first offset
            t = ax.annotate(lab, (x, y), xytext=OFFS[0], textcoords='offset points', fontsize=7.5, color=c, zorder=5)
            best = (t, None, *OFFS[0])
        t, box, dx, dy = best
        if abs(dy) > 9 or abs(dx) > 9 and dy != 3:
            t.arrow_patch = None
            t.remove()
            ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=7.5, color=c,
                        ha='left' if dx > 0 else 'right' if dx < 0 else 'center', va='bottom' if dy >= 0 else 'top',
                        arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6, shrinkA=1, shrinkB=5), zorder=5)
        if box:
            taken[ax].append(box)
for row, D in enumerate(DIRS):
    A = AGG[AGG.direction == D['name']]
    for col, (lev, metric, xlab) in enumerate(LEVELS):
        ax = axes[row, col]
        a = A[A.level == lev].set_index('model')
        for M, lab, coh, hollow in D['models']:
            if M not in a.index:
                continue
            v = a.loc[M]
            ax.errorbar(v.r, v.W2, xerr=[[v.r - v.r_lo], [v.r_hi - v.r]], yerr=[[v.W2 - v.W2_lo], [v.W2_hi - v.W2]],
                        fmt='none', ecolor=INK2, elinewidth=0.8, capsize=2, zorder=2)
            mk = 's' if not hollow else 'o'
            ax.scatter(v.r, v.W2, s=70, marker=mk, facecolor='white' if hollow else COL[coh], edgecolor=COL[coh],
                       linewidth=1.8, zorder=3)
            LABELS.append((ax, lab, v.r, v.W2, INK))
        if CEIL in a.index:
            v = a.loc[CEIL]
            ax.scatter(v.r, v.W2, s=90, marker='*', color=INK2, zorder=3)
            LABELS.append((ax, 'True cell values', v.r, v.W2, INK2))
        if CONST in a.index:
            ax.axhline(a.loc[CONST].W2, color=INK2, lw=1, ls=(0, (1, 2)), zorder=1)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, ytop[lev])
        ax.set_xlabel(xlab)
        ax.set_ylabel(f'Distribution mismatch of {lev} values\n' +
                      ('W2, exact' if lev == 'cell' else 'W2, moment bound') + f' ({D["name"]})')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(color='#e6e5e1', lw=0.6, zorder=0)
        ax.text(-0.16, 1.02, 'abcdef'[row * 3 + col], transform=ax.transAxes, fontsize=13, fontweight='bold',
                va='bottom', color=INK)
handles = [Line2D([], [], ls='', marker='s', ms=8, mfc=JAC, mec=JAC, label='Image model, Jackson-trained'),
           Line2D([], [], ls='', marker='s', ms=8, mfc=DAN, mec=DAN, label='Image model, Danenberg-trained'),
           Line2D([], [], ls='', marker='s', ms=8, mfc=NONE, mec=NONE, label='Image model, zero-shot (EVA)'),
           Line2D([], [], ls='', marker='o', ms=8, mfc='white', mec=JAC, mew=1.8, label='Cell model, Jackson-trained'),
           Line2D([], [], ls='', marker='o', ms=8, mfc='white', mec=DAN, mew=1.8, label='Cell model, Danenberg-trained'),
           Line2D([], [], ls='', marker='*', ms=10, color=INK2, label='True cell values'),
           Line2D([], [], color=INK2, lw=1, ls=(0, (1, 2)), label='Constant (training mean)')]
fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.0, h_pad=2.2)
place_labels()
for ext in ('png', 'pdf'):
    fig.savefig(f'{HERE}/fig_realism_vs_correlation.{ext}', dpi=300, bbox_inches='tight')
print('saved')
