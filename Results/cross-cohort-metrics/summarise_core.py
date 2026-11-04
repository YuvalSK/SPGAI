"""Shared by both directions (summarise.py here, and ../cross-cohort-metrics-jackson/summarise.py).

Per model and marker: pooled pixel r, per-image pixel r, R4 (cell-level r within image), and the errors
pooled pixel RMSE, per-image pixel RMSE and cell RMSE. Then the mean and median over the 10 markers (ERα excluded),
with 95% patient-bootstrap CIs (the kernel's draws, same seed), paired per-marker calls of each per-cell model against
each image model, and the normalisation control.

Errors are in the target space. Per-cell models and the U-Net already predict there; ImmuVis and EVA supply a
target-scale copy of their output (pixel sums Sp_t / Spp_t / Sgp_t and per-cell predt__), mapped without test data.
'Constant (training mean)' predicts the training cohort's mean target value everywhere (errors only).
"""
import os
import numpy as np, pandas as pd

R_METRICS = ('pooled pixel r', 'per-image pixel r', 'pooled patch r', 'per-image patch r', 'R4')
E_METRICS = ('pooled pixel RMSE', 'per-image pixel RMSE', 'patch RMSE', 'cell RMSE')
MIN_PATCHES = 10                       # per-image patch r only for images with at least this many 32 µm patches
NEEDS_T = {'ImmuVis', 'EVA'}           # output not in the target space: errors only from the target-scale copy
CONST = 'Constant (training mean)'
ORACLE = 'Painted true cell means'


def r_from(n, Sg, Sp, Sgg, Spp, Sgp):
    with np.errstate(invalid='ignore', divide='ignore'):
        return (Sgp - Sg * Sp / n) / np.sqrt((Sgg - Sg ** 2 / n) * (Spp - Sp ** 2 / n))


def run(OUT, NPZ, patient_fn, boot_seed, MODELS, EXTRA, IMAGE_MODELS, PER_CELL, CTRL, ctrl_col, data_dir, ref_csv,
        SEED=20261013, N_BOOT=1000, EXCL='estrogen_receptor_alpha', PATCH_CSVS=()):
    """PATCH_CSVS: per-image sums over 32 µm patch means (same columns as the pixel sums), from the *-patch kernels."""
    MODELS = list(MODELS) + [M + CTRL for M in PER_CELL]
    Z = np.load(NPZ, allow_pickle=False)
    image = Z['image']
    patient = patient_fn(Z)
    MARKERS = sorted(k.split('__')[1] for k in Z.files if k.startswith('truth__'))
    tp, tcode = np.unique(patient, return_inverse=True)
    Wt = np.random.default_rng(SEED + boot_seed).multinomial(len(tp), np.full(len(tp), 1 / len(tp)),
                                                              size=N_BOOT).astype(np.float64)
    img_names, img_code = np.unique(image, return_inverse=True)
    img_pat = pd.Series(tcode, index=image).groupby(level=0).first()
    ms10 = [m for m in MARKERS if m != EXCL]

    def r4(a, b):
        """Per-cell r after removing each image's mean (same maths as stats_fn in the notebooks)."""
        ok = np.isfinite(a) & np.isfinite(b)
        a, b, im, pc = a[ok].astype(np.float64), b[ok].astype(np.float64), img_code[ok], tcode[ok]
        k = img_code.max() + 1
        cnt = np.maximum(np.bincount(im, minlength=k), 1)
        a = a - (np.bincount(im, a, k) / cnt)[im]
        b = b - (np.bincount(im, b, k) / cnt)[im]
        sxy, sxx, syy = (np.bincount(pc, a * b, len(tp)), np.bincount(pc, a * a, len(tp)), np.bincount(pc, b * b, len(tp)))
        return float(sxy.sum() / np.sqrt(sxx.sum() * syy.sum())), (Wt @ sxy) / np.sqrt((Wt @ sxx) * (Wt @ syy))

    def cell_rmse(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        d2 = (a[ok].astype(np.float64) - b[ok].astype(np.float64)) ** 2
        s, c = np.bincount(tcode[ok], d2, len(tp)), np.bincount(tcode[ok], minlength=len(tp)).astype(np.float64)
        return float(np.sqrt(s.sum() / c.sum())), np.sqrt((Wt @ s) / (Wt @ c))

    SUMS = pd.read_csv(f'{OUT}/pixel_sums_test.csv')
    PRED = {k: Z[k] for k in Z.files if k.startswith('pred__')}
    PREDT = {}
    extra_pts = {}
    for kernel, tag, name in EXTRA:
        fs = os.path.join(OUT, '..', kernel, f'pixel_sums_test_{tag}.csv')
        fz = os.path.join(data_dir, kernel, f'percell_pred_{tag}.npz')
        if not (os.path.exists(fs) and os.path.exists(fz)):
            print(f'{name}: no outputs from {kernel} yet')
            continue
        E = np.load(fz, allow_pickle=False)
        assert (E['image'] == image).all() and (E['object_number'] == Z['object_number']).all(), f'{name}: cell order differs'
        S_new = pd.read_csv(fs)
        if name in set(SUMS.model):
            # the same model already scored inside the main kernel: its raw r must be reproduced, then it is replaced
            old, new = SUMS[SUMS.model == name], S_new
            for m in MARKERS:
                a = r_from(*(old[old.marker == m][c].sum() for c in ['n', 'Sg', 'Sp', 'Sgg', 'Spp', 'Sgp']))
                b = r_from(*(new[new.marker == m][c].sum() for c in ['n', 'Sg', 'Sp', 'Sgg', 'Spp', 'Sgp']))
                assert abs(a - b) < 2e-3, (name, m, a, b)
            print(f'{name}: raw pooled r of {kernel} reproduces the main kernel (all markers within 2e-3)')
            SUMS = SUMS[SUMS.model != name]
        PRED.update({k: E[k] for k in E.files if k.startswith('pred__')})
        PREDT.update({k: E[k] for k in E.files if k.startswith('predt__')})
        SUMS = pd.concat([SUMS, S_new], ignore_index=True)
        if name not in MODELS:
            MODELS.append(name)
        fp = os.path.join(OUT, '..', kernel, f'metrics_point_{tag}.csv')
        if os.path.exists(fp):
            extra_pts[name] = pd.read_csv(fp).set_index('marker')
    MODELS = [M for M in MODELS if f'pred__{M}__{MARKERS[0]}' in PRED]

    cols6 = ['n', 'Sg', 'Sp', 'Sgg', 'Spp', 'Sgp']
    BOOT = {}

    def pixel_errors(M, m, se, n, pc):
        se = np.maximum(se, 0)
        sp, nn = np.bincount(pc, se, len(tp)), np.bincount(pc, n, len(tp))
        BOOT[(M, 'pooled pixel RMSE', m)] = (float(np.sqrt(sp.sum() / nn.sum())), np.sqrt((Wt @ sp) / (Wt @ nn)))
        ri = np.sqrt(se / n)
        sr, cn = np.bincount(pc, ri, len(tp)), np.bincount(pc, minlength=len(tp)).astype(np.float64)
        BOOT[(M, 'per-image pixel RMSE', m)] = (float(sr.sum() / cn.sum()), (Wt @ sr) / (Wt @ cn))

    for (M, m), df in SUMS.groupby(['model', 'marker'], sort=False):
        pc = img_pat.reindex(df.image).values.astype(np.int64)
        per_pat = np.stack([np.bincount(pc, df[c].values, len(tp)) for c in cols6])
        BOOT[(M, 'pooled pixel r', m)] = (float(r_from(*per_pat.sum(1))), r_from(*(Wt @ per_pat.T).T))
        ri = r_from(*(df[c].values for c in cols6))
        ok = np.isfinite(ri)
        sr, cn = np.bincount(pc[ok], ri[ok], len(tp)), np.bincount(pc[ok], minlength=len(tp)).astype(np.float64)
        BOOT[(M, 'per-image pixel r', m)] = (float(sr.sum() / cn.sum()), (Wt @ sr) / (Wt @ cn))
        if 'Sp_t' in df.columns and df['Sp_t'].notna().all():
            pixel_errors(M, m, (df.Sgg - 2 * df.Sgp_t + df.Spp_t).values, df.n.values, pc)
        elif M not in NEEDS_T:
            pixel_errors(M, m, (df.Sgg - 2 * df.Sgp + df.Spp).values, df.n.values, pc)
    # 32 µm patch level: the same maths on per-image sums over patch means
    PS = [pd.read_csv(f) for f in PATCH_CSVS if os.path.exists(f)]
    PSUMS = pd.concat(PS, ignore_index=True) if PS else pd.DataFrame(columns=['model', 'marker', 'image'])
    PSUMS = PSUMS[PSUMS.image.isin(img_pat.index) & PSUMS.model.isin(MODELS + [ORACLE])]
    assert not PSUMS.duplicated(['model', 'marker', 'image']).any(), 'duplicate patch sums'
    for (M, m), df in PSUMS.groupby(['model', 'marker'], sort=False):
        pc = img_pat.reindex(df.image).values.astype(np.int64)
        per_pat = np.stack([np.bincount(pc, df[c].values, len(tp)) for c in cols6])
        BOOT[(M, 'pooled patch r', m)] = (float(r_from(*per_pat.sum(1))), r_from(*(Wt @ per_pat.T).T))
        ri = r_from(*(df[c].values for c in cols6))
        ok = np.isfinite(ri) & (df.n.values >= MIN_PATCHES)
        sr, cn = np.bincount(pc[ok], ri[ok], len(tp)), np.bincount(pc[ok], minlength=len(tp)).astype(np.float64)
        BOOT[(M, 'per-image patch r', m)] = (float(sr.sum() / cn.sum()), (Wt @ sr) / (Wt @ cn))
        if 'Sp_t' in df.columns and df['Sp_t'].notna().all():
            se = (df.Sgg - 2 * df.Sgp_t + df.Spp_t).values
        elif M not in NEEDS_T:
            se = (df.Sgg - 2 * df.Sgp + df.Spp).values
        else:
            continue
        sp, nn = np.bincount(pc, np.maximum(se, 0), len(tp)), np.bincount(pc, df.n.values, len(tp))
        BOOT[(M, 'patch RMSE', m)] = (float(np.sqrt(sp.sum() / nn.sum())), np.sqrt((Wt @ sp) / (Wt @ nn)))
    if len(PSUMS):
        print('patch sums for:', sorted(PSUMS.model.unique()))

    for M in MODELS:
        for m in MARKERS:
            BOOT[(M, 'R4', m)] = r4(Z[f'truth__{m}'], PRED[f'pred__{M}__{m}'])
            p = PREDT.get(f'predt__{M}__{m}') if M in NEEDS_T else PRED[f'pred__{M}__{m}']
            if p is not None:
                BOOT[(M, 'cell RMSE', m)] = cell_rmse(Z[f'truth__{m}'], p)

    # constant prediction at the training cohort's mean (errors only)
    if os.path.exists(ref_csv):
        REF = pd.read_csv(ref_csv).set_index('marker')
        for m in MARKERS:
            df = SUMS[(SUMS.model == ORACLE) & (SUMS.marker == m)]
            pc = img_pat.reindex(df.image).values.astype(np.int64)
            c = REF.loc[m, 'pixel_mean']
            pixel_errors(CONST, m, (df.Sgg - 2 * c * df.Sg + df.n * c ** 2).values, df.n.values, pc)
            dp = PSUMS[(PSUMS.model == ORACLE) & (PSUMS.marker == m)]
            if len(dp):
                pp = img_pat.reindex(dp.image).values.astype(np.int64)
                se = np.maximum((dp.Sgg - 2 * c * dp.Sg + dp.n * c ** 2).values, 0)
                sp, nn = np.bincount(pp, se, len(tp)), np.bincount(pp, dp.n.values, len(tp))
                BOOT[(CONST, 'patch RMSE', m)] = (float(np.sqrt(sp.sum() / nn.sum())), np.sqrt((Wt @ sp) / (Wt @ nn)))
            t = Z[f'truth__{m}']
            BOOT[(CONST, 'cell RMSE', m)] = cell_rmse(t, np.full(len(t), REF.loc[m, 'cell_mean'], np.float32))
        MODELS.append(CONST)
    else:
        print('no training-cohort reference yet:', ref_csv)

    # the recomputed per-marker values must equal the kernels'
    PM = pd.read_csv(f'{OUT}/metrics_per_marker.csv')
    PM = PM[PM.model.isin(MODELS) & ~PM.model.isin(list(extra_pts))]
    dev = max(abs(BOOT[(r.model, r.metric, r.marker)][0] - r.value) for r in PM.itertuples())
    for name, P in extra_pts.items():
        for col, metric in [('pooled', 'pooled pixel r'), ('per_image', 'per-image pixel r'), ('R4', 'R4'),
                            ('pixel_rmse', 'pooled pixel RMSE'), ('per_image_rmse', 'per-image pixel RMSE'),
                            ('cell_rmse', 'cell RMSE')]:
            if col in P.columns and (name, metric, MARKERS[0]) in BOOT:
                dev = max(dev, max(abs(BOOT[(name, metric, m)][0] - P.loc[m, col]) for m in MARKERS))
    print(f'max |local - kernel| over the per-marker point values: {dev:.2e}')
    assert dev < 1e-3

    ROWS = []
    for M in MODELS:
        for metric in R_METRICS + E_METRICS:
            if not all((M, metric, m) in BOOT for m in ms10):
                continue
            pts = np.array([BOOT[(M, metric, m)][0] for m in ms10])
            bv = np.stack([BOOT[(M, metric, m)][1] for m in ms10])
            for stat, f in (('mean', np.mean), ('median', np.median)):
                b = f(bv, 0)
                ROWS.append(dict(model=M, metric=metric, stat=stat, value=float(f(pts)),
                                 lo=float(np.nanpercentile(b, 2.5)), hi=float(np.nanpercentile(b, 97.5)), n_markers=len(ms10)))
    SUMM = pd.DataFrame(ROWS)
    SUMM.to_csv(f'{OUT}/metrics_summary.csv', index=False)

    # per-marker values with CIs for every model and metric (the kernel's file has only its own models and r)
    PMA = pd.DataFrame([dict(model=M, metric=metric, marker=m, value=v[0], lo=float(np.nanpercentile(v[1], 2.5)),
                             hi=float(np.nanpercentile(v[1], 97.5))) for (M, metric, m), v in BOOT.items()])
    PMA.to_csv(f'{OUT}/metrics_per_marker_all.csv', index=False)

    # each per-cell model vs each image model, per marker: paired difference (cell - image) with a CI (same draws)
    DIFF = []
    for M in [x for x in PER_CELL + [p + CTRL for p in PER_CELL] if x in MODELS]:
        for B in [x for x in IMAGE_MODELS if x in MODELS]:
            for metric in R_METRICS + E_METRICS:
                for m in MARKERS:
                    if (M, metric, m) not in BOOT or (B, metric, m) not in BOOT:
                        continue
                    a, b = BOOT[(M, metric, m)], BOOT[(B, metric, m)]
                    lo, hi = np.nanpercentile(a[1] - b[1], [2.5, 97.5])
                    good, bad = (lo > 0, hi < 0) if metric in R_METRICS else (hi < 0, lo > 0)   # lower RMSE is better
                    DIFF.append(dict(model=M, image_model=B, metric=metric, marker=m, diff=a[0] - b[0], lo=lo, hi=hi,
                                     call='better' if good else 'worse' if bad else 'n.s.'))
    DIFF = pd.DataFrame(DIFF)
    DIFF.to_csv(f'{OUT}/per_cell_vs_image_per_marker.csv', index=False)

    # normalisation control: the model with its own cohort scaling minus the main version, mean over the 10 markers
    NC = []
    for M in [x for x in PER_CELL if x + CTRL in MODELS]:
        for metric in R_METRICS + E_METRICS:
            if not all((M, metric, m) in BOOT and (M + CTRL, metric, m) in BOOT for m in ms10):
                continue
            own = np.array([BOOT[(M, metric, m)][0] for m in ms10]).mean()
            js = np.array([BOOT[(M + CTRL, metric, m)][0] for m in ms10]).mean()
            d = np.stack([BOOT[(M, metric, m)][1] - BOOT[(M + CTRL, metric, m)][1] for m in ms10]).mean(0)
            NC.append({'model': M, 'metric': metric, 'own_scaling': own, ctrl_col: js, 'diff': own - js,
                       'lo': float(np.nanpercentile(d, 2.5)), 'hi': float(np.nanpercentile(d, 97.5))})
    NC = pd.DataFrame(NC)
    if len(NC):
        NC.to_csv(f'{OUT}/normalisation_control.csv', index=False)

    # every model vs the constant prediction, errors only: model - constant (negative = model better), mean over the
    # 10 markers with a CI, and per-marker calls (same draws)
    VC = []
    if CONST in MODELS:
        for M in [x for x in MODELS if x != CONST]:
            for metric in E_METRICS:
                if not all((M, metric, m) in BOOT for m in ms10):
                    continue
                d = np.stack([BOOT[(M, metric, m)][1] - BOOT[(CONST, metric, m)][1] for m in ms10])
                calls = []
                for m, dm in zip(ms10, d):
                    lo, hi = np.nanpercentile(dm, [2.5, 97.5])
                    calls.append('better' if hi < 0 else 'worse' if lo > 0 else 'n.s.')
                dmean = d.mean(0)
                VC.append(dict(model=M, metric=metric,
                               diff=float(np.mean([BOOT[(M, metric, m)][0] - BOOT[(CONST, metric, m)][0] for m in ms10])),
                               lo=float(np.nanpercentile(dmean, 2.5)), hi=float(np.nanpercentile(dmean, 97.5)),
                               n_better=calls.count('better'), n_ns=calls.count('n.s.'), n_worse=calls.count('worse')))
        VC = pd.DataFrame(VC)
        VC.to_csv(f'{OUT}/vs_constant.csv', index=False)

    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        pv = SUMM[SUMM.stat == 'mean'].pivot_table(index='model', columns='metric', values='value')
        print(pv[[c for c in R_METRICS + E_METRICS if c in pv.columns]].round(3).to_string())
        d10 = DIFF[DIFF.marker != EXCL]
        t = d10.groupby(['image_model', 'metric', 'model']).call.value_counts().unstack(fill_value=0)
        print(t.to_string())
        if len(VC):
            print(VC.round(3).to_string())
    return SUMM, DIFF, NC
