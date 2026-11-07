"""Cohort mixing (iLISI) vs cell-type mixing (cLISI) for every cell representation we built, by scale.

Local analysis on saved outputs (no Kaggle run):
- cell tables and contact graphs from `danenberg-cells-prep` / `jackson-cells-prep`;
- pixel features from `pixel-features`; KRONOS / KRONOS2 embeddings from `kronos-zeroshot` / `kronos2-zeroshot`;
- Novae latents (replicate 0) from `novae-cells`.

Cells, labels, inputs, LISI code and draws are those of `novae-cells` (replicate-0 test patients, 5 draws x 5,000
labelled test cells per cohort, perplexity 30, every space z-scored per column over the pooled draw), so its
"Table" and "Table + nbr" rows are a reproduction check against `Results/novae-cells/lisi.csv`.

Run: python scale_lisi.py   (writes scale_lisi.csv, scale_lisi_summary.csv, fig_scale_lisi.png/pdf here)
"""
import os, json
import numpy as np, pandas as pd, duckdb
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "IMC data")
SRC = {"Danenberg": ("Danenberg2022", "danenberg"), "Jackson": ("Jackson2020", "jackson")}

SEED, COF, MIN_CELLS, R_CAP = 20261013, 5.0, 200, 32
LISI_PERP, LISI_N, LISI_DRAWS, REP = 30, 5000, 5, 0
N_PC = 32
NUC = ["histone_h3", "dna1", "dna2"]
MARKERS = ["c_erb_b_2_her2", "cd20", "cd3", "cd45", "cd68", "cytokeratin_5", "cytokeratin_8_18",
           "estrogen_receptor_alpha", "ki_67", "pan_cytokeratin", "sma"]
FEATS = NUC + MARKERS
D_TABLE = {"c_erb_b_2_her2": "c_erb_b_2_her2_3b5"}
COARSE = ["epithelial", "T cell", "B cell", "myeloid", "endothelial", "stroma"]
D_COARSE = {"CD8^{+} T cells": "T cell", "CD4^{+} T cells": "T cell", "T_{Reg} & T_{Ex}": "T cell",
            "B cells": "B cell",
            "Macrophages": "myeloid", "Granulocytes": "myeloid", "Macrophages & granulocytes": "myeloid",
            "Endothelial": "endothelial",
            "Fibroblasts": "stroma", "Fibroblasts FSP1^{+}": "stroma", "Myofibroblasts": "stroma",
            "Myofibroblasts PDPN^{+}": "stroma"}
J_COARSE = {1: "T cell", 2: "B cell", 4: "myeloid", 6: "myeloid", 7: "endothelial",
            **{m: "stroma" for m in range(8, 14)}, **{m: "epithelial" for m in range(14, 28)}}
PIX = [f"{s}_full__{c}" for c in FEATS for s in ("sd", "dio", "mi")] + ["ecc", "perim", "dmax"]


def sql(q):
    return duckdb.sql(q).df()                      # local pyarrow < 21 cannot read the Kaggle parquets


def harm(X):
    med = np.nanmedian(X, 0)
    q05, q95 = np.nanpercentile(X, [5, 95], 0)
    sc = np.where(q95 - q05 > 1e-3, q95 - q05, np.nanstd(X, 0) + 1e-6)
    return np.nan_to_num(((X - med) / sc).astype(np.float32))


def knn_adj(xy, starts, k=None, radius=None):
    rows, cols = [], []
    for a, b in zip(starts[:-1], starts[1:]):
        p = xy[a:b]
        kk = min(k if k else R_CAP, b - a - 1)
        d, nb = cKDTree(p).query(p, k=kk + 1, distance_upper_bound=radius if radius else np.inf)
        d, nb = d[:, 1:], nb[:, 1:]
        ok = np.isfinite(d)
        rows.append(np.repeat(np.arange(a, b), kk)[ok.ravel()])
        cols.append(nb[ok] + a)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    n = len(xy)
    A = sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float32).tocsr()
    A.sort_indices()
    return A


def grid_mean(img, xy, X, size):
    gx = np.floor(xy[:, 0] / size).astype(np.int64)
    gy = np.floor(xy[:, 1] / size).astype(np.int64)
    code, inv = np.unique((img * 100_000 + gy) * 100_000 + gx, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    M = np.stack([np.bincount(inv, X[:, j], len(code)) for j in range(X.shape[1])], 1) / cnt[:, None]
    return M[inv].astype(np.float32), np.log1p(cnt[inv]).astype(np.float32)


def row_index(names, objs, ref_names, ref_objs):
    """Position of each (image_name, object_number) in a reference list; -1 if absent."""
    ref = pd.Series(np.arange(len(ref_names)), index=pd.MultiIndex.from_arrays([np.asarray(ref_names).astype(str), np.asarray(ref_objs).astype(np.int64)]))
    out = ref.reindex(pd.MultiIndex.from_arrays([np.asarray(names).astype(str), np.asarray(objs).astype(np.int64)])).values
    return np.where(np.isnan(out), -1, out).astype(np.int64)


def load(key):
    folder, low = SRC[key]
    base = os.path.join(DATA, folder)
    f = os.path.join(base, f"{low}-cells-prep", "cells.parquet").replace("\\", "/")
    cols = [D_TABLE.get(c, c) for c in FEATS] if key == "Danenberg" else FEATS
    pid = "metabric_id" if key == "Danenberg" else "patient_id"
    extra = "cell_phenotype, is_epithelial" if key == "Danenberg" else "cell_phenotype, metacluster"
    filt = "is_tumour = 1 and is_hot_aggregate = 0" if key == "Danenberg" else "is_tumour = 1"
    cells = sql(f"select image_number, object_number, image_name, {pid}, x, y, area, {extra}, {', '.join(cols)} from '{f}' where {filt}")
    n_per = cells.groupby("image_name").size()
    cells = cells[cells.image_name.isin(n_per.index[n_per >= MIN_CELLS])]
    cells = cells.sort_values(["image_name", "object_number"]).reset_index(drop=True)
    n = len(cells)
    img, img_names = pd.factorize(cells.image_name, sort=True)
    starts = np.r_[0, np.flatnonzero(np.diff(img)) + 1, n]
    xy = cells[["x", "y"]].values.astype(np.float64)
    raw = np.arcsinh(cells[cols].values.astype(np.float64) / COF)
    X = np.c_[raw, np.log(np.maximum(cells["area"].values, 1))].astype(np.float32)

    phen = cells.cell_phenotype.astype(str).values
    fine = sorted(set(phen))
    if key == "Danenberg":
        epi = cells.groupby(cells.cell_phenotype.astype(str)).is_epithelial.mean()
        cmap = {p: ("epithelial" if epi[p] > 0.5 else D_COARSE.get(p)) for p in fine}
    else:
        mc = cells.groupby(cells.cell_phenotype.astype(str)).metacluster.first()
        cmap = {p: J_COARSE.get(int(mc[p])) for p in fine}
    coarse = np.array([COARSE.index(cmap[p]) if cmap[p] else -1 for p in phen], np.int64)

    patient = cells[pid].astype(str).values
    if key == "Danenberg":
        s0 = pd.read_csv(os.path.join(base, "danenberg-cells-prep", "danenberg_patient_split.csv"))
        s0 = s0.set_index(s0.metabric_id.astype(str)).split
        test = pd.Series(patient).map(s0).fillna("none").values == "test"
    else:
        jp, jcode = np.unique(patient, return_inverse=True)
        u = np.random.default_rng(SEED + 11 + REP).random(len(jp))
        test = (u >= 0.8)[jcode]

    # contact graph (for the Table + nbr reproduction)
    loc = pd.Series(np.arange(n), index=pd.MultiIndex.from_arrays([cells.image_number.values, cells.object_number.values]))
    con = sql(f"select ImageNumber, \"from\", \"to\" from '{os.path.join(base, f'{low}-cells-prep', 'cell_neighbours.parquet').replace(chr(92), '/')}'")
    ca = loc.reindex(pd.MultiIndex.from_arrays([con.ImageNumber.values, con["from"].values])).values
    cb = loc.reindex(pd.MultiIndex.from_arrays([con.ImageNumber.values, con["to"].values])).values
    ok = ~(np.isnan(ca) | np.isnan(cb))
    ca, cb = ca[ok].astype(np.int64), cb[ok].astype(np.int64)
    ca, cb = ca[ca != cb], cb[ca != cb]
    CON = sp.coo_matrix((np.ones(len(ca)), (ca, cb)), shape=(n, n)).tocsr()
    n_con = np.diff(((CON + CON.T) > 0).tocsr().indptr)

    I = harm(X)                                   # "intensity": arcsinh means, robust-scaled within each cohort (as `Table`)
    S = {"intensity": I, "pooled": X, "rank": (rankdata(raw[:, len(NUC):], axis=1) / len(MARKERS)).astype(np.float32)}

    A10 = knn_adj(xy, starts, k=10)
    A30 = knn_adj(xy, starts, radius=30.0)
    deg = np.diff(A10.indptr).astype(np.float32)[:, None]
    mean = (A10 @ I) / deg
    std = np.sqrt(np.maximum((A10 @ (I * I)) / deg - mean * mean, 0))
    mx = np.maximum.reduceat(I[A10.indices], A10.indptr[:-1], axis=0)
    hop2 = (A10 @ mean) / deg
    dst = np.repeat(np.arange(n), np.diff(A10.indptr))
    dist = np.bincount(dst, np.linalg.norm(xy[A10.indices] - xy[dst], axis=1), n) / deg.ravel()
    S["nbr"] = np.c_[I, mean, mx, std, hop2, np.log1p(np.diff(A30.indptr)), np.log1p(n_con), dist / 30.0].astype(np.float32)
    g64, n64 = grid_mean(img, xy, I, 64.0)
    g128, n128 = grid_mean(img, xy, I, 128.0)
    S["patch"] = np.c_[I, g64, g128, n64, n128]
    del A10, A30, CON, mean, std, mx, hop2

    pf = os.path.join(base, "pixel-features", f"pixfeat_{low}.parquet").replace("\\", "/")
    P = sql(f"select image_name, object_number, {', '.join(PIX)} from '{pf}'")
    pi = row_index(cells.image_name, cells.object_number, P.image_name, P.object_number)
    PX = np.full((n, len(PIX)), np.nan, np.float32)
    PX[pi >= 0] = P[PIX].values[pi[pi >= 0]]
    S["pix_only"] = harm(PX)
    S["pix"] = np.c_[I, S["pix_only"]]

    emb = {}
    for tag, sub in (("kronos", "kronos-zeroshot"), ("kronos2", "kronos2-zeroshot")):
        kc = sql(f"select image_name, object_number from '{os.path.join(base, sub, f'{tag}_cells_{low}.parquet').replace(chr(92), '/')}'")
        emb[tag] = (np.load(os.path.join(base, sub, f"{tag}_emb_{low}.npy"), mmap_mode="r"),
                    row_index(cells.image_name, cells.object_number, kc.image_name, kc.object_number))
    z = np.load(os.path.join(base, "novae-cells", f"novae_rep0_{folder[:-4]}.npz"), allow_pickle=True)
    ni = row_index(cells.image_name, cells.object_number, z["image_name"], z["object_number"])
    nov_ok = (ni >= 0)
    nov_ok[nov_ok] = z["valid_cohort"][ni[nov_ok]] & z["valid_img"][ni[nov_ok]]
    for name, fld in (("nov_cohort", "cohort"), ("nov_cohort_corr", "cohort_corr"), ("nov_img", "img")):
        L = np.zeros((n, 64), np.float32)
        L[ni >= 0] = z[fld][ni[ni >= 0]].astype(np.float32)
        S[name] = L

    test_imgs = np.flatnonzero(test[starts[:-1]])
    pool = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in test_imgs])
    pool = pool[(coarse[pool] >= 0) & nov_ok[pool]]
    print(f"{key}: {n:,} cells; LISI pool {len(pool):,} labelled test cells from {len(set(patient[pool]))} patients")
    return dict(S=S, emb=emb, coarse=coarse, pool=pool)


def knn(V, k=LISI_PERP * 3):
    Ds, Is = [], []
    for s in range(0, len(V), 2000):
        D = cdist(V[s:s + 2000], V)
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        d = np.take_along_axis(D, idx, 1)
        o = np.argsort(d, axis=1, kind="stable")
        Is.append(np.take_along_axis(idx, o, 1))
        Ds.append(np.take_along_axis(d, o, 1))
    D, I = np.vstack(Ds), np.vstack(Is)
    return D[:, 1:], I[:, 1:]


def calibrate(D, perplexity=LISI_PERP, tol=1e-5, n_tries=50):
    n = len(D)
    logU = np.log(perplexity)
    beta, bmin, bmax = np.ones(n), np.full(n, -np.inf), np.full(n, np.inf)

    def hp(b):
        P = np.exp(-D * b[:, None])
        s = P.sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            H = np.where(s > 0, np.log(s) + b * (D * P).sum(1) / s, 0.0)
            P = np.where(s[:, None] > 0, P / s[:, None], 0.0)
        return H, P
    H, P = hp(beta)
    for _ in range(n_tries):
        act = np.abs(H - logU) >= tol
        if not act.any():
            break
        pos, neg = act & (H - logU > 0), act & ~(H - logU > 0)
        bmin[pos] = beta[pos]
        beta[pos] = np.where(np.isfinite(bmax[pos]), (beta[pos] + bmax[pos]) / 2, beta[pos] * 2)
        bmax[neg] = beta[neg]
        beta[neg] = np.where(np.isfinite(bmin[neg]), (beta[neg] + bmin[neg]) / 2, beta[neg] / 2)
        Hn, Pn = hp(beta)
        H[act], P[act] = Hn[act], Pn[act]
    return H, P


def lisi_from(H, P, I, codes, n_cat):
    n, k = I.shape
    w = np.bincount((np.arange(n)[:, None] * n_cat + codes[I]).ravel(), P.ravel(), n * n_cat).reshape(n, n_cat)
    return 1.0 / ((w ** 2).sum(1) + np.where(H == 0, -1.0, 0.0))


# space -> (scale, label, how to build it for a set of picked cells)
SPACES = {
    "Intensity":                ("all",   "Intensity"),
    "Intensity, no cohort scaling": ("cell", "Intensity, pooled scaling"),
    "Rank":                     ("cell",  "Rank"),
    "Intensity + neighbours":   ("cell",  "+ neighbours (LightGBM input)"),
    "Novae [cohort]":           ("cell",  "Novae"),
    "Novae [cohort]-corr":      ("cell",  "Novae, batch-corrected"),
    "Novae [img]":              ("cell",  "Novae, per image"),
    "Intensity + pixel features": ("pixel", "+ pixel features"),
    "Pixel features only":      ("pixel", "Pixel features only"),
    "Intensity + patch means":  ("patch", "+ 64/128 µm patch means"),
    "KRONOS":                   ("patch", "KRONOS"),
    "KRONOS2":                  ("patch", "KRONOS2"),
    "Intensity + KRONOS2":      ("patch", "+ KRONOS2"),
}
KEYS = {"Intensity": "intensity", "Intensity, no cohort scaling": "pooled", "Rank": "rank", "Intensity + neighbours": "nbr",
        "Novae [cohort]": "nov_cohort", "Novae [cohort]-corr": "nov_cohort_corr", "Novae [img]": "nov_img",
        "Intensity + pixel features": "pix", "Pixel features only": "pix_only", "Intensity + patch means": "patch"}


def pcs(E, k=N_PC):
    E = (E - E.mean(0)) / (E.std(0) + 1e-9)
    U, s, _ = np.linalg.svd(E, full_matrices=False)
    return U[:, :k] * s[:k]


def build(space, CO, pick):
    order = ("Jackson", "Danenberg")
    if space in KEYS:
        return np.vstack([CO[k]["S"][KEYS[space]][pick[k]] for k in order]).astype(np.float64)
    tag = "kronos2" if "KRONOS2" in space else "kronos"
    E = []
    for k in order:
        arr, idx = CO[k]["emb"][tag]
        r = idx[pick[k]]
        assert (r >= 0).all(), f"{tag} embedding missing for some {k} cells"
        o = np.argsort(r)
        e = np.empty((len(r), arr.shape[1]), np.float64)
        e[o] = arr[r[o]].astype(np.float64)
        E.append(e)
    E = pcs(np.vstack(E))
    if space.startswith("Intensity +"):
        E = np.c_[np.vstack([CO[k]["S"]["intensity"][pick[k]] for k in order]), E]
    return E


def main():
    CO = {k: load(k) for k in ("Jackson", "Danenberg")}
    rs = np.random.default_rng(SEED + 41 + REP)       # same draws as novae-cells replicate 0
    rows = []
    for d in range(LISI_DRAWS):
        pick = {k: rs.choice(C["pool"], min(LISI_N, len(C["pool"])), replace=False) for k, C in CO.items()}
        coh = np.r_[np.zeros(len(pick["Jackson"]), np.int64), np.ones(len(pick["Danenberg"]), np.int64)]
        lab = np.r_[CO["Jackson"]["coarse"][pick["Jackson"]], CO["Danenberg"]["coarse"][pick["Danenberg"]]]
        for space in SPACES:
            V = build(space, CO, pick)
            V = (V - V.mean(0)) / (V.std(0) + 1e-9)
            D, I = knn(V)
            H, P = calibrate(D)
            il, cl = lisi_from(H, P, I, coh, 2), lisi_from(H, P, I, lab, len(COARSE))
            rows.append(dict(draw=d, space=space, dim=V.shape[1], ilisi_median=np.median(il), ilisi_mean=il.mean(),
                             clisi_median=np.median(cl), clisi_mean=cl.mean()))
            print(f"draw {d} {space:32s} iLISI {il.mean():.3f}  cLISI {cl.mean():.3f}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(HERE, "scale_lisi.csv"), index=False)
    agg = R.groupby("space")[["ilisi_mean", "clisi_mean", "ilisi_median", "clisi_median"]].agg(["mean", "min", "max"])
    agg.to_csv(os.path.join(HERE, "scale_lisi_summary.csv"))
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
