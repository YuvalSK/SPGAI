"""H3K27me3 along the epithelial-mesenchymal axis of breast tumour cells (`future_plan.md` §19, pre-registered 2026-11-07).

Local CPU on saved outputs of `jackson-cells-prep` and `keren-cells-prep`. Writes results.csv, deciles.csv,
exploratory_*.csv, decision.json and fig_epi_emt.png/pdf next to this file.
"""
import os, json
import numpy as np, pandas as pd, duckdb
from scipy.stats import norm, spearmanr
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
J_CELLS = os.path.join(ROOT, "IMC data", "Jackson2020", "jackson-cells-prep", "cells.parquet").replace("\\", "/")
J_CLIN = os.path.join(ROOT, "IMC data", "Jackson2020", "jackson-cells-prep", "clinical.parquet").replace("\\", "/")
K_CELLS = os.path.join(ROOT, "IMC data", "Keren2018", "keren-cells-prep", "cells.parquet").replace("\\", "/")
SEED, N_BOOT, MIN_CELLS, COF = 20261107, 1000, 100, 5.0

J_M = ["vimentin", "twist", "slug"]
J_E = ["e_cadherin_p_cadherin", "pan_cytokeratin", "cytokeratin_8_18", "cytokeratin_19"]
J_COV = ["histone_h3", "dna1", "dna2", "log_area"]
K_COV = ["ds_dna", "log_area", "pan_keratin", "phospho_s6", "cd68"]     # Keren: no total H3; mass-neighbour/oxide channels


def nscore(df, cols, by="image_name"):
    """Rank-based normal scores within each image (average ranks for ties)."""
    r = df.groupby(by)[cols].rank(method="average")
    n = df.groupby(by)[cols[0]].transform("size").values[:, None]
    return pd.DataFrame(norm.ppf(r.values / (n + 1)), columns=cols, index=df.index)


def knn_means(df, cols, k=10):
    out = np.zeros((len(df), len(cols)), np.float64)
    V = df[cols].values
    for _, idx in df.groupby("image_name").indices.items():
        xy = df[["x", "y"]].values[idx]
        kk = min(k, len(idx) - 1)
        _, nb = cKDTree(xy).query(xy, k=kk + 1)
        out[idx] = V[idx][nb[:, 1:]].mean(1)
    return pd.DataFrame(out, columns=[f"nbr_{c}" for c in cols], index=df.index)


def partial_r(y, x, C):
    """Correlation of y and x after both are regressed on C (with intercept)."""
    C = np.c_[np.ones(len(y)), C]
    b = np.linalg.lstsq(C, np.c_[y, x], rcond=None)[0]
    R = np.c_[y, x] - C @ b
    s = R.std(0)
    return float((R[:, 0] * R[:, 1]).mean() / (s[0] * s[1])) if (s > 1e-12).all() else np.nan


def per_patient(df, y, x, covs, fine=False, shuffle=False, rs=None):
    rows = []
    for pid, d in df.groupby("patient_id"):
        if len(d) < MIN_CELLS:
            continue
        xv = d[x].values.copy()
        if shuffle:
            for _, ii in d.groupby("image_name").indices.items():
                xv[ii] = rs.permutation(xv[ii])
        C = [d[covs].values]
        if d.image_name.nunique() > 1:
            C.append(pd.get_dummies(d.image_name, drop_first=True).values.astype(float))
        if fine:
            C.append(pd.get_dummies(d.cell_phenotype, drop_first=True).values.astype(float))
        rows.append(dict(patient_id=pid, n=len(d), r=partial_r(d[y].values, xv, np.hstack(C))))
    return pd.DataFrame(rows).dropna()


def summarise(P, rs):
    z = np.arctanh(np.clip(P.r.values, -0.999, 0.999))
    bs = np.array([z[rs.integers(0, len(z), len(z))].mean() for _ in range(N_BOOT)])
    return dict(r=float(np.tanh(z.mean())), lo=float(np.tanh(np.percentile(bs, 2.5))), hi=float(np.tanh(np.percentile(bs, 97.5))),
                n_pat=int(len(z)), n_cells=int(P.n.sum()), frac_pos=float((P.r > 0).mean()))


def load_jackson():
    cols = ["image_name", "object_number", "patient_id", "cohort", "is_tumour", "metacluster", "cell_phenotype", "x", "y", "area",
            "histone_h3_trimethylate", "histone_h3", "dna1", "dna2", *J_M, *J_E]
    A = duckdb.sql(f"select {', '.join(cols)} from '{J_CELLS}'").df()
    A = A.rename(columns={"histone_h3_trimethylate": "h3k27me3"})
    for c in ["h3k27me3", "histone_h3", "dna1", "dna2", *J_M, *J_E]:
        A[c] = np.arcsinh(A[c].astype(float) / COF)
    A["log_area"] = np.log(np.maximum(A.area.astype(float), 1))
    A["patient_id"] = A.patient_id.astype(str)
    nb = knn_means(A, ["vimentin", "pan_cytokeratin", "h3k27me3"])       # neighbourhood over all cells of the image
    A = pd.concat([A, nb], axis=1)
    T = A[(A.is_tumour == 1) & A.metacluster.between(14, 27)].copy()
    return T


def load_keren():
    K = duckdb.sql(f"select * from '{K_CELLS}'").df()
    K["log_area"] = np.log(np.maximum(K.area.astype(float), 1))
    K["image_name"] = K.image_name.astype(str)
    K["patient_id"] = K.patient_id.astype(str)
    chans = ["h3k27me3", "h3k9ac", "vimentin", "keratin6", "keratin17", "pan_keratin", "phospho_s6", "cd68", "ds_dna"]
    nb = knn_means(K, ["vimentin", "keratin6", "keratin17", "h3k27me3"])
    K = pd.concat([K, nb], axis=1)
    return K[K.tumor_yn == 1].copy(), chans


def add_scores(T, cols):
    S = nscore(T, cols)
    for c in cols:
        T[f"z_{c}"] = S[c].values
    return T


def main():
    rs = np.random.default_rng(SEED)
    rows, expl, dec, between = [], [], [], []

    def record(set_, test, P, extra=None):
        s = summarise(P, rs)
        rows.append(dict(set=set_, test=test, **s, **(extra or {})))
        print(f"{set_:8s} {test:42s} r {s['r']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}]  patients {s['n_pat']}  pos {s['frac_pos']:.2f}", flush=True)
        return s

    # ---------------- Jackson ----------------
    J = load_jackson()
    jcols = ["h3k27me3", "histone_h3", "dna1", "dna2", "log_area", *J_M, *J_E, "nbr_vimentin", "nbr_pan_cytokeratin", "nbr_h3k27me3"]
    J = add_scores(J, jcols)
    J["axis"] = J[[f"z_{c}" for c in J_M]].mean(1) - J[[f"z_{c}" for c in J_E]].mean(1)
    J["axis_common"] = J.z_vimentin - J.z_pan_cytokeratin
    J = add_scores(J, ["axis", "axis_common"])                      # axes back to normal scores within image
    cov = [f"z_{c}" for c in J_COV]
    nbr = ["z_nbr_vimentin", "z_nbr_pan_cytokeratin", "z_nbr_h3k27me3"]
    for coh, name in (("basel", "Basel"), ("zurich", "Zurich")):
        d = J[J.cohort == coh]
        print(f"{name}: {len(d):,} epithelial tumour cells, {d.patient_id.nunique()} patients, {d.image_name.nunique()} images")
        prim = per_patient(d, "z_h3k27me3", "z_axis", cov)
        prim.assign(set=name).to_csv(os.path.join(HERE, f"per_patient_{name}.csv"), index=False)
        record(name, "primary", prim)
        record(name, "C1 total H3 (negative control)", per_patient(d, "z_histone_h3", "z_axis", ["z_dna1", "z_dna2", "z_log_area"]))
        record(name, "C2a + fine phenotype", per_patient(d, "z_h3k27me3", "z_axis", cov, fine=True))
        record(name, "C2b + neighbour means", per_patient(d, "z_h3k27me3", "z_axis", cov + nbr))
        record(name, "C2c common axis (vimentin - pan-CK)", per_patient(d, "z_h3k27me3", "z_axis_common", cov))
        record(name, "C0 shuffled axis", per_patient(d, "z_h3k27me3", "z_axis", cov, shuffle=True, rs=rs))
        # exploratory: single markers (sign: + = higher marker, more H3K27me3)
        for m in J_M + J_E:
            s = summarise(per_patient(d, "z_h3k27me3", f"z_{m}", cov), rs)
            expl.append(dict(set=name, test=f"marker {m}", **s))
        # deciles of the residualised axis (pooled within cohort): mean residual H3K27me3
        C = np.c_[np.ones(len(d)), d[cov].values]
        ry = d.z_h3k27me3.values - C @ np.linalg.lstsq(C, d.z_h3k27me3.values, rcond=None)[0]
        rx = d.z_axis.values - C @ np.linalg.lstsq(C, d.z_axis.values, rcond=None)[0]
        q = pd.qcut(rx, 10, labels=False)
        g = pd.DataFrame(dict(q=q, ry=ry, pid=d.patient_id.values)).groupby(["pid", "q"]).ry.mean().unstack()
        for k in range(10):
            v = g[k].dropna().values
            bs = [v[rs.integers(0, len(v), len(v))].mean() for _ in range(300)]
            dec.append(dict(set=name, decile=k + 1, mean=float(v.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5))))
        # between patients (raw scale): patient means
        pm = d.assign(k27=d.h3k27me3 - d.histone_h3, ax=d.vimentin - d.pan_cytokeratin).groupby("patient_id")[["k27", "ax"]].mean()
        pm = pm[d.groupby("patient_id").size().reindex(pm.index) >= MIN_CELLS]
        rho = spearmanr(pm.k27, pm.ax)
        between.append(dict(set=name, rho=float(rho.statistic), p=float(rho.pvalue), n_pat=len(pm)))

    # subtype strata (exploratory)
    clin = duckdb.sql(f"select * from '{J_CLIN}'").df()
    sub_col = next((c for c in ("clinical_type", "Subtype") if c in clin.columns), None)
    if sub_col:
        sub = clin.drop_duplicates("patient_id").set_index(clin.drop_duplicates("patient_id").patient_id.astype(str))[sub_col].astype(str)
        J["subtype"] = J.patient_id.map(sub).fillna("NA")
        for st, d in J.groupby("subtype"):
            P = per_patient(d, "z_h3k27me3", "z_axis", cov)
            if len(P) >= 10:
                expl.append(dict(set="Jackson (both)", test=f"subtype {st}", **summarise(P, rs)))

    # ---------------- Keren ----------------
    K, kch = load_keren()
    kcols = kch + ["log_area", "nbr_vimentin", "nbr_keratin6", "nbr_keratin17", "nbr_h3k27me3"]
    K = add_scores(K, kcols)
    K["axis_common"] = K.z_vimentin - K[["z_keratin6", "z_keratin17"]].mean(1)
    K["axis_pank"] = K.z_vimentin - K.z_pan_keratin
    K = add_scores(K, ["axis_common", "axis_pank"])
    kcov = [f"z_{c}" for c in K_COV]
    print(f"Keren: {len(K):,} tumour cells, {K.patient_id.nunique()} patients")
    record("Keren", "R2 common axis (vimentin - K6/K17)", per_patient(K, "z_h3k27me3", "z_axis_common", kcov))
    record("Keren", "C0 shuffled axis", per_patient(K, "z_h3k27me3", "z_axis_common", kcov, shuffle=True, rs=rs))
    expl.append(dict(set="Keren", test="+ neighbour means", **summarise(per_patient(K, "z_h3k27me3", "z_axis_common",
                kcov + ["z_nbr_vimentin", "z_nbr_keratin6", "z_nbr_keratin17", "z_nbr_h3k27me3"]), rs)))
    expl.append(dict(set="Keren", test="no crosstalk covariates", **summarise(per_patient(K, "z_h3k27me3", "z_axis_common", ["z_ds_dna", "z_log_area"]), rs)))
    expl.append(dict(set="Keren", test="pan-keratin axis, crosstalk regressed", **summarise(per_patient(K, "z_h3k27me3", "z_axis_pank",
                ["z_ds_dna", "z_log_area", "z_phospho_s6", "z_cd68"]), rs)))
    expl.append(dict(set="Keren", test="H3K9ac on common axis", **summarise(per_patient(K, "z_h3k9ac", "z_axis_common", kcov), rs)))
    for m in ("vimentin", "keratin6", "keratin17"):
        expl.append(dict(set="Keren", test=f"marker {m}", **summarise(per_patient(K, "z_h3k27me3", f"z_{m}", kcov), rs)))

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(HERE, "results.csv"), index=False)
    E = pd.DataFrame(expl)
    E.to_csv(os.path.join(HERE, "exploratory.csv"), index=False)
    pd.DataFrame(dec).to_csv(os.path.join(HERE, "deciles.csv"), index=False)
    pd.DataFrame(between).to_csv(os.path.join(HERE, "between_patients.csv"), index=False)
    print(E.round(3).to_string())
    print(pd.DataFrame(between).round(3).to_string())

    # ---------------- decision (§19 rules) ----------------
    g = lambda s, t: R[(R.set == s) & (R.test == t)].iloc[0]
    ex0 = lambda x: x.lo > 0 or x.hi < 0
    b = g("Basel", "primary")
    s = np.sign(b.r)
    D1 = bool(abs(b.r) >= 0.05 and ex0(b))
    none = bool(abs(b.r) < 0.05 and b.lo > -0.10 and b.hi < 0.10)
    z = g("Zurich", "primary")
    R1 = bool(D1 and np.sign(z.r) == s and ex0(z) and abs(z.r) >= 0.05)
    k = g("Keren", "R2 common axis (vimentin - K6/K17)")
    R2 = bool(D1 and np.sign(k.r) == s and ex0(k))
    C1 = {c: bool(abs(g(c, "C1 total H3 (negative control)").r) <= abs(g(c, "primary").r) / 2) for c in ("Basel", "Zurich")}
    C2 = {f"{c}|{t}": bool(np.sign(g(c, t).r) == s and ex0(g(c, t))) for c in ("Basel", "Zurich")
          for t in ("C2a + fine phenotype", "C2b + neighbour means", "C2c common axis (vimentin - pan-CK)")}
    C0 = {c: bool(abs(g(c, "C0 shuffled axis").r) < 0.01) for c in ("Basel", "Zurich", "Keren")}
    ok = all(C1.values()) and all(C2.values())
    level = ("replicated across hospitals and technologies" if D1 and R1 and R2 and ok else
             "replicated within IMC" if D1 and R1 and ok else
             "Basel only" if D1 and not R1 else
             "no association" if none else "inconclusive (rules not all met)")
    band = "substantive" if abs(b.r) >= 0.10 else "weak" if abs(b.r) >= 0.05 else "none"
    dec_ = dict(sign=int(s), basel_r=float(b.r), band=band, D1=D1, R1=R1, R2=R2, C1=C1, C2=C2, C0=C0, claim=level)
    json.dump(dec_, open(os.path.join(HERE, "decision.json"), "w"), indent=1)
    print(json.dumps(dec_, indent=1))


if __name__ == "__main__":
    main()
