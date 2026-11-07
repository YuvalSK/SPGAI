"""Figure for §19: (A) partial r of H3K27me3 with the E→M axis per set and test; (B) residual H3K27me3 over axis deciles."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = pd.read_csv(os.path.join(HERE, "results.csv"))
D = pd.read_csv(os.path.join(HERE, "deciles.csv"))
COL = {"Basel": "#0072B2", "Zurich": "#56B4E9", "Keren": "#D55E00"}
LAB = {"primary": "Primary axis", "C2c common axis (vimentin - pan-CK)": "Vimentin − keratin axis",
       "R2 common axis (vimentin - K6/K17)": "Vimentin − keratin axis",
       "C2a + fine phenotype": "+ cell-cluster adjustment", "C2b + neighbour means": "+ neighbourhood adjustment",
       "C1 total H3 (negative control)": "Total H3 (negative control)", "C0 shuffled axis": "Shuffled axis"}
ORDER = ["primary", "C2c common axis (vimentin - pan-CK)", "R2 common axis (vimentin - K6/K17)", "C2a + fine phenotype",
         "C2b + neighbour means", "C1 total H3 (negative control)", "C0 shuffled axis"]

fig, (a, c, b) = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1.3, 1.1, 1]))
rows = []
for t in ORDER:
    for s in ("Basel", "Zurich", "Keren"):
        r = R[(R.set == s) & (R.test == t)]
        if len(r):
            rows.append((LAB[t], s, r.iloc[0]))
labels = list(dict.fromkeys(l for l, _, _ in rows))
off = {"Basel": -0.22, "Zurich": 0.0, "Keren": 0.22}
for lab, s, r in rows:
    y = labels.index(lab) + off[s]
    a.errorbar(r.r, y, xerr=[[r.r - r.lo], [r.hi - r.r]], fmt="o", ms=6, color=COL[s], capsize=0, lw=1.6)
a.axvline(0, color="0.5", lw=0.8)
a.set_yticks(range(len(labels)))
a.set_yticklabels(labels)
a.invert_yaxis()
a.set_xlabel("Partial correlation of H3K27me3 with the epithelial → mesenchymal axis\n(mean over patients, 95% patient-bootstrap CI)")
for s in ("top", "right"):
    a.spines[s].set_visible(False)

for s in ("Basel", "Zurich"):
    d = D[D.set == s]
    b.fill_between(d.decile, d.lo, d.hi, color=COL[s], alpha=0.2, lw=0)
    b.plot(d.decile, d["mean"], "o-", color=COL[s], ms=4, lw=1.6)
b.axhline(0, color="0.5", lw=0.8)
b.set_xticks(range(1, 11))
b.set_xlabel("Epithelial → mesenchymal axis decile (after covariates)")
b.set_ylabel("H3K27me3 after covariates (normal score)")
for s in ("top", "right"):
    b.spines[s].set_visible(False)

G = pd.read_csv(os.path.join(HERE, "posthoc_generic.csv"))
MK = [("vimentin", "Vimentin"), ("twist", "Twist"), ("slug", "Slug"), ("e_cadherin_p_cadherin", "E/P-cadherin"),
      ("pan_cytokeratin", "Pan-CK"), ("cytokeratin_8_18", "CK8/18"), ("cytokeratin_19", "CK19")]
for s_ in ("Basel", "Zurich"):
    for i, (m, l) in enumerate(MK):
        r = G[(G.set == s_) & (G.test == f"marker {m} + generic covariate")].iloc[0]
        y = i + off[s_]
        c.errorbar(r.r, y, xerr=[[r.r - r.lo], [r.hi - r.r]], fmt="o", ms=6, color=COL[s_], capsize=0, lw=1.6)
c.axvline(0, color="0.5", lw=0.8)
c.axhline(2.5, color="0.8", lw=0.8, ls="--")
c.set_yticks(range(len(MK)))
c.set_yticklabels([l for _, l in MK])
c.invert_yaxis()
c.set_xlabel("Partial correlation of H3K27me3 with each marker\n(also adjusted for a generic per-cell signal factor)")
for s_ in ("top", "right"):
    c.spines[s_].set_visible(False)

h = [plt.Line2D([], [], marker="o", color=COL[s], ls="", ms=6, label=l) for s, l in
     (("Basel", "Jackson Basel (IMC)"), ("Zurich", "Jackson Zurich (IMC)"), ("Keren", "Keren (MIBI, TNBC)"))]
fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"fig_epi_emt.{ext}"), dpi=200, bbox_inches="tight")
