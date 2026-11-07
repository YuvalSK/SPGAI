"""Figure: iLISI (cohort mixing) vs cLISI (cell-type mixing) per scale; intensity baseline in black.
Shaded = better than intensity on both axes (more cohort mixing AND types kept more apart)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
R = pd.read_csv(os.path.join(HERE, "scale_lisi.csv"))
A = R.groupby("space").agg(i=("ilisi_mean", "mean"), i_lo=("ilisi_mean", "min"), i_hi=("ilisi_mean", "max"),
                           c=("clisi_mean", "mean"), c_lo=("clisi_mean", "min"), c_hi=("clisi_mean", "max"))

OFF = {"+ pixel features": (-8, -16, "right"), "Pixel features only": (8, 4, "left"), "KRONOS": (-8, -14, "right"),
       "KRONOS2": (8, 4, "left"), "Novae, corrected": (-8, 8, "right"), "Novae": (8, -14, "left"),
       "Novae, per image": (8, -4, "left")}
FEAT, FM = "#0072B2", "#D55E00"                    # hand-built feature sets (hollow) vs learned embeddings (filled)
PANELS = [
    ("Pixel (within each cell)", [("Intensity + pixel features", "+ pixel features", FEAT, False),
                                  ("Pixel features only", "Pixel features only", FEAT, False)]),
    ("Patch (32–128 µm)", [("Intensity + patch means", "+ patch means", FEAT, False),
                           ("KRONOS", "KRONOS", FM, True),
                           ("KRONOS2", "KRONOS2", FM, True),
                           ("Intensity + KRONOS2", "+ KRONOS2", FM, True)]),
    ("Cell and neighbourhood", [("Intensity, no cohort scaling", "No cohort scaling", FEAT, False),
                                ("Rank", "Rank", FEAT, False),
                                ("Intensity + neighbours", "+ neighbours", FEAT, False),
                                ("Novae [cohort]", "Novae", FM, True),
                                ("Novae [cohort]-corr", "Novae, corrected", FM, True),
                                ("Novae [img]", "Novae, per image", FM, True)]),
]
b = A.loc["Intensity"]
xs = np.r_[A.i_lo.values, A.i_hi.values]
ys = np.r_[A.c_lo.values, A.c_hi.values]
xl = (1.0, 2.0)                                     # iLISI range for 2 cohorts
yl = (1.3, 2.3)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True, sharey=True)
for ax, (scale, items) in zip(axes, PANELS):
    ax.add_patch(Rectangle((b.i, yl[0]), xl[1] - b.i, b.c - yl[0], color="0.9", zorder=0, lw=0))
    ax.axvline(b.i, color="0.6", lw=0.8, zorder=1)
    ax.axhline(b.c, color="0.6", lw=0.8, zorder=1)
    ax.errorbar(b.i, b.c, xerr=[[b.i - b.i_lo], [b.i_hi - b.i]], yerr=[[b.c - b.c_lo], [b.c_hi - b.c]],
                fmt="o", ms=8, color="black", mfc="black", zorder=4, capsize=0)
    ax.annotate("Intensity", (b.i, b.c), xytext=(8, -14), textcoords="offset points", fontsize=9)
    for space, lab, col, filled in items:
        r = A.loc[space]
        ax.errorbar(r.i, r.c, xerr=[[r.i - r.i_lo], [r.i_hi - r.i]], yerr=[[r.c - r.c_lo], [r.c_hi - r.c]],
                    fmt="s" if filled else "o", ms=8, color=col, mfc=col if filled else "white", mew=1.8,
                    zorder=3, capsize=0)
        dx, dy, ha = OFF.get(lab, (8, 4, "left"))
        ax.annotate(lab, (r.i, r.c), xytext=(dx, dy), textcoords="offset points", fontsize=9, color="0.15", ha=ha)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_xlabel("Cohort mixing, iLISI (higher = more mixed)")
    ax.tick_params(labelleft=True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.0, 1.02, scale, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
axes[0].set_ylabel("Cell-type mixing, cLISI (lower = types kept apart)")
handles = [Line2D([], [], marker="o", ls="", color="black", ms=8, label="Intensity (baseline)"),
           Line2D([], [], marker="o", ls="", color=FEAT, mfc="white", mew=1.8, ms=8, label="Hand-built features"),
           Line2D([], [], marker="s", ls="", color=FM, ms=8, label="Foundation-model embedding"),
           Rectangle((0, 0), 1, 1, color="0.9", label="Better than intensity on both")]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"fig_scale_lisi.{ext}"), dpi=200, bbox_inches="tight")
print(A.round(3).to_string())
