# Profile proximity vs physical proximity: Danenberg IMC

## Version 3 update (2026-09-27): does ranking lose information? Panel-size curve
**Outputs:** `v3/`. The figure is `v3/fig_panel_size_curve.png`, the numbers are `v3/panel_size_curve.csv`, and the version 2 outputs are in `v2/`.

**What changed:** the same-type null bug is fixed (no infinite values now). The version 2 numbers below barely moved: A2 beyond type for the 6 markers with intensity went from −0.055 to −0.053, so they stand.

**Setting:** random marker sets of 6, 12, 18, 24 and 31, plus all 36; k = 30; touching cells excluded. Each difference is rank − intensity, with patient-bootstrap 95% CIs.

| Markers | Context R² (fine) | Same-phenotype share among the 30 profile neighbours | Closeness (log2 distance ratio; negative = rank closer) |
|---|---|---|---|
| 6 | −0.078 (0.187 → 0.109) | −0.084 | +0.041 (rank worse) |
| 12 | −0.039 | −0.076 | +0.022 (rank worse) |
| 18 | −0.019 | −0.069 | 0.000 |
| 24 | −0.015 | −0.068 | −0.007 |
| **31 (CyTOF size)** | **−0.011** (0.272 vs 0.262, ~4% relative) | **−0.071** | **−0.015 (rank better)** |
| 36 | −0.007 | −0.068 | −0.022 (rank better) |

**Reading:**
- **Physical context and closeness (label-free):** the rank penalty shrinks as the panel grows. From about 18 markers, rank is as good as intensity for closeness, and at 31–36 it is slightly better. For context R², the loss at 31 markers is 0.011, about 4%. So "ranking does no harm" holds approximately for tissue organisation at CyTOF-size panels.
- **Cell-type grouping:** rank has 7–8 points fewer same-phenotype neighbours at every panel size; it does not catch up.
  - **Caveat:** the reference phenotypes were defined by clustering intensity data, which favours intensity. The broad epithelial/immune/stromal version shows the same gap, shrinking from −0.09 to −0.04.
- **Bug in the automatic "no harm" flag for closeness:** the `no_harm` column in `panel_size_curve.csv` treats higher as better for every metric, but for the log2 distance ratio lower is better. The flag is therefore inverted for that metric. The values in the table above are correct.

---
*(Version 2 write-up below.)*

**Run:** Kaggle kernel `profile-vs-physical`, version 2, 2026-09-27.
**Data:** 1,062,219 cells, 704 tumour images, 650 patients. Hot-aggregate cells are removed, and images need at least 200 cells.
**Setting:** each cell's 30 nearest cells by marker profile (touching cells excluded, as a spillover guard) are compared with where those cells actually sit in the tissue. Stats are a patient bootstrap with 95% CI.
**Notebook:** `ProfileVsPhysical.ipynb`. The raw outputs are in this folder.

## Main numbers
| | 6 CyTOF-matched markers, intensity | 6 markers, rank | 36 markers, intensity | 36 markers, rank |
|---|---|---|---|---|
| Profile neighbours that are also among the 30 physical neighbours (chance = 3.0%) | 4.0% | 3.7% | 5.2% | 5.5% |
| How much closer profile neighbours sit than random cells | 9% | 7% | 15% | 16% |
| …of which is left after matching cell type | 4% | 3% | 9% | 10% |
| R² for guessing a cell's surroundings (cell-type mix of its 10 nearest cells) from its profile neighbours | 0.22 | 0.15 | 0.28 | 0.27 |
| Same, using random cells of the same types instead | 0.20 | 0.15 | 0.23 | 0.23 |
| **Gain beyond cell type (R²)** | **+0.019** | **+0.006** | **+0.043** | **+0.041** |
| Cross-patient transfer: surroundings R² when profile neighbours come from other patients | 0.015 | −0.017 | 0.045 | 0.044 |

All CIs are narrow (about ±0.001–0.01) because of the cell count, so every non-zero value above is statistically clear.

**Profile similarity by physical distance** (profile distance relative to the image average; 36 markers, intensity):

| Distance | 0–15 µm | 15–30 µm | 30–60 µm | 60–120 µm | 120–240 µm |
|---|---|---|---|---|---|
| Relative profile distance | 0.85 | 0.91 | 0.95 | 0.98 | 0.99 |

The similarity is short-range and fades by about 100 µm.

## What it means
1. **Look-alike cells are only slightly closer in tissue.** Among a cell's 30 profile neighbours, about 1–1.5 are among its 30 physical neighbours. Chance alone gives about 0.9.
2. **Most of the weak link is cell type.** Random cells of the same types reproduce most of it. After matching types, profile neighbours carry little extra information about surroundings: +0.02 R² with the 6 CyTOF-matched markers and +0.04 with all 36 markers.
3. **Transfer across patients fails for detailed surroundings.** R² is 0.015 with the 6 markers and 0.045 with 36. The broad class mix (epithelial / immune / stromal) transfers better (0.20–0.32), but again that is mostly cell type.
4. **Rank encoding is not better at this.** With the 6 shared markers, rank is worse than intensity on every measure; with 36 markers they are about equal. Rank's tighter grouping in marker space does not mean closer correspondence to the tissue.
5. **The 6 shared markers are a typical set of 6.** Their scores sit at the median of 20 random 6-marker sets. More markers help: 36 markers give about twice the gain beyond cell type.
6. **Spillover** explains part of the signal but not all of it. Excluding touching cells cuts the gain beyond type by about a third.

## Against the pre-registered rule
- Formally, every rule is met: closer than chance, and closer and more context-predictive beyond cell type, even with the 6 matched markers.
- **But the rule had no minimum effect size, and the effects are small.** In practice:
  - treat profile neighbours in CyTOF as "cells in a similar state, mostly of the same type", not as spatial neighbours;
  - **drop the "virtual space" transfer idea** (future_plan §13 Step 3). Cross-patient transfer of detailed surroundings is near zero.

## Known issue
- **The type-matched null could occasionally pick the cell itself.** This happens when the cell's phenotype has very few members in the image.
  - For the closeness numbers, that inflates the null distance, so "closer beyond type" is slightly overstated. It also produced 12 infinite values, only at k = 10 with 36 markers.
  - For the surroundings R², it makes the null slightly too good, so the gain beyond type is, if anything, understated.
  - Fix: exclude the cell itself when sampling the null, then re-run to firm up the beyond-type closeness numbers.
- Local pyarrow cannot read `a3_bands.parquet` and `per_image_metrics.parquet`; use `engine="fastparquet"`.
