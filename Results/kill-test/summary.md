# Kill test: does the image model beat the cell table? (`future_plan.md` §0.0)

**Run:** Kaggle kernel `kill-test`, version 1, 2026-10-09. T4 GPU, 156 min (75 min on GPU).
**Notebook:** `Scripts/KillTest.ipynb`.
**Large per-cell outputs:** `IMC data/Danenberg2022/kill-test/`.

**Setting:**
- The 3 existing Jackson → Danenberg checkpoints (joint ResNet-UNet, residual / gamma / immuvis), 11 shared markers.
- 1.06M Danenberg cells in tumour images, per-cell means over the published masks.
- Cell-table baselines trained on the Jackson table:
  - **B1:** the cell's own inputs;
  - **B2:** + neighbours;
  - **B1r:** rank harmonisation.
- Patient-bootstrap CIs.

## Headline: the cell table beats the image model at the cell level, for 10 of 11 markers
**R4** (per-cell r, image mean removed):

| Marker | B1 (table) | Residual | Gamma | Immuvis |
|---|---|---|---|---|
| panCK | **0.84** | 0.25 | 0.52 | 0.54 |
| CK8/18 | **0.73** | 0.19 | 0.25 | 0.47 |
| HER2 | **0.70** | 0.44 | 0.43 | 0.36 |
| CD3 | **0.66** | 0.09 | 0.22 | 0.23 |
| ERα | **0.61** | 0.53 | 0.34 | 0.31 |
| CD45 | **0.56** | 0.34 | 0.34 | 0.29 |
| CD20 | **0.30** | 0.26 | 0.27 | 0.09 |
| CK5 | **0.26** | 0.11 | 0.09 | 0.13 |
| Ki-67 | **0.24** | 0.14 | 0.14 | 0.06 |
| CD68 | **0.19** | −0.05 | −0.04 | −0.05 |
| SMA | −0.16 | **0.19** | **0.25** | **0.23** |

- B1 − image has a CI < 0 for every marker except SMA. SMA is the marker whose cross-cohort table transfer collapses (`cell-table-followup`).
- Shifted-map nulls give R4 ≈ −0.03 to 0. So the image model does carry real cell-level signal, but much less than a per-cell regression on the same channels.

**Downstream readouts follow the same order:**
- **Phenotype classifier, macro-F1 change on substitution** (measured F1 0.645):
  - The image arms lose more than B1, e.g. panCK −0.14 to −0.21 vs −0.06, ERα −0.05 to −0.07 vs −0.02.
  - The image arms are close to their own shifted-map null (panCK residual −0.215 vs shifted −0.239).
  - Substituting the measured truth costs about 0, as it should.
- **M+ fraction per patient × class, r with measured:**
  - B1: 0.28–0.78 (SMA −0.13);
  - image arms: mostly −0.08 to 0.47;
  - shifted null: up to 0.22.
- **HER2 status from patient means:** the measured target gives AUROC 0.86. Every imputation drops to 0.43–0.58, image and table alike. Amplification is a magnitude signal that the other markers don't carry.

## A data problem found along the way: Danenberg ERα saturates in the model's target space
- The target normalisation uses Jackson's ER percentiles, but Jackson's "ER" is a different antibody and metal (Gd156 "Rabbit IgG H L") from Danenberg's (Ho165).
- In Danenberg, per-cell ERα in target space is compressed against the ceiling: median 0.76, p10 0.34 (high background), p90 0.98, and 8% of cells at the clip value of 1.
- As a result:
  - the measured target-space ERα predicts clinical ER status at only AUROC 0.57;
  - the image predictions are **inverted** (0.29–0.34, shifted maps too), so the predicted per-image ER level tracks background, not ER positivity.
- **This affects earlier claims.** E1/E2's "ERα is the best-imputed marker (between-image r 0.82)" was measured in the same saturated space, so it largely reflects background level, not ER biology. Recheck before citing.

## Pre-registered decisions
| Rule | Result |
|---|---|
| Image beats table (R4 vs B1, CI > 0, ≥ 2 markers in one arm) | **Fails.** Only SMA, in every arm. The table is better for the other 10. |
| Positive controls (R5 must be ≈ 0 for the state-only controls) | **Fail:** state mean R5 −0.17 to −0.45; shuffled-within-state R5 +0.06 to +0.48. 0 of 11 markers pass. R5 is **not validated**. |
| Beyond-panel (R5 arm > shifted null, ≥ 2 markers) | Formally met (5–7 markers per arm, R5 0.03–0.12), but **not interpretable** because R5 failed its controls. B2 (table + neighbours) reaches similar R5 (up to 0.10). |
| D1 (pooled r and R4 rank arms oppositely) | Met for 2 markers: CD45 (residual vs immuvis), ERα (gamma vs immuvis). |
| D2 (R4 + R5 add ≥ 0.1 LOMO R² for the readouts) | Fails (gains −0.18, −0.05). |
| Automatic outcome | "GO", but **invalid**: it relies on the unvalidated R5. |

**Corrected reading:**
- The decisive rule (image beats table) fails, and the R5 rung is not yet valid.
- Under §0.0 this is **PIVOT-A**, or a KILL for the image-model zoo in its current form.
- Pooled r and R4 do dissociate (D1). That supports the metric critique, but within a model family that loses to a simple per-cell regression.

## Why the R5 controls failed (to fix before R5 is used again)
- **"State mean" control:** it was itself cross-fitted, so each half's control values come from a model trained on the *other* half's truth. Re-residualising it leaks the evaluated half's own truth, which produces the negative R5.
- **"Shuffled within state" control:** it shuffled within image × state decile, so it keeps each image's state-specific offset. R5 credits that offset as beyond-panel information.
- **Fix:** residualise on the cell's markers *and* the image, i.e. image × state fixed effects. Then rebuild the controls so that each is computed from the same fold model that R5 uses.

## What this means for the research question
At the cell level, the generative image models trained here don't enable "virtual cells" beyond what a per-cell regression on the measured channels gives; they are worse. The cell-table model is the stronger "virtual cell" engine. Its information is redundant with the panel by construction, and in `cell-table-map` it added no clinical value.
