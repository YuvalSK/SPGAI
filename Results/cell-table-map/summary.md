# Cell-table information map and virtual tumour state (Danenberg + Jackson IMC)

**Run:** Kaggle kernel `cell-table-map`, version 1, 2026-10-08. CPU, 198 min.
**Notebook:** `Scripts/CellTableMap.ipynb`. Plan: `future_plan.md` §14 (Steps 0–1) and §15-lite.

**Data:**

| Cohort | Cells | Images | Patients | Markers |
|---|---|---|---|---|
| Danenberg 2022 | 1.06M | 668 with contacts | 650 | 36 |
| Jackson 2020 | 1.13M | 542 | 349 | 32 + H3K27me3/H3 |

- Danenberg survival: 500 patients with complete covariates, 160 DSS events.
- Jackson Basel survival: 260 patients, 55 DSS events.

**Stats:** 5-fold CV grouped by patient; 1000× patient-bootstrap CIs.

## Step 0: immune cells avoid touching tumour cells
| | Danenberg | Jackson |
|---|---|---|
| Images with immune–tumour contact **depleted** vs label permutation (p < 0.05) | 90% | 88% |
| … enriched | 0.3% | 0.9% |
| Median log2(observed / expected contacts) | −1.10 (about half of chance) | −1.09 |
| Link to subtype | none (p = 0.87) | none (p = 0.07) |
| Link to ER | ER+ slightly less depleted (+0.25 log2, p = 0.015) | none (p = 0.52) |

- Immune and tumour cells are segregated in almost every image: there is no uniform immune–tumour attraction at the contact level.
- **Caveat:** the null shuffles labels within the image. Tumour cells forming nests are therefore enough to produce depletion.
- "Near but not touching" (for example at the nest border) was not tested here.

## Step 1: what predicts each marker in a cell
**Own other markers (redundancy) explain most of the variance for most markers.** Median R² by group:

| Group | Danenberg R² | Jackson R² |
|---|---|---|
| Antigen presentation | 0.83 | not in panel |
| Clinical | 0.71 | 0.65 |
| Epithelial lineage | 0.64 | 0.74 |
| Immune lineage | 0.62 | 0.64 |
| Stroma and vessels | 0.56 | 0.65 |
| Cell death | 0.46 | 0.59 |
| Immune state | 0.44 | not in panel |
| Tumour state | not in panel | 0.51 |
| Epigenetic (H3K27me3) | not in panel | 0.64 |
| Proliferation | 0.16 | 0.32 |

Least predictable markers:
- Danenberg: Ki-67 0.16, ICOS 0.16, OX40 0.14, CD15 0.17, CK5 0.19, CD57 0.19.
- Jackson: Ki-67 0.25, p53 0.36, pS6 0.40, pHH3 0.40.

**Neighbours add little.**
- The context gain (ΔR², own + neighbours − own) has a CI above 0 for 24/36 Danenberg and 22/33 Jackson markers, so the GK §14 reading is formally met in both cohorts.
- The gains are small: median ΔR² 0.004–0.02 per group.
- Largest gains:
  - Jackson: fibronectin +0.062, CD20 +0.056, CA-IX +0.051, CD44 +0.038, SMA +0.028, cleaved PARP +0.026, Ki-67 +0.025, c-Myc +0.025.
  - Danenberg: HER2 (D8F12) +0.044, CK5 +0.034, CD57 +0.027, SMA +0.024, CD16 +0.022.
- Almost all of the gain comes from **non-touching** cells within 30 µm. Touching cells add ≤ 0.015, so this is tissue context, not spillover. The only gains that come from contacts alone: Danenberg CD4; Jackson vimentin and EGFR.
- Like `profile-vs-physical`, this is statistically clear because of the cell count, but small in size. The pre-registered rule had no minimum effect size.

**H3K27me3 (Jackson):**
- R² = 0.64 from the cell's other markers, and 0.58 from Danenberg's channels only. Neighbours add nothing.
- **Artefact:** the per-H3 version (R² 0.92) is inflated by construction. The target contains H3, which is also an input. Do not use it. The fix is to predict H3K27me3 from H3/DNA only vs the full panel.

## Gate 1: Jackson-only markers predicted from Danenberg's channels
- All 21 pass (R² CI > 0). The bar is low.
- R² ranges from pHH3 0.20, pS6 0.24 and vWF 0.25 up to GATA3 0.74 and CK19 0.75.
- Tumour state: CA-IX 0.37, p53 0.27, EGFR 0.38, CD44 0.51, c-Myc 0.54, Twist 0.62, p-mTOR 0.64.

## Calibration: what the move from Jackson to Danenberg costs, cell table, 11 shared markers
- Cell-level r falls from 0.37–0.92 within Jackson to 0.02–0.85 in Danenberg.
- The worst are SMA (0.74 → 0.02), CD68 (0.63 → 0.32), CD20 (0.58 → 0.33) and CK5 (0.46 → 0.24).
- **The SMA collapse needs checking:** same clone, but a very different intensity scale (p99 56 in Jackson vs 9 in Danenberg).
- **Danenberg's "cd45" column is labelled CD45RO in its own table.** Verify this before treating it as shared.

## §15-lite: virtual tumour state in Danenberg (cell-table model)
**Against patient labels.** ΔAUROC is the gain over the patient means of the 14 input channels.

| Test | Virtual marker alone | Inputs | Inputs + virtual | ΔAUROC [95% CI] |
|---|---|---|---|---|
| PR+ vs PR− (all) | 0.69 | 0.73 | 0.74 | +0.008 [−0.001, 0.018] |
| PR+ vs PR− (ER+ only) | 0.57 | 0.58 | 0.60 | +0.020 [−0.002, 0.044] |
| Basal-like, CK14 | 0.77 | 0.85 | 0.85 | +0.002 [−0.005, 0.009] |
| **Basal-like, EGFR** | 0.68 | 0.85 | 0.88 | **+0.027 [0.005, 0.052]** |
| LumA vs LumB, GATA3 | 0.52 | 0.78 | 0.78 | −0.002 [−0.009, 0.005] |
| LumA vs LumB, PR | 0.57 | 0.78 | 0.78 | −0.002 [−0.011, 0.006] |

**Survival (DSS, adjusted): no virtual marker is prognostic.**
- **T1, direction:** the sign matches the measured marker in Jackson for 8 of 16 markers, which is chance level. No HR CI in Danenberg excludes 1.
- **T2, beyond clinical:** ΔC between −0.003 and +0.001 for every marker; none survives BH.
- **T3, beyond the inputs:** same, ΔC ≈ 0 everywhere.

**Observation (not pre-registered): imputation keeps the redundant part and loses the prognostic part, even within Jackson.**

| Marker | Measured HR per SD (Jackson Basel) | Out-of-fold virtual HR, same patients |
|---|---|---|
| pS6 | 1.44 [1.13, 1.83] | 1.08 [0.83, 1.40] |
| CD44 | 1.26 [0.97, 1.65] | 1.13 [0.85, 1.49] |
| CA-IX | 1.21 [0.95, 1.55] | 1.07 [0.81, 1.40] |

## Reading
1. **At the cell-table level, a virtual marker is mostly panel redundancy.** Most markers are 40–90% predictable from the cell's own other markers. Tissue context within 30 µm adds at most ~6% R² (CA-IX, CD20, fibronectin). Touching cells add almost nothing.
2. **Virtual tumour-state markers carry no clinical information beyond the measured channels.** The one exception is a small EGFR gain for basal-like status. Where the measured marker is prognostic (pS6 in Jackson), the virtual one is not.
3. **This does not settle the image-model question.** Images also contain subcellular morphology and texture that the cell table lacks. The kill test (§0.0) must show whether an image model beats these cell-table baselines. These results set a low prior for that, and make PIVOT-A ("virtual staining = panel redundancy") more likely.
4. **Your immune–tumour intuition:** at the contact level, immune cells are about 2× less likely than chance to touch tumour cells, in almost every image.

## Known issues
- The H3K27me3/H3 target is inflated by construction (see above).
- The cross-cohort SMA collapse and the CD45 vs CD45RO identity in Danenberg are unverified.
- Gate 1 has no minimum effect size, and neither does the context-gain rule.
- Danenberg broad classes come from the published phenotypes (rule: > 50% epithelial; stromal by name), as in earlier runs.
