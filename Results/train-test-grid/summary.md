# Train-on × test-on grid: Jackson, Danenberg and both cohorts (11 shared markers)

**Run:** Kaggle kernel `train-test-grid`, version 1, 2026-10-11 (GPU, 282 min of the 660-min budget, no units skipped).
**Notebook:** `Scripts/TrainTestGrid.ipynb`.
**Combined locally:** `combine_reps.py`.

**Question:**
- How much of the Jackson → Danenberg drop is a cohort-transfer gap?
- Does training on both cohorts close it?

**Grid:** train on {Jackson, Danenberg, Both} × test on {held-out Jackson patients, held-out Danenberg patients}, over 3 replicates.

**Models:** one per target marker, same as `model-comparison`:
- Table (LightGBM);
- MLP;
- GCN;
- GraphSAGE.

**Inputs:** the other 10 shared markers, H3, DNA1, DNA2 and log area. Values are arcsinh-transformed and harmonised per cohort.

**Splits:**
- **Jackson:** 70/10/20 by patient, the same draws as `model-comparison(-reps)`. The test set is 55–76 patients.
- **Danenberg:** the frozen test patients from `danenberg-cells-prep` (130 patients, 228,511 cells). Train and validation are redrawn per replicate.

**Scoring:**
- R4 = per-cell r after removing each image's mean, with a 1000× patient bootstrap. A call counts when its CI excludes 0.
- Danenberg truth = per-cell image means (`cell mean`, primary). The kill-test `target map` gives the same picture, about +0.02 to +0.06 higher.
- ERα is excluded from medians and call counts. Its R4 is fine here (0.80–0.84 within Danenberg, in both truth spaces), because removing the image mean also removes most of the saturation problem.

**Replicate spread:** small for Danenberg test medians (SD ≤ 0.02, except Jackson-trained GraphSAGE at 0.04). Larger for Jackson test medians (SD up to 0.06), mostly from the Jackson patient split.

## Median R4 over 10 markers (mean ± SD over 3 replicates)
| Test on → | Danenberg (130 patients) | | | Jackson (held-out patients) | | |
|---|---|---|---|---|---|---|
| **Train on →** | **Danenberg** | **Jackson** | **Both** | **Jackson** | **Danenberg** | **Both** |
| Table | 0.72 ± 0.00 | 0.44 ± 0.01 | 0.72 ± 0.00 | 0.63 ± 0.03 | 0.37 ± 0.01 | 0.62 ± 0.03 |
| MLP | 0.73 ± 0.00 | 0.47 ± 0.02 | 0.70 ± 0.00 | 0.64 ± 0.04 | 0.30 ± 0.01 | 0.64 ± 0.04 |
| GraphSAGE | **0.74 ± 0.01** | 0.50 ± 0.04 | **0.74 ± 0.00** | **0.65 ± 0.04** | 0.34 ± 0.03 | **0.64 ± 0.03** |
| GCN | 0.50 ± 0.01 | 0.42 ± 0.02 | 0.49 ± 0.00 | 0.51 ± 0.05 | 0.29 ± 0.06 | 0.50 ± 0.05 |
| ResNet-UNet residual / gamma / immuvis (Jackson-trained only) | — | 0.15 / 0.21 / 0.25 | — | — | — | — |

**The median-of-medians gap (0.72 vs 0.44) overstates the typical transfer gap.** The 10 markers split into a high group and a low group, and the Jackson-trained median falls between them. The median *per-marker* gap (own cohort − other cohort, paired) is:
- Danenberg test: +0.06 to +0.11;
- Jackson test: +0.08 to +0.17.

## Per-marker R4, Table model (mean over 3 replicates)
| Marker | Danenberg test: own | Jackson → | Both | Jackson test: own | Danenberg → | Both |
|---|---|---|---|---|---|---|
| panCK | 0.88 | 0.85 | 0.88 | 0.92 | 0.88 | 0.91 |
| CD45 | 0.79 | 0.68 | 0.78 | 0.81 | 0.67 | 0.80 |
| CD3 | 0.77 | 0.69 | 0.76 | 0.59 | 0.43 | 0.58 |
| CK8/18 | 0.73 | 0.74 | 0.74 | 0.82 | 0.76 | 0.82 |
| HER2 | 0.71 | 0.64 | 0.71 | 0.67 | 0.53 | 0.67 |
| SMA | 0.79 | **−0.17** | 0.73 | 0.80 | **−0.13** | 0.75 |
| CD68 | 0.51 | 0.22 | 0.51 | 0.49 | 0.31 | 0.47 |
| CD20 | 0.45 | 0.23 | 0.41 | 0.41 | 0.23 | 0.41 |
| Ki-67 | 0.28 | 0.23 | 0.30 | 0.24 | 0.11 | 0.16 |
| CK5 | 0.26 | 0.19 | 0.25 | 0.36 | 0.23 | 0.32 |
| *(ERα)* | *0.83* | *0.67* | *0.81* | *0.55* | *0.48* | *0.56* |

## Calls (better / worse, out of 10, for replicates 0 / 1 / 2)
| Comparison | Test | Table | MLP | GraphSAGE | Same non-n.s. call in all 3 (Table) |
|---|---|---|---|---|---|
| Own − other cohort | Danenberg | 9/0, 9/0, 8/0 | 7/0, 7/0, 9/0 | 7/0, 6/0, 8/0 | better: HER2, CD20, CD3, CD45, CD68, Ki-67, panCK, SMA |
| Own − other cohort | Jackson | 10/0 ×3 | 10/0, 10/0, 9/0 | 10/0, 9/0, 9/0 | better: all 10 |
| Both − other cohort | Danenberg | 8/0, 9/0, 8/0 | 7/0, 7/0, 9/0 | 7/0, 8/0, 8/0 | better: 8 markers |
| Both − other cohort | Jackson | 9/0, 9/0, 10/0 | 10/0 ×3 | 10/0, 9/0, 8/0 | better: 9 markers |
| Both − own cohort | Danenberg | 2/3 ×3 | 3/4, 1/4, 1/5 | 3/5, 1/3, 1/4 | worse: CD20, CD45, SMA; better: CK8/18, panCK |
| Both − own cohort | Jackson | 0/6, 0/4, 0/4 | 0/6, 0/4, 1/1 | 0/6, 0/7, 0/5 | worse: CD45, SMA |
| ResNet-UNet (each arm) − Table, Jackson-trained | Danenberg | 1/8–1/9 | | | worse: 8–9 markers; better: SMA |
| ResNet-UNet (each arm) − Table, Danenberg- or Both-trained | Danenberg | 0/10 ×3 | | | worse: all 10 (median −0.38 to −0.55) |

**Within-cohort spatial context** (GraphSAGE vs MLP / Table, same training cohort):
- **Danenberg-trained on Danenberg:** small gains for CD3 and CD68 in all 3 replicates, e.g. CD68 0.54 vs 0.50 / 0.51 and CD3 0.79 vs 0.78 / 0.77. Median +0.01.
- **Jackson-trained on Jackson:** CD45, SMA, and Ki-67 vs the MLP, as in `model-comparison-reps`.
- The two cohorts show different context markers, and every gain is ≤ 0.04.

## What holds
1. **Within Danenberg, the 11 shared markers are highly redundant at the cell level.**
   - A per-cell model on the other 10 markers + nuclear channels reaches a median R4 of 0.72–0.74.
   - panCK reaches 0.88, CD45 0.79, CD3 0.77 and SMA 0.79.
   - This is the within-cohort ceiling the image models would have to beat, and they were never trained within Danenberg.
2. **The cohort-transfer gap is real but concentrated.**
   - Jackson → Danenberg loses most on SMA (≈ 0.9–1.0, the sign flips), CD68 (≈ 0.25–0.29), CD20 (≈ 0.15–0.22), and CD45 / CD3 / HER2 (≈ 0.06–0.13).
   - CK8/18, panCK, Ki-67 and CK5 lose ≤ 0.07.
   - Transfer is worse in the other direction: Danenberg → Jackson loses on 9–10 of 10 markers, with a median per-marker loss of 0.14–0.17 (Table, MLP, GraphSAGE).
3. **Pooling both cohorts closes nearly all of the gap.**
   - On each test cohort, the median paired difference between Both and the own-cohort model is 0.00 to −0.02. For Table, MLP and GraphSAGE, Both beats the other-cohort model on 7–10 markers in every replicate.
   - SMA, which flips sign across cohorts, recovers to 0.73–0.75 under Both. So the input–SMA relation differs between cohorts, and a pooled model can fit both.
4. **Pooling has a small, marker-specific cost.**
   - SMA is worse under Both than own-cohort for every model, in both cohorts, in all 3 replicates (≈ −0.05 to −0.08).
   - Other costs depend on the model: MLP CD20 in Danenberg 0.49 → 0.32; GraphSAGE Ki-67 and CK5 in Jackson 0.28 → 0.14 and 0.39 → 0.27.
   - Among Table, MLP and GraphSAGE, the Table model has the smallest pooling cost.
5. **The image models are far below every per-cell model.**
   - Against the like-for-like Jackson-trained Table model, the median loss is −0.24 to −0.27, with SMA the only win. This replicates `kill-test` and `model-comparison-reps`.
   - Against a per-cell model trained in Danenberg or on both cohorts, they lose all 10 markers. That comparison mixes the model gap with the transfer gap, because no Danenberg-trained image model exists.
6. **GCN has the lowest median of the per-cell models in every cell of the grid.** Its median paired loss vs Table / MLP is −0.10 to −0.19, except Danenberg → Jackson (−0.04 to −0.08).

## Caveats
- The Danenberg-trained model learns and is scored in the same feature space: per-cell image means over the published masks. The Jackson-trained model learns from Jackson's published table. Part of the "transfer gap" may therefore be a feature-definition mismatch rather than biology.
- Danenberg medians use only the 130 frozen test patients. `model-comparison(-reps)` scored all Danenberg cells. That explains small differences in the Jackson-trained numbers (e.g. GraphSAGE 0.50 here vs 0.47 there).

## What this means for the research question
- For the 11 shared markers, cell-level imputation is mostly a panel-redundancy problem. A per-cell model trained in the target cohort, or on pooled cohorts, gets most of the way, and spatial context adds ≤ 0.04.
- The cross-cohort penalty that the image-model pilot carried is largely removable by pooling training data. Most of the remaining gap to the within-cohort ceiling is model choice (image vs per-cell), not cohort.
- This strengthens the PIVOT-A reading of `kill-test`.

## Files
- `median_r4_by_rep.csv`: per test × truth × train × model, median R4 for each replicate, plus mean and SD.
- `median_r4_summary.csv`: the kernel's own version of the same.
- `r4_per_marker_mean_sd.csv`: per test × truth × train × model × marker, mean / SD / min / max over replicates.
- `calls_by_rep.csv`: per-marker call and paired difference in each replicate.
- `call_counts_by_rep.csv`: better/worse counts per comparison and replicate, plus markers with the same call in all 3.
- Raw files: `scores_rep{0,1,2}.csv`, `differences_R4_rep{0,1,2}.csv`, `training_log_rep{0,1,2}.csv`, `run_info.json`, and the kernel log `train-test-grid.log`.
