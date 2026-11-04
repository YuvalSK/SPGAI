# Cross-cohort harmonisation: a CytoNorm-like correction for IMC (`future_plan.md` §18.7, §0.3 item 2b)

**Run:** Kaggle kernel `harmonise`, version 1, 2026-10-29 (CPU, 74 min, no failed blocks).
**Notebook:** `Scripts/Harmonise.ipynb` (setup cells shared verbatim with `prob-impute`).

**Setup:**
- Transfer only, Jackson → Danenberg and Danenberg → Jackson.
- LightGBM point, 9-level quantile and residual sampling, on the replicate-0 split.
- One model per source × marker; only the treatment of the test cohort's inputs changes.
- Scored on arcsinh(x/5) of the test cohort's measured values.
- Means are over 10 markers (ERα excluded), with 1000× patient-bootstrap CIs.

## Headline: correcting the inputs doesn't fix transfer; the missing piece is the target marker's own level
Quantile LightGBM. gap = mean |cov₈₀ − 0.80|.

| Arm | J→D cov₈₀ | J→D gap | J→D CRPS | D→J cov₈₀ | D→J gap | D→J CRPS |
|---|---|---|---|---|---|---|
| 1 source scaling (= `prob-impute`) | 0.57 | 0.27 | 0.129 | 0.44 | 0.36 | 0.135 |
| 2 cohort z | 0.58 | 0.25 | 0.122 | 0.38 | 0.42 | 0.144 |
| 3 per-cluster ComBat (cyCombine-like) | 0.57 | 0.25 | 0.125 | 0.38 | 0.42 | 0.142 |
| 4 per-cluster quantile (CytoNorm-like) | 0.57 | 0.25 | 0.124 | 0.39 | 0.41 | 0.139 |
| 5 GMM landmarks | **0.61** | **0.21** | **0.114** | 0.41 | 0.40 | 0.136 |
| 6a oracle: target level + scale | 0.58 | 0.22 | 0.096 | 0.64 | 0.16 | 0.125 |
| 6b oracle: target full marginal | 0.67 | 0.14 | 0.094 | 0.62 | 0.18 | 0.114 |

1. **Decision (pre-registered): not adopted.**
   - J→D: the best input correction (GMM landmarks) closes 46% of the gap to the oracle, just under the 50% bar. Its CRPS is better than arm 1 for 7/10 markers.
   - D→J: every input correction makes coverage *worse* than arm 1. GMM landmarks close −15%.
   - `decision.json`: `adopt_for_bayes_impute = false`.
2. **The level shift is real, but it sits in the target marker, not in the inputs.** Giving the model the test cohort's target marginal (the oracles) halves the gap in both directions: 0.27 → 0.14 and 0.36 → 0.16. Correcting only the inputs can't supply that information.
3. **Even the oracle stays overconfident** (cov₈₀ 0.62–0.67). The remaining gap of 0.13–0.18 is a change in how the markers relate. Climatology under the oracle map is calibrated (0.78–0.79), so the conditional intervals themselves are too narrow.
4. **Input correction moves errors between markers rather than removing them.**
   - Coverage improves for HER2 (0.54 → 0.74 J→D; 0.29 → 0.62 D→J), CD68 and CK5.
   - It collapses for CD20 (0.86 → 0.29–0.40; 0.65 → 0.11–0.20), CD45 (to 0.04–0.25) and Ki-67 in D→J.
   - The likely cause: once the inputs are aligned, the immune markers' predicted level is set by the source cohort's target scale, which is wrong for the other cohort.
5. **Cell-level accuracy barely moves.** R4 changes from 0.40 to 0.41–0.43 (J→D) and from 0.37 to 0.37–0.39 (D→J). Point RMSE improves only under the oracles.

## Biology check (all 15 inputs corrected)
| Arm | Cohort AUROC J→D | Patient-mean Spearman J→D (median / min) | Cohort AUROC D→J | Patient-mean Spearman D→J (median / min) |
|---|---|---|---|---|
| 1 source scaling | 1.00 | 1.00 / 1.00 | 1.00 | 1.00 / 1.00 |
| 2 cohort z | 1.00 | 1.00 / 1.00 | 1.00 | 1.00 / 1.00 |
| 3 per-cluster ComBat | 0.99 | 0.97 / 0.78 | 0.97 | 0.97 / 0.89 |
| 4 per-cluster quantile | 0.87 | 0.97 / 0.75 | 0.93 | 0.98 / 0.90 |
| 5 GMM landmarks | 1.00 | 1.00 / 1.00 | 1.00 | 1.00 / 1.00 |

- **The cohorts stay almost perfectly separable after every correction.** Per-cluster quantile matching mixes them the most (AUROC 0.87–0.93).
- **Between-patient structure within the test cohort is kept** (median Spearman ≥ 0.97). The per-cluster methods distort cell area and CK5 the most (min 0.75–0.90).
- So no correction "erased biology". They are mild, and they still don't help, which points at the target marker rather than the inputs.

## What this means for `bayes-impute` (§18.6.1)
- **Supports the planned design.** The fix has to come from the target marker's cohort offset, i.e. the cohort random effect (`c_cohort`, τ_b), and not from preprocessing.
- **For an unseen cohort (S2)**, intervals must widen by the uncertainty of that offset. Quantile LightGBM can't do this.
- **With a few labelled patients (S3)**, the offset is learnable. The oracle arms bound the gain at about half the coverage gap and about 25% of the CRPS.
- **Keep source scaling (arm 1) as the S2 preprocessing.** GMM landmarks are an optional J→D sensitivity check only.

## Caveats
- One replicate. The 50% bar was missed narrowly in J→D (46%).
- The oracles use the test cohort's target marginal, so they are upper bounds, not methods.
- k-means (30 clusters) on rank-normalised inputs is a stand-in for cyCombine's SOM, and ComBat here is location–scale without empirical-Bayes shrinkage.
- The biology check replaces the cell-type-fraction check in §18.7: input correction doesn't change the published labels, so the fractions can't move.
