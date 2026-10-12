# Does rank recover the subtype and receptor information that per-sample normalisation removes?

**Run:** Kaggle kernel `rank-recovery`, version 1, 2026-10-04.
**Notebook:** `RankRecovery.ipynb`. The setup, loaders, clinical tables, k-means and classifier are copied unchanged from `RankUtilityClinical`.
**Check:** the k = 20 and 40 AUCs for `int_img`, `rank_img` and `int_coh` reproduce `rank-utility-clinical` exactly (66 values, maximum difference 0.0).
**Outputs:**
- `fig_recovery.png`;
- `recovery_table.md` / `.csv`;
- `recovery_gap_closed.csv` (per target and pooled, all k);
- `recovery_auc.csv`;
- `oof_predictions.npz` (out-of-fold probabilities, so any CI can be recomputed locally);
- `summary_recovery.json`.

## Headline number (fixed before the run)
**% of the normalisation gap closed by rank** = Σ(AUC rank, per-sample z − AUC intensity, per-sample z) / Σ(AUC intensity, cohort z − AUC intensity, per-sample z).
- It is pooled over the subtype and receptor targets:
  - Danenberg: PAM50 + claudin-low, IntClust, ER, PR, HER2;
  - Jackson: clinical subtype, ER, PR, HER2.
- 95% CI from 1000 paired patient-bootstrap resamples.
- **Decision rule:** the claim holds in a cohort only if the CI is above 0 for ≥ 3 of 4 k values **and** for the cluster-free features (each patient's mean encoded vector).

## Result: the claim does not hold as stated, in either cohort
| | k = 10 | k = 20 | k = 40 | k = 80 | No clusters (patient mean) | Rule |
|---|---|---|---|---|---|---|
| Danenberg 2022 | −48% [−130, −7] | 6% [−14, 23] | 15% [−13, 36] | 25% [−2, 46] | −2% [−23, 14] | not met (0 of 4 k; no clusters ✗) |
| Jackson 2020 | **61% [37, 84]** | **59% [25, 92]** | **51% [14, 91]** | **34% [2, 64]** | −18% [−66, 9] | not met (4 of 4 k, but no clusters ✗) |

**Per target in Jackson, k = 20** (ΔAUC rank − intensity, per-sample z):

| Target | ΔAUC [95% CI] | % of gap closed |
|---|---|---|
| Clinical subtype | +0.101 [+0.052, +0.152] | 70% |
| ER | +0.094 [+0.033, +0.150] | 102% |
| PR | +0.053 [+0.003, +0.103] | 69% |
| HER2 | +0.001 [−0.099, +0.098] | 1% |

## Reading
1. **Jackson: rank closes a consistent 34–61% of the gap at every k, but only when patients are described by cell-state composition.**
   - With the patient's mean encoded vector there is no gain (−18%).
   - In fact, mean per-sample intensity predicts clinical subtype better (AUC 0.714) than per-sample intensity clusters do (0.670). Rank clusters (0.771) beat both.
2. **So rank does not restore sample-level level information.** The earlier wording in `rank-utility-clinical` was wrong. What rank does in Jackson is produce cell states that are more comparable across patients:
   - per-sample intensity states partly split patients rather than biology;
   - rank's states are shared by more patients (M1 in `rank-utility-clinical`, and rank's higher iLISI);
   - so rank-state compositions are better patient features.
3. **Danenberg does not replicate.** The gain grows with k (−48% at k = 10, 25% at k = 80), but no CI excludes 0. At k = 10 rank is significantly worse.
4. **HER2 is never recovered** (Jackson 1%, Danenberg −7%). HER2 status is a magnitude signal (amplification), which rank discards and cluster composition does not restore.
5. **Neither per-sample encoding reaches the ceiling.** Cohort-z intensity or plain pseudo-bulk remain the best patient features in both cohorts.

## How to present this
- **Defensible statement:** "In one of two cohorts (Jackson), cell-state compositions built on the rank encoding predicted tumour subtype and ER/PR status better than those built on per-sample-normalised intensity. They closed 59% [25, 92] of the gap to cohort-normalised intensity at k = 20, and 34–61% across k = 10–80. The gain was absent without clustering and did not replicate in Danenberg (6% [−14, 23]). It reflects more patient-comparable cell states, not recovered marker levels."
- **Main figure:** `fig_recovery.png`.
  - Panel A shows the gap (light bar), what rank recovers (dark line) and the ΔAUC with CI per target.
  - Panel B shows the pooled number across k and without clusters. That panel carries the caveat.
- **Supplementary table:** `recovery_table.md`.
