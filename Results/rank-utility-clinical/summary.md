# Does the rank encoding carry more clinical information than intensity?

**Run:** Kaggle kernel `rank-utility-clinical`, version 2, 2026-09-29. Version 1 crashed in the survival step: Basel nodal stage is coded "0", "1a" without the "n" prefix. Its classification results are identical to version 2's.
**Notebook:** `RankUtilityClinical.ipynb`.
**Data:**
- Danenberg: 1.06M cells, 650 patients, 553 of them with clinical data.
- Jackson: 1.13M cells, 349 patients; survival is available for the 265 Basel patients.

**Figure:** `fig_utility.png`.

## Design (fixed before the run)
- **Setting:** the CyTOF use case. Only single-cell profiles are used, no spatial information.
- **Encodings:** each encoding gets its own k-means cell states (k = 20 and 40, 3 seeds). A patient is described by the CLR of their state shares.
- **Normalisations:**
  - `img`: per-image robust z, as in all earlier runs;
  - `coh`: one robust z per marker over the whole cohort, which keeps absolute between-patient levels.
- **Reference feature sets:** the published phenotypes' composition, and pseudo-bulk (mean arcsinh intensity).
- **Models:** logistic regression or penalised Cox. 5 × 5-fold cross-validation, with the same folds for every feature set, and a paired patient-bootstrap CI for rank − intensity.
- **Heterogeneity test:** entropy of tumour-cell states (k-means on epithelial cells only) against disease-specific survival, adjusted for clinical covariates.

## Verdict: no clinical utility shown (`summary_utility.json`)
| Pre-set test | Rule | Result |
|---|---|---|
| T1 classification | Rank better (CI > 0) for ≥ half the Danenberg targets, same sign in Jackson | **Not met.** 1 of 7 in Danenberg (lymph nodes). Jackson goes the same way for all 5. |
| T2 survival | Rank adds DSS C-index on top of clinical covariates (CI > 0) in both cohorts | **Not met.** Δ −0.007 in Danenberg, −0.021 in Jackson; both CIs include 0. |
| T3 heterogeneity | Rank tumour-state entropy more prognostic than intensity's | **Not met.** HR per SD 1.02 vs 1.02 in Danenberg, 1.24 vs 1.12 in Jackson; all CIs include 1. |

## Key numbers (k = 20)
**Classification AUC, intensity → rank:**

| | `img` | `coh` | best feature set overall |
|---|---|---|---|
| Danenberg PAM50 | 0.69 → 0.70 | 0.74 → 0.73 | pseudo-bulk 0.74 |
| Danenberg HER2 | 0.69 → 0.67 | **0.94 → 0.78** | `int_coh` 0.95 |
| Danenberg grade 3 | 0.77 → 0.74 | 0.80 → 0.79 | pseudo-bulk 0.82 |
| Danenberg lymph nodes | 0.52 → **0.58** | 0.58 → 0.57 | `rank_img` k = 40, 0.61 |
| Jackson clinical subtype | 0.67 → **0.77** | 0.81 → 0.79 | pseudo-bulk 0.84 |
| Jackson grade 3 | 0.67 → **0.76** | 0.80 → 0.78 | `rank_img` k = 40 / pseudo-bulk, 0.81 |
| Jackson ER | 0.79 → **0.89** | 0.89 → 0.89 | `rank_coh` / `rank_img` k = 40, 0.90 |

**Survival C-index:**
- Clinical covariates alone: Danenberg DSS 0.704, RFS 0.668; Jackson DSS 0.741, OS 0.750.
- No feature set adds more than 0.005 to clinical alone in Danenberg. In Jackson the largest gain is +0.027 (`rank_coh`, k = 40, DSS). It was not tested against clinical-only with a CI; against `int_coh` it is +0.013 (CI −0.012 to 0.038).
- Features alone: rank is lower than intensity for Danenberg DSS (−0.035 to −0.042, about 0.56 vs 0.60).

## Reading
1. **Rank adds no clinical information beyond intensity once intensity is normalised sensibly.** With cohort-wide z, rank is never better, and it is clearly worse for HER2 (−0.16 AUC). HER2 amplification is a magnitude signal, and rank discards magnitude.
2. **Rank's gains appear only against per-image-normalised intensity, and mainly in Jackson** (+0.05 to +0.10 AUC for subtype, grade, ER and PR). Even then, cohort-z intensity or pseudo-bulk match or beat rank.
   - **Corrected 2026-10-04 by `rank-recovery`:** the earlier explanation, that rank restores the between-patient level differences per-image z removes, is wrong. In Jackson the gain holds at every k from 10 to 80 (34–61% of the gap closed), but it vanishes when patients are described by their mean encoded vector instead of cell-state composition.
   - So rank gives cell states that are more comparable across patients; it does not recover marker levels.
   - Danenberg does not replicate at any k.
   - See `../rank-recovery/summary.md`.
3. **There is no prognostic value beyond clinical covariates** for any single-cell composition feature set in either cohort, so survival cannot separate the encodings here.
4. **The "rank exposes tumour heterogeneity" idea was not supported:**
   - tumour-state entropy is not prognostic for either encoding;
   - rank and intensity entropies correlate only moderately (ρ 0.36–0.65), so they do measure different things;
   - rank entropy differs more across PAM50 subtypes than per-image intensity entropy does (η² 0.033 vs ≈ 0), but cohort-z intensity does as well or better (0.044).
5. **Mechanism (M1):**
   - rank's states are shared by somewhat more patients: median effective patients per state 443 vs 395 in Danenberg (`img`, k = 20), 247 vs 231 in Jackson. This is consistent with rank's higher iLISI;
   - no state was private to one patient in any encoding, so the pre-set "private-state share" metric is uninformative.

## Implication for the C2S / CyTOF story
Rank is a legitimate alternative encoding with a characterised trade-off. It gives sharper co-high marker grouping, weaker lineage grouping and slightly more cross-patient sharing, but no measurable clinical gain.

The one practical niche: when data must be normalised per sample (as with CyTOF batches), rank-based cell states can give better patient-level subtype and receptor features. This held in one of the two cohorts (Jackson), and only for cell-state composition (see `../rank-recovery/summary.md`).

## Caveats
- k-means is fitted on all patients without labels (transductive). The folds apply to the supervised step only.
- Danenberg images are about 1 per patient, so patient features come from small tissue areas.
- Jackson survival rests on 56 DSS events.
- k = 40 results are in the CSVs. They point the same way, with somewhat larger rank gains under `img` normalisation.
