# Future Plan: When Can a Virtually Stained Marker Be Trusted?

**Working title:** *A resolution ladder for virtual staining: which imputed IMC markers preserve which biology*
**Scope:** Combined paper of Direction 1 (validated evaluation metric) + Direction 3 (panel substitution / downstream biology).
**Grant fit (CZI, "Towards a unified atlas of spatial proteomics"):** Aim 1 (SpatialProteomicsNet as data backbone), Aim 2 (breast stain-to-stain translation), Aim 4 (web dictionary reframed as a per-marker trust table).
**Target venues:** NeurIPS Datasets & Benchmarks, RECOMB, ISMB; journal fallback *Cell Systems* / *Bioinformatics*.

---

## 1. Motivation

**Known.**
- Virtual staining models (ImmuVis, VirTues, EVA, our ResNet-UNet) are ranked by pooled pixel correlation. FID is also used, but it relies on ImageNet features and has no clear meaning for IMC.
- Pooled correlation mixes two components exactly. With η = the fraction of a channel's variance that lies between images:
  `r_pooled = √(η_true·η_pred)·r_between + √((1−η_true)(1−η_pred))·r_within`
- Our pilot (Jackson → Danenberg, 11 markers, 3 preprocessing arms) shows:
  - Predictions recover real within-image structure. ERα: between-image r ≈ 0.82, within-image r ≈ 0.52.
  - Markers fall into tiers that are stable across arms. High: ERα, panCK. Mid: HER2, CD45, CK8/18, SMA. Low: CD20, Ki-67, CD3, CK5, CD68.
  - Low-pass preprocessing inflates between-image r for weak markers without improving within-image r. CD3: between 0.31 → 0.60, within 0.09 → 0.11.

**Unknown.**
1. At what spatial level is an imputed marker actually correct: patient, image, tissue compartment, cell, or within a cell state?
2. Within-image r mixes "where cells are" (almost free from the DNA channels) with marker-specific signal. How much of it is marker-specific?
3. Which downstream analyses stay valid when a measured marker is replaced by an imputed one?

## 2. Hypotheses

- **H1 (metric).** Pooled correlation and the rung-by-rung readout rank models and preprocessing pipelines differently. Pooled r rewards abundance and coarse compartment accuracy.
- **H2 (substitution).** The finest rung a marker reaches predicts which downstream analyses survive substitution:
  - abundance or compartment rung → patient-level analyses survive (ER/HER2 status, survival stratification)
  - within-cell-state rung → cell phenotyping and neighborhood analyses also survive
  - The rung predicts this better than pooled r does.
- **H3 (cross-cohort cost).** Moving from within-cohort to cross-cohort imputation costs most at the fine rungs. Simple intensity calibration recovers only part of that cost.

Any outcome is reported. For example, if all rungs degrade equally, that is a result, not a failure.

---

## 3. Data and evaluation settings

| Setting | Train | Test | Purpose |
|---|---|---|---|
| **S1 (main)** | Danenberg 2022 (METABRIC IMC, ~718 patients / 794 images) | Held-out Danenberg patients (5-fold, patient-level CV) | Within-cohort reference; same antibodies for input and target |
| **S2 (replication)** | Jackson 2020 (Basel) | Held-out Jackson patients (patient-level CV) | Do the rungs replicate in an independent cohort? |
| **S3 (harmonization)** | Jackson 2020 | Danenberg 2022 | Cost of crossing cohorts (S1 → S3 drop), which is what the atlas needs |

- **Why Danenberg is the main cohort:** most patients; METABRIC clinical data (ER/HER2, PAM50, long survival follow-up); published cell masks and single-cell tables.
- **Target markers:** measured in both cohorts and spanning the tiers. ERα and panCK (high); HER2 and CD45 (mid); CD3 and Ki-67 (low, as negative controls for signal quality). Extend to all 11 shared markers if compute allows.
- **Ground-truth caveat:** check the antibody clone and metal tag for each shared marker in both panels. A clone mismatch caps S3 performance and must be reported.
- **Out of scope for the main paper:** MIBI-TOF (Keren 2018, ~41 patients) is too small for patient-level endpoints. It is a possible supplementary check across technologies.
- **Dropped:** a foundation-model pretraining comparison (the old Direction 2). Hidden overlap with the IMC17M / SPM-47M pretraining sets cannot be ruled out without unpublished data. Revisit only if a collaborator provides an unpublished cohort.

**Frozen before any training:** patient folds for S1 and S2, segmentation masks, the choice of preprocessing arm (on validation folds only), and every threshold used in section 7.

---

## 4. Preprocessing

**Rules:**
1. **The target marker is never normalized per image.** Use arcsinh with a fixed cofactor, scaled with **training-cohort** statistics. Per-image min-max erases the abundance rung.
2. **Inputs** are scaled with training-cohort statistics. The gamma correction is allowed: it uses the DNA and H3 channels within each image as an anchor, and those channels are always measured and never imputed.
3. **The main arm is chosen on validation folds**, between gamma and ImmuVis-style but with global (not per-image) scaling. The other arm goes to the supplement.

**Access to the test cohort in S3:**

| Level | Uses from Danenberg | Status |
|---|---|---|
| A | Nothing | **Main result** |
| B | Per-channel quantile matching of input markers to Jackson (unlabeled) | Secondary; labeled as using the test inputs |
| C | Any statistics of the target marker | **Forbidden** (leaks the answer) |

The A-vs-B gap measures how much of the cross-cohort cost is intensity calibration and how much is deeper (antibody clone, staining).

## 5. Models

- **Main model:** the ResNet-UNet + HyperConv joint model (existing code), trained per setting.
- **3 seeds minimum** per setting × preprocessing arm. All comparisons use patient-level bootstrap CIs across seeds.
- **Optional:** one public foundation model with native imputation (e.g. ImmuVis), used only in S1/S2 **as an example of applying the ladder**, not as a comparison between models. Given likely pretraining overlap, its numbers are not interpreted as generalization.

---

## 6. D1 — The resolution ladder

### 6.1 Segmentation (independent of the target marker)
- Nuclei from **DNA1, DNA2 and H3 only**, with a fixed off-the-shelf model (Cellpose or Mesmer), then a fixed pixel expansion. No membrane channels, because panCK and CD45 are targets.
- One frozen mask set per cohort, shared by all models and markers.
- Cross-check against the published Danenberg masks.

### 6.2 Rungs (per target marker M)
Computed on tissue pixels or cells only, never on background.

| Rung | Question | Computation |
|---|---|---|
| R1 Patient | Right patients high or low? | r of pixel-weighted patient means |
| R2 Image | Right images high or low? | r of per-image means |
| R3 Compartment | Right coarse regions? | Within-image r after Gaussian smoothing at scales s ∈ {16, 32, 64} px, image mean removed |
| R4 Cell | Right cells? | r of per-cell mean intensity, image mean removed; AUROC for M-positive cells (threshold set on the training cohort) |
| R5 Within cell state | Knows M beyond the cell's other markers? | Residual of true M = true M − mean true M of its k nearest neighbors in **non-M marker space**; correlate predicted M with that residual |

- **R5 without discrete cell types:** "cell state" is defined by the k nearest neighbors in non-M marker space. This avoids clustering bias and the circularity of defining cell types with the marker being tested.
- **Also report:** a scale-resolved correlation curve (Fourier ring correlation, or block sizes 1/4/16/64 px). It is a principled replacement for FID.

### 6.3 Baselines and nulls (each rung is reported relative to these)
- **Abundance oracle:** every pixel set to the true image mean. This is the ceiling for R1/R2 from abundance alone.
- **Per-pixel model with no spatial context:** linear model or gradient boosting on the input channels at the same pixel.
- **Nuclear-density-only model:** DNA1/DNA2/H3 → M. Also report partial r at R3/R4 controlling for DNA density.
- **Cyclic-shift null:** the prediction shifted cyclically within each image (keeps its autocorrelation). This gives a null distribution for R3–R5.

**A marker "reaches" a rung** if it beats both the null and the best baseline at that rung (patient-bootstrap 95% CI excludes zero), and this holds across the sensitivity checks in 6.5. Its **trust level** is the finest rung it reaches.

### 6.4 Positive controls (validating the ladder itself)
Degraded versions of the true marker whose correct rung is known:

| Control | Must reach | Must fail |
|---|---|---|
| Every pixel = image mean | R1–R2 | R3+ |
| True map, Gaussian-smoothed at scale s | up to R3 | R4–R5 |
| Each cell = mean of its k nearest neighbors in non-M marker space | up to R4 | R5 |
| True values shuffled among cells with similar non-M marker profiles | up to R4 | R5 |
| True map (identity) | R5 | — |

The ladder is valid only if every control lands on its expected rung.

### 6.5 Sensitivity checks
- Nucleus expansion 0–4 px, and nucleus-only vs. whole-cell masks. A score that rises as masks grow indicates spillover from neighboring cells.
- Cellpose vs. Mesmer.
- k ∈ {10, 30, 100}.
- Simulated segmentation errors (erode, dilate, merge neighboring cells). The positive controls must stay on their rungs.

### 6.6 Test of H1
- Rank models, preprocessing arms and seeds by pooled r and by each rung.
- Report Spearman agreement between the rankings.
- Report cases where the preprocessing choice changes pooled r but not R4/R5, like the CD3 low-pass example.

---

## 7. D3 — Panel substitution: which biology survives imputation

**Protocol:** mask one target marker M at a time → impute it → rerun each analysis with imputed M and with measured M → measure how much the result changes. All thresholds are set on training folds (or on Jackson for S3) and never refit on test data.

| Analysis | Level | Readout of agreement (imputed vs measured) |
|---|---|---|
| ER / HER2 status from patient-mean ERα / HER2 | Patient | ΔAUROC against clinical status (threshold-free) |
| Survival stratification (patients split at the training-set median of M) | Patient | Agreement of risk groups (Cohen's κ); ΔC-index |
| Cell phenotyping (clustering with M included) | Cell | Adjusted Rand index against phenotypes from measured M |
| M-positive cell fraction within epithelium (panCK-high) | Compartment/cell | Absolute error against a pre-set tolerance |
| Neighborhood enrichment (e.g. M+ cells near immune cells) | Spatial | Correlation of per-patient enrichment scores |

**Test of H2:**
- Across markers × analyses, test whether each marker's trust level (section 6) predicts how much the analysis result changes.
- Compare rank correlation or logistic fits: trust level vs. pooled r as the predictor.
- **Signal-quality confound:** report a signal-quality score per marker (fraction of positive pixels, SNR) and control for it. The goal is to separate "biologically non-redundant" from "just noisy".

**Framing:** ER/HER2 status and survival are **biological readouts of imputation quality**, not clinical claims. In the atlas, imputed values are always flagged with their trust level and never stored as if measured.

---

## 8. Test of H3 (cross-cohort cost)
- For each marker and rung: S1 vs. S3 performance, and S3 level A vs. level B (section 4).
- Report which rungs lose most, and how much of the loss input-only calibration recovers.
- The antibody clone check from section 3 explains outliers.

---

## 9. Deliverables
1. **Paper** (D1 + D3).
2. **`ladder` evaluation module** inside SpatialProteomicsNet: rungs, baselines, nulls, positive controls, as a small API.
3. **Marker trust table** for the CZI web dictionary: marker × cohort setting → trust level, plus the analyses it supports.
4. **Frozen folds, masks and checkpoints**, released.

## 10. Timeline (indicative)

| Weeks | Task |
|---|---|
| 1–2 | Antibody clone check; freeze folds; DNA-only segmentation + comparison to published masks |
| 3–4 | Implement the ladder + positive controls on **true** data only (no models). Validate the controls. |
| 5–8 | Train S1/S2/S3 × 2 arms × 3 seeds; baselines and nulls |
| 9–10 | Rung readouts, sensitivity checks, H1 test |
| 11–13 | D3 downstream analyses, H2 test; S3 level A vs. B (H3) |
| 14–16 | Writing; package release; trust table |

**Checkpoint for the PI after week 4:** if the positive controls do not land on their rungs, fix the ladder before training anything.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Positive controls misassigned → metric invalid | Validate on true data first (weeks 3–4); adjust rung definitions before any model results exist |
| Low-signal markers fail every rung, so the result is trivial | Report signal quality; the main claims use high/mid-tier markers; low tier serves as negative controls |
| Segmentation bias at R4/R5 | Masks from DNA only; cell state from non-M markers; sensitivity checks (6.5) |
| Spillover from neighboring cells inflates R4/R5 | Expansion sweep; flag scores that depend on mask size |
| Antibody clone mismatch in S3 | Check panels up front; report it as a limit on the ground truth |
| Seed variance swamps arm differences (seen in pilot: 10-epoch run beat 50-epoch run) | ≥3 seeds; patient-bootstrap CIs; no claims without non-overlapping CIs |
| Reviewers see D1 as "just a metric" | D3 shows the metric predicts which downstream biology survives, and pooled r does not |
| Tuning on test data | Freeze everything listed in section 3 before training; S3 thresholds come from Jackson only |

## 12. Open decisions for the PI
1. Whether an unpublished cohort is available (it would reopen the pretraining comparison, D2).
2. Whether to include one foundation model in S1/S2 as an example application (section 5).
3. Compute budget: 3 settings × 2 arms × 3 seeds × (joint model) ≈ 18 training runs.
