# Future Plan: When Can a Virtually Stained Marker Be Trusted?

**Working title:** *Which biology survives virtual staining? A validated resolution ladder for imputed IMC markers*
**Scope:** Combined paper of Direction 1 (validated evaluation metric) + Direction 3 (panel substitution / downstream biology).
**Target venue:** **MLCB 2027, full paper with PMLR proceedings** (primary). Afterwards, an extended journal version (*Cell Systems* / *Bioinformatics*) carrying the P2 items.
**Compute:** Kaggle only (T4x2, ~30 GPU-h/week, 12 h sessions; the notebooks resume from the checkpoint cache).
**Grant fit (CZI, "Towards a unified atlas of spatial proteomics"):** Aim 1 (SpatialProteomicsNet as data backbone), Aim 2 (breast stain-to-stain translation), Aim 4 (the marker map in this paper is the first version of the per-marker trust table for the web dictionary).

---

## 0. Publication strategy and priorities

### 0.00 Pivot (2026-10-20): does an image model know a marker beyond the cell table? (supersedes 0.0–0.3 where they conflict)
**Why.** The metric-critique headline has thin novelty: Sengupta 2025, 7-UP, Sun/Li/Murphy 2024 and MAXIM already cover it. The unanswered question is whether a virtual-staining image model carries information about marker M that the measured cell table (own and neighbour markers) does not. Plan file: `~/.claude/plans/go-over-this-project-scalable-bachman.md`.

**Estimator (replaces the kill-test R5).**
- **Statistic:** R5 = out-of-fold ΔR² of LightGBM(C + prediction) vs LightGBM(C).
- **Patient split, three ways:**
  - **S (20%):** fits every control that is a function of C, and the state deciles;
  - **A / B (40% / 40%):** carry the cross-fit.
  - Cross-fitted controls switch functions between fit and scored rows, which gave a spurious R5 ≈ −0.011 on synthetic data.
- **Rungs:**
  - **R5_own:** within image.
  - **R5_cell:** within image × state decile.
  - **R5_table:** R5_cell plus ring and contact neighbour means. This is the headline rung.

**Phase 0 (`Scripts/R5Validate.ipynb`, kernel `r5-validate`).** True data only. It runs the controls, the planted signal, a split-half noise ceiling (checkerboard and random pixel splits), the eroded-mask spillover check and mask hashes.
- **Gate G-R5** (all must hold):
  - every negative control |R5| < 0.01;
  - identity ≥ 0.9 × the residual share and ≥ the ceiling;
  - for main markers (ceiling ≥ 0.05), the planted curve is monotone and α = 0.1 is detected on R5_cell and R5_table.
- **No Phase 1 code is written before G-R5 passes.**
- **Status (run 2026-10-21, `Results/r5-validate/summary.md`): G-R5 FAILED.**
  - Passed: masks, power, noise ceiling, spillover.
  - Failed: the shuffled and shifted controls leak image- and state-level brightness (up to +0.20).
  - **Next:** apply the five fixes listed in that summary and rerun `r5-validate` on Kaggle:
    1. same subsample and seed for base and full models;
    2. cofactor 1 plus a 99.9th-percentile clip;
    3. leave-one-out group mean and SD as base features;
    4. candidate entered only as its deviation from the group mean;
    5. drop the "identity ≥ ceiling" criterion.

**Phase 1 decision (pre-registered 2026-10-20, before any image-model R5 is computed).**
- **Setting:** within cohort. ImmuVis-Danenberg is scored on Danenberg test patients, and ImmuVis-Jackson on Jackson test patients. Pixel outputs are aggregated with the Phase 0 masks, whose hashes must match.
- **Cell-table predictions** scored as candidates (B1/B2) must be fixed functions, never cross-fitted.
- **GO:** ≥ 2 markers with R5_table ≥ 0.05, a CI excluding 0, above that marker's minimum detectable R5 from the planted curve, and stable under 2-px erosion.
- **NO-GO** otherwise. Then train one full-panel (36-marker) seed and write the validated negative result.

### 0.0 Flagship analysis and kill-test outcome (read this first)
**Flagship: "same pooled r, different surviving biology"** (C3, with R5 as its key ingredient).
- Replace each measured marker with its imputed version.
- Rerun the standard analyses: patient status, phenotype classification, M+ fraction per cell type, neighbourhood enrichment.
- Show that **the rung a marker reaches predicts which analyses survive, and pooled r does not.** R5 (information beyond the rest of the panel) should be the rung that separates markers.
- Everything else in this plan exists to make this one figure credible.

**Why this is the flagship and not another claim:**
- **C1 (metric validity)** passes almost by construction, because the positive controls are designed to fool pooled r. It certifies the rungs; it cannot carry the paper.
- **C2 (marker map)** is descriptive.
- **C4 (selection) and C5 (cross-cohort)** are secondary.
- Only C3 answers the reviewer objection "just another metric". Only C3 changes what a biologist does with an imputed marker.

**Flagship figure:**
- x = pooled r; y = agreement of a downstream result (imputed vs measured); one point per (marker, model).
- Colour = trust level (finest rung reached).
- Statistic: leave-one-marker-out incremental prediction of downstream agreement by the rungs, over pooled r alone.

**Kill test: run 2026-10-09** (`Results/kill-test/summary.md`).
- The image models lose to a per-cell regression on the same channels for 10 of 11 markers (about half the R4).
- R5 failed its positive controls, so the automatic "GO" is invalid.
- The current reading is **PIVOT-A** or **KILL**; GK is not yet decided. The outcomes:
  - **GO:** a validated R5 > 0 for ≥ 2 markers, plus dissociation between pooled r and the cell-level readouts. Full paper as planned.
  - **PIVOT-A:** re-test R5 with the S1 main model at G0. If R5 is still ≈ 0, write a "virtual staining = panel redundancy" paper.
  - **PIVOT-B:** no dissociation, but R5 > 0. Weight moves to C2 (the marker map); decide with the PI.
  - **KILL:** drop the ladder paper; E1/E2 becomes a short methods note.
- **Open before GK:** pass G-R5 (section 0.00).

### 0.1 Venue
- **MLCB 2026 format** (verify against the 2027 call):
  - Deadline July 3–5 (AoE).
  - 8 pages, excluding references.
  - Single-blind.
  - PMLR is opt-in and includes an author response in mid-August.
  - Reviewers look for new ML methods that yield biological insight, or existing methods applied to new biological problems.
  - **MLCB 2027 is expected around early July 2027, about 39 weeks from 2026-09-28.**
- **What follows from that:**
  - Code and data can be public from the start; no anonymization is needed.
  - PMLR is archival, so the same paper cannot later go to NeurIPS or ICML. This trade-off is accepted. The follow-up is the journal extension.
  - 8 pages leaves room for at most 4 main figures and 1 table. Everything else goes to the appendix.
- **Main figures:**
  1. The ladder schematic and the pooled-r decomposition.
  2. Metric validity: the positive controls scored by the ladder and by standard metrics (C1).
  3. The marker map: finest rung per Danenberg marker, R5 (information beyond the panel), and signal quality (C2). C5 is a panel here if P1 is done.
  4. Downstream substitution: which analyses survive, and how well the rungs predict it compared with pooled r (C3).
  - Table 1: the model zoo ranked by pooled r and by each rung (C4).

### 0.2 What is new compared with prior work
- **Downstream evaluation of imputed markers already exists:**
  - 7-UP: cell typing and survival from imputed CODEX.
  - MIM-CyCIF.
  - Cell-level metrics in "Deep learning model imputes missing stains in multiplex images".
  - "On the utility of virtual staining for downstream applications".
  - VirTues and ImmuVis report pooled reconstruction metrics.
- **The novelty of this paper therefore rests on three things:**
  a. The exact decomposition of pooled r into abundance and structure, and the pilot result that preprocessing can inflate the abundance part.
  b. Validity of the metric itself: positive controls whose answer is known, nulls, and a direct comparison with the standard metrics (pooled r, SSIM, PSNR, FID, per-cell r).
  c. **R5: does the image model know M beyond what the cell's other markers already imply?** A model can score well per cell, and support cell typing, just by regressing M onto correlated markers. That is redundancy, not new information.
- **Open task (weeks 3–6):** a positioning memo covering 7-UP, MIM-CyCIF, VirTues, ImmuVis, EVA and the downstream-utility papers.

### 0.3 Run order (reprioritised 2026-10-30)

**Done since the last ranking (details in `project_recap.md`, items 35–39):**
- correlation vs W2 diagnostic;
- `prob-impute`: calibrated within a cohort, not across (cov₈₀ 0.57 / 0.44);
- `harmonise`: input correction not adopted; the shift sits in the target marker's level;
- `tabpfn-cells`: ties LightGBM; few-shot (k = 5 / 20 target patients) is the strong lever;
- LISI of virtual cells: transferred virtual cells keep cell types but don't mix with real cells (iLISI ≤ 0.09).

**The paper story these results support (MLCB 2027, early July):**
- **Working title:** *Virtual cells are not trustworthy across cohorts, and how to make them honest.*
- **Story, step by step:**
  1. Pixel metrics mislead.
  2. A cell table matches or beats image models.
  3. Across cohorts, virtual cells are confidently wrong, because of a level shift in each target marker. They keep cell types but don't mix with real cells.
  4. Input harmonisation doesn't fix it.
  5. A hierarchical Bayesian model with a cohort effect gives honest intervals for an unseen cohort, and a few measured patients fix the rest.
- **This frame holds under either R5 outcome.**
  - If G-R5 passes and R5 > 0, the ladder (C1–C3) is added as the "beyond the panel" part.
  - If not, this is PIVOT-A: the paper is the frame above.
- **Main figures:**
  1. The metric disagreement (`fig_cross_cohort_metrics`).
  2. iLISI vs cLISI plus calibration within vs across cohorts.
  3. Harmonisation oracle and the few-shot curve.
  4. Bayesian model: S2 / S3 and downstream biology.
  - Table 1: the per-marker trust table.

**How items are ranked:**
- **Must:** without it, a main claim is unsupported, or a reviewer will reject.
- **High value:** strengthens a main claim or answers an obvious reviewer question.
- **Nice to have:** appendix or journal version.

CPU and GPU kernels run in parallel on Kaggle; run the must items in the order given.

#### Tier 1: must for submission
| # | Run | Compute | Why it is a must | Gate / what it decides |
|---|---|---|---|---|
| M1 | **`r5-validate` v3: state-mean fix** (§18.5 item 1; a stronger base LightGBM with early stopping, or a stacked base) | CPU, ~7 h | Closes GK, which sets the framing: ladder + calibration, or PIVOT-A. Cheap, so do it now and stop carrying it | Passes → R5 Phase 1 (within cohort, ImmuVis vs table). Fails → R5 dropped for good; PIVOT-A |
| M2 | **`bayes-impute`** (§18.6.1, pre-registered): simulation-based calibration first, then S1–S4 | CPU, ~4–8 h | The ML-method contribution. Without it the paper is only negative results plus baselines | H1–H4 and the kill rule in §18.6.1. Comparators: quantile LightGBM (S2), TabPFN k = 5 (S3: cov₈₀ 0.62 / 0.64, CRPS 0.085 / 0.104) |
| M3 | **`fewshot-curve`**: k ∈ {0, 1, 2, 5, 10, 20} target patients × 5 draws of which patients × 3 split replicates; quantile LightGBM, TabPFN and (once M2 runs) Bayes. Also reps 1–2 of `prob-impute` and `harmonise` | 1 T4 + CPU, ~6–8 h | The few-shot result is the strongest positive finding, but it rests on one draw and one replicate. The curve answers the practical question "how many patients must be measured?" | Headline figure 3. If the k = 5 gain varies by > 0.05 cov₈₀ across draws, report the spread, not a point |
| M4 | **ImmuVis seed 2 (J→D) + variance export in both directions**; also inference on each cohort's own test patients | GPU, ~4.5 h | (a) The metric-disagreement and asymmetry claims rest on one ImmuVis run per direction. (b) The variance makes ImmuVis the image-model comparator in S2. (c) The own-cohort predictions fill in the LISI figure | If seed 2 doesn't reproduce the per-image pixel r flip (LightGBM ≤ 2 wins / ≥ 6 losses), downgrade the metric-disagreement claim to "model-dependent" |
| M5 | **Downstream biology on virtual cells** (C3-lite on cell models; §18.6 item 3; §18.4 item 3): measured vs virtual (point, sampled, few-shot, Bayes) for phenotype composition, immune–tumour contact depletion, HER2 status AUROC and patient-level marker means | CPU, ~2–3 h | "Trustworthy" has to mean the same biological conclusion, not only CRPS. This is the reframed flagship | Which analyses survive transfer, and whether calibration or few-shot rescues them |
| M6 | **Complete the LISI / distribution figure**: save per-cell predictions for the within-cohort MLP, GraphSAGE and GCN; add samples from quantile LightGBM, TabPFN, Bayes and ImmuVis (M4); add a classifier two-sample AUROC next to iLISI | CPU, ~1–2 h, after M2 and M4 | The figure currently has only LightGBM within a cohort, and a reviewer will ask for the full set | None |
| M7 | **No compute:** positioning memo (§0.2, plus scDesign2/3 LISI, Lingshu-Cell, Astir, SHADE, Tonks 2026, TabPFN, cyCombine / CytoNorm); **per-marker trust table** (§18.4 item 5) from existing outputs | Local | Novelty framing for the calibration angle; Table 1 | – |

#### Tier 2: high value (run if time allows before the deadline; most are cheap)
| # | Run | Compute | Why |
|---|---|---|---|
| H1 | **Uncertainty as a trust signal** (§18.4 item 2): do predicted intervals flag the cells and markers that fail (AUROC for large errors), for Bayes, quantile LightGBM, TabPFN and ImmuVis? | CPU, < 1 h, after M2 / M4 | Turns calibration into a usable per-cell trust flag |
| H2 | **A third cohort for the cohort-effect variance**: Keren on the 7 shared markers (§18.4 item 4), plus a search for another public IMC breast cohort | CPU, ~2 h; search has no compute | With two cohorts, τ_b is set by the prior (§18.6.1 caveat). Reviewers will raise this first |
| H3 | **ImmuVis asymmetry** (§18.5 item 4): Danenberg subsampled to 397 images; per-cohort clip | GPU, 1–2 runs | Explains the one direction in which an image model wins. Can share a session with M4 |
| H4 | **Joint generator** (§18.6 item 4): a conditional flow or mixture density network over all held-out markers | GPU, ~3–4 h | A real "virtual cell" with correlation structure. Run only if M2 shows the per-marker model is calibrated |
| H5 | **Crosstalk audit** (§18.4 item 1; §18.5 item 6) | CPU, < 1 h | Needed only if the epigenetic results (§17) go into the paper |
| H6 | **Scale-resolved curve + exact patch W2** (§18.5 item 3) | Mostly CPU | Turns the three-level metric figure into a curve |

#### Tier 3: nice to have (appendix, journal version or reviewer requests)
- TabPFN cell typing (§18.8 task 2); TabPFN 2.5 (gated weights, 50k context).
- Tokenisation ablation, then KRONOS2 adaptation (§18.5 items 2, 7). Cell typing is not a paper claim, and the embedding gain is +0.01.
- Label propagation for cell typing (§16).
- KID/FID on cell crops with a KRONOS encoder (§18.6 item 5).
- H3K27me3 survival test (§17 item 3) and §15 with the image model.
- Other baselines: Nimbus, CytoBERT / GPCT (§18.3).
- VirTues, on reviewer request only.
- Phase 2–4 below (GPU main models, zoo, PatchGAN, diffusion), **only if M1 passes and GK = GO**.

**Dropped or superseded:**
- Targeted CD3 / CD45 quantile matching (§18.5 item 5): superseded by `harmonise`, where per-cluster quantile matching didn't help.
- Input harmonisation as the S2 preprocessing (§18.7): not adopted.
- CyTOF / C2S rank work: a separate paper, parked.

**No compute, ongoing:** look for an IMC cohort with H3K27me3 measured (§17 item 1).

**KRONOS weights on Kaggle.** The model is gated on Hugging Face (`HF_TOKEN` is in `.env`). Don't put the token in notebook source. Download the weights locally once and upload them as a private Kaggle dataset (`yuvalsamoilov/kronos-weights`; academic-use licence, so keep it private). Alternatively, use a Kaggle notebook secret, which can only be set in the Kaggle UI.

**Phase 2: main models (weeks 7–19). Starts only after GK = GO (Tier 3 unless M1 passes).**

4. **G0:** smoke-test the main model on S1 and time one run.
5. **S1 main model** (gamma arm, 3 seeds) and **S2 Jackson main model** (3 seeds).
   - Interleave the runs (S1 seed 1, S2 seed 1, then the rest), so section 15 gate 2 is known early.
   - **Marker-dropout schedule:**
     - S1: often drop every channel Jackson lacks;
     - S2: often drop every channel Danenberg lacks.

     This lets the same weights serve S1-shared, S3 and section 15 without dedicated runs. Train dedicated S1-shared/S3 models only if the dropout version is clearly worse at G0.
   - Weight the targets toward the markers section 14 found context-dependent.
6. **`ladder` package, full positive controls and metric meta-evaluation** (section 6.4). Runs on CPU while the GPU queue runs. Includes the VirTues domain FID (inference). Gate G1.
7. **GPU zoo members, staged.**
    - Run the low-pass arm and the DNA/H3-only U-Net with 1 seed each first.
    - Add seeds 2–3 only if one lands within the main model's CI on any rung.
    - The panel-redundancy regressor and the per-pixel GBM are CPU and already exist from Phase 1.

**Phase 3: results (weeks 18–27).**
8. **Marker map (C2) and the P0 sensitivity checks.** The expansion sweep matters most, because spillover can fake R5 and the interaction terms. Gate G2.
9. **Flagship C3 on the full zoo** (section 7).
10. **Section 15 with the image model.** Same tests as 15-lite (done in `cell-table-map`). Report what the image model adds beyond the cell-table virtual markers.

**Phase 4: P1 tail (weeks 28–31). If late, cut from the bottom.**
11. C5 cross-cohort: S1-shared vs S3 from the dropout models, plus the antibody clone check (sections 3, 8).
12. VirTues audit as zoo member 8 (inference only).
13. C4 rankings and the model-selection case study (section 6.6).
14. PatchGAN member, the cheapest adversarial ("generative") member.
15. Scale-resolved correlation curve (section 6.2).

**P2: journal extension or appendix.**
- Level B calibration.
- Survival stratification.
- Simulated segmentation errors.
- Cellpose vs Mesmer.
- 5-fold CV.
- Diffusion model.
- Web trust dictionary.
- (The MIBI check was done for H3K27me3 on Keren 2018; see section 17.)

**Out of this paper.**
- The CyTOF/C2S side project (section 13) becomes a separate paper. It is parked: the rank analyses are done, and the PDAC IMC replication waits on Zenodo access.

### 0.4 Gates
| Gate | When | Pass condition | If it fails |
|---|---|---|---|
| **GK** | End of week 6 | The kill test (0.0) returns GO. It is read together with section 14: at least 2 markers show a context gain (ΔR² CI > 0), or the image model beats the cell-table baselines at R4/R5 | PIVOT-A, PIVOT-B or KILL, as defined in 0.0. If no marker shows a context gain and the image model never beats the cell table, take PIVOT-A ("virtual staining = panel redundancy"). Section 15 then continues on the cell table. No zoo training starts before GK. |
| **G0** | End of week 9 | The main model trains stably on Danenberg, and the wall-clock time per run is measured | Fall back to the pilot joint ResNet-UNet with an explicit target-marker input. Resize the zoo to the Kaggle quota (0.5). |
| **G1** | End of week 12 | Every positive control lands on its expected rung | Fix the ladder before any model results exist |
| | | At least one standard metric fails to separate the controls | If every standard metric already separates every control, C1 shrinks to a validity appendix; the flagship is unaffected |
| **G2** | End of week 22 | Trust levels are stable across seeds and differ across markers | If trust levels are identical across markers, the flagship cannot work. Report the negative result and consult the PI. |

### 0.5 GPU budget (Kaggle only)
| Tier | Runs | Count |
|---|---|---|
| Phase 2 | S1 main model (gamma) × 3; S2 Jackson main model × 3; low-pass arm × 1 (+2 if needed); DNA/H3-only U-Net × 1 (+2 if needed) | 8 (up to 12) |
| Phase 4 | PatchGAN × 3; dedicated S1-shared / S3 models only if the dropout versions fail at G0 | 3 (up to 9) |
| CPU | Per-pixel GBM, panel-redundancy regressor, all ladder / D3 analyses (Cellpose: brief GPU use or local CPU) | — |

- **Rule:** at G0, time one full run.
- If a run takes more than about 10 h, drop Phase 4 GPU items before cutting any Phase 2 seeds.

---

## 1. Motivation

**Known.**
- **How models are ranked today.** Virtual staining models (ImmuVis, VirTues, EVA, our joint ResNet-UNet) are ranked by pooled pixel correlation. FID is also used, but it relies on ImageNet features and has no clear meaning for IMC.
- **Downstream checks exist but rest on unvalidated metrics.** Downstream checks of imputed markers exist (7-UP, MIM-CyCIF). But the metrics behind them have not been validated. They also do not separate what a model adds beyond the rest of the panel.
- **Pooled correlation mixes two components exactly.** With η = the fraction of a channel's variance that lies between images:
  `r_pooled = √(η_true·η_pred)·r_between + √((1−η_true)(1−η_pred))·r_within`
- **Our pilot** (Jackson → Danenberg, 11 markers, 3 preprocessing arms). It used the joint ResNet-UNet *without* HyperConv (`nopca_augFalse`), with one seed per arm. It shows:
  - **Predictions recover real within-image structure.** ERα: between-image r ≈ 0.82, within-image r ≈ 0.52.
  - **Markers fall into tiers that are stable across arms:**
    - High: ERα, panCK.
    - Mid: HER2, CD45, CK8/18, SMA.
    - Low: CD20, Ki-67, CD3, CK5, CD68.
  - **Low-pass preprocessing inflates between-image r for weak markers** without improving within-image r. CD3: between 0.31 → 0.60, within 0.09 → 0.11.

**Unknown.**
1. At what spatial level is an imputed marker actually correct: patient, image, tissue compartment, cell, or within a cell state?
2. Within-image r mixes "where cells are" (almost free from the DNA channels) with marker-specific signal. How much of it is marker-specific?
3. How much of a model's cell-level accuracy is only redundancy with the rest of the panel, and how much is new information from tissue context?
4. Which downstream analyses stay valid when a measured marker is replaced by an imputed one?

## 2. Claims

The old hypotheses map as follows: H1 → C4, H2 → C3, H3 → C5. **C3 is the flagship** (section 0.0); the others support it.

- **C3 (downstream validity, THE FLAGSHIP).** A marker's rung profile predicts which downstream analyses survive substitution, and predicts it better than pooled r, SSIM or FID. R5 is expected to be the rung that separates markers with similar pooled r.
- **C1 (metric validity).** Standard metrics give indistinguishable scores to degradations that destroy different biology. These metrics are pooled r, SSIM, PSNR, FID and per-cell r. The ladder separates those degradations.
- **C2 (marker map, the biological headline).** For every Danenberg marker, the map gives two things:
  - the finest rung at which the marker can be imputed;
  - whether it carries information beyond the rest of the panel (R5).

  This sorts markers into three groups:
  - redundant (candidates to drop from a panel);
  - imputable only with spatial context;
  - not imputable.

  The map controls for signal quality.
- **C4 (selection, P1).** The choice of metric changes which model, preprocessing arm or checkpoint gets selected. Pooled r rewards abundance and coarse compartment accuracy.
- **C5 (cross-cohort cost, P1).** Moving from within-cohort to cross-cohort imputation costs most at the fine rungs. The comparison holds panel size fixed.
- **C6 (virtual tumour state, P1; section 15).** Jackson-only tumour-state markers, imputed into Danenberg from the shared channels, have two properties:
  - they reproduce Danenberg's patient-level labels (PR status, PAM50);
  - they reproduce the survival associations measured in Jackson.

  Their prognostic value is tested beyond both clinical covariates and the input channels themselves.

Any outcome is reported. For example, if all rungs degrade equally, that is a result, not a failure.

---

## 3. Data and evaluation settings

| Setting | Priority | Train | Test | Purpose |
|---|---|---|---|---|
| **S1 (main)** | P0 | Danenberg 2022 (METABRIC IMC, ~718 patients / 794 images), all channels | Held-out Danenberg patients | Within-cohort reference. Every marker is both input and target. |
| **S1-shared** | P1 | Danenberg, only the 14 channels shared with Jackson | Same S1 test patients | Same panel as S3, so S1-shared vs S3 isolates the cohort shift from the panel size |
| **S3 (harmonization)** | P1 | Jackson 2020 | S1 test patients (full cohort in the appendix) | Cost of crossing cohorts, which is what the atlas needs |
| **S2 (Jackson within-cohort)** | P1 | Jackson 2020, all channels | Held-out Jackson patients (patient-level split from the `jackson-cells-prep` patient IDs; the old Jackson split was random at the image level) | The only ground truth for the tumour-state and epigenetic markers. Also the gate for section 15. The rungs replicate in a second cohort. |

- **Split.** One frozen patient-level split, train/val/test ≈ 60/20/20.
  - It is stratified by ER status, HER2 status and PAM50, so every D3 group has enough test patients.
  - The split is `danenberg_patient_split.csv`, written by the Kaggle kernel `danenberg-cells-prep` (`DanenbergCellsPrep.ipynb`). A copy lives in `splits/`.
  - That kernel also builds, from Zenodo 5850952 and cBioPortal `brca_metabric`, what the Kaggle image dataset lacks:
    - the cell table;
    - the contact graph;
    - the relabelled cell masks;
    - the clinical data.
  - Each GPU model gets 3 seeds, with patient-bootstrap CIs.
  - 5-fold CV moves to P2, for compute reasons.
- **Why Danenberg is the main cohort:**
  - It has the most patients.
  - It has METABRIC clinical data (ER/HER2, PAM50, long survival follow-up).
  - It has published cell masks and single-cell tables (Zenodo 5850952; prepared on Kaggle, see Split).
- **Target markers.**
  - In S1, every Danenberg protein channel is a target: 36 channels after removing H3, DNA1 and DNA2, including the two HER2 clones (`IMC data/Danenberg2022/panel.parquet`).
  - The main claims use markers above a signal-quality threshold, and that threshold is fixed before training.
  - Low-signal markers are reported as negative controls.
  - In S1-shared and S3, the targets are the 11 shared markers.
- **Ground-truth caveat (P1, with S3).**
  - Check the antibody clone and metal tag for each shared marker in both panels. The panel parquets have clone and metal columns. Danenberg has two HER2 clones.
  - A clone mismatch caps S3 performance and must be reported.
- **Out of scope for this paper:** MIBI-TOF (Keren 2018, ~41 patients). It is too small for patient-level endpoints. A cross-technology check is P2.
- **Dropped:** a foundation-model pretraining comparison (the old Direction 2).
  - Hidden overlap with the IMC17M / SPM-47M pretraining sets cannot be ruled out without unpublished data.
  - Revisit only if a collaborator provides an unpublished cohort.

**Frozen before any training:**
- the S1 patient split;
- the segmentation masks;
- the signal-quality threshold for main-claim markers;
- every threshold used in section 7.

---

## 4. Preprocessing

**Rules:**
1. **The target marker is never normalized per image.**
   - Use arcsinh with a fixed cofactor, scaled with **training-cohort** statistics.
   - Per-image min-max erases the abundance rung.
   - The current code already does this: arcsinh, then global training-set 1st–99th percentiles.
2. **Inputs are scaled with training-cohort statistics.** The gamma correction is allowed. It uses the DNA and H3 channels within each image as an anchor, and those channels are always measured and never imputed.
3. **Both arms are trained as zoo members** (section 5): gamma, and ImmuVis-style with global rather than per-image scaling.
   - The arm that wins on the validation split is called the main model.
   - The other arm stays in the zoo, because the pilot showed that low-pass filtering inflates the abundance part of r. That makes it a test case for C4.

**Access to the test cohort in S3:**

| Level | Uses from Danenberg | Status |
|---|---|---|
| A | Nothing | **Main result (P1)** |
| B | Per-channel quantile matching of input markers to Jackson (unlabeled) | P2; labeled as using the test inputs |
| C | Any statistics of the target marker | **Forbidden** (leaks the answer) |

The A-vs-B gap (P2) measures how much of the cross-cohort cost is intensity calibration and how much is deeper (antibody clone, staining).

## 5. Models

**Correction to earlier notes.**
- The pilot and E1/E2 used the joint ResNet-UNet *without* HyperConv (`use_hyperconv=False`, tag `nopca_augFalse` in `AbundanceVsStructure.ipynb`). `session_summary.md` and `Results/J2D/HyperConv/` say otherwise.
- That model is marker-agnostic: it sees K sorted channels and is not told which marker it is predicting.
- HyperConv has only a 10-epoch POC.

**Main model (P0).**
- The target-conditioned HyperConv rewrite in `Scripts/hyperconv - full.ipynb`. It has a marker-conditional decoder and is trained with marker dropout.
- Each setting trains one model that can predict any masked marker.
- G0 checks that it trains stably.
- Fallback: the pilot joint ResNet-UNet with an explicit target-marker input.

**Model zoo (S1).** The zoo spans the behaviours the ladder should tell apart.

| # | Member | Compute | What it tests | Priority |
|---|---|---|---|---|
| 1 | Per-pixel gradient boosting on the input channels at the same pixel | CPU | No spatial context | P0 |
| 2 | DNA/H3-only small U-Net | GPU × 3 seeds | Nuclear density only | P0 |
| 3 | Panel-redundancy regressor: gradient boosting that predicts each cell's M from that cell's other markers (training patients only). Per-cell predictions are painted into the masks for pixel metrics. | CPU | Redundancy. Should reach R4 with R5 ≈ 0. | P0 |
| 4 | Main model, gamma arm | GPU × 3 seeds | — | P0 |
| 5 | Main model, ImmuVis-style arm | GPU × 3 seeds | Preprocessing; low-pass inflates abundance | P0 |
| 6 | Checkpoints every N epochs from 4–5 | Free | Model selection (C4) | P0 |
| 7 | Main model + PatchGAN loss | GPU × 3 seeds | Sharp outputs that may hallucinate texture; stress test of FID vs the ladder | P1 |
| 8 | VirTues (public weights) | Inference | See the note below | P1 (P0 if an unseen cohort appears) |
| 9 | Pilot `nopca` joint ResNet-UNet (existing checkpoints) | Existing | Continuity with the pilot in S3 | P1 |
| 10 | Diffusion model | GPU | — | P2 |

- **VirTues audit (member 8).** Its pretraining includes Danenberg, Jackson Basel and Jackson Zurich (confirmed 2026-10-08 from the dataset table in Uchal 2026). The weights are public at `huggingface.co/bunnelab/virtues`. It is used only to show what the ladder reveals about it, never as a generalization result. ImmuVis's IMC17M corpus includes the same cohorts; I found no public ImmuVis checkpoint in its paper.
- **Seeds.** Every GPU member gets 3 seeds. All comparisons use patient-bootstrap CIs across seeds.

---

## 6. D1 — The resolution ladder

### 6.1 Segmentation (independent of the target marker)
- Nuclei come from **DNA1, DNA2 and H3 only**, using a fixed off-the-shelf model (Cellpose), then a fixed pixel expansion.
  - No membrane channels are used, because panCK and CD45 are targets.
- One frozen mask set per cohort, shared by all models and markers.
- Cross-check against the published Danenberg masks (`ai4bmr_datasets`).
- Mesmer as a second segmenter is P2.

### 6.2 Rungs (per target marker M)
Computed on tissue pixels or cells only, never on background.

| Rung | Question | Computation |
|---|---|---|
| R1 Patient | Right patients high or low? | r of pixel-weighted patient means |
| R2 Image | Right images high or low? | r of per-image means |
| R3 Compartment | Right coarse regions? | Within-image r after Gaussian smoothing at scales s ∈ {16, 32, 64} px, image mean removed |
| R4 Cell | Right cells? | r of per-cell mean intensity, image mean removed; AUROC for M-positive cells (threshold set on the training data) |
| R5 Within cell state | Knows M beyond the cell's other markers? | Residual of true M = true M − mean true M of its k nearest neighbors in **non-M marker space**; correlate predicted M with that residual |

- **R5 without discrete cell types.** "Cell state" is defined by the k nearest neighbors in non-M marker space. This avoids clustering bias, and it avoids the circularity of defining cell types with the marker being tested.
- **Two versions of R5:**
  - **`R5_input`:** the non-M space is the model's own input markers. Does the image model know M beyond what the cell's measured inputs already imply? This version is used for the kill test and the flagship.
  - **`R5_panel`:** the non-M space is every other marker in the full panel. This is the panel-design question: is M redundant with the rest of the panel? It is reported in the C2 map.
- **Why R5 matters.**
  - The panel-redundancy regressor (zoo member 3) should score R5 ≈ 0 by construction.
  - An image model scores R5 > 0 only if tissue context tells it about M beyond the cell's own profile.
- **P1: a scale-resolved correlation curve** (Fourier ring correlation, or block sizes 1/4/16/64 px). It is a principled replacement for FID.
- **Reuse.** The between/within decomposition, patient bootstrap and signal-quality code from E1/E2 (`AbundanceVsStructure.ipynb`) become R1/R2 and the covariates.

### 6.3 Baselines and nulls (each rung is reported relative to these)
- **Abundance oracle:** every pixel is set to the true image mean. This is the ceiling for R1/R2 from abundance alone.
- **Zoo members 1–3** (section 5): no spatial context; nuclear density only; panel redundancy only. For the DNA-only model, also report partial r at R3/R4, controlling for DNA density.
- **Cyclic-shift null:** the prediction is shifted cyclically within each image, which keeps its autocorrelation. This gives a null distribution for R3–R5.

**When a marker "reaches" a rung:**
- it beats both the null and the best baseline at that rung (patient-bootstrap 95% CI excludes zero);
- this holds across the P0 sensitivity checks in 6.5.

Its **trust level** is the finest rung it reaches.

### 6.4 Positive controls and the metric meta-evaluation (P0, Fig. 2)
Degraded versions of the true marker whose correct rung is known:

| Control | Must reach | Must fail |
|---|---|---|
| Every pixel = image mean | R1–R2 | R3+ |
| True map, cyclically shifted within its image (realistic texture, wrong place) | R1–R2 | R3+ |
| True map, Gaussian-smoothed at scale s | up to R3 | R4–R5 |
| Each cell = mean of its k nearest neighbors in non-M marker space | up to R4 | R5 |
| True values shuffled among cells with similar non-M marker profiles | up to R4 | R5 |
| True map (identity) | R5 | — |

**Meta-evaluation.**
- Score every control with the ladder *and* with the standard metrics:
  - pooled r;
  - SSIM / MS-SSIM;
  - PSNR;
  - FID (existing clean-fid code);
  - VirTues embedding distance, a "domain FID" (P1; inference only, public weights). Expected to miss the shifted-map and kNN-mean controls, as FID does;
  - per-cell r;
  - cell-typing ARI.
- Report which controls each metric cannot tell apart.
  - Expected example: FID barely penalizes the cyclically shifted true map, because the image statistics are unchanged.
  - Pooled r, and even cell-typing ARI, cannot tell the kNN-mean control from the truth.
- This runs on true data only, with no training. It is the first result produced.

**Validity condition.** The ladder is valid only if every control lands on its expected rung (G1).

### 6.5 Sensitivity checks
- **P0:**
  - Nucleus expansion 0–4 px, and nucleus-only vs. whole-cell masks. A score that rises as masks grow indicates spillover from neighboring cells. Spillover can fake R5, so this check is required.
  - k ∈ {10, 30, 100}.
  - DNA-only vs. published masks.
- **P2:**
  - Cellpose vs. Mesmer.
  - Simulated segmentation errors (erode, dilate, merge neighboring cells). The positive controls must stay on their rungs.

### 6.6 Test of C4 (metric choice changes selection)
- Rank zoo members, preprocessing arms, seeds and checkpoints by pooled r, SSIM, FID and by each rung.
- Report Spearman/Kendall agreement between the rankings.
- **Model-selection case study.**
  - Which checkpoint or arm does each metric pick?
  - What does that choice do to one downstream result from section 7?
- Report cases where the preprocessing choice changes pooled r but not R4/R5, like the CD3 low-pass example.

---

## 7. D3 — Panel substitution: which biology survives imputation (THE FLAGSHIP, C3)

This section is the flagship (section 0.0). It is built on the **full S1 zoo** (weeks 22–27) and produces the paper's flagship figure: x = pooled r, y = downstream agreement, colour = trust level.

**Protocol:**
1. Mask one target marker M at a time.
2. Impute it.
3. Rerun each analysis twice: with imputed M and with measured M.
4. Measure how much the result changes.

All thresholds are set on the training split (or on Jackson for S3) and are never refit on test data.

| Analysis | Level | Readout of agreement (imputed vs measured) | Priority |
|---|---|---|---|
| ER / HER2 status from patient-mean ERα / HER2 (worked example) | Patient | ΔAUROC against clinical status (threshold-free) | P0 |
| Patient-level effect size of M across clinical groups (ER, HER2, PAM50) | Patient | Agreement of effect sizes; applies to every marker | P0 |
| Cell phenotyping: a classifier trained on measured features of train patients predicts the published `cellPhenotype`; M is swapped for imputed M at test | Cell | Macro-F1 drop (fixed classifier, no refitting) | P0 |
| Neighborhood enrichment (e.g. M+ cells near immune cells) | Spatial | Correlation of per-patient enrichment scores | P0 |
| M-positive cell fraction per cell type per patient (threshold from train patients) | Compartment/cell | Correlation and absolute error, measured vs imputed | P0 |
| Survival stratification (patients split at the training-set median of M) | Patient | Agreement of risk groups (Cohen's κ); ΔC-index | P2 |

**Test of C3:**
- **Data points:** markers × zoo members × analyses. That is hundreds of points, not the 30 a 6-marker design would give.
- **Predictors compared:**
  - the rung profile / trust level;
  - pooled r;
  - SSIM;
  - FID.
- **Main test:** leave-one-marker-out prediction of how much each analysis result changes.
  - The points are not independent, so use a mixed model or a bootstrap clustered by marker.
- **Resolution matching.** Report which rung best predicts which analysis. Expected: R1/R2 predicts the patient analyses, and R4/R5 predicts the cell and spatial analyses.
- **Signal-quality confound:**
  - Include a signal-quality score per marker (fraction of positive pixels, SNR; E2 code) as a covariate.
  - The goal is to separate "biologically non-redundant" from "just noisy".

**Framing:**
- ER/HER2 status is a **biological readout of imputation quality**, not a clinical claim.
- In the atlas, imputed values are always flagged with their trust level and never stored as if measured.

---

## 8. Test of C5 (cross-cohort cost, P1)
- For each shared marker and rung, compare S1-shared vs. S3 at level A. Both are evaluated on the same test patients, with the same panel.
- Report which rungs lose most.
- The antibody clone check from section 3 explains outliers.
- The level A vs. B comparison (how much input-only calibration recovers) is P2.

---

## 9. Deliverables
1. **MLCB paper** (D1 + D3; 8 pages + appendix; PMLR).
2. **`ladder` evaluation package**, public, inside SpatialProteomicsNet. It contains:
   - the rungs, baselines and nulls;
   - the positive controls;
   - the metric meta-evaluation;
   - a small API.
3. **Frozen split, masks and checkpoints**, released.
4. **Marker map** (the C2 figure). This is the first version of the trust table: marker × setting → trust level, plus the analyses it supports. The CZI web dictionary is built from it after submission.

## 10. Timeline (to MLCB 2027, starting 2026-09-28)

| Weeks | Dates (approx.) | Work | Gate |
|---|---|---|---|
| 3–6 | Oct 12 – Nov 8 | `r5-validate` rerun with fixes, then Phase 1 (section 0.00); positioning memo; pull E2 CSVs | **GK** |
| 7–9 | Nov 9 – Nov 29 | DNA-only masks, compared with the published masks; smoke-test the main model on Danenberg (S1) with the dropout schedule and time one run | **G0** |
| 8–12 | Nov 16 – Dec 20 | `ladder` package (reuses the kill-test R4/R5 code); full positive controls and metric meta-evaluation on **true** data only, with the VirTues domain FID | **G1** |
| 10–19 | Nov 30 – Feb 7 | GPU queue: S1 seed 1 → S2 seed 1 (section 15 gate 2) → remaining S1/S2 seeds → low-pass and DNA-only (1 seed each, more if needed) | — |
| 18–22 | Jan 25 – Feb 28 | Rung readouts; marker map (C2); P0 sensitivity checks | **G2** |
| 22–27 | Feb 22 – Apr 4 | **Flagship on the full zoo:** D3 analyses and the C3 test; section 15 with the image model | — |
| 28–31 | Apr 5 – May 2 | Phase 4 in order: C5 (S1-shared vs S3 from the dropout models, clone check), VirTues audit, C4 rankings, PatchGAN, scale-resolved curve | Hard stop May 2 |
| 32–39 | May 3 – ~Jul 1 | Writing (8 pages + appendix); code and data release; PI review 2 weeks before the deadline | Submit |
| — | mid-Aug 2027 | PMLR author response | — |
| — | after Nov 2027 | Journal extension with the P2 items | — |

**Checkpoints for the PI:**
- **Week 6 (GK):** the GK decision (section 0.0) decides whether any zoo training happens.
- **Week 12 (G1):** if the positive controls do not land on their rungs, fix the ladder before interpreting any model results.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Published masks used membrane channels (possible target leakage at R4/R5) | Used for the kill test only; the paper uses DNA-only masks (6.1) |
| Positive controls misassigned → metric invalid | Validate on true data first (weeks 8–12, G1); adjust rung definitions before any model results exist |
| Standard metrics already separate every control → C1 is weak | G1 detects this; C1 shrinks to a validity appendix, and the flagship (C3) is unaffected |
| Novelty challenged (7-UP, VirTues, ImmuVis already evaluate downstream) | Positioning memo in weeks 3–6; claim only the decomposition, metric validity and R5 (section 0.2) |
| Main model (HyperConv rewrite) unstable | G0; fall back to the pilot joint ResNet-UNet with an explicit target-marker input |
| Kaggle GPU quota too small | Time one run at G0; drop P1 GPU items before cutting P0 seeds; baselines run on CPU |
| Low-signal markers fail every rung, so the result is trivial | Signal-quality threshold fixed before training; main claims use markers above it; low tier serves as negative controls |
| Segmentation bias at R4/R5 | Masks from DNA only; cell state from non-M markers; sensitivity checks (6.5) |
| Spillover from neighboring cells inflates R4 or fakes R5 | Expansion sweep and nucleus-only masks (P0); flag scores that depend on mask size |
| Antibody clone mismatch in S3 | Check panels up front (the parquets have clone columns); report it as a limit on the ground truth |
| Seed variance swamps arm differences (seen in pilot: 10-epoch run beat 50-epoch run) | 3 seeds; patient-bootstrap CIs; no claims without non-overlapping CIs |
| Foundation-model pretraining overlap | VirTues used only as an audit, never as a generalization result |
| Reviewers see D1 as "just a metric" | C2 gives the biological map, and C3 shows the metric predicts which downstream biology survives while pooled r does not |
| PMLR proceedings block a later NeurIPS submission | Accepted trade-off; the follow-up is a journal extension |
| Tuning on test data | Freeze everything listed in section 3 before training; S3 thresholds come from Jackson only |

## 12. Open decisions for the PI
1. Is an unseen cohort available? It would reopen the pretraining comparison (D2) and make the VirTues audit P0. An **IMC cohort with H3K27me3 measured** is the one thing that would let the epigenetic result be validated (section 17).
2. Should the PatchGAN zoo member be included? It is P1 and costs 3 GPU runs.
3. Should a non-archival workshop version be presented before MLCB (e.g. an ICLR 2027 workshop, ~Feb 2027)? Check MLCB's policy on prior presentations first.
4. If extra compute (e.g. a university cluster) becomes available, it would restore 5-fold CV and S2.

## 13. Side project: link to the breast-cancer CyTOF / C2S project

Done and parked (`project_recap.md` Part 2; `Results/c2s-imc-summary.md`). The PDAC IMC replication waits on Zenodo access.

## 14. Cell-table information map and neighbour interactions

Done (`cell-table-map`, `cell-table-followup`; see `project_recap.md`). Two parts stay in use.

**Marker groups** (results are reported by group):

| Group | Danenberg (36) | Jackson (32) |
|---|---|---|
| Clinical | ER*, HER2* (clones 3B5 and D8F12) | ER* (Gd156 "Rabbit IgG H L"), PR, HER2* |
| Proliferation | Ki-67* | Ki-67*, pHH3 |
| Cell death | cleaved caspase-3 | cleaved PARP / caspase-3 (one channel) |
| Epigenetic | — | H3K27me3 |
| Tumour state (signalling, EMT, hypoxia, stemness) | — | pS6, p-mTOR, c-Myc, EGFR, p53, Twist, Slug, CD44, CA-IX |
| Antigen presentation | HLA-ABC, β2M, HLA-DR | — |
| Immune state (activation, checkpoint, differentiation) | PD-1, ICOS, OX40, GITR, CD38, CD57, CD45RA | — |
| Immune lineage | CD45*, CD3*, CD4, CD8a, CD20*, CD68*, CD163, CD11c, CD15, CD16, FOXP3 | CD45*, CD3*, CD20*, CD68* |
| Epithelial lineage | panCK*, CK8/18*, CK5* | panCK*, CK8/18*, CK5*, CK7, CK14, CK19, E/P-cadherin, GATA3 |
| Stroma and vessels | SMA*, FSP1, PDGFRB, podoplanin, caveolin-1, CD31/vWF, CXCL12 | SMA*, vimentin, fibronectin, vWF/CD31 |

\* = one of the 11 shared markers.
- Total histone H3, DNA1 and DNA2 are nuclear controls in both cohorts and are not targets.
- Across the two cohorts, only Ki-67 is a shared state marker. Cleaved caspase-3 is a partial match.

**Decision rule** (sets the image-model targets in sections 5 and 7):
- Markers with a context gain (ΔR² CI > 0) or a significant interaction term become the main targets for the image model and the flagship.
- High own-marker R² with no context gain → reported as panel-redundant.
- Low on both → "must measure".

## 15. Virtual tumour state: Jackson-only markers in Danenberg, tested against METABRIC survival (P1, C6)

**Why.**
- The two cohorts have complementary strengths:
  - Danenberg (METABRIC) has 718 patients with long survival follow-up but no tumour-state markers;
  - Jackson has the tumour-state and epigenetic markers but weaker survival data (265 Basel patients, 56 DSS events).
- Imputing Jackson's markers into Danenberg is the "virtual cell" use case: cells gain markers that were never measured, in a cohort where the clinical questions can be asked.

**Targets** (Jackson-only):
- tumour state: CA-IX, p53, c-Myc, pS6, p-mTOR, EGFR, Twist, Slug, CD44;
- epigenetic: H3K27me3, normalised per cell to H3;
- positive controls with Danenberg labels: PR, GATA3, CK14, CK7, CK19.

Only markers that pass both gates below are carried forward.

**Model.**
- The S2 main model is trained within Jackson with marker dropout. The dropout patterns must include the shared-only input set (11 shared markers + H3, DNA1, DNA2) often, so the same 3 seeds serve both S2 and this section with no extra GPU cost.
- Inputs at inference in Danenberg: the shared-only set, level A (no Danenberg statistics; section 4).

**15-lite (cell-table version): done** (`cell-table-map`).
- No virtual marker was prognostic.
- Only EGFR added to basal-like AUROC (+0.027).
- The image model is judged by what it adds beyond 15-lite.

**H3K27me3 status (2026-10-22): see section 17.**
- A Danenberg imputation already exists (LightGBM, `IMC data/Danenberg2022/epi-relations/`).
- The survival test (low H3K27me3 → worse) has not been run.
- Run it only with section 17's caveat: cross-cohort validity of the mark is unproven.

**Gates (thresholds fixed before running).**
1. **CPU:** passed for all 21 Jackson-only markers (`cell-table-map`).
2. **GPU:** the S2 model with shared-only inputs, on held-out Jackson patients:
   - it must beat the restricted cell-table model at R4 (per-cell r, image mean removed);
   - it must beat the cyclic-shift null.
- A marker that fails gate 2 can still be imputed from the cell table, but it is then labelled panel-redundant: no spatial model is needed for it.

**Cross-cohort sanity check (no ground truth for the targets).**
- Treat the 11 shared markers as pseudo-targets: hold each one out of the inputs, impute it into Danenberg, and score it against its measured value.
- This calibrates how much the Jackson → Danenberg move costs.
- It also tests whether a consistency check matches real accuracy. The check: does a virtual marker's correlation with held-out (non-input) markers match the true correlation in Jackson?
  - If the check also passes for poorly imputed markers, it is uninformative and is not used.
- Per cell type, compare the distribution of each virtual marker in Danenberg with the measured distribution in Jackson.

**Patient-level validation against Danenberg labels.**
- Patient features: the mean virtual marker in tumour cells (Danenberg published phenotypes), and the fraction of marker-high tumour cells (threshold set on Jackson training patients).
- **PR:**
  - AUROC for clinical PR status;
  - the key test is **within ER+ patients**, because ER is an input and PR follows ER.
- **Basal markers (CK14, EGFR):** AUROC for PAM50 basal-like.
- **Luminal markers (GATA3, PR):** AUROC for PAM50 luminal A vs B.
- Every AUROC is compared with a baseline built from the same input channels (patient means of the 11 shared markers). Only the gain over that baseline counts.

**Survival.**
- Model: Cox for DSS, with pre-registered covariates (age, grade, tumour size, nodal status, ER, HER2).
- Per marker, three tests, with BH correction across markers:
  1. **Direction:** does the hazard ratio match the measured marker's HR in Jackson Basel and the literature priors (verify each)?
     - high CA-IX → worse;
     - p53 accumulation → worse;
     - EGFR → worse in triple-negative disease;
     - low H3K27me3 → worse (Wei 2008, Holm 2012).
  2. **Beyond clinical:** ΔC-index over clinical covariates (patient-bootstrap CI).
  3. **Beyond the inputs:** ΔC-index over clinical covariates plus patient means of the input channels.
- A virtual marker is calculated from the input images, so it cannot carry information that is not in them.
  - Passing test 3 means the image model extracts prognostic structure that patient-mean inputs miss, for example spatial patterns.
  - Failing test 3 but passing 1–2 means the virtual marker is an interpretable re-expression of the inputs (e.g. a "hypoxia score"), not new information. That is still worth reporting.
- Prior from `rank-utility-clinical`: no cell-composition feature set added to clinical covariates for survival, so expect small gains.

**Framing.**
- Virtual values are always labelled as imputed, with their gate results. They are never treated as measurements.
- Agreement in Danenberg shows consistency with Jackson biology, not ground truth. Only re-staining METABRIC tissue would give that.

## 16. Cell typing: topology and label propagation (LiveJournal DOI-paper analogue)

**Background.**
- `cell-type-graph` (2026-10-16): LightGBM with neighbour-marker features is best within a cohort; SAGE+ is best across cohorts (J↔D).
- `cell-type-topology` (done, 3 replicates): adds 13 graph-position features per cell (contact graph and 30 µm graph).
  - Topology alone reaches coarse macro-F1 ≈ 0.25 (chance ≈ 0.15).
  - It adds ≤ 0.002 on top of Table + nbr in every setting.
- The DOI paper (Samoilov-Katz, Louzoun, Muchnik & Zaidel, arXiv 2106.13672) combined:
  - 20 topology features;
  - 4 **label-propagation** features per DOI: counts of first neighbours with and without that DOI, separately for in- and out-edges. They were counted only over neighbours with known (training) labels, from an 80/20 node split.
  - The two sets were concatenated as input to a 3-layer feed-forward network.
  - The paper reports no topology-only or propagation-only ablation.
- None of our arms uses neighbours' *labels*. Table + nbr propagates neighbours' *markers* only.

**Problem.** The paper is transductive: test nodes sit in the same graph as labelled nodes. We hold out whole images and patients, so test images have no labelled cells.

**A. Predicted-label propagation (main arm; works on unseen images and across cohorts).**
1. Stage 1: Table + nbr, with out-of-fold predictions (grouped by image) on the training images, so stage 2 never sees in-sample predictions.
2. Stage 2: add each cell's neighbours' predicted class probabilities:
   - mean and argmax counts per class on the contact graph and the 30 µm graph;
   - a 2-hop mean.
3. Optional: iterate 2–3 rounds (iterative classification), recomputing neighbour predictions from the stage-2 output.
4. Arms:
   - Propagation only;
   - Topology + propagation (the paper's combination);
   - Table + nbr + propagation;
   - Table + nbr + topology + propagation.
5. Same tasks (fine/coarse; D, J, J→D, D→J, Both), patient bootstrap, 3 replicates. Report per-class F1, especially endothelial.

**B. Partly labelled test images (the paper's setting; side analysis).**
- Reveal the true labels of a random 20% of cells in each test image (also try 5% and 50%). Score only the hidden cells.
- Add the paper's features: per class, the counts of revealed neighbours with and without that class. The graph is undirected, so there are 2 features per class instead of 4.
- Interpretation: how much does a pathologist annotating a few cells per image help?
- Not comparable with the other arms, because it uses test labels.

**Compute.** CPU kernel (LightGBM only), about 3 h. Run after `cell-type-topology` finishes, reusing its notebook (`Scripts/CellTypeTopology.ipynb`).

## 17. Epigenetic virtual cells (H3K27me3, pHH3)

**Done (2026-10-21/22; `project_recap.md` items 29–31).**
- `epi-immuvis-train` and `epi-immuvis-infer`.
- `epi-relations` v1–v3 (pre-registered), including post-hoc Amendment 1.
- Keren 2018: Kaggle dataset, `keren-cells-prep`, `keren-epi-within`.

**Results.**
- **Within Jackson:** H3K27me3 is imputable. LightGBM ≈ ImmuVis (partial r 0.60), so the image adds nothing. pHH3 fails.
- **Danenberg:** passes only after dropping the anchors that don't transfer (SMA, CD68, CD20).
- **Keren:** not usable as an external test. Even the measured mark keeps only 2/18 Jackson relations. It is weakly predictable even within Keren (0.23 from the shared inputs) and likely affected by crosstalk from the neighbouring mass (pS6, Yb173).

**Open (in order).**
1. **Find an IMC cohort with H3K27me3 measured** (same technology as Jackson). This is the only route to a real external test. Section 12, item 1.
2. **Optional, cheap (CPU):** confirm the Keren crosstalk. Regress H3K27me3 pixels on the pS6 and panCK pixels, and test whether the cell-level H3K27me3–pS6 link follows the fixed isotope ratio rather than biology. This decides whether Keren's H3K27me3 can be used after correction.
3. **Section 15 survival test with the imputed H3K27me3.** Only if item 1 or 2 rescues its validity; otherwise report it as exploratory.
4. **Rule fix for later tests:** the anchor-transfer ratio needs a floor on the within-cohort r. In Keren, SMA "transferred" with r ≈ 0.15 both ways.

## 18. Further analyses (ideas, 2026-10-22)

Ranked by information per unit of compute. None of these starts R5 Phase 1 before G-R5 passes (§0.00).

### 18.1 What drives cell typing: XGBoost + SHAP (CPU or one T4; ~2 h)
**Question.** Which inputs drive each cell type, and does that differ between cohorts? Candidate inputs: own markers, neighbour features, morphology/area, nuclear channels. A cohort difference would explain where transfer fails (e.g. endothelial cells, SMA).

**Design.**
- **Model:** XGBoost (`device="cuda"`) on the `cell-type-graph` features (Table + nbr), with coarse and fine labels, trained on D, J and Both.
- **TreeSHAP** via `pred_contribs` / `pred_interactions`:
  - mean |SHAP| per class and per feature group;
  - rank agreement of the importances between the D and J models;
  - SHAP on misclassified J→D cells: which features mislead the model;
  - own × neighbour interaction values, especially for endothelial cells.
- **Robustness:** correlated markers split SHAP credit. Report SHAP summed per marker group, and check it against grouped permutation importance.
- **Same tool for imputation:**
  - SMA in J vs D: why the sign flips;
  - H3K27me3 in Jackson: which markers carry it. Compare with Keren, where pS6 dominates.

**Decision value:** interpretation for the paper and the trust table. Cheap.

### 18.2 KRONOS as a clean zero-shot image encoder, then IMC adaptation (2×T4)
**Facts** (Shaban et al. 2025, arXiv 2506.03373; weights at github.com/mahmoodlab/KRONOS, academic use only):
- ViT-S/16, DINO-v2.
- Pretrained on SPM-47M: 8 fluorescence platforms (CODEX, MxIF, COMET, IBEX, CellDive). **No IMC or MIBI.**
- Inputs: a shared per-marker conv stem with fixed sinusoidal marker encodings, so it accepts any panel.
- Cell embeddings: from nucleus-centred 64×64 px patches at ~0.5 µm/px, with pixels outside the cell zeroed.
- **Why it matters here:** neither of our cohorts is in its pretraining. That makes it the only clean zero-shot foundation-model test. ImmuVis, VirTues and EVA all have overlap or scale issues.

**Steps.**
1. **Zero-shot.**
   - IMC is 1 µm/px: resample ×2 (or use 32×32 native patches; compare both).
   - Map our markers onto KRONOS's marker vocabulary.
   - Normalise intensities per marker as in their recipe.
   - Readouts:
     - linear and LightGBM probes for coarse/fine cell types (D, J, J→D), against Table and Table + nbr;
     - a marker-imputation probe: predict a held-out marker from the embedding of the other channels, against the cell table;
     - an H3K27me3 probe in Jackson, scored as ΔR² of LightGBM(table + embedding) over LightGBM(table). This is a cheap, generator-free version of the R5 question. Score it with the same leak controls once G-R5 passes.
2. **Domain adaptation.** Continue DINO-v2 self-supervised training on the Jackson + Danenberg train images (no labels). Compare frozen vs LoRA (r = 8–16 on attention) vs full fine-tuning. Rerun the step-1 probes.
3. **Supervised.** Add a cell-type head with LoRA vs full fine-tuning.
- **On QLoRA:** not needed. ViT-S has about 22 M parameters, so full fine-tuning in fp16 fits on one T4. 4-bit quantisation only adds error at this size; QLoRA is for multi-billion-parameter models.
  - Checked again for KRONOS2 (ViT-B, about 86 M parameters): full fine-tuning plus Adam is about 1.4 GB, still small. Use LoRA (r 8–16 on qkv/proj, plus the marker-embedding MLP) vs full fine-tuning.
  - Train in fp16 with a grad scaler (no bf16 on T4), and use DDP across the 2×T4, not DataParallel.
- **Step 1 result (2026-10-27; `Results/kronos2-zeroshot/summary.md`):**
  - Zero-shot embeddings alone are 0.04–0.07 below Table + nbr.
  - Adding 32 PCs gives +0.01, on Danenberg only.
  - KRONOS2 ≈ KRONOS overall. Both pass the gate; KRONOS2 meets (b) in every replicate.
- **KRONOS2 on DataParallel:** the replicas have no `parameters()`. Cache the marker embedding as a buffer, as the notebook does.
- **Risks:** mass-ion vs fluorescence intensity statistics; 1 vs 0.5 µm/px; markers missing from KRONOS's vocabulary.

### 18.3 Other foundation models to consider
| Model | Data | Relevance | Status |
|---|---|---|---|
| **ImmuVis** (Uchal 2026, arXiv 2602.04585) | IMC17M: 28 IMC cohorts, 265 markers, incl. ours | Already reimplemented and trained from scratch | No public weights found (2026-10-13) |
| **VirTues** (Wenckstern et al., *Nature* 2026; github.com/bunnelab/virtues) | >12k images, >30 cohorts, 4 technologies | Marker reconstruction, cell typing, niches, zero-shot across panels | **Not run (decided 2026-10-27).** The public weights saw both cohorts, and ImmuVis beats it in its own paper. If a reviewer asks: public-weights inference in both directions, about 1 GPU run each, scored with the existing pixel/patch/cell code and labelled "pretrained on both cohorts" |
| **EVA** (Liu 2026) | – | Done (zero-shot) | Cannot represent histone marks (one shared token, H3F3A) |
| **KRONOS / KRONOS2** (Mahmood lab) | Fluorescence only. KRONOS2: SPM-56M, 8 platforms, no IMC or MIBI, 268 markers, ViT-B/16 | Clean zero-shot test (18.2) | **Done:** zero-shot cell typing, 3 reps (`kronos2-probes`). Weights: private dataset `kronos2-weights` |
| **Nimbus** (Angelo lab; *Nature Methods*) | Pan-M, 197 M marker-positivity labels | Pretrained marker-positive/negative classifier; a baseline for "M+" readouts | Check platform coverage (IMC?) |
| **CytoBERT** (arXiv 2608.14414), **GPCT** (bioRxiv 2026) | Cytometry tables, no images | Cell-table foundation models: a stronger table baseline, and a link to the CyTOF/C2S project | Unread |

### 18.4 Cheap follow-ups that close open questions (CPU, existing outputs)
1. **Crosstalk audit, every cohort.**
   - For each target, test whether the ±1 Da and +16 Da (oxide) channels carry it.
   - Jackson H3K27me3 (La139) has no neighbour or oxide source, so it is clean. Jackson pHH3 (Eu153) sits next to CD3 (Sm152), and its oxide falls on EGFR (Tm169).
   - Apply spillover compensation (CATALYST-style matrix) to Jackson, which is uncorrected, and recheck the key relations.
   - For Keren, fit the H3K27me3 → pS6 leakage as a fixed isotope ratio (§17 item 2).
2. **Trust signal from uncertainty.**
   - ImmuVis's β-NLL gives a per-pixel variance. Does it rank the cells and markers where imputation fails (AUROC for detecting large errors)?
   - Compare with the LightGBM ensemble or quantile spread.
   - This feeds the per-marker trust table.
3. **HER2 as a "must-measure" marker.**
   - All J→D imputations lose HER2 status (AUROC 0.86 → 0.43–0.58).
   - Test HER2 status from ImmuVis-D on Jackson (cell-level HER2 r 0.65), and pseudo-bulk vs cell-level.
   - Is any model able to keep the magnitude signal?
4. **IMC → MIBI cost for the shared markers.** Train on Jackson / Danenberg / Both and test on Keren for the 7 shared markers (cell table). Compare with within-Keren cross-validation.
5. **Per-marker trust table** (the deliverable from §9). Columns:
   - within-cohort R4;
   - transfer ratio;
   - pooling cost;
   - crosstalk flag;
   - clinical retention (HER2 status, prognostic HR);
   - uncertainty calibration.

   Everything in it already exists or comes from items 1–4.

### 18.5 New ideas (2026-10-27, after the patch level and KRONOS2)
Ranked by information per unit of compute.

1. **R5 state-mean fix** (CPU, about 7 h). The base LightGBM probably underfits. Give it more capacity with early stopping on held-out patients, then rerun `r5-validate`.
   - Gate as before: the state-mean control must fall below 0.01.
   - If it doesn't, report R5 against a stacked base, LightGBM(C, f_S(C)).
2. **Tokenisation ablation for image encoders** (GPU, about 2 h per arm with the KRONOS2 notebook). We never compared tokenisations. Arms, all scored with the same cell-typing probes and 3 replicates:
   - crop 16 / 32 / 64 µm;
   - cell mask applied vs not;
   - upsampling ×1 / ×2 / ×2.7 (0.37 µm/px, KRONOS2's reference);
   - CLS only vs CLS + per-marker PCs.

   This decides whether the +0.01 gain is the ceiling or a tokenisation artefact, before any fine-tuning.
3. **Scale-resolved correlation curve** (mostly CPU). Extend the 32 µm patch metric to 8 / 16 / 64 / 128 µm.
   - Per-cell models and the oracle need only `cross-cohort-metrics-patch` with a list of patch sizes.
   - Image models need one more pass each, or save pixel maps once.
   - This turns the three-level figure into a curve showing where the image models' advantage ends (§0.3 Phase 4 item 15).
4. **Why ImmuVis transfers D→J but not J→D** (GPU, 1–2 runs). Two candidate causes:
   - the number of training images: subsample Danenberg to 397 images;
   - the fixed clip saturating bright Danenberg CD45/CD20: retrain on Jackson with the per-cohort clip (Danenberg 2.9).

   Currently there is one run per direction.
5. **Cross-cohort CD3 scale shift** (CPU). SHAP pins the J→D T-cell errors on own CD3/CD45 values. Test per-marker quantile matching of the inputs, for those two markers only, as a targeted fix. Score both cell typing and imputation.
6. **Keren H3K27me3 vs pS6 spillover** (CPU, existing data). pS6 (Yb172, the mass neighbour) carries 0.31 of the SHAP weight.
   - Fit the leakage as a fixed isotope ratio and subtract it.
   - Then recheck the 2 / 18 Jackson relations that Keren keeps (§17 item 2; §18.4 item 1).
7. **KRONOS2 adaptation**, only if item 2 finds a tokenisation that clears +0.02 (§18.2 steps 2–3):
   - LoRA vs full fine-tuning, DDP on 2×T4, fp16;
   - continued DINO pretraining on our training images, then the cell-type head.
8. **VirTues reference run**, only on reviewer request (§18.3).

### 18.6 Generative evaluation: distributions, not point predictions (2026-10-27)
**Why:** every model we compare returns one value per cell or pixel (the conditional mean). LightGBM, MLP and GraphSAGE are regressors. The U-Net, ImmuVis and EVA, as run, give one deterministic map; ImmuVis's β-NLL head also predicts a variance. So we have not yet tested any *generative* ability. A mean predictor is under-dispersed by construction: its predictions vary less than the truth by exactly the variance it cannot explain.

**Done (diagnostic only):** `Results/cross-cohort-metrics/realism_vs_correlation.py` → `fig_realism_vs_correlation.png`. It plots correlation against Wasserstein-2 between the predicted and true marginals at pixel, patch and cell level, in both directions.
- Pixel and patch W2 is a moment bound (only sums saved). The cell-level W2 is exact; the bound is 90–97% of it.
- **J→D:** the transferred cell models are no closer than a constant at the cell level (LightGBM 0.144 vs 0.139). They are over-dispersed (SD ratio 1.4×, ImmuVis 1.6×): a symptom of cohort shift.
- **D→J:** every model is under-dispersed (0.47–0.72×), as expected of mean predictors.
- Read this as a distribution-match diagnostic, not realism. To do: relabel the figure; add bootstrap CIs on W2.

**Plan** (Kaggle CPU kernel, cell tables; both directions; the same splits and seeds as `train-test-grid` replicate 0):
1. **Give the cell models a predictive distribution:**
   - quantile LightGBM, with 19 quantiles per marker (sort the quantiles to stop them crossing);
   - residual sampling: the prediction plus residuals drawn from training cells of the same type and prediction decile.
2. **Give ImmuVis one too.** Its predicted variance per pixel, pooled per cell, defines a Gaussian. The U-Net and EVA stay as point predictions; they are scored with CRPS, where a point prediction gets its absolute error.
3. **Scoring:**
   - **CRPS:** one scale for distributions and points;
   - **log score / information gain** (Tonks et al. 2026, arXiv 2602.23305);
   - **calibration:** a PIT histogram and the coverage of the 80% and 95% intervals, within each cohort vs across cohorts;
   - **distribution match on samples:** W2 per marker, energy distance and a classifier two-sample test (AUROC) on the joint 10-marker vector, within each cell type; the marker–marker correlation matrix;
   - **downstream checks on sampled cells:** phenotype composition with the Danenberg classifier; immune–tumour contact depletion.
4. **Joint generator (optional, GPU):** a mixture density network or conditional flow over all held-out markers given the measured ones. This is the actual "virtual cell". Compare it against independent per-marker quantile models on energy score and correlation structure.
5. **Optional:** single-cell KID/FID on cell crops with a microscopy encoder (KRONOS), not Inception, with a real-vs-real baseline. Secondary only: it compares distributions, not cells.

**Gate:** call a model "generative-valid" for a marker only if its 80% interval covers 75–85% of cells and its CRPS beats the constant's, on the *transfer* cohort.

#### 18.6.1 Bayesian arm: hierarchical imputation with cohort effects (pre-registered 2026-10-27, before any code)
**Question:** does a hierarchical Bayesian imputation model stay calibrated under cohort shift, without losing accuracy? And how fast does it adapt when a few target patients have the marker measured?

**Prior work found** (search 2026-10-27): IMC has Bayesian *cell typing* (Astir) and Bayesian *spatial interaction* models (SHADE; Eliason et al. 2025, a log-Gaussian Cox process). We found none for marker imputation with a calibrated posterior. ImmuVis has a heteroscedastic head but no posterior over cohorts. Tonks et al. 2026 frame virtual staining as posterior inference, but on brightfield data.

**Model, per target marker m** (NumPyro):
- **Mean:** y = f(x) + a_cohort + a_patient + Σ_k b_k,cohort · x_k.
  - x = the other 10 shared markers, H3, DNA1, DNA2 and log area (harmonised, as in `train-test-grid`);
  - f = additive natural splines with 4 df per input;
  - b_k,cohort = cohort deviations of the linear slopes, with prior b_k,cohort ~ N(0, τ_b);
  - a_patient ~ N(0, τ_p).
- **Likelihood (amended 2026-10-27, before any code):** a two-component mixture on the unclipped harmonised arcsinh scale, left-censored at the floor (Tobit):
  - y ~ π(x) · Background + (1 − π(x)) · Expressed, with logit π(x) = h(x) + c_cohort;
  - Background = Student-t with a small scale;
  - Expressed = Student-t (ν learned, robust to spillover and segmentation errors), with mean μ(x) as above and log σ = g(x) + s_cohort;
  - both components are censored at the observed floor: a cell at the floor contributes the CDF, not the density.
  - **Why** (checked on the test-cell truth): exact zeros are rare (0–4% of cells), but up to 50% of cells sit within 0.02 of the floor (Ki-67, CK8/18, CD68 in Danenberg). That is a continuous off-population, not a point mass, so a zero-only hurdle would model almost nothing. The skew of log y has opposite signs across markers (−1.65 to +1.18), so no single positive family (Gamma, Log-Normal) fits every marker. Censoring, not truncation, because floor values are real observations at the detection limit.
  - **Sensitivity models:** the plain heteroscedastic Student-t (the original plan) and a Log-Normal hurdle, compared by PSIS-LOO on training patients and by the same test metrics. The primary model stays the mixture whichever wins.
- **Priors:** τ_b, τ_p ~ HalfNormal(0.3); the spline weights get a weak N(0, 1).
- **Space:** fit on the unclipped harmonised arcsinh scale (this avoids the ERα ceiling). Map the samples through the monotone target transform, clip them to [0, 1], and score everything in target space, so ImmuVis is directly comparable.
- **Inference:**
  - NUTS on 50k training cells per cohort (4 chains, 1000 + 1000 draws);
  - ADVI on the full training set, checked against NUTS on the same 50k.

**Identifiability caveat (decided in advance):** with only two cohorts, the between-cohort variance is driven by the prior. So zero-shot transfer is reported with the prior-sensitivity range (τ_b ∈ {0.1, 0.3, 1}), not as a single number. Keren (third cohort) is optional and exploratory: its table is standardised and its panel overlap is partial.

**Scenarios** (splits and seeds of `train-test-grid` replicate 0; both directions):
| | Training data | Test | Purpose |
|---|---|---|---|
| S1 | target cohort | held-out target patients | sanity: within-cohort accuracy and calibration |
| S2 | source cohort | target cohort, new-cohort effect drawn from the prior (zero-shot) | the main shift test |
| S3 | source + k target patients with the marker measured, k ∈ {1, 5, 20} | the remaining target test patients | few-shot adaptation |
| S4 | both cohorts | each cohort's held-out patients | the "Both" result, with uncertainty |

**Comparators**, the same cells and scenarios:
- point LightGBM (CRPS = MAE);
- quantile LightGBM (19 quantiles);
- LightGBM + residual sampling (§18.6);
- ImmuVis Gaussian (S2, from its predicted variance);
- the constant (training mean);
- complete-pooling Bayesian (no cohort or patient effects): this isolates what the hierarchy adds.

**Metrics** (10 markers, ERα excluded; 1000× patient bootstrap; paired differences use the same draws):
- **Primary:** cell CRPS.
- **Calibration:** coverage of the 80% and 95% intervals; the PIT histogram (KS distance from uniform); mean interval width, as a sharpness check, so that wide intervals can't buy calibration.
- **Secondary:**
  - R4 of the posterior mean;
  - NLL;
  - W2 and energy distance on one posterior draw per cell;
  - correlation structure, if fitted jointly (later);
  - phenotype composition on sampled cells.

**Hypotheses and decision rules (fixed now):**
- **H1, calibration under shift (S2):** the Bayesian |coverage₈₀ − 0.80|, averaged over markers, is lower than quantile LightGBM's (paired CI < 0), in both directions. This must hold across the τ_b range.
- **H2, no loss of accuracy (S1, S4):** the R4 of the posterior mean is within 0.05 of LightGBM (non-inferiority margin, lower CI > −0.05).
- **H3, few-shot (S3):** with k = 5, CRPS closes ≥ 50% of the gap between S2 and the target-trained model.
- **H4, useful, not just calibrated:** in S2, CRPS beats the constant (CI < 0) for ≥ 6/10 markers.
- **Kill rule:** if H1 fails *and* S2 CRPS is not better than quantile LightGBM, drop the Bayesian arm and report it as negative.

**Controls and diagnostics:**
- **Simulation-based calibration:** simulate data from the fitted model with a known cohort shift, refit, and check that coverage recovers within 0.78–0.82. Run this before any real-data claim.
- **Negative control:** permute the cohort labels within the training data. The cohort effects should then shrink to about 0 (posterior τ_b near its prior mode, no gain over complete pooling).
- **Convergence:** R-hat < 1.01 and bulk ESS > 400 for every hyperparameter. For ADVI, report coverage against NUTS on the same 50k cells (difference ≤ 0.03), else use NUTS only.
- **Replicates 1–2:** only if H1 or H3 passes on replicate 0.

**Compute:** Kaggle CPU kernel `bayes-impute`. About 2–4 h per direction for NUTS (10 markers × 50k cells); ADVI on the full data takes under 1 h. No GPU needed.
**Order:** run after the §18.6 quantile and residual baselines. They share the scoring code, and they are the bar the Bayesian arm has to clear.

### 18.7 Cross-cohort harmonisation: a CytoNorm-like correction (2026-10-29)
**Why:** part of the transfer failure is a plain level shift (§18.6: Danenberg CD45, SMA and ERα sit off the floor). Our current fixes (cohort z, rank) are global. CytoNorm (Van Gassen 2020) can't be used, because the cohorts share no reference sample. cyCombine (Pedersen 2022) is the reference-free analogue: SOM clusters, then ComBat within each cluster.

**Arms** (kernel `harmonise`, CPU; LightGBM point + quantile, rep 0 split; J→D and D→J):
1. current: source-cohort scaling (= `prob-impute`);
2. global cohort z (per marker);
3. per-cluster ComBat (cyCombine style; SOM on the input markers);
4. per-cluster quantile matching (CytoNorm-style splines, but cohort-to-cohort, no reference sample);
5. per-marker GMM landmark registration (align the floor and the high component; UniFORM/MxNorm style);
6. oracle: the target marker's marginal from the test cohort (§0.3 item 2 follow-up). Upper bound only.

**No leakage:** arms 2–5 fit the correction on the **input** markers only. The target marker is never corrected with test-cohort information; its output is mapped back with the source cohort's statistics, as now.

**Metrics:** R4, cell RMSE vs the constant, CRPS, coverage₈₀, W2 (§18.6 scoring code).

**Biology check (the risk):** reference-free correction can erase real composition differences. Report the coarse cell-type fractions per cohort before and after correction (`cell-type-graph` labels). A method that moves them by > 5 points flattens biology, so flag it.

**Decision:** if the best of arms 2–5 closes ≥ 50% of the coverage gap between arm 1 and the oracle (arm 6) in both directions, it becomes the default preprocessing for `bayes-impute` (S2). If even the oracle doesn't fix coverage, the failure is a change in how markers relate, not a level shift, and harmonisation is dropped.

### 18.8 Tabular foundation models (TabPFN v2/2.5, TabICL) as the cell model (2026-10-29)
**Why:** the cell table is our best model, and the literature search (2026-10-29) found **no** published test of a tabular foundation model on IMC, CyTOF or spatial proteomics single-cell data. The only cytometry use found is a vendor case study (BostonGene, flow cytometry, patient-level classification). TabPFN also returns a predictive distribution (it approximates the Bayesian posterior predictive), so it doubles as a §18.6 generative baseline.

**Tasks** (kernel `tabpfn-cells`, one T4; rep 0 split):
1. **Marker imputation:** TabPFN regressor vs LightGBM; within cohort, J→D, D→J, Both. Score R4 and RMSE; CRPS and coverage from its output distribution, vs quantile LightGBM.
2. **Cell typing:** TabPFN classifier on the same features as Table and Table + nbr (§16); macro-F1.
3. **Few-shot (S3 analogue):** context = source cells + k = 0, 5, 20 target-cohort patients. In-context learning is where TabPFN should beat a refit LightGBM.

**Constraints:** TabPFN v2 context ~10k rows (2.5: ~50k). Subsample the context per test batch (stratified by image), and report the context size. Licence: check the TabPFN-2.5 weights licence before use (v2 is open).

**Decision:** report it as one row in the model comparison. It is a claim only if it beats LightGBM on transfer coverage or few-shot CRPS (paired CI), not on within-cohort R4, where the cell models already tie.
