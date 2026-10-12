# Future Plan: When Can a Virtually Stained Marker Be Trusted?

**Working title:** *Which biology survives virtual staining? A validated resolution ladder for imputed IMC markers*
**Scope:** Combined paper of Direction 1 (validated evaluation metric) + Direction 3 (panel substitution / downstream biology).
**Target venue:** **MLCB 2027, full paper with PMLR proceedings** (primary). Afterwards, an extended journal version (*Cell Systems* / *Bioinformatics*) carrying the P2 items.
**Compute:** Kaggle only (T4x2, ~30 GPU-h/week, 12 h sessions; the notebooks resume from the checkpoint cache).
**Grant fit (CZI, "Towards a unified atlas of spatial proteomics"):** Aim 1 (SpatialProteomicsNet as data backbone), Aim 2 (breast stain-to-stain translation), Aim 4 (the marker map in this paper is the first version of the per-marker trust table for the web dictionary).

---

## 0. Publication strategy and priorities

### 0.0 Flagship analysis and kill test (read this first)
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

**Kill test (weeks 3–6, no new training; notebook `FlagshipKillTest.ipynb`):**
- **Inputs:**
  - the pilot J2D checkpoints (11 markers × 3 preprocessing arms = 33 units);
  - the published Danenberg masks and cell table;
  - the frozen split.

  All of these come from the `danenberg-cells-prep` Kaggle kernel.
- **Kaggle GPU pass.** Built from AbundanceVsStructure E2 `_collect`. It writes per-cell means of:
  - the true target;
  - the predicted target;
  - 5 cyclically shifted predictions (the null);
  - all input channels.

  Everything stays in the model's target space.
- **CPU pass:**
  - R4 (per-cell r, image mean removed).
  - `R5_input` and `R5_panel` (section 6.2).
  - Three readouts:
    - ER/HER2 status AUROC;
    - phenotype-classifier substitution (macro-F1 drop; classifier trained on measured features of train patients);
    - M+ fraction per cell type.
- **Caveat:** the published masks used membrane channels. That is acceptable for this test only.
- **Decision** (thresholds fixed in the notebook before running):
  - **Dissociation** holds if either condition does:
    - **D1:** for ≥2 markers, the arm ranking by pooled r and the ranking by a cell-level readout go in opposite directions, and both CIs exclude 0.
    - **D2:** adding R4/R5 to pooled r raises leave-one-marker-out R² of the cell-level readouts by ≥ 0.1.
  - **Beyond-panel signal:** `R5_input` beats the cyclic-shift null for ≥2 markers.

| | R5 > 0 (≥2 markers) | R5 ≈ 0 everywhere |
|---|---|---|
| **Dissociation** | **GO**: full paper as planned | **PIVOT-A**: re-test R5 with the S1 main model at G0. If R5 is still 0, write a "virtual staining = panel redundancy" paper |
| **No dissociation** | **PIVOT-B**: weight moves to C2 (the marker map); decide with the PI | **KILL**: drop the ladder paper; E1/E2 becomes a short methods note |

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
- **Weeks 3–6 task (alongside the kill test):** a positioning memo covering 7-UP, MIM-CyCIF, VirTues, ImmuVis, EVA and the downstream-utility papers.

### 0.3 Run order (reprioritised 2026-10-08)
**How items are ranked.**
- By how much each tells us about the research question, per unit of compute. The question: can a generative model give trustworthy virtual cells for breast-cancer IMC?
- Then by what it unblocks.
- Cheap CPU items that can stop or redirect GPU work come first.

**Already done.** Cell tables, masks, clinical data and patient IDs for both cohorts (`danenberg-cells-prep`, `jackson-cells-prep`), plus the frozen Danenberg split.

**Phase 1: no training (weeks 3–6, Oct 12 – Nov 8). Decides GK.**
1. **Cell-table information map** (section 14, Steps 0–1; CPU; one Kaggle notebook).
   - Both cohorts, full panels, reported by marker group.
   - Includes the Jackson arm restricted to Danenberg's channels, which is gate 1 for section 15.
   - It tells which markers can be virtual at all, which need spatial context and which must be measured, so it sets the targets for everything after.
2. **Virtual tumour state from the cell table** (section 15-lite; CPU).
   - Apply the restricted Jackson cell-table models to Danenberg cells, then run the section 15 tests (PR/PAM50 and survival).
   - This is the first answer to the virtual-cell question with no GPU, and the baseline the image model must beat later.
3. **Kill test with cell-table baselines** (section 0.0; one GPU inference pass on the existing J2D checkpoints, then CPU).
   - Readouts: R4, R5_input and the D3 readouts.
   - Comparators: the panel-redundancy regressor and the context regressor from section 14.
   - Run the R5 positive controls here (kNN-mean; shuffle within state), so R5 is certified before GK relies on it.
4. **Neighbour interactions** (section 14, Step 2; CPU), including H3K27me3 vs CA-IX.
5. **Positioning memo** (0.2). Add SVCA/MISTy/NCEM and prior work that used imputed markers for survival.
6. **Housekeeping:** pull the E2 CSVs from Kaggle.

**Phase 2: main models (weeks 7–19). Starts only after GK.**

7. **G0:** smoke-test the main model on S1 and time one run.
8. **S1 main model** (gamma arm, 3 seeds) and **S2 Jackson main model** (3 seeds).
   - Interleave the runs (S1 seed 1, S2 seed 1, then the rest), so section 15 gate 2 is known early.
   - **Marker-dropout schedule:**
     - S1: often drop every channel Jackson lacks;
     - S2: often drop every channel Danenberg lacks.

     This lets the same weights serve S1-shared, S3 and section 15 without dedicated runs. Train dedicated S1-shared/S3 models only if the dropout version is clearly worse at G0.
   - Weight the targets toward the markers section 14 found context-dependent.
9. **`ladder` package, full positive controls and metric meta-evaluation** (section 6.4). Runs on CPU while the GPU queue runs. Includes the VirTues domain FID (inference). Gate G1.
10. **GPU zoo members, staged.**
    - Run the low-pass arm and the DNA/H3-only U-Net with 1 seed each first.
    - Add seeds 2–3 only if one lands within the main model's CI on any rung.
    - The panel-redundancy regressor and the per-pixel GBM are CPU and already exist from Phase 1.

**Phase 3: results (weeks 18–27).**
11. **Marker map (C2) and the P0 sensitivity checks.** The expansion sweep matters most, because spillover can fake R5 and the interaction terms. Gate G2.
12. **Flagship C3 on the full zoo** (section 7).
13. **Section 15 with the image model.** Same tests as 15-lite. Report what the image model adds beyond the cell-table virtual markers.

**Phase 4: P1 tail (weeks 28–31). If late, cut from the bottom.**
14. C5 cross-cohort: S1-shared vs S3 from the dropout models, plus the antibody clone check (sections 3, 8).
15. VirTues audit as zoo member 8 (inference only).
16. C4 rankings and the model-selection case study (section 6.6).
17. PatchGAN member, the cheapest adversarial ("generative") member.
18. Scale-resolved correlation curve (section 6.2).

**P2: journal extension or appendix.**
- Level B calibration.
- Survival stratification.
- Simulated segmentation errors.
- Cellpose vs Mesmer.
- 5-fold CV.
- Diffusion model.
- MIBI check.
- Web trust dictionary.

**Out of this paper.**
- The CyTOF/C2S side project (section 13) becomes a separate paper. It is parked: the rank analyses are done, and the PDAC IMC replication waits on Zenodo access.

### 0.4 Gates
| Gate | When | Pass condition | If it fails |
|---|---|---|---|
| **GK** | End of week 6 | The kill test (0.0) returns GO. It is read together with section 14: at least 2 markers show a context gain (ΔR² CI > 0), or the image model beats the cell-table baselines at R4/R5 | PIVOT-A, PIVOT-B or KILL, as in the 0.0 decision table. If no marker shows a context gain and the image model never beats the cell table, take PIVOT-A ("virtual staining = panel redundancy"). Section 15 then continues on the cell table. No zoo training starts before GK. |
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

This section is the flagship (section 0.0). It is built twice:
1. As the **kill test** on the pilot checkpoints (weeks 3–6; 11 markers × 3 arms).
2. On the **full S1 zoo** (weeks 22–27). This version produces the paper's flagship figure: x = pooled r, y = downstream agreement, colour = trust level.

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
| 1–2 | Sep 28 – Oct 11 | Kaggle prep kernel (cell table, masks, clinical data, frozen split); CyTOF profile-vs-physical analysis (section 13) | CyTOF decision |
| 3–4 | Oct 12 – Oct 25 | **Phase 1, first:** cell-table information map, both cohorts (section 14, Steps 0–1); section 15-lite (virtual tumour state from the cell table, PR/PAM50 and survival); pull E2 CSVs | — |
| 4–6 | Oct 19 – Nov 8 | **Kill test** with cell-table baselines and the R5 positive controls (section 0.0); neighbour interactions (section 14, Step 2); positioning memo | **GK** |
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
- **Week 6 (GK):** the kill-test decision table decides whether any zoo training happens.
- **Week 12 (G1):** if the positive controls do not land on their rungs, fix the ladder before interpreting any model results.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The flagship fails: rungs add nothing over pooled r | Kill test at week 6 (GK) on existing checkpoints, before any zoo compute is spent; decision table in 0.0 |
| Kill test underpowered (33 units) | Decide on paired within-marker arm comparisons with patient-bootstrap CIs; ambiguous results go to PIVOT, not KILL |
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
1. Is an unseen cohort available? It would reopen the pretraining comparison (D2) and make the VirTues audit P0.
2. Should the PatchGAN zoo member be included? It is P1 and costs 3 GPU runs.
3. Should a non-archival workshop version be presented before MLCB (e.g. an ICLR 2027 workshop, ~Feb 2027)? Check MLCB's policy on prior presentations first.
4. If extra compute (e.g. a university cluster) becomes available, it would restore 5-fold CV and S2.

## 13. Side project: link to the breast-cancer CyTOF / C2S project

> **Out of scope for the MLCB paper.** This becomes a separate paper, for example an MLCB 2-page abstract or a workshop paper. It shares the leave-target-out kNN code with rung R5.

**What and why.**
- **The CyTOF finding.** The CyTOF project (7 breast tumours, 31 markers) compares two ways of representing each cell: rank encoding (the "sentence" C2S reads) and intensity.
  - **Supported by the CyTOF repo:** rank grouping of marker-high cells is tighter, but on **PDAC** only. Mean kNN enrichment for intensity / rank vector / C2S is 2.09 / 2.73 / 2.71 (`PDAC_reanalysis.md`).
  - **BC** never had a model-free rank-vector arm; it compared only the C2S embedding with intensity.
  - **VERIFY (not found in the CyTOF repo on 2026-09-27):** the claim that "intensity is better at predicting a held-out marker from the cell's neighbours in marker space". No leave-target-out kNN predictor or result exists there. The closest is `level3_conditional.csv`: masked-marker prediction where C2S-LM Spearman ≈ 0 vs Ridge/XGBoost 0.53–0.67, which is not kNN. Treat this claim as a hypothesis until the analysis is run.
- **Those neighbours are not physical.** They are cells with similar profiles, not cells that are physically close, because CyTOF dissociates the tissue and loses spatial coordinates.
- **Rejected idea: a "virtual spatial" layout.** Building one for the CyTOF cells from an IMC model was considered and rejected:
  - A single cell's markers carry only a weak signal about its location.
  - Any such layout would therefore mostly copy the IMC prior.
  - That would make every spatial finding circular and impossible to validate on CyTOF.
- **The honest question.** Does the encoding that groups similar profiles better also better reflect a cell's real physical surroundings? IMC cell tables have both profiles and coordinates, so they can answer that directly.
- **What it adds and reuses:**
  - It extends the C2S result from similarity in marker space to tissue organisation.
  - The marker-space kNN code is shared with rung R5 (section 6.2).

**How.**
- **Data:** the Danenberg cell table prepared on Kaggle (`danenberg-cells-prep`).
- **Panel overlap with the 31-marker BC CyTOF panel:**
  - **Danenberg: only 6 markers** (panCK, CK5, CK8/18, SMA, ER, Ki-67).
  - **Jackson: ~11 markers** (those plus E-cadherin, vimentin, CD44, GATA3, H3K27me3 and pHH3). ER maps only if Jackson's "Rabbit IgG H L" channel is the ER antibody; verify this.
  - Profile spaces therefore run on three marker sets: the matched 6, all 36 Danenberg proteins, and random 6-marker subsets (to separate panel size from panel composition).
- **Encodings, copied from the CyTOF code:**
  - intensity = per-tumour (per-image) robust z `(x−median)/(1.4826·MAD+1e-6)`, clipped to ±5 (`scripts/bc_knn_enrichment.py`);
  - rank vector = each marker's within-cell position, computed from the clipped z, ties broken by column order (`scripts/rankvec_control.py` @ `9dad454`);
  - Euclidean kNN, k = 30.
  - The C2S LLM embedding is out of scope: it needs the CyTOF-trained adapter.
- **Step 2 runs first** (moved ahead of Step 1 on 2026-09-27).
  - Notebook: `ProfileVsPhysical.ipynb` (Kaggle kernel `profile-vs-physical`).
  - Results and the pre-registered decision rule: `Results/ProfileVsPhysical/summary.md`.
  - The nulls include a **type-matched** null (random cells with the same phenotype as each profile neighbour), which isolates the signal beyond cell type.
  - Touching cells are excluded, as a spillover guard.
- **Replication:** Jackson with the matched ~11 markers, run after Danenberg if there is a signal.
- **Step 1 (go/no-go):** measure how well a cell's own profile predicts its physical surroundings:
  - Surroundings = the immune, stromal and epithelial fractions among the cell's 10 nearest cells in physical space, and a cellular-neighbourhood label.
  - Model: gradient boosting.
  - Validation: patient-level CV.
  - Stop if the neighbourhood label is predicted at AUROC ≲ 0.65. Transferring any spatial information to CyTOF would then be meaningless.
- **Step 2 (main test, run first):** for each encoding (intensity and rank):
  - Build the profile kNN graph (k = 30).
  - For each cell, check two things:
    - are its profile neighbours physically close?
    - do they share its physical-neighbourhood composition (R² of context from the profile neighbours, and agreement of CN labels)?
  - Compare against a uniform null and a type-matched null, with a patient bootstrap for CIs.
  - Result: does the rank advantage inside the representation carry over to physical space, or does intensity win?
- **Step 3 (only if Step 1 passes):** train the Step 1 predictor on IMC and apply it to CyTOF cells.
  - It gives each cell a predicted neighbourhood with a calibrated probability (the Bayesian version of the virtual-space idea).
  - It is validated only on held-out IMC patients, never on CyTOF.
  - It is reported as a hypothesis generator, with two caveats:
    - CyTOF and IMC intensities differ.
    - IMC cell measurements include signal spilling over from neighbouring cells.

## 14. Cell-table information map and neighbour interactions (weeks 3–6, CPU, alongside the kill test)

**Why.**
- Measure, per marker, how much a cell's own other markers and its neighbours tell us about it, before any image model is trained.
  - A marker predictable from the cell's own profile can be imputed without an image model (panel redundancy).
  - A marker that depends on neighbours is the only kind for which a spatial generative model can beat a cell-table model.
- This gives R5_panel and the C2 map early and without a model. It also tells the main model (section 5) which targets matter.
- `profile-vs-physical` found that a cell's profile adds little about its surroundings beyond cell type (R² +0.02 to +0.08). That number was averaged over all cells and markers, so rare but strong interactions could be hidden in it. Steps 0 and 2 test that.

**Data.**
- Cell tables, contact graphs and masks:
  - Danenberg (36 proteins, spillover-corrected) from `danenberg-cells-prep`;
  - Jackson (32 proteins, not spillover-corrected) from `jackson-cells-prep`.
- Splits:
  - Danenberg: the frozen patient split.
  - Jackson: a new patient-level split built from the `jackson-cells-prep` patient IDs. This is also the split S2 needs.

**Scope: full panels, per dataset, reported by marker group.**
- The virtual-staining work so far used only the 11 shared markers. Those are almost all lineage markers, plus ER, HER2 and Ki-67.
- Nearly every state marker belongs to one cohort only:
  - Danenberg covers immune state;
  - Jackson covers tumour-cell state and the only chromatin mark.
- So Steps 1–2 run on the full panel of each dataset (36 Danenberg, 32 Jackson), and every result is reported per group. Groups:

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

**Step 0: immune–tumour contact check (hours).**
- Per image, count immune–tumour contacts in the contact graph. Compare with 1000 label permutations within the image (positions fixed).
- Report the share of images where contact is enriched, depleted or neither, and whether that split follows PAM50, ER or grade.
- Expected from the literature: patient-specific (mixed vs compartmentalised, Keren 2018), not uniformly enriched.

**Step 1: per-marker map.** For every marker M, per cell, with gradient boosting on the patient split:
- **(a) own:** the cell's other markers;
- **(b) own + neighbours:** (a), plus the cell-type counts among its contacts and within 30 µm, plus the mean profile of non-touching cells within 30 µm.

Readouts, with patient-bootstrap CIs and signal quality (positive fraction, SNR; E2 code) as a covariate:
- R² of (a): redundancy;
- ΔR² = (b) − (a): context.

**Jackson arm restricted to Danenberg's channels (the CPU gate for section 15).**
- Repeat (a) and (b) for every Jackson-only marker, using as inputs only the channels Danenberg also has: the 11 shared markers, H3, DNA1 and DNA2.
- A marker whose cell-level R² is no better than an intercept-only model here does not go to section 15.

**Step 2: interaction terms.**
- For each cell type A, neighbour type B and marker M: ΔR² from adding "number of contacts with B" to model (a), fitted within type-A cells.
- **Spillover guard:** only test markers that type B does not express itself (e.g. never panCK on T cells touching tumour).
- **Pre-specified examples:**
  - PD-1, ICOS and Ki-67 on CD8 T cells touching tumour;
  - HLA-ABC and β2M on tumour cells touching T cells;
  - CD163 on macrophages near tumour;
  - **H3K27me3 (Jackson only):** per-cell H3K27me3/H3 in tumour cells vs their own CA-IX and their distance to CA-IX-high cells. Hypoxia raises H3K27me3 through the oxygen-sensing demethylase KDM6A (Chakraborty 2019; verify).
- Benjamini–Hochberg correction across all (A, B, M) triples.
- **Spillover checks:**
  - Danenberg is primary.
  - Compare contact effects with non-touching effects at 10–30 µm.
  - The effects must not grow as the masks are expanded.

**H3K27me3 as a target.**
- It is the only chromatin mark in either cohort (Jackson only). Report it separately in Step 1, normalised per cell to total H3 so it does not just track nuclear density.
- Both outcomes are a result:
  - predictable from the other proteins → chromatin state largely follows cell identity, and a virtual H3K27me3 is possible;
  - not predictable → the mark carries information a protein panel misses, and must be measured.
- Earlier image models for H3K27me3 within Jackson failed (r −0.14 to −0.21, Nov 2025, old pipeline).

**Decision.**
- Markers with a context gain (ΔR² CI > 0) or a significant interaction term become the main targets for the image model (section 5) and the flagship (section 7).
- High (a) with no context gain → reported as panel-redundant.
- Low on both → "must measure".

**Novelty.** Neighbour-dependent marker variance is an established question (SVCA, MISTy, NCEM; verify details). Here it serves as the ceiling for virtual staining, not as a standalone biology claim.

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

**15-lite: the cell-table version (CPU, weeks 3–4; run first).**
- Train the restricted Jackson cell-table models (section 14) and apply them to the Danenberg cell table. This gives a virtual value per Danenberg cell for each target that passes gate 1.
- **Input harmonisation:** the published tables differ (Danenberg is spillover-corrected, Jackson is not). Inputs are therefore standardised per cohort. This uses Danenberg input statistics only, never target statistics (level B, labelled as such).
- Run the patient-level validation and the three survival tests below, unchanged.
- The image model (below) is later judged by what it adds beyond 15-lite.

**Gates (thresholds fixed before running).**
1. **CPU, weeks 3–6:** the section 14 Jackson arm restricted to Danenberg's channels. Cell-level R² > 0 with the CI excluding 0.
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
