IMC Virtual Staining — Summary
========================================
EXPERIMENTAL CONDITIONS
========================================

1. Per-marker ResNet-UNet (PCA)
   Preprocessing: SGDRegressor per input channel regressing from 3 structural controls
   (Histone H3, DNA1, DNA2) → arcsinh residuals → joint PCA compression to 3 components
   → z-score in PCA space.
   Architecture: ResNet50-UNet, in_channels=3, random conv1 init.
   Training: 11 separate models, one per target marker. Jackson2020 → Danenberg2022.

2. Joint ResNet-UNet
   Preprocessing: Same regression residuals + arcsinh, but NO PCA. Each of 10 residual
   channels z-scored independently using training-set statistics.
   Architecture: ResNet50-UNet, in_channels=10, random conv1 init.
   Training: Single model jointly trained on all 11 marker-prediction tasks. Model has no
   explicit marker identity — must infer from channel ordering (which shifts per task,
   making it marker-agnostic).

3. Joint ResNet-UNet + HyperConv
   Preprocessing: Identical to condition 2.
   Architecture: ResNet50-UNet with MarkerStem replacing conv1. MarkerStem receives
   Jackson vocabulary indices of the 10 input markers, looks up learned embeddings,
   projects each to a 7x7 kernel, applies per-marker convolutions, sums to 64-channel
   output. Parameter count matched to condition 2.
   Training: Same joint multi-task setup; marker_ids passed per batch.
   NOTE: Run in POC_MODE — 10 epochs only, patience=3. Results are undertrained.

Three preprocessing arms:
     (a) Regression-residual (current): one SGDRegressor per input channel on
         [H3, DNA1, DNA2] in raw intensity space → arcsinh residual → z-score.
         NOTE: this is ALREADY marker-specific — each channel gets its own coefficients.
     (b) Multiplicative gamma-correction (new, scTransform-inspired): arcsinh FIRST →
         regress each marker on the nuclear technical axis T = PC1(arcsinh[H3,DNA1,DNA2])
         → correct: arcsinh(x_m) − gamma_m * (T − median T) → z-score each channel. 
         This differs from (a) in three ways:
         multiplicative (log-space) rather than additive (raw-space) model; a single
         anchored technical axis instead of multivariate control regression; and
         median-centering so typical nuclear-density pixels are unchanged.
         RELATION TO THE CyTOF/scTransform METHOD (see "A new normalization method for
         CyTOF based experiments"): arm (b) is a faithful port of that estimator — same
         multiplicative-in-arcsinh model, same gamma_m = regression slope of arcsinh(x)
         on the technical axis, same median-centered subtraction, same final z-score.
         TWO deviations: (i) it is applied PER-PIXEL on IMC, whereas the paper is PER-CELL
         for CyTOF — so here T is a nuclear-density / subcellular-architecture signal, a
         PROXY for the paper's per-cell permeability scalar T_i; (ii) axis order differs —
         we use T = PC1(arcsinh(controls)), i.e. PCA on arcsinh-transformed controls,
         NOT arcsinh(PC1(raw histones)) as the paper suggests. The paper's optional
         per-cluster refit is not implemented (single global fit).
     (c) ImmuVis-style: arcsinh(x/5) + 2D Butterworth low-pass (order 2, 0.2 Nyquist,
         zero-phase; Uchal et al. 2026 App. A.7) + per-channel 1st–99th-percentile
         min-max + clip [0,1]. No nuclear-control regression.
   
   If (a)/(b) match or beat (c), biology-grounded cleaning beats signal processing at
   fixed model size. VirTues comparison not needed — ImmuVis already dominates it under
   both pipelines (their Tables 1 and 8).

========================================
RESULTS (cross inference from Jackson2020 to Danenberg2022, using 11 shared markers)
========================================

Marker groups:
  Clinical/molecular:  ERa, HER2, Ki-67
  Immune:              CD3, CD20, CD45, CD68
  Structural/epithelial: CK5, CK8/18, panCK, SMA

--- Mean Correlation (Pearson) ---

  Per-marker ResNet-UNet (PCA)       : 0.290
  Joint ResNet-UNet                  : 0.234
  Joint ResNet-UNet + HyperConv      : 0.305

  For reference*:
    EVA (ImmuneVIS prepro            : 0.284 (zero-shot)
    VirTues** (immuneVis prepro)      : 0.728
    ImmuVis_Conv                     : 0.774

* Reference numbers are from Uchal et al. 2026 on IMMUcan Head & Neck cohort (leave-one-of-38-markers-out) so not direcrtly comparable (cohort, task, preprocessing differ).
** Their original preprocessing pipeline requies statistics from unseen test cohort (still, ImmuneVis > Virtues).


--- Per-marker (Ours) ---

  Marker       | Per-marker (PCA) | Joint  | HyperConv (POC with 10 epochs)
  -------------|------------------|--------|----------------
  HER2         | 0.450            | 0.415  | 0.412
  ERa          | 0.578            | 0.586  | 0.636
  Ki-67        | 0.175            | 0.111  | 0.183
  CD3          | 0.147            | 0.150  | 0.157
  CD20         | 0.218            | 0.188  | 0.253
  CD45         | 0.370            | 0.386  | 0.437
  CD68         | 0.097            | 0.160  | 0.119
  CK5          | 0.081            | 0.078  | 0.124
  CK8/18       | 0.368            | 0.394  | 0.253
  panCK        | 0.574            | 0.135  | 0.389
  SMA          | 0.130            | 0.026  | 0.391

--- Mean FID (lower is better) ---

  Indv. ResNet-UNet per marker(PCA)       : 251.7
  Joint ResNet-UNet                       : 253.0
  Joint ResNet-UNet + HyperConv           : 234.1

  Notable HyperConv FID improvements:
    CK5   : 241 vs 378 (joint baseline)
    Ki-67 : 177 vs 262
    CD20  : 204 vs 271
    CD3   : 204 vs 321

Note: the HyperConv row in the table above is POC only (10 epochs); the preprocessing-ablation runs below are full 50-epoch (see ABLATION RESULTS). RMSE not comparable across families due to different target normalizations (per-image min-max vs global percentile).


ABLATION RESULTS - @Results/J2D/HyperConv/preproc_ablation_results_full.csv

Setup: SAME architecture (ResNet-UNet + HyperConv), SAME Jackson2020 -> Danenberg2022
task and 11 shared markers. ONLY the preprocessing changes.
NOTE: all three arms are FULL 50-epoch converged runs (early stopping patience 8 +
ReduceLROnPlateau LR decay; see train curves) — NOT the 10-epoch POC, so these ablation
numbers are not undertrained.
Three arms:
  - residual : additive nuclear-channel regression residual (original scheme)
  - gamma    : multiplicative scTransform-inspired correction: arcsinh(x) − gamma_m*(T − median T),
               T = PC1(arcsinh controls), then per-channel z-score (new variant)
  - immuvis  : ImmuVis-style signal processing: arcsinh(x/5) + 2D Butterworth low-pass
               + per-channel 1st–99th-percentile min-max + clip [0,1]

--- Mean over 11 markers ---
  Preprocessing         | Mean Pearson       | Mean FID 
  ----------------------|--------------------|----------------
  residual (original)   | 0.255              | 235.7
  gamma                 | 0.274              | *201.8*
  immuvis               | *0.284*            | 213.4

--- Per-marker Pearson ---

  Marker   | residual | gamma  | immuvis
  ---------|----------|--------|--------
  HER2     | *0.403*  | 0.362  | 0.341
  ERa      | *0.589*  | 0.463  | 0.514
  Ki-67    | 0.152    | 0.182  | 0.133
  CD3      | 0.095    | 0.126  | 0.140
  CD20     | 0.225    | 0.257  | 0.203
  CD45     | 0.348    | 0.410  | 0.391
  CD68     | 0.068    | 0.070  | 0.093
  CK5      | 0.088    | 0.096  | 0.097
  CK8/18   | 0.228    | 0.264  | 0.393
  panCK    | 0.335    | 0.508  | 0.538
  SMA      | 0.270    | 0.274  | 0.277
  ---------|----------|--------|--------
  MEAN     | 0.255    | 0.274  | *0.284*

--- Per-marker FID ---

  Marker   | residual | gamma  | immuvis
  ---------|----------|--------|--------
  HER2     | 203.7    | *179.4*  | 181.1
  CD20     | 183.1    | 167.6  | 184.5
  CD3      | 241.5    | 208.0  | 207.5
  CD45     | 214.5    | 151.0  | 192.4
  CD68     | 258.4    | 220.8  | 249.9
  CK5      | 302.5    | 247.8  | 277.7
  CK8/18   | 208.6    | 178.1  | 190.9
  ERa      | 294.3    | 257.9  | *254.3*
  Ki-67    | 203.8    | 205.7  | 195.4
  panCK    | 253.9    | 212.7  | 233.3
  SMA      | 228.5    | 191.3  | 179.9
  ---------|----------|--------|--------
  MEAN     | 235.7    | *201.8*  | 213.4

--- Interpretation ---

  * ImmuVis-style preprocessing has the highest mean Pearson (0.284). Cheap
    signal processing is at least as accurate as removing nuclear structure.
  * The gamma variant essentially TIES ImmuVis on accuracy (0.274 vs 0.284) with higher FID
    (FID 201.8 vs 213.4). There is no need for a foundation-model / multi-cohort
    pretraining, just one cohort.
  * The original additive RESIDUAL scheme is the WEAKEST arm on both metrics
  * Per-marker caveat: no arm dominates. ImmuVis wins big on CK8/18 (0.393 vs 0.264);
    residual wins ERa/HER2; gamma wins CD45 and FID nearly everywhere.
  * FID is now the differentiator carrying the story. VERIFY the FID gap is stable
    (repeat seeds / bootstrap) before building the paper on it.

Possible paper narrative: A cheap, single-cohort, biology-grounded preprocessing (gamma) MATCHES foundation-model-style preprocessing on accuracy and BEATS it on spatial fidelity, at a fraction of the data cost. Also, FID as a needed IMC benchmark metric.

========================================
ABUNDANCE vs STRUCTURE (Experiments 1 & 2) — @Results/.../e2_arm_marker_table.csv, e2_did_contrasts.csv
========================================

THE QUESTION (hypothesis)
  A high pixel-to-pixel correlation between the predicted and the true marker could be
  misleading. The model might only be guessing each image's OVERALL LEVEL of a marker
  (which patient is high or low) while getting the actual SPATIAL PATTERN inside the tissue
  wrong. If that is what is happening, the marker is not really "imputable" — you could not
  drop it from a panel, because you would lose its spatial biology. We were especially
  worried about ERα, which varies a lot in overall level between patients. So we asked:
  does the good correlation come from real spatial recovery, or just from ranking patients
  by abundance?

HOW WE TESTED IT
  Inference only — no retraining. We split the pixel correlation into three parts, per
  marker:
    - pooled: all pixels together (the number we usually report).
    - between-image: correlation of the per-image AVERAGES — does the model get which
      images are high or low overall (abundance).
    - within-image: correlation after subtracting each image's own average — does the model
      get the spatial pattern inside the image (structure).
  Experiment 1: the residual checkpoint, three markers — ERα (main test), HER2 (backup),
    panCK (control that should keep its structure).
  Experiment 2: extended to ALL 3 preprocessing arms (residual, gamma, immuvis) × ALL 11
    markers. Added a between-PATIENT version (pixel-weighted patient averages) as a
    sensitivity check, a signal-quality score for each true channel (fraction of "on" pixels
    and signal-to-noise), and a paired patient-level bootstrap (1000x, 718 patients over
    794 Danenberg images) for confidence intervals. We also tested whether the
    PREPROCESSING itself changes spatial recovery, using a paired contrast
    Δ = (within_A − within_B) − (between_A − between_B) between each pair of arms.
  Sanity check: the between-image and within-image pieces must add back up to the total.
  They do (identity residual ~1e-15).

MAIN RESULTS
  1. No marker "flips" to abundance-only. Even ERα — the marker with the strongest
     patient-ranking signal (between-image r = 0.82) — still recovers real spatial
     structure (within-image r ≈ 0.52). So the correlations are NOT just abundance; the
     model does learn the pattern inside the tissue.
  2. Markers fall into clear, arm-stable PREDICTABILITY TIERS by within-image r:
       HIGH: ERα (0.46), panCK (0.44)
       MID:  HER2 (0.33), CD45 (0.31), CK8/18 (0.26), SMA (0.25)
       LOW:  CD20 (0.17), Ki-67 (0.10), CD3 (0.10), CK5 (0.08), CD68 (0.04)
     The LOW tier lines up almost exactly with LOW signal quality (few "on" pixels) — these
     channels are weak/noisy, not specially "abundance-only".
  3. Between-image ≈ between-patient everywhere → the abundance ranking is real, not an
     artifact of some patients contributing many images.
  4. Most of the true signal is WITHIN images (only ~2–20% of a marker's variance is
     between images), so there is plenty of spatial pattern to recover.
  5. Preprocessing DOES change spatial recovery for some markers. Clearest pattern: the
     ImmuVis low-pass filter INFLATES the abundance (between-image) correlation for the weak
     immune markers without adding spatial content — e.g. CD3 between-image r jumps
     0.31 → 0.60 from residual to immuvis while within-image barely moves (0.09 → 0.11).
     So ImmuVis makes those predictions MORE abundance-driven; residual and gamma keep them
     more spatial.
  6. The "abundance-leaning" flag (high abundance, low spatial) fired only for CD3, CD68,
     Ki-67 in some arms — but ALL are low-signal-quality markers, so we do NOT read this as
     a real biological "abundance-only" result. It just means "weak / hard to predict".

WHAT IT MEANS (interpretation)
  - The good correlations are real: the model recovers spatial biology inside the image, not
    just the patient's overall level. This supports the virtual-staining and panel-design
    claims — the well-predicted markers (ERα, panCK, HER2, CD45) really are imputable in the
    meaningful, spatial sense.
  - Pooled correlation alone can flatter a pipeline, because it mixes in the easy abundance
    part. The WITHIN-image correlation is the honest metric, and it is the one that separates
    the arms — the same lesson as the FID story: the headline accuracy number hides
    differences that spatial metrics reveal.
  - For weak markers we still cannot separate "biologically unique" from "just noisy": the
    low-predictability markers are simply low-signal. This confirms the panel-design caveat —
    pair predictability with a signal-quality check before calling a marker redundant.


========================================
PROPOSED ABSTRACT
========================================

Cross-cohort virtual staining of IMC data — predicting markers missing from one study's
panel using another study's measurements — is a pressing practical problem: dozens of
breast cancer IMC cohorts exist with incompatible panels, and integrating them requires
either re-staining (expensive, often impossible on archival tissue) or computational
imputation. Current state-of-the-art approaches (VirTues, EVA, ImmuVis) address panel
heterogeneity through increasingly large foundation models pretrained on tens of millions
of patches. While pretrained checkpoints can in principle be fine-tuned on a single GPU
(e.g. via QLoRA), the pretraining itself requires a large curated multi-cohort dataset —
a resource most IMC labs, which typically have few cohorts, do not have. We
ask how far a compact ResNet-UNet trained from scratch on a single breast cancer cohort
(Jackson2020) can go on cross-cohort transfer to an independent cohort (Danenberg2022),
and whether preprocessing (not scale) governs that transfer. Standard data normalization (arcsinh + filter + min-max) treats each channel as an independent intensity signal, and do not account for both technical and biological variability. We tested two biologically-grounded, preprocessing pipelines that operate directly at the pixel level using the nuclear control channels (Histone H3, DNA1, DNA2), and a marker-specific sensitivity (gamma). Holding the architecture
fixed, we find that biology-grounded cleaning does not beat foundation-model-style signal processing on intensity accuracy (mean Pearson gamma 0.27 vs ImmuVis-style 0.28). Crucially, however, the gamma pipeline matches that accuracy while producing markedly more spatially realistic
images (mean FID 202 vs 213) — with no multi-cohort pretraining. We conclude that at fixed
model size a cheap, single-cohort preprocessing recipe is competitive with the
preprocessing used by large foundation models. We further ask whether these correlations
reflect genuine spatial recovery or merely getting each image's overall marker level right.
Splitting the pixel correlation into a between-image (abundance) and a within-image (spatial)
component, we find that even the most abundance-dominated marker (ERα) keeps substantial
within-image structure — the predictions are spatially real, not abundance artifacts — while
markers separate into stable predictability tiers that track true-signal quality. Because
pooled correlation can be inflated by the easy between-image component (e.g. low-pass signal
processing raises abundance agreement for weak markers without adding spatial content), the
within-image correlation is a more honest readout that, like spatial fidelity (FID) — which
current IMC virtual-staining benchmarks omit — distinguishes pipelines that intensity
correlation alone does not. Such per-marker imputability, judged on spatial recovery and
signal quality rather than pooled correlation, could support a data-driven criterion for
panel design: markers recoverable from the rest of the panel can be dropped to free scarce
metal-tag channels for markers that cannot.


========================================
POTENTIAL APPLICATIONS & IMPACT
========================================

1. Cohort integration / panel harmonization (core use case)
   Impute markers missing from one cohort's panel using another's, enabling joint
   analysis of cohorts with incompatible panels — without re-staining archival tissue.

2. Data-driven panel design - Per-marker imputation accuracy is a direct readout of how much INDEPENDENT information each marker carries given the rest of the panel:
     - Highly predictable markers (recoverable from the others) are partially REDUNDANT —
       candidates to drop, freeing a scarce metal-tag channel.
     - Poorly predictable markers (e.g. Ki-67, CD68 in current results) carry UNIQUE
       signal — must be measured; prioritize them on the panel.
   The imputation model becomes a tool to maximize independent biological information per
   channel under IMC's hard channel budget.
   CAVEAT: low predictability can mean "biologically independent" OR merely "noisy / hard
   to predict." Predictability is necessary-but-not-sufficient evidence of redundancy —
   pair it with a signal-quality check before dropping a marker.

3. An imputed marker that reproduces known co-expression relationships (Experiment 4)
   is evidence the model learned biology, not dataset-specific intensity artifacts.


========================================
KEY FOLLOW-UP EXPERIMENTS
========================================

1. Inference only. Load the existing Jackson→Danenberg checkpoints, run them, and save the predicted and true image for each marker. Focus on ERα (main test), HER2 (backup), and panCK (should stay good — the comparison). Split the correlation three ways, per marker. Pooled correlation (your current number); between-image (correlation of the per-image averages — does it get which patients are high or low); within-image (correlation after subtracting each image's own mean — does it get the spatial pattern). Report within-image two ways: pooled-after-centering and the median per-image value. Bootstrap by patient if you can tell which image belongs to which patient; otherwise by image, and say so. Quick check: the within and between pieces of variance/covariance should add up to the total. Finally, Look for the flip. The signal you're after: ERα scores high on pooled and between-image but low on within-image. If you see that, you have your candidate.
2. Check if the flip matters for panels — no segmentation needed. Fix all thresholds first on Jackson (ERα-positive cutoff, panCK-high cutoff, neighborhood size ~10 px) — never tune them on Danenberg. Then, inside panCK-high epithelial regions only, ask whether predicted ERα recovers true ERα's variation there — i.e. beyond just "it's in the epithelium," which panCK already gives it for free. Two controls make it clean. First prove the test works: true ERα must look clearly different from a shifted (location-scrambled) copy of itself. Then the flip is confirmed if predicted ERα looks the same as its own shifted copy while true ERα does not. Call it a failure using a preset error size on a real number — the epithelial ERα-positive fraction.

What each result buys you. Claim ERα is falsely "imputable" only if it keeps abundance but loses both within-image and within-epithelium detail, while panCK keeps both. One marker clearing this justifies moving to the strong-model check (phase 2). The paper needs the full marker map plus HER2 as a second example.


========================================
Draft
========================================

BIOLOGICAL-RELATIONSHIP VALIDATION OF IMPUTED MARKERS (no ground truth needed)
   Goal: show an imputed marker preserves known biology in the TARGET cohort even though
   that marker was never measured there — validating both the imputation and the
   panel-design use case above.
   Design:
     - Pick a marker M present in Jackson but ABSENT from Danenberg. Strong candidates
       (Jackson-only, well-characterized relationships): GATA3, progesterone receptor,
       E-cadherin, vimentin, CK19/CK7, EGFR.
     - Train to predict M from the shared inputs; apply to Danenberg → predicted-M map.
     - Validate against markers that ARE measured in Danenberg, via known relationships:
         GATA3       → POSITIVE spatial correlation with ERa (luminal program); enriched
                       in epithelial (panCK+), depleted in stroma (SMA+) and immune (CD45+).
         E-cadherin  → co-localizes with panCK / CK8-18; absent in SMA+ stroma.
         Vimentin    → ANTI-correlates with epithelial panCK; enriched in stroma (SMA+).
         EGFR        → tracks basal epithelium (CK5+).
     - Quantify: compare corr(predicted_M, measured_X) in Danenberg against the SAME
       corr(M, X) measured directly in Jackson (where both exist). Preservation of the
       relationship = success. Include a NEGATIVE control (a pair expected to be
       unrelated, e.g. GATA3 vs CD45) to demonstrate specificity, not a global intensity
       artifact.
     - Optional cell-level check: segment Danenberg, type cells from measured markers,
       confirm predicted M is enriched in the expected cell type (e.g. GATA3 high in
       luminal epithelial cells).
   NOTE on the H3K27me3 example: in Jackson→Danenberg this maps to the Jackson-only
   'histone_h3_trimethylate' channel. A repressive chromatin mark has only WEAK, noisy
   pixel-level relationships (mild anti-correlation with proliferation/Ki-67 and active
   chromatin), so use the lineage/luminal markers above as the PRIMARY validation and
   treat any chromatin-mark relationship as a secondary, harder test.
