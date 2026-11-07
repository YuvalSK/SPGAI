# IMC project recap (updated 2026-11-06)

## Main results
Cell-level r = per-cell Pearson within each image (image mean removed). In the virtual-cell results, unless marked as medians, numbers are means over the 10 shared markers (ERα excluded); calls use 1000× patient-bootstrap CIs. Item numbers point to "Experiments in order".

### Virtual cells across cohorts
1. **The field's pixel metrics hide, and can reverse, the cell-level ranking** (items 11, 26–27; main figure `Results/cross-cohort-metrics/fig_cross_cohort_metrics.png`).
   - Jackson → Danenberg, per-image pixel r (the VirTues metric) ranks ImmuVis first among Jackson-trained models (0.31). That equals a LightGBM trained on Danenberg itself, yet at the cell level ImmuVis scores 0.26 and that model 0.66.
   - Painting the true cell values into the masks scores only 0.53 pooled and 0.38 per image. ImmuVis beats this oracle on per-image r for HER2, CK8/18 and pan-CK, so pixel r rewards texture rather than which cells are high.
   - Pooled pixel r (the ImmuVis metric) compresses the gap: LightGBM 0.31 vs 0.26–0.28 for the image models, against 0.43 vs 0.26–0.30 at the cell level.
   - Low-pass preprocessing doubles CD3's between-image r (0.31 → 0.60) with no gain within images (0.09 → 0.11).
   - **Three levels: pixel, 32 µm patch, cell** (item 33; figure now 2 × 4). Jackson → Danenberg, the image models' pixel advantage fades as the scale coarsens.
     - Pooled r (pixel / patch / cell): LightGBM 0.31 / 0.59 / 0.43, ImmuVis 0.27 / 0.38 / 0.26, U-Net 0.26 / 0.48 / 0.26, EVA 0.28 / 0.48 / 0.30.
     - LightGBM vs ImmuVis per marker (better / worse): per-image pixel r 1 / 8; per-image patch r 3 / 5; pooled patch r 6 / 0; cell r 6 / 1.
     - True cell values score 0.53 per pixel but 0.98 per patch. A one-value-per-cell map loses almost nothing at 32 µm, so its pixel-level penalty is about scale, not information.
     - Danenberg → Jackson, ImmuVis keeps its lead at patch scale (0.68 vs 0.66 pooled), as at the cell level.
     - Patch RMSE: no Jackson-trained model beats a constant prediction at the training mean (U-Net 0.103 vs 0.100).
2. **Jackson → Danenberg, per-cell regression beats every image model at the cell level** (items 22–26).
   - LightGBM 0.43, against U-Net 0.26, ImmuVis 0.26 and EVA (zero-shot) 0.30. Per marker, LightGBM wins 9 / 1 against the U-Net, 6 / 1 against ImmuVis and 6 / 2 against EVA.
   - The ~2× gap over the U-Net holds in 3 replicates.
   - It is not a normalisation artefact: scaling Danenberg inputs with Jackson statistics moves LightGBM only from 0.45 to 0.43.
3. **Danenberg → Jackson, an image model wins: ImmuVis trained on Danenberg is the best model on all six metrics** (item 27).
   - Cell-level r 0.44 vs LightGBM 0.41 (5 markers won, 3 lost); cell RMSE 0.128 vs 0.154 for a constant prediction.
   - The image model's transfer is asymmetric (cell-level r 0.26 J→D vs 0.44 D→J); the cell models' is not (LightGBM 0.43 vs 0.41). One run per direction; the cause is untested.
   - The ImmuVis implementation is sound: within cohort, in its own space, validation pixel r is 0.65 (Jackson) and 0.62 (Danenberg).
4. **The shared markers are highly redundant within a cohort, and pooling cohorts closes the transfer gap** (item 25).
   - From the other 10 markers plus nuclear channels, a per-cell model reaches median cell-level r 0.72–0.74 within Danenberg and 0.63–0.65 within Jackson.
   - Training on both cohorts closes nearly all of the cross-cohort gap (median paired difference vs own cohort 0.00 to −0.02).
   - The gap sits in a few markers: SMA (sign flips), CD68 and CD20.
5. **Error metrics need a constant-prediction baseline, which current benchmarks omit** (items 26–27).
   - Jackson → Danenberg, no Jackson-trained model beats predicting the Jackson-train mean on any RMSE (constant 0.191 / 0.181 / 0.139: pooled pixel / per-image pixel / cell). Only training on the target cohort beats it clearly (cell RMSE 0.080).
   - Level offsets make up 25–40% of the squared cell error. Cross-cohort models get *which* cells are high partly right, but not *how* high.
   - This also explains the gap to the ImmuVis paper (pooled r 0.77, MSE 0.068): it pretrains on 26–27 cohorts, uses 37 inputs and a denoised target, and compares only against VirTues and EVA. Where we can compare (ImmuVis vs EVA, D→J), we agree with it.
6. **Cell tables recover tissue biology** (items 20–21).
   - Immune cells touch tumour cells about half as often as chance, in about 90% of images.
   - Hypoxia rises away from vessels (r 0.10).
   - H3K27me3 follows cell identity (R² 0.63, vs 0.19 from nuclear content) and rises with the cell's own CA-IX (r 0.20).
7. **For a single-cohort U-Net, cheap preprocessing matches foundation-model preprocessing** (items 9–11).
   - Gamma vs ImmuVis-style: mean r 0.274 vs 0.284, with a better FID (202 vs 213; one seed per arm).
   - Marker predictability falls into the same tiers under every preprocessing and tracks signal quality.

8. **Image foundation-model embeddings add little to cell typing** (items 32, 34; 3 replicates).
   - KRONOS and KRONOS2 saw no IMC in pretraining. Their embeddings alone sit 0.04–0.07 macro-F1 below LightGBM with neighbour features, within each cohort.
   - Added to the table, 32 embedding components give about +0.01 macro-F1, on Danenberg only.
   - KRONOS2 vs KRONOS: no overall winner. KRONOS2 is better on J→D (+0.06) and worse on D→J (−0.10).
   - Both pass the pre-registered adaptation gate, only just. KRONOS2 meets the "+0.01 with CI > 0" criterion in every replicate (+0.012–0.013 on D→D).
   - Cell-type drivers (XGBoost + TreeSHAP, item 32):
     - immune types are typed by their own immune markers; neighbours matter mainly for endothelial and stroma cells;
     - J→D T-cell errors come from a cross-cohort CD3 scale shift;
     - the SMA model does not transfer (R² 0.00);
     - Keren's H3K27me3 predictability is dominated by its mass neighbour phospho-S6 (possible spillover).

### Bottom line of the virtual-cell comparison (2026-10-27)
- **Which model:** across cohorts at single-cell resolution, cell-table models (LightGBM ≈ MLP ≈ GraphSAGE; GCN worst) beat every image model in Jackson → Danenberg. ImmuVis wins only at pixel level, and in Danenberg → Jackson (one run per direction).
- **The biggest factor is cohort shift, not architecture.** A LightGBM trained on the target cohort beats every transferred model at every level.
- **Update 2026-10-30:**
  - The shift is a level offset in each target marker. It makes transferred virtual cells overconfident (coverage 0.44–0.59) and separates them from real cells (iLISI ≈ 0).
  - Input harmonisation doesn't fix it. A few labelled target patients do (D→J coverage 0.44 → 0.71 with 20 patients).
  - See "Virtual cells as distributions" below.
- **Preprocessing:**
  - U-Net inputs: ImmuVis-style (arcsinh → Butterworth → p1–p99) ≥ gamma > residual (cell r 0.26 / 0.26 / 0.19).
  - Own-cohort input scaling for cell models: +0.02 cell r; not used in the figure, to stay fair to the image models.
  - Rank harmonisation: no gain.
  - EVA: arcsinh ≈ linear.
  - KRONOS normalisation: model stats ≈ own-cohort stats.
- **Tokenisation was not compared.** KRONOS used one recipe (32 µm cell crop ×2, mask applied); ImmuVis used fixed 128 px tiles. See `future_plan.md` §18.5.
- **VirTues was not run.** Its public weights saw both cohorts, and ImmuVis already beats it in its own paper. Run it only as a labelled, leaky reference if reviewers ask.

### Virtual cells as distributions: calibration, harmonisation, mixing (items 35–39, 2026-10-27 to 10-30)
The main result: **virtual cells are trustworthy within a cohort and not across cohorts.** Across cohorts, the failure is a shift in each target marker's level. Correcting the inputs doesn't fix it, but a few labelled patients from the new cohort do.

1. **Within a cohort, a predictive distribution is nearly calibrated; across cohorts it is badly overconfident** (`prob-impute`, item 36).
   - Quantile LightGBM cuts CRPS by about 30% against point LightGBM, on 10/10 markers in every setting.
   - 80% interval coverage is 0.74–0.76 within a cohort or pooled. It is slightly too narrow because patient-to-patient variation is missing.
   - Under transfer, coverage is 0.57 (J→D) and 0.44 (D→J), and 0/10 markers are calibrated. In J→D the model barely beats drawing from the source cohort's marginal.
   - Several markers have different marginals in the two cohorts: Danenberg CD45, SMA and ERα sit off the floor.
2. **Correlation and distribution match rank models differently** (cell-level W2, item 35; a diagnostic, not "realism").
   - J→D: the Jackson-trained LightGBM has the best r (0.43), but its W2 (0.144) is no better than a constant prediction (0.139). The U-Net, which shrinks its predictions toward the mean, has the lowest W2 (0.096).
   - Trained on the target cohort, LightGBM reaches a W2 of 0.036 in both directions.
3. **Harmonising the inputs, CytoNorm-style, doesn't fix transfer** (`harmonise`, item 37; pre-registered; not adopted).
   - Arms tested: cohort z, per-cluster ComBat (cyCombine-like), per-cluster quantile matching (CytoNorm-like) and GMM landmarks.
   - The best arm, GMM landmarks, closes 46% of the J→D coverage gap and makes D→J worse (−15%).
   - Input corrections move error between markers: HER2 and CD68 improve, while CD20 and CD45 collapse.
   - An oracle that gives the model the target marker's own marginal in the test cohort halves the gap (0.27 → 0.14 and 0.36 → 0.16). So the shift sits in the target marker's level, not in the inputs.
   - Even with the oracle, coverage only reaches 0.62–0.67, which leaves a change in how the markers relate to each other.
4. **A tabular foundation model (TabPFN v2) ties LightGBM; a few target patients are the real lever** (`tabpfn-cells`, item 38). To our knowledge this is the first test of TabPFN on IMC.
   - With all the data, TabPFN ties LightGBM on CRPS and coverage. At equal data (10k cells) it wins on 7–8/10 markers.
   - The pre-registered J→D claim passes only narrowly (coverage 0.57 → 0.59).
   - **Few-shot:** adding k target-cohort patients to the training context lifts D→J coverage from 0.44 to 0.64 (k = 5) and 0.71 (k = 20), with CRPS falling from 0.135 to 0.089. With k = 20 this beats the harmonisation oracle.
   - In few-shot, TabPFN is slightly better calibrated than LightGBM (Δgap −0.01 to −0.06).
5. **Transferred virtual cells keep cell types but don't mix with real cells** (iLISI vs cLISI, item 39; `Results/cross-cohort-metrics/fig_lisi_virtual_cells.png`).
   - iLISI is the "real vs synthetic" LISI of scDesign2/3; their mLISI = iLISI + 1. cLISI is computed on virtual cells, using their true coarse types (6 classes).
   - The ceiling is measured cells against measured cells from other test patients: iLISI 0.70 / 0.65 (Danenberg / Jackson test).
   - Every model trained on the other cohort, including ImmuVis, U-Net, EVA and the graph models, has iLISI ≤ 0.09. Their cLISI is 0.80–0.90, close to the measured 0.89–0.92. So the cell types are kept, but the whole population is shifted, consistent with item 3.
   - Own-cohort LightGBM reaches iLISI 0.39 / 0.32, because point predictions are under-dispersed. Adding residual noise lifts this to 0.50 / 0.46 and costs about 0.03 cLISI. Across cohorts the same noise helps little (0.07 / 0.13).
   - Image and graph models give no gain over LightGBM or the MLP. GCN, U-Net and EVA have the lowest cLISI.
   - Caveat: within a cohort, only LightGBM has saved per-cell predictions.

**Implication for the Bayesian arm (`bayes-impute`, §18.6.1, pre-registered):**
- The fix must model the target marker's cohort offset, through a cohort random effect, not through preprocessing.
- For an unseen cohort (S2), its intervals have to widen by the uncertainty of that offset.
- For few-shot (S3), the bar to beat is TabPFN at k = 5: coverage 0.62 / 0.64, CRPS 0.085 / 0.104.

### Across scales: pixel, patch, cell (items 24–27, 32–34, 40–42; updated 2026-11-06)
The main result: **for marker imputation, the information sits in per-cell means.** Sub-cellular pixels, patches, neighbourhoods and graph foundation models add ≤ 0.007 R4. Extra scales help only cell typing: pixel morphology within a cohort, and graph context or known neighbour labels across cohorts.

**Positive findings:**
1. **Sub-cellular morphology and texture help fine cell typing within a cohort** (`multiscale-audit`, item 41; 3 replicates).
   - Fine typing gains +0.021 macro-F1 (Danenberg) and +0.018 (Jackson); coarse Danenberg gains +0.010.
   - The gain survives the 2-px erosion and outside-band controls, so it isn't spillover.
   - It comes from shape-defined classes (stroma "small elongated / circular" +0.06–0.07, Jackson "epithelial low" +0.13) and endothelial cells (+0.015 to +0.027).
2. **Known neighbour labels help typing, including across cohorts** (`label-counts`, item 42; pre-registered).
   - Revealing 20% of the target cohort's cell labels gives +0.020 / +0.026 macro-F1 on the hidden cells (D→J / J→D); 50% gives +0.04. 5% is not enough.
   - This parallels imputation: the lever is a little target-cohort ground truth, not a better model.
3. **Graph learning wins only cross-cohort typing:** the SAGE+ GNN beats Table + nbr on J→D (0.590 vs 0.566) and D→J, driven by endothelial cells (item 24, `cell-type-graph`).
4. **Typing errors sit at compartment borders.** Endothelial and myeloid cells are heterophilous (same-type share 0.35–0.85 of chance). A cell's error rate is far higher when most of its neighbours are another type (epithelial 0.43–0.54 vs 0.03–0.07).

**Cohort mixing vs type separation (item 43, `Results/scale-lisi/fig_scale_lisi.png`):** no representation at any scale beats cohort-scaled intensity on both iLISI and cLISI. Patch embeddings (KRONOS, KRONOS2) separate the cohorts more; Novae mixes them only by blurring cell types.

**Negative findings (support PIVOT-A):** pixel features don't help imputation (≤ +0.007 R4) or transfer. Patch and typed-edge features fail (B2). Predicted-type neighbour counts add nothing, and iterating them hurts D→J. Novae (graph FM, trained from scratch) adds ±0.002 and blurs cell types when aligning cohorts.

**Best model and preprocessing by scale**

| Scale | Within cohort | Across cohorts |
|---|---|---|
| **Pixel** (image → marker map) | Image models win only on pixel metrics (ImmuVis validation pixel r 0.62–0.65), which reward texture. Scored per cell, the table wins (H3K27me3: LightGBM 0.60 ≈ ImmuVis 0.59). As extra table columns, pixel features help **typing only** (+0.01–0.02). U-Net inputs: ImmuVis-style ≥ gamma > residual. | ImmuVis wins pixel metrics J→D (per-image r 0.31) and every metric D→J (cell r 0.44 vs 0.41; one run per direction). Pixel features: no transfer gain. |
| **Patch** (32–128 µm) | Not needed: true cell values painted into masks score r 0.98 per patch. Patch means (B2) add nothing; KRONOS embeddings add ~+0.01 typing (Danenberg only). | J→D: LightGBM pooled patch r 0.59 vs ImmuVis 0.38. D→J: ImmuVis 0.68 vs 0.66. B2 and KRONOS: no consistent gain. |
| **Cell** (per-cell means) | **Imputation: LightGBM** (R4 0.72–0.74 Danenberg); MLP, GraphSAGE and TabPFN tie (TabPFN wins at equal data). Quantile LightGBM gives the distribution (CRPS −30%, coverage 0.74–0.76). **Typing: LightGBM Table + nbr**; every GNN is below it. Own-cohort input scaling +0.02 r. | **Imputation: LightGBM** J→D (0.43 vs 0.26–0.30 for image models); D→J ImmuVis edges it. **Typing: SAGE+ GNN.** No input harmonisation helps (ComBat, CytoNorm-like, GMM, rank). What works: pooling cohorts in training, k target patients (TabPFN k = 5: coverage 0.44 → 0.64), or ~20% revealed target labels. |

**So, still gradient-boosted trees?** Yes for imputation and within-cohort typing: LightGBM on cell means (+ neighbour features for typing, + pixel features for fine typing). XGBoost was used only for SHAP; TabPFN ties. The exceptions are cross-cohort typing (SAGE+) and D→J imputation (ImmuVis, one run). Across cohorts, architecture and preprocessing matter less than target-cohort data.

### H3K27me3 and the epithelial–mesenchymal axis (item 44, 2026-11-07; pre-registered)
- **Measured, not imputed: tumour cells with more keratin carry more H3K27me3,** in both Jackson hospitals (Basel 264 patients, Zurich 71).
  - Per keratin, r ≈ +0.4 to +0.5 within patients, in 97–99% of patients.
  - It holds beyond total H3, DNA, cell size, cell cluster, neighbourhood and a generic per-cell signal factor.
- **It is not an EMT effect.** Vimentin is flat, and Twist / Slug go slightly the other way.
- **Cross-technology replication (Keren, MIBI, TNBC) is weak** (−0.06), vimentin-driven, and not robust to neighbourhood adjustment.
- **Methods point:** in Jackson, H3K27me3 correlates +0.12 to +0.16 even with immune markers inside tumour cells. That points to a generic per-cell signal factor, which any IMC epigenetic association must model.

### Epigenetic marks: imputing a mark the target cohort never measured (items 28–31)
Can H3K27me3 (and pHH3) be given to cells from the 14 channels shared across cohorts? **Within Jackson, yes. Across cohorts, not shown.**

| Cohort | H3K27me3 measured? | Accuracy (partial r beyond nuclear content) | Reliable Jackson relations kept |
|---|---|---|---|
| Jackson (train, held-out patients) | yes | LightGBM 0.60 ≈ ImmuVis 0.59 (nuclear only: 0.17) | 29/33 and 25/33 |
| Danenberg | no | – | 19/33; after a post-hoc amendment that drops anchors which don't transfer (SMA, CD68, CD20), LightGBM 17/23 and ImmuVis 12/23 |
| Keren 2018 (MIBI, TNBC) | yes | Jackson-trained 0.12; Keren-trained 0.23 (same inputs) | only 2/18, even with the *measured* mark |

- **The image model adds nothing over the cell table.** pHH3 fails in every cohort, at 0.29–0.30.
- **EVA can't represent these marks:** H3K27me3, pHH3 and H3 map to the same gene token (H3F3A), so its outputs are identical.
- **Keren is not a usable external test.**
  - Its H3K27me3 is weakly tied to the shared markers, even inside Keren.
  - Jackson's own triple-negative patients keep the relations, so the cause is not TNBC biology.
  - Most of Keren's apparent predictability is likely isotope crosstalk from a neighbouring mass channel: pS6 (Yb173) sits next to H3K27me3 (Yb172). Removing the mass neighbours drops partial r from 0.54 to 0.28.
- **The Danenberg-style check, which judges an imputed mark against Jackson's relations, can wrongly pass relations.** On Keren it agreed with the true reference on only 14/18.
- **What's needed:** another IMC cohort with H3K27me3 measured.

### Rank vs intensity (CyTOF/C2S side project)
1. **Rank groups marker-high cells more tightly than intensity, from about 24 markers, in both cohorts** (items 14–15, 17).
   - kNN enrichment +0.14 at 31 markers (Danenberg) and +0.12 (Jackson); rank wins for 35 of 36 and 29 of 32 markers. Ki-67 loses in both.
   - About half the gain is built in: it disappears when the scored marker is left out.
2. **The price is lineage purity.** Rank lowers the same-type share by about 7 points at every panel size and raises cLISI and iLISI: neighbourhoods are blurrier, not batch-corrected (items 13, 16–17).
3. **Profile neighbours are cell-state neighbours, not spatial ones.** Look-alike cells are only 9–15% (Danenberg) and 13–23% (Jackson) closer in tissue than random cells, mostly because they share a cell type (items 13, 15).
4. **Rank's one niche is per-sample-normalised data.** There it closes 34–61% of the gap to cohort-normalised intensity in Jackson, through more comparable cell states (not in Danenberg; item 19).

### What did not hold (shapes the claims)
- **A single-cohort image model does not give trustworthy virtual cells across cohorts:** cell-level r ≤ 0.30 Jackson → Danenberg, against 0.66 when training on the target cohort, and no RMSE gain over a constant.
- **Spatial context adds little:** neighbours add ≤ 0.06 R², and graph models only tie the table model and MLP (items 20, 24–25).
- **Virtual tumour-state markers carry no survival information** (item 20).
- **Epigenetic marks don't transfer verifiably** (items 28–31).
  - H3K27me3 is imputable within Jackson, but the only external cohort with the mark (Keren) does not reproduce Jackson's relations.
  - pHH3 fails everywhere.
  - The early image-only attempt (item 3) failed outright.
- **Rank has no clinical gain** over cohort-normalised intensity and loses HER2 (AUC 0.94 → 0.78; pre-registered, item 18).
- **Borrowing tissue context from look-alike cells in other patients fails** (R² ≤ 0.045; item 13).
- **Uncertainty doesn't transfer:** under transfer, 80% intervals cover only 44–59% of cells, even with TabPFN (items 36, 38).
- **Input harmonisation (ComBat, CytoNorm-like, GMM landmarks) doesn't rescue transfer** (pre-registered, item 37).
- **Virtual cells from another cohort don't mix with real cells** (iLISI ≤ 0.09 for every model, item 39).
- **Finer or coarser scales don't help imputation:** sub-cellular pixel features (≤ +0.007 R4), patch / typed-edge features and Novae (±0.002) add nothing beyond cell means and Table + nbr (items 40–41).
- **Not yet validated:** the R5 rung ("information beyond the rest of the panel") failed its positive controls. Earlier ERα results were measured in a saturated target space.

## Part 1: Virtual staining across cohorts (Jackson 2020 → Danenberg 2022)
*Written before the per-cell and foundation-model comparisons (items 1–11). The current results are in "Main results" above.*

**Intro.** The headline metric in IMC virtual staining, pooled pixel correlation, can be raised by preprocessing without any gain in spatial accuracy. In our cross-cohort transfer, the low-pass preprocessing used by ImmuVis nearly doubled the between-image correlation for CD3 (0.31 → 0.60), while the within-image correlation barely moved (0.09 → 0.11). A cheap correction built from the nuclear channels and trained on one cohort matched ImmuVis-style accuracy (mean r 0.27 vs 0.28) and gave more realistic images (mean FID 202 vs 213). Current methods (VirTues, EVA, ImmuVis) handle mismatched panels with foundation models pretrained on many cohorts and report mainly pooled correlation. We tested how far a compact model trained on one cohort gets on a new cohort, whether preprocessing rather than scale drives that transfer, and whether a good correlation means real spatial recovery or only the right overall level per image.

**Method.** We trained ResNet50-UNets on Jackson 2020 and tested them, without fine-tuning, on Danenberg 2022 (794 images, 718 patients), predicting each of 11 shared markers (ERα, HER2, Ki-67, CD3, CD20, CD45, CD68, CK5, CK8/18, panCK, SMA) from the other 10. We first compared three designs: one model per marker on PCA-compressed inputs, one joint model for all markers, and a joint model with a HyperConv stem that knows which markers it receives. With the joint model (without HyperConv) fixed and trained for up to 50 epochs, we then compared three preprocessings. The first was an additive residual after regressing each channel on H3, DNA1 and DNA2. The second was a multiplicative "gamma" correction ported from a CyTOF normalisation method, which removes each marker's dependence on a nuclear-density axis. The third was the ImmuVis pipeline: arcsinh, Butterworth low-pass and percentile min-max. We scored pixel Pearson and FID. We then split each marker's correlation into a between-image part (overall level) and a within-image part (spatial pattern), using a 1000× patient bootstrap and a paired test of whether a preprocessing shifts one part more than the other.

**Results.** Mean Pearson was 0.29 for the per-marker models, 0.23 for the joint model and 0.30 for HyperConv; the HyperConv figure comes from a 10-epoch test run. In the 50-epoch preprocessing comparison, ImmuVis-style had the highest mean Pearson (0.284), gamma was close (0.274) with the best FID (202 vs 213), and the residual was weakest on both (0.255, FID 236). No preprocessing won every marker. No marker's correlation came only from overall level: even ERα, the most level-driven marker (between-image r 0.82), kept a within-image r of 0.52. Markers fell into the same tiers under every preprocessing:
- High: ERα and panCK (within-image r ≈ 0.45).
- Mid: HER2, CD45, CK8/18 and SMA (0.25–0.33).
- Low: CD20, Ki-67, CD3, CK5 and CD68 (≤ 0.17).

The low tier is also the low-signal tier. The ImmuVis low-pass filter raised between-image agreement for the weak immune markers without raising within-image agreement.

**Discussion.** In this setting, a compact single-cohort model with cheap preprocessing reaches the accuracy of foundation-model-style preprocessing, so preprocessing matters at least as much as scale. We have not yet compared against the foundation models themselves, whose published numbers come from a different cohort and task. Because pooled correlation flatters pipelines that smooth images, within-image correlation and FID should be reported alongside it. Gamma's FID advantage rests on one seed per arm and needs repeat seeds or a bootstrap before a paper can rely on it. The well-predicted markers (ERα, panCK, HER2, CD45) look imputable in the spatial sense, which could guide panel design. A poorly predicted marker, though, may be noisy rather than unique, so predictability has to be read together with signal quality.

## Part 2: Rank vs intensity on IMC cells (side project for the CyTOF/C2S paper)

**Intro.** In CyTOF, marker-high cells group more tightly under the rank encoding than under intensity. This advantage replicates in two breast-cancer IMC cohorts, but only with panels of about 24 markers or more. It costs about 7 points of cell-type purity at every panel size and brings no clinical gain over cohort-normalised intensity. The CyTOF project treats cells with similar marker profiles as neighbours, encoding each cell either by its intensities or by the rank order of its markers (the encoding used by Cell2Sentence). Spatial IMC lets us check what these profile neighbours mean in intact tissue. We tested whether look-alike cells sit close together and share surroundings, which encoding better groups marker-high cells and cell types, and whether rank-based cell states carry more clinical information.

**Method.** We built cell tables on Kaggle from the public releases of Danenberg 2022 (1.06M cells, 704 images, 650 patients) and Jackson 2020 (1.13M cells, 542 images, 349 patients). Each cell was encoded by per-image robust z (intensity) or by the rank order of those z-scores (rank). We then took its 30 nearest cells in that space, excluding touching cells to limit spillover. On panels from 6 markers up to the full panel, we measured:
- physical closeness;
- how well neighbours predict a cell's surroundings beyond cell type, and whether that transfers across patients;
- kNN enrichment of marker-high cells, with and without the scored marker;
- same-type share;
- cLISI and iLISI.

For clinical value, we built k-means cell states for each encoding, described each patient by state composition, and predicted subtype, receptor status, grade and survival with cross-validation. All comparisons used patient-bootstrap CIs and decision rules fixed before the run. The Jackson runs reused the Danenberg code unchanged.

**Results.** Look-alike cells were only 9–15% closer in tissue than random cells in Danenberg (13–23% in Jackson), mostly because they were the same cell type, and borrowing surroundings from look-alikes in other patients almost failed (R² ≤ 0.045). Rank beat intensity on kNN enrichment from about 24 markers (+0.14 at 31 markers in Danenberg, +0.12 in Jackson). It did so for 35 of 36 and 29 of 32 markers, with Ki-67 losing in both cohorts. With 6 markers rank lost clearly (−0.62 and −0.83). About half the gain disappeared when the scored marker was left out, and the gain did not follow how marker-high cells cluster in the tissue. Rank lowered the same-type share by about 7 points at every panel size and raised cLISI and iLISI by similar amounts. Clinically, rank never beat cohort-normalised intensity and lost HER2 (AUC 0.94 → 0.78). Its only gain, over per-image-normalised intensity in Jackson (34–61% of the gap closed), came from more comparable cell states rather than recovered marker levels, and it did not replicate in Danenberg.

**Discussion.** In CyTOF, profile neighbours are best read as cells in a similar state, mostly of the same type, not as spatial neighbours. Inferring tissue context for CyTOF cells from look-alike IMC cells is not supported, so that idea was dropped. Rank is a legitimate encoding with a clear trade-off. It groups cells that share their top markers more tightly, but groups lineages less well and gives blurrier neighbourhoods. It suits panels of about 24 markers or more, not small ones. It is not a route to better clinical features, and it loses magnitude signals such as HER2 amplification. Its one niche is data that must be normalised per sample, where rank-based states can be more comparable across patients; so far this holds in one of two cohorts. Both cohorts are breast cancer, and a PDAC IMC replication is waiting on data access.

## Where things stand (2026-10-14)
**Can a generative image model give trustworthy virtual cells?** Not in its current form.
- **The kill test went against the image models** (items 22–25).
  - A per-cell regression on the same channels beats the image models for 10 of 11 markers, at about 2× the R4. This holds in all 3 replicates.
  - Within Danenberg, the 11 shared markers are predicted at a median R4 of 0.72–0.74 from each other plus the nuclear channels.
  - Training per-cell models on both cohorts removes most of the cross-cohort gap.
  - Spatial context adds ≤ 0.06 R² anywhere.
  - Virtual tumour-state markers carry no survival information.
- **One exception (2026-10-14, item 27):** ImmuVis trained on Danenberg beats the cell models on Danenberg → Jackson (cell-level r 0.44 vs 0.41; 5 markers won, 3 lost). Trained on Jackson, the same recipe loses on Jackson → Danenberg (0.26 vs 0.43).
  - So "image models lose" holds in one direction only, and rests on one ImmuVis run per direction.
  - In absolute error, no Jackson-trained model beats a constant prediction (item 26).
- **The R5 rung is not validated.** It is the rung meant to measure "information beyond the rest of the panel", and its positive controls failed. The fix is known: residualise on image × state fixed effects, and build the controls from the same fold models R5 uses.
- **The plan's decision table points to PIVOT-A** ("virtual staining = panel redundancy"), or to stopping the image-model zoo. The go/no-go gate (GK) is not yet formally decided.
- **ERα caveat for Part 1.** E1/E2's "ERα best imputed (between-image r 0.82)" was measured in a saturated target space (the two cohorts use different ER antibodies), so it largely reflects background. Recheck it before citing. ERα's within-image R4 is fine (0.80–0.84).

- **Epigenetic virtual cells (2026-10-22, items 28–31).**
  - H3K27me3 can be imputed within Jackson, by the cell table as well as by ImmuVis.
  - Cross-cohort validity is unproven: Danenberg passes only under a post-hoc rule, and Keren's measured mark doesn't reproduce Jackson's relations.
- **R5 Phase 0 gate failed** (`r5-validate`, 116 failures: the leak controls are > 0.01). Phase 1 stays blocked; the fixes are listed in `Results/r5-validate/summary.md`.
  - **Update 2026-10-22 (v2, five fixes):** down to 11 failures.
    - The shuffled, shifted and kNN controls now pass. Power: 66 / 66 cases detected at α 0.1. Identity ≥ 0.97 × residual share.
    - Only the state-mean control fails: up to +0.033 on R5_own and +0.024 on R5_table; R5_cell is clean.
    - Likely cause: the base LightGBM underfits (untested). Phase 1 is still blocked.

**Open next steps** (updated 2026-10-30; `future_plan.md` §0.3):
1. `bayes-impute` (§18.6.1): simulation-based calibration first, then scenarios S1–S4. The comparators are quantile LightGBM (S2) and TabPFN at k = 5 (S3).
2. Fix the R5 state-mean control in `r5-validate`, then decide GK.
3. Run ImmuVis seed 2 and export its variance, so the image model gets a distribution too.
4. TabPFN cell typing (§18.8 task 2).
5. Find an IMC cohort with H3K27me3 measured, for a real external test of epigenetic imputation.

## Experiments in order
1. **Jun–Aug 2025, Danenberg exploration:** compared normalisations, PCA/UMAP and an XGBoost cell-type classifier with SHAP. Exploratory, with figures only and no recorded conclusion.
2. **Sep–Oct 2025, within-cohort pilots** (one ResNet-UNet per marker, 4-fold CV, 9 markers, each cohort alone): only panCK was predicted well (r 0.49 Danenberg, 0.64 Jackson). Every other marker had r ≤ 0.16.
3. **Nov 2025, Jackson epigenetic marks** (H3K27me3, pHH3) predicted from the other channels: failed, with negative correlations (−0.07 to −0.21).
4. **Dec 2025, marker correlations and stain-area statistics** in both cohorts: used to pick the 11 markers shared by Jackson and Danenberg.
5. **Dec 2025, within-cohort models on the shared markers only:** failed, with r between −0.17 and 0.16.
6. **Dec 2025–Jan 2026, first Jackson → Danenberg transfer** (11 per-marker models, residual + PCA inputs): the first working result, with mean r 0.29 and FID 252 (ERα 0.58, panCK 0.57, HER2 0.45).
7. **May 2026, model variants** (pretrained first layer; joint model without PCA): r 0.28 and 0.23. Neither beat the per-marker baseline.
8. **Jun 2026, HyperConv test run** (joint model with a marker-aware stem, 10 epochs): mean r 0.30 and FID 234, the best so far but undertrained.
9. **Jul 2026, preprocessing comparison** (joint model without HyperConv, up to 50 epochs): residual, gamma and ImmuVis-style gave r 0.255, 0.274 and 0.284, and FID 236, 202 and 213.
10. **Jul 2026, abundance vs structure, E1** (residual arm; ERα, HER2, panCK): no marker was level-only. ERα had between-image r 0.82 and within-image r 0.52.
11. **Jul 2026, abundance vs structure, E2** (3 preprocessings × 11 markers, patient bootstrap): predictability tiers were stable and tracked signal quality. The low-pass filter inflated between-image r for weak immune markers.
12. **27 Sep 2026, `danenberg-cells-prep`:** built the Danenberg cell table, masks, contact graph, clinical table and frozen patient split on Kaggle.
13. **27 Sep 2026, `profile-vs-physical`** (Danenberg): look-alike cells are only slightly closer in tissue, mostly because they share a cell type, and cross-patient transfer of surroundings is near zero. Rank costs 7 points of type purity at every panel size.
14. **28 Sep 2026, `knn-enrichment-imc`** (Danenberg): rank beats intensity from about 24 markers (+0.14 at 31) and loses at 6 (−0.62). Half the gain is built in.
15. **28 Sep 2026, Jackson replication** (`jackson-cells-prep`, `profile-vs-physical-jackson`, `knn-enrichment-jackson`): most results replicated, with effects 1.5–2× larger. Rank's closeness never caught up with intensity.
16. **28 Sep 2026, `lisi-imc`** (both cohorts): rank has higher cLISI and iLISI, so its neighbourhoods are blurrier rather than batch-corrected.
17. **29 Sep 2026, `rank-tradeoff-markers`:** per marker, rank raises enrichment for most markers (35/36 Danenberg, 27/32 Jackson with CIs excluding 0) and lowers the same-type share for every marker.
18. **29 Sep 2026, `rank-utility-clinical`** (pre-registered): rank gives no clinical gain over cohort-normalised intensity, loses HER2 and adds nothing to survival. Tumour-state entropy is not prognostic.
19. **4 Oct 2026, `rank-recovery`:** rank closes 34–61% of the per-sample-normalisation gap in Jackson, only through cell-state composition, and not in Danenberg. The claim that rank recovers marker levels fails.
20. **8 Oct 2026, `cell-table-map`** (both cohorts, full panels):
    - Immune cells touch tumour cells about half as often as chance in about 90% of images.
    - Most markers are 40–90% predictable from the cell's own other markers. Neighbours add at most about 6% R².
    - Jackson tumour-state markers imputed into Danenberg add no survival information.
21. **9 Oct 2026, `cell-table-followup`:**
    - Neighbour effects within each cell class stay small (ΔR² ≤ 0.05).
    - A known effect is recovered: hypoxia grows away from vessels, r 0.10.
    - H3K27me3 follows cell identity (R² 0.63 vs 0.19 from nuclear content) and rises with the cell's own CA-IX (r 0.20).
22. **9 Oct 2026, `kill-test`** (existing image checkpoints vs cell-table baselines on Danenberg cells):
    - A per-cell regression on the Jackson table beats the image models for 10 of 11 markers (e.g. panCK R4 0.84 vs 0.25–0.54).
    - The R5 rung failed its positive controls.
    - Danenberg ERα saturates in the model's target space (the two cohorts use different ER antibodies), which undercuts the earlier ERα results.
23. **9 Oct 2026, `model-comparison`** (table model, MLP, GCN, GraphSAGE and ResNet-UNet; Jackson → Danenberg):
    - Every per-cell model beats the ResNet-UNet by about 2× (median R4 0.42–0.46 vs 0.19–0.25).
    - Within Jackson, graph context adds a little.
    - Across cohorts, a plain MLP is best. Item 24 shows this was a tie.
24. **9–10 Oct 2026, `model-comparison-reps`** (2 more replicates, each with a new Jackson patient split and new seeds):
    - The 2× gap between per-cell and image models holds in all 3 replicates.
    - The GCN is again the weakest per-cell model.
    - Table, MLP and GraphSAGE are tied: median R4 0.63–0.64 in Jackson and 0.44–0.47 in Danenberg, which is within the replicate spread.
    - Neighbour context never hurts within Jackson. It gives small, consistent gains only for CD45, SMA and Ki-67 or panCK.
25. **11 Oct 2026, `train-test-grid`** (per-cell models; train on Jackson, Danenberg or both × test on held-out patients of each; 3 replicates):
    - Within Danenberg, the 11 shared markers are highly redundant at the cell level: median R4 0.72–0.74 from the other 10 markers plus nuclear channels.
    - The cohort-transfer gap is concentrated in a few markers. The median per-marker loss is 0.06–0.11 for Jackson → Danenberg, mostly SMA (sign flips), CD68 and CD20. It is 0.14–0.17 for Danenberg → Jackson.
    - Pooling both cohorts closes nearly all of the gap (median paired difference vs own cohort 0.00 to −0.02). The cost is small and marker-specific; SMA pays it in both cohorts.
    - Spatial context within Danenberg helps only CD3 and CD68, by ≤ 0.04.
    - The image models remain far below every per-cell model.
26. **12 Oct 2026, `cross-cohort-metrics`** (main figure: Jackson → Danenberg, one set of predictions scored three ways; `Results/cross-cohort-metrics/fig_cross_cohort_metrics.png`):
    - Scores are means over 10 markers on the Danenberg test patients. Table / MLP / GraphSAGE vs ResNet-UNet:
      - pooled pixel r (ImmuVis-style): 0.31–0.32 vs 0.26;
      - per-image pixel r (VirTues-style): 0.25–0.26 vs 0.22;
      - R4: 0.45–0.47 vs 0.26.
    - Per marker, the per-cell models win on 9 of 10 markers for R4 and per-image r, and 7 of 10 for pooled r. SMA is the only loss.
    - **The pixel metrics compress the gap.** Even the true cell means painted into the masks score only 0.53 pooled and 0.38 per-image, because pixel r rewards within-cell texture and the background around cells.
    - A median over markers misleads here: on per-image r it puts the image model ahead. Report the mean (as the papers do) together with paired per-marker calls.
    - **GCN** (`cross-cohort-metrics` v2): 0.29 / 0.23 / 0.37. It is the weakest per-cell model but still ahead of both image models on R4.
    - **EVA, zero-shot** (`eva-zeroshot`; Liu 2026; its pretraining includes neither cohort): 0.28 / 0.26 / 0.30.
      - It edges out the ResNet-UNet on all three metrics.
      - On per-image pixel r it ties the best per-cell models and wins more markers than it loses (5–6 vs 3).
      - On R4 the per-cell models beat it on 9 of 10 markers.
      - So the pixel metrics can reverse the ranking that R4 shows.
    - **ImmuVis trained on Jackson** (`immuvis-jackson-train` v3 → `immuvis-jackson` v2, 2026-10-13): 0.27 / **0.31** / 0.26.
      - Getting it to run: the first runs died from a host-RAM leak (about 50 MB per step) at about 8 s per step. With cuDNN disabled (`immuvis-profile` v2) it ran at 0.53 s per step with flat RAM. It trained for 3 h from scratch on 397 Jackson images; validation pixel r was 0.646.
      - **Per-image pixel r (VirTues metric) makes ImmuVis the best Jackson-trained model.** It beats every cell model on 8–9 of 10 markers and equals the Danenberg-trained LightGBM (0.31).
      - **At the cell level it ties the ResNet-UNet** (0.26), and LightGBM beats it 6 / 1.
      - It exceeds the painted true-cell oracle on per-image r for HER2, CK8/18 and pan-CK, which is direct evidence that pixel r rewards texture rather than which cells are high.
      - It is strong on the epithelial markers (cell-level r: pan-CK 0.81, HER2 0.65) and weak on the immune markers (CD3 0.17, CD68 0.08).
    - **Normalisation control** (`cross-cohort-metrics` v3, 2026-10-13): the same cell models, with Danenberg inputs scaled using Jackson's statistics instead of Danenberg's own.
      - LightGBM is nearly unchanged: R4 0.45 → 0.43.
      - The MLP and the graph models lose 0.07–0.11 R4.
      - The main figure now uses these Jackson-scaled versions, since no image model gets cohort-level adaptation.
    - **Final figure values** (Jackson-scaled cell models; pooled / per-image / cell-level): LightGBM 0.31 / 0.25 / 0.43, MLP 0.31 / 0.24 / 0.40, GCN 0.22 / 0.19 / 0.27, GraphSAGE 0.24 / 0.20 / 0.34; U-Net 0.26 / 0.22 / 0.26, ImmuVis 0.27 / 0.31 / 0.26, EVA 0.28 / 0.26 / 0.30; Danenberg-trained LightGBM 0.41 / 0.31 / 0.66.
    - **Errors added (2026-10-14; the figure is now 2 × 3, r on top and RMSE below).** RMSE is in the target space: pooled pixel, per-image pixel and cell level.
      - ImmuVis and EVA were re-run (`immuvis-jackson` v3, `eva-zeroshot` v4) to save outputs mapped to the target space using only Jackson-train statistics. Their r values were unchanged.
      - The new reference line is a constant prediction at the Jackson-train mean.
      - **No Jackson-trained model beats the constant on any error metric.** Constant 0.191 / 0.181 / 0.139 (pooled / per-image / cell). The U-Net ties it on pixels (0.191 / 0.182) and is worse at the cell level (0.147). The others are worse: ImmuVis 0.214 / 0.198 / 0.199; LightGBM 0.209 / 0.200 / 0.188; EVA 0.198 / 0.184 / 0.165.
      - Only the Danenberg-trained LightGBM beats it (0.165 / 0.156 / 0.080).
      - Why: r ≈ 0.3 leaves at most about 5% RMSE to gain, and level offsets (25–40% of squared error) use it up. Models that spread their predictions widely (ImmuVis) do worst; models that shrink toward the mean (U-Net) land at the constant.
      - So the errors rank by calibration, while cell-level r ranks by which cells are high. Cross-cohort models get the second partly right and the first not at all.
    - **Comparison with the ImmuVis paper** (pooled Pearson 0.77, MSE 0.068 on a held-out head & neck cohort; `Results/cross-cohort-metrics/summary.md`):
      - Not a contradiction. The paper pretrains on 26–27 cohorts (including ours), uses 37 input markers, scores a Butterworth-denoised target scaled by each cohort's own limit (Danenberg 2.9, Jackson 2.3), and compares only against VirTues and EVA.
      - Our MSE in the paper's aggregation is 0.036–0.051, below theirs (different scale). Our ImmuVis reaches pixel r 0.62–0.65 within cohort in its own space.
      - D→J agrees with the paper: ImmuVis beats EVA on everything. Our negative results are against baselines the paper did not run: cell regression and a constant.
      - No public ImmuVis weights exist to run the published model.
27. **13 Oct 2026, `cross-cohort-metrics-jackson`** (reverse direction, Danenberg → Jackson; 66 Jackson test patients):
    - Means over 10 markers (pooled / per-image / R4):
      - LightGBM trained on Danenberg: 0.34 / 0.27 / 0.41;
      - EVA zero-shot: 0.21 / 0.20 / 0.18;
      - LightGBM trained on Jackson (reference): 0.44 / 0.34 / 0.69;
      - oracle: 0.55 / 0.42.
    - The cell models beat EVA on all three metrics. Cohort scaling barely matters in this direction (≤ 0.02).
    - **ImmuVis trained on Danenberg** (`immuvis-danenberg-train` → `image-models-jackson`, 2026-10-14). It used the same recipe and 3 h budget as the Jackson model, on 494 train images; validation pixel r was 0.622.
      - **It is the best Danenberg → Jackson model on all six metrics:** 0.35 / 0.29 / **0.44** r and RMSE 0.184 / 0.173 / **0.128**.
      - It beats LightGBM (0.41) on 5 of 10 markers at the cell level and loses 3, including CD3 (0.05 vs 0.52).
      - Unlike in J→D, the models here do beat the constant (0.195 / 0.185 / 0.154), but only slightly (ImmuVis by −0.026 at the cell level; LightGBM by −0.013). EVA does not.
      - **The image model's transfer is asymmetric; the cell models' is not.** ImmuVis scores 0.26 J→D and 0.44 D→J at the cell level, while LightGBM scores 0.43 and 0.41. The gain is in CD45, CD20, CD68, CK5 and CK8/18.
      - Candidate causes are more training images and brighter Danenberg CD45/CD20 saturating the fixed clip for a Jackson-trained model. They are untested, and there is one run per direction.
28. **20–21 Oct 2026, `r5-validate`** (R5 Phase 0, both cohorts):
    - Masks, power (α = 0.1 detected in 43/44 cases), noise ceiling and spillover all passed.
    - The gate failed: the shuffled and shifted controls leak image- and state-level brightness (up to +0.20).
29. **21–22 Oct 2026, `epi-immuvis-train` → `epi-immuvis-infer`:**
    - ImmuVis trained on Jackson to predict H3K27me3 and pHH3 from the 14 shared channels. Validation pixel r is 0.65 and 0.50.
    - Also produced leave-anchor-out predictions and EVA zero-shot predictions. EVA's tokens for the marks are identical.
30. **22 Oct 2026, `epi-relations` v1–v3** (pre-registered 2026-10-21; Keren block pre-registered 2026-10-22). Results are in the epigenetic table above:
    - accuracy;
    - 33 / 16 reliable Jackson relations, checked in the Jackson test patients, Danenberg and Keren;
    - post-hoc Amendment 1 for anchors that don't transfer.
31. **22 Oct 2026, Keren 2018 added** (`keren2018` dataset, `keren-cells-prep`, `keren-epi-within`):
    - 197,678 cells, 40 TNBC patients. The masks match the published table exactly.
    - Within-Keren H3K27me3, partial r: 0.23 from the shared inputs, 0.54 from all channels, 0.28 without the mass neighbours of Yb172.

32. **22 Oct 2026, `kronos-zeroshot` and `cell-type-shap`:**
    - KRONOS (ViT-S, no IMC in pretraining) zero-shot cell typing, replicate 0.
    - XGBoost + TreeSHAP drivers of cell typing and imputation.
    - `r5-validate` v2 (fixes; 11 failures, state-mean control only).
33. **25 Oct 2026, 32 µm patch level for the cross-cohort figure:**
    - Kernels: `cross-cohort-metrics-patch`, `cross-cohort-metrics-jackson-patch`, `unet-patch-sums`, `immuvis-jackson-patch`, `eva-zeroshot-patch`, `image-models-jackson-patch`.
    - Every rerun reproduces the earlier pixel and cell values (exactly, or within 5e-6 for ImmuVis on GPU).
    - Summary code: `summarise_core.run(PATCH_CSVS=...)`.
34. **25–27 Oct 2026, `kronos2-zeroshot` → `kronos2-probes`:**
    - KRONOS2: ViT-B/16, pretrained on SPM-56M (fluorescence only). CK5 and CK8/18 were added through its text-embedding marker route.
    - The embedding run hit an out-of-memory error in the pooled probes. The probes were rerun on CPU from the saved embeddings (dataset `kronos2-embeddings`).
    - Replicates 0–2, with KRONOS v1 re-scored on the same splits. Summary: `Results/kronos2-zeroshot/summary.md`.
    - Kaggle changes found along the way:
      - Python 3.13 image;
      - `imagecodecs==2024.9.22` is gone, so a fallback is needed;
      - at most 2 GPU batch sessions at a time.
35. **27 Oct 2026, correlation vs distribution match** (local, `Results/cross-cohort-metrics/realism_vs_correlation.py`):
    - Plots r against W2 between the predicted and true marginals, at pixel, patch and cell level, in both directions.
    - Cell-level W2 is exact. Pixel and patch W2 is a moment bound (90–97% of the exact value).
    - It is a diagnostic, not "realism": mean predictors score well on it without producing realistic cells.
36. **27 Oct 2026, `prob-impute`** (CPU): quantile LightGBM and residual sampling, scored with CRPS, coverage, PIT and W2.
    - Calibrated within a cohort, not across cohorts (coverage 0.57 / 0.44).
    - Mixture likelihood check for the Bayesian arm. The Bayesian arm (§18.6.1) was pre-registered the same day.
37. **29 Oct 2026, `harmonise`** (CPU, pre-registered): five input corrections and two oracles. Not adopted. The level shift is in the target marker. `Results/harmonise/summary.md`.
38. **29 Oct 2026, `tabpfn-cells`** (T4, `tabpfn==9.2.0`, v2 weights, 10k context): ties full LightGBM, and the few-shot runs are the strong result. `Results/tabpfn-cells/summary.md`.
39. **30 Oct 2026, LISI of virtual cells** (local, `Results/cross-cohort-metrics/lisi_virtual_cells.py`):
    - iLISI against measured cells from the other half of the test patients; cLISI on 6 coarse types.
    - 5,000 cells per side, 3 sampling replicates.
    - Cohort labels are read with duckdb, because local pyarrow fails on `cells.parquet`.

40. **4–5 Nov 2026, `novae-cells`** (2×T4, Novae 1.1.2 trained from scratch on both cohorts, pre-registered in `future_plan.md` §18.9):
    - Both claims fail. Its niche latent adds ±0.002 macro-F1 to Table + nbr; alone it reaches 0.39–0.46.
    - Its alignment mixes the cohorts but blurs cell types (cLISI +0.40).
    - Training was shallow and partly collapsed (2/3 `[cohort]` fits at loss ln 512). `Results/novae-cells/summary.md`.

41. **5 Nov 2026, `pixel-features` + `multiscale-audit`** (CPU; Track B0–B2, `future_plan.md` §18.10):
    - Sub-cellular pixel features (membrane vs centre, texture, shape) add ≤ +0.007 R4 to imputation and nothing in transfer.
    - They do help within-cohort typing: fine +0.02, coarse D→D +0.010; stable under 2-px erosion and beyond the outside band.
    - Patch / typed-edge features (B2) fail.
    - Endothelial and myeloid cells are heterophilous. `Results/multiscale-audit/summary.md`.
42. **5 Nov 2026, `label-counts`** (CPU; §16 arms A/B, pre-registered):
    - Counts of neighbours by *predicted* type add nothing (round 2 hurts D→J).
    - Counts from *revealed true* labels help: 20% revealed gives +0.02–0.03 macro-F1 in transfer and +0.07 / +0.15 on fine labels (partly label leakage).
    - `Results/label-counts/summary.md`.

43. **6 Nov 2026, `scale-lisi`** (local, on saved outputs; same cells, draws and LISI code as `novae-cells`, reproduced within 6e-4):
    - iLISI (cohort) vs cLISI (coarse type) for 13 spaces at pixel, patch and cell scale; baseline = cohort-scaled intensity.
    - No space beats intensity on both axes. Patch spaces (incl. KRONOS / KRONOS2) separate cohorts more; Novae mixes more but blurs types most.
    - Cohort scaling is the one step that matters (iLISI 1.06 → 1.46). `Results/scale-lisi/summary.md`.

44. **7 Nov 2026, `epi-emt`** (local CPU; pre-registered `future_plan.md` §19):
    - H3K27me3 vs the epithelial → mesenchymal axis in tumour cells, as a within-patient rank partial correlation.
    - The pre-registered rules pass in Basel (−0.34), Zurich (−0.34) and Keren (−0.06).
    - Per marker, the effect is keratin-driven (+0.4 to +0.5, robust to a generic per-cell signal factor). Vimentin ≈ 0, and Twist / Slug go the opposite way.
    - Keren's weak signal is vimentin-driven and not robust to the neighbourhood adjustment. `Results/epi-emt/summary.md`.

Details: `Results/<experiment>/summary.md` and `Results/c2s-imc-summary.md`. The previous long version is in git: `git show HEAD:session_summary.md`.
