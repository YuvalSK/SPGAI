# Cross-cohort inference, Jackson → Danenberg: one set of predictions, three correlations and three errors (main figure)

**Runs (Kaggle, 2026-10-12/13, T4 GPU):**
- `cross-cohort-metrics` v3 (2026-10-13): per-cell models plus the normalisation control; 18 min; no failed blocks. Notebook `Scripts/CrossCohortMetrics.ipynb`. The v2 values are reproduced exactly.
- `eva-zeroshot` v3: EVA foundation model, zero-shot; 17 min. Notebook `Scripts/EvaZeroShot.ipynb`. Outputs are in `Results/eva-zeroshot/`.
- `immuvis-jackson-train` v3 (2026-10-13): ImmuVis trained from scratch on Jackson; 190 min, 19,331 steps, no NaN steps. Notebook `Scripts/ImmuVisJacksonTrain.ipynb`. Logs in `Results/immuvis-jackson-train/`.
- `immuvis-jackson` v2 (2026-10-13): that checkpoint scored on the Danenberg test images; 32 min; no failed blocks. Notebook `Scripts/ImmuVisJackson.ipynb`. Outputs in `Results/immuvis-jackson/`.
- **Error reruns (2026-10-14), for the RMSE row:** `immuvis-jackson` v3 (34 min) and `eva-zeroshot` v4 (20 min) also save a target-scale copy of each output. Their r values reproduce v2 / v3 exactly, for every marker.

**Combined locally:**
- `summarise.py`: mean and median over markers, plus paired per-marker calls.
- `make_figure.py` → `fig_cross_cohort_metrics.png` / `.pdf`.

**Large outputs:**
- Per-cell predictions:
  - `IMC data/Danenberg2022/cross-cohort-metrics/percell_predictions_test.npz`
  - `IMC data/Danenberg2022/eva-zeroshot/percell_pred_eva.npz`
  - `IMC data/Danenberg2022/immuvis-jackson/percell_pred_immuvis.npz`
- ImmuVis checkpoint: `IMC data/Jackson2020/immuvis-jackson-train/immuvis_jackson_best.pth`.

**Setting:**
- Every model is scored on the 130 frozen Danenberg test patients (228,511 cells, 145 images).
- All models except the reference were trained on Jackson. EVA was trained on neither cohort.
- **Markers:** the 11 shared markers, each predicted with its own channel hidden. The inputs and preprocessing differ by model (table below).
- ERα is left out of the summaries because it saturates in target space.
- **Truth:** the image model's target space, `clip(arcsinh(x/5) / p99_Jackson, 0, 1)`. The Jackson p01 is 0 for every marker.

**Models in the figure:**

| Figure label (CSV name) | Algorithm | Input | Input preprocessing |
|---|---|---|---|
| U-Net (`ResNet-UNet immuvis`) | ResNet50 encoder (ImageNet init) and U-Net decoder. One network serves all 11 tasks. Trained on Jackson (`kill-test` checkpoint) | Pixels: the 10 other markers (no DNA or H3) | arcsinh(x/5) → Butterworth low-pass (order 2, 0.2) → p1–p99 min-max with **Jackson training** percentiles |
| ImmuVis (`ImmuVis`) | ImmuVis-Conv masked autoencoder (Uchal 2026; code from github `dav3794/multiplex-image-model`). Trained from scratch on 397 Jackson training images with the paper's channel and spatial masking and β-NLL loss. 128-px crops, batch 16, 180 min. Best Jackson validation pixel r 0.646 (held-out channel), reached at step 19,000. Prediction: the encoder sees the 13 other channels; the decoder is queried for the hidden marker. Tiles are 128 px with 32 px of context | Pixels: DNA1, DNA2, H3 + the 10 other markers | arcsinh(x/5) → Butterworth low-pass (0.2) → clip(0, 2.3)/2.3, fixed (the ImmuVis recipe) |
| EVA (`EVA`) | EVA foundation model (Liu 2026), 122 M parameters, zero-shot; pretrained on neither cohort | Pixels: DAPI (mean of DNA1, DNA2), H3 + the 10 other markers | arcsinh(x/5) → **per-image**, per-channel p0.5–p99.5 min-max |
| LightGBM (`Table [Jackson-scaled inputs]`) | LightGBM, 250 trees, 31 leaves, lr 0.06; 300k Jackson training cells | Cell means: the 10 other markers + H3, DNA1, DNA2 + log area | arcsinh(x/5) → robust scaling ((x − median) / (p95 − p5)) with **Jackson** statistics, applied to the Danenberg inputs too |
| MLP (`MLP [Jackson-scaled inputs]`) | 2 hidden layers × 128, ReLU, dropout 0.1; Adam, early stopping on Jackson val | Same as LightGBM | Same as LightGBM |
| GCN (`GCN [Jackson-scaled inputs]`) | 2-layer GCN (Kipf normalisation, self-loops), 128 hidden. The cell and its neighbours are averaged with shared weights | Same, plus the 10 nearest cells (symmetrised kNN graph) | Same as LightGBM |
| GraphSAGE (`GraphSAGE [Jackson-scaled inputs]`) | 2-layer GraphSAGE (mean aggregator), 128 hidden. Separate weights for the cell and its neighbours | Same, plus the 10 nearest cells | Same as LightGBM |
| LightGBM, green (`Table (Danenberg-trained)`) | As LightGBM, trained on the Danenberg train patients | As LightGBM | Danenberg statistics. The target is the target-space truth itself |
| Dashed line, "True cell values" (`Painted true cell means`) | An oracle, not a model: the measured target marker, averaged per cell and painted into the masks | | |

Notes on the table:
- **The oracle** uses the answer, so it is the ceiling for any one-value-per-cell map. Image models can exceed it on pixel r, and ImmuVis does for three markers.
- **Splits and seeds:** all Jackson-trained per-cell models use the replicate-0 split and seeds of `train-test-grid`. With Danenberg's own scaling, their harmonised R4 reproduces that run exactly (median 0.456 / 0.488 / 0.486 / 0.506).
- **The figure shows the Jackson-scaled versions,** so that no model gets cohort-level adaptation to Danenberg (see the normalisation control).
- **Outputs and scale:**
  - The per-cell models predict in harmonised units, which are mapped back to target space with Jackson statistics.
  - The U-Net and ImmuVis predict in target space or a fixed scale.
  - EVA's output stays in its own per-image scale.

**Metrics:**
- **(a) Pooled pixel r** (ImmuVis): per marker, over all pixels of all test images.
- **(b) Per-image pixel r** (VirTues): r within each image, then the mean over images.
- **(c) Cell-level r within image** (CSV name `R4`): per-cell r after subtracting each image's mean.
- **(d) Pooled pixel RMSE:** per marker, over all pixels of all test images, `sqrt(Σ(Sgg − 2Sgp + Spp) / Σn)` from the same per-image sums.
- **(e) Per-image pixel RMSE:** RMSE within each image, then the mean over images.
- **(f) Cell-level RMSE:** over all test cells, predicted vs true cell value. Image means are *not* removed, so level offsets count.

**Errors need every output in the target space, mapped without test data:**
- The per-cell models and the U-Net already predict there.
- **ImmuVis:** its output is clip(filtered arcsinh, 0, 2.3)/2.3, so `2.3 × output` is arcsinh, then mapped with the fixed Jackson p99.
- **EVA:** its output is in per-image [0, 1] units. It is mapped back as `lo + output × (hi − lo)`, where lo / hi are the *median* per-image p0.5 / p99.5 of that marker over the 397 Jackson train images (`Results/eva-zeroshot/source_reference_jackson.csv`). A test image's own percentiles would leak the answer.
- **Dotted line, "Constant (training mean)":** predicts the Jackson-train mean target value of the marker everywhere: the pixel mean for (d, e), the cell mean for (f). It is the reference any useful model must beat on RMSE.

**How each is computed:**
- **Per-cell models in (a) and (b):** each cell's prediction fills its mask pixels, and the background is 0. Pixel r is computed exactly from per-image sums. A direct pixel check on 2 images matched within 3e-5.
- **EVA:**
  - Inputs and scaling as in the table. Arcsinh was chosen over linear on 20 Jackson val images (0.163 vs 0.159).
  - It predicts in 224-px tiles, with the target channel zeroed and masked.
  - The output is scored as is, against the same target-space truth.
- **Image models' cell-level scores:** their cell means come from their pixel maps.
- **Summary:** the mean over the 10 markers, with 95% patient-bootstrap CIs (1000 draws; the same draws for every model, metric and marker).

## Mean over 10 markers [95% CI] (the models in the figure)
| Model | (a) Pooled pixel r | (b) Per-image pixel r | (c) Cell-level r |
|---|---|---|---|
| U-Net, Jackson-trained | 0.26 [0.24, 0.28] | 0.22 [0.21, 0.23] | 0.26 [0.23, 0.28] |
| ImmuVis, Jackson-trained | 0.27 [0.24, 0.31] | **0.31 [0.30, 0.33]** | 0.26 [0.23, 0.31] |
| EVA, zero-shot | 0.28 [0.27, 0.29] | 0.26 [0.25, 0.27] | 0.30 [0.28, 0.32] |
| LightGBM, Jackson-trained | 0.31 [0.29, 0.32] | 0.25 [0.24, 0.26] | **0.43 [0.41, 0.45]** |
| MLP, Jackson-trained | 0.31 [0.29, 0.32] | 0.24 [0.23, 0.25] | 0.40 [0.37, 0.42] |
| GCN, Jackson-trained | 0.22 [0.21, 0.24] | 0.19 [0.18, 0.19] | 0.27 [0.25, 0.30] |
| GraphSAGE, Jackson-trained | 0.24 [0.23, 0.26] | 0.20 [0.19, 0.21] | 0.34 [0.32, 0.36] |
| LightGBM, Danenberg-trained (reference) | 0.41 [0.39, 0.42] | 0.31 [0.30, 0.33] | 0.66 [0.64, 0.68] |
| Oracle: true cell values painted | 0.53 [0.51, 0.54] | 0.38 [0.37, 0.39] | 1 |

The exact values and CIs, and the cohort-scaled versions of the cell models, are in `metrics_summary.csv`.

## Errors: mean RMSE over 10 markers [95% CI] (lower is better; target space [0, 1])
| Model | (d) Pooled pixel RMSE | (e) Per-image pixel RMSE | (f) Cell-level RMSE | (f) vs constant: better / n.s. / worse markers |
|---|---|---|---|---|
| U-Net, Jackson-trained | 0.191 [0.183, 0.199] | 0.182 [0.175, 0.190] | 0.147 [0.139, 0.155] | 3 / 1 / 6 |
| ImmuVis, Jackson-trained | 0.214 [0.197, 0.233] | 0.198 [0.186, 0.210] | 0.199 [0.168, 0.233] | 2 / 2 / 6 |
| EVA, zero-shot | 0.198 [0.189, 0.207] | 0.184 [0.176, 0.192] | 0.165 [0.156, 0.172] | 1 / 3 / 6 |
| LightGBM, Jackson-trained | 0.209 [0.199, 0.217] | 0.200 [0.191, 0.209] | 0.188 [0.179, 0.196] | 2 / 1 / 7 |
| MLP, Jackson-trained | 0.199 [0.190, 0.207] | 0.191 [0.183, 0.199] | 0.170 [0.162, 0.177] | 3 / 1 / 6 |
| GCN, Jackson-trained | 0.223 [0.214, 0.233] | 0.210 [0.201, 0.220] | 0.209 [0.199, 0.218] | 1 / 3 / 6 |
| GraphSAGE, Jackson-trained | 0.221 [0.211, 0.231] | 0.210 [0.201, 0.219] | 0.205 [0.196, 0.214] | 2 / 1 / 7 |
| LightGBM, Danenberg-trained (reference) | 0.165 [0.158, 0.172] | 0.156 [0.150, 0.162] | **0.080 [0.075, 0.085]** | 9 / 1 / 0 |
| Constant: Jackson-train mean | 0.191 [0.183, 0.199] | 0.181 [0.174, 0.188] | 0.139 [0.130, 0.146] | |
| Oracle: true cell values painted | 0.153 [0.147, 0.159] | 0.146 [0.141, 0.152] | 0 | |

Source: `metrics_summary.csv`; the comparison with the constant (paired, same bootstrap draws, a call needs the CI to exclude 0) is in `vs_constant.csv`.

**What the errors show:**
1. **No Jackson-trained model beats a constant prediction on any error metric.**
   - The best ones tie it: the U-Net on pixel RMSE (0.191 vs 0.191; difference CI [−0.002, 0.003]).
   - At the cell level, every Jackson-trained model and EVA is worse than the constant (U-Net +0.008 [0.004, 0.014]; ImmuVis +0.060; LightGBM +0.049).
   - Only the Danenberg-trained LightGBM beats it, on 9 of 10 markers (cell RMSE 0.080 vs 0.139).
2. **Why: low r leaves almost no room, and level offsets use it up.**
   - With r ≈ 0.3, even a perfectly calibrated prediction lowers RMSE by only about 5% (√(1 − r²)) below the target cohort's own mean.
   - Level offsets make up 25–40% of each model's squared cell-level error. The models' offsets are about twice the cohorts' own difference in means: mean |bias| over markers is 0.11 for ImmuVis, LightGBM, the MLP and EVA, 0.07 for the U-Net, and 0.06 for the constant.
   - Spread that does not track the truth adds error. ImmuVis's predictions vary more than the truth (SD 0.16 vs 0.12) and score worst; the U-Net shrinks toward the mean (SD 0.08) and lands next to the constant. EVA's mapped output is nearly flat (SD 0.03), so its error is mostly offset.
3. **So the error row ranks by calibration, not by which cells are high.**
   - The U-Net has the lowest error of the Jackson-trained models but the lowest cell-level r of them (0.26).
   - LightGBM has the highest cell-level r (0.43) but one of the highest errors (0.188), because its Jackson-scaled inputs shift its levels on Danenberg.
   - With Danenberg's own input scaling, the cell models' errors fall to the constant's level (MLP cell RMSE 0.147, LightGBM 0.159; `normalisation_control.csv`). They still do not beat it.
4. **For the paper:** RMSE alone would call every cross-cohort model useless, and r alone hides that their absolute levels are off. The two rows together show that cross-cohort models get *which cells are high* partly right (r up to 0.43) but not *how high* (no gain over a constant). Only training on the target cohort gets both.
5. **The reverse direction differs** (`Results/cross-cohort-metrics-jackson/summary.md`).
   - ImmuVis trained on Danenberg is the best Danenberg → Jackson model on all six metrics (cell-level r 0.44 vs LightGBM 0.41). It beats the constant on errors (cell RMSE 0.128 vs 0.154).
   - The cell models score the same in both directions, so the image model's transfer is asymmetric. With one training run per direction, the cause is untested.

## Paired per marker: cell model better / worse than each image model
Source: `per_cell_vs_image_per_marker.csv`; Jackson-scaled cell models; 10 markers; a call needs the CI to exclude 0.

| vs | Metric | LightGBM | MLP | GCN | GraphSAGE |
|---|---|---|---|---|---|
| U-Net | Pooled pixel r | 8 / 1 | 7 / 1 | 3 / 3 | 5 / 3 |
| U-Net | Per-image pixel r | 9 / 1 | 8 / 1 | 4 / 4 | 6 / 3 |
| U-Net | Cell-level r | 9 / 1 | 8 / 1 | 4 / 3 | 6 / 2 |
| ImmuVis | Pooled pixel r | 5 / 2 | 6 / 2 | 4 / 5 | 3 / 4 |
| ImmuVis | Per-image pixel r | **1 / 8** | 1 / 9 | 1 / 9 | 1 / 9 |
| ImmuVis | Cell-level r | **6 / 1** | 7 / 1 | 4 / 5 | 4 / 3 |
| EVA | Pooled pixel r | 3 / 2 | 5 / 3 | 1 / 5 | 3 / 3 |
| EVA | Per-image pixel r | 3 / 6 | 3 / 6 | 0 / 8 | 2 / 6 |
| EVA | Cell-level r | 6 / 2 | 6 / 1 | 3 / 4 | 5 / 2 |

**LightGBM vs ImmuVis, marker by marker:**
- **Cell-level r:** LightGBM wins CD20, CD3, CD45, CK5, CK8/18 and pan-CK. It loses SMA. HER2, CD68 and Ki-67 are n.s.
- **Per-image pixel r:** ImmuVis wins every marker except CD3 (a LightGBM win) and CK5 (n.s.).

## What the figure shows
1. **At the cell level, the per-cell models beat all three image models.**
   - LightGBM reaches 0.43 and the MLP 0.40, against 0.26 for the U-Net, 0.26 for ImmuVis and 0.30 for EVA.
   - LightGBM wins 9 of 10 markers against the U-Net, 6 / 1 against ImmuVis and 6 / 2 against EVA.
2. **Per-image pixel r (the VirTues metric) reverses the order.**
   - ImmuVis is the best Jackson-trained model on it (0.31). It beats every Jackson-trained cell model on 8–9 of 10 markers.
   - It matches the Danenberg-trained LightGBM (0.31), which is trained on the test cohort and reaches 0.66 at the cell level.
   - Judged only by this metric, ImmuVis would look like the best cross-cohort model and as good as training on the target cohort. At the cell level it ties the U-Net and trails LightGBM by 0.17.
3. **Pixel r rewards texture inside and around cells, not which cells are high.**
   - Painting the *true* cell means scores only 0.53 pooled and 0.38 per image.
   - ImmuVis exceeds this oracle on per-image r for HER2 (0.54 vs 0.47), CK8/18 (0.45 vs 0.44) and pan-CK (0.73 vs 0.65).
   - Pan-CK shows the split within one marker: LightGBM wins at the cell level (0.86 vs 0.81), while ImmuVis wins both pixel metrics.
4. **Pooled pixel r compresses the gap.** LightGBM scores 0.31, against 0.26–0.28 for the three image models.
5. **ImmuVis is uneven across markers.**
   - It is strong on the epithelial markers: cell-level r of 0.81 for pan-CK and 0.65 for HER2.
   - It is weak on the immune markers: CD3 0.17, CD45 0.21, CD20 0.14, CD68 0.08.
   - LightGBM scores 0.67 for CD3 and 0.56 for CD45.
   - This spread explains its wide CI (cell-level 0.23–0.31).
6. **Without cohort scaling, the graph models fall back** (GCN 0.27, GraphSAGE 0.34). LightGBM and the MLP keep most of their lead (normalisation control below).
7. **Within-cohort training is still the largest single gain.** The Danenberg-trained LightGBM reaches 0.66 at the cell level and 0.41 pooled.
8. **SMA is the one marker where the image models win.** Cross-cohort transfer of SMA fails for every per-cell model (LightGBM cell-level r −0.06).

## Normalisation control: Danenberg inputs scaled with Jackson statistics (`normalisation_control.csv`)

The same Jackson-trained cell models, with Danenberg inputs scaled using Jackson's median and p5–p95 range instead of Danenberg's own. Like the image models, they then get no adaptation to Danenberg. Values are the mean over 10 markers; Δ = own scaling − Jackson scaling, with a 95% patient-bootstrap CI.

| Model | Pooled pixel r: own → Jackson | Per-image pixel r: own → Jackson | Cell-level r: own → Jackson |
|---|---|---|---|
| LightGBM | 0.315 → 0.308 (Δ 0.007) | 0.253 → 0.252 (Δ 0.001, n.s.) | 0.447 → 0.426 (Δ 0.021) |
| MLP | 0.322 → 0.305 (Δ 0.017) | 0.255 → 0.244 (Δ 0.011) | 0.466 → 0.395 (Δ 0.071) |
| GCN | 0.293 → 0.224 (Δ 0.070) | 0.226 → 0.187 (Δ 0.039) | 0.366 → 0.275 (Δ 0.091) |
| GraphSAGE | 0.314 → 0.243 (Δ 0.072) | 0.248 → 0.203 (Δ 0.045) | 0.454 → 0.341 (Δ 0.113) |

**What it means:**
- **LightGBM barely depends on the cohort scaling.** Its cell-level r drops only 0.02, so its advantage over the image models is not a normalisation artefact.
- **The neural and graph models depend on it more.**
  - The MLP loses 0.07, and GraphSAGE and GCN lose 0.09–0.11.
  - Graph models aggregate neighbour features, so any offset in the inputs compounds.
- **The figure therefore shows the Jackson-scaled versions,** because no image model gets cohort-level adaptation.

## Comparison with the ImmuVis paper (Uchal et al., arXiv 2602.04585 v2)

**Their headline:** Table 1 reports virtual staining on a held-out cohort.

| Model | MSE | Pearson |
|---|---|---|
| ImmuVis-Conv | 0.068 | 0.774 |
| ImmuVis-Swin | 0.061 | 0.778 |
| VirTues | 0.101 | 0.728 |
| EVA | 1.452 | 0.284 |

**Our numbers do not contradict it; they measure a different setting.**

| | ImmuVis paper (Table 1) | This figure |
|---|---|---|
| Test set | IMMUcan head & neck cohort (743 images), held out of pretraining; one central crop per image | The other cohort's test patients (Danenberg or Jackson), whole images |
| Inputs | 37 markers (leave one out of 38) | 13 channels: 10 markers + DNA1, DNA2, H3 |
| Target | Preprocessed: arcsinh → Butterworth low-pass (denoised) → [0, 1] | Unfiltered arcsinh, scaled per marker by the Jackson p99, clipped to [0, 1] |
| Scaling | Each cohort by its own panel-wide p99 (code config: Jackson-Basel 2.3, **Danenberg 2.9**) | One fixed scale for both cohorts (Jackson, 2.3), so no test-cohort statistics |
| Model | Pretrained for 200 epochs on the rest of IMC17M (26–27 cohorts, ~19k training images), including Jackson, Danenberg and two other cohorts from the same consortium as the test cohort (IMMUcan BC2, NSCLC2) | Same architecture and recipe, trained from scratch for 3 h on one cohort (397 or 494 images) |
| Compared against | VirTues and EVA only | Also per-cell regression, a constant prediction and the true-cell-value oracle |
| Pearson | Per marker, pooled over images, mean over markers | Same (panel a), plus per-image and cell-level |
| MSE | Per pixel, mean over images and markers | Reported as RMSE (panels d–f) |

**Errors, in the paper's own aggregation** (per-image MSE, mean over images and 10 markers):
- ImmuVis: 0.051 (J→D) and 0.036 (D→J).
- EVA: 0.047 and 0.041.
- Constant prediction: 0.043 and 0.040.
- All of these are *below* the paper's 0.068. The scales differ, so neither number transfers to the other setting.
- The paper itself is not on one scale. Its Fig. 14 (same zero-shot task, 149 test patches) starts at RMSE ≈ 0.08, i.e. MSE ≈ 0.006.
- EVA's 1.452 in the paper is larger than any MSE possible for predictions inside the [0, 1] target range, so it reflects EVA's output never being mapped to their scale. Here EVA's output is mapped with training-cohort statistics, and its MSE is in the same range as the others.

**"Better than all current methods" means better than VirTues and EVA.**
- On our data, ImmuVis trained on Danenberg beats EVA on all six metrics (Danenberg → Jackson), which agrees with the paper.
- Jackson → Danenberg, ImmuVis beats EVA on per-image r and ties it on pooled r (0.27 vs 0.28), but has the higher error.
- Our negative results come from comparisons the paper did not make: against per-cell regression and against a constant prediction.

**Why their correlation is 0.77 and ours is 0.27–0.35** (the factors were not separated here):
1. **Cross-cohort transfer.** Within a cohort, in the paper's preprocessed space, our ImmuVis reaches a validation pixel r of 0.646 (Jackson) and 0.622 (Danenberg; mean over 11 markers, 13 inputs), within 0.13 of the paper's 0.77. The drop to 0.27–0.35 combines the change of cohort with our unfiltered target (point 2).
2. **Denoised target.** Low-pass filtering removes the pixel noise that no model can predict, which raises r and lowers MSE. Our target keeps it.
3. **Many more inputs:** 37 markers, including close relatives of the target, against 10 markers + 3 nuclear channels.
4. **Pretraining breadth.** The held-out head & neck cohort shares its consortium (and so likely its staining and acquisition) with cohorts in pretraining, which is a milder shift than Jackson ↔ Danenberg.
5. **Per-cohort scaling.** The paper scales each cohort by its own p99. We used Jackson's 2.3 for Danenberg too; ImmuVis's own config uses 2.9, so the brighter Danenberg channels saturate for the Jackson-trained model. This may contribute to the Jackson → Danenberg weakness (see `../cross-cohort-metrics-jackson/summary.md`).

**Public weights:** none were found. The code repository has no checkpoint and the anonymous release returns 403, so the published model cannot be scored on our images. In any case it saw both of our cohorts in pretraining.

**Paper-ready wording (methods / discussion):**
> Our ImmuVis scores are lower than published values because the settings differ, not because the implementation does. Uchal et al. report pooled Pearson 0.77 and MSE 0.068 for a model pretrained on 26–27 cohorts, evaluated on a held-out cohort with 37 input markers against a low-pass-filtered target scaled with that cohort's own statistics, and compared only with other image foundation models. We trained the same architecture and recipe on a single cohort, predicted each marker from 13 shared channels, and scored unfiltered intensities in a fixed scale without test-cohort statistics. Within the training cohort and in the original preprocessed space, our models reach pixel r 0.62–0.65. Across cohorts, pooled r falls to 0.27–0.35, and the MSE (in the paper's aggregation) is worse than predicting the training-cohort mean in one direction (0.051 vs 0.043) and only 10% better in the other (0.036 vs 0.040). Error metrics are therefore only interpretable alongside a constant-prediction baseline, which current virtual-staining benchmarks omit.

## Caveats
- **This ImmuVis is not the published model.**
  - It is the ImmuVis architecture and training recipe, trained from scratch on 397 Jackson images for 3 h.
  - The published weights were pretrained on IMC17M, which contains both cohorts, so they cannot give a clean cross-cohort test.
  - A larger pretraining set might change ImmuVis's numbers. The comparison here is at equal training data: Jackson only.
- **Median vs mean:** a median over markers gives a misleading order here.
  - For per-image pixel r, the median would put the U-Net ahead of the cell models, even though it loses to them on 9 of 10 markers.
  - The median would also cut ImmuVis's cell-level score from 0.26 to 0.16, because its strength is concentrated in a few markers.
  - The papers report the mean over markers, so the figure uses the mean. Both are in `metrics_summary.csv`.
- **Target space for the Jackson-trained per-cell models:**
  - It uses arcsinh of the cell mean, not the mean of arcsinh pixels. This affects only the cell-to-background contrast in (a) and (b).
  - Clipping to [0, 1] raised cell-level r slightly (LightGBM median 0.456 → 0.469).
- **EVA's scaling is per image.**
  - EVA sees every channel rescaled within its image, so it cannot know a marker's absolute level in an image. That should cost it mostly on pooled pixel r.
  - Mapping its output back exactly would need the hidden channel's percentiles in the test image, which would leak the answer. So r uses the output as is, and the errors use a fixed mapping from the Jackson-train median percentiles (see Metrics). Any per-image mismatch of that mapping counts against EVA on RMSE.
  - Cell-level r and per-image r do not depend on image-level offsets. Cell-level r still depends on the scale of the predictions within each image.
- **RMSE depends on the target space.** Here it is the image model's `clip(arcsinh(x/5) / p99_Jackson, 0, 1)` for every model. Another scale would change the numbers but not the comparison with the constant, which uses the same space.
- **One replicate, one training run per model.** `train-test-grid` showed that the per-cell ranking is stable across 3 replicates (SD ≤ 0.04). ImmuVis's validation score had plateaued (0.644–0.646 over the last 3,500 steps).
- **U-Net arm:**
  - The ImmuVis-preprocessing arm was chosen in advance, as the field's standard preprocessing.
  - The gamma arm is equivalent (cell-level r 0.26, pooled 0.26); the residual arm is lower (0.19, 0.21).
  - All arms are in the CSVs for the ablation table.

## Files
- `metrics_summary.csv`: mean and median over the 10 markers, per model × metric, with CIs.
- `metrics_per_marker.csv`: per model × metric × marker, with CIs, for the models run in `cross-cohort-metrics`. ImmuVis and EVA per-marker point values are in `Results/immuvis-jackson/` and `Results/eva-zeroshot/`.
- `metrics_per_marker_all.csv`: per model × metric × marker, with CIs, for every model including ImmuVis, EVA and the constant.
- `per_cell_vs_image_per_marker.csv`: paired differences of each per-cell model against each image model, with calls. For RMSE, "better" means a lower error.
- `vs_constant.csv`: every model minus the constant on the three errors (mean over markers with CI, and per-marker calls).
- `normalisation_control.csv`, `metrics_median.csv` (the kernel's own median summary).
- `pixel_sums_test.csv`: pixel sums per model × marker × test image; they recompute (a) and (b) exactly.
- `direct_pixel_check.csv`, `training_log.csv`, `run_info.json`, `cross-cohort-metrics.log`.
- `summarise.py`, `make_figure.py`, `fig_cross_cohort_metrics.png` / `.pdf`.

## 32 µm patch level (added 2026-10-25)

**Runs:**
- `cross-cohort-metrics-patch`: per-cell models and the painted oracle.
- `unet-patch-sums`: the U-Net GPU pass again, immuvis arm.
- `immuvis-jackson-patch` and `eva-zeroshot-patch`.

These reruns reproduce every earlier pixel and cell value: per-cell models and EVA exactly, ImmuVis within 5e-6, U-Net pixel sums within 4e-5 relative.

**Metric:**
- Each test image is tiled into non-overlapping 32 × 32 px patches; partial edge tiles are dropped (median ≈ 300 patches per image).
- Truth and prediction are averaged per patch, in target space. Per-cell models are painted into masks with background 0.
- Pooled patch r, per-image patch r (images with ≥ 10 patches) and pooled patch RMSE. The bootstrap draws are shared with the other metrics.

**Mean over 10 markers [95% CI], figure models:**

| Model | Pixel pooled | Pixel per image | Patch pooled | Patch per image | Cell (R4) | Patch RMSE |
|---|---|---|---|---|---|---|
| U-Net | 0.26 | 0.22 | 0.48 [0.45, 0.51] | 0.57 | 0.26 | 0.103 |
| ImmuVis | 0.27 | **0.31** | 0.38 [0.34, 0.47] | **0.71** | 0.26 | 0.137 |
| EVA | 0.28 | 0.26 | 0.48 [0.46, 0.51] | 0.61 | 0.30 | 0.113 |
| LightGBM (Jackson-scaled) | 0.31 | 0.25 | **0.59 [0.57, 0.62]** | 0.69 | **0.43** | 0.117 |
| MLP (Jackson-scaled) | 0.31 | 0.24 | 0.60 | 0.68 | 0.40 | 0.106 |
| LightGBM, Danenberg-trained | 0.41 | 0.31 | 0.77 | 0.81 | 0.66 | 0.051 |
| True cell values (oracle) | 0.53 | 0.38 | 0.98 | 0.94 | 1 | 0.021 |
| Constant (training mean) | | | | | | 0.100 |

**LightGBM (Jackson-scaled) vs ImmuVis, per marker** (better / n.s. / worse for LightGBM):

| Metric | LightGBM better | n.s. | LightGBM worse |
|---|---|---|---|
| Per-image pixel r | 1 | 1 | 8 |
| Per-image patch r | 3 | 2 | 5 |
| Pooled patch r | 6 | 4 | 0 |
| R4 | 6 | 3 | 1 |

The reversal on per-image pixel r fades at 32 µm: pooled, ImmuVis no longer wins any marker. Per image, ImmuVis still wins 5 of 10.

**Errors at patch level:** only the U-Net comes near the constant, 0.103 vs 0.100. No Jackson-trained model beats it.
