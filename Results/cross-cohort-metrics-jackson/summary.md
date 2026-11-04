# Cross-cohort inference, Danenberg → Jackson: the reverse direction of the main figure

**Runs (Kaggle, T4 GPU):**
- `cross-cohort-metrics-jackson` v1 (2026-10-13; 31 min, no failed blocks): cell models, truth and the oracle. Notebook `Scripts/CrossCohortMetricsJackson.ipynb`.
- `immuvis-danenberg-train` v1 (2026-10-14; 196 min): ImmuVis trained from scratch on Danenberg. See `Results/immuvis-danenberg-train/`.
- `image-models-jackson` v1 (2026-10-14; 37 min, no failed blocks): ImmuVis-Danenberg and EVA scored on the Jackson test patients, with target-scale outputs for the errors. Notebook `Scripts/ImageModelsJackson.ipynb`; outputs in `Results/image-models-jackson/`.
  - EVA's r values equal those of the first run exactly (max difference 1e-16), although the EVA repo had moved to a new commit (67bae70).

**Combined locally:** `summarise.py` and `make_figure.py` → `fig_cross_cohort_metrics.png` / `.pdf`. The maths and drawing code are shared with the main figure (`../cross-cohort-metrics/summarise_core.py`, `figure_core.py`).

**Large outputs:**
- Per-cell predictions:
  - `IMC data/Jackson2020/cross-cohort-metrics-jackson/percell_predictions_test.npz`
  - `IMC data/Jackson2020/image-models-jackson/percell_pred_{immuvis,eva}.npz`
- ImmuVis checkpoint: `IMC data/Danenberg2022/immuvis-danenberg-train/immuvis_danenberg_best.pth`.

**Setting:**
- **Test set:** the Jackson test patients of the `train-test-grid` replicate-0 split: 66 patients, 106 images, 229,818 cells.
- **Training:** every model except the reference and EVA is trained on the Danenberg train patients, with the same split and seeds as `train-test-grid`. ImmuVis uses the 494 train and 65 val images of those patients.
- **Truth:** the same target space as the main figure, built from the Kaggle Jackson TIFFs and the masks relabelled by `jackson-cells-prep`.
  - These truth cell means agree with the published Jackson cell table (r 0.80–0.99 per marker), which confirms the masks line up with the cells and the channels are in the right order.
  - In `image-models-jackson`, the truth sums equal those of `cross-cohort-metrics-jackson` for every image and marker (asserted).
  - RMSE from the sums equals a direct RMSE on 2 images (asserted, < 1e-6).
- **Markers:** the 10 used in the main figure; ERα is left out of the summaries so the two directions stay comparable.
- **Main version of the cell models:** Jackson inputs are scaled with **Danenberg** statistics, so there is no cohort-level adaptation to the test cohort. This is the same rule as the Jackson-scaled models in the main figure.
- **Errors:** the same three RMSEs as the main figure, in the target space.
  - ImmuVis is mapped with `to_target_arc(m, 2.3 × output)`.
  - EVA is mapped with the median per-image p0.5 / p99.5 of the **Danenberg** train images (`Results/image-models-jackson/source_reference_danenberg.csv`).
  - The constant predicts the Danenberg-train mean everywhere.
- **No U-Net in this direction:** the ResNet-UNet was trained on Jackson, and training one on Danenberg would take about 12 GPU hours.

## Mean over 10 markers [95% patient-bootstrap CI]
| Model | (a) Pooled pixel r | (b) Per-image pixel r | (c) Cell-level r |
|---|---|---|---|
| ImmuVis, Danenberg-trained | **0.35 [0.33, 0.37]** | **0.29 [0.28, 0.30]** | **0.44 [0.42, 0.45]** |
| EVA, zero-shot | 0.21 [0.20, 0.22] | 0.20 [0.19, 0.22] | 0.18 [0.16, 0.21] |
| LightGBM, Danenberg-trained | 0.34 [0.31, 0.36] | 0.27 [0.25, 0.28] | 0.41 [0.39, 0.43] |
| MLP, Danenberg-trained | 0.33 [0.30, 0.35] | 0.26 [0.24, 0.27] | 0.41 [0.38, 0.44] |
| GCN, Danenberg-trained | 0.29 [0.27, 0.32] | 0.23 [0.21, 0.24] | 0.31 [0.28, 0.33] |
| GraphSAGE, Danenberg-trained | 0.31 [0.29, 0.33] | 0.26 [0.24, 0.27] | 0.40 [0.37, 0.43] |
| LightGBM, Jackson-trained (within-cohort reference) | 0.44 [0.41, 0.47] | 0.34 [0.32, 0.35] | 0.69 [0.66, 0.71] |
| Oracle: true cell values painted | 0.55 [0.52, 0.57] | 0.42 [0.41, 0.44] | 1 |

| Model | (d) Pooled pixel RMSE | (e) Per-image pixel RMSE | (f) Cell-level RMSE | (f) vs constant: better / n.s. / worse markers |
|---|---|---|---|---|
| ImmuVis, Danenberg-trained | 0.184 [0.172, 0.196] | 0.173 [0.161, 0.185] | **0.128 [0.117, 0.138]** | 8 / 0 / 2 |
| EVA, zero-shot | 0.195 [0.180, 0.209] | 0.180 [0.166, 0.194] | 0.160 [0.146, 0.173] | 1 / 2 / 7 |
| LightGBM, Danenberg-trained | 0.184 [0.171, 0.198] | 0.173 [0.160, 0.186] | 0.141 [0.128, 0.152] | 5 / 3 / 2 |
| MLP, Danenberg-trained | 0.183 [0.170, 0.197] | 0.171 [0.158, 0.185] | 0.138 [0.126, 0.149] | 5 / 3 / 2 |
| GCN, Danenberg-trained | 0.193 [0.179, 0.208] | 0.180 [0.166, 0.195] | 0.159 [0.144, 0.173] | 2 / 5 / 3 |
| GraphSAGE, Danenberg-trained | 0.187 [0.173, 0.202] | 0.175 [0.162, 0.189] | 0.146 [0.131, 0.159] | 5 / 2 / 3 |
| LightGBM, Jackson-trained (reference) | 0.165 [0.154, 0.176] | 0.156 [0.146, 0.168] | 0.087 [0.079, 0.094] | 10 / 0 / 0 |
| Constant: Danenberg-train mean | 0.195 [0.183, 0.207] | 0.185 [0.174, 0.196] | 0.154 [0.145, 0.163] | |
| Oracle: true cell values painted | 0.151 [0.143, 0.160] | 0.144 [0.135, 0.154] | 0 | |

**Model minus constant, mean over markers [CI]** (`vs_constant.csv`; negative = lower error than the constant):
- **ImmuVis:** −0.011 [−0.012, −0.009] pooled; −0.012 per image; **−0.026 [−0.029, −0.023] cell**.
- **LightGBM:** −0.010 / −0.012 / −0.013. **MLP:** −0.011 / −0.013 / −0.016. **GraphSAGE:** −0.007 / −0.009 / −0.008.
- **GCN:** −0.001 (n.s.) / −0.005 / +0.005 (n.s.).
- **EVA:** −0.000 (n.s.) / −0.005 / +0.006 [0.001, 0.010], i.e. worse at the cell level.

**Paired per marker, ImmuVis vs LightGBM** (ImmuVis better / n.s. / worse; `per_cell_vs_image_per_marker.csv`):

| Metric | vs LightGBM | vs MLP | vs GCN | vs GraphSAGE |
|---|---|---|---|---|
| Pooled pixel r | 5 / 2 / 3 | 5 / 3 / 2 | 6 / 3 / 1 | 6 / 1 / 3 |
| Per-image pixel r | 7 / 0 / 3 | 7 / 1 / 2 | 8 / 0 / 2 | 7 / 1 / 2 |
| Cell-level r | 5 / 2 / 3 | 6 / 2 / 2 | 8 / 0 / 2 | 5 / 2 / 3 |
| Cell-level RMSE | 6 / 3 / 1 | 6 / 3 / 1 | 8 / 2 / 0 | 7 / 3 / 0 |

- **Cell-level r, ImmuVis vs LightGBM:**
  - ImmuVis wins HER2 (0.65 vs 0.44), CD20 (0.41 vs 0.17), CK5 (0.46 vs 0.26), CK8/18 and pan-CK.
  - It loses CD3 (0.05 vs 0.52), CD45 (0.65 vs 0.70) and SMA (−0.19 vs −0.14, both failing). CD68 and Ki-67 are n.s.
- **LightGBM vs EVA (better / worse):** cell-level r 5 / 1, per-image r 6 / 1, pooled r 5 / 1, cell RMSE 7 / 2.

## What it shows
1. **In this direction the image model trained on the other cohort wins.**
   - ImmuVis trained on Danenberg is the best cross-cohort model on all six metrics. Its cell-level r is 0.44, against 0.41 for LightGBM and the MLP, and it beats LightGBM on 5 of 10 markers at the cell level (loses 3).
   - In the main figure (Jackson → Danenberg), the same recipe trained on Jackson scored 0.26 at the cell level and lost to LightGBM 1 / 6.
2. **The two directions are asymmetric for ImmuVis, not for the cell models.**
   - The cell models score about the same in both directions: LightGBM 0.43 (J→D) and 0.41 (D→J).
   - ImmuVis goes from 0.26 to 0.44. The gain is in the immune and luminal markers. Cell-level r, J→D vs D→J: CD45 0.21 → 0.65, CD20 0.14 → 0.41, CD68 0.08 → 0.41, CK8/18 0.30 → 0.80, CK5 0.04 → 0.46.
   - CD3 fails in both directions (0.17 / 0.05), and ImmuVis also fails to learn CD3 within either cohort (validation pixel r ≤ 0.07).
   - **Not tested:** why the directions differ. Danenberg gives ImmuVis more training images (494 vs 397). Its intensities are also brighter for CD45 (3.8× the Jackson mean) and CD20 (1.9×). Under the fixed clip at arcsinh 2.3, a Jackson-trained model would then see saturated Danenberg inputs, while a Danenberg-trained model sees Jackson inputs within its range. ImmuVis's own code config scales Danenberg at 2.9 (Jackson-Basel 2.3), so the published protocol would not clip Danenberg this hard. We used 2.3 for both cohorts so that no model sees test-cohort statistics. Brightness does not explain every marker: CK8/18 is dimmer in Danenberg (0.6×) and also improves. One training run per direction cannot separate these causes.
3. **Here cross-cohort models do beat a constant on errors, but only slightly.**
   - ImmuVis lowers cell-level RMSE by 0.026 against the constant (0.128 vs 0.154). LightGBM and the MLP lower it by 0.013–0.016; EVA is worse.
   - Within-cohort training is still far ahead: the Jackson-trained LightGBM reaches 0.087.
   - In the main figure, no Jackson-trained model beat the constant at all.
4. **EVA is weaker on Jackson than on Danenberg** (cell-level r 0.18 vs 0.30; per-image pixel r 0.20 vs 0.26). It is the weakest model here on all three correlations, and ties GCN for the largest errors.
5. **Cohort scaling changes little for r in this direction** (≤ 0.02, `normalisation_control.csv`). On errors, Danenberg scaling is better than Jackson's own: LightGBM cell RMSE is 0.141 with Danenberg scaling and 0.167 with Jackson's own.
6. **Pixel metrics again cap any cell map** (oracle 0.55 pooled, 0.42 per image) and compress the gaps between models.

## Caveats
- **One training run per model and direction.** The ImmuVis asymmetry is the largest effect here and rests on one run per direction. The per-cell ranking was stable across 3 replicates in `train-test-grid`.
- **Comparison with the published ImmuVis numbers** (pooled Pearson 0.77, MSE 0.068): different protocol and pretraining; see "Comparison with the ImmuVis paper" in `../cross-cohort-metrics/summary.md`.
- **ImmuVis's mean hides marker failures.** It wins on 5 markers but scores 0.05 on CD3, which LightGBM gets to 0.52.
- **SMA fails for every cross-cohort model** in this direction (cell-level r −0.19 to −0.03), while the Jackson-trained LightGBM reaches 0.80.

## Files
- From `summarise.py`:
  - `metrics_summary.csv`: mean and median over markers, with CIs.
  - `metrics_per_marker_all.csv`.
  - `per_cell_vs_image_per_marker.csv` (for RMSE, "better" = lower error).
  - `vs_constant.csv`.
  - `normalisation_control.csv`.
- From the kernel: `metrics_per_marker.csv`, `pixel_sums_test.csv`, `truth_vs_table_check.csv`, `direct_pixel_check.csv`, `training_log.csv`, `run_info.json`, `cross-cohort-metrics-jackson.log`.
- ImmuVis / EVA sums, point values and the Danenberg reference statistics: `Results/image-models-jackson/`.
- `summarise.py`, `make_figure.py`, `fig_cross_cohort_metrics.png` / `.pdf`.

## 32 µm patch level (added 2026-10-25)

**Runs:** `cross-cohort-metrics-jackson-patch` and `image-models-jackson-patch` (ImmuVis-Danenberg and EVA). They reproduce the earlier values, to within 3e-6.

**Mean over 10 markers:**

| Model | Patch pooled | Patch per image | Patch RMSE |
|---|---|---|---|
| ImmuVis (Danenberg-trained) | **0.68 [0.65, 0.71]** | **0.66** | 0.078 |
| LightGBM (Danenberg-scaled) | 0.66 [0.62, 0.70] | 0.63 | 0.082 |
| EVA | 0.38 | 0.45 | 0.102 |
| LightGBM, Jackson-trained | 0.84 | 0.79 | 0.051 |
| True cell values (oracle) | 0.98 | 0.94 | 0.021 |
| Constant (training mean) | | | 0.101 |

The ImmuVis advantage in this direction holds at patch scale: LightGBM is worse on 7 of 10 markers per image, and wins 1 of 10 pooled. That matches the cell level.
