# ImmuVis trained on Jackson, scored on Danenberg (main figure model)

**Runs (Kaggle, 2026-10-13, T4 GPU):**
- **Training:** `immuvis-jackson-train` v3, 190 min, notebook `Scripts/ImmuVisJacksonTrain.ipynb`; logs in `Results/immuvis-jackson-train/`.
- **Scoring:** `immuvis-jackson` v2, 32 min, no failed blocks, notebook `Scripts/ImmuVisJackson.ipynb`.
- **Rescoring for errors:** `immuvis-jackson` v3 (2026-10-14), 34 min. It also saves the output mapped to the target space (`to_target_arc(m, 2.3 × output)`, fixed Jackson p99), as pixel sums `Sp_t` / `Spp_t` / `Sgp_t` and per-cell `predt__`. Its r values equal v2's exactly. The files here are from v3.

**Model:**
- ImmuVis-Conv (Uchal 2026), code from github `dav3794/multiplex-image-model`. The anonymous release returns 403.
- Trained from scratch on the 397 tumour images of the Jackson train patients, with 39 validation images.
- **Training recipe:** the paper's channel masking (0.75 / 0.5 with subset sampling) and spatial masking (0.6, patch 8), β-NLL loss (β 0.5).
- **Settings:** 128-px crops, batch 16, cosine schedule, fp16, cuDNN disabled (it leaked host RAM and ran about 10× slower on the T4; see `Results/immuvis-profile/`).
- **Stopping and checkpoint:** stopped at 180 min (19,331 steps). The best checkpoint is from step 19,000, with validation pixel r 0.646 (each channel predicted with itself hidden).
- **Checkpoint file:** `IMC data/Jackson2020/immuvis-jackson-train/immuvis_jackson_best.pth`.
- **Input:** DNA1, DNA2, H3 and the 11 shared markers.
  - Preprocessing: arcsinh(x/5) → Butterworth low-pass (0.2) → clip(0, 2.3)/2.3.
- **Prediction on Danenberg:**
  - For each marker, the encoder sees the other 13 channels and the decoder is queried for the hidden one.
  - Tiles are 128 px with 32 px of context.
  - The output is scored as is, against the main figure's target space.

**Result** (mean over 10 markers, ERα excluded; 95% patient-bootstrap CIs from `Results/cross-cohort-metrics/summarise.py`):

| Pooled pixel r | Per-image pixel r | Cell-level r (within image) |
|---|---|---|
| 0.27 [0.24, 0.31] | 0.31 [0.30, 0.33] | 0.26 [0.23, 0.31] |

| Pooled pixel RMSE | Per-image pixel RMSE | Cell-level RMSE |
|---|---|---|
| 0.214 [0.197, 0.233] | 0.198 [0.186, 0.210] | 0.199 [0.168, 0.233] |

A constant prediction at the Jackson-train mean scores 0.191 / 0.181 / 0.139.

**Interpretation:**
- On per-image pixel r, ImmuVis is the best Jackson-trained model.
- At the cell level, it ties the U-Net and trails LightGBM (0.43).
- On errors it is the worst of the image models and worse than the constant (cell RMSE +0.060 [0.030, 0.093]).
  - Its predictions vary more than the truth (SD 0.16 vs 0.12), with large level offsets per marker (HER2 +0.25, CD3 +0.19, CD45 −0.20).
- See `Results/cross-cohort-metrics/summary.md`.
- **Why these numbers are below the paper's** (pooled Pearson 0.77, MSE 0.068): the paper pretrains on 26–27 cohorts, uses 37 input markers, scores a denoised target and scales each cohort by its own limit. We clipped Danenberg at Jackson's 2.3, where ImmuVis's config uses 2.9. Details are in "Comparison with the ImmuVis paper" in `Results/cross-cohort-metrics/summary.md`.

**Files:**
- `metrics_point_immuvis.csv`: per-marker point values.
- `pixel_sums_test_immuvis.csv`: per-image pixel sums.
- `run_info.json` and `immuvis-jackson.log`.
- **Per-cell predictions:** `IMC data/Danenberg2022/immuvis-jackson/percell_pred_immuvis.npz`.
