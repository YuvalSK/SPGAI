# Image models scored on the Jackson test patients (Danenberg → Jackson)

**Run (Kaggle, 2026-10-14, T4 GPU):** `image-models-jackson` v1, 37 min, no failed blocks. Notebook `Scripts/ImageModelsJackson.ipynb`.

**What it does:** scores the two image models of the reverse direction on the 106 images (66 patients, 229,818 cells) of the Jackson test patients.
- **ImmuVis, trained on Danenberg** (checkpoint from `immuvis-danenberg-train`, step 16,500, val pixel r 0.622).
- **EVA, zero-shot.**

Both are scored the same way as in the main figure: raw-output sums for r, plus a target-scale copy for the errors.

**Inputs and checks:**
- **Cell order and truth:** from the `cross-cohort-metrics-jackson` outputs (input kernel). Per image and marker, the truth sum and pixel count equal that kernel's (asserted).
- **RMSE:** RMSE from the sums equals a direct painted-map RMSE on 2 images (asserted, < 1e-6).
- **EVA reproduction:** its r values equal those of `cross-cohort-metrics-jackson` v1 to 1e-16, although the EVA repo was at a newer commit (67bae70).
- **Danenberg reference statistics** (`source_reference_danenberg.csv`, from the 494 Danenberg train images):
  - the median per-image p0.5 / p99.5 of each marker, which maps EVA's output back without test data;
  - the mean target value per pixel and per cell, which is the constant baseline.

**Result** (mean over 10 markers, ERα excluded; CIs and comparisons in `Results/cross-cohort-metrics-jackson/summary.md`):

| Model | Pooled pixel r | Per-image pixel r | Cell-level r | Pooled pixel RMSE | Per-image pixel RMSE | Cell-level RMSE |
|---|---|---|---|---|---|---|
| ImmuVis (Danenberg-trained) | 0.353 | 0.293 | 0.439 | 0.184 | 0.173 | 0.128 |
| EVA (zero-shot) | 0.210 | 0.205 | 0.184 | 0.195 | 0.180 | 0.160 |

The kernel does not compute cell-level r; `summarise.py` computes it from the per-cell predictions.

**Files:**
- `metrics_point_{immuvis,eva}.csv`, `pixel_sums_test_{immuvis,eva}.csv` (raw and `_t` target-scale sums).
- `source_reference_danenberg.csv`, `run_info.json`, `image-models-jackson.log`.
- **Per-cell predictions** (`pred__` raw, `predt__` target scale): `IMC data/Jackson2020/image-models-jackson/percell_pred_{immuvis,eva}.npz`.
