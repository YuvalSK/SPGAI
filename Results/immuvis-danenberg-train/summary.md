# ImmuVis trained on Danenberg (the image model for Danenberg → Jackson)

**Run (Kaggle, 2026-10-14, T4 GPU):** `immuvis-danenberg-train` v1, 196 min (180 min training budget); child exit code 0. Notebook `Scripts/ImmuVisDanenbergTrain.ipynb`.

**What it is:** the mirror of `immuvis-jackson-train`. It uses the same code, architecture, preprocessing, masking, β-NLL loss and 180-min budget, trained from scratch on Danenberg instead of Jackson.
- **Code:** ImmuVis-Conv (Uchal 2026), from github `dav3794/multiplex-image-model`.
- **Images:** 494 train and 65 val images, from the Danenberg train / val patients of replicate 0 of `train-test-grid`. These are the same patients as the Danenberg-trained cell models in `cross-cohort-metrics-jackson`.
  - Each image belongs to one split, and none is a frozen test image (both asserted).
- **Channels:** DNA1, DNA2, H3 and the 11 shared markers, mapped by name.
- **Preprocessing:** arcsinh(x/5) → Butterworth low-pass (0.2) → clip(0, 2.3)/2.3. It is fixed and uses no cohort statistics.
- **Training:** cuDNN off, in a child process reading memory-mapped preprocessed images (171 M pixels, 4.5 GB).

**Training run:**
| | Danenberg (this run) | Jackson (`immuvis-jackson-train` v3) |
|---|---|---|
| Train / val images | 494 / 65 | 397 / 39 |
| Steps in 180 min | 19,553 (0.52 s/step) | 19,331 (0.53 s/step) |
| Best val LOO pixel r (step) | 0.622 (16,500) | 0.646 (19,000) |
| NaN steps | 0 | 0 |

- Validation r plateaued: it stayed at 0.616–0.622 over the last 4,500 steps. Host RAM was flat at 7.0 GB.
- **Per-marker val r at the best step** (Danenberg / Jackson model, each on its own cohort's val images):
  - HER2 0.78 / 0.76; CD20 0.52 / 0.42; CD3 0.03 / 0.07; CD45 0.81 / 0.78; CD68 0.55 / 0.66;
  - CK5 0.45 / 0.59; CK8/18 0.69 / 0.88; ERα 0.92 / 0.72; Ki-67 0.38 / 0.46; pan-CK 0.88 / 0.92; SMA 0.83 / 0.83.
  - **CD3 is not learnt in either cohort** (pixel r ≤ 0.07, even within cohort).

**Use:** `image-models-jackson` scores the checkpoint on the Jackson test patients (D→J figure).

**Files:**
- `run_info.json`, `child_done.json`.
- `immuvis_training_log.csv`: every 500 steps, including per-marker val r.
- `train_progress.log`, `child_stdout.log`, `immuvis-danenberg-train.log`.
- **Checkpoint:** `IMC data/Danenberg2022/immuvis-danenberg-train/immuvis_danenberg_best.pth`.
