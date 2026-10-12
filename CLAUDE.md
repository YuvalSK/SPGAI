# Running things on Kaggle

All data and compute are on Kaggle (account `yuvalsamoilov`). Do not use data from folders outside this project.

## Setup
- The token is in `.env` (`KAGGLE_API_TOKEN`). Load it with `set -a && . ./.env && set +a`.
- CLI: `C:\Users\Yuval\anaconda3\Scripts\kaggle.exe`.
- Always `export PYTHONUTF8=1` first. Without it the CLI crashes on Unicode in notebooks.

## Push and run a notebook
1. Put the `.ipynb` and a `kernel-metadata.json` in their own folder. Copy a known-good metadata file with `kaggle kernels pull <id> -m -p <dir>`.
2. The notebook must have a kernelspec (`python3`) in its metadata, or Kaggle fails before running any code.
3. `kaggle kernels push -p <dir>` saves and runs it.
4. Check with `kaggle kernels status <id>`. Finished states are COMPLETE, ERROR and CANCEL.
5. Get results with `kaggle kernels output <id> -p <dir>`. The log is included, so read it on ERROR.

## Save every finished run locally
When a kernel finishes, download its outputs (the log comes with them) into this project:
- Results (tables, figures, json, log) go to `Results/<kernel-name>/`.
- Large data (cell tables, masks) go to `IMC data/<cohort>/<kernel-name>/`, which is gitignored.
- The notebook source stays in the repo root under the same name.

## kernel-metadata.json
- GPU: `"enable_gpu": true, "machine_shape": "NvidiaTeslaT4"` (2×T4, so the code uses `nn.DataParallel`).
- CPU only: `"enable_gpu": false` and no `machine_shape`.
- Internet (downloads, `pip install`): `"enable_internet": true`.
- Inputs:
  - `dataset_sources` for datasets.
  - `kernel_sources` to use another notebook's output (e.g. `yuvalsamoilov/danenberg-cells-prep`). It only works if that notebook's last run ended COMPLETE; a run that ended in ERROR passes on nothing.

## Data
| Source | Contents |
|---|---|
| `yuvalsamoilov/danenberg2022` | 794 image TIFFs + `panel.parquet`; images only |
| `yuvalsamoilov/jackson2020` | images only |
| `yuvalsamoilov/your-notebook-output-files2` | model checkpoint / resume cache |
| `yuvalsamoilov/cytof-data` | CyTOF BC tumours 13, 18, 19 |
| kernel `danenberg-cells-prep` | Danenberg cell table, contact graph, cell masks, clinical data, frozen patient split |

- Mount paths change; find files with `glob("/kaggle/input/**/<name>", recursive=True)`.
- The TIFFs are ZSTD-compressed. Install with `!pip install -q "imagecodecs==2024.9.22"` (works with Kaggle's numpy).

## Notes
- Checkpoint names include `ARCH_TAG`. Bump it when the model shape changes.
- Notebooks resume from the cache: a `_done` file means that run is finished.
- The GPU quota is ~30 h/week, with a 12 h limit per run.
