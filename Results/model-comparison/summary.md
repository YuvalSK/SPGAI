# Table model vs ResNet-UNet vs graph networks (11 shared markers, Jackson → Danenberg)

**Run:** Kaggle kernel `model-comparison`, version 1, 2026-10-09. T4 GPU, about 75 min.
**Notebook:** `Scripts/ModelComparison.ipynb`.
**Setting:** one model per marker.
- Inputs: the other 10 shared markers + H3, DNA1, DNA2 + log cell area.
- Trained on 70% of Jackson patients; early stopping on 10%.
- Scored on the held-out 20% of Jackson patients (66) and on all Danenberg cells (cross-cohort).
- The ResNet-UNet predictions are from `kill-test`. Your existing checkpoints were trained on Jackson images, so they are scored on Danenberg only.
- **Graph:** the 10 nearest cells per cell, within each image, symmetrised (mean degree 11.4). There is no distance cutoff.

**Scores:** R4 = per-cell r after removing each image's mean. CIs from 1000× patient bootstrap. ERα is excluded from the medians and counts, because its Danenberg truth is saturated (`kill-test`).

## Median R4 over 10 markers
| Model | Jackson, held-out patients | Danenberg, cross-cohort |
|---|---|---|
| Table (LightGBM, own cell) | 0.60 | 0.42 |
| Table + ring (LightGBM, + neighbours within 30 µm) | 0.60 | 0.42 |
| MLP (own cell, no graph) | 0.60 | **0.46** |
| GCN (Kipf, 2 layers) | 0.46 | 0.45 |
| GraphSAGE (2 layers) | 0.60 | 0.44 |
| ResNet-UNet: residual / gamma / immuvis | not scorable | 0.19 / 0.25 / 0.23 |

## Per-marker calls (R4 difference, CI excludes 0; 10 markers)
**Jackson (held-out patients):**
- GraphSAGE vs MLP: better 8, n.s. 2, worse 0.
- GraphSAGE vs Table: better 5, n.s. 5, worse 0.
- Table + ring vs Table: better 7, n.s. 3, worse 0.
- GCN vs Table: worse 10.

**Danenberg (cross-cohort):**
- MLP vs Table: better 6, n.s. 4, worse 0.
- GraphSAGE vs MLP: better 1, n.s. 5, worse 4.
- GraphSAGE vs Table: better 4, n.s. 3, worse 3.
- GCN vs Table: better 2 (CD20, CD68), worse 6.
- ResNet-UNet vs Table: worse in 8–9 of 10; better only for SMA.

## Reading
1. **Every per-cell model beats the image model by about 2×** in Danenberg (median R4 0.42–0.46 vs 0.19–0.25).
2. **Within a cohort, neighbourhood context helps a little.**
   - GraphSAGE (separate weights for the cell and its neighbours) and Table + ring add +0.01 to +0.13.
   - The largest gains: CD20 0.36 → 0.49 (B cells cluster in aggregates), CK5 0.43 → 0.52, Ki-67 0.35 → 0.40.
   - The Kipf GCN is worse, because averaging a cell with its neighbours blurs the cell's own signal.
3. **Across cohorts, the spatial gain mostly disappears.**
   - The plain MLP is the best model. GraphSAGE ties or loses to it for 9 of 10 markers.
   - The exception is CD20 (GCN 0.42 vs Table 0.29, MLP 0.33).
4. **SMA fails for every per-cell model across cohorts** (R4 −0.13 to −0.24). The image models keep about 0.2. This is the SMA behaviour difference between the cohorts found in `cell-table-followup`.
5. **This agrees with Ali et al. 2025** (Nature Communications): in breast IMC, non-spatial single-cell models match graph networks.

## Caveats
- One run per model, with no hyperparameter search. The graph models might gain a little with tuning.
- **Replicated:** see `Results/model-comparison-reps/summary.md` (3 replicates). These hold:
  - point 1;
  - point 4;
  - the Kipf GCN being worst (point 2).

  Points 2 and 3 weaken otherwise: the neighbour gains are smaller than they looked, and the MLP's cross-cohort lead is a tie. Table, MLP and GraphSAGE are tied within the replicate spread.
- The kNN graph has no distance cutoff.
- In Jackson the truth is the published cell table; in Danenberg it is the image-derived target map.
