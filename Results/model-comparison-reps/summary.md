# Model comparison: robustness across 3 replicates

**Run:** Kaggle kernel `model-comparison-reps`, version 1, 2026-10-09 (T4 GPU, 182 min). This adds replicates 1 and 2 to `model-comparison`, which is replicate 0.
**Notebook:** `Scripts/ModelComparisonReps.ipynb`.
**Combined locally:** `combine_reps.py`.

**What changes per replicate:**
- the Jackson patient split (70/10/20);
- the network initialisation;
- the LightGBM seed and subsample.

**What stays fixed:**
- features and graphs;
- the ResNet-UNet predictions (your existing checkpoints), so their SD is 0.

The setup is the same as `model-comparison`:
- one model per marker;
- inputs: the other 10 shared markers, H3, DNA1, DNA2 and log area;
- graph: 10 nearest cells per cell, within each image, symmetrised.

**Scoring:**
- R4 = per-cell r after removing each image's mean.
- Calls are per replicate: a 1000× patient bootstrap, where the CI excludes 0.
- ERα is excluded (saturated Danenberg truth).
- Jackson scores use 20% held-out patients (about 66); Danenberg scores use all cells.

## Median R4 over 10 markers (replicates 0 / 1 / 2 → mean ± SD)
| Model | Jackson, held-out patients | Danenberg, cross-cohort |
|---|---|---|
| Table (LightGBM) | 0.60 / 0.66 / 0.63 → 0.63 ± 0.03 | 0.42 / 0.48 / 0.43 → 0.44 ± 0.03 |
| Table + ring | 0.60 / 0.67 / 0.64 → 0.64 ± 0.03 | 0.42 / 0.47 / 0.43 → 0.44 ± 0.03 |
| MLP | 0.60 / 0.68 / 0.64 → 0.64 ± 0.04 | 0.46 / 0.49 / 0.46 → **0.47 ± 0.02** |
| GraphSAGE | 0.60 / 0.68 / 0.65 → **0.64 ± 0.04** | 0.44 / 0.48 / 0.49 → **0.47 ± 0.03** |
| GCN (Kipf) | 0.46 / 0.54 / 0.53 → 0.51 ± 0.05 | 0.45 / 0.44 / 0.42 → 0.43 ± 0.01 |
| ResNet-UNet residual / gamma / immuvis | not scorable | 0.19 / 0.25 / 0.23 (fixed) |

**Best model by median R4:**
- Jackson: Table + ring in replicate 0, GraphSAGE in replicates 1 and 2.
- Danenberg: MLP in replicates 0 and 1, GraphSAGE in replicate 2.

## Per-marker calls (better / n.s. / worse, out of 10, for replicates 0 / 1 / 2)
| Cohort | Comparison | Better | Worse | Same non-n.s. call in all 3 |
|---|---|---|---|---|
| Jackson | GraphSAGE vs MLP | 8 / 4 / 5 | 0 / 0 / 0 | better: CD45, Ki-67, SMA |
| Jackson | GraphSAGE vs Table | 5 / 4 / 5 | 0 / 0 / 0 | better: CD45, SMA |
| Jackson | Table + ring vs Table | 7 / 5 / 5 | 0 / 0 / 0 | better: CD45, panCK, SMA |
| Jackson | GCN vs Table | 0 / 0 / 0 | 10 / 10 / 8 | worse: 8 markers |
| Danenberg | MLP vs Table | 6 / 4 / 6 | 0 / 4 / 2 | better: HER2, CD45, SMA |
| Danenberg | GraphSAGE vs MLP | 1 / 3 / 2 | 4 / 4 / 6 | worse: CD3, SMA |
| Danenberg | GCN vs MLP | 1 / 2 / 1 | 7 / 7 / 7 | worse: 6 markers |
| Danenberg | ResNet-UNet (each arm) vs Table | 1 / 1 / 1 | 8–9 each | worse: 8–9 markers; better: SMA |

**Mean R4 gain over 3 replicates (± SD), Jackson:**
- GraphSAGE − MLP:
  - CD20 +0.08 ± 0.02;
  - Ki-67 +0.08 ± 0.07;
  - CK5 +0.08 ± 0.07;
  - SMA +0.04 ± 0.01;
  - all other markers between −0.01 and +0.02.
- Table + ring − Table:
  - CK5 +0.08 ± 0.03;
  - CD20 +0.06 ± 0.02;
  - Ki-67 +0.03 ± 0.01;
  - all other markers ≤ 0.01.

## What holds
1. **Per-cell models beat the image models by about 2× in every replicate.**
   - Danenberg median R4 is 0.43–0.47 for the per-cell models vs 0.19–0.25 for the ResNet-UNet arms.
   - The image models lose for 8–9 of 10 markers in all 3 replicates.
   - SMA is the only marker where the image models win (0.19–0.25 vs −0.12 to −0.22), again in all 3.
2. **The Kipf GCN is the weakest per-cell model within Jackson in every replicate** (about −0.13 median). It is worse than the Table model for 8–10 markers each time.
3. **Neighbour context gives a small, consistent gain within Jackson, and is never worse.**
   - Neither GraphSAGE nor Table + ring is ever significantly worse than its non-spatial counterpart.
   - The gains are small and marker-specific. CD45, SMA, and Ki-67 or panCK are significant in all 3 replicates.
   - CD20 and CK5 have the largest mean gains (about +0.06 to +0.08) but are not significant in every split.
4. **Across cohorts, the spatial gain is gone.**
   - In Danenberg, GraphSAGE ties the MLP (median 0.47 vs 0.47) and is significantly worse for CD3 and SMA in all 3 replicates.
   - CD20 is the one marker where a graph model has the top point estimate in all 3 replicates (0.38–0.39 vs MLP 0.35, Table 0.33). The gap is not significant in every replicate.
5. **SMA fails cross-cohort for every per-cell model in every replicate.**

## What does not hold
- **"GraphSAGE beats the MLP for 8 of 10 markers in Jackson"** was the high end. It is 8, 4 and 5 across replicates.
- **"The MLP is best across cohorts"** is a tie: the MLP and GraphSAGE share first place.
- The MLP's lead over the Table model in Danenberg (+0.03 median) is consistent only for HER2, CD45 and SMA.
- **Model ranking is not stable.** Differences among Table, Table + ring, MLP and GraphSAGE (≤ 0.01 median in Jackson, ≤ 0.03 in Danenberg) are smaller than the spread across replicates (SD 0.03–0.04 in Jackson, mostly from the patient split). Those four models are effectively tied; only GCN and the image models separate.

## Files
- `r4_per_marker_mean_sd.csv`: per cohort × model × marker, mean / SD / min / max R4 over the 3 replicates.
- `median_r4_by_rep.csv`
- `best_model_per_marker.csv`
- `calls_by_rep.csv`: per-marker call in each replicate.
- `call_counts_by_rep.csv`
- `diff_r4_mean_sd.csv`
- `fig_model_comparison_reps.png`
- Raw files: `scores_rep{1,2}.csv`, `differences_R4_rep{1,2}.csv`, `training_log_rep{1,2}.csv`, and the kernel log. Replicate 0 is in `Results/model-comparison/`.
