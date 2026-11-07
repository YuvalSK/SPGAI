# Neighbour label counts for cell typing (`future_plan.md` §16 arms A and B; pre-registered 2026-11-05)

**Run:** Kaggle `label-counts` v2 (CPU, 3 replicates, nothing skipped). Smoke run in `smoke/`.
- **Notebook:** `Scripts/LabelCounts.ipynb`; it reuses `multiscale-audit`'s loading, graphs and stage-1.
- **Sanity check:** arm 0 reproduces `multiscale-audit` arm 0 exactly (max |Δ| 0.0 over 18 replicate × unit cells).

**Setup:**
- **Count features (DOI style):** for each class, neighbours *with* the label, neighbours *without* it, and neighbours with a known label. Computed on the contact graph and on the 30 µm graph.
- **Arm A, predicted labels:**
  - A1: counts from out-of-fold predicted coarse types;
  - A2: A1 + predicted-probability mixtures;
  - A3 (coarse only): round 2, from out-of-fold A2 predictions.
- **Arm B, revealed true labels:** a random 5 / 20 / 50% of cells per image (fine: 20%), in training and test images alike. Counts over revealed neighbours; trained and scored on hidden cells only, paired with arm 0 on the same cells. In transfer, the target cohort's labels are revealed.
- **Claim rule:** mean Δ macro-F1 ≥ +0.01, with CI > 0 in 3/3 replicates.

## Verdict
**Arm A (predicted labels): fails everywhere.**

| Setting | A1 | A2 | A3 (round 2) |
|---|---|---|---|
| coarse J→J / D→D | +0.002 / +0.002 | +0.002 / +0.001 | +0.003 / +0.002 |
| coarse J→D | +0.002 | +0.003 | +0.004 |
| coarse D→J | −0.006 | −0.007 | **−0.009 (CI < 0 in 3/3)** |
| fine D / J | +0.002 / +0.002 | +0.002 / +0.002 | – |

- Predicted neighbour types add nothing beyond Table + nbr, which already sees the neighbours' markers.
- In D→J, iterating hurts: prediction errors from the source cohort feed back into the features.

**Arm B (revealed true labels): passes in 9 of 14 cells.**

| Setting | 5% | 20% | 50% |
|---|---|---|---|
| coarse J→J | +0.005 | **+0.013** | **+0.027** |
| coarse D→D | +0.003 | +0.008 | **+0.015** |
| coarse J→D (target labels revealed) | +0.006 | **+0.026** | **+0.039** |
| coarse D→J | +0.003 | **+0.020** | **+0.044** |
| fine D→D | – | **+0.071** | – |
| fine J→J | – | **+0.149** | – |

Bold = claim (mean ≥ +0.01, CI > 0 in 3/3). At 5%, the CI is > 0 in 2–3 of 3 replicates, but the gain stays below +0.01 everywhere.

## Reading
- **Known neighbour labels carry real information. Predicted ones don't.**
  - Across cohorts, annotating 20% of the target cells gives +0.02–0.03 macro-F1 on the rest, and 50% gives +0.04.
  - This parallels the imputation result: the lever is a little target-cohort ground truth, not a better model (few-shot patients: D→J cov₈₀ 0.44 → 0.64).
- **Caveat on the fine-label gains** (+0.07 / +0.15):
  - The published labels come from clustering the *full* panel, so a revealed neighbour's label carries information from markers outside the 11 shared ones.
  - Clustering labels are also spatially autocorrelated through segmentation and spillover.
  - So part of the fine gain is "label leakage through neighbours", not biology. Coarse gains, especially in transfer, are less exposed.
- **Practical reading:** a pathologist annotating ~20% of cells per image is worth about as much as the best graph model (SAGE+, +0.02–0.03 on J↔D in `cell-type-graph`); 5% is not.
