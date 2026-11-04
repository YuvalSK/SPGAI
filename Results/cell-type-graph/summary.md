# Cell types: table model vs graph models (Danenberg, Jackson, J→D, D→J)

**Run:** Kaggle kernel `cell-type-graph`, version 1, 2026-10-16. T4 GPU, 167 min, 3 replicates, no units skipped.
**Notebook:** `Scripts/CellTypeGraph.ipynb` (the design is in its first cell).

**Labels:**
- Fine: Danenberg 32 phenotypes; Jackson 27 metaclusters.
- Coarse (6 classes): epithelial, T, B, myeloid, endothelial, stroma.
  - Jackson MC01 → T and MC02 → B by marker evidence (the published names don't match the markers).
  - Jackson MC03/MC05 and 4 mixed Danenberg phenotypes are left unlabelled.

**Inputs:** the 11 shared markers + H3, DNA1, DNA2 + log area.

**Scores:** macro-F1, mean ± SD over 3 replicates. Calls are paired patient-bootstrap CIs, one per replicate.

## Macro-F1

| Task | Train → test | Table | Table + nbr | MLP | GCN | GraphSAGE | SAGE+ | GATv2-e |
|---|---|---|---|---|---|---|---|---|
| fine | D → D | 0.437 | **0.454** | 0.389 | 0.243 | 0.385 | 0.382 | 0.392 |
| fine | J → J | 0.374 | **0.421** | 0.330 | 0.270 | 0.364 | 0.355 | 0.379 |
| coarse | D → D | 0.680 | **0.704** | 0.662 | 0.486 | 0.680 | 0.678 | 0.692 |
| coarse | J → J | 0.763 | **0.786** | 0.734 | 0.524 | 0.760 | 0.762 | 0.759 |
| coarse | J → D | 0.534 | 0.566 | 0.537 | 0.393 | 0.561 | **0.590** | 0.568 |
| coarse | D → J | 0.558 | 0.574 | 0.576 | 0.440 | 0.570 | **0.599** | 0.587 |
| coarse | Both → D | 0.671 | **0.692** | 0.648 | 0.485 | 0.670 | 0.673 | 0.680 |
| coarse | Both → J | 0.749 | **0.771** | 0.716 | 0.515 | 0.736 | 0.750 | 0.758 |

`[img-z]` (inputs standardised per image) is worse than the per-cohort scaling everywhere: Table 0.47–0.60, SAGE+ 0.52–0.64.

## Findings
1. **Neighbourhood information helps cell typing, unlike marker imputation.**
   - Table + nbr beats Table in all 24 within-cohort and pooled comparisons (8 rows × 3 replicates).
   - Gains: coarse +0.02, fine D +0.02, fine J +0.05.
   - For D→J it is better in 2 of 3 replicates.
2. **Within a cohort, LightGBM with neighbour features beats every GNN** (calls worse in 3/3 replicates):
   - SAGE+, GraphSAGE and GATv2-e are 0.01–0.03 below it on coarse labels and 0.04–0.07 below on fine labels.
   - On fine labels, the GNNs are also below the plain Table model.
3. **Across cohorts, the graph networks win.**
   - J→D: SAGE+ 0.590 vs Table 0.534 and Table + nbr 0.566 (better in 3/3 replicates against both).
   - D→J: SAGE+ 0.599 vs Table 0.558 (3/3 better); against Table + nbr it is +0.026 (better in 1/3, n.s. in 2/3).
4. **Endothelial carries most of the gain**, since the shared panel has no vessel marker:
   - within Danenberg: F1 0.37 (Table) → 0.45 (Table + nbr, GATv2-e);
   - J→D: 0.09 → 0.30 (SAGE+).
   - Stroma and epithelial also gain in J→D. B and T cells barely move.
5. **The Kipf GCN is far worst**, 0.15–0.25 below everything else. Averaging a cell with its neighbours loses the cell's own signal.
6. **Sweep (SAGE+, graph × depth):**
   - Graph type and k change macro-F1 by ≤ 0.02.
   - Depth: 2 layers are best for J→J and J→D; 1 layer is best for D→J (0.614 vs 0.596–0.612).
   - So the GNN deficit within a cohort is not a tuning problem.
7. **Pooling both cohorts** gives close to within-cohort performance in each cohort (as in `train-test-grid`). Table + nbr is again best.

## Caveats
- The labels come from clustering each cohort's full panel, which favours a per-cell model fitted to own-cell features.
- Danenberg's T and myeloid classes are hard (F1 ≈ 0.6) because of the shared panel (CD3 is weak; CD68 only for myeloid).
- One hyperparameter set for the networks (sweep on SAGE+ only).
