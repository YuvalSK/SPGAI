# cell-type-shap: summary

Run finished on 2026-10-22 in 96 min (XGBoost 3.2.0, TreeSHAP). It uses the same cells, coarse labels, replicate-0 split and Table + nbr features as `cell-type-graph`. Descriptive only: no decision depends on it.

## XGBoost matches the cell-type-graph LightGBM
Macro-F1: J → J 0.756, D → D 0.679, J → D 0.570, D → J 0.555, Both → J 0.745, Both → D 0.649. Table + nbr LightGBM in `kronos-zeroshot` scored 0.748, 0.670, 0.572 and 0.552. Endothelial is the weakest class everywhere (F1 0.13–0.44).

## 1. What drives each cell type
- **Immune types are typed by their own immune markers.** In J → J, own immune markers hold 55–57% of |SHAP| for T, B and myeloid cells. Permuting that group drops their F1 by 0.66–0.90.
- **The cohorts use different cues.** Danenberg leans on own epithelial markers for every class: permuting them drops macro-F1 by 0.28, including T cell −0.28 and myeloid −0.22, so "keratin-negative" helps define immune cells. Jackson relies on them mainly for epithelial, stroma and endothelial cells (macro-F1 drop 0.09).
- **Neighbour features matter mainly for endothelial and stroma.** These classes have the largest neighbour share of |SHAP|. Permuting neighbour size/density or neighbour stroma features drops J → J endothelial and stroma F1 by 0.04–0.05. No neighbour group costs more than 0.03 macro-F1 (largest: Danenberg neighbour epithelial, 0.029).
- **J vs D driver agreement is moderate:** Spearman of feature importance 0.42 (B cell) to 0.69 (endothelial); top-10 overlap 5–7 of 10.

## 2. Why Jackson → Danenberg fails
Largest confusions (20k-cell sample): epithelial → stroma (1,143), endothelial → stroma (590), stroma → endothelial (568), stroma → epithelial (551), T cell → stroma (412). For T → stroma, the wrong push comes mostly from own CD3 (mean push 4.0) and CD45 (1.2). That fits a cross-cohort CD3 scale difference, not missing context. Misclassified epithelial cells are pushed by own CD68, SMA and HER2.

## 3. Own × neighbour interactions (Jackson, 2,000 cells)
Interactions are a large part of the SHAP mass for every class: own × neighbour 24–29%, neighbour × neighbour 16–31%. For endothelial and stroma, neighbour × neighbour is the largest single part (29%, 31%) and own main effects are small (17–18%). The endothelial gain therefore comes from neighbourhood structure and its interactions, which fits a panel with no vessel marker.

## 4. Imputation drivers
- **SMA:** Jackson-trained R² 0.64, driven by CK8/18, panCK, CD68, CD20. Danenberg-trained R² 0.19, driven by panCK, H3, CD68, CD45. The Jackson model applied to Danenberg gives R² 0.00: the SMA relation does not transfer.
- **H3K27me3, Jackson (14 shared channels):** R² 0.51, driven by histone H3 (0.21), CK8/18, CK5, panCK.
- **H3K27me3, Keren (34 channels):** R² 0.70, dominated by phospho-S6 (0.31), then dsDNA (0.14). Phospho-S6 is H3K27me3's mass neighbour, so much of Keren's predictability may be channel spillover, not biology.

## Files
`xgb_scores.csv`, `shap_by_group.csv`, `shap_by_feature.csv`, `permutation_by_group.csv`, `driver_agreement_j_vs_d.csv`, `misclassification_drivers_j_to_d.csv`, `confusion_j_to_d_sample.csv`, `interactions_jackson.csv`, `imputation_shap.csv`, `fig_shap_groups.png`, `summary.json`, `cell-type-shap.log`.
