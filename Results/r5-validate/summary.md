# r5-validate (Phase 0), v2: summary

The full v2 run finished on 2026-10-22 in 410 min, CPU only. Same cells, images and masks as v1: the per-cell tables and mask md5s are byte-identical to v1. v1 results are in `v1/`, and the 60-image v2 smoke run is in `smoke/`.

v2 applies the five fixes proposed after v1 (see the notebook's first cell):
1. base and full models share the training subsample and seed;
2. arcsinh cofactor 1, clipped per channel at the 99.9th percentile of S cells;
3. the base model gets the mean and SD of true M over the rung's group (image for R5_own, image × state decile otherwise);
4. the candidate enters only as its deviation from its group mean;
5. identity criterion = R5 ≥ 0.9 × residual share (split-half ceiling reported, not gated).

## Verdict: gate G-R5 FAILED (11 failures, down from 116). Phase 1 is still blocked.

## What now passes
| Check | v2 result |
|---|---|
| Shuffled within image × state | max ΔR² +0.007 over 66 marker × rung cases (v1: up to +0.14) |
| Shifted true map | max +0.0095 (v1: up to +0.20) |
| kNN mean of M over S cells | max +0.010, none above the 0.01 tolerance |
| Power: planted α = 0.1 | detected (CI > 0) in 66/66 cases; ΔR² 0.005–0.105 |
| Identity | ≥ 0.968 × residual share everywhere (lowest: Jackson CK5) |
| Masks | per-cell means vs published tables r = 0.99 (Danenberg H3, DNA1, panCK), 0.71 (Danenberg ERα), ≥ 0.997 (Jackson) |
| Spillover | context gain (table − own) ≤ 0.05 except panCK (Jackson 0.051–0.079; Danenberg 0.051 at 2-px erosion); it does not shrink under erosion, so it is not membrane spillover |
| Split-half agreement | checkerboard vs random within 0.05, except Danenberg HER2, ERα, SMA (0.051–0.054) |

## What still fails: the "state mean" control
All 11 failures are the state-mean control (9) or planted α = 0 (2), which is the state mean plus pure noise.

| Rung | Failures | Worst |
|---|---|---|
| R5_cell | 0 (max +0.0099, Jackson panCK) | — |
| R5_own | 5 | Danenberg CK5 +0.033; Jackson CK5 +0.016, CD68 +0.015, SMA +0.014 |
| R5_table (headline) | 6 | Jackson SMA +0.024, panCK +0.020, CK5 +0.018; Danenberg panCK +0.014, CD45 +0.011 |

The state-mean control is the rung's own model fitted once on S: a fixed function of the conditioning set C, with the same N_FIT (100k) and LightGBM settings (250 trees, lr 0.06, 31 leaves) as the base model. A positive ΔR² therefore cannot come from information outside C. The likely cause (not yet tested) is base-model underfitting: adding a second, independently fitted model of the same C acts as an ensemble and lowers error. It concentrates in R5_table (most features) and in heavy-tailed markers (CK5, SMA, panCK).

Possible fixes to test: give the base model more capacity (more trees with early stopping on held-out patients) and check that the state-mean control falls below 0.01; or report R5 against a stacked base, LightGBM(C, f_S(C)).

## Files
- `gate.json`: verdict and failure list. `r5_controls.csv`: ΔR² with CI for every rung × control.
- `noise_ceiling.csv`, `planted_detection.csv`, `spillover_by_mask.csv`, `mask_hashes_*.csv`.
- `fig_r5_validation.png`, `r5-validate.log`, `run_info.json`.
- Per-cell tables (unchanged from v1): `IMC data/{Danenberg2022,Jackson2020}/r5-validate/percell_full_mask_*.parquet`; smoke versions in `smoke/` there.
