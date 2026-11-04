# r5-validate (Phase 0): summary

The full run (kernel version 4) finished in 289 min, CPU only. It covers both cohorts: Danenberg (1.06 M tumour cells, 704 images, 650 patients) and Jackson (1.13 M tumour cells, 542 images, 349 patients). Every result here was checked against the files in this folder.

## Verdict: gate G-R5 FAILED (116 failures). Phase 1 is blocked.

## What passed
- **Masks.** Per-cell means computed from the images under the cell masks match the published tables. Danenberg: H3 0.99, DNA1 0.99, panCK 0.99, ERα 0.71. Jackson: ≥ 0.997. Mask md5s are in `mask_hashes_*.csv`.
- **Power.** The planted signal is monotone in α in every case. α = 0.1 is detected (CI > 0) in 43 of 44 marker × rung cases; the miss is Jackson CK5 R5_table, at +0.038. The R5 at α = 0.1 is 0.010–0.075, which gives the minimum detectable effect per marker. See `planted_detection.csv`.
- **Noise ceiling.** The residual is highly reliable beyond the panel. R5_cell ceilings are 0.48–0.98 (checkerboard split), so a future "no" would not be explained by noise. Checkerboard and random splits agree within 0.05, except three markers at 0.050–0.052: Danenberg ERα and SMA, Jackson CK8/18. See `noise_ceiling.csv`.
- **Spillover.** The context gain (table − own) is ≤ 0.04 and does not shrink under 1-px or 2-px erosion. Neighbour information is small, and it is not membrane spillover. See `spillover_by_mask.csv`.

## What failed
| Failure | Worst cases | Likely cause |
|---|---|---|
| Shuffled-within-group control > 0.01 | CK5 R5_cell +0.12 (both cohorts), R5_table +0.14 (Danenberg); 8 other markers +0.01–0.07 | The full model gets p and p − group mean(p), so it recovers the group mean of M (exactly, for this control). The base model lacks it, and the deviation scale grows with group level. |
| Shifted true map > 0.01 | R5_own: CK8/18 +0.20, CK5 +0.19, HER2 +0.14 (Danenberg); ERα +0.14 (Jackson). R5_cell/R5_table up to +0.07 | Same image-level M leak. It also points to long-range within-image structure that a cyclic shift keeps. |
| State mean, planted α = 0, kNN ≠ 0 | Jackson CK5 R5_own −0.046; SMA +0.02; kNN up to +0.029 | Heavy tails: in Jackson, the top 0.1% of cells hold 30–43% of the within-image sum of squares for CD3, CK5 and CD20 (arcsinh cofactor 5 on raw counts with median ≈ 0.1). Base and full fits also use different subsamples and seeds. |
| Identity below 0.9 × residual share | Jackson CD3 0.64 vs 0.91; Danenberg Ki-67 0.89 vs 1.00 | Heavy tails, as above. |
| Identity below the split-half ceiling | 14 cases, mostly short by 0.01–0.04 | Mostly the criterion: the ceiling is an upper bound (it counts base-model misfit as reliable variance). |

The planted-signal detection holds even though the controls fail. The test has power but no specificity: right now it would credit an image model that only knows image- or state-level brightness.

## Proposed fixes (none run yet)
1. Base and full models use the same training subsample and the same seed.
2. Arcsinh cofactor 1, then clip at the 99.9th percentile per channel.
3. The base model gets the leave-one-out mean and SD of M within the image (R5_own) and within image × state (R5_cell, R5_table).
4. The candidate enters only as its deviation from its group mean.
5. Identity check: keep "≥ 0.9 × residual share" and drop "≥ split-half ceiling". Report the ceiling as an upper bound.

A local check of these fixes was stopped (low memory) after only the unchanged variant. It reproduced the Kaggle failures for Jackson CK5: shuffled +0.20 on R5_own and +0.09 on R5_cell; state mean −0.07 on R5_own. Every further test should run on Kaggle.

## Files
- `r5_controls.csv`: ΔR² with CI for every rung × control.
- `noise_ceiling.csv`, `planted_detection.csv`, `spillover_by_mask.csv`.
- `gate.json`: the gate verdict and the full failure list.
- `fig_r5_validation.png`, `r5-validate.log`.
- `smoke/`: the 60-image smoke run.
- Per-cell tables: `IMC data/{Danenberg2022,Jackson2020}/r5-validate/percell_full_mask_*.parquet`. Local pyarrow 19 cannot read them; duckdb can.
