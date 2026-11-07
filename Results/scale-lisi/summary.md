# Cohort mixing vs cell-type separation, by scale (local, 2026-11-06)

**Question:** does any representation, at any scale, mix the Jackson and Danenberg cohorts better than plain cell intensity *and* keep cell types at least as far apart?

**Run:** local, `scale_lisi.py` + `plot_scale_lisi.py`, on saved outputs (cell tables, `pixel-features`, `kronos-zeroshot`, `kronos2-zeroshot`, `novae-cells` replicate-0 latents).
- **Cells:** replicate-0 test patients of `novae-cells` (66 Jackson, 130 Danenberg). 5 draws × 5,000 labelled cells per cohort.
- **LISI:** harmonypy algorithm, perplexity 30. iLISI over cohort (1–2, higher = more mixed); cLISI over 6 coarse types (lower = types kept apart). Every space is z-scored per column over the pooled draw. KRONOS / KRONOS2 are first reduced to 32 PCs.
- **Baseline, "intensity":** the 11 shared markers + H3, DNA1, DNA2 (arcsinh) + log area, robust-scaled within each cohort. This is the LightGBM input.
- **Reproduction check:** the intensity, + neighbours and Novae rows match `Results/novae-cells/lisi.csv` (replicate 0) within 6e-4.

## Result: no representation beats intensity on both axes, at any scale
Mean over cells, mean [min–max] over 5 draws.

| Scale | Space | iLISI | cLISI |
|---|---|---|---|
| – | **Intensity (baseline)** | **1.456** [1.449–1.463] | **1.511** [1.505–1.519] |
| Pixel | + pixel features (texture, membrane vs centre, shape) | 1.421 | 1.513 |
| Pixel | Pixel features only | 1.470 | 1.559 |
| Patch | + 64/128 µm patch means | 1.290 | 1.591 |
| Patch | KRONOS (32 PCs) | 1.261 | 1.805 |
| Patch | KRONOS2 (32 PCs) | 1.282 | 1.816 |
| Patch | + KRONOS2 | 1.234 | 1.637 |
| Cell | Intensity, no cohort scaling | 1.059 | 1.513 |
| Cell | Rank | 1.048 | 1.623 |
| Cell | + kNN-10 neighbours (Table + nbr) | 1.268 | 1.635 |
| Cell | Novae (`[cohort]`) | 1.769 | 2.186 |
| Cell | Novae, batch-corrected | 1.680 | 2.199 |
| Cell | Novae, per image | 1.787 | 2.010 |

- **Pixel:** essentially identical to intensity (Δ iLISI −0.035, Δ cLISI +0.002). Pixel features alone mix slightly more (+0.014) but blur types (+0.05).
- **Patch:** every patch-scale space, including both image foundation models, *separates* the cohorts more (iLISI −0.17 to −0.22) and blurs types (+0.08 to +0.31). Image patches carry cohort identity (staining, acquisition).
- **Cell / neighbourhood:** neighbour features separate the cohorts (the neighbourhood carries cohort style). Novae is the only space that mixes more (+0.23 to +0.33), and it pays with the largest type blurring (+0.50 to +0.69). This is mixing by losing biology, not by alignment.
- **Cohort scaling is the one preprocessing step that matters:** without it, iLISI falls from 1.46 to 1.06 at no gain in cLISI. Rank is no substitute (1.05).

## Reading
Intensity, robust-scaled within each cohort, sits on the trade-off frontier. Every richer representation moves along or behind it: better mixing is bought with worse type separation, never both. This matches the supervised results (pixel features: within-cohort typing only; patch / KRONOS: ≤ +0.01; Novae: ±0.002).

Figure: `fig_scale_lisi.png` / `.pdf`.
