# Multi-scale audit (Track B0–B2, `future_plan.md` §18.10)

**Runs:**
- `pixel-features` v2 (CPU, 8.5 min): per-cell sub-cellular features for 1.06M Danenberg and 1.13M Jackson cells. Masks are byte-identical to `r5-validate` (704/704, 542/542 md5).
- `multiscale-audit` v2 (CPU, 235 min, 3 replicates, nothing skipped).
- Smoke runs are in `smoke/`.
- **Notebooks:** `Scripts/PixelFeatures.ipynb`, `Scripts/MultiscaleAudit.ipynb`.
- **Large outputs:** `IMC data/<cohort>/pixel-features/`.

**Setup:**
- **Base (arm 0):** LightGBM Table + nbr on image-derived cell means. For imputation, the target channel is removed from every block.
- **Pixel features** (arm 1), per channel: outer-minus-inner third, within-cell SD, within-cell Moran's I; plus shape.
- **Spillover controls:**
  - arm 2: the same features on the 2-px-eroded mask;
  - arm 4 vs 3: pixel features beyond the outside 2-px band.
- **Arm 5 (B2):** 64 / 128 µm patch means, contact vs non-contact neighbour means, and (typing only) out-of-fold predicted-type mixtures and counts.
- **Scores:** imputation R4; typing macro-F1. Paired patient-bootstrap CIs.

## Verdict (`decision.json`): B1 passes formally, but only for within-cohort typing. B2 fails.

| Gate | Passes in | Does not pass |
|---|---|---|
| **B1, pixel scale** | **Typing within cohort:** coarse D→D (+0.010), fine D→D (+0.021), fine J→J (+0.018). In all three, the gain survives 2-px erosion and stays beyond the outside band, with CI > 0 in 3/3 replicates | **Imputation:** 0 markers reach +0.02 in any setting (largest mean Δ +0.0065, within cohort). **Typing across cohorts:** J→D −0.002, D→J +0.003 (n.s.). Coarse J→J +0.009 (just under the bar) |
| **B2, patch / typed edges** | – | Imputation: only 1 marker per transfer setting (CK5 J→D, HER2 D→J; the rule needs ≥ 2). Typing transfer: J→D +0.005, D→J −0.010 |

### Mean Δ vs Table + nbr (3 replicates)
| | Imputation R4, within D / J | Imputation R4, J→D / D→J | Coarse typing, D→D / J→J | Coarse typing, J→D / D→J | Fine typing, D / J |
|---|---|---|---|---|---|
| 1 +pix | +0.007 / +0.006 | +0.001 / −0.002 | **+0.010** / +0.009 | −0.002 / +0.003 | **+0.021 / +0.018** |
| 2 +pix (eroded) | +0.002 / +0.001 | +0.002 / −0.004 | +0.006 / +0.004 | −0.004 / +0.006 | +0.009 / +0.018 |
| 4 vs 3 (beyond band) | +0.006 / +0.006 | −0.000 / −0.007 | +0.007 / +0.008 | −0.002 / +0.002 | +0.014 / +0.016 |
| 5 +B2 | +0.003 / +0.001 | +0.002 / +0.003 | +0.004 / +0.004 | +0.005 / −0.010 | −0.003 / +0.002 |

**What the pixel scale carries: morphology and texture, which help fine cell types within a cohort.**
- The largest per-class gains (fine typing, within cohort):
  - Jackson "MC21 epithelial low" (+0.13), "MC11 fibronectin hi" (+0.09), "MC10 small elongated" (+0.07), "MC09 small circular" (+0.06);
  - Danenberg "Ep CD57⁺" (+0.08), "CD38⁺ lymphocytes" (+0.06), "CK^lo ER^med" (+0.06).
- Coarse typing: endothelial gains most (+0.015 to +0.027).
- **The signal does not transfer between cohorts, and does not help marker imputation.**

## B0: homophily (same-type share among neighbours vs a within-image permutation null)
| Class | Danenberg contact / 30 µm | Jackson contact / 30 µm |
|---|---|---|
| epithelial | 1.51 / 1.41 | 1.47 / 1.34 |
| B cell | 1.67 / 1.62 | 1.43 / 1.25 |
| T cell | 0.80 / 0.72 | 1.54 / 1.39 |
| stroma | 1.03 / 0.86 | 1.14 / 0.99 |
| myeloid | 0.57 / 0.47 | 0.75 / 0.52 |
| endothelial | 0.85 / 0.52 | 0.78 / 0.35 |

- Epithelial and B cells cluster with their own type. **Endothelial and myeloid cells are heterophilous** (0.35–0.85). So are Danenberg T cells, but not Jackson's.
- **Typing errors concentrate where neighbours are mostly a different type.** Error rate, Table + nbr, within cohort, same-type contact share < 0.25 vs > 0.75:
  - epithelial 0.43 vs 0.03 (D), 0.54 vs 0.07 (J);
  - endothelial 0.68 vs 0.21 (D), 0.67 vs 0.45 (J).

  This is correlational: cells at compartment borders are also harder for other reasons.

## Implications
- **Formally, the B1 pass opens B3.** But the only signal is within-cohort typing (+0.01–0.02 macro-F1) from morphology and texture. It touches neither imputation nor transfer, which are the paper's axes. A hierarchical network with a cohort node has nothing new to propagate across cohorts here.
- **For the paper:** "sub-cellular structure adds nothing to marker imputation beyond cell means (≤ +0.007 R4), with spillover controls" is a clean negative that supports PIVOT-A. "Morphology helps fine typing within a cohort" is a minor positive for an appendix.
- Caveat: eroded-mask features are missing for 47% of Jackson cells (median 16 eroded pixels), so the erosion control is weaker in Jackson.
