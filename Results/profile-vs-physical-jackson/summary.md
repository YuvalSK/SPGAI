# Profile proximity vs physical proximity (Jackson IMC), a replication of Danenberg version 3

**Run:** Kaggle kernel `profile-vs-physical-jackson`, version 1, 2026-09-28 (per-image stage 194 min).
**Notebook:** `ProfileVsPhysicalJackson.ipynb`, generated from `ProfileVsPhysical.ipynb` version 3. The code, settings and pre-registered rules are the same.
**Data:** `jackson-cells-prep` (Zenodo 4607374): 1,126,124 cells, 542 tumour images (at least 200 cells each), 349 patients.
- Basel: 284 cores, 278 patients.
- Zurich: 258 centre/periphery cores, 71 patients.

**Differences from Danenberg:**
- The full panel is 32 proteins (`full32`) instead of 36.
- Phenotypes are the 27 published metaclusters (Danenberg: 32 phenotypes).
- The contact graph is built from the masks, because Jackson published none. Contact = within 4 px; mean degree 5.1, against Danenberg's 5.87.
- The values are the published mean counts, which are not spillover-corrected. Danenberg's were.
- The `no_harm` flag is now direction-aware for A2.

**Setting:** k = 30, touching cells excluded. Every number has a patient-bootstrap 95% CI, and the CIs are narrow (about ±0.002–0.02).

## Main numbers next to Danenberg
| | Danenberg, 6 markers (int / rank) | Jackson, 6 markers (int / rank) | Danenberg, 36 (int / rank) | Jackson, 32 (int / rank) |
|---|---|---|---|---|
| Profile neighbours that are also physical neighbours (chance: D 3.0%, J 1.9%) | 4.0% / 3.7% | 3.4% / 2.8% | 5.2% / 5.5% | 4.8% / 4.7% |
| Closer than random cells | 9% / 7% | 13% / 10% | 15% / 16% | 23% / 22% |
| …after matching cell type | 4% / 3% | 8% / 5% | 9% / 10% | 16% / 16% |
| Context R² (fine) | 0.22 / 0.15 | 0.32 / 0.24 | 0.28 / 0.27 | 0.39 / 0.36 |
| **Context gain beyond cell type (R²)** | **+0.022 / +0.009** | **+0.055 / +0.029** | **+0.048 / +0.045** | **+0.078 / +0.060** |
| Cross-patient transfer, context R² | 0.015 / −0.017 | −0.012 / −0.020 | 0.045 / 0.044 | 0.025 / 0.030 |
| Cross-patient transfer, beyond type | 0.008 / 0.003 | −0.005 / −0.001 | 0.015 / 0.015 | 0.002 / 0.006 |

All pre-registered rules are met in both cohorts, for both encodings and both panels: closer than chance, closer beyond type, and context beyond type. The verdict string is the same: "Profile neighbours carry physical context beyond cell type with the CyTOF-matched markers".

## Panel-size curve (rank − intensity, obs)
| Markers | 6 | 12 | 18 | 24 | 31 | full |
|---|---|---|---|---|---|---|
| Context R², D | −0.078 | −0.039 | −0.019 | −0.015 | −0.011 | −0.007 (36) |
| Context R², J | −0.086 | −0.044 | −0.031 | −0.027 | **−0.023** | −0.023 (32) |
| Same-phenotype share, D | −0.084 | −0.076 | −0.069 | −0.068 | −0.071 | −0.068 |
| Same-phenotype share, J | −0.074 | −0.064 | −0.067 | −0.067 | −0.069 | −0.070 |
| Closeness (log2 ratio; + = rank worse), D | +0.041 | +0.022 | 0.000 | −0.007 | −0.015 | −0.021 |
| Closeness, J | +0.074 | +0.056 | +0.039 | +0.030 | **+0.018** | +0.016 |

Per cohort, closeness rank − intensity at 31 markers is +0.021 in Basel and +0.003 in Zurich.

## What replicates
1. **Look-alike cells are only somewhat closer in tissue, and mostly because they share cell type.** The context gain beyond cell type stays small: an R² of at most 0.08.
2. **Cross-patient transfer of detailed surroundings fails, even more clearly than in Danenberg.**
   - Context R² is −0.01 to 0.03, and about 0 beyond cell type.
   - Broad-class transfer (0.19–0.33) is again cell type.
   - **The decision to drop the CyTOF "virtual space" transfer stands.**
3. **Rank groups same-type cells about 7 points less well at every panel size.** This matches Danenberg almost exactly (−0.064 to −0.074 vs −0.068 to −0.084).
4. **Rank loses more with small panels, and the loss shrinks as the panel grows.** This holds for both context and closeness.

## What does not replicate
- **"Rank does no harm for tissue organisation at CyTOF-size panels" is weaker in Jackson.**
  - At 31 markers, rank loses 0.023 context R² (0.383 → 0.361, 6% relative). Danenberg lost 0.011 (4%).
  - Rank stays slightly *less* close than intensity at every panel size (+0.018 at 31). In Danenberg it became slightly closer from 24 markers on.
  - By the pre-set bar (CI above −0.01), "no harm" fails at every size in Jackson.
  - The gap is small, but the Danenberg sentence "at 31–36 rank is slightly better for closeness" should not be generalised.
- **Every within-image effect is larger in Jackson** (for example, the beyond-type context gain is +0.055 vs +0.022 with 6 markers). Possible reasons, none verified:
  - the published values are not spillover-corrected, and the contact guard is a little looser. However, excluding touching cells cuts the beyond-type gain by 14–16% in Jackson, against 27–33% in Danenberg, so leftover spillover alone is unlikely to explain it;
  - the images are larger TMA cores with more cells;
  - the metaclusters differ from Danenberg's phenotypes, so the type-matched null removes a different amount.

## Known issue
Because of a notebook-generator bug, the Kaggle HTML export failed on a markdown cell. The analysis ran fully and the outputs are complete; the repo notebook is fixed.
