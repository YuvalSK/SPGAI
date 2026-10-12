# kNN enrichment of marker-high cells: rank vs intensity (Jackson IMC), a replication of the Danenberg run

**Run:** Kaggle kernel `knn-enrichment-jackson`, version 1, 2026-09-28 (per-image stage 147 min).
**Notebook:** `KnnEnrichmentJackson.ipynb`, generated from `KnnEnrichmentIMC.ipynb`. The code and pre-set readings are the same; only the data loading and the panel size differ.
**Data:** `jackson-cells-prep` (Zenodo 4607374): 1,126,124 cells, 542 tumour images (at least 200 cells each), 349 patients.
- Basel: 284 tumour cores, 278 patients.
- Zurich: 258 centre/periphery cores, 71 patients.

**Differences from Danenberg:**
- The full panel is 32 protein channels (`full32`) instead of 36.
- Phenotypes are the 27 published metaclusters.
- There is no hot-aggregate flag.

**Score and stats:** the same as Danenberg. "High" = robust z > 1; k = 30; enrichment = (fraction of high neighbours) / (share of high cells). Patient bootstrap, 95% CI.

## Results next to Danenberg (rank − intensity)
| | Danenberg | Jackson |
|---|---|---|
| 6 CyTOF markers, intensity → rank | 3.13 → 2.51 (**−0.62**) | 3.44 → 2.61 (**−0.83**, CI −0.86 to −0.79) |
| Full panel (36 / 32), intensity → rank | 2.17 → 2.34 (**+0.17**) | 2.52 → 2.65 (**+0.13**, CI 0.11 to 0.14) |
| Full panel, marker left out | +0.08 | +0.06 (CI 0.05 to 0.07) |
| Full panel, beyond cell type | +0.16 | +0.11 (CI 0.10 to 0.12) |
| Full panel, beyond type, marker left out | +0.08 | +0.05 (CI 0.04 to 0.06) |
| Tracking of tissue clustering, marker left out (ρ rank vs intensity) | 0.59 vs 0.60 | 0.68 vs 0.70 (diff −0.02, CI −0.03 to −0.02) |

**Panel-size curve** (rank − intensity):

| Markers | 6 | 12 | 18 | 24 | 31 | full |
|---|---|---|---|---|---|---|
| Danenberg | −0.84 | −0.25 | −0.04 | +0.07 | +0.14 | +0.17 (36) |
| Jackson | −0.93 | −0.29 | −0.07 | +0.04 | +0.12 | +0.13 (32) |

**Per cohort** (full panel): Basel +0.12 (CI 0.11 to 0.13) and Zurich +0.15 (0.13 to 0.18). With the marker left out: Basel +0.06, Zurich +0.09. The crossover sits between 18 and 24 markers in both cohorts.

**Per marker (full32):** rank wins for 29 of 32 markers. The exceptions are Ki-67 (−0.08; also the only loser in Danenberg), CD68 (−0.06) and CD3 (−0.01). The largest gains are GATA3, CK19 and HER2 (+0.24 to +0.31).

## Reading
Every Danenberg conclusion replicates, with the same sign and a similar size:
1. **Rank beats intensity only with large panels.** Rank wins from about 24 markers, with +0.12 at the CyTOF panel size (31), and loses clearly with 6.
2. **About half the advantage is built in.** With the marker left out, the advantage roughly halves.
3. **The advantage holds beyond cell type.**
4. **It does not reflect tissue biology.** With the marker left out, rank tracks real tissue clustering slightly worse than intensity.

The only difference is magnitude. At 6 markers rank loses more in Jackson (−0.83 vs −0.62), and with the full panel it gains a little less (+0.13 vs +0.17), which fits a 32-marker panel against 36.

**Known issue:** because of a notebook-generator bug, the Kaggle HTML export failed on a markdown cell. The analysis cells ran and the outputs are complete; the repo notebook is fixed.
