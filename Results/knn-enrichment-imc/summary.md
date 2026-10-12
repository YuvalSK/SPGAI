# kNN enrichment of marker-high cells: rank vs intensity (Danenberg IMC)

**Run:** Kaggle kernel `knn-enrichment-imc`, version 1, 2026-09-28.
**Data:** 1,062,219 cells, 704 tumour images, 650 patients.
**Score:** the CyTOF definition. "High" = robust z > 1; neighbours = the 30 nearest in the encoded profile; score = fraction of a high cell's neighbours that are also high, divided by the overall share of high cells. So 1 = no enrichment.
**Stats:** patient bootstrap, 95% CI. The marker sets are the same as ProfileVsPhysical version 3 (identical `spaces.json`).

## Results
| | 6 CyTOF-matched markers | 36 markers |
|---|---|---|
| Enrichment, intensity → rank | 3.13 → 2.51 (**rank −0.62**) | 2.17 → 2.34 (**rank +0.17**, CI 0.16–0.18) |
| Same, with the marker left out of the profile | rank −0.25 | rank +0.08 (CI 0.08–0.09) |
| Beyond cell type (minus the same-type null) | rank −0.46 | rank +0.16 |
| Beyond cell type, marker left out | rank −0.09 | rank +0.08 |
| Tracks real tissue clustering (Spearman with physical enrichment), rank vs intensity | 0.39 vs 0.33 | 0.54 vs 0.52 |
| Same, marker left out | 0.67 vs 0.69 | 0.59 vs 0.60 |

**Panel-size curve** (rank − intensity):

| Markers | 6 | 12 | 18 | 24 | 31 | 36 |
|---|---|---|---|---|---|---|
| rank − intensity | −0.84 | −0.25 | −0.04 | +0.07 | +0.14 | +0.17 |

**Per marker (36 markers):** rank wins for 35 of 36 markers, by +0.03 to +0.33. The one exception is **Ki-67** (2.12 → 1.72).

## Reading
1. **Your CyTOF finding replicates, but only with large panels.** Rank beats intensity from about 24 markers, and by +0.14 at 31 markers (the CyTOF panel size). With 6 markers it clearly loses.
2. **About half the advantage is built in.** When the marker itself is left out of the profile, rank still wins, but by half as much (+0.08).
3. **It is not just cell type.** The advantage holds after the same-type null, too.
4. **It is not tissue biology.** With the marker left out, rank's enrichment tracks real tissue clustering slightly *worse* than intensity does (0.59 vs 0.60). Rank pulls together cells that share their top markers; that grouping is real in marker space, but it is not a closer match to how those cells sit in the tissue.
5. **This fits ProfileVsPhysical version 3.** There, rank grouped same-type cells about 7 points less well. So rank groups cells by "which markers are my highest" (a state) more than by lineage. That raises marker-high enrichment and lowers type purity.
