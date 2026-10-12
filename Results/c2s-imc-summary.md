# C2S encodings tested on spatial IMC data

## Main effects in both cohorts
Settings for every row: each cell's 30 most similar cells ("profile neighbours"), with touching cells excluded, intensity encoding unless the row says rank. "6" = the 6 markers shared with the CyTOF panel; "full" = all protein markers (36 in Danenberg, 32 in Jackson); 31 = the CyTOF panel size. All differences have patient-bootstrap 95% CIs that exclude 0. The one exception is Jackson's 6-marker transfer (−0.012, CI −0.026 to 0.000).

| Effect | Danenberg 2022 | Jackson 2020 | Replicates? |
|---|---|---|---|
| Data | 1.06M cells, 704 images, 650 patients | 1.13M cells, 542 images, 349 patients | – |
| Profile neighbours closer in tissue than random cells (6 / full) | 9% / 15% | 13% / 23% | Yes (larger) |
| …left after matching cell type (6 / full) | 4% / 9% | 8% / 16% | Yes (larger) |
| Surroundings R² gained beyond cell type (6 / full) | +0.022 / +0.048 | +0.055 / +0.078 | Yes (larger) |
| Borrowing surroundings from other patients, R² (6 / full) | 0.015 / 0.045 | −0.012 / 0.025 | Yes (near zero) |
| Rank − intensity: same-type share among neighbours (every panel size) | −7 points | −7 points | Yes |
| Rank − intensity: surroundings R² at 31 markers | −0.011 (4% loss) | −0.023 (6% loss) | Partly (loss twice as large) |
| Rank − intensity: closeness at 31 markers | rank 1% closer | rank 1% further | **No** |
| kNN enrichment, rank − intensity (6 / 31 / full) | −0.62 / +0.14 / +0.17 | −0.83 / +0.12 / +0.13 | Yes |
| …with the scored marker left out (full) | +0.08 | +0.06 | Yes |
| …beyond cell type (full) | +0.16 | +0.11 | Yes |
| Markers where rank gives higher enrichment (full) | 35 of 36 (Ki-67 loses) | 29 of 32 (Ki-67, CD68, CD3 lose) | Yes |
| Enrichment follows tissue clustering, marker left out (ρ rank vs intensity) | 0.59 vs 0.60 | 0.68 vs 0.70 | Yes |
| cLISI, fine cell type, pooled 10 patients (6 / full; lower = types kept apart), intensity → rank | 4.24 → 5.59 / 3.30 → 4.02 | 4.50 → 5.56 / 3.19 → 3.80 | Yes |
| iLISI, patient, pooled 10 patients (6 / full; higher = patients mixed), intensity → rank | 3.87 → 5.15 / 3.36 → 3.90 | 3.99 → 4.99 / 3.03 → 3.55 | Yes |

LISI rows: 100 draws of 10 patients × 500 cells; brackets are ranges over draws, given in `lisi-imc/summary.md`. Every rank − intensity range excludes 0.

## Danenberg 2022

CyTOF breaks tissue into single cells, so the C2S project treats cells with similar marker profiles as each other's neighbours. We used spatial IMC data to check what these profile neighbours mean in intact tissue. The data were the Danenberg 2022 breast-cancer cohort: 1.06 million cells, 704 tumour images and 650 patients, with each cell's markers and position. We used the same two encodings as the CyTOF study: intensity (robust z-scores per image) and rank (each cell's markers ordered from highest to lowest, the C2S "sentence"). For every cell we found its 30 most similar cells. We then asked whether they sit close to it in the tissue, whether they share its surroundings (the mix of cell types around it), whether they are the same cell type, and whether they are enriched for cells high in the same marker, which is the kNN-enrichment score from the CyTOF study. Each answer was compared with random cells and with random cells of the same type. We repeated everything with panels of 6 to 36 markers: 6 markers are shared with the CyTOF panel, and 31 is the CyTOF panel size. Touching cells were excluded to guard against signal spillover. For enrichment we also left the scored marker out of the profile, to remove the part that is built in. All runs used Kaggle, with patient-level bootstrap confidence intervals.

Look-alike cells were only slightly closer in the tissue than random cells: 9% closer with the 6 shared markers and 15% with all 36. Most of that came from being the same cell type. Beyond cell type, look-alikes said little about a cell's real surroundings, and borrowing surroundings from look-alike cells in other patients almost did not work (R² 0.015 with 6 markers, 0.045 with 36). With small panels (6–12 markers), rank lost information on every measure. From about 18 markers, rank kept tissue information as well as intensity: at 31 markers the loss in predicting surroundings was about 4%, and look-alikes were slightly closer. At every panel size, however, rank grouped same-type cells about 7 points less well, which matches the breast-cancer CyTOF observation that intensity beats rank. Rank gave higher kNN enrichment of marker-high cells from about 24 markers: +0.14 at 31 markers, with 35 of 36 markers favouring rank and Ki-67 the exception. About half of that advantage remained with the marker left out, and it held beyond cell type, but it did not follow how marker-high cells actually cluster in the tissue.

These results suggest that, in CyTOF, profile neighbours should be described as cells in a similar state, mostly of the same type, and not as spatial neighbours. Inferring tissue context for CyTOF cells from look-alike IMC cells is not supported. Rank and intensity appear to capture different kinds of similarity. Rank groups cells by which markers are their highest, a cell state, whereas intensity groups them more by lineage. This would explain both rank's higher enrichment and its lower cell-type purity. Rank therefore looks like a reasonable encoding for panels of about 24 markers or more, where it loses no tissue information and sharpens state-based grouping, but not for small panels. The main caveat is that this is breast cancer, whereas most C2S results come from PDAC and bone marrow. A PDAC IMC replication (Zenodo access pending) is needed before generalising.

## Jackson 2020 (replication)

We repeated both analyses, unchanged, on a second breast-cancer IMC cohort: Jackson 2020 (Basel and Zurich tissue microarrays). After the same filters it had 1.13 million cells, 542 tumour images and 349 patients: 278 Basel patients with about one core each, and 71 Zurich patients with several cores each. Cell tables, positions and cell masks came from the public Zenodo release, and cell types were Jackson's 27 published clusters. The code, settings and decision rules were those of the Danenberg run. Four things differed:
- the Jackson panel has 32 protein markers, so the full panel was 32 rather than 36. The 6 CyTOF-shared markers and the random panels of 6–31 markers were used as before;
- Jackson published no list of touching cells, so we derived one from the cell masks. It is slightly looser than Danenberg's (5.1 vs 5.9 touching cells per cell);
- Jackson's published values are not corrected for signal spillover;
- Jackson has no hot-aggregate flag, so no cells were removed for that reason.

Most results replicated. Look-alike cells were again only modestly closer in the tissue than random cells: 13% closer with the 6 shared markers and 23% with all 32. Much of that came from being the same cell type (8% and 16% remained after matching type). Beyond cell type, look-alikes added little about a cell's surroundings (R² +0.055 and +0.078). Borrowing surroundings from look-alike cells in other patients failed even more clearly than in Danenberg: R² was −0.012 with 6 markers and 0.025 with 32, and about zero beyond cell type. Rank lost information with small panels, and the loss shrank as the panel grew. Rank again grouped same-type cells about 7 points less well at every panel size. Rank gave higher kNN enrichment from about 24 markers: +0.12 at 31 markers, with 29 of 32 markers favouring rank and Ki-67 again the clearest exception. About half of the advantage remained with the marker left out, and it held beyond cell type, but it again did not follow how marker-high cells cluster in the tissue. The enrichment result held in the Basel and Zurich patients separately. Two results did not fully replicate:
- rank never caught up with intensity on closeness. At 31 markers its look-alikes were about 1% further away, where in Danenberg they were 1% closer;
- rank's loss in predicting surroundings at 31 markers was 6% rather than 4%.

All within-image effects were 1.5–2 times larger in Jackson. We have not established why. Candidate reasons are the uncorrected spillover, the larger images and the different cell-type definitions. Excluding touching cells reduced the effects less than in Danenberg, so leftover spillover alone seems unlikely to explain it.

The replication supports the Danenberg conclusions. In CyTOF, profile neighbours are best described as cells in a similar state, mostly of the same type, and not as spatial neighbours, and inferring tissue context for CyTOF cells from look-alike IMC cells is not supported in either cohort. The rank-versus-intensity pattern also holds in both. Rank sharpens state-based grouping (higher enrichment from about 24 markers) at the cost of lineage grouping (7 points lower cell-type purity), and it is clearly worse with small panels. The one claim to soften is that rank "loses no tissue information" at CyTOF-size panels. In Jackson it loses a little: about 6% of the surroundings signal, and slightly less close look-alikes. Rank therefore remains a reasonable encoding for panels of about 24 markers or more, with a small tissue-information cost rather than none. Both cohorts are breast cancer, so the PDAC replication is still needed before generalising.

## cLISI and iLISI (both cohorts)

We also scored the pooled profile space with the standard integration metrics (LISI, as in Harmony; perplexity 30). Each draw pooled 500 cells from each of 10 random patients, as when CyTOF samples are pooled. We computed cLISI (how many cell types are mixed among a cell's neighbours; lower is better) and iLISI (how many patients are mixed; higher means better mixing). The result was the same in both cohorts. Rank had higher cLISI than intensity (full panel: 3.30 → 4.02 in Danenberg, 3.19 → 3.80 in Jackson) and higher iLISI (3.36 → 3.90 and 3.03 → 3.55). Both gaps were largest with 6 markers and shrank but did not close at CyTOF panel size. Rank's gain in patient mixing was about as large as its loss in cell-type separation (about 0.12 on a 0–1 scale). It therefore looks like a generally "blurrier" neighbourhood, not a selective removal of patient batch effects. This matches rank's lower cell-type purity above. In Jackson, the Basel and Zurich cohorts were already well mixed with intensity (iLISI 1.60 of a maximum 2), and rank added little.

Details: `profile-vs-physical/summary.md`, `knn-enrichment-imc/summary.md`, `profile-vs-physical-jackson/summary.md`, `knn-enrichment-jackson/summary.md`, `lisi-imc/summary.md`.
