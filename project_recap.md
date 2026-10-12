# IMC project recap (updated 2026-10-08)

## Part 1: Virtual staining across cohorts (Jackson 2020 → Danenberg 2022)

**Intro.** The headline metric in IMC virtual staining, pooled pixel correlation, can be raised by preprocessing without any gain in spatial accuracy. In our cross-cohort transfer, the low-pass preprocessing used by ImmuVis nearly doubled the between-image correlation for CD3 (0.31 → 0.60), while the within-image correlation barely moved (0.09 → 0.11). A cheap correction built from the nuclear channels and trained on one cohort matched ImmuVis-style accuracy (mean r 0.27 vs 0.28) and gave more realistic images (mean FID 202 vs 213). Current methods (VirTues, EVA, ImmuVis) handle mismatched panels with foundation models pretrained on many cohorts and report mainly pooled correlation. We tested how far a compact model trained on one cohort gets on a new cohort, whether preprocessing rather than scale drives that transfer, and whether a good correlation means real spatial recovery or only the right overall level per image.

**Method.** We trained ResNet50-UNets on Jackson 2020 and tested them, without fine-tuning, on Danenberg 2022 (794 images, 718 patients), predicting each of 11 shared markers (ERα, HER2, Ki-67, CD3, CD20, CD45, CD68, CK5, CK8/18, panCK, SMA) from the other 10. We first compared three designs: one model per marker on PCA-compressed inputs, one joint model for all markers, and a joint model with a HyperConv stem that knows which markers it receives. With the joint model (without HyperConv) fixed and trained for up to 50 epochs, we then compared three preprocessings. The first was an additive residual after regressing each channel on H3, DNA1 and DNA2. The second was a multiplicative "gamma" correction ported from a CyTOF normalisation method, which removes each marker's dependence on a nuclear-density axis. The third was the ImmuVis pipeline: arcsinh, Butterworth low-pass and percentile min-max. We scored pixel Pearson and FID. We then split each marker's correlation into a between-image part (overall level) and a within-image part (spatial pattern), using a 1000× patient bootstrap and a paired test of whether a preprocessing shifts one part more than the other.

**Results.** Mean Pearson was 0.29 for the per-marker models, 0.23 for the joint model and 0.30 for HyperConv; the HyperConv figure comes from a 10-epoch test run. In the 50-epoch preprocessing comparison, ImmuVis-style had the highest mean Pearson (0.284), gamma was close (0.274) with the best FID (202 vs 213), and the residual was weakest on both (0.255, FID 236). No preprocessing won every marker. No marker's correlation came only from overall level: even ERα, the most level-driven marker (between-image r 0.82), kept a within-image r of 0.52. Markers fell into the same tiers under every preprocessing:
- High: ERα and panCK (within-image r ≈ 0.45).
- Mid: HER2, CD45, CK8/18 and SMA (0.25–0.33).
- Low: CD20, Ki-67, CD3, CK5 and CD68 (≤ 0.17).

The low tier is also the low-signal tier. The ImmuVis low-pass filter raised between-image agreement for the weak immune markers without raising within-image agreement.

**Discussion.** In this setting, a compact single-cohort model with cheap preprocessing reaches the accuracy of foundation-model-style preprocessing, so preprocessing matters at least as much as scale. We have not yet compared against the foundation models themselves, whose published numbers come from a different cohort and task. Because pooled correlation flatters pipelines that smooth images, within-image correlation and FID should be reported alongside it. Gamma's FID advantage rests on one seed per arm and needs repeat seeds or a bootstrap before a paper can rely on it. The well-predicted markers (ERα, panCK, HER2, CD45) look imputable in the spatial sense, which could guide panel design. A poorly predicted marker, though, may be noisy rather than unique, so predictability has to be read together with signal quality.

## Part 2: Rank vs intensity on IMC cells (side project for the CyTOF/C2S paper)

**Intro.** In CyTOF, marker-high cells group more tightly under the rank encoding than under intensity. This advantage replicates in two breast-cancer IMC cohorts, but only with panels of about 24 markers or more. It costs about 7 points of cell-type purity at every panel size and brings no clinical gain over cohort-normalised intensity. The CyTOF project treats cells with similar marker profiles as neighbours, encoding each cell either by its intensities or by the rank order of its markers (the encoding used by Cell2Sentence). Spatial IMC lets us check what these profile neighbours mean in intact tissue. We tested whether look-alike cells sit close together and share surroundings, which encoding better groups marker-high cells and cell types, and whether rank-based cell states carry more clinical information.

**Method.** We built cell tables on Kaggle from the public releases of Danenberg 2022 (1.06M cells, 704 images, 650 patients) and Jackson 2020 (1.13M cells, 542 images, 349 patients). Each cell was encoded by per-image robust z (intensity) or by the rank order of those z-scores (rank). We then took its 30 nearest cells in that space, excluding touching cells to limit spillover. On panels from 6 markers up to the full panel, we measured:
- physical closeness;
- how well neighbours predict a cell's surroundings beyond cell type, and whether that transfers across patients;
- kNN enrichment of marker-high cells, with and without the scored marker;
- same-type share;
- cLISI and iLISI.

For clinical value, we built k-means cell states for each encoding, described each patient by state composition, and predicted subtype, receptor status, grade and survival with cross-validation. All comparisons used patient-bootstrap CIs and decision rules fixed before the run. The Jackson runs reused the Danenberg code unchanged.

**Results.** Look-alike cells were only 9–15% closer in tissue than random cells in Danenberg (13–23% in Jackson), mostly because they were the same cell type, and borrowing surroundings from look-alikes in other patients almost failed (R² ≤ 0.045). Rank beat intensity on kNN enrichment from about 24 markers (+0.14 at 31 markers in Danenberg, +0.12 in Jackson). It did so for 35 of 36 and 29 of 32 markers, with Ki-67 losing in both cohorts. With 6 markers rank lost clearly (−0.62 and −0.83). About half the gain disappeared when the scored marker was left out, and the gain did not follow how marker-high cells cluster in the tissue. Rank lowered the same-type share by about 7 points at every panel size and raised cLISI and iLISI by similar amounts. Clinically, rank never beat cohort-normalised intensity and lost HER2 (AUC 0.94 → 0.78). Its only gain, over per-image-normalised intensity in Jackson (34–61% of the gap closed), came from more comparable cell states rather than recovered marker levels, and it did not replicate in Danenberg.

**Discussion.** In CyTOF, profile neighbours are best read as cells in a similar state, mostly of the same type, not as spatial neighbours. Inferring tissue context for CyTOF cells from look-alike IMC cells is not supported, so that idea was dropped. Rank is a legitimate encoding with a clear trade-off. It groups cells that share their top markers more tightly, but groups lineages less well and gives blurrier neighbourhoods. It suits panels of about 24 markers or more, not small ones. It is not a route to better clinical features, and it loses magnitude signals such as HER2 amplification. Its one niche is data that must be normalised per sample, where rank-based states can be more comparable across patients; so far this holds in one of two cohorts. Both cohorts are breast cancer, and a PDAC IMC replication is waiting on data access.

## Experiments in order
1. **Jun–Aug 2025, Danenberg exploration:** compared normalisations, PCA/UMAP and an XGBoost cell-type classifier with SHAP. Exploratory, with figures only and no recorded conclusion.
2. **Sep–Oct 2025, within-cohort pilots** (one ResNet-UNet per marker, 4-fold CV, 9 markers, each cohort alone): only panCK was predicted well (r 0.49 Danenberg, 0.64 Jackson). Every other marker had r ≤ 0.16.
3. **Nov 2025, Jackson epigenetic marks** (H3K27me3, pHH3) predicted from the other channels: failed, with negative correlations (−0.07 to −0.21).
4. **Dec 2025, marker correlations and stain-area statistics** in both cohorts: used to pick the 11 markers shared by Jackson and Danenberg.
5. **Dec 2025, within-cohort models on the shared markers only:** failed, with r between −0.17 and 0.16.
6. **Dec 2025–Jan 2026, first Jackson → Danenberg transfer** (11 per-marker models, residual + PCA inputs): the first working result, with mean r 0.29 and FID 252 (ERα 0.58, panCK 0.57, HER2 0.45).
7. **May 2026, model variants** (pretrained first layer; joint model without PCA): r 0.28 and 0.23. Neither beat the per-marker baseline.
8. **Jun 2026, HyperConv test run** (joint model with a marker-aware stem, 10 epochs): mean r 0.30 and FID 234, the best so far but undertrained.
9. **Jul 2026, preprocessing comparison** (joint model without HyperConv, up to 50 epochs): residual, gamma and ImmuVis-style gave r 0.255, 0.274 and 0.284, and FID 236, 202 and 213.
10. **Jul 2026, abundance vs structure, E1** (residual arm; ERα, HER2, panCK): no marker was level-only. ERα had between-image r 0.82 and within-image r 0.52.
11. **Jul 2026, abundance vs structure, E2** (3 preprocessings × 11 markers, patient bootstrap): predictability tiers were stable and tracked signal quality. The low-pass filter inflated between-image r for weak immune markers.
12. **27 Sep 2026, `danenberg-cells-prep`:** built the Danenberg cell table, masks, contact graph, clinical table and frozen patient split on Kaggle.
13. **27 Sep 2026, `profile-vs-physical`** (Danenberg): look-alike cells are only slightly closer in tissue, mostly because they share a cell type, and cross-patient transfer of surroundings is near zero. Rank costs 7 points of type purity at every panel size.
14. **28 Sep 2026, `knn-enrichment-imc`** (Danenberg): rank beats intensity from about 24 markers (+0.14 at 31) and loses at 6 (−0.62). Half the gain is built in.
15. **28 Sep 2026, Jackson replication** (`jackson-cells-prep`, `profile-vs-physical-jackson`, `knn-enrichment-jackson`): most results replicated, with effects 1.5–2× larger. Rank's closeness never caught up with intensity.
16. **28 Sep 2026, `lisi-imc`** (both cohorts): rank has higher cLISI and iLISI, so its neighbourhoods are blurrier rather than batch-corrected.
17. **29 Sep 2026, `rank-tradeoff-markers`:** per marker, rank raises enrichment for most markers (35/36 Danenberg, 27/32 Jackson with CIs excluding 0) and lowers the same-type share for every marker.
18. **29 Sep 2026, `rank-utility-clinical`** (pre-registered): rank gives no clinical gain over cohort-normalised intensity, loses HER2 and adds nothing to survival. Tumour-state entropy is not prognostic.
19. **4 Oct 2026, `rank-recovery`:** rank closes 34–61% of the per-sample-normalisation gap in Jackson, only through cell-state composition, and not in Danenberg. The claim that rank recovers marker levels fails.
20. **8 Oct 2026, `cell-table-map`** (both cohorts, full panels):
    - Immune cells touch tumour cells about half as often as chance in about 90% of images.
    - Most markers are 40–90% predictable from the cell's own other markers. Neighbours add at most about 6% R².
    - Jackson tumour-state markers imputed into Danenberg add no survival information.
21. **9 Oct 2026, `cell-table-followup`:**
    - Neighbour effects within each cell class stay small (ΔR² ≤ 0.05).
    - A known effect is recovered: hypoxia grows away from vessels, r 0.10.
    - H3K27me3 follows cell identity (R² 0.63 vs 0.19 from nuclear content) and rises with the cell's own CA-IX (r 0.20).
22. **9 Oct 2026, `kill-test`** (existing image checkpoints vs cell-table baselines on Danenberg cells):
    - A per-cell regression on the Jackson table beats the image models for 10 of 11 markers (e.g. panCK R4 0.84 vs 0.25–0.54).
    - The R5 rung failed its positive controls.
    - Danenberg ERα saturates in the model's target space (the two cohorts use different ER antibodies), which undercuts the earlier ERα results.
23. **9 Oct 2026, `model-comparison`** (table model, MLP, GCN, GraphSAGE and ResNet-UNet; Jackson → Danenberg):
    - Every per-cell model beats the ResNet-UNet by about 2× (median R4 0.42–0.46 vs 0.19–0.25).
    - Within Jackson, graph context adds a little.
    - Across cohorts, a plain MLP is best. Item 24 shows this was a tie.
24. **9–10 Oct 2026, `model-comparison-reps`** (2 more replicates, each with a new Jackson patient split and new seeds):
    - The 2× gap between per-cell and image models holds in all 3 replicates.
    - The GCN is again the weakest per-cell model.
    - Table, MLP and GraphSAGE are tied: median R4 0.63–0.64 in Jackson and 0.44–0.47 in Danenberg, which is within the replicate spread.
    - Neighbour context never hurts within Jackson. It gives small, consistent gains only for CD45, SMA and Ki-67 or panCK.

Details: `Results/<experiment>/summary.md` and `Results/c2s-imc-summary.md`. The previous long version is in git: `git show HEAD:session_summary.md`.
