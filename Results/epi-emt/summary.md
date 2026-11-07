# H3K27me3 along the epithelial–mesenchymal axis (`future_plan.md` §19, pre-registered 2026-11-07)

**Run:** local CPU, 2026-11-07.
- **Scripts:** `epi_emt.py` (pre-registered analysis), `posthoc_generic.py` (post-hoc, labelled as such), `plot_epi_emt.py`.
- **Data:** `jackson-cells-prep` (epithelial metaclusters 14–27 in tumour images) and `keren-cells-prep` (`tumor_yn == 1`). Patients with ≥ 100 cells.
- **Statistic:** rank-based partial correlation within each patient.
  - Variables are converted to normal scores within each image.
  - Covariates: total H3, DNA1/2, log area and image. Keren has no total H3, so it uses dsDNA, log area, and the mass-neighbour / oxide channels pan-keratin, phospho-S6 and CD68.
  - Summary: mean Fisher z over patients, with a 1000× patient-bootstrap CI.

## Pre-registered verdict: passes, "replicated across hospitals and technologies"
`decision.json`: D1, R1, R2, C0, C1 and C2 (all three variants) pass. Sign: negative (more mesenchymal, less H3K27me3). Band: substantive.

| Test | Basel (264 patients, 0.40M cells) | Zurich (71, 0.19M) | Keren (40, 0.11M) |
|---|---|---|---|
| Primary axis (M = vimentin, Twist, Slug; E = E/P-cadherin, pan-CK, CK8/18, CK19) | **−0.335** [−0.357, −0.314]; 97% of patients negative | **−0.342** [−0.373, −0.306]; 99% | – |
| Vimentin − keratin axis | −0.342 | −0.306 | **−0.060** [−0.106, −0.011] (keratin = K6/K17) |
| + cell-cluster adjustment | −0.272 | −0.301 | – |
| + neighbourhood adjustment | −0.297 | −0.316 | (exploratory) −0.027 [−0.062, +0.013] |
| C1, total H3 instead of H3K27me3 | −0.122 (passes ≤ half) | −0.122 | – |
| C0, shuffled axis | −0.003 | +0.009 | −0.002 |

The profile over axis deciles is monotonic, with no hybrid-state bump. It is the same in Basel and Zurich (`deciles.csv`).

## But the effect is carried by the epithelial side, not by mesenchymal gain (exploratory and post-hoc)
1. **Per marker, Jackson:** H3K27me3 rises with the keratins (pan-CK +0.55, CK19 +0.53, CK8/18 +0.48) and E/P-cadherin (+0.29).
   - Vimentin is flat: −0.007 in Basel, +0.023 in Zurich.
   - Twist and Slug go the *opposite* way to the hypothesis (+0.15 to +0.25).
2. **A generic per-cell signal factor** (post-hoc, `posthoc_generic.csv`).
   - In Jackson tumour cells, H3K27me3 correlates positively with almost every marker, including CD3, CD45 and CD20 (+0.12 to +0.16). The mean of 18 off-target markers gives r +0.37, and total H3 does the same against it (+0.43 to +0.47).
   - So part of every per-cell correlation in Jackson is technical: section thickness, ablation, segmentation.
   - After adjusting for this factor:
     - the axis effect stays (−0.33 / −0.33);
     - the keratins stay strong (+0.37 to +0.48);
     - E/P-cadherin drops to +0.14 / +0.17;
     - vimentin is −0.06 (Basel) and −0.02 n.s. (Zurich);
     - Twist and Slug are ±0.08.
3. **The Keren replication is weak and rests on a different marker.**
   - The Keren signal comes from vimentin (−0.08), not from keratins (K6 0.00, K17 +0.05).
   - It loses significance with the neighbourhood adjustment (−0.027, CI includes 0), so border spillover from stroma is a live explanation.
   - Without the crosstalk covariates it is −0.10. With the generic covariate it is −0.050 [−0.094, −0.005].
4. **Exploratory strata:**
   - Ki-67: −0.09 / −0.10. Proliferating cells carry less H3K27me3, consistent with dilution at replication.
   - Subtypes, both Jackson sets: HR+HER2− −0.37, HR+HER2+ −0.31, HR−HER2+ −0.32, TNBC −0.24 (all CI < 0).
   - Between patients (raw patient means): Basel ρ +0.18 (p 0.004), Zurich −0.09 (n.s.). There is no consistent between-patient relation.
   - Keren H3K9ac on the axis: +0.02 (n.s.).

## Reading
- **What the pre-registered rule says:** the test passes in all three sets.
- **What the data support:**
  - Within IMC, in two hospitals, tumour cells with more keratin carry more H3K27me3. Per keratin r ≈ +0.4 to +0.5, beyond nuclear content, cell cluster, neighbourhood and a generic per-cell signal factor.
  - Mesenchymal markers add little.
  - The cross-technology replication is weak (−0.06), driven by vimentin rather than keratins, and not robust to the neighbourhood adjustment.
- **So:** this is a robust **"H3K27me3 tracks epithelial (keratin) differentiation"** result in Jackson, not a demonstrated **EMT** result. The pre-registered axis could not tell loss of E from gain of M. That was a design gap, now visible.
- **Next tests that would separate these:**
  - (a) Pre-register an M-only test, vimentin among keratin-positive tumour cells, and an E-only test, as separate claims.
  - (b) Check whether the keratin relation is cytoplasm-volume / segmentation driven, using eroded nuclear-only H3K27me3. This needs Jackson pixel features for H3K27me3, which `pixel-features` did not compute.
  - (c) Orthogonal data, O1 (CCLE chromatin) and O2 (METABRIC EZH2), as pre-registered.

Figure: `fig_epi_emt.png` / `.pdf`.
- Left: the pre-registered tests.
- Middle: per marker, after the generic adjustment (Jackson).
- Right: H3K27me3 over axis deciles.
