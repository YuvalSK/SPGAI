# cLISI and iLISI of the C2S encodings (Danenberg and Jackson IMC)

**Run:** Kaggle kernel `lisi-imc`, version 1, 2026-09-28 (35.7 min).
**Notebook:** `LisiIMC.ipynb`.
**Data:** the same cells, encodings and marker sets as ProfileVsPhysical; `spaces.json` is identical to both earlier runs.
- Danenberg: 1.06M cells, 650 patients.
- Jackson: 1.13M cells, 349 patients.

**LISI:** the `harmonypy.compute_lisi` algorithm (Korsunsky 2019): perplexity 30, 90 neighbours. My port matches harmonypy 0.0.10 to 4e-15. The value is the effective number of label categories among a cell's profile neighbours.
- **cLISI**, with cell type as the label: lower means types are kept apart.
- **iLISI**, with patient as the label: higher means patients are mixed.

**Design:**
- One draw = 10 random patients × 500 cells, pooled as in CyTOF. There are 100 draws for the 6 CyTOF markers and the full panel, and 20 for the random panels.
- Rank and intensity are compared on the same cells in each draw.
- Jackson also has draws of 5 Basel + 5 Zurich patients, with cohort as the label.
- Values: the median over cells, then the mean over draws. Brackets are the 2.5–97.5% range over draws.
- "Relative" rescales against the same neighbours with shuffled labels: 1 = types perfectly separated (cLISI) or patients perfectly mixed (iLISI).

## Results
| | Danenberg, intensity → rank | Jackson, intensity → rank |
|---|---|---|
| **cLISI, fine type** (6 markers) | 4.24 → 5.59 (+1.35 [1.04, 1.78]) | 4.50 → 5.56 (+1.06 [0.80, 1.40]) |
| **cLISI, fine type** (full panel) | 3.30 → 4.02 (+0.72 [0.32, 1.08]) | 3.19 → 3.80 (+0.61 [0.29, 0.83]) |
| cLISI, broad class (full panel) | 1.26 → 1.50 (+0.24) | 1.30 → 1.49 (+0.18) |
| Type separation, relative (full panel) | 0.61 → 0.47 | 0.62 → 0.50 |
| **iLISI, patient** (6 markers) | 3.87 → 5.15 (+1.29 [1.00, 1.55]) | 3.99 → 4.99 (+1.01 [0.73, 1.30]) |
| **iLISI, patient** (full panel) | 3.36 → 3.90 (+0.54 [0.27, 0.87]) | 3.03 → 3.55 (+0.52 [0.22, 0.76]) |
| Patient mixing, relative (full panel) | 0.47 → 0.59 | 0.40 → 0.51 |
| iLISI, Basel vs Zurich (6 / full; max 2) | – | 1.77 → 1.87 (+0.10 [0.05, 0.16]) / 1.60 → 1.66 (+0.06 [−0.02, 0.15]) |

**Panel-size curve** (rank − intensity):

| Markers | 6 | 12 | 18 | 24 | 31 | full |
|---|---|---|---|---|---|---|
| cLISI, D | +1.32 | +0.87 | +0.81 | +0.81 | +0.80 | +0.74 |
| cLISI, J | +1.12 | +0.64 | +0.68 | +0.62 | +0.61 | +0.60 |
| iLISI, D | +0.99 | +0.62 | +0.64 | +0.64 | +0.63 | +0.55 |
| iLISI, J | +0.93 | +0.62 | +0.63 | +0.57 | +0.49 | +0.49 |

## Reading
1. **Same result in both datasets.** At every panel size, rank has higher cLISI (cell types more mixed among profile neighbours) and higher iLISI (patients more mixed).
2. **cLISI agrees with the earlier purity result.** Rank keeps cell types apart less well, about 0.14 lower relative separation with the full panel in both cohorts. That matches the 7-point lower same-type share (T1) in ProfileVsPhysical. The penalty shrinks from 6 to 12 markers, then stays roughly flat up to the full panel, so a CyTOF-size panel does not remove it.
3. **iLISI: rank mixes patients better, but this is not evidence that it removes batch effects.**
   - The gain in patient mixing (+0.11–0.12 relative) is about as large as the loss in type separation (−0.12 to −0.14). So rank's neighbourhoods look generally "blurrier", not selectively cleaned of patient effects.
   - Patients also differ biologically (tumour subtype and composition), so higher iLISI is not automatically better here.
4. **Basel vs Zurich (Jackson):** the two cohorts already mix fairly well with intensity (1.60 of a maximum 2, with the full panel). Rank adds a little, but with the full panel the CI includes 0.

**Caveat:** the fine phenotypes were defined by clustering intensity data, so cLISI favours intensity. The broad-class cLISI is less sensitive to this and gives the same pattern. iLISI does not use cell-type labels at all.
