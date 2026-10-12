# Cell-table follow-up: neighbour interactions (§14 Step 2) and three checks

**Run:** Kaggle kernel `cell-table-followup`, version 1, 2026-10-09. CPU.
**Notebook:** `Scripts/CellTableFollowup.ipynb`.
**Data and settings:** the same cells and settings as `cell-table-map`. Patient-bootstrap CIs, 1000×.

## Fix 1: H3K27me3 is mostly explained by cell identity, not nuclear content (Jackson)
Cell-level R², out of fold:

| Inputs | All cells | Tumour cells |
|---|---|---|
| Nuclear content only (H3, DNA1, DNA2, area) | 0.19 [0.11, 0.24] | 0.13 [0.05, 0.19] |
| Danenberg's channels (nuclear + 11 shared) | 0.57 | 0.60 |
| Full Jackson panel | 0.63 [0.61, 0.66] | 0.65 [0.62, 0.69] |

- The protein markers add +0.45 R² over nuclear content (CI 0.39–0.52), so the mark follows cell identity and state.
- About 35% of its variance is not predictable from the panel.
- The H3K27me3/H3 R² of 0.92 in `cell-table-map` was an artefact, as flagged there.

## Fix 2: the SMA collapse is biological or technical, not a scaling problem
- With rank harmonisation instead of median / p5–p95, SMA is still not recovered in Danenberg (r 0.02 → 0.06).
- SMA relates to the other shared markers differently in the two cohorts:

| Correlation with SMA (all cells) | Jackson | Danenberg |
|---|---|---|
| panCK | −0.25 | −0.03 |
| H3 | −0.15 | +0.24 |
| CD45 | −0.02 | +0.22 |

- **In Jackson,** SMA is the "neither epithelial nor immune" signal. The model predicts it mainly from CK8/18, CD20, CD68 and panCK.
- **In Danenberg,** SMA has a compressed range (stromal median arcsinh 0.77 vs 1.48 in Jackson) and tracks nuclear content.
- **Rank harmonisation helps most other markers:** median cross-cohort r goes from 0.61 to 0.64. The biggest gains are CD20 (0.34 → 0.64), Ki-67 (0.23 → 0.32) and CK5 (0.24 → 0.34). It should be the default for cross-cohort transfer.

## Fix 3: Danenberg's "cd45" behaves like pan-CD45, not CD45RO
- The channel is high on every immune lineage:

| Phenotype | Mean arcsinh |
|---|---|
| B cells | 1.66 |
| CD8 T cells | 1.63 |
| CD4 T cells | 1.72 |
| Treg/Tex | 2.17 |
| Macrophages & granulocytes | 2.16 |

- Epithelial background is 1.0–1.4.
- B cells are CD45RO-negative, so a high B-cell signal argues for pan-CD45. That matches the panel's clone, 2B11.
- Its correlation with CD45RA in immune cells is only 0.27.
- **Reading:** the "CD45RO" in the table header is probably a label issue. Treating the channel as shared with Jackson's CD45 is reasonable.

## Step 2a: neighbour effects within each cell class are small
ΔR² from the mix of touching cells, or of non-touching cells within 30 µm, added to the cell's own markers. Fitted within each class.

| Cohort | Class | Markers | Ring gain (CI > 0) | Max ΔR² ring | Median ΔR² |
|---|---|---|---|---|---|
| Danenberg | epithelial | 16 | 3 | 0.030 (CK5) | 0.000 |
| Danenberg | immune | 24 | 6 | 0.015 | 0.001 |
| Danenberg | stromal | 16 | 7 | 0.008 | 0.001 |
| Jackson | epithelial | 24 | 3 | 0.047 (CA-IX) | 0.002 |
| Jackson | immune | 17 | 5 | 0.051 (CD20) | 0.002 |
| Jackson | stromal | 17 | 4 | 0.021 (fibronectin) | 0.000 |

Splitting by cell class does not reveal hidden large effects. The per-class gains are as small as the overall ones in `cell-table-map`.

## Step 2b: pre-specified neighbourhood tests
Readout: within-image r between the marker's residual (after the cell's own markers) and a neighbourhood feature.

| Test | Result |
|---|---|
| **Positive control:** tumour CA-IX vs distance to the nearest vessel (Jackson) | **r = +0.097 [0.069, 0.125]**, as expected: hypoxia grows away from vessels. Endothelial counts nearby give the matching negative sign. The method detects known spatial biology. |
| CD8 T cells: PD-1, ICOS, Ki-67 vs tumour contact or proximity (Danenberg) | Nothing: \|r\| < 0.01, every CI includes 0 |
| Tumour cells: HLA-DR vs nearby T cells | **+0.04** (contacts 0.041, within 30 µm 0.037, distance −0.038); consistent with IFN-γ-induced MHC-II |
| Tumour cells: HLA-ABC and β2M vs nearby T cells | +0.01 to +0.03, same direction across features |
| Macrophages: CD163 vs tumour proximity | **Opposite to expected:** CD163 is lower near tumour (contacts −0.040, within 30 µm −0.056, distance +0.074) |
| **H3K27me3 (nuclear-content residual) vs the cell's own CA-IX** (Jackson tumour cells) | **r = +0.20 [0.18, 0.22]**: hypoxic tumour cells carry more H3K27me3, as the KDM6A oxygen-sensing mechanism predicts |
| H3K27me3 (full-model residual) vs vessel distance or CA-IX-high neighbours | Not detected (\|r\| ≤ 0.02, mostly n.s.) |

## Reading
1. **Neighbourhood effects on a cell's markers are real but small,** |r| ≤ 0.1 and ΔR² ≤ 0.05. The method finds the expected ones: hypoxia vs vessel distance, MHC-II near T cells. Splitting by cell class adds nothing large.
2. **H3K27me3 is about 60% predictable from protein markers.** It goes up with the cell's own hypoxia marker, not with its neighbourhood. That is a cell-intrinsic link, testable on real data, and of interest for the lab.
   - **Caveat:** the nuclear-content residual still contains cell identity, so part of the 0.20 may be cell type. The next check is the partial correlation controlling for the other markers except CA-IX.
3. **Two issues from `cell-table-map` are resolved:**
   - Danenberg's CD45 behaves as pan-CD45.
   - The SMA failure comes from different SMA behaviour across cohorts, not from harmonisation. Rank harmonisation is better overall.
