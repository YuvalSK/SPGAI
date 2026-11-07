# Novae (graph foundation model) on IMC: niche features for cell typing, and cohort alignment

**Run:** Kaggle kernel `novae-cells`, version 5, 2026-11-04/05. 2×T4, 405 min, 3 replicates, no units skipped.
- Versions 1–3 failed on installation and a smoke-only learning-rate schedule bug. Version 4 was the smoke run (`smoke/`).
- **Notebook:** `Scripts/NovaeCells.ipynb`.
- **Pre-registration:** `future_plan.md` §18.9 (2026-11-04).
- **Large outputs:** replicate-0 latents and domains in `IMC data/<cohort>/novae-cells/`.

**Setup:**
- Novae 1.1.2, trained from scratch on the 15 `cell-type-graph` inputs:
  - Delaunay graph within each image, edges ≤ 40 µm (mean degree 5.8, median edge 13 µm);
  - fitted per replicate on the train + val cells of both cohorts (1.69–1.79M cells), without labels.
- Two variants:
  - `[cohort]`: one Novae "slide" per cohort;
  - `[img]`: one slide per image.
- The arms add Novae's 64-d latent to LightGBM Table + nbr. Everything else is identical to `cell-type-graph`.
- Scores: macro-F1, mean over 3 replicates; paired patient-bootstrap calls against Table + nbr.

**Sanity check:** Table + nbr reproduces `cell-type-graph` exactly in all 8 task × train × test cells (difference 0.0, `check_vs_cell_type_graph.csv`).

## Verdict: both pre-registered claims fail (`decision.json`)

| Claim | Rule | Result |
|---|---|---|
| H-N1: cross-cohort cell typing | + Novae `[cohort]` beats Table + nbr on J→D or D→J (CI > 0 in 3/3 replicates, mean ≥ +0.01), with no loss elsewhere | **Fail.** Mean change −0.0014 (J→D) and −0.0004 (D→J); no replicate has CI > 0. J→J loses in 2/3 replicates (−0.0015) |
| H-N2: cohort alignment | `[cohort]`-corr iLISI > Table + nbr in every draw, and cLISI rise < 0.25 | **Fail.** iLISI is higher in 15/15 draws, but cLISI rises by 0.40: cell types are blurred, not just cohorts mixed |

## Macro-F1, mean over 3 replicates (SD ≤ 0.04)

| Task | Train → test | Table + nbr | Novae `[cohort]` alone | + Novae `[cohort]` | + `[cohort]`-corr | + Novae `[img]` |
|---|---|---|---|---|---|---|
| coarse | J → J | **0.786** | 0.464 | 0.784 | 0.785 | 0.785 |
| coarse | D → D | **0.704** | 0.437 | 0.704 | 0.703 | 0.704 |
| coarse | J → D | 0.566 | 0.401 | 0.564 | 0.566 | **0.567** |
| coarse | D → J | 0.574 | 0.387 | 0.573 | **0.575** | 0.573 |
| coarse | Both → J | 0.771 | 0.455 | 0.771 | 0.772 | **0.773** |
| coarse | Both → D | **0.692** | 0.436 | 0.692 | 0.692 | 0.692 |
| fine | D → D | **0.454** | – | 0.453 | – | – |
| fine | J → J | 0.421 | – | **0.422** | – | – |

1. **Novae adds nothing on top of the table plus neighbour features.**
   - Every Novae arm is within ±0.002 of Table + nbr, in every setting.
   - Of 54 paired calls (3 combined arms × 6 coarse settings × 3 replicates), 3 are "better" and 6 are "worse". The rest are n.s.
   - For comparison, SAGE+ gave +0.02 to +0.03 on J↔D (`cell-type-graph`).
2. **On its own, the niche embedding is a poor cell-type classifier** (macro-F1 0.39–0.46 vs 0.57–0.79).
   - Epithelial F1 is fine (0.79–0.83).
   - Endothelial F1 is close to zero (0.03–0.15), even though endothelial is the class where neighbours help most (`cell-type-graph`).
   - T-cell and myeloid F1 are 0.20–0.54.
   - This matches the homophily caveat in §18.9: the 2-hop SwAV objective encodes the compartment, not the cell.
3. **The `[img]` variant** (Novae's default per-image z-score) **is no better than `[cohort]`** when added to the table.

## Cohort mixing (5 draws × 3 replicates, 5,000 labelled test cells per cohort, perplexity 30)

| Space | iLISI, median (1–2; higher = cohorts mixed) | cLISI, mean (1–6; lower = types kept apart) |
|---|---|---|
| Table | 1.43 | **1.51** |
| Table + nbr | 1.16 | 1.63 |
| Novae `[cohort]` | 1.75 | 2.03 |
| Novae `[cohort]`-corr | 1.67 | 2.03 |
| Novae `[img]` | **1.82** | 2.05 |

- Novae mixes the cohorts more than any table space, but loses cell-type separation by about the same amount.
  - The cLISI rise is +0.40 against Table + nbr and +0.52 against Table.
  - This is the overcorrection that the H-N2 cLISI guard was meant to catch. As with rank encoding (`lisi-imc`), it is a generally blurrier space, not a selective removal of cohort.
- Novae's own `batch_effect_correction` (domain-centroid shift) *lowers* iLISI (1.75 → 1.67) and leaves cLISI unchanged. It does not improve the alignment.
- The neighbour features separate the cohorts more than the cell's own inputs do: iLISI 1.16 vs 1.43.
  - This is a design flaw in the H-N2 comparator, found in the smoke run and not amended.
  - The verdict is the same against either comparator (cLISI rise 0.40 vs Table + nbr, 0.52 vs Table).

## Caveat: Novae training was unstable or shallow
- Early stopping (Novae's default `min_delta` 0.05) ended 5 of 6 fits after 6 epochs (50k cells per epoch). Only replicate 0 `[cohort]` ran to 13 epochs.
- In replicates 1 and 2, the `[cohort]` training loss climbed back to **6.238 = ln(512)**, the value for uniform prototype assignment. That is, the SwAV objective collapsed.
- `[img]` plateaued at 5.03 from epoch 2 in every replicate.
- Even so, the conclusion does not depend on this:
  - replicate 0 `[cohort]` (not collapsed, loss 5.85) also adds nothing to the table (−0.0013 to +0.0022);
  - its standalone macro-F1 (0.35–0.40) is in fact *lower* than in the collapsed replicates (0.38–0.53).
- A longer, more tuned Novae run (smaller `min_delta`, more cells per epoch, a learning-rate sweep) could still change the standalone embedding. It is unlikely to change "nothing beyond Table + nbr": with a ±0.002 effect, there is no signal to amplify.

## Caveat: no pretraining (trained from scratch, not fine-tuned)
- **What was tested:** Novae's architecture and training recipe, trained from scratch on our two cohorts only (`model.fit`, about 1.7M unlabelled cells per replicate). The run did **not** use its 30M-cell pretraining or fine-tuning (`fine_tune` from `novae-human-0`).
- **Why:** the pretrained weights are RNA-only (Xenium / MERSCOPE / CosMx). The first layer is a per-gene-name embedding, and our protein panel (pan-cytokeratin, CK8/18 and others) has no matching gene tokens. The Novae paper also retrains for its protein data.
- **What this means:**
  - The result says nothing about Novae's zero-shot or pretrained performance.
  - It shows that this architecture and objective, trained on our IMC data, adds nothing beyond Table + nbr.
  - A protein-pretrained Novae would be a different test; we have not checked whether one exists.

## Implications
- **Not adopted.** Report it as one row in the cell-typing table: a graph foundation model trained on the target data adds nothing beyond hand-built neighbour features.
- **For cohort shift,** Novae's slide alignment does not separate cohort from biology. This agrees with `harmonise`: input-side correction doesn't fix transfer.
- **Novae for imputation:** not pursued, as pre-registered.
- **Unused so far:** replicate-0 latents and level-10 domains are saved, for a possible niche → survival test (§15). Given the results above, this is low priority.
