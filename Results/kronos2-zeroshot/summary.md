# kronos2-zeroshot: KRONOS2 vs KRONOS (v1), zero-shot cell typing, replicates 0–2

**Runs:**
- `kronos2-zeroshot` v1 (GPU, 2×T4) embedded every cell with KRONOS2, then ran out of RAM in the pooled ('Both') probes. Its log and calibration are in `failed-v1/`; the 60-image smoke run is in `smoke/`.
- `kronos2-probes` v3 (CPU, 364 min, no failed blocks) ran the probes and the gate from those saved embeddings (dataset `kronos2-embeddings`), together with KRONOS v1's embeddings from `kronos-zeroshot`.
- Embeddings are in `IMC data/<cohort>/kronos2-zeroshot/`.

**Pretraining data:** KRONOS2 (ViT-B/16) was trained on SPM-56M: 8 fluorescence platforms, no IMC or MIBI, and only 0.06% breast tissue. It is a clean zero-shot test.
- CK5 and CK8/18 were registered as new markers through KRONOS2's text-embedding route. Their nearest vocabulary markers are CK20, CK17 and panCK.
- Normalisation: KRONOS2's own marker stats won the calibration narrowly (validation macro-F1 0.708 vs 0.707 for our cohort stats).

## Macro-F1, mean over 3 replicates
| Task | Table + nbr | KRONOS linear | KRONOS2 linear | + K32 (v1) | + K2-32 (v2) |
|---|---|---|---|---|---|
| J→J | 0.755 | 0.694 | 0.686 | 0.759 | 0.759 |
| D→D | 0.672 | 0.627 | 0.627 | 0.681 | 0.684 |
| J→D | 0.571 | 0.445 | 0.504 | 0.572 | 0.572 |
| D→J | 0.542 | 0.555 | 0.457 | 0.552 | 0.550 |
| Both→J | 0.739 | 0.687 | 0.675 | 0.741 | 0.741 |
| Both→D | 0.639 | 0.593 | 0.602 | 0.646 | 0.651 |
| Fine J | 0.421 | 0.350 | 0.348 | 0.427 | 0.422 |
| Fine D | 0.451 | 0.377 | 0.379 | 0.469 | 0.469 |

"Table + nbr" is the cell table plus neighbour features; "+ K32" adds the encoder's first 32 per-marker principal components to it.

## Head-to-head, KRONOS2 − KRONOS (paired, all 3 replicates)
**Linear probes (embedding alone):**
- No consistent winner.
- KRONOS2 is better on J→D (+0.059) and Both→D (+0.009).
- It is worse on D→J (−0.099), J→J (−0.008) and Both→J (−0.013).
- The two tie on D→D.

**Added to the table (+32 PCs):**
- KRONOS2 is slightly better on D→D (+0.003) and Both→D (+0.005), 3/3 replicates each.
- Elsewhere they tie.

## Adaptation gate (pre-registered; must hold in every replicate)
**Both encoders pass.**
- **(a) D→D linear probe within 0.05 of Table + nbr:**
  - KRONOS: −0.042 / −0.047 / −0.045;
  - KRONOS2: −0.042 / −0.047 / −0.044.
- **(b) Table + nbr + 32 PCs ≥ +0.01 with CI > 0:**
  - KRONOS2: D→D +0.013 / +0.012 / +0.012 in all 3 replicates; Both→D +0.011 to +0.014.
  - KRONOS v1 misses the 0.01 minimum in replicates 1–2 (+0.009); it passes those replicates through (a) only.

**Reading:**
- The image embedding alone stays 0.04–0.07 below the cell table within cohort.
- What the image adds on top of the table is about +0.01 macro-F1, and only on Danenberg.
- KRONOS2 is the stronger candidate for adaptation: it meets (b) in every replicate.
- The margins are small, so a fine-tuning study should expect gains around 0.01 unless adaptation changes the picture.
