# kronos-zeroshot: summary

Run finished on 2026-10-22 in 61 min on 2×T4, replicate 0 only. KRONOS (ViT-S/16, commit 26f1bc0) saw no IMC data in pretraining. Cell patches are 32 px around the mask centroid, upsampled to 64 px, with pixels outside the mask set to 0. The embedding is CLS (384) plus 256 PCs of the per-marker features (95.5% of variance). Normalisation was chosen on Jackson calibration images: own cohort statistics (val macro-F1 0.712) beat KRONOS's statistics (0.705). The 60-image smoke run is in `smoke/`.

## Coarse cell typing, macro-F1 (patient bootstrap, test images)
| Train → test | Table | Table + nbr | Table + K32 | Table + nbr + K32 | KRONOS linear |
|---|---|---|---|---|---|
| J → J | 0.721 | 0.748 | 0.731 | **0.752** | 0.693 |
| J → D | 0.552 | 0.572 | 0.560 | **0.573** | 0.466 |
| D → D | 0.644 | 0.670 | 0.662 | **0.681** | 0.628 |
| D → J | 0.517 | 0.552 | 0.532 | 0.558 | **0.561** |
| Both → J | 0.710 | 0.735 | 0.720 | **0.736** | 0.686 |
| Both → D | 0.618 | 0.640 | 0.632 | **0.647** | 0.597 |

Fine labels (within cohort): Table + nbr 0.398 (J) / 0.452 (D); + K32 0.402 / 0.472; KRONOS linear 0.333 / 0.378.

## Paired differences against Table + nbr (95% CI)
- **KRONOS linear (image only):** worse in every within-cohort and pooled task (−0.04 to −0.06 coarse; −0.065 and −0.074 fine). Worse on J → D (−0.105). The only exception is D → J: +0.009 [−0.024, +0.029], n.s.
- **Adding 32 KRONOS PCs:** small gains. J → J +0.004 [+0.002, +0.006]; D → D +0.010 [+0.008, +0.013]; D → J +0.006 [+0.001, +0.011]; Both → D +0.007; fine D → D +0.020 [+0.006, +0.031]. J → D and Both → J are n.s.
- **K32 instead of neighbours:** Table + K32 is below Table + nbr everywhere (−0.008 to −0.020). The image embedding adds less than the neighbour features.

## Gate for KRONOS adaptation: passed, but only just
- (a) KRONOS linear within 0.05 of Table + nbr: D → D −0.042 passes; J → J −0.055 fails.
- (b) Table + nbr + K32 beats Table + nbr by ≥ 0.01 with CI > 0: D → D +0.010 [+0.008, +0.013], exactly at the threshold.

`proceed_to_adaptation` is true. Both criteria pass on Danenberg only, by small margins, and with one replicate.

## Files
- `scores.csv`, `diffs.csv`, `perclass.csv`, `calibration.json`, `summary.json`, `kronos-zeroshot.log`.
- Embeddings (1.4 GB each) and cell indices: `IMC data/{Danenberg2022,Jackson2020}/kronos-zeroshot/`; smoke versions in `smoke/` there. They are kept for the deferred imputation and H3K27me3 probes, which must be scored with the G-R5 leak controls.
