# Probabilistic imputation: are virtual cells calibrated under cohort shift? (`future_plan.md` §18.6, §0.3 item 2)

**Run:** Kaggle kernel `prob-impute`, version 1, 2026-10-27 (CPU, 43 min, no failed blocks).
**Notebook:** `Scripts/ProbImpute.ipynb`.

**Setup:**
- Grid as in `train-test-grid` (replicate-0 split and seeds); LightGBM only; 11 shared markers.
- Scored on arcsinh(x/5) of the test cohort's measured values.
- In transfer, inputs are scaled and outputs mapped back with the *training* cohort's statistics, so the target marker's distribution in the test cohort is never used.

**Four predictive distributions:**
- point LightGBM (CRPS = absolute error);
- quantile LightGBM (9 levels, 50 draws per cell);
- residual sampling (validation-patient residuals within each prediction decile);
- climatology (draws from the training cohort's marginal).

Means are over the 10 markers (ERα excluded), with 1000× patient-bootstrap CIs.

## Headline: calibrated within a cohort, badly miscalibrated across cohorts
| Test | Training data | Quantile CRPS | Point CRPS | Climatology CRPS | Coverage, 80% interval | Coverage, 95% interval | PIT KS |
|---|---|---|---|---|---|---|---|
| Danenberg | own | 0.070 | 0.101 | 0.125 | 0.74 | 0.88 | 0.05 |
| Danenberg | both | 0.071 | 0.102 | 0.126 | 0.75 | 0.89 | 0.05 |
| Danenberg | Jackson (transfer) | 0.129 | 0.189 | 0.146 | **0.57** | 0.72 | 0.37 |
| Jackson | own | 0.080 | 0.115 | 0.146 | 0.74 | 0.89 | 0.04 |
| Jackson | both | 0.084 | 0.122 | 0.148 | 0.76 | 0.90 | 0.04 |
| Jackson | Danenberg (transfer) | 0.135 | 0.161 | 0.163 | **0.44** | 0.59 | 0.37 |

1. **A distribution beats a point everywhere.** Quantile LightGBM and residual sampling lower CRPS by about 30% against point LightGBM, for 10/10 markers in every setting.
2. **Within a cohort, the intervals are nearly calibrated but a little too narrow** (80% → 0.74–0.76; 95% → 0.88–0.90).
   - The likely cause: the residuals come from cells, so patient-to-patient variation is missing. This is what the patient term of the Bayesian model (§18.6.1) addresses.
   - Residual sampling reproduces the marginal (SD ratio 0.93–1.00, W2 0.050 = climatology's), where the point model is under-dispersed (0.68–0.70, W2 0.085).
3. **Under transfer, calibration fails in both directions.** 0/10 markers have |coverage₈₀ − 0.80| < 0.03, and the PIT KS is about 0.37. The worst markers:
   - J→D: CD45 0.18, CD68 0.35, SMA 0.39;
   - D→J: SMA 0.14, CD45 0.24, HER2 0.29.
4. **Under transfer, a distribution from the source cohort barely beats the source cohort's marginal.**
   - J→D, quantile vs climatology: 4 better / 2 n.s. / 4 worse per marker. The point model is *worse* than climatology for 8/10 markers.
   - D→J, quantile vs climatology: 7 / 1 / 2.
5. **The cause is visible in the histograms** (`target_histograms.png`). Several markers have different *marginals* in the two cohorts:
   - Danenberg CD45, SMA and ERα sit off the floor, where Jackson's pile up at it.
   - HER2, Ki-67, CK5 and pan-CK look alike.

   The transfer failure is partly a level/background shift that no model trained on one cohort can know. It is not only a change in how the markers relate to each other.

## Decision (§0.3 item 2)
`decision.json`: quantile LightGBM is **not** calibrated under transfer (0/10 markers, both directions).
- So the Bayesian arm (§18.6.1) keeps its full H1 test: there is room to improve calibration under shift.
- It must beat quantile LightGBM, whose transfer coverage is 0.57 / 0.44.

## Likelihood check for §18.6.1
- **A dominant near-floor component plus a separate high component** fits most markers: CD20, CD3, CD68, CK5 and Ki-67 (low-component weight 0.80–0.96, at about 0.02), and pan-CK (bimodal).
- **No floor component in Danenberg** for CD45, SMA, ERα or HER2 (low-component mean 0.19–0.78).
- **The mixture likelihood is justified,** but the weight of the floor component has to vary by cohort (the c_cohort term), and for some markers it will be near 0.
- BIC picks 3 components everywhere; with 50k cells it is not discriminative, so read the component weights instead.

## Caveats
- One replicate; CIs are over patients only.
- Quantile CRPS comes from 50 draws per cell (unbiased estimator); the quantiles are clamped beyond 0.025 / 0.975.
- **Transfer is scored on the raw arcsinh scale of the test cohort.** The cross-cohort level shift therefore counts as error, which is right for "virtual values".
- **Still to run:** a variant scored in the test cohort's own harmonised space. It would give the model the target marker's marginal (an oracle), and so separate the level shift from a change in how the markers relate.
- ImmuVis is not included; its predicted variance has not been exported yet (§0.3 item 4).
