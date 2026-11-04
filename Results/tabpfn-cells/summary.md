# TabPFN v2 as the cell model (`future_plan.md` §18.8, §0.3 item 2c)

**Run:** Kaggle kernel `tabpfn-cells`, version 1, 2026-10-29 (one T4, 312 min, no failed blocks, nothing skipped).
**Notebook:** `Scripts/TabPFNCells.ipynb` (setup cells shared with `prob-impute`).

**Setup:**
- TabPFN v2 (`tabpfn==9.2.0`, open weights), 10k context cells, 4 estimators, 9 quantiles → 50 draws per cell.
- Compared with quantile LightGBM, both *full* (up to 300k / 100k cells, as `prob-impute`) and fitted on the *same* 10k context cells.
- The same 40,000 test cells per cohort for every model.
- Means are over 10 markers (ERα excluded), with 1000× patient-bootstrap CIs.
- Few-shot: k target-cohort patients (from its training split) join the context, plus a cohort-flag feature; everything is in the source cohort's scaling.

## Headline: TabPFN ties full LightGBM; it is better with little data and slightly better calibrated in few-shot transfer
| Setting | Test | cov₈₀ LGB full | cov₈₀ TabPFN | Δgap₈₀ (CI) | CRPS LGB full | CRPS TabPFN | TabPFN vs LGB full, CRPS per marker (better / n.s. / worse) |
|---|---|---|---|---|---|---|---|
| own | Danenberg | 0.74 | 0.74 | +0.004 (−0.003, 0.008) | 0.069 | 0.069 | 1 / 8 / 1 |
| own | Jackson | 0.74 | 0.76 | −0.012 (−0.015, −0.006) | 0.080 | 0.079 | 2 / 8 / 0 |
| both | Danenberg | 0.75 | 0.75 | +0.008 (0.004, 0.012) | 0.071 | 0.072 | 2 / 6 / 2 |
| both | Jackson | 0.76 | 0.76 | +0.003 (−0.001, 0.007) | 0.084 | 0.082 | 2 / 7 / 1 |
| transfer J→D | Danenberg | 0.57 | 0.59 | **−0.025 (−0.029, −0.021)** | 0.128 | 0.125 | 4 / 2 / 4 |
| transfer D→J | Jackson | 0.43 | 0.44 | −0.005 (−0.013, 0.004) | 0.135 | 0.135 | 2 / 4 / 4 |
| J→D + 5 patients | Danenberg | 0.59 | 0.62 | **−0.031** | 0.084 | 0.085 | 3 / 6 / 1 |
| J→D + 20 patients | Danenberg | 0.67 | 0.68 | **−0.010** | 0.076 | 0.077 | 3 / 6 / 1 |
| D→J + 5 patients | Jackson | 0.59 | 0.64 | **−0.058** | 0.107 | 0.104 | 4 / 5 / 1 |
| D→J + 20 patients | Jackson | 0.69 | 0.71 | **−0.021** | 0.089 | 0.089 | 2 / 8 / 0 |

Δgap₈₀ = TabPFN − LightGBM (full) in mean |cov₈₀ − 0.80|; negative = TabPFN better calibrated. Bold = paired CI < 0.

1. **Pre-registered decision: passes, but only on the letter of the rule.**
   - TabPFN lowers the J→D transfer coverage gap (CI < 0). But coverage only moves from 0.57 to 0.59, far from 0.80, and D→J is n.s.
   - The few-shot CRPS criterion (≥ 6/10 markers better) fails in every case (2–4/10).
   - `decision.json`: `claim = true`. **Read it as "a small, real calibration gain", not "TabPFN fixes transfer".**
2. **At equal data, TabPFN is clearly better.** Against LightGBM fitted on the same 10k cells, it wins CRPS on 7–8/10 markers within a cohort and 5–6/10 pooled. Its 80% coverage is 0.74–0.76 vs 0.70.
   - With all the data (30× more cells), LightGBM catches up. So TabPFN's advantage is data efficiency.
3. **Few-shot: TabPFN is consistently better calibrated** (Δgap −0.010 to −0.058, CI < 0 in all four cases), with CRPS tied.
4. **The few-shot result matters more than the model choice.** Adding a few target-cohort patients helps far more than any model or harmonisation change:
   - D→J coverage: 0.44 → 0.64 (5 patients) → 0.71 (20 patients).
   - D→J CRPS: 0.135 → 0.104 → 0.089.
   - R4: 0.33–0.41 → 0.47–0.55.
   - With 20 patients this beats the `harmonise` target-marginal oracle (cov₈₀ 0.62–0.67, CRPS 0.094–0.114), so the k patients teach relations as well as level.
5. **Cell-level accuracy:** TabPFN's median ties LightGBM on R4 (within 0.02 everywhere, except D→J, where it is lower: 0.33 vs 0.36).
   - "TabPFN (median)" beats *point* LightGBM on CRPS for 4–8/10 markers. That is mostly a median-vs-mean effect under an L1 score, not a better model; RMSE is equal or worse.

## Implications
- **Not a new main model.** TabPFN is a well-calibrated, data-efficient tie with LightGBM. Report it as one row in the model comparison.
- **For `bayes-impute`:**
  - H3 (few-shot) is now the most promising hypothesis. The comparator to beat at k = 5 is TabPFN (cov₈₀ 0.62 / 0.64, CRPS 0.085 / 0.104).
  - S2 (no target data) remains the hard case: no model gets past cov₈₀ 0.59.
- **Few-shot is cheap and strong.** Even 5 labelled patients from a new cohort nearly halve the D→J coverage gap.

## Caveats
- One replicate, and one draw of the k patients (the k = 5 numbers will vary with which patients are drawn).
- 40k-cell test subsample.
- TabPFN context capped at 10k cells, with the v2 weights. The later, gated versions (2.5+) take 50k and were not tested.
- Cell typing (§18.8 task 2) was not run (needs the coarse-label mapping; TabPFN v2 handles at most 10 classes).
