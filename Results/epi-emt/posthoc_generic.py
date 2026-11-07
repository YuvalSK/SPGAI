"""POST-HOC (not pre-registered, run 2026-11-07 after seeing epi_emt.py results).

The axis effect in Jackson is carried by the epithelial markers (keratins, E/P-cadherin: +0.3 to +0.55 with H3K27me3),
while Twist and Slug are also positive and vimentin is flat. Question: does H3K27me3 co-vary with *every* marker in
tumour cells (a generic per-cell signal factor: section thickness, ablation, segmentation), rather than with the EMT axis?
1. Partial r of H3K27me3 with off-target markers that should not relate to EMT in tumour cells.
2. The primary axis after adding a generic-intensity covariate (mean normal score of all non-axis, non-nuclear markers).
"""
import os
import numpy as np, pandas as pd, duckdb
import epi_emt as E

HERE = os.path.dirname(os.path.abspath(__file__))
rs = np.random.default_rng(E.SEED + 1)
J_OFF = ["cd45", "cd3", "cd20", "cd68", "sma", "c_erb_b_2_her2", "estrogen_receptor_alpha", "progesterone_receptor_a_b", "gata3",
         "ki_67", "p53", "carbonic_anhydrase_ix", "egfr", "s6", "m_tor", "c_myc", "cleaved_parp", "v_wf", "fibronectin", "cd44",
         "cytokeratin_5", "keratin_14_krt14", "cytokeratin_7"]
K_OFF = ["cd45", "cd3", "cd20", "cd68", "sma", "hla_class_1", "hla_dr", "beta_catenin", "p53", "egfr", "ki67", "cd63", "pd_l1", "ido"]
rows = []

cols = ["image_name", "patient_id", "cohort", "is_tumour", "metacluster", "histone_h3_trimethylate", "histone_h3", "dna1", "dna2",
        "area", *E.J_M, *E.J_E, *J_OFF]
J = duckdb.sql(f"select {', '.join(cols)} from '{E.J_CELLS}'").df().rename(columns={"histone_h3_trimethylate": "h3k27me3"})
J = J[(J.is_tumour == 1) & J.metacluster.between(14, 27)].copy()
J["patient_id"] = J.patient_id.astype(str)
J["log_area"] = np.log(np.maximum(J.area.astype(float), 1))
J = E.add_scores(J, ["h3k27me3", "histone_h3", "dna1", "dna2", "log_area", *E.J_M, *E.J_E, *J_OFF])
J["axis"] = J[[f"z_{c}" for c in E.J_M]].mean(1) - J[[f"z_{c}" for c in E.J_E]].mean(1)
generic = [m for m in J_OFF if m not in ("fibronectin", "cd44", "cytokeratin_5", "keratin_14_krt14", "cytokeratin_7")]
J["generic"] = J[[f"z_{c}" for c in generic]].mean(1)
J = E.add_scores(J, ["axis", "generic"])
cov = [f"z_{c}" for c in E.J_COV]
for coh, name in (("basel", "Basel"), ("zurich", "Zurich")):
    d = J[J.cohort == coh]
    for m in J_OFF:
        rows.append(dict(set=name, test=f"off-target {m}", **E.summarise(E.per_patient(d, "z_h3k27me3", f"z_{m}", cov), rs)))
    rows.append(dict(set=name, test="generic intensity (mean of off-target)", **E.summarise(E.per_patient(d, "z_h3k27me3", "z_generic", cov), rs)))
    rows.append(dict(set=name, test="primary axis + generic covariate", **E.summarise(E.per_patient(d, "z_h3k27me3", "z_axis", cov + ["z_generic"]), rs)))
    rows.append(dict(set=name, test="total H3 vs generic (control)", **E.summarise(E.per_patient(d, "z_histone_h3", "z_generic", ["z_dna1", "z_dna2", "z_log_area"]), rs)))
    for m in E.J_M + E.J_E:
        rows.append(dict(set=name, test=f"marker {m} + generic covariate", **E.summarise(E.per_patient(d, "z_h3k27me3", f"z_{m}", cov + ["z_generic"]), rs)))
    print(name, "done", flush=True)

K, kch = E.load_keren()
K = E.add_scores(K, list(dict.fromkeys(kch + ["log_area", *K_OFF, "nbr_vimentin", "nbr_keratin6", "nbr_keratin17", "nbr_h3k27me3"])))
K["axis_common"] = K.z_vimentin - K[["z_keratin6", "z_keratin17"]].mean(1)
K["generic"] = K[[f"z_{c}" for c in K_OFF if c != "beta_catenin"]].mean(1)
K = E.add_scores(K, ["axis_common", "generic"])
kcov = [f"z_{c}" for c in E.K_COV]
for m in K_OFF:
    rows.append(dict(set="Keren", test=f"off-target {m}", **E.summarise(E.per_patient(K, "z_h3k27me3", f"z_{m}", kcov), rs)))
rows.append(dict(set="Keren", test="generic intensity (mean of off-target)", **E.summarise(E.per_patient(K, "z_h3k27me3", "z_generic", kcov), rs)))
rows.append(dict(set="Keren", test="common axis + generic covariate", **E.summarise(E.per_patient(K, "z_h3k27me3", "z_axis_common", kcov + ["z_generic"]), rs)))

P = pd.DataFrame(rows)
P.to_csv(os.path.join(HERE, "posthoc_generic.csv"), index=False)
print(P[["set", "test", "r", "lo", "hi", "n_pat", "frac_pos"]].round(3).to_string())
