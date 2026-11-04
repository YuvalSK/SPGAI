"""Danenberg → Jackson: metrics, mean and median over the 10 markers with patient-bootstrap CIs, paired per-marker
calls and the normalisation control. The maths is in ../cross-cohort-metrics/summarise_core.py (shared with the main
figure). EVA and ImmuVis (Danenberg-trained) come from image-models-jackson once it has run; until then EVA comes from
this kernel (correlations only).
"""
import os, sys
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(OUT, '..', 'cross-cohort-metrics'))
from summarise_core import run

DATA = os.path.join(OUT, '..', '..', 'IMC data', 'Jackson2020')
run(OUT=OUT,
    NPZ=os.path.join(DATA, 'cross-cohort-metrics-jackson', 'percell_predictions_test.npz'),
    patient_fn=lambda Z: Z['patient'],
    boot_seed=12,
    MODELS=['EVA', 'Table', 'MLP', 'GCN', 'GraphSAGE', 'Table (Jackson-trained)', 'Painted true cell means'],
    EXTRA=[('image-models-jackson', 'immuvis', 'ImmuVis'), ('image-models-jackson', 'eva', 'EVA')],
    IMAGE_MODELS=['ImmuVis', 'EVA'],
    PER_CELL=['Table', 'MLP', 'GCN', 'GraphSAGE'],
    CTRL=' [Danenberg-scaled inputs]',    # Jackson inputs scaled with Danenberg statistics (the main version)
    ctrl_col='danenberg_scaling',
    data_dir=DATA,
    # 32 µm patch sums from the *-patch kernels (per-cell models and oracle, image models)
    PATCH_CSVS=[os.path.join(OUT, '..', k, f) for k, f in [('cross-cohort-metrics-jackson-patch', 'patch_sums_test.csv'),
                                                       ('image-models-jackson-patch', 'patch_sums_test_immuvis.csv'),
                                                       ('image-models-jackson-patch', 'patch_sums_test_eva.csv')]],
    ref_csv=os.path.join(OUT, '..', 'image-models-jackson', 'source_reference_danenberg.csv'))
