"""Jackson → Danenberg (main figure): metrics, mean and median over the 10 markers with patient-bootstrap CIs,
paired per-marker calls and the normalisation control. The maths is in summarise_core.py (shared with the
Danenberg → Jackson direction). Writes metrics_summary.csv, metrics_per_marker_all.csv,
per_cell_vs_image_per_marker.csv and normalisation_control.csv.
"""
import os, sys
import numpy as np
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)        # run with python -I
from summarise_core import run

DATA = os.path.join(OUT, '..', '..', 'IMC data', 'Danenberg2022')
run(OUT=OUT,
    NPZ=os.path.join(DATA, 'cross-cohort-metrics', 'percell_predictions_test.npz'),
    # METABRIC id from the image name (checked against kill-test: 704/704 images)
    patient_fn=lambda Z: np.char.add('MB-', np.array([s.split('_')[1] for s in Z['image']])),
    boot_seed=13,
    MODELS=['ResNet-UNet residual', 'ResNet-UNet gamma', 'ResNet-UNet immuvis', 'Table', 'MLP', 'GCN', 'GraphSAGE',
            'Table (Danenberg-trained)', 'Painted true cell means'],
    # image foundation models run in their own kernels: (kernel, file tag, model name)
    EXTRA=[('immuvis-jackson', 'immuvis', 'ImmuVis'), ('eva-zeroshot', 'eva', 'EVA')],
    IMAGE_MODELS=['ResNet-UNet immuvis', 'ImmuVis', 'EVA'],
    PER_CELL=['Table', 'MLP', 'GCN', 'GraphSAGE'],
    CTRL=' [Jackson-scaled inputs]',      # Danenberg inputs scaled with Jackson statistics (the main version)
    ctrl_col='jackson_scaling',
    data_dir=DATA,
    # 32 µm patch sums from the *-patch kernels (per-cell models and oracle, image models)
    PATCH_CSVS=[os.path.join(OUT, '..', k, f) for k, f in [('cross-cohort-metrics-patch', 'patch_sums_test.csv'),
                                                       ('unet-patch-sums', 'patch_sums_per_image.csv'),
                                                       ('immuvis-jackson-patch', 'patch_sums_test_immuvis.csv'),
                                                       ('eva-zeroshot-patch', 'patch_sums_test_eva.csv')]],
    ref_csv=os.path.join(OUT, '..', 'eva-zeroshot', 'source_reference_jackson.csv'))
