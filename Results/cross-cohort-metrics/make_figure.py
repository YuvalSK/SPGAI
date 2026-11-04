"""Main figure: Jackson → Danenberg cross-cohort inference, one set of predictions scored six ways
(metrics_summary.csv, mean over 10 markers). Drawing code in figure_core.py (shared with the reverse direction)."""
import os, sys
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)
from figure_core import draw

# groups of (model in the CSV, x label, style key); a gap separates the groups
draw(OUT, [[('ResNet-UNet immuvis', 'U-Net', 'imageJ'),
            ('ImmuVis', 'ImmuVis', 'imageJ'),
            ('EVA', 'EVA', 'fm')],
           # Jackson-scaled inputs: no cohort-level adaptation to Danenberg, like the image models (normalisation control)
           [('Table [Jackson-scaled inputs]', 'LightGBM', 'cellJ'),
            ('MLP [Jackson-scaled inputs]', 'MLP', 'cellJ'),
            ('GCN [Jackson-scaled inputs]', 'GCN', 'cellJ'),
            ('GraphSAGE [Jackson-scaled inputs]', 'GraphSAGE', 'cellJ')],
           [('Table (Danenberg-trained)', 'LightGBM', 'cellD')]], first_cohort='Jackson')
