"""Reverse direction: Danenberg → Jackson cross-cohort inference scored six ways (metrics_summary.csv, mean over
10 markers). Drawing code in ../cross-cohort-metrics/figure_core.py (shared with the main figure)."""
import os, sys
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(OUT, '..', 'cross-cohort-metrics'))
from figure_core import draw

draw(OUT, [[('ImmuVis', 'ImmuVis', 'imageD'),
            ('EVA', 'EVA', 'fm')],
           # Danenberg-scaled inputs: no cohort-level adaptation to Jackson, like the image models (normalisation control)
           [('Table [Danenberg-scaled inputs]', 'LightGBM', 'cellD'),
            ('MLP [Danenberg-scaled inputs]', 'MLP', 'cellD'),
            ('GCN [Danenberg-scaled inputs]', 'GCN', 'cellD'),
            ('GraphSAGE [Danenberg-scaled inputs]', 'GraphSAGE', 'cellD')],
           [('Table (Jackson-trained)', 'LightGBM', 'cellJ')]], first_cohort='Danenberg')
