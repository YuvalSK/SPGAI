"""Shared figure code for both directions: top row correlations (a-d: pixel pooled, pixel per image, 32 µm patch,
cell), bottom row the matching errors (e-h), mean over markers.

Colour = training cohort (Jackson blue, Danenberg green, neither purple); fill = model type (generative image model
filled, non-generative cell model hollow). Models without outputs yet are skipped.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

CEIL = 'Painted true cell means'      # best any per-cell map can score on the pixel metrics
CONST = 'Constant (training mean)'
JAC, DAN, NONE, GREY = '#2a78d6', '#008300', '#4a3aa7', '#52514e'
STYLE = {'imageJ': dict(color=JAC, hollow=False), 'imageD': dict(color=DAN, hollow=False),
         'fm': dict(color=NONE, hollow=False),
         'cellJ': dict(color=JAC, hollow=True), 'cellD': dict(color=DAN, hollow=True)}
PANELS = [('pooled pixel r', 'a', 'Pixel-level correlation\n(pooled across images)'),
          ('per-image pixel r', 'b', 'Pixel-level correlation\n(per image, averaged)'),
          ('pooled patch r', 'c', 'Patch-level correlation (32 µm)\n(pooled across images)'),
          ('R4', 'd', 'Cell-level correlation\n(within image)'),
          ('pooled pixel RMSE', 'e', 'Pixel-level RMSE\n(pooled across images)'),
          ('per-image pixel RMSE', 'f', 'Pixel-level RMSE\n(per image, averaged)'),
          ('patch RMSE', 'g', 'Patch-level RMSE (32 µm)\n(pooled across images)'),
          ('cell RMSE', 'h', 'Cell-level RMSE')]
NCOL = len(PANELS) // 2
INK, INK2 = '#0b0b0b', '#52514e'
DASH, DOT = (0, (4, 3)), (0, (1, 2))


def bar_kw(st):
    if st['hollow']:
        return dict(facecolor='none', edgecolor=st['color'], linewidth=1.8)
    return dict(facecolor=st['color'], edgecolor=st['color'], linewidth=1.8)


def draw(OUT, GROUPS, first_cohort='Jackson', name='fig_cross_cohort_metrics'):
    SUM = pd.read_csv(f'{OUT}/metrics_summary.csv')
    SUM = SUM[SUM.stat == 'mean']
    present = set(SUM.model)
    MODELS, X, x = [], [], 0.0
    for grp in GROUPS:
        grp = [g for g in grp if g[0] in present]
        for g in grp:
            MODELS.append(g)
            X.append(x)
            x += 1
        if grp:
            x += 0.5
    names = [m for m, _, _ in MODELS]
    err = SUM[SUM.metric.isin([p[0] for p in PANELS[NCOL:]]) & SUM.model.isin(names + [CEIL, CONST])]
    etop = max(0.05, -(-err.hi.max() * 1.18 // 0.05) * 0.05) if len(err) else 0.3

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK})
    fig, axes = plt.subplots(2, NCOL, figsize=(4.4 * NCOL * max(1.0, len(MODELS) / 8), 7.8))
    for ax, (metric, letter, ylab) in zip(axes.ravel(), PANELS):
        is_err = 'RMSE' in metric
        for xi, (nm, _, key) in zip(X, MODELS):
            r = SUM[(SUM.model == nm) & (SUM.metric == metric)]
            if not len(r):
                continue                  # e.g. errors of a model whose target-scale outputs are not in yet
            row = r.iloc[0]
            ax.bar(xi, row.value, width=0.66, zorder=2, **bar_kw(STYLE[key]))
            ax.errorbar(xi, row.value, yerr=[[row.value - row.lo], [row.hi - row.value]], fmt='none',
                        ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
            ax.text(xi, max(row.hi, row.value) + (0.006 if is_err else 0.025),
                    f'{row.value:.3f}' if is_err else f'{row.value:.2f}', ha='center', va='bottom',
                    fontsize=7 if is_err else 7.5, color=INK, bbox=dict(facecolor='#ffffff', edgecolor='none', pad=0.5),
                    zorder=4)
        for ref, ls in ((CEIL, DASH), (CONST, DOT)):
            if metric in ('R4', 'cell RMSE') and ref == CEIL:
                continue                  # the true cell values are exact at the cell level (r = 1, RMSE = 0)
            c = SUM[(SUM.model == ref) & (SUM.metric == metric)]
            if len(c):
                ax.axhline(c.value.iloc[0], color=INK2, lw=1, ls=ls, zorder=1)
        ax.set_xticks(X)
        ax.set_xticklabels([lab for _, lab, _ in MODELS], fontsize=8, rotation=35, ha='right', rotation_mode='anchor')
        ax.set_xlim(X[0] - 0.6, X[-1] + 0.6)
        if is_err:
            ax.set_ylim(0, etop)
        else:
            ax.set_ylim(min(0.0, SUM[SUM.metric == metric].value.min() - 0.05), 1)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.axhline(0, color=INK2, lw=0.8, zorder=1)
        ax.set_ylabel(ylab, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', length=0)
        ax.text(-0.2, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom', color=INK)

    keys = {k for _, _, k in MODELS}
    cohorts = {'Jackson': Patch(color=JAC, label='Jackson'), 'Danenberg': Patch(color=DAN, label='Danenberg')}
    trained = [cohorts[first_cohort]] + [v for k, v in cohorts.items() if k != first_cohort]
    if 'fm' in keys:
        trained.append(Patch(color=NONE, label='Neither (zero-shot)'))
    kind = [Patch(**bar_kw(dict(color=GREY, hollow=False)), label='Generative (image)'),
            Patch(**bar_kw(dict(color=GREY, hollow=True)), label='Non-generative (cell)'),
            Line2D([0], [0], color=INK2, lw=1, ls=DASH, label='True cell values')]
    if CONST in present:
        kind.append(Line2D([0], [0], color=INK2, lw=1, ls=DOT, label='Constant (training mean)'))
    lg = dict(loc='upper center', frameon=False, fontsize=9, title_fontsize=9, alignment='left')
    fig.legend(handles=trained, title='Trained on', ncol=3, bbox_to_anchor=(0.27, 1.01), **lg)
    fig.legend(handles=kind, title='Model', ncol=2, bbox_to_anchor=(0.7, 1.01), **lg)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.5, h_pad=2.0)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/{name}.{ext}', dpi=300, bbox_inches='tight')
    print('saved:', names)
