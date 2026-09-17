"""Task B part 2 figure: the omitted route-start range d0, one histogram per arena (one value per group).

Small multiples, one series per facet, shared axes, single categorical hue (slot 1). Numbers for the same
histogram are in shift.json -> d0_histogram.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
TAGS = ['f104', 'g203', 'g216', 'g217', 'g228', 'g231']
SURFACE, INK, INK2, SERIES = '#fcfcfb', '#0b0b0b', '#52514e', '#2a78d6'
H = 110.0

vals = {}
for t in TAGS:
    z = np.load(HERE / 'points' / f'{t}.npz', allow_pickle=True)
    d0 = z['d0_all'].astype(float)
    g = z['group_all'].astype(str)
    _, first = np.unique(g, return_index=True)
    vals[t] = d0[first]

lo = min(v.min() for v in vals.values()); hi = max(v.max() for v in vals.values())
edges = np.linspace(lo - .2, hi + .2, 46)
fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.2), sharex=True, sharey=True, facecolor=SURFACE)
for ax, t in zip(axes.ravel(), TAGS):
    v = vals[t]
    ax.hist(v, edges, color=SERIES, edgecolor=SURFACE, linewidth=.6)
    ax.set_facecolor(SURFACE)
    ax.grid(axis='y', color='#e6e5e1', linewidth=.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#d5d4d0')
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    name = 'arena_f104_50h_v1 (training)' if t == 'f104' else f'arena_g{t[1:]}'
    ax.set_title(name, color=INK, fontsize=10, loc='left', pad=6)
    ax.text(.98, .93, f'n={len(v)} groups\nmean {v.mean():.1f} m\nsd {v.std():.2f} m\n'
                      f'span {v.max()-v.min():.1f} m',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5, color=INK2, linespacing=1.5)
for ax in axes[1]:
    ax.set_xlabel('route-start range $d_0$  (m)', color=INK2, fontsize=9.5)
for ax in axes[:, 0]:
    ax.set_ylabel('start/goal groups', color=INK2, fontsize=9.5)
fig.suptitle('The range the v1 raw-depth arm subtracts and never passes', color=INK, fontsize=12.5,
             x=.011, ha='left', y=.985)
fig.text(.011, .925, 'One value per start/goal group; 400 groups per arena. '
                     r'$d_0 = (H-z_0)\,\sec\theta_0$ — spread comes almost entirely from where the start sits in the image.',
         color=INK2, fontsize=9.5, ha='left')
fig.tight_layout(rect=[0, 0, 1, .90])
fig.savefig(HERE / 'd0_histogram.png', dpi=170, facecolor=SURFACE)
print('wrote', HERE / 'd0_histogram.png')
print(json.dumps({t: dict(n=len(v), mean=round(float(v.mean()), 3), std=round(float(v.std()), 3),
                          min=round(float(v.min()), 3), max=round(float(v.max()), 3)) for t, v in vals.items()}, indent=1))
