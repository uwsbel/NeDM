#!/usr/bin/env python3
"""Small figures for the arena_gator_20260925 final report (REPORT.md).

Reads only the frozen evaluation indexes (e6/index/*.json), the arena table (arenas/map_lookup_error.json) and the
overhead RGB renders of the arena maps; writes PNGs to <K3>/figures/. No number is typed by hand.

  PYTHONPATH=src:scripts python scripts/ag_report_figures.py

fig_taskA_unseen.png   task A: pooled rate on the 8 unseen arenas per model (soil speed free, goal not reached;
                       rigid fixed 2 m/s, unsafe), with each arena's own rate as a thin grey line.
fig_taskB_soil.png     task B: soil goal reached on the 800-pair f104 suite, by planner and driven vehicle.
fig_arenas.png         overhead renders of the 12 arenas (training, dev, near and spread test arenas).
"""
import json
import os
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

K3 = 'artifacts/traverse/arena_gator_20260925'
OUT = os.path.join(K3, 'figures')

# reference palette (dataviz skill, light mode); validated: blue/orange all checks pass
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#b9b8b3'
BLUE = '#2a78d6'
ORANGE = '#eb6834'

plt.rcParams.update({
    'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.titlecolor': INK, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#ebeae6',
    'grid.linewidth': 0.8, 'axes.axisbelow': True,
})


def load_rows(name):
    with open(os.path.join(K3, 'e6/index', name)) as f:
        return json.load(f)['rows']


def rate(rows, arm, label, sets, vehicle=None, arena=None):
    v = [int(r[label]) for r in rows if r['arm'] == arm and r['set'] in sets
         and (vehicle is None or r['vehicle'] == vehicle) and (arena is None or r['arena'] == arena)]
    return Fraction(100 * sum(v), len(v)), len(v)


def exact(fr):
    # exact decimal of a percentage whose denominator has only factors 2 and 5 (group counts 150-2,000): no rounding
    d = Decimal(fr.numerator) / Decimal(fr.denominator)
    t = format(d.normalize(), 'f')
    return t if '.' in t else t + '.0'


def one_decimal(fr):
    # half-up to one decimal, as the results files print these rates
    d = Decimal(fr.numerator) / Decimal(fr.denominator)
    return format(d.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP), 'f')


def composite(rows, arms, label, sets, arena=None):
    # declared composite (M1 = mean of M1a, M1b per group; pooled rate = mean of the ensembles' rates, same groups)
    vals = [rate(rows, a, label, sets, arena=arena) for a in arms]
    assert len({n for _, n in vals}) == 1, vals
    return sum(r for r, _ in vals) / len(vals), vals[0][1]


def fig_task_a():
    soil = load_rows('soil_eval_v1.json')
    rigid = load_rows('rigid_eval_v1.json')
    arenas_near = ['g260', 'g271', 'g251', 'g247']
    arenas_spread = ['g258', 'g268', 'g263', 'g241']
    models = [('f104 only', 'M1'), ('two arenas,\nsame total', 'M2'), ('three arenas,\nsame total', 'M3'),
              ('three arenas,\nall data', 'A3')]
    panels = [
        ('Soil, speed free: goal not reached', soil, 'fail', '_free',
         {'M1': ['M1a', 'M1b'], 'M2': ['M2'], 'M3': ['M3a', 'M3b'], 'A3': ['A3']}, 'M1a_free'),
        ('Rigid, fixed 2 m/s: unsafe', rigid, 'unsafe', '_fx2',
         {'M1': ['M1a', 'M1b'], 'M2': ['M2'], 'M3': ['M3a', 'M3b'], 'A3': ['A3']}, None),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    x = np.arange(len(models))
    record = {}
    for ax, (title, rows, label, suf, parts, indist_arm) in zip(axes, panels):
        pooled = [composite(rows, [p + suf for p in parts[m]], label, {'unseen'})[0] for _, m in models]
        for a in arenas_near + arenas_spread:
            ys = [composite(rows, [p + suf for p in parts[m]], label, {'unseen'}, arena=a)[0] for _, m in models]
            ax.plot(x, [float(v) for v in ys], color=MUTED, lw=1.0, zorder=1, ls='-' if a in arenas_near else (0, (3, 2)))
        ax.plot(x, [float(v) for v in pooled], color=BLUE, lw=2, marker='o', ms=8, zorder=3, markeredgecolor=SURFACE, markeredgewidth=2)
        for xi, y in zip(x, pooled):
            ax.annotate(f'{exact(y)} %', (xi, float(y)), textcoords='offset points', xytext=(0, 9), ha='center', color=INK,
                        fontsize=9, zorder=4)
        # in-distribution reference: the f104-only planner on f104's own held-out hill/crater groups
        if indist_arm:
            ref = rate(rows, indist_arm, label, {'indist_f104'}, vehicle='hmmwv')[0]
            ref_lab = f'f104 only (ensemble 1) on 200 held-out f104 pairs: {exact(ref)} %'
        else:
            ref = composite(rows, ['M1a_fx2', 'M1b_fx2'], label, {'indist_f104'})[0]
            ref_lab = f'f104 only on 200 held-out f104 pairs: {exact(ref)} %'
        ax.axhline(float(ref), color=INK2, lw=1, ls=':', zorder=2)
        ax.annotate(ref_lab, (x[-1] + 0.35, float(ref)), textcoords='offset points', xytext=(0, 3), ha='right',
                    color=INK2, fontsize=8)
        ax.set_xticks(x, [m for m, _ in models])
        ax.set_xlim(-0.35, len(models) - 0.65)
        ax.set_ylim(0, None)
        ax.set_ylabel('% of start/goal pairs')
        ax.set_title(title, loc='left', fontsize=10)
        record[title] = dict(pooled=dict(zip([m for _, m in models], map(exact, pooled))), indist_ref=exact(ref))
    axes[1].plot([], [], color=MUTED, lw=1.0, label='one near test arena')
    axes[1].plot([], [], color=MUTED, lw=1.0, ls=(0, (3, 2)), label='one spread test arena')
    axes[1].plot([], [], color=BLUE, lw=2, marker='o', ms=6, label='8 unseen arenas pooled')
    axes[1].legend(loc='upper right', frameon=False, fontsize=8)
    fig.suptitle('Task A: planners on 8 arenas none of them was trained on (lower is better)', x=0.01, ha='left',
                 fontsize=11, color=INK)
    p = os.path.join(OUT, 'fig_taskA_unseen.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p, record


def fig_task_b():
    rows = load_rows('soil_eval_bfull.json')
    s800 = {'f104_suite', 'indist_f104'}
    bars = [
        ('Gator', 'straight route, 6 m/s', 'straight6_gator'),
        ('Gator', 'HMMWV-trained, all data', 'Hfull_free_gator'),
        ('Gator', 'HMMWV-trained, tiers 0-6', 'H_free_gator'),
        ('Gator', 'Gator-trained, tiers 0-6', 'G_free_gator'),
        ('Gator', 'Gator-trained, all data', 'Gfull_free_gator'),
        ('HMMWV', 'straight route, 6 m/s', 'straight6'),
        ('HMMWV', 'HMMWV-trained, tiers 0-6', 'M1a_free'),
        ('HMMWV', 'HMMWV-trained, all data', 'Hfull_free'),
    ]
    vals = []
    for veh, lab, arm in bars:
        r, n = rate(rows, arm, 'fail', s800, vehicle=veh.lower())
        assert n == 800, (arm, n)
        vals.append(100 - r)
    fig, ax = plt.subplots(figsize=(7.6, 3.7), constrained_layout=True)
    y = np.array([8.7, 7.7, 6.7, 5.7, 4.7, 3.0, 2.0, 1.0])  # gap between the Gator and the HMMWV group
    for yi, (veh, lab, arm), v in zip(y, bars, vals):
        ax.barh(yi, float(v), height=0.72, color=ORANGE if veh == 'Gator' else BLUE, edgecolor=SURFACE, linewidth=2)
        ax.text(float(v) + 1.0, yi, f'{one_decimal(v)} %', va='center', ha='left', color=INK, fontsize=9)
    ax.set_yticks(y, [lab for _, lab, _ in bars])
    ax.set_xlim(0, 105)
    ax.set_xlabel('goal reached, % of the 800 f104 start/goal pairs (soil, speed free)')
    ax.grid(axis='y', visible=False)
    ax.legend(handles=[Patch(color=ORANGE, label='driven by the Gator'), Patch(color=BLUE, label='driven by the HMMWV')],
              loc='upper right', frameon=False, fontsize=8)
    ax.set_title('Task B, soil: goal reached on f104 by planner and vehicle (higher is better)', loc='left', fontsize=10)
    p = os.path.join(OUT, 'fig_taskB_soil.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p, {f'{veh}: {lab}': exact(v) for (veh, lab, _), v in zip(bars, vals)}  # exact; the bars print 1 decimal


def fig_arenas():
    with open(os.path.join(K3, 'arenas/map_lookup_error.json')) as f:
        tab = json.load(f)['arenas']
    order = [('f104', 'training'), ('g203', 'training'), ('g228', 'training'), ('g217', 'dev'),
             ('g260', 'near test'), ('g271', 'near test'), ('g251', 'near test'), ('g247', 'near test'),
             ('g258', 'spread test'), ('g268', 'spread test'), ('g263', 'spread test'), ('g241', 'spread test')]
    fig, axes = plt.subplots(3, 4, figsize=(8.0, 6.6), constrained_layout=True)
    for ax, (a, role) in zip(axes.ravel(), order):
        png = ('artifacts/traverse/crm_f104_v1/map_root/static_map_v1/rgb.png' if a == 'f104'
               else os.path.join(K3, f'maps/arena_{a}/rgb.png'))
        img = plt.imread(png)
        h, w = img.shape[:2]
        # crop the sky border around the 80 x 80 m arena, then subsample 3x
        c = img[int(0.09 * h):int(0.91 * h), int(0.09 * w):int(0.91 * w)][::3, ::3]
        ax.imshow(c, interpolation='lanczos')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        d = tab[a]['distance_f104']
        ax.set_title(f'{a} ({role})\ndistance to f104 {d:.2f}', fontsize=8, color=INK)
    fig.suptitle('The 12 arenas (overhead renders of the undeformed terrain, 80 x 80 m each)', x=0.01, ha='left',
                 fontsize=10, color=INK)
    p = os.path.join(OUT, 'fig_arenas.png')
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    pa, ra = fig_task_a()
    pb, rb = fig_task_b()
    pc = fig_arenas()
    rec = {'script': 'scripts/ag_report_figures.py', 'figures': [pa, pb, pc], 'task_a_rates_pct': ra,
           'task_b_goal_reached_pct': rb}
    with open(os.path.join(OUT, 'figures_numbers.json'), 'w') as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
