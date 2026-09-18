"""Planner scoring picture for one start-goal pair: all 256 candidate routes on the terrain coloured by predicted risk,
the three driven routes highlighted, and every route's score against its speed / detour ("scenario")."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm, colors
import matplotlib.patheffects as pe
sys.path.insert(0, 'src')
from nedm.traverse.terrain import TerrainMap

D = Path(sys.argv[1]); GROUP = sys.argv[2]
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
PICKS = [(145, 'A  planner pick', '#2a78d6'), (183, 'B  suboptimal', '#1baf7a'), (0, 'C  risky', '#4a3aa7')]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE})
z = np.load(D / 'pool.npz'); p = z['p_crm']; wp = z['wp']; sp = z['sp']; n = len(p)
case = json.load(open(f'artifacts/traverse/crm_f104_v1/cases_eval/cases/{GROUP}.json'))
tm = TerrainMap.from_dir(Path('assets/traverse/arena_f104_50h_v1'))
out = {i: json.load(open(D / f'run_{i:03d}' / 'outcome.json')) for i, _, _ in PICKS}
mean_speed = sp[:, 1:-1].mean(1)
start, goal = np.asarray(case['layout']['start_xy']), np.asarray(case['goal_xy'])
chord = (goal - start) / np.linalg.norm(goal - start); normal = np.array([-chord[1], chord[0]])
lateral = np.abs((wp - start) @ normal).max(1)
order = np.argsort(p); rank = np.empty(n, int); rank[order] = np.arange(n)
cmap = colors.LinearSegmentedColormap.from_list('risk', cm.Oranges(np.linspace(0.18, 1.0, 256)))
norm = colors.Normalize(0, 1)

fig = plt.figure(figsize=(13.6, 8.2))
gs = fig.add_gridspec(2, 2, width_ratios=[0.82, 1], height_ratios=[1, 1], wspace=0.14, hspace=0.36, left=0.035, right=0.975, top=0.885, bottom=0.075)
ax = fig.add_subplot(gs[:, 0])
xs = np.linspace(-40, 40, 321); XX, YY = np.meshgrid(xs, xs); HH = tm.height(XX, YY)
ax.contourf(XX, YY, HH, levels=np.linspace(-2, 4, 13), cmap=colors.LinearSegmentedColormap.from_list('g', ['#d9d8d3', '#ffffff']), alpha=1.0)
ax.contour(XX, YY, HH, levels=np.arange(-1.5, 4.1, 0.5), colors='#8a8984', linewidths=0.4)
for i in order[::-1]:                                   # risky first, safest on top
    lw = 0.7 if p[i] > 0.5 else 1.8
    ax.plot(wp[i][:, 0], wp[i][:, 1], color=cmap(norm(p[i])), lw=lw, alpha=0.55 if p[i] > 0.5 else 1.0, zorder=2 + (1 - p[i]),
            path_effects=None if p[i] > 0.5 else [pe.Stroke(linewidth=lw + 1.2, foreground='#3a3936'), pe.Normal()])
for i, name, col in PICKS:
    ax.plot(wp[i][:, 0], wp[i][:, 1], color=col, lw=3.0, zorder=6, path_effects=[pe.Stroke(linewidth=4.6, foreground='white'), pe.Normal()])
    k = int(len(wp[i]) * {145: 0.46, 183: 0.66, 0: 0.62}[i]); off = {145: (-17, -2), 183: (9, 0), 0: (7, -2)}[i]
    ax.annotate(name.split()[0], wp[i][k], xytext=off, textcoords='offset points', fontsize=13, fontweight='bold',
                                                                 color=INK, zorder=8, path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])
ax.plot(*start, 'o', ms=9, mfc='white', mec=INK, mew=1.6, zorder=9); ax.annotate('start', start, xytext=(8, -12), textcoords='offset points', fontsize=10, zorder=9)
ax.add_patch(plt.Circle(goal, 2.5, fill=False, ec=INK, lw=1.8, zorder=9)); ax.annotate('goal', goal, xytext=(9, -4), textcoords='offset points', fontsize=10, zorder=9)
allx, ally = wp[..., 0].ravel(), wp[..., 1].ravel(); pad = 7
ax.set_xlim(allx.min() - pad, allx.max() + pad); ax.set_ylim(ally.min() - pad, ally.max() + pad); ax.set_aspect('equal')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.tick_params(length=0)
for s_ in ax.spines.values(): s_.set_visible(False)
ax.set_title('256 candidate routes, coloured by predicted risk\n(terrain contours every 0.5 m; lighter ground = higher)', loc='left', fontsize=10.5)
cb = fig.colorbar(cm.ScalarMappable(norm, cmap), ax=ax, location='bottom', fraction=0.035, pad=0.075, aspect=40); cb.set_label('predicted probability of not reaching the goal'); cb.outline.set_visible(False)

def style(a_):
    for s_ in ('top', 'right'): a_.spines[s_].set_visible(False)
    a_.grid(True, color=GRID, lw=0.8); a_.set_axisbelow(True); a_.tick_params(length=0)

a1 = fig.add_subplot(gs[0, 1])
a1.scatter(rank + 1, 100 * p[np.arange(n)], s=14, c=p, cmap=cmap, norm=norm, edgecolors='#3a3936', linewidths=0.3, zorder=3)
for i, name, col in PICKS:
    o = out[i]; res = f"goal reached in {o['elapsed_s']:.1f} s" if o['status'] == 'goal_reached' else f"bogged down at {o['elapsed_s']:.1f} s"
    a1.scatter([rank[i] + 1], [100 * p[i]], s=120, facecolors='none', edgecolors=col, linewidths=2.6, zorder=5)
    dx, dy, ha = {145: (16, -24, 'left'), 183: (16, -4, 'left'), 0: (-12, -30, 'right')}[i]
    a1.annotate(f"{name}: risk {100 * p[i]:.1f} %, rank {rank[i] + 1}\nChrono CRM: {res}", (rank[i] + 1, 100 * p[i]), xytext=(dx, dy), textcoords='offset points',
                ha=ha, fontsize=9.2, color=INK, zorder=6)
a1.set_xscale('log'); a1.set_xlim(0.8, 300); a1.set_ylim(-16, 108); a1.set_yticks([0, 20, 40, 60, 80, 100]); a1.set_xlabel('route rank by predicted risk (1 = safest, log scale)'); a1.set_ylabel('predicted risk (%)')
a1.set_title(f'Score of every route: {int((p < 0.5).sum())} of {n} are rated below 50 % risk', loc='left', fontsize=10.5); style(a1)

a2 = fig.add_subplot(gs[1, 1])
a2.scatter(mean_speed, lateral, s=16, c=p, cmap=cmap, norm=norm, edgecolors='#3a3936', linewidths=0.3, zorder=3)
for i, name, col in PICKS:
    a2.scatter([mean_speed[i]], [lateral[i]], s=120, facecolors='none', edgecolors=col, linewidths=2.6, zorder=5)
    a2.annotate(name.split()[0], (mean_speed[i], lateral[i]), xytext=(9, 5), textcoords='offset points', fontsize=11, fontweight='bold', zorder=6)
a2.set_xlabel('mean commanded speed along the route (m/s)'); a2.set_ylabel('largest sideways detour from the straight line (m)')
a2.set_title('What each scored route looks like: speed and detour (same colours)', loc='left', fontsize=10.5); style(a2)
fig.suptitle(f'Planner scoring for one held-out start-goal pair on CRM soil ({GROUP[-10:]}): CRM-trained network, 5-member ensemble', x=0.04, ha='left', fontsize=12.5)
fig.savefig(D / 'planner_scoring.png', dpi=150); print('saved', D / 'planner_scoring.png')
json.dump({str(i): dict(name=nm, risk=float(p[i]), rank=int(rank[i] + 1), mean_speed=float(mean_speed[i]), lateral_m=float(lateral[i]),
                        length_m=float(z['st'][i][-1]), status=out[i]['status'], elapsed_s=out[i]['elapsed_s']) for i, nm, _ in PICKS}, open(D / 'demo_summary.json', 'w'), indent=1)
