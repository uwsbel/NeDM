"""fig_planner_f104.png for K4/RESULTS_planner.md (Polaris planner on f104 soil).
Inputs (read-only): K4/e6/analysis/results_ov_v1.json (stage 2, frozen read-out), /tmp/k4_planner/s1_results.json
(the same frozen spec run on the stage-1 arms, renamed), K4/e6/index/soil_eval_ov_final.json, soil_eval_polaris_s1.json,
arena_gator_20260925/e6/index/soil_eval_bfull.json (stored Gator / HMMWV CEM and Gator straight drives)."""
import collections, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

K4 = 'artifacts/traverse/offroad_vehicles_20260927'
K3 = 'artifacts/traverse/arena_gator_20260925'
OUT = K4 + '/figures/fig_planner_f104.png'

SURF, INK, INK2, MUTED, GRID, NEUTRAL = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#f0efec'
C_POL, C_GAT, C_HMM, CRIT = '#2a78d6', '#1baf7a', '#4a3aa7', '#d03b3b'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'savefig.facecolor': SURF, 'axes.titlecolor': INK, 'text.color': INK})

R2 = json.load(open(K4 + '/e6/analysis/results_ov_v1.json'))
R1 = json.load(open('/tmp/k4_planner/s1_results.json'))


def bar(res, vehicle, arm, s='f104_800'):
    for b in res['bar']:
        if b['vehicle'] == vehicle and b['arm'] == arm and b['label'] == 'unsafe_belly' and not b.get('key_suffix'):
            p = b['per_set'][s]
            return p['goal_rate'], p['group_ci95'], round(p['n'] * (100 - p['goal_rate']) / 100)
    raise KeyError(arm)


# straight6_gator has no bar block: same one-sample pair bootstrap as ov_analyze (seeded), on the unsafe label
by = collections.defaultdict(dict)
for r in json.load(open(K4 + '/e6/index/soil_eval_ov_final.json'))['rows'] + json.load(open(K4 + '/e6/index/soil_eval_polaris_s1.json'))['rows']:
    by[r['group']][r['arm']] = r
for r in json.load(open(K3 + '/e6/index/soil_eval_bfull.json'))['rows']:
    if r['world'] == 'crm' and r['arena'] == 'f104' and r['arm'] in ('Gfull_free_gator', 'Hfull_free', 'straight6_gator'):
        by[r['group']][r['arm']] = r
assert len(by) == 800
ub = lambda r: r.get('unsafe_belly', r['unsafe'])
v = np.array([ub(by[g]['straight6_gator']) for g in sorted(by)], float)
rng = np.random.default_rng(0)
gb = 100 * (1 - v[rng.integers(0, len(v), (4000, len(v)))].mean(1))
s6g = (100 * (1 - v.mean()), list(np.percentile(gb, [2.5, 97.5])), int(v.sum()))

rows = [  # (label, (rate, ci, n_failed), colour, marker, filled, section)
    ('Planner: sampling + gradient refinement (declared default)', bar(R2, 'polaris', 'polaris_grad'), C_POL, 'o', True, 'Polaris, stage 2 model (all 15,229 drives)'),
    ('Planner: sampling only', bar(R2, 'polaris', 'polaris_cem'), C_POL, 'o', False, None),
    ('Same refined routes, power-corrected Polaris', bar(R2, 'polaris_pc', 'polaris_grad_pc'), C_POL, 'D', True, None),
    ('Planner: sampling + gradient refinement', bar(R1, 'polaris', 'polaris_grad'), C_POL, 'o', True, 'Polaris, stage 1 model (8,395 drives; interim)'),
    ('Planner: sampling only', bar(R1, 'polaris', 'polaris_cem'), C_POL, 'o', False, None),
    ('Same refined routes, power-corrected Polaris', bar(R1, 'polaris_pc', 'polaris_grad_pc'), C_POL, 'D', True, None),
    ('Straight route at 6 m/s (no model)', bar(R2, 'polaris', 'straight6_polaris'), C_POL, 's', True, 'Polaris, no planner'),
    ('Own planner: sampling + gradient refinement', bar(R2, 'gator', 'Gfull_grad_gator'), C_GAT, 'o', True, 'Gator (Gator-trained model), same pairs'),
    ('Own planner: sampling only (stored drives)', bar(R2, 'gator', 'Gfull_free_gator'), C_GAT, 'o', False, None),
    ('Straight route at 6 m/s (stored drives)', s6g, C_GAT, 's', True, None),
    ('Own planner: sampling + gradient refinement', bar(R2, 'hmmwv', 'Hfull_grad_hmmwv'), C_HMM, 'o', True, 'HMMWV (HMMWV-trained model), same pairs'),
    ('Own planner: sampling only (stored drives)', bar(R2, 'hmmwv', 'Hfull_free'), C_HMM, 'o', False, None),
]

fig = plt.figure(figsize=(19.5, 9.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.1], height_ratios=[1.55, 1.0], wspace=0.78, hspace=0.36,
                      left=0.205, right=0.99, top=0.885, bottom=0.075)
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[1, 0]); axC = fig.add_subplot(gs[:, 1])
fig.suptitle('Polaris planner on f104 soil, standing start: goal reached safely on the 800 suite pairs', x=0.012, ha='left',
             fontsize=13, fontweight='bold', y=0.975)
fig.text(0.012, 0.935, 'Goal reached safely = goal reached, no roll-back on a climb, no belly-in-soil flag. Bars: 95 % pair-bootstrap '
         'intervals. Dashed line: the 90 % bar. Sources: e6/analysis/results_ov_v1.json (frozen), e6/index/*.json.',
         fontsize=9, color=INK2, ha='left')


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_visible(False); ax.spines['bottom'].set_color(GRID)
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)


def draw_rows(ax, rows, lo, hi, label_fmt):
    y = 0; ys = []; secs = []
    for lab, (rate, ci, nf), col, mk, filled, sec in rows:
        if sec:
            if ys:
                y += 0.55
            secs.append((y, sec)); y += 0.85
        ys.append(y)
        ax.plot([max(ci[0], lo), min(ci[1], hi)], [y, y], color=col, lw=2, solid_capstyle='round', zorder=2)
        ax.plot([rate], [y], marker=mk, ms=8, color=col, mfc=(col if filled else SURF), mew=2, zorder=3, ls='none', clip_on=False)
        ax.text(hi + (hi - lo) * 0.03, y, label_fmt(rate, nf), va='center', ha='left', fontsize=8.8, color=INK)
        y += 1
    ax.set_ylim(y - 0.4, -0.9)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=8.8, color=INK2)
    for yy, sec in secs:
        ax.text(-0.02, yy, sec, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=9.2, fontweight='bold', color=INK)
    return ys


style(axA)
axA.set_xlim(0, 100)
axA.axvline(90, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
axA.text(90, -0.9, '90 % bar ', ha='right', va='bottom', fontsize=8.5, color=INK2)
draw_rows(axA, rows, 0, 100, lambda r, nf: f'{r:.1f} %')
axA.set_xlabel('goal reached safely, % of 800 pairs')
axA.set_title('(a) All planners on the same 800 pairs', loc='left', fontsize=10.5, fontweight='bold')

# (b) zoom on the Polaris arms
style(axB)
pol = [r for r in rows if r[2] == C_POL]
axB.set_xlim(97.5, 100)
draw_rows(axB, pol, 97.5, 100, lambda r, nf: f'{r:.2f} % ({nf} failed)')
axB.set_xlabel('goal reached safely, % of 800 pairs (zoom: 97.5-100 %; the 90 % bar is far to the left)')
axB.set_title('(b) Polaris only, zoomed: every arm is within 1 point of 100 %', loc='left', fontsize=10.5, fontweight='bold')

# (c) the 14 pairs where any Polaris drive failed
cols = [('straight6_polaris', 'straight\n6 m/s'), ('polaris_s1_cem', 'sampling'), ('polaris_s1_grad', '+ gradient'),
        ('polaris_s1_grad_pc', '+ gradient,\npower-\ncorrected'), ('polaris_cem', 'sampling'), ('polaris_grad', '+ gradient'),
        ('polaris_grad_pc', '+ gradient,\npower-\ncorrected'), (None, None), ('Gfull_grad_gator', 'Gator'), ('Hfull_grad_hmmwv', 'HMMWV')]
short = {'timeout': 'time-\nout', 'rollover': 'rolled', 'terrain_bounds_exit': 'left\narena', 'prolonged_blockage_terminated': 'stall',
         'soil_breakthrough_terminated': 'dug in'}
kind = {'crater_cross_slope': 'crater, side slope', 'crater_entry_cross_exit': 'crater, in-across-out', 'hill_cross_slope': 'hill, side slope',
        'hill_entry_cross_exit': 'hill, in-across-out', 'roughness_transfer': 'rough patch'}
pol_arms = [c for c, _ in cols[:7]]
sel = sorted(g for g in by if any(ub(by[g][a]) for a in pol_arms))
assert len(sel) == 14, len(sel)
sel.sort(key=lambda g: (-ub(by[g]['straight6_polaris']), -ub(by[g]['polaris_grad']), -ub(by[g]['polaris_s1_grad']), g))
axC.set_xlim(-0.5, len(cols) - 0.5); axC.set_ylim(len(sel) - 0.5, -3.0)
for i, g in enumerate(sel):
    for j, (a, _) in enumerate(cols):
        if a is None:
            continue
        r = by[g][a]; bad = ub(r)
        axC.add_patch(plt.Rectangle((j - 0.46, i - 0.44), 0.92, 0.88, fc=(CRIT if bad else NEUTRAL), ec=SURF, lw=2, zorder=2))
        if bad:
            axC.text(j, i, short.get(r['status'], 'unsafe'), ha='center', va='center', fontsize=7.6, color='white', fontweight='bold', zorder=3,
                     linespacing=0.9)
ylabels = []
for g in sel:
    r = by[g]['straight6_polaris']
    num = g.split('_')[-1]
    part = 'tuning 200' if r['suite_part'] == 'tuned200' else 'fresh 600'
    ylabels.append(f"pair {num}, {part}: {kind.get(r['stratum'], r['stratum'])}")
axC.set_yticks(range(len(sel))); axC.set_yticklabels(ylabels, fontsize=8.6, color=INK2)
axC.set_xticks([]); axC.tick_params(length=0)
for j, (a, l) in enumerate(cols):
    if a:
        axC.text(j, -0.62, l, ha='center', va='bottom', fontsize=7.9, color=INK2, linespacing=1.0)
for x0, x1, lab in [(0, 0, 'Polaris,\nno model'), (1, 3, 'Polaris planner,\nstage-1 model'), (4, 6, 'Polaris planner,\nstage-2 model'),
                    (8, 9, 'own planners\n(context)')]:
    axC.text((x0 + x1) / 2, -2.2, lab, ha='center', va='bottom', fontsize=8.4, color=INK, fontweight='bold', linespacing=1.0)
    axC.plot([x0 - 0.42, x1 + 0.42], [-2.12, -2.12], color=INK2, lw=0.9)
for s_ in axC.spines.values():
    s_.set_visible(False)
axC.set_title('(c) The 14 of 800 pairs where any Polaris drive failed', loc='left', fontsize=10.5, fontweight='bold')
axC.axhline(3.5, color=INK2, lw=0.8, ls=(0, (2, 2)), xmin=0, xmax=0.7)
axC.text(-0.5, len(sel) - 0.5 + 0.35, 'Red = failed, with the end state (stall = stopped moving; dug in = a wheel dug through the soil layer).\n'
         'Grey = goal reached safely. Above the dotted line: the 4 pairs where the straight route fails;\n'
         'every Polaris planner arm reaches the goal on them.\nGator / HMMWV columns: their own sampling + gradient planners.', ha='left', va='top', fontsize=8.3, color=INK2, transform=axC.transData)
# legend for markers in (a)
h = [Line2D([], [], marker='o', color=INK2, mfc=INK2, ls='none', ms=7, label='sampling + gradient refinement'),
     Line2D([], [], marker='o', color=INK2, mfc=SURF, mew=2, ls='none', ms=7, label='sampling only'),
     Line2D([], [], marker='D', color=INK2, mfc=INK2, ls='none', ms=6, label='power-corrected Polaris'),
     Line2D([], [], marker='s', color=INK2, mfc=INK2, ls='none', ms=7, label='straight route, 6 m/s')]
axA.legend(handles=h, loc='upper left', bbox_to_anchor=(0.0, 0.985), frameon=False, fontsize=8.3, ncol=2, columnspacing=1.2)
fig.savefig(OUT, dpi=150)
print('wrote', OUT, 'straight6_gator', s6g)
