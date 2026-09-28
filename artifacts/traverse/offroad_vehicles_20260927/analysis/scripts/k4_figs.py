"""Figures for offroad_vehicles_20260927 RESULTS_smoke / RESULTS_collection (written by the results agent, 2026-09-28).
Inputs: /tmp/k4_extract.jsonl (read-only extraction of the cluster runs, script /tmp/k4_extract.py),
K4/scratch/S3/sample_A.json, K4/e6/index/soil_eval_probe_v1.json, K4/e6/index/soil_eval_gradref_v1.json,
arena_gator_20260925/e6/index/soil_eval_bfull.json, search_probe_20260927/groups/*.txt and A4 picks."""
import collections, json, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

K4 = 'artifacts/traverse/offroad_vehicles_20260927'
SP = 'artifacts/traverse/search_probe_20260927'
OUT = K4 + '/figures'
os.makedirs(OUT, exist_ok=True)

SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
C_POL, C_M113, C_GAT, C_HMM = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'savefig.facecolor': SURF, 'axes.titlecolor': INK, 'text.color': INK})
fail = lambda s: s != 'goal_reached'


def wilson(k, n, z=1.959963984540054):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def style(ax, axis='x'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(GRID); ax.spines['bottom'].set_color(GRID)
    ax.grid(axis=axis, color=GRID, lw=0.8, ls='-'); ax.set_axisbelow(True)
    ax.tick_params(length=0)


def hbar(ax, y, v, h, color, r=0.9):
    """horizontal bar from 0 to v (in data units), rounded data end, square baseline."""
    if v <= 0:
        return
    rr = min(r, v / 2)
    ax.add_patch(FancyBboxPatch((0, y - h / 2), v, h, boxstyle=f'round,pad=0,rounding_size={rr}', fc=color, ec='none', zorder=2,
                                mutation_aspect=h / (2 * rr) if rr else 1))
    ax.add_patch(Rectangle((0, y - h / 2), max(v - rr, 0), h, fc=color, ec='none', zorder=2))


def vbar(ax, x, v, w, color, r=0.9):
    if v <= 0:
        return
    ax.add_patch(FancyBboxPatch((x - w / 2, 0), w, v, boxstyle='round,pad=0,rounding_size=0.02', fc=color, ec='none',
                                mutation_aspect=1 / 0.02 * min(r, v / 2)))
    ax.add_patch(Rectangle((x - w / 2, 0), w, max(v - min(r, v / 2), 0), fc=color, ec='none'))


L = [json.loads(l) for l in open('/tmp/k4_extract.jsonl')]

# ---------------------------------------------------------------- fig_smoke
A = collections.defaultdict(dict)
for x in L:
    if x['sample'] == 'A':
        A[x['arm']][x['pair_id']] = x
arms = [('gatorctl', 'Gator, re-driven (1 ms step)', C_GAT), ('gatorh', 'Gator at the M113\'s 0.5 ms step', C_GAT),
        ('polaris', 'Polaris, stock driveline (primary)', C_POL), ('polaris_pc', 'Polaris, power-corrected driveline', C_POL),
        ('polaris_4wd', 'Polaris, open differentials', C_POL), ('polaris_w08', 'Polaris, 0.33 m soil wheels', C_POL),
        ('m113', 'M113, stock gearing', C_M113), ('m113_g4', 'M113, re-geared (gear ratios / 4)', C_M113)]
rng = np.random.default_rng(0)
smoke_rows = []
for a, lab, col in arms:
    v = {p: x for p, x in A[a].items() if x['run'].get('complete') and x['run'].get('qa_ok')}
    g = collections.defaultdict(list)
    for p, x in v.items():
        g[x['group']].append(fail(x['run']['status']))
    keys = sorted(g); s = np.array([sum(g[k]) for k in keys], float); n = np.array([len(g[k]) for k in keys], float)
    bs = []
    for _ in range(4000):
        i = rng.integers(0, len(keys), len(keys)); bs.append(s[i].sum() / n[i].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    k = int(s.sum()); N = int(n.sum())
    smoke_rows.append((a, lab, col, k, N, 100 * k / N, 100 * lo, 100 * hi))
fig, ax = plt.subplots(figsize=(8.6, 4.6))
style(ax, 'x')
H = 0.56
for i, (a, lab, col, k, N, f, lo, hi) in enumerate(smoke_rows):
    y = len(smoke_rows) - 1 - i
    hbar(ax, y, f, H, col, r=0.8)
    ax.plot([lo, hi], [y, y], color=INK, lw=1.2, solid_capstyle='butt', zorder=3)
    for xx in (lo, hi):
        ax.plot([xx, xx], [y - 0.12, y + 0.12], color=INK, lw=1.2, zorder=3)
    ax.text(hi + 1.2, y, f'{f:.1f} %  ({k}/{N})', va='center', ha='left', fontsize=9, color=INK)
for xv in (80.6, 69.4):
    ax.axvline(xv, color=MUTED, lw=1.0, zorder=1)
ax.text(69.4 - 0.8, -0.95, 'HMMWV, stored runs: 69.4 %', ha='right', va='center', fontsize=7.8, color=INK2)
ax.text(80.6 + 0.8, -0.95, 'rule 5 limit: 80.6 %\n(Gator, 0.08 m larger wheels)', ha='left', va='center', fontsize=7.8, color=INK2)
ax.set_yticks(range(len(smoke_rows))); ax.set_yticklabels([r[1] for r in smoke_rows][::-1], color=INK)
ax.set_xlim(0, 118); ax.set_ylim(-1.45, len(smoke_rows) - 0.4)
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels([f'{t} %' for t in (0, 25, 50, 75, 100)])
ax.set_xlabel('goal not reached on sample A (144 routes, 24 groups)')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc=C_GAT, label='Gator'), Patch(fc=C_POL, label='Polaris'), Patch(fc=C_M113, label='M113')],
          loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=8.5, handlelength=1.0, borderaxespad=0.2)
fig.suptitle('Smoke test: how often each vehicle fails to reach the goal on the same 144 soil routes', x=0.01, ha='left',
             fontsize=11, color=INK, y=0.985)
fig.text(0.01, 0.925, 'Bars: share of routes where the goal was not reached; whiskers: 95 % interval, bootstrap over the 24 route groups. '
         'M113 re-geared: 143 routes (1 crash).', fontsize=7.8, color=INK2, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT + '/fig_smoke.png', dpi=170)
plt.close(fig)
print('smoke', [(r[0], r[3], r[4], round(r[5], 1), round(r[6], 1), round(r[7], 1)) for r in smoke_rows])

# ---------------------------------------------------------------- fig_vehicles
C = [x for x in L if x['src'] == 'collection' and x['run'].get('qa_ok') and x['gator'].get('qa_ok') and x['hmmwv'].get('qa_ok')]
def rates(xs):
    n = len(xs)
    return n, [100 * sum(fail(x[k]['status']) for x in xs) / n for k in ('run', 'hmmwv', 'gator')]
panels = [
    ('By route kind', [('all routes', C), ('designed', [x for x in C if x['kind'] == 'designed']),
                       ('planner proposals', [x for x in C if x['kind'] == 'on_policy'])]),
    ('Designed routes by speed profile', [(f'constant {v} m/s', [x for x in C if x['kind'] == 'designed' and x['route']['speed_profile_id'] == f'constant_{v}']) for v in (2, 4, 6)]
     + [('2 -> 6 -> 2 m/s', [x for x in C if x['kind'] == 'designed' and x['route']['speed_profile_id'] == 'smooth_2_6_2'])]),
    ('Planner proposals by mean commanded speed', [(f'{lo}-{hi} m/s' if hi < 9 else f'>= {lo} m/s', [x for x in C if x['kind'] == 'on_policy' and lo <= x['route']['v_mean'] < hi])
                                                   for lo, hi in ((0.5, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 9))]),
]
fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.9), gridspec_kw=dict(width_ratios=[3, 4, 5]), sharey=True)
W = 0.26
vt = []
for ax, (title, cats) in zip(axs, panels):
    style(ax, 'y')
    ticks = []
    for j, (lab, xs) in enumerate(cats):
        n, rr = rates(xs); vt.append((title, lab, n, [round(v, 1) for v in rr]))
        ticks.append(f'{lab}\nn = {n:,}')
        for q, (v, col) in enumerate(zip(rr, (C_POL, C_HMM, C_GAT))):
            xpos = j + (q - 1) * (W + 0.02)
            vbar(ax, xpos, v, W, col, r=1.2)
            ax.text(xpos, v + 1.2, f'{v:.0f}' if v >= 9.95 else f'{v:.1f}', ha='center', va='bottom', fontsize=7.4, color=INK)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(ticks, color=INK, fontsize=8.2)
    ax.set_xlim(-0.6, len(cats) - 0.4); ax.set_ylim(0, 105)
    ax.set_title(title, fontsize=9.5, color=INK, loc='left')
axs[0].set_yticks([0, 20, 40, 60, 80, 100]); axs[0].set_yticklabels([f'{t} %' for t in (0, 20, 40, 60, 80, 100)])
axs[0].set_ylabel('goal not reached')
fig.legend(handles=[Patch(fc=C_POL, label='Polaris'), Patch(fc=C_HMM, label='HMMWV'), Patch(fc=C_GAT, label='Gator')],
           loc='upper right', bbox_to_anchor=(0.995, 0.93), ncol=3, frameon=False, fontsize=8.5, handlelength=1.0)
fig.suptitle('The same soil routes driven by three vehicles: Polaris (this study), HMMWV and Gator (stored runs)', x=0.01, ha='left',
             fontsize=11, color=INK, y=0.99)
fig.text(0.01, 0.925, f'{len(C):,} collection routes on f104 soil, valid for all three vehicles (tiers 0-12). Designed = geometric routes at a '
         'set speed profile; planner proposals = routes drawn at random from the planner\'s route family (bends, speeds 0.5-6 m/s).',
         fontsize=7.8, color=INK2)
fig.tight_layout(rect=(0, 0.0, 1, 0.88))
fig.savefig(OUT + '/fig_vehicles.png', dpi=170)
plt.close(fig)
for r in vt:
    print('veh', r)

# ---------------------------------------------------------------- fig_probe
pr = json.load(open(K4 + '/e6/index/soil_eval_probe_v1.json'))
gr = json.load(open(K4 + '/e6/index/soil_eval_gradref_v1.json'))
grp = lambda f: {l.strip() for l in open(SP + '/groups/' + f) if l.strip()}
F50, FAIL, CON = grp('fail_p50.txt'), grp('fail.txt'), grp('control.txt')
FO = FAIL - F50
P = collections.defaultdict(dict)
for r in pr['rows']:
    P[r['arm']][r['group']] = r
sets = [('recorded pick failed; model had\nrated it over 50 % likely to fail', F50), ('recorded pick failed; model had\nrated it 50 % or less', FO), ('recorded pick\nreached the goal', CON)]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.7), gridspec_kw=dict(width_ratios=[3.2, 1.6]))
style(ax, 'y'); style(ax2, 'y')
cols = [('recorded planner (search 4 x 64)', C1), ('bigger search (16 x 512), same route shapes', C2), ('wider route shapes, same bigger search', C3)]
probe_vals = []
for j, (lab, S) in enumerate(sets):
    n = len(S)
    rec = 0 if S is not CON else n
    vals = [(rec, None)]
    for arm in ('Gbig_gator', 'Gwide_gator'):
        k = sum(1 - P[arm][g]['unsafe_belly'] for g in S); vals.append((k, wilson(k, n)))
    for q, ((k, ci), (_, col)) in enumerate(zip(vals, cols)):
        xpos = j + (q - 1) * 0.28
        v = 100 * k / n
        vbar(ax, xpos, v, 0.25, col, r=1.2)
        top = v
        if ci:
            ax.plot([xpos, xpos], [100 * ci[0], 100 * ci[1]], color=INK, lw=1.1, zorder=3)
            for yy in ci:
                ax.plot([xpos - 0.04, xpos + 0.04], [100 * yy, 100 * yy], color=INK, lw=1.1, zorder=3)
            top = 100 * ci[1]
        ax.text(xpos, top + 1.5, f'{k}/{n}' if ci else ('0 (all\nfailed)' if k == 0 else f'{k}/{n}'), ha='center', va='bottom',
                fontsize=7.4, color=INK)
        probe_vals.append((lab.replace('\n', ' '), q, k, n))
ax.set_xticks(range(3)); ax.set_xticklabels([f'{s[0]}\n({len(s[1])} pairs)' for s in sets], fontsize=8.3, color=INK)
ax.set_ylim(0, 112); ax.set_xlim(-0.55, 2.55)
ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_yticklabels([f'{t} %' for t in (0, 20, 40, 60, 80, 100)])
ax.set_ylabel('goal reached safely (Gator drives)')
ax.set_title('The 361 probe pairs, by what the recorded planner did', fontsize=9.5, loc='left', color=INK)
ax.legend(handles=[Patch(fc=c, label=l) for l, c in cols], loc='upper left', frameon=False, fontsize=8, handlelength=1.0)
# right: 800-pair picture
gg = {r['group']: r for r in gr['rows'] if r['arm'] == 'Gfull_grad_gator'}
grad_ok = 100 * sum(1 - r['fail'] for r in gg.values()) / len(gg)
k_c = {a: sum(1 - P[a][g]['unsafe_belly'] for g in CON) for a in ('Gbig_gator', 'Gwide_gator')}
k_1 = {a: sum(1 - P[a][g]['unsafe_belly'] for g in F50) for a in ('Gbig_gator', 'Gwide_gator')}
k_2 = {a: sum(1 - P[a][g]['unsafe_belly'] for g in FO) for a in ('Gbig_gator', 'Gwide_gator')}
est = {a: 100 * (539 * k_c[a] / 100 + k_1[a] + k_2[a]) / 800 for a in k_c}
bars = [('recorded\n(driven)', 100 * 539 / 800, C1), ('bigger\nsearch\n(estimate)', est['Gbig_gator'], C2), ('wider\nshapes\n(estimate)', est['Gwide_gator'], C3)]
for j, (lab, v, col) in enumerate(bars):
    vbar(ax2, j, v, 0.5, col, r=1.2)
    ax2.text(j, v + 1.5, f'{v:.1f} %', ha='center', va='bottom', fontsize=8, color=INK)
ax2.axhline(90, color=MUTED, lw=1.0)
ax2.text(2.3, 91, '90 % bar', ha='right', va='bottom', fontsize=7.8, color=INK2)
ax2.set_xticks(range(3)); ax2.set_xticklabels([b[0] for b in bars], fontsize=8, color=INK)
ax2.set_ylim(0, 112); ax2.set_xlim(-0.55, 2.55)
ax2.set_yticks([0, 20, 40, 60, 80, 100]); ax2.set_yticklabels([f'{t} %' for t in (0, 20, 40, 60, 80, 100)])
ax2.set_title('All 800 suite pairs', fontsize=9.5, loc='left', color=INK)
fig.suptitle('Gator on f104 soil: driving the offline wider-search probe (same Gator-trained model, standing start)', x=0.01, ha='left',
             fontsize=11, color=INK, y=0.99)
fig.text(0.01, 0.935, 'Whiskers: 95 % Wilson intervals. "Safely" = goal reached, no roll-back on a climb, no belly-in-soil flag. '
         'Right: recorded = goal reached by the recorded planner on all 800 pairs;\nthe other two are estimates if the new search replaced it, '
         '(539 x control share + rescued pairs) / 800, because only 100 of the 539 recorded successes were re-driven.', fontsize=7.8, color=INK2, va='top')
fig.tight_layout(rect=(0, 0, 1, 0.87))
fig.savefig(OUT + '/fig_probe.png', dpi=170)
plt.close(fig)
print('probe', probe_vals, {a: round(v, 1) for a, v in est.items()}, 'grad', round(grad_ok, 1))
