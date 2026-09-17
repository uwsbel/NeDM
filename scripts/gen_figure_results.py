"""gen_v1 results figure: single start/goal test (speed free and fixed 2 m/s) and five-goal missions."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

G = 'artifacts/traverse/fdm_f104_50h_20260909/gen_v1'
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
COL = {'model': '#2a78d6', 'hand rule': '#eb6834', 'straight line': '#1baf7a'}
t = json.load(open(G + '/test_results.json')); m = json.load(open(G + '/mission_results.json'))


def wilson(p, n, z=1.96):
    if n == 0: return (0, 0)
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0, c - h), min(1, c + h)


def bars(ax, groups, values, ns, title, ylabel, pct=True, ci=True):
    arms = list(COL); w = 0.26; x = np.arange(len(groups))
    for k, arm in enumerate(arms):
        v = np.array([values[g][arm] for g in range(len(groups))], float)
        pos = x + (k - 1) * (w + 0.02)
        ax.bar(pos, 100 * v if pct else v, width=w, color=COL[arm], label=arm, zorder=3)
        if ci:
            lo_hi = [wilson(v[g], ns[g][arm]) for g in range(len(groups))]
            ax.errorbar(pos, 100 * v, yerr=[100 * v - [100 * a for a, b in lo_hi], [100 * b for a, b in lo_hi] - 100 * v],
                        fmt='none', ecolor=INK2, elinewidth=1, capsize=2.5, zorder=4)
        for p_, val in zip(pos, v):
            ax.text(p_, (100 * val if pct else val), f"{100*val:.1f}" if pct else f"{val:.0f}", ha='center', va='bottom',
                    fontsize=8, color=INK, zorder=5, bbox=dict(boxstyle='square,pad=0.1', fc=SURF, ec='none', alpha=.8))
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9.5, color=INK2)
    ax.set_title(title, fontsize=11, color=INK, loc='left'); ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.set_facecolor(SURF); ax.grid(axis='y', color=GRID, lw=.8, zorder=0)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#cfcec9'); ax.tick_params(colors=INK2, labelsize=8.5, length=0)


fig, axs = plt.subplots(2, 3, figsize=(17, 9.2), facecolor=SURF)
R = t['rates']
grp = ['f104 (trained)', 'five new arenas']; keys = ['f104', 'new_pooled']
free = [{'model': R[k]['n2']['unsafe'], 'hand rule': R[k]['rule']['unsafe'], 'straight line': R[k]['straight6']['unsafe']} for k in keys]
nfree = [{'model': R[k]['n2']['n'], 'hand rule': R[k]['rule']['n'], 'straight line': R[k]['straight6']['n']} for k in keys]
bars(axs[0, 0], grp, free, nfree, 'Single goal, speed free: failed or slid (%)', '% of start/goals')
fx = [{'model': R[k]['n2_fixed2']['unsafe'], 'hand rule': R[k]['rule_fixed2']['unsafe'], 'straight line': R[k]['straight2']['unsafe']} for k in keys]
nfx = [{'model': R[k]['n2_fixed2']['n'], 'hand rule': R[k]['rule_fixed2']['n'], 'straight line': R[k]['straight2']['n']} for k in keys]
bars(axs[0, 1], grp, fx, nfx, 'Single goal, speed fixed at 2 m/s: failed or slid (%)', '% of start/goals')
tl = [{'model': R[k]['n2']['tilt30'], 'hand rule': R[k]['rule']['tilt30'], 'straight line': R[k]['straight6']['tilt30']} for k in keys]
bars(axs[0, 2], grp, tl, nfree, 'Single goal, speed free: leaned past 30 deg (%)', '% of start/goals')
S = m['summary']; mk = ['f104', 'new_pooled']; mg = ['f104 (100 missions)', 'new arenas (100 missions)']
arm_of = {'model': 'n2', 'hand rule': 'rule', 'straight line': 'straight6'}
ms = [{a: 1 - S[k][arm_of[a]]['success'] for a in COL} for k in mk]; mn = [{a: S[k][arm_of[a]]['n'] for a in COL} for k in mk]
bars(axs[1, 0], mg, ms, mn, 'Five goals in a row: mission failed (%)', '% of missions')
sl = [{a: S[k][arm_of[a]]['any_slide'] for a in COL} for k in mk]
bars(axs[1, 1], mg, sl, mn, 'Five goals in a row: slid backwards at least once (%)', '% of missions')
tm = [{a: S[k][arm_of[a]]['median_time_successful_s'] for a in COL} for k in mk]
bars(axs[1, 2], mg, tm, mn, 'Five goals in a row: median time of completed missions (s)', 'seconds', pct=False, ci=False)
axs[0, 0].legend(frameon=False, fontsize=9.5, loc='upper left', labelcolor=INK)
fig.text(.01, .985, 'Tonight: longer tasks and never-seen arenas (frozen model, no retraining; 6,639 single-goal runs + 600 missions)',
         fontsize=15, color=INK, weight='bold', va='top')
fig.text(.01, .955, 'model = the deployed planner (lowest predicted risk of 256 candidates);  hand rule = non-learned terrain+speed score on the same 256;  '
         'straight line = 6 m/s (2 m/s in the fixed-speed panel). Bars: 95% intervals; the time panel shows medians.', fontsize=9.5, color=INK2, va='top')
fig.tight_layout(rect=(0, 0, 1, .94))
fig.savefig(G + '/results.png', dpi=130, facecolor=SURF)
print('wrote', G + '/results.png')
