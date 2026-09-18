"""Report figures for the CRM f104 night session (static PNG; palette = validated default slots 1-2, surface #fcfcfb).

  fig 1  goal-not-reached rate by route type: CRM episode vs the rigid-ground twin of the SAME route id
  fig 2  held-out single-goal CRM missions: goal reached per planner arm with 95 % Wilson intervals
  python scripts/crm_figures.py --ds <crm npz> [--results <eval results.json>] --out <dir>
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'figure.facecolor': SURFACE,
                     'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE})


def style(ax):
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)


def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def fig_twins(ds, out):
    d = np.load(ds, allow_pickle=True)
    r = np.load('artifacts/traverse/crm_f104_v1/rigid_labels_station_ds_all.npz', allow_pickle=True)
    rig = {i: int(f) for i, f in zip(r['id'], r['fail'])}
    names = ['2 m/s', '4 m/s', '6 m/s', '2-6-2 m/s', 'planner\nproposals']
    kind = np.where(d['profile'] >= 0, d['profile'], 4)
    crm, rg, ns = [], [], []
    for k in range(5):
        m = (kind == k) & np.array([i in rig for i in d['id']])
        crm.append(100 * d['fail'][m].mean()); rg.append(100 * np.mean([rig[i] for i in d['id'][m]])); ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(7.2, 3.8)); x = np.arange(5); w = 0.34
    b1 = ax.bar(x - w / 2 - 0.01, rg, w, color=BLUE, label='rigid ground (same routes)')
    b2 = ax.bar(x + w / 2 + 0.01, crm, w, color=ORANGE, label='CRM soil')
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f'{b.get_height():.0f}', ha='center', va='bottom', fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([f'{n}\nn={k:,}' for n, k in zip(names, ns)])
    ax.set_ylabel('goal not reached (%)'); ax.set_ylim(0, 105); style(ax)
    ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.14))
    ax.set_title('Same routes, same controller: failure on CRM soil vs rigid ground', loc='left', fontsize=11, pad=26)
    fig.tight_layout(); fig.savefig(f'{out}/fig_crm_vs_rigid_twins.png', dpi=160); plt.close(fig)
    return dict(names=names, n=ns, crm_fail_pct=crm, rigid_fail_pct=rg)


def fig_eval(results, out):
    s = json.load(open(results))['summary']; R = s['rates']
    arms = [('crm', 'CRM-trained\nspeed free'), ('rigid', 'rigid-trained\nspeed free'), ('crm_pess', 'CRM-trained\npessimistic'),
            ('rigid_pess', 'rigid-trained\npessimistic'), ('straight6', 'straight\n6 m/s'), ('crm_fixed2', 'CRM-trained\nfixed 2 m/s'),
            ('rigid_fixed2', 'rigid-trained\nfixed 2 m/s'), ('straight2', 'straight\n2 m/s')]
    fig, ax = plt.subplots(figsize=(8.6, 3.9)); x = np.arange(len(arms))
    vals = [R[a]['goal_reached'] for a, _ in arms]; n = R['crm']['n']
    cis = [wilson(round(v * n / 100), n) for v in vals]
    cols = [ORANGE if a.startswith('crm') else BLUE for a, _ in arms]
    ax.bar(x, vals, 0.56, color=cols)
    ax.errorbar(x, vals, yerr=[[v - lo for v, (lo, hi) in zip(vals, cis)], [hi - v for v, (lo, hi) in zip(vals, cis)]], fmt='none', ecolor=INK2, elinewidth=1, capsize=3)
    for xi, v, (lo, hi) in zip(x, vals, cis):
        ax.text(xi, hi + 1.5, f'{v:.0f}', ha='center', va='bottom', fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in arms], fontsize=8.5); ax.set_ylim(0, 108); ax.set_ylabel('goal reached (%)'); style(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label='network trained on CRM'), Patch(color=BLUE, label='frozen rigid-trained network / no network')],
              frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15))
    ax.set_title(f'Held-out single-goal missions on CRM soil ({n} start-goal pairs, 95 % intervals)', loc='left', fontsize=11, pad=26)
    fig.tight_layout(); fig.savefig(f'{out}/fig_eval_goal_reached.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--ds'); ap.add_argument('--results'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import os; os.makedirs(a.out, exist_ok=True)
    if a.ds: print(json.dumps(fig_twins(a.ds, a.out)))
    if a.results: fig_eval(a.results, a.out)
