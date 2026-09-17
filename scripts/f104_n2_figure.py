"""Night-2 figures: what the proposal change did, what more data did, and what happened in Chrono."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
N2 = ROOT + '/night2_v1'
SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
C1, C2, C3 = '#b0692c', '#2a78d6', '#008300'


def chord_offset(wp):
    """Max lateral distance from the straight start-goal chord; one definition for both proposals."""
    a = wp[:, 0, :]; b = wp[:, -1, :]
    d = b - a; L = np.linalg.norm(d, axis=1, keepdims=True); L[L < 1e-9] = 1e-9
    u = d / L
    rel = wp - a[:, None, :]
    along = (rel * u[:, None, :]).sum(-1)
    perp = rel - along[..., None] * u[:, None, :]
    return np.abs(np.linalg.norm(perp, axis=-1)).max(1)


def proposal_panel(ax1, ax2):
    lat1, sp1, lat2, sp2 = [], [], [], []
    for f in sorted(glob.glob(N2 + '/testcand/*__night1.npz'))[:60]:
        z = np.load(f); lat1.append(chord_offset(z['wp'])); sp1.append(z['sp'][:, 10:-10].mean(1))
        y = np.load(f.replace('night1', 'night2')); lat2.append(chord_offset(y['wp'])); sp2.append(y['sp'][:, 10:-10].mean(1))
    lat1 = np.concatenate(lat1); sp1 = np.concatenate(sp1); lat2 = np.concatenate(lat2); sp2 = np.concatenate(sp2)
    ax1.scatter(lat2, sp2, s=3, alpha=.12, color=C2, label='night-2 proposal')
    ax1.scatter(lat1, sp1, s=3, alpha=.12, color=C1, label='night-1 proposal')
    ax1.set_xlabel('detour width (m)', fontsize=9.5, color=INK2); ax1.set_ylabel('route mean speed (m/s)', fontsize=9.5, color=INK2)
    ax1.set_title('What the planner is allowed to consider', fontsize=11, color=INK)
    leg = ax1.legend(fontsize=9, markerscale=4, frameon=False); 
    for h in leg.legend_handles: h.set_alpha(1)
    e1, e2 = [], []
    for f in sorted(glob.glob(N2 + '/testcand/*__night1.npz'))[:60]:
        z = np.load(f); e1.append(z['sp'][:, int(.85 * z['sp'].shape[1])])
        y = np.load(f.replace('night1', 'night2')); e2.append(y['sp'][:, int(.85 * y['sp'].shape[1])])
    e1 = np.concatenate(e1); e2 = np.concatenate(e2)
    bins = np.linspace(0, 6.2, 40)
    ax2.hist(e1, bins=bins, color=C1, alpha=.65, label='night-1')
    ax2.hist(e2, bins=bins, color=C2, alpha=.65, label='night-2')
    ax2.set_xlabel('commanded speed at 85% of the route (m/s)', fontsize=9.5, color=INK2)
    ax2.set_ylabel('candidates', fontsize=9.5, color=INK2)
    ax2.set_title('Approach speed near the goal', fontsize=11, color=INK)
    ax2.legend(fontsize=9, frameon=False)


def curve_panel(ax):
    rows = json.load(open(N2 + '/scaling.json'))
    des = [r for r in rows if r['tag'] == 'designed']
    fr = sorted({r['frac'] for r in des})
    x = [np.mean([r['n_rows'] for r in des if r['frac'] == f]) for f in fr]
    for key, c, lab in (('G_unsafe', C2, 'ranking AUC (same group, same speed)'),):
        mu = [np.mean([r[key] for r in des if r['frac'] == f]) for f in fr]
        sd = [np.std([r[key] for r in des if r['frac'] == f]) for f in fr]
        ax.errorbar(x, mu, yerr=sd, marker='o', color=c, capsize=3, label=lab)
    ax.set_xscale('log'); ax.set_xlabel('training routes (designed)', fontsize=9.5, color=INK2)
    ax.set_ylabel('ranking AUC on held-out groups', fontsize=9.5, color=INK2)
    ax.set_title('Does more data help?  (still rising)', fontsize=11, color=INK)
    ax.legend(fontsize=8.5, frameon=False, loc='lower right')


def onpolicy_panel(ax):
    rows = json.load(open(N2 + '/onpolicy_eval.json'))
    tags = ['designed only', 'designed + on-policy']
    mu = [np.mean([r['auc_unsafe'] for r in rows if r['tag'] == t]) for t in tags]
    sd = [np.std([r['auc_unsafe'] for r in rows if r['tag'] == t]) for t in tags]
    ax.bar([0, 1], mu, yerr=sd, color=[C1, C3], width=.6, capsize=4)
    for i, v in enumerate(mu): ax.text(i, v + .004, f'{v:.3f}', ha='center', fontsize=10, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['trained on\ndesigned routes', "trained also on\nthe planner's own\nproposals"],
                       fontsize=8.5, color=INK2)
    ax.set_ylim(.90, 1.0)
    ax.set_ylabel("ranking AUC on the planner's proposals", fontsize=9, color=INK2)
    ax.set_title('The data that matters is\nwhat the planner asks about', fontsize=11, color=INK)


def arms_panel(ax, key='fail', compact=False):
    res = json.load(open(N2 + '/closed/results.json'))
    arms = ('control', 'model', 'sampler', 'pessimist', 'anchor6', 'fixed2_control', 'fixed2_new')
    names = {'control': "last night's\nplanner", 'model': '+ new model\n(same routes\noffered)',
             'sampler': '+ new route\nproposals', 'pessimist': '+ pessimistic\nensemble',
             'anchor6': 'always 6 m/s\nstraight', 'fixed2_control': 'at 2 m/s:\nold model',
             'fixed2_new': 'at 2 m/s:\nnew model'}
    G = list(res)
    rng = np.random.default_rng(0)
    vals, los, his = [], [], []
    for a in arms:
        v = np.array([res[g][a][key] for g in G])
        b = [v[i].mean() for i in (rng.integers(0, len(G), len(G)) for _ in range(4000))]
        vals.append(100 * v.mean()); los.append(100 * np.percentile(b, 2.5)); his.append(100 * np.percentile(b, 97.5))
    cols = [C1, '#7b5cb8', C2, '#00897b', INK2, '#b8b5ae', '#8a867c']
    ax.bar(range(len(arms)), vals, color=cols, width=.68)
    ax.errorbar(range(len(arms)), vals, yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
                fmt='none', ecolor=INK, capsize=4, lw=1.2)
    for i, v in enumerate(vals):
        ax.text(i, v + (max(his) - min(los)) * .04, f'{v:.1f}%', ha='center', fontsize=10, color=INK)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([names[a] for a in arms], fontsize=6.5 if compact else 8.5, color=INK2,
                       rotation=35 if compact else 0, ha='right' if compact else 'center')
    ax.set_ylabel(f'{"did not reach goal" if key == "fail" else "failed or slid backwards"} (%)', fontsize=9.5, color=INK2)
    ax.set_title(f'Chrono, {len(G)} fresh start/goals: {"real failures" if key == "fail" else "unsafe runs"}',
                 fontsize=11, color=INK)


def main():
    has_closed = os.path.exists(N2 + '/closed/results.json')
    fig = plt.figure(figsize=(20, 10 if has_closed else 5.2), facecolor=SURF)
    if has_closed:
        gs = fig.add_gridspec(2, 4, hspace=.46, wspace=.32, left=.055, right=.985, top=.88, bottom=.10)
        proposal_panel(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]))
        curve_panel(fig.add_subplot(gs[0, 2])); onpolicy_panel(fig.add_subplot(gs[0, 3]))
        arms_panel(fig.add_subplot(gs[1, :3]), 'unsafe')
        arms_panel(fig.add_subplot(gs[1, 3]), 'fail', compact=True)
    else:
        gs = fig.add_gridspec(1, 4, wspace=.32, left=.055, right=.985, top=.82, bottom=.16)
        proposal_panel(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]))
        curve_panel(fig.add_subplot(gs[0, 2])); onpolicy_panel(fig.add_subplot(gs[0, 3]))
    fig.text(.06, .955, 'Night 2: wider, faster route proposals + more data', fontsize=16, color=INK, weight='bold')
    fig.text(.06, .925, 'One fixed arena. Test start/goals are at least 6 m (in start+goal space) from every group the model trained on.',
             fontsize=10, color=INK2)
    for ax in fig.axes:
        ax.set_facecolor(SURF); ax.tick_params(colors=INK2, labelsize=8.5)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'): ax.spines[s].set_color('#d9d8d4')
    fig.savefig(N2 + '/night2_summary.png', dpi=130, facecolor=SURF)
    print('wrote', N2 + '/night2_summary.png')


if __name__ == '__main__':
    main()
