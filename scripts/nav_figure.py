"""One figure for nav_v1: outcomes by arm, the sensing-to-planning budget, and a matched mission side by side."""
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DT = 0.05
DEV = {'f104', 'g228', 'g203', 'g217', 'g216', 'g231'}


def load(root, mission, arm):
    d = os.path.join(root, 'runs', f'{mission}__{arm}')
    o = json.load(open(d + '/mission_outcome.json'))
    z = np.load(d + '/trajectory.npz')
    dec = json.load(open(d + '/decisions.json'))
    return o, z, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--summary', required=True)
    ap.add_argument('--mission', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--arms', nargs='+', default=['W', 'R2', 'R1', 'R1L'])
    a = ap.parse_args()
    S = json.load(open(a.summary))
    fig = plt.figure(figsize=(14.5, 8.6), dpi=115)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=.32, wspace=.28)

    # A: outcomes -------------------------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    arms = [x for x in a.arms if x in S['by_arm']]
    w = .38
    xs = np.arange(len(arms))
    comp = [100 * S['by_arm'][k]['complete_rate'] for k in arms]
    uns = [100 * S['by_arm'][k]['unsafe'] / max(S['by_arm'][k]['n'], 1) for k in arms]
    ax.bar(xs - w / 2, comp, w, color='#2a7f3f', label='missions completed')
    ax.bar(xs + w / 2, uns, w, color='#b32b2b', label='missions with a slide or a failed leg')
    for x, v in zip(xs, comp):
        ax.text(x - w / 2, v + 1.5, f'{v:.0f}', ha='center', fontsize=8)
    for x, v in zip(xs, uns):
        ax.text(x + w / 2, v + 1.5, f'{v:.0f}', ha='center', fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(arms); ax.set_ylim(0, 132); ax.set_ylabel('% of missions')
    ax.set_title(f"outcomes, {S['by_arm'][arms[0]]['n']} missions x 10 arenas", fontsize=10)
    ax.legend(fontsize=7, loc='upper center', ncol=1, framealpha=.95); ax.grid(axis='y', alpha=.25)

    # B: development vs unseen arenas ------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for i, (key, col, lab) in enumerate((('by_arm_dev', '#3b6fb5', 'development arenas'),
                                         ('by_arm_unseen', '#e08a1e', 'unseen arenas'))):
        v = [100 * S[key].get(k, {}).get('waypoint_rate', float('nan')) for k in arms]
        ax.bar(xs + (i - .5) * w, v, w, color=col, label=lab)
        for x, y in zip(xs, v):
            if np.isfinite(y):
                ax.text(x + (i - .5) * w, y + 1, f'{y:.0f}', ha='center', fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(arms); ax.set_ylim(0, 128)
    ax.set_ylabel('% of waypoints reached'); ax.set_title('waypoints reached', fontsize=10)
    ax.legend(fontsize=7, loc='upper center', framealpha=.95); ax.grid(axis='y', alpha=.25)

    # C: latency budget --------------------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    stages = [('render_s', 'depth render (simulator)', '#999999'),
              ('backproject_s', 'back-project', '#6a9fd8'),
              ('propose_s', 'candidates', '#8fc98f'),
              ('corridor_s', 'corridors', '#e6c35c'),
              ('score_s', 'risk model', '#c0504d')]
    bottom = np.zeros(len(arms))
    for key, lab, col in stages:
        v = np.array([S['latency'][k].get(key, {}).get('p50', 0.) for k in arms])
        ax.bar(xs, v, .55, bottom=bottom, color=col, label=lab)
        bottom += v
    for x, tot in zip(xs, bottom):
        ax.text(x, tot + .08, f'{tot:.1f}s', ha='center', fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(arms); ax.set_ylabel('seconds per decision (median)')
    ax.set_ylim(0, bottom.max() * 1.55)
    ax.set_title('sensing-to-planning wall clock', fontsize=10)
    ax.legend(fontsize=6.5, loc='upper center', ncol=2, framealpha=.95); ax.grid(axis='y', alpha=.25)

    # D/E: one matched mission -------------------------------------------------------------------------------
    mission = a.mission
    pair = [k for k in ('W', 'R1') if os.path.exists(
        os.path.join(a.root, 'runs', f'{mission}__{k}', 'mission_outcome.json'))]
    axm = fig.add_subplot(gs[1, :2])
    axs = fig.add_subplot(gs[1, 2])
    cols = {'W': '#3b6fb5', 'R1': '#c0504d'}
    for arm in pair:
        o, z, dec = load(a.root, mission, arm)
        p = z['pose']
        vx = z['state'][:, 0]; thr = z['action'][:, 1]
        sl = ((vx < -.10) & (thr > .3)) | (vx < -.30); sl[:20] = False
        axm.plot(p[:, 0], p[:, 1], '-', color=cols[arm], lw=2.0,
                 label=f"{arm}: {o['goals_reached']}/{o['n_goals']} waypoints, {o['total_time_s']:.0f} s, "
                       f"{o['n_decisions']} decisions")
        if sl.any():
            axm.plot(p[sl, 0], p[sl, 1], 'x', color='k', ms=6, mew=1.6,
                     label=f'{arm}: backward slide ({sl.sum() * DT:.1f} s)')
        t = np.arange(len(vx)) * DT
        axs.plot(t, vx, '-', color=cols[arm], lw=1.2, label=arm)
    mj = json.load(open(Path(a.root).parent / 'missions' / f'{mission}.json'))
    g = np.asarray(mj['goals'])
    axm.scatter(g[:, 0], g[:, 1], s=130, marker='*', c='gold', edgecolors='k', zorder=6, label='waypoints')
    axm.scatter(*mj['layout']['start_xy'], s=70, marker='s', c='w', edgecolors='k', zorder=6, label='start')
    axm.set_xlim(-40, 40); axm.set_ylim(-40, 40); axm.set_aspect('equal'); axm.grid(alpha=.25)
    axm.set_title(f"{mission}: one decision per waypoint vs 1 Hz replanning (same node, same mission)", fontsize=10)
    axm.legend(fontsize=7, loc='lower left', ncol=2, framealpha=.92, borderpad=.3, columnspacing=.9)
    axs.set_xlabel('simulated time (s)'); axs.set_ylabel('forward speed (m/s)')
    axs.axhline(0, color='k', lw=.6); axs.grid(alpha=.25); axs.legend(fontsize=8)
    axs.set_title('speed', fontsize=10)
    fig.savefig(a.out, bbox_inches='tight')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
