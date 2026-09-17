"""One map per demo scenario: the three routes the planner ranked, and what Chrono did with each."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patheffects as pe

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
OUT = ROOT + '/night2_v1/demo_v1'
VID = 'artifacts/f104_demo_v1'
SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
COL = {'optimal': '#0a8f2a', 'suboptimal': '#d18b00', 'risky': '#c81e1e'}
NAME = {'optimal': "planner's choice", 'suboptimal': 'middling', 'risky': 'worst-ranked'}
ROLES = ('optimal', 'suboptimal', 'risky')


def elevation():
    o = json.load(open(f'{ROOT}/static_map_v1/observation.json'))['camera']
    z = np.load(f'{ROOT}/static_map_v1/observation.npz')
    e = np.where(z['rgbd'][3] > -1.999, z['rgbd'][3].astype(float) * o['elevation_scale_m'], np.nan)
    n = e.shape[0]; mpp = (2 * o['cam_height_m'] * np.tan(o['hfov_rad'] / 2)) / n; h = n / 2 * mpp
    return np.flipud(e), [-h, h, -h, h]


def main():
    E, EXT = elevation(); ground = float(np.nanmedian(E))
    meta = {(m['scenario'], m['role']): m for m in json.load(open(OUT + '/video_meta.json'))}
    scen = sorted({k[0] for k in meta})
    fig = plt.figure(figsize=(4.2 * len(scen), 6.2), facecolor=SURF)
    gs = fig.add_gridspec(1, len(scen), wspace=.10, left=.015, right=.985, top=.84, bottom=.20)
    norm = TwoSlopeNorm(vmin=np.nanpercentile(E, 1), vcenter=ground, vmax=np.nanpercentile(E, 99.5))
    for k, s in enumerate(scen):
        ax = fig.add_subplot(gs[0, k]); ax.set_facecolor(SURF)
        ax.imshow(E, extent=EXT, origin='lower', cmap='BrBG_r', norm=norm, interpolation='bilinear')
        pts = np.concatenate([np.load(f"{OUT}/ribbons/{meta[(s, r)]['id']}.npy")[:, :2] for r in ROLES])
        c = pts.mean(0); half = max(np.abs(pts - c).max() + 8.0, 16.0)
        lo = np.clip(c - half, EXT[0], EXT[1] - 2 * half); ax.set_aspect('equal')
        ax.set_xlim(lo[0], lo[0] + 2 * half); ax.set_ylim(lo[1], lo[1] + 2 * half)
        for r in ROLES:
            m = meta[(s, r)]
            wp = np.asarray(json.load(open(f"{OUT}/routes/{m['id']}.json"))['waypoints'], float)
            ax.plot(wp[:, 0], wp[:, 1], color=COL[r], lw=1.6, ls=(0, (4, 3)), alpha=.85, zorder=3)
            z = np.load(f"{OUT}/blend/{m['id']}/run/trajectory.npz") if os.path.exists(
                f"{OUT}/blend/{m['id']}/run/trajectory.npz") else np.load(f"{OUT}/runs/{m['id']}/trajectory.npz")
            xy = z['pose'][:, :2]
            outcome = ('stalled' if 'blockage' in m['status'] else
                       'never arrived' if m['fail'] else '%.0f s to goal' % m['elapsed'])
            ax.plot(xy[:, 0], xy[:, 1], color=COL[r], lw=2.8, alpha=.95, zorder=4,
                    label='%s   %.2f%%  ->  %s' % (NAME[r], 100 * m['risk'], outcome))
            if m['fail']:
                ax.plot(*xy[-1], marker='X', ms=11, color=COL[r], mec='white', mew=1.4, zorder=6)
        g = meta[(s, 'optimal')]['goal']
        ax.add_patch(plt.Circle((g[0], g[1]), g[2], fill=False, ec='#111', lw=1.4, zorder=5))
        st = np.load(f"{OUT}/ribbons/{meta[(s, 'optimal')]['id']}.npy")[0, :2]
        ax.plot(*st, marker='o', ms=8, color='#111', zorder=6)
        ax.text(st[0], st[1] - 2.6, 'start', ha='center', fontsize=8.5, color=INK,
                path_effects=[pe.withStroke(linewidth=2.4, foreground='white')])
        ax.text(g[0], g[1] + 3.4, 'goal', ha='center', fontsize=8.5, color=INK,
                path_effects=[pe.withStroke(linewidth=2.4, foreground='white')])
        ax.set_title(f'Scenario {s}', fontsize=12, color=INK)
        ax.legend(fontsize=8.2, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.02),
                  handlelength=1.6, borderpad=0.2, labelspacing=.45)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color('#d9d8d4')
    fig.text(.015, .945, 'Five scenarios, three routes each: what the model predicted and what Chrono did',
             fontsize=16, color=INK, weight='bold')
    fig.text(.015, .905, 'Dashed = the route the planner proposed.  Solid = the path the vehicle actually drove.  '
                         'X = where it stalled.  Brown is high ground, teal is a dip; the model sees only this '
                         'terrain and the route it is asked about.', fontsize=10, color=INK2)
    os.makedirs(VID, exist_ok=True)
    fig.savefig(VID + '/scenarios.png', dpi=140, facecolor=SURF)
    print('wrote', VID + '/scenarios.png')


if __name__ == '__main__':
    main()
