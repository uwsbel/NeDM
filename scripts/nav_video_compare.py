"""Four planners, one mission, one video: the same route driven by each arm, synchronised on simulated time.

Each cell shows that arm's own sensed height map (the frame its current decision was made from), the routes it has
committed, the track it has driven and where it slid backwards. An arm that has finished or failed freezes with its
outcome. The strip underneath overlays the four speed traces with a time cursor, so the moment one arm stalls or
leaves the terrain is visible against what the others were doing.

  python nav_video_compare.py --runs DIR_W DIR_R2 DIR_R1 DIR_R1L --labels W R2 R1 R1L --out OUT.mp4
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vehicle_corridor as VC

HALF, DT = 40.0, 0.05
COLS = ['#1f5fbf', '#c0504d', '#2a8f3f', '#8a4fbf']
NICE = {'W': 'one decision per waypoint', 'R2': 'replan every 2 s', 'R1': 'replan every 1 s',
        'R1L': 'replan every 1 s, delay charged'}
STATUS = {'mission_complete': 'all waypoints reached', 'prolonged_blockage_terminated': 'STUCK — stall rule',
          'terrain_bounds_exit': 'LEFT THE ARENA', 'timeout': 'TIMED OUT', 'rollover': 'ROLLED OVER',
          'no_route': 'NO ROUTE', 'no_route_reroute': 'NO ROUTE', 'mission_timeout': 'TIMED OUT'}


def add_zone(ax, pose, margin, **kw):
    L, W = 2 * (VC.HALF_LENGTH_M + margin), 2 * (VC.HALF_WIDTH_M_VEH + margin)
    r = Rectangle((-L / 2, -W / 2), L, W, fill=False, **kw)
    r.set_transform(Affine2D().rotate(float(pose[2])).translate(float(pose[0]), float(pose[1])) + ax.transData)
    ax.add_patch(r)


def load(run):
    run = Path(run)
    o = json.load(open(run / 'mission_outcome.json'))
    decs = [d for d in json.load(open(run / 'decisions.json')) if 'route_id' in d]
    routes = json.load(open(run / 'routes.json'))
    z = np.load(run / 'trajectory.npz')
    vx, thr = z['state'][:, 0], z['action'][:, 1]
    slide = ((vx < -.10) & (thr > .3)) | (vx < -.30); slide[:20] = False
    frames = sorted((run / 'frames').glob('dec_*.npz'))
    if not frames:
        raise SystemExit(f'{run} has no saved frames; re-run with --save-frames')
    return dict(o=o, decs=decs, routes=routes, pose=z['pose'], vx=vx, slide=slide, frames=frames,
                dec_frames=[d['frame'] for d in decs], n=len(vx), cache={})


def cell(ax, R, fi, margin):
    """Draw one arm at simulated frame fi (clamped to its own end)."""
    done = fi >= R['n']
    k = min(fi, R['n'] - 1)
    active = max([i for i, f in enumerate(R['dec_frames']) if f <= k], default=0)
    active = min(active, len(R['frames']) - 1)
    if active not in R['cache']:
        R['cache'].clear()
        with np.load(R['frames'][active]) as z:
            R['cache'][active] = {q: z[q] for q in z.files}
    d = R['cache'][active]
    zz = d['z'].astype(float); zz[d['cover'] == 0] = np.nan
    lo, hi = np.nanpercentile(zz, [1, 99]) if np.isfinite(zz).any() else (0, 1)
    ax.imshow(zz, origin='lower', extent=[-HALF, HALF, -HALF, HALF], cmap='terrain',
              vmin=lo, vmax=hi, alpha=.55 if done else 1.0)
    for rr in R['routes'][:active + 1]:
        w = np.asarray(rr['waypoints'])
        ax.plot(w[:, 0], w[:, 1], '-', color='#555555', lw=.5, alpha=.35, zorder=2)
    r = d['route']
    ax.plot(r[:, 0], r[:, 1], '-', color='#ff2d00', lw=2.0, zorder=5)
    ax.plot(R['pose'][:k + 1, 0], R['pose'][:k + 1, 1], '-', color='#0b2fff', lw=2.0, zorder=6)
    sl = R['slide'][:k + 1]
    if sl.any():
        ax.plot(R['pose'][:k + 1][sl, 0], R['pose'][:k + 1][sl, 1], 'x', color='k', ms=6, mew=1.8, zorder=7)
    add_zone(ax, R['pose'][k], margin, ec='k', lw=1.1, zorder=8)
    ax.plot(*R['pose'][k, :2], 'o', color='w', ms=7, mec='k', mew=1.4, zorder=9)
    return done, k, active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs=4, required=True); ap.add_argument('--labels', nargs=4, required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--fps', type=int, default=20); ap.add_argument('--margin-m', type=float, default=1.5)
    a = ap.parse_args()
    Rs = [load(r) for r in a.runs]
    mid = Rs[0]['o']['mission']
    mfile = next((q for q in (p / 'missions' / f'{mid}.json' for p in Path(a.runs[0]).resolve().parents)
                  if q.exists()), None)
    goals = np.asarray(json.load(open(mfile))['goals']) if mfile else np.zeros((0, 2))
    n_max = max(R['n'] for R in Rs)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / (out.stem + '_png'); tmp.mkdir(exist_ok=True)
    t_all = np.arange(n_max) * DT
    n_png = 0
    for fi in range(0, n_max, a.stride):
        fig = plt.figure(figsize=(12.6, 12.0), dpi=95)
        gs = fig.add_gridspec(3, 2, height_ratios=[3, 3, 1.15], hspace=.30, wspace=.06)
        for i, (R, lab) in enumerate(zip(Rs, a.labels)):
            ax = fig.add_subplot(gs[i // 2, i % 2])
            done, k, active = cell(ax, R, fi, a.margin_m)
            if len(goals):
                ax.scatter(goals[:, 0], goals[:, 1], s=95, marker='*', c='gold', edgecolors='k', lw=.6, zorder=10)
            ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(COLS[i]); sp.set_linewidth(2.4)
            o = R['o']
            tag = ('\n' + STATUS.get(o['status'], o['status']) if done else '\n ')
            reached = sum(1 for L in o['legs'] if L.get('status') == 'goal_reached'
                          and L.get('end_frame', 10 ** 9) <= k + 1)
            ax.set_title(f"{lab} — {NICE.get(lab, lab)}\n{reached}/{o['n_goals']} waypoints · "
                         f"{active + 1} decisions · {R['vx'][k]:.1f} m/s{tag}",
                         fontsize=9.5, color=COLS[i], fontweight='bold' if done else 'normal', linespacing=1.35)
        axs = fig.add_subplot(gs[2, :])
        for i, (R, lab) in enumerate(zip(Rs, a.labels)):
            k = min(fi, R['n'] - 1)
            axs.plot(np.arange(k + 1) * DT, R['vx'][:k + 1], '-', color=COLS[i], lw=1.2, label=lab)
        axs.axhline(0, color='k', lw=.6)
        axs.axvline(fi * DT, color='#888888', lw=1.0, ls='--')
        axs.set_xlim(0, t_all[-1]); axs.set_ylim(-3.2, 7.2)
        axs.set_xlabel('simulated time (s)', fontsize=9); axs.set_ylabel('forward speed (m/s)', fontsize=9)
        axs.tick_params(labelsize=8); axs.grid(alpha=.25); axs.legend(fontsize=8, ncol=4, loc='upper right')
        fig.suptitle(f"{mid} — same mission, same frozen model, same controller;  t = {fi * DT:5.1f} s\n"
                     f"blue = driven, red = the route being followed, grey = every route committed so far, "
                     f"x = sliding backwards", fontsize=12)
        fig.savefig(tmp / f'{n_png:05d}.png', bbox_inches='tight'); plt.close(fig)
        n_png += 1
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(a.fps), '-i', str(tmp / '%05d.png'),
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '21', str(out)], check=True)
    for q in tmp.glob('*.png'):
        q.unlink()
    tmp.rmdir()
    print('wrote', out, n_png, 'frames')


if __name__ == '__main__':
    main()
