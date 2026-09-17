"""Video of a continuous navigation run: what the sensor saw at each decision, and how the route was updated.

Panels, left to right: the height map back-projected from that decision's own depth frame (whatever the sensor
did not see is white), a zoom on the vehicle showing the footprint exclusion the planner applies to its own
corridors, and the driven track with every route the planner has committed to. Underneath: the speed trace with a
tick at every decision. The vehicle moves at the simulation's 20 Hz; the sensed frame changes at decision rate.

  python nav_video.py --run DIR --out OUT.mp4 [--stride 2]
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


def add_zone(ax, pose, margin, **kw):
    L, W = 2 * (VC.HALF_LENGTH_M + margin), 2 * (VC.HALF_WIDTH_M_VEH + margin)
    r = Rectangle((-L / 2, -W / 2), L, W, fill=False, **kw)
    r.set_transform(Affine2D().rotate(float(pose[2])).translate(float(pose[0]), float(pose[1])) + ax.transData)
    ax.add_patch(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=int, default=20); ap.add_argument('--margin-m', type=float, default=1.5)
    ap.add_argument('--stride', type=int, default=2); ap.add_argument('--zoom-m', type=float, default=11.)
    a = ap.parse_args()
    run = Path(a.run)
    out_mp4 = Path(a.out); out_mp4.parent.mkdir(parents=True, exist_ok=True)
    mission = json.load(open(run / 'mission_outcome.json'))
    decs = [d for d in json.load(open(run / 'decisions.json')) if 'route_id' in d]
    routes = json.load(open(run / 'routes.json'))
    traj = np.load(run / 'trajectory.npz')
    pose_t = traj['pose']; vx = traj['state'][:, 0]; thr = traj['action'][:, 1]
    slide = ((vx < -.10) & (thr > .3)) | (vx < -.30); slide[:20] = False
    frames = sorted((run / 'frames').glob('dec_*.npz'))
    if not frames:
        raise SystemExit('no saved frames; re-run with --save-frames')
    dec_frames = [d['frame'] for d in decs]
    mid = mission['mission']
    mfile = next((q for q in (p / 'missions' / f'{mid}.json' for p in run.parents) if q.exists()), None)
    goal_xy = np.asarray(json.load(open(mfile))['goals']) if mfile else np.zeros((0, 2))
    tmp = out_mp4.parent / (out_mp4.stem + '_png'); tmp.mkdir(exist_ok=True)
    NICE = {('waypoint', 2.0, 0.0): 'one decision per waypoint',
            ('periodic', 2.0, 0.0): 'replan every 2 s',
            ('periodic', 1.0, 0.0): 'replan every 1 s',
            ('periodic', 1.0, -2.0): 'replan every 1 s, planning delay charged'}
    tag = NICE.get((mission['mode'], mission['period_s'], mission.get('latency_s_setting', 0.0)),
                   mission['mode'] + f" {mission['period_s']:.0f} s")
    STATUS = {'mission_complete': 'all waypoints reached', 'prolonged_blockage_terminated': 'STUCK (stall rule)',
              'terrain_bounds_exit': 'LEFT THE ARENA', 'timeout': 'TIMED OUT'}
    outcome = (f"{mission['goals_reached']}/{mission['n_goals']} waypoints - "
               f"{STATUS.get(mission['status'], mission['status'])}")
    n_png, cache = 0, {}
    t_all = np.arange(len(vx)) * DT
    for fi in range(0, len(pose_t), a.stride):
        active = max([i for i, f in enumerate(dec_frames) if f <= fi], default=0)
        if active not in cache:
            cache.clear()
            with np.load(frames[min(active, len(frames) - 1)]) as z:
                cache[active] = {k: z[k] for k in z.files}
        d = cache[active]
        fig = plt.figure(figsize=(14.2, 7.4), dpi=105)
        gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.0], hspace=.22, wspace=.16)
        ax0, ax1, ax2 = (fig.add_subplot(gs[0, i]) for i in range(3))
        axs = fig.add_subplot(gs[1, :])
        zz = d['z'].astype(float); zz[d['cover'] == 0] = np.nan
        lo, hi = np.nanpercentile(zz, [1, 99]) if np.isfinite(zz).any() else (0, 1)
        for ax in (ax0, ax1):
            ax.imshow(zz, origin='lower', extent=[-HALF, HALF, -HALF, HALF], cmap='terrain', vmin=lo, vmax=hi)
        r = d['route']
        ax0.plot(r[:, 0], r[:, 1], '-', color='#ff2d00', lw=2.4, zorder=5)
        if len(goal_xy):
            ax0.scatter(goal_xy[:, 0], goal_xy[:, 1], s=60, marker='*', c='w', edgecolors='k', zorder=7)
        add_zone(ax0, d['pose'], a.margin_m, ec='k', lw=1.4, zorder=8)
        ax0.set_xlim(-HALF, HALF); ax0.set_ylim(-HALF, HALF)
        ax0.set_title(f'what the depth frame gave the planner\n(white = not seen)  decision '
                      f'{active + 1}/{len(decs)}', fontsize=9.5)
        px, py = float(d['pose'][0]), float(d['pose'][1])
        ax1.plot(r[:, 0], r[:, 1], '-', color='#ff2d00', lw=3.0, zorder=5)
        add_zone(ax1, d['pose'], a.margin_m, ec='k', lw=2.2, zorder=8)
        add_zone(ax1, d['pose'], 0.0, ec='w', lw=1.4, ls='--', zorder=9)
        ax1.plot(pose_t[max(0, fi - 200):fi + 1, 0], pose_t[max(0, fi - 200):fi + 1, 1], '-', color='#0050ff', lw=2)
        ax1.set_xlim(px - a.zoom_m, px + a.zoom_m); ax1.set_ylim(py - a.zoom_m, py + a.zoom_m)
        ax1.set_title('vehicle footprint + 1.5 m margin, excluded\nfrom every candidate corridor', fontsize=9.5)
        for rr in routes[:active + 1]:
            w = np.asarray(rr['waypoints'])
            ax2.plot(w[:, 0], w[:, 1], '-', color='#999999', lw=.7, alpha=.55, zorder=2)
        ax2.plot(r[:, 0], r[:, 1], '-', color='#ff2d00', lw=2.2, zorder=4, label='active route')
        ax2.plot(pose_t[:fi + 1, 0], pose_t[:fi + 1, 1], '-', color='#0050ff', lw=2.0, zorder=5, label='driven')
        sl = slide[:fi + 1]
        if sl.any():
            ax2.plot(pose_t[:fi + 1][sl, 0], pose_t[:fi + 1][sl, 1], 'x', color='k', ms=5, mew=1.4,
                     zorder=6, label='backward slide')
        if len(goal_xy):
            ax2.scatter(goal_xy[:, 0], goal_xy[:, 1], s=90, marker='*', c='gold', edgecolors='k', zorder=7)
        ax2.plot(*pose_t[fi, :2], 'o', color='k', ms=6, zorder=8)
        ax2.set_xlim(-HALF, HALF); ax2.set_ylim(-HALF, HALF); ax2.set_aspect('equal'); ax2.grid(alpha=.25)
        ax2.set_title(f'routes committed: {active + 1}', fontsize=9.5)
        ax2.legend(fontsize=7.5, loc='upper right')
        for ax in (ax0, ax1, ax2):
            ax.set_xticks([]); ax.set_yticks([])
        axs.plot(t_all[:fi + 1], vx[:fi + 1], '-', color='#0050ff', lw=1.3)
        axs.plot(np.asarray(dec_frames[:active + 1]) * DT, np.zeros(active + 1) - .6, '|', color='#ff2d00', ms=7)
        axs.axhline(0, color='k', lw=.6)
        axs.set_xlim(0, t_all[-1]); axs.set_ylim(min(-1.2, vx.min() - .3), max(7., vx.max() + .3))
        axs.set_ylabel('speed (m/s)', fontsize=8.5); axs.set_xlabel('simulated time (s)', fontsize=8.5)
        axs.tick_params(labelsize=8); axs.grid(alpha=.25)
        fig.suptitle(f"{mid}  |  {tag}  |  outcome: {outcome}\n"
                     f"t = {fi * DT:5.1f} s    speed {vx[fi]:.1f} m/s    "
                     f"decision {active + 1}/{len(decs)}    sense-to-plan {decs[active]['sense_to_plan_s']:.1f} s",
                     fontsize=11)
        fig.savefig(tmp / f'{n_png:05d}.png', bbox_inches='tight'); plt.close(fig)
        n_png += 1
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(a.fps), '-i', str(tmp / '%05d.png'),
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '20', str(out_mp4)], check=True)
    for p in tmp.glob('*.png'):
        p.unlink()
    tmp.rmdir()
    print('wrote', out_mp4, n_png, 'frames')


if __name__ == '__main__':
    main()
