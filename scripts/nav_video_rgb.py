"""Videos of continuous navigation from Chrono::Sensor's own RGB cameras, with the planner drawn on top.

Frames come from a run made with `nav_runner.py --video-dir DIR --chase-cam`: the overhead RGB camera (the same
110 m / 47 deg camera whose depth the planner reads) and a chase camera that follows the vehicle's position and
heading. Planner geometry is projected into the overhead image with the camera model
(u = cx + f x / (110 - z), v = cy - f y / (110 - z), f = (W/2) / tan(hfov/2), image row 0 = +y, z from the arena
heightmap). Nothing in the camera images is synthetic.

What is drawn, and when:
  * routes in use (white) - a route is drawn from the frame it actually took effect, which for the delay-charged
    planner is after its charged delay; routes planned but never used are not drawn;
  * the route being followed (red); rescue routes (planned when no normal route fit) dashed;
  * driven track (blue), backward sliding (black x), vehicle footprint mask (yellow box);
  * waypoints: gold = active, green = reached, white = not yet;
  * speed strip: a tick for every planning attempt - red when it produced a route, grey x when no route was found.

  single:  python nav_video_rgb.py single --run RUN_DIR --out OUT.mp4
  compare: python nav_video_rgb.py compare --runs W R2 R1 R1L --labels W R2 R1 R1L --out OUT.mp4
"""
import argparse, json, math, shutil, subprocess, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DT = 0.05
NICE = {'W': 'one decision per waypoint', 'R2': 'replan every 2 s', 'R1': 'replan every 1 s',
        'R1L': 'replan every 1 s, planning delay charged'}
STATUS = {'mission_complete': 'all waypoints reached', 'prolonged_blockage_terminated': 'STUCK (stall rule)',
          'terrain_bounds_exit': 'LEFT THE ARENA', 'timeout': 'TIMED OUT', 'mission_timeout': 'TIMED OUT',
          'rollover': 'ROLLED OVER', 'no_route': 'NO VALID ROUTE', 'no_route_reroute': 'NO VALID ROUTE'}
COLS = ['#e69f00', '#009e73', '#cc79a7', '#555555']      # planner colours: never the overlay's red/blue
HALF_LENGTH_M, HALF_WIDTH_M, MARGIN_M = 2.6, 1.3, 1.5
HOLD_S = 2.5


def arm_of(o):
    if o['mode'] == 'waypoint':
        return 'W'
    if o.get('latency_s_setting', 0.) != 0:
        return 'R1L'
    return 'R2' if abs(o['period_s'] - 2.0) < 1e-9 else 'R1'


def fmt_speed(v):
    return f"{0.0 if abs(v) < 0.05 else v:.1f}"


class Run:
    def __init__(self, run_dir):
        sys.path.insert(0, str(ROOT / 'src'))
        from nedm.traverse.terrain import TerrainMap
        self.dir = Path(run_dir)
        self.o = json.load(open(self.dir / 'mission_outcome.json'))
        self.idx = json.load(open(self.dir / 'video' / 'index.json'))
        cam = self.idx['camera']
        self.W, self.H, self.cam_h = cam['width'], cam['height'], cam['cam_height_m']
        self.f = (self.W / 2) / math.tan(cam['hfov_rad'] / 2)
        self.frames = self.idx['frames']
        self.by_frame = {fr['frame']: fr for fr in self.frames}
        self.routes = {r['route_id']: r for r in json.load(open(self.dir / 'routes.json'))}
        self.attempts = json.load(open(self.dir / 'decisions.json'))
        rescue = {d['route_id'] for d in self.attempts if 'route_id' in d and
                  (d.get('used_fallback_base') or 'surrogate_goal' in d or 'straight_ahead' in d)}
        self.rescue = rescue
        # the frame each route took effect: first captured frame on which it was the followed route
        self.active_from = {}
        for fr in self.frames:
            rid = fr.get('active_route_id')
            if rid is not None and rid not in self.active_from:
                self.active_from[rid] = fr['frame']
        z = np.load(self.dir / 'trajectory.npz')
        self.pose, self.vx = z['pose'], z['state'][:, 0]
        thr = z['action'][:, 1]
        self.slide = ((self.vx < -.10) & (thr > .3)) | (self.vx < -.30); self.slide[:20] = False
        self.n = len(self.vx)
        mid = self.o['mission']
        cands = [ROOT / 'artifacts/traverse/fdm_f104_50h_20260909/nav_v1'] + list(self.dir.resolve().parents)
        mfile = next(q for q in (p / 'missions' / f'{mid}.json' for p in cands) if q.exists())
        self.mission = json.load(open(mfile))
        self.goals = np.asarray(self.mission['goals'])
        self.tmap = TerrainMap.from_dir(ROOT / self.mission['arena'])
        self.goal_end = sorted(L['end_frame'] for L in self.o['legs'] if L.get('status') == 'goal_reached')
        self.last_video_frame = self.frames[-1]['frame']

    def px(self, xy):
        xy = np.atleast_2d(np.asarray(xy, float))
        z = self.tmap.height(np.clip(xy[:, 0], -39.9, 39.9), np.clip(xy[:, 1], -39.9, 39.9))
        axial = self.cam_h - z
        return (self.W - 1) / 2 + self.f * xy[:, 0] / axial, (self.H - 1) / 2 - self.f * xy[:, 1] / axial

    def k(self, fi):
        return min(fi, self.n - 1)

    def ended(self, fi):
        return fi >= self.n - 1

    def reached(self, fi):
        """Waypoints reached by trajectory frame k: one test used for both the stars and the titles."""
        return sum(1 for e in self.goal_end if e <= self.k(fi) + 1)

    def frame_at(self, fi):
        fi = min(fi, self.last_video_frame)
        fi -= fi % self.idx['every_frames']
        while fi not in self.by_frame and fi > 0:
            fi -= self.idx['every_frames']
        return self.by_frame[fi]


def draw_overhead(ax, R, fi, faded=False, legend=False):
    from PIL import Image
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D
    vf = R.frame_at(fi)
    img = np.asarray(Image.open(R.dir / 'video' / f"ov_{vf['frame']:06d}.jpg"))
    ax.imshow(img, alpha=.55 if faded else 1.0)
    for rid, start in R.active_from.items():
        if start <= vf['frame']:
            u, v = R.px(R.routes[rid]['waypoints'])
            ax.plot(u, v, '--' if rid in R.rescue else '-', color='white', lw=.6, alpha=.5, zorder=2)
    rid = vf.get('active_route_id')
    if rid is not None and rid in R.routes:
        u, v = R.px(R.routes[rid]['waypoints'])
        ax.plot(u, v, '--' if rid in R.rescue else '-', color='#ff2d00', lw=2.2, zorder=4)
    k = R.k(fi)
    u, v = R.px(R.pose[:k + 1, :2]); ax.plot(u, v, '-', color='#0b2fff', lw=2.0, zorder=5)
    sl = R.slide[:k + 1]
    if sl.any():
        u, v = R.px(R.pose[:k + 1][sl, :2]); ax.plot(u, v, 'x', color='k', ms=4.5, mew=1.4, zorder=6)
    nr = R.reached(fi)
    for j, g in enumerate(R.goals):
        u, v = R.px(g)
        colour = '#2bd12b' if j < nr else ('gold' if j == nr and not R.ended(fi) else 'white')
        ax.plot(u, v, marker='*', ms=14 if j == nr else 11, mfc=colour, mec='k', mew=.8, zorder=7, ls='none')
        ax.text(float(u[0]) + 9, float(v[0]) - 9, str(j + 1), fontsize=8, zorder=8,
                bbox=dict(boxstyle='round,pad=.12', fc='white', ec='none', alpha=.7))
    p = R.pose[k]
    c, s = math.cos(p[2]), math.sin(p[2])
    L, W = HALF_LENGTH_M + MARGIN_M, HALF_WIDTH_M + MARGIN_M
    corners = np.array([[p[0] + c * a - s * b, p[1] + s * a + c * b] for a, b in ((L, W), (L, -W), (-L, -W), (-L, W))])
    u, v = R.px(corners)
    ax.add_patch(Polygon(np.c_[u, v], closed=True, fill=False, ec='yellow', lw=1.2, zorder=8))
    half = 45.0
    u0 = (R.W - 1) / 2 - R.f * half / R.cam_h; u1 = (R.W - 1) / 2 + R.f * half / R.cam_h
    v0 = (R.H - 1) / 2 - R.f * half / R.cam_h; v1 = (R.H - 1) / 2 + R.f * half / R.cam_h
    ax.set_xlim(u0, u1); ax.set_ylim(v1, v0); ax.set_xticks([]); ax.set_yticks([])
    if legend:
        star = lambda c: Line2D([], [], marker='*', ms=11, mfc=c, mec='k', ls='none')
        handles = [Line2D([], [], color='#ff2d00', lw=2.2), Line2D([], [], color='#ff2d00', lw=2.2, ls='--'),
                   Line2D([], [], color='white', lw=1.5), Line2D([], [], color='#0b2fff', lw=2),
                   Line2D([], [], color='k', marker='x', ls='none'), Line2D([], [], color='yellow', lw=1.5),
                   star('gold'), star('#2bd12b'), star('white')]
        labels = ['route being followed', 'rescue route (no normal route fit)', 'routes used so far', 'driven',
                  'sliding backwards', 'footprint masked from the planner', 'active waypoint', 'reached', 'not yet']
        leg = ax.legend(handles, labels, loc='lower left', fontsize=7, framealpha=.9, ncol=2)
        leg.get_frame().set_facecolor('#dddddd')
    rid_rescue = rid in R.rescue if rid is not None else False
    return vf, rid_rescue


def speed_strip(ax, runs, fi, t_end, colours, labels, attempts_of=None):
    for R, col, lab in zip(runs, colours, labels):
        k = R.k(fi)
        ax.plot(np.arange(k + 1) * DT, R.vx[:k + 1], '-', color=col, lw=1.3, label=lab)
    if attempts_of is not None:
        R = attempts_of
        ok = [d['frame'] * DT for d in R.attempts if d['frame'] <= R.k(fi) + 1 and 'route_id' in d]
        bad = [d['frame'] * DT for d in R.attempts if d['frame'] <= R.k(fi) + 1 and 'route_id' not in d]
        ax.plot(ok, np.full(len(ok), -2.6), '|', color='#ff2d00', ms=8, clip_on=False)
        if bad:
            ax.plot(bad, np.full(len(bad), -2.6), 'x', color='#777777', ms=7, mew=1.6, clip_on=False)
    ax.axhline(0, color='k', lw=.6)
    ax.set_xlim(-1.0, t_end + 0.5); ax.set_ylim(-3.2, 8.0); ax.grid(alpha=.25); ax.tick_params(labelsize=8)
    ax.set_ylabel('speed (m/s)', fontsize=9)


def render_single_chunk(args):
    run_dir, items, tmp = args
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    R = Run(run_dir)
    arm = arm_of(R.o)
    for n, fi in items:
        fig = plt.figure(figsize=(16, 9), dpi=90)
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], height_ratios=[4.2, 1], hspace=.14, wspace=.03,
                              left=.045, right=.985, top=.88, bottom=.07)
        ax = fig.add_subplot(gs[0, 0])
        vf, on_rescue = draw_overhead(ax, R, fi, legend=True)
        ax.set_title("overhead RGB camera - the planner reads this camera's depth", fontsize=10)
        axc = fig.add_subplot(gs[0, 1])
        ch = R.dir / 'video' / f"ch_{vf['frame']:06d}.jpg"
        if ch.exists():
            axc.imshow(np.asarray(Image.open(ch)))
        axc.set_xticks([]); axc.set_yticks([]); axc.set_title('chase camera (follows position and heading)', fontsize=10)
        axs = fig.add_subplot(gs[1, :])
        speed_strip(axs, [R], fi, R.n * DT, ['#0b2fff'], [arm], attempts_of=R)
        axs.set_xlabel('simulated time (s)  -  red ticks: planning attempts that produced a route;  grey x: no valid route found', fontsize=9)
        done = R.ended(fi)
        n_att = sum(1 for d in R.attempts if d['frame'] <= R.k(fi) + 1)
        notes = []
        if on_rescue and not done:
            notes.append(f"no normal route to waypoint {min(R.reached(fi) + 1, len(R.goals))} fit - following a rescue route")
        if arm == 'R1L' and not done:
            latest = [d for d in R.attempts if d['frame'] <= R.k(fi) + 1 and 'route_id' in d]
            if latest and latest[-1]['route_id'] not in R.active_from or \
                    (latest and R.active_from.get(latest[-1]['route_id'], 10 ** 9) > vf['frame']):
                wait = latest[-1]['frame'] * DT + latest[-1].get('latency_charged_s', 0.) - fi * DT
                if wait > 0:
                    notes.append(f"new route planned, takes effect in {wait:.1f} s")
        t_show = R.o['total_time_s'] if done else fi * DT
        head = (f"{R.o['mission']}  |  {NICE[arm]}  |  outcome of this run: {R.o['goals_reached']}/{R.o['n_goals']} waypoints - "
                f"{STATUS.get(R.o['status'], R.o['status'])}")
        line2 = (f"t = {t_show:5.1f} s    speed {fmt_speed(R.vx[R.k(fi)])} m/s    waypoints {R.reached(fi)}/{len(R.goals)}    "
                 f"planning attempts {n_att}/{len(R.attempts)}" + ('    (run ended)' if done else ''))
        fig.text(.5, .965, head, ha='center', fontsize=12.5)
        fig.text(.5, .93, line2 + ('    ·    ' + ';  '.join(notes) if notes else ''), ha='center', fontsize=11,
                 color='#b00000' if notes else 'k')
        fig.savefig(tmp / f'{n:05d}.png'); plt.close(fig)
    return len(items)


def render_compare_chunk(args):
    run_dirs, labels, items, tmp, t_end = args
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    Rs = [Run(d) for d in run_dirs]
    for n, fi in items:
        fig = plt.figure(figsize=(19.2, 12.0), dpi=80)
        gs = fig.add_gridspec(3, 4, height_ratios=[3.2, 1.75, 1.05], hspace=.2, wspace=.04,
                              left=.04, right=.99, top=.815, bottom=.05)
        for i, (R, lab) in enumerate(zip(Rs, labels)):
            done = R.ended(fi)
            ax = fig.add_subplot(gs[0, i]); vf, on_rescue = draw_overhead(ax, R, fi, faded=done)
            for sp in ax.spines.values():
                sp.set_color(COLS[i]); sp.set_linewidth(3.5)
            status = STATUS.get(R.o['status'], R.o['status']) if done else ('rescue route' if on_rescue else ' ')
            ax.set_title(f"{lab}: {NICE[lab]}\n{R.reached(fi)}/{len(R.goals)} waypoints · {fmt_speed(R.vx[R.k(fi)])} m/s\n{status}",
                         fontsize=10.5, color=COLS[i], fontweight='bold' if done else 'normal')
            axc = fig.add_subplot(gs[1, i])
            ch = R.dir / 'video' / f"ch_{vf['frame']:06d}.jpg"
            if ch.exists():
                axc.imshow(np.asarray(Image.open(ch)), alpha=.55 if done else 1.)
            axc.set_xticks([]); axc.set_yticks([])
            for sp in axc.spines.values():
                sp.set_color(COLS[i]); sp.set_linewidth(3.5)
        axs = fig.add_subplot(gs[2, :])
        speed_strip(axs, Rs, fi, t_end, COLS, labels)
        tclamp = min(fi * DT, t_end)
        axs.axvline(tclamp, color='#888888', lw=1, ls='--')
        axs.set_xlabel('simulated time (s)', fontsize=10); axs.legend(fontsize=9, ncol=4, loc='upper right')
        fig.text(.5, .975, f"{Rs[0].o['mission']} - one continuous Chrono rollout per planner, rendered by Chrono::Sensor (OptiX, RTX 5090);  "
                 f"t = {tclamp:5.1f} s", ha='center', fontsize=13.5)
        fig.text(.5, .953, "same mission, frozen model and controller; the planners differ in when they plan, and each draws its own random candidate routes",
                 ha='center', fontsize=11)
        fig.text(.5, .931, "red: route being followed (dashed = rescue route) · white: routes used so far · blue: driven · x: sliding backwards · "
                 "yellow box: footprint masked from the planner · stars: gold active, green reached, white not yet", ha='center', fontsize=10.5)
        fig.savefig(tmp / f'{n:05d}.png'); plt.close(fig)
    return len(items)


def encode(tmp, out, fps):
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', str(tmp / '%05d.png'),
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
                    '-movflags', '+faststart', str(out)], check=True)
    shutil.rmtree(tmp)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s1 = sub.add_parser('single'); s1.add_argument('--run', required=True); s1.add_argument('--out', required=True)
    s2 = sub.add_parser('compare'); s2.add_argument('--runs', nargs=4, required=True)
    s2.add_argument('--labels', nargs=4, required=True); s2.add_argument('--out', required=True)
    for s in (s1, s2):
        s.add_argument('--every', type=int, default=2, help='simulation frames per video frame (2 = 2.5x real time at 25 fps)')
        s.add_argument('--fps', type=int, default=25); s.add_argument('--procs', type=int, default=14)
        s.add_argument('--max-frames', type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / (out.stem + '_png')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    if a.cmd == 'single':
        n_frames = len(np.load(Path(a.run) / 'trajectory.npz')['pose'])
    else:
        n_frames = max(len(np.load(Path(r) / 'trajectory.npz')['pose']) for r in a.runs)
    sim = list(range(0, n_frames, a.every))
    if sim[-1] != n_frames - 1:
        sim.append(n_frames - 1)
    sim += [n_frames - 1] * int(round(HOLD_S * a.fps))          # hold the final state
    if a.max_frames:
        sim = sim[:a.max_frames]
    items = list(enumerate(sim))
    chunks = [items[i::a.procs] for i in range(a.procs)]
    if a.cmd == 'single':
        jobs = [(a.run, c, tmp) for c in chunks if c]; fn = render_single_chunk
    else:
        jobs = [(a.runs, a.labels, c, tmp, (n_frames - 1) * DT) for c in chunks if c]; fn = render_compare_chunk
    with Pool(len(jobs)) as pool:
        total = sum(pool.map(fn, jobs))
    encode(tmp, out, a.fps)
    print('wrote', out, total, 'frames')


if __name__ == '__main__':
    main()
