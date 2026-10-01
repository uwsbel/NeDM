"""M1: live depth-camera navigation: one Chrono drive through a mission's waypoints, planning from one overhead depth
frame per decision with the vehicle in it. A port of nav_runner.py, nav_online.py, sensor_map_v2.py,
vehicle_corridor.py and gen_planner.GridRiskModel at 901d6c9 (test_nav.py, B9). There is no RGB camera (R, G, B are 0,
NaN where empty), so models reading colour are refused (nav_runner.py:125-129). One rng per mission, md5(mission + mode
+ str(period)), is drawn across all its decisions; the original patched gen_planner's bound and reversal check at
import, here they are arguments of routes.validate.
  NavHook    decisions after frame k's physics, at frame t = k + 1: the first after the settle (none: no_route_leg0);
             then per frame: a delay-charged route due at t; the stop rules (leg clock per waypoint, mission 700 s); a
             reroute when parked at the route end > 20 frames after the last decision (none: no_route_reroute); on a
             waypoint the next leg's decision (none: no_route) or mission_complete; periodic decisions skipped within
             6 m of the waypoint (a no-route keeps the route). A route change builds a follower WITH Initialize() on
             sensed path heights (none valid: chassis z - 0.75). Delay is charged only by replaying a recorded
             decisions.json (ceil(latency / 0.05 - 1e-9) frames); a replay miss raises (the original measured anew).
"""

from __future__ import annotations

import json
import math
import time
from functools import partial

import numpy as np
import torch

from .config import DT
from .planner import Ensemble, layout_pose, record_numerics
from .routes import (CORRIDOR_HALF_M, GOAL_RADIUS_M, HALF_LENGTH_M, HALF_WIDTH_M, N_LATERAL as NL, N_STATION as NS,
                     arc_line, base_route, constant_speed, corridor_grid, draw, family_anchors, from_params, geom5,
                     validate, widened_bound)

CAMERA = dict(width=1024, height=1024, hfov_rad=math.radians(47.), cam_height_m=110., max_depth_m=180.)
MPP, HALF, N = 80.0 / 512, 40.0, 512                    # the sensed grid (sensor_map_v2)
CHANNELS = ('z_rel', 'grade', 'cross', 'speed', 'valid', 'range_abs', 'range_rel', 'sec1', 'R', 'G', 'B', 'cover1')
MARGIN_M, PLAN_HALF_M, REVERSAL_DEG, RESCUE_MPS, N_CAND = 1.5, 37.0, 45.0, 2.0, 256
MIN_REPLAN_M, MISSION_S = 6.0, 700.0


# ------------------------------------------------------------------------------------------------ sensing and grid
class Camera:
    """The overhead depth camera (scene.build_scene), attached after the scene is built and before any follower."""

    def __init__(self, sim, cam=CAMERA):
        import pychrono as chrono
        import pychrono.sensor as sens
        self.manager, self.taken = sens.ChSensorManager(sim.system), 0
        pose = chrono.ChFramed(chrono.ChVector3d(0., 0., cam['cam_height_m']),
                               chrono.QuatFromAngleZ(math.pi / 2.) * chrono.QuatFromAngleY(math.pi / 2.))
        self.cam = sens.ChDepthCamera(sim.terrain.GetPatches()[0].GetGroundBody(), 1. / sim.dt, pose, cam['width'],
                                      cam['height'], cam['hfov_rad'], cam['max_depth_m'])
        self.cam.SetName('overhead_depth')
        self.cam.SetLag(0.)
        self.cam.SetCollectionWindow(0.)
        self.manager.AddSensor(self.cam)

    def render(self, timeout_s=10.):
        """One render (manager.Update) -> depth (H, W) f32, row 0 = +y (scene._Tap: exactly the next launch)."""
        self.manager.Update()
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            b = self.cam.GetMostRecentDepthBuffer()
            if b.HasData() and b.LaunchedCount > self.taken:
                if b.LaunchedCount != self.taken + 1:
                    raise RuntimeError(f'depth frames skipped: launch {b.LaunchedCount}, expected {self.taken + 1}')
                self.taken = b.LaunchedCount
                d = np.asarray(b.GetDepthData(), dtype=np.float32)
                d = d.reshape(b.Height, b.Width) if d.ndim == 1 else d.squeeze()
                return np.ascontiguousarray(d[::-1, :])
            time.sleep(.0005)
        raise RuntimeError(f'depth frame of launch {self.taken + 1} never arrived')


def grid_from_depth(depth, cam=CAMERA) -> dict:
    """sensor_map_v2.grid_from_arrays, blank RGB image: z, range_m, sec, rgb (3, N, N) f32 and cover (pixels) f32."""
    depth = np.asarray(depth, np.float64)
    h, w = depth.shape
    H, f = float(cam['cam_height_m']), (w / 2) / math.tan(float(cam['hfov_rad']) / 2)
    v, u = np.mgrid[0:h, 0:w]
    ray_x, ray_y = (u - (w - 1) / 2) / f, -(v - (h - 1) / 2) / f             # image row 0 is +y
    sec = np.sqrt(1 + ray_x ** 2 + ray_y ** 2)
    ok = np.isfinite(depth) & (depth > 0) & (depth < float(cam['max_depth_m']) - 1e-6)
    axial = np.where(ok, depth / sec, np.nan)
    x, y, z = ray_x * axial, ray_y * axial, H - axial
    inside = ok & (np.abs(x) < HALF) & (np.abs(y) < HALF)
    flat = (np.clip(((y[inside] + HALF) / MPP).astype(int), 0, N - 1) * N
            + np.clip(((x[inside] + HALF) / MPP).astype(int), 0, N - 1))
    cnt = np.bincount(flat, minlength=N * N).astype(np.float32)
    with np.errstate(invalid='ignore', divide='ignore'):
        g = {k: (np.bincount(flat, weights=q[inside], minlength=N * N) / cnt).reshape(N, N).astype(np.float32)
             for k, q in (('z', z), ('range_m', depth), ('sec', sec))}
        g['rgb'] = np.repeat((np.zeros(N * N) / cnt).reshape(1, N, N), 3, 0).astype(np.float32)
    g['cover'] = cnt.reshape(N, N)
    return g


def sample(img, x, y):
    """Bilinear sample of grid image(s) at world (x, y), cell centres at -40 + (i + 0.5) 80/512 (sensor_dataset_v2)."""
    fx, fy = (x + HALF) / MPP - 0.5, (y + HALF) / MPP - 0.5
    i0, j0 = np.clip(np.floor(fy).astype(int), 0, N - 2), np.clip(np.floor(fx).astype(int), 0, N - 2)
    ty, tx = np.clip(fy - i0, 0, 1), np.clip(fx - j0, 0, 1)
    a, b, c, d = img[..., i0, j0], img[..., i0, j0 + 1], img[..., i0 + 1, j0], img[..., i0 + 1, j0 + 1]
    return a * (1 - ty) * (1 - tx) + b * (1 - ty) * tx + c * ty * (1 - tx) + d * ty * tx


def footprint(pose, margin_m=MARGIN_M):
    """Grid cells under the vehicle (half extents 2.6 x 1.3 m + margin) at the measured pose (vehicle_corridor)."""
    xs = -HALF + (np.arange(N) + 0.5) * MPP
    X, Y = np.meshgrid(xs, xs)
    dx, dy = X - float(pose[0]), Y - float(pose[1])
    c, s = np.cos(float(pose[2])), np.sin(float(pose[2]))
    return (np.abs(dx * c + dy * s) <= HALF_LENGTH_M + margin_m) & (np.abs(-dx * s + dy * c) <= HALF_WIDTH_M + margin_m)


def corridors12(cands, g, exclude=None):
    """nav_online.corridors12_batch: X (n, 12, 96, 32) f32 in CHANNELS order, lengths (n,) f32, invalid fractions.
    Valid: inside +-(40 - 1e-6) m, bilinear cover > 0.999 (`exclude` zeroed) and finite z; references from the first
    station with >= 4 valid samples; range 110 and secant 1 where invalid; no float16 rounding (unlike planner.Scorer)."""
    n, (GX, GY, SP, LEN) = len(cands), corridor_grid(cands)
    DS = np.maximum(LEN / (NS - 1), 1e-3)
    cover = g['cover'] if exclude is None else np.where(exclude, 0.0, g['cover'])
    inside = (np.abs(GX) < HALF - 1e-6) & (np.abs(GY) < HALF - 1e-6)
    cov, zs, rng = sample(cover, GX, GY), sample(g['z'], GX, GY), sample(g['range_m'], GX, GY)
    valid = inside & (cov > 0.999) & np.isfinite(zs)
    has, rows = valid.sum(2) >= 4, np.arange(n)
    ref = np.where(has.any(1), has.argmax(1), -1)

    def reference(field):                   # mean over the reference station's valid samples, else over all valid
        vr, fr = valid[rows, ref], field[rows, ref]
        with np.errstate(invalid='ignore'):
            out = np.where(vr.any(1), np.nansum(np.where(vr, fr, 0.), 1) / np.maximum(vr.sum(1), 1), np.nan)
        whole = np.nansum(np.where(valid, field, 0.), (1, 2)) / np.maximum(valid.sum((1, 2)), 1)
        return np.where(ref >= 0, out, np.where(valid.any((1, 2)), whole, np.nan))

    z0, r0 = reference(zs), reference(rng)
    z0, r0 = np.where(np.isfinite(z0), z0, 0.0), np.where(np.isfinite(r0), r0, CAMERA['cam_height_m'])
    zf = np.where(valid, zs, z0[:, None, None])
    ga = np.clip(np.gradient(zf, axis=1) / DS[:, None, None], -2, 2)
    gc = np.clip(np.gradient(zf, 2 * CORRIDOR_HALF_M / (NL - 1), axis=2), -2, 2)
    rngm = np.where(valid, rng, CAMERA['cam_height_m'])
    sec, rgb = np.where(valid, sample(g['sec'], GX, GY), 1.0), sample(g['rgb'], GX, GY)
    X = np.stack([zf - z0[:, None, None], ga, gc, np.repeat(SP[:, :, None], NL, 2), valid.astype(float), rngm,
                  rngm - r0[:, None, None], sec - 1.0, rgb[0], rgb[1], rgb[2], np.clip(cov, 0, 8) / 8.0], 1)
    return X.astype(np.float32), LEN.astype(np.float32), 1.0 - valid.mean((1, 2))


def path_heights(xy, g, fallback_z):
    """Sensed ground under each waypoint: valid cells (finite z, cover > 0.999) interpolated along the route by index,
    `fallback_z` everywhere when none is valid (nav_online.path_heights)."""
    xy = np.asarray(xy, float)
    z, cov = sample(g['z'], xy[:, 0], xy[:, 1]), sample(g['cover'], xy[:, 0], xy[:, 1])
    ok = np.isfinite(z) & (cov > 0.999)
    if not ok.any():
        return np.full(len(xy), float(fallback_z))
    idx = np.arange(len(xy))
    return np.interp(idx, idx[ok], z[ok])


# ---------------------------------------------------------------------------------------- candidates and rescue
def m1_valid(bound):
    return partial(validate, bound=bound, reversal_deg=REVERSAL_DEG)


def pool(base, pose, rng, valid, fixed=None, n=N_CAND):
    """(routes, tries): fixed None = gen_planner.proposal_pool (the valid anchors, then prior draws, 8n tries); else
    nav_online.fixed_pool (the base at `fixed` m/s with candidate 'fallback_base', unvalidated, then fixed-speed draws,
    6n tries, reported in full). Draws are f104_n2_sampler.sample_one routes ('n2_wide')."""
    st = np.asarray(base['stations'], float)
    L = float(st[-1] - st[0])
    if fixed is None:
        out, cap = [r for r in family_anchors(base) if valid(r, pose)], 8 * n
    else:
        out, cap = [dict(base, speeds=constant_speed(st, fixed), meta=dict(base.get('meta', {}),
                                                                             candidate='fallback_base'))], 6 * n
    tries = 0
    while len(out) < n and tries < cap:
        tries += 1
        r = from_params(base, draw(rng, L, fixed), fixed)
        if valid(r, pose):
            r['meta'] = dict(candidate='n2_wide', max_lateral_m=r['meta']['max_lateral_m'],
                             mean_speed_mps=r['meta']['mean_speed_mps'])
            out.append(r)
    return out, tries if fixed is None else cap


def passes(pose):
    """(bound, aim limit) of the two rescue passes: inside the planning margin, then out to the terrain edge."""
    h = widened_bound(pose)
    return (h, min(h, PLAN_HALF_M) - 0.5), (max(h, 39.5), 39.0)


def fallback_base(pose, goal):
    """A tight arc (radius 8.5, 8.2, 8.05; short way, then long way) to the waypoint, or None."""
    for bound, _ in passes(pose):
        for R in (8.5, 8.2, 8.05):
            for lw in (False, True):
                r = arc_line(pose, goal, R, long_way=lw)
                if r is not None and m1_valid(bound)(r, pose):
                    r['meta'] = dict(r['meta'], candidate='fallback_arc', arc_radius_m=R, long_way=lw,
                                     arena_half_extent_m=bound)
                    return r
    return None


def any_base(pose, goal, bound=PLAN_HALF_M):
    r = base_route(pose, goal, m1_valid(PLAN_HALF_M))
    return r if m1_valid(bound)(r, pose) else fallback_base(pose, goal)


def surrogate_goal(pose, goal):
    """The reachable point (25-10 m ahead, +-180 deg in 10 deg steps) closest to the waypoint, or None."""
    for bound, half in passes(pose):
        best = None
        for d in (25., 20., 15., 12., 10.):
            for dth in np.arange(0., 180. + 1e-9, 10.):
                for sgn in ((1,) if dth in (0., 180.) else (1, -1)):
                    th = pose[2] + math.radians(sgn * dth)
                    q = pose[:2] + d * np.array([math.cos(th), math.sin(th)])
                    if np.max(np.abs(q)) > half or any_base(pose, q, bound) is None:
                        continue
                    if best is None or float(np.linalg.norm(q - goal)) < best[0]:
                        best = float(np.linalg.norm(q - goal)), q
        if best is not None:
            return best[1]
    return None


def straight_ahead(pose):
    fwd = np.array([math.cos(pose[2]), math.sin(pose[2])])
    for bound, half in passes(pose):
        for d in (12., 10., 8.):
            q = pose[:2] + d * fwd
            if np.max(np.abs(q)) <= half and m1_valid(bound)(r := base_route(pose, q, m1_valid(PLAN_HALF_M)), pose):
                r['meta'] = dict(r['meta'], candidate='straight_ahead')
                return q, r
    return None, None


def propose(pose, goal, rng, pick, base=None, fixed=None):
    """One nav_online.Navigator.decide: pool towards `goal` (base: base_route under the widened bound unless given),
    `pick(cands, pose, goal)` -> (index, record); -> (route | None, record)."""
    valid = m1_valid(widened_bound(pose))
    cands, tries = pool(base_route(pose, goal, valid) if base is None else base, pose, rng, valid, fixed)
    rec = dict(aim_xy=[float(goal[0]), float(goal[1])], n_candidates=len(cands), tries=tries,
               plan_half_m=widened_bound(pose))
    if not cands:
        return None, rec
    i, more = pick(cands, pose, goal)
    return cands[i], dict(rec, index=i, **more)


def ladder(pose, goal, rng, pick):
    """nav_runner.decide's ladder: the waypoint, then the rescue rungs at 2 m/s; the record names the rung used."""
    r, rec = propose(pose, goal, rng, pick)
    if r is None and (fb := fallback_base(pose, goal)) is not None:
        r, rec = propose(pose, goal, rng, pick, fb, RESCUE_MPS)
        rec['rung'] = 'fallback_base'
    if r is None and (q := surrogate_goal(pose, goal)) is not None:
        r, rec = propose(pose, q, rng, pick, any_base(pose, q), RESCUE_MPS)
        rec['rung'] = 'surrogate_goal'
    if r is None and (q := straight_ahead(pose))[0] is not None:
        r, rec = propose(pose, q[0], rng, pick, q[1], RESCUE_MPS)
        rec['rung'] = 'straight_ahead'
    return r, rec


# ------------------------------------------------------------------------------------------------------- the hook
class NavHook:
    """The episode loop's hook for a mission (module docstring), recording nav_runner's decisions, routes and legs."""

    def __init__(self, cfg, task, case, env):
        self.goals = [np.asarray(g, float) for g in case['goals']]
        self.radius = float(case.get('goal_radius_m', GOAL_RADIUS_M))
        self.route0 = base_route(layout_pose(case), self.goals[0], m1_valid(PLAN_HALF_M))     # the settle follower's
        self.periodic = cfg.replan != 'waypoint'
        self.period = round((cfg.replan if self.periodic else 2.0) / DT)
        self.rng = np.random.default_rng(cfg.seed(task.id))
        rows = json.loads(env.file(cfg.latency_replay, task).read_text()) if cfg.latency_replay else None
        self.replay = rows and {int(d['frame']): float(d['latency_charged_s']) for d in rows
                                if 'latency_charged_s' in d}
        self.info = dict(camera=CAMERA, seed=cfg.seed(task.id), **self.load_models(cfg, env))
        self.goal_i, self.pending, self.next, self.last, self.grid_t = 0, None, 0, -1000, None
        self.decisions, self.routes, self.legs, self.frame_leg = [], [], [], []      # frame_leg: goal_i per frame

    def load_models(self, cfg, env) -> dict:
        """The direct-depth ensemble on CUDA with torch threads 1 (nav_local_batch.py:37); its provenance."""
        torch.set_num_threads(1)
        self.ens = Ensemble.load(env.glob(cfg.models), env.device)
        chans = {tuple(ck.get('channels') or ()) for _, ck, _ in self.ens.members}
        c = chans.pop() if len(chans) == 1 else ()
        if self.ens.kind != 'legacy' or not c or not set(c) <= set(CHANNELS) - set('RGB'):
            raise ValueError(f'{cfg.models}: M1 needs a legacy ensemble on one set of uncoloured {CHANNELS}')
        self.sel, self.zs = [CHANNELS.index(x) for x in c], self.ens.encode(None, None)
        return dict(channels=list(c), numerics=record_numerics(env.device),
                    models=[dict(path=str(q), sha256=h) for q, h in zip(self.ens.paths, self.ens.sha256)])

    def attach(self, sim):
        self.camera = Camera(sim)

    def route_at(self, pose, goal, t):
        """The decision at frame t: one render per frame, the sensed grid, the ladder scored by the ensemble."""
        if self.grid_t != t:
            self.grid, self.grid_t = grid_from_depth(self.camera.render()), t
        return ladder(pose, goal, self.rng, self.pick)

    def pick(self, cands, pose, goal):
        X, L, inval = corridors12(cands, self.grid, footprint(pose))
        z = self.ens.logits(X[:, self.sel], geom5(pose, goal, L), self.zs).mean(0)
        i = int(np.argmin(z))
        return i, dict(logit=float(z[i]), invalid_fraction=float(inval[i]))

    def decide(self, ep, trigger, t):
        pose, goal = ep.sim.chassis_pose()[0], self.goals[self.goal_i]
        if trigger == 'periodic' and np.linalg.norm(pose[:2] - goal) < MIN_REPLAN_M:
            return None
        t0 = time.perf_counter()
        r, rec = self.route_at(pose, goal, t)
        self.last = t
        self.decisions.append(dict(frame=t, goal_index=self.goal_i, trigger=trigger, pose=pose.tolist(),
                                   v_now=float(ep.sim.vehicle.GetSpeed()), **rec, wall_s=time.perf_counter() - t0))
        if r is not None:
            self.decisions[-1]['route_id'] = len(self.routes)
            self.routes.append(dict({k: np.asarray(r[k]).tolist() for k in ('waypoints', 'speeds', 'stations')},
                                    meta=r['meta'], route_id=len(self.routes), frame=t, goal_index=self.goal_i))
        return r

    def follow(self, ep, r):
        ep.switch(r, initialize=True, z=path_heights(r['waypoints'], self.grid, ep.sim.chassis_pose()[1] - .75))

    def start(self, ep):
        if (r := self.decide(ep, 'leg', 0)) is None:
            return 'no_route_leg0'
        self.follow(ep, r)
        self.legs.append(dict(goal_index=0, start_frame=0))
        self.next = self.period
        return None

    def end_frame(self, ep):
        t, leg = ep.k + 1, self.legs[-1]
        self.frame_leg.append(self.goal_i)
        if self.pending is not None and t >= self.pending[1]:          # a delay-charged route is due
            self.follow(ep, self.pending[0])
            self.pending = None
        s = ep.stops.check(ep)
        if s is None and ep.wp >= len(ep.xy) - 2 and np.linalg.norm(ep.after.pose[:2] - ep.xy[-1]) < 3. \
                and t - self.last > 20 and self.pending is None:       # parked short of the waypoint: plan again
            if (r := self.decide(ep, 'reroute', t)) is None:
                s = 'no_route_reroute'
            else:
                self.follow(ep, r)
        if s is not None:
            leg.update(status=s, elapsed_s=(t - leg['start_frame']) * DT, end_frame=t)
            if s != 'goal_reached':
                return s
            if self.goal_i == len(self.goals) - 1:
                return 'mission_complete'
            self.goal_i += 1
            self.legs.append(dict(goal_index=self.goal_i, start_frame=t))
            if (r := self.decide(ep, 'leg', t)) is None:
                self.legs[-1].update(status='no_route', elapsed_s=0.)
                return 'no_route'
            self.follow(ep, r)
            ep.stops.new_leg(t, self.goals[self.goal_i])
            self.pending, self.next = None, t + self.period
        if self.periodic and self.pending is None and t >= self.next:
            self.next = t + self.period
            if (r := self.decide(ep, 'periodic', t)) is not None:
                if self.replay is not None and t not in self.replay:
                    raise RuntimeError(f'latency replay: no recorded decision at frame {t}')
                n = math.ceil((0. if self.replay is None else self.replay[t]) / DT - 1e-9)
                self.decisions[-1]['latency_charged_s'] = float(n * DT)
                if n <= 0:
                    self.follow(ep, r)
                else:
                    self.pending = (r, t + n)
        return None

    def outcome(self) -> dict:
        """Record fields of the mission (labels.load_drive reads goals_reached and n_goals) and its JSON documents."""
        return dict(goals_reached=sum(g.get('status') == 'goal_reached' for g in self.legs), n_goals=len(self.goals),
                    legs=self.legs, docs={'decisions.json': self.decisions, 'routes.json': self.routes})
