"""Online sensing -> planning for continuous waypoint navigation: one overhead RGB-D frame per decision.

Everything here runs INSIDE the Chrono simulation, once per planning decision, with the vehicle present in the
frame. It is the vehicle-included pilot's preprocessing (sensor_v2/VEHICLE_PILOT.md) turned into a live loop:

    render -> back-project to the metric grid (sensor_map_v2) -> footprint exclusion zone at the MEASURED pose
           -> candidate routes from the measured pose to the active waypoint -> 12-channel corridors from THIS
              frame (samples inside the zone marked invalid, coordinates preserved) -> frozen ensemble -> argmin

Nothing is carried over between decisions: each decision sees one frame. The model, candidate generator and risk
definition are unchanged and the checkpoints are frozen.

Supporting fixes live here because continuous replanning needs them; none of them changes the model, and each was
found by a failure that hit the replanning arms harder than the plan-once arm, i.e. would have decided the
comparison for a reason unrelated to planning (nav_v1/LOG.md has the measurements):
  * SpeedPI - Chrono's ChSpeedController and the path-follower's throttle/brake mapping, reimplemented so the
    speed integrator SURVIVES a route change (rebuilding ChPathFollowerDriver resets it: -0.09 m/s and 1.1 m of
    divergence when rebuilding every 2 s on an UNCHANGED route), with anti-windup, which Chrono's own controller
    does not need because the collector rebuilds the driver every 10-20 s episode and this loop does not.
  * plan_bound - the candidate validation bound: 3 m inside the terrain, widened to the vehicle's own radius when
    tracking error has already carried it past that.
  * the rescue ladder (fallback_base -> surrogate_goal -> straight_ahead), commanded at 2 m/s by the runner
    because the follower cannot hold an 8 m radius at 4 m/s.
  * reachable_speed - OFF by default, and the reason is worth recording. Clipping a candidate's commanded speed
    to what the vehicle can reach from its current speed looks obviously right, and it is a trap: every route
    starts at the vehicle, the path follower reads its speed command at the vehicle's own station, and the cone
    caps that station at exactly the current speed. The command then equals the measured speed at every replan and
    the vehicle can never accelerate (measured: 0.3 m/s after 5 s of 1 Hz replanning). Commanding a speed the
    vehicle is not yet at is also what the training episodes did - they all commanded 2/4/6 m/s from rest - so the
    unclipped profile is the in-distribution one. The flag is kept because the diagnostic is worth repeating.
"""
import dataclasses, glob, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sensor_map_v2 as M
import sensor_dataset_v2 as V2
import vehicle_corridor as VC
import gen_planner as P
import f104_n2_sampler as S

# The vehicle-free arena capture and the vehicle-included pilot both used this camera; kept identical so the
# frozen checkpoints see the frames they were evaluated on.
CAMERA = {'width': 1024, 'height': 1024, 'hfov_rad': math.radians(47.), 'cam_height_m': 110.,
          'depth_ray_scale': 1., 'max_depth_m': 180., 'backend': 'Vulkan_RT_lavapipe',
          'depth_measurement': 'Euclidean_ray_range_m', 'model_image_size': 512, 'elevation_scale_m': 10.,
          'observation_mode': 'one overhead RGB-D frame per planning decision, vehicle present'}
MARGIN_M = 1.5          # footprint buffer; the pilot measured 0.50 m as the worst vehicle+shadow reach
# Routes must stay 3 m inside the terrain. The route validator's own bound is the terrain edge itself (+-40 m),
# which is fine for a single leg starting near the middle but not for a mission that works the whole arena: the
# path follower's tracking error reaches ~3 m at 4-5 m/s on this terrain, so a route that touches 37-38 m puts the
# vehicle over the edge and ends the run. Applied identically to every arm.
PLAN_HALF_M = 37.0
P.CFG = dataclasses.replace(P.CFG, arena_half_extent_m=PLAN_HALF_M)

# Routes that reverse on themselves are rejected. The frozen validator estimates curvature from three consecutive
# points, and a route that runs straight out and straight back along the same line has collinear points at the cusp,
# so it scores exactly 0 and passes. The rescue ladder produced such routes whenever it aimed at a stand-in point
# directly behind the vehicle (the Hermite shape to a point behind folds back after ~0.5 m): 67 of 133 rescue routes
# in the first local run and AMD campaign did, and every one of the 14 AMD arena exits followed one. Across the same
# runs no normal or fallback route ever turned more than 6.6 deg between consecutive points, so 45 deg only catches
# reversals. Patched into gen_planner so the proposal sampler and the frozen route builder use it too.
MAX_STEP_TURN_DEG = 45.0
_FROZEN_SAFE_VALIDATE = P.safe_validate


def max_step_turn_deg(route):
    w = np.asarray(route['waypoints'], float)
    d = np.diff(w, axis=0); d = d[np.linalg.norm(d, axis=1) > 1e-6]
    if len(d) < 2:
        return 0.0
    h = np.arctan2(d[:, 1], d[:, 0])
    return float(np.degrees(np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi).max()))


def safe_validate_no_reversal(route, obstacles, cfg, anchor):
    if max_step_turn_deg(route) > MAX_STEP_TURN_DEG:
        return {'valid': False, 'reasons': ['reverses on itself']}
    return _FROZEN_SAFE_VALIDATE(route, obstacles, cfg, anchor)


P.safe_validate = safe_validate_no_reversal
TERRAIN_HALF_M = 40.0


class plan_bound:
    """Candidate validation bound for one decision: the 3 m margin, or the vehicle's own distance from the centre
    if it is already outside it.

    A fixed margin has a nasty failure mode. Every candidate starts at the vehicle, so once tracking error has
    carried the vehicle past the margin, EVERY candidate fails the arena test and the planner is reduced to its
    rescue ladder at exactly the moment it needs a normal route pointing back inside. Widening the bound to the
    vehicle's own radius keeps the full 256-candidate pool available there."""

    def __init__(self, pose):
        # the validator rejects a route whose swept corridor (half width 1.3 m) crosses the bound, so a vehicle at
        # radius r needs a bound of at least r + 1.3 before ANY route from it can validate
        r = float(np.max(np.abs(np.asarray(pose, float)[:2])))
        self.half = min(max(PLAN_HALF_M, r + 2.5), TERRAIN_HALF_M + 5.0)

    def __enter__(self):
        self.saved = P.CFG
        if abs(self.half - P.CFG.arena_half_extent_m) > 1e-9:
            P.CFG = dataclasses.replace(P.CFG, arena_half_extent_m=self.half)
        return self.half

    def __exit__(self, *exc):
        P.CFG = self.saved
        return False
# Rescue shapes are tried twice: first inside the same margin, then out to the terrain edge, because a cornered
# vehicle with no valid shape at all is worse than one driving close to the wall at 2 m/s.
CFG_FALLBACK = P.CFG


def passes_for(pose):
    """Rescue shapes are tried inside the planning margin first, then out to the terrain edge; both bounds are
    widened if the vehicle is already outside them."""
    h = plan_bound(pose).half
    return ((dataclasses.replace(P.CFG, arena_half_extent_m=h), min(h, PLAN_HALF_M) - 0.5),
            (dataclasses.replace(P.CFG, arena_half_extent_m=max(h, 39.5)), 39.0))


class SpeedPI:
    """ChSpeedController (trapezoidal integral, backward-difference derivative) + the path-follower's
    throttle/brake mapping, with the integral carried across route changes and anti-windup.

    Chrono's own controller has no anti-windup; it does not need one, because the collector rebuilds the driver
    for every episode and an episode is 10-20 s long. Carrying the integral across a 90 s mission is a different
    regime: with Ki = 0.05 the integral only has to reach 20 m/s.s to saturate the command on its own, and it does,
    on any long stretch where the vehicle is slower than commanded. The observed failure was full throttle at
    9-10.8 m/s on a downhill against a 6 m/s command, 7 m off the route, ending outside the arena. Conditional
    integration (stop integrating in the direction that is already saturated) plus a clamp at +-1/Ki fixes it and
    leaves the controller identical to Chrono's whenever the command is inside [-1, 1]."""

    def __init__(self, kp=.6, ki=.05, kd=0., throttle_threshold=.2):
        self.kp, self.ki, self.kd, self.tt = kp, ki, kd, throttle_threshold
        self.err = 0.; self.erri = 0.; self.throttle = 0.; self.braking = 0.
        self.saturated_steps = 0

    def reset(self):
        self.err = 0.; self.erri = 0.; self.throttle = 0.; self.braking = 0.

    def advance(self, speed, target, step):
        err = float(target) - float(speed)
        errd = (err - self.err) / step
        erri = self.erri + (err + self.err) * step / 2
        raw = self.kp * err + self.ki * erri + self.kd * errd
        if (raw > 1. and err > 0.) or (raw < -1. and err < 0.):
            erri = self.erri                                   # conditional integration
            raw = self.kp * err + self.ki * erri + self.kd * errd
            self.saturated_steps += 1
        if self.ki:
            erri = float(np.clip(erri, -1. / self.ki, 1. / self.ki))
        self.erri, self.err = erri, err
        out = float(np.clip(raw, -1., 1.))
        if out > 0:
            self.braking, self.throttle = 0., out
        elif self.throttle > self.tt:
            self.braking, self.throttle = 0., 1. + out
        else:
            self.braking, self.throttle = -out, 0.
        return self.throttle, self.braking


def reachable_speed(route, v_now, a_acc=S.A_ACC):
    """Clip a candidate's commanded speed to what is reachable from the current speed (in place, returns the
    fraction of stations the clip touched)."""
    st = np.asarray(route['stations'], float); v = np.asarray(route['speeds'], float).copy()
    cap = np.sqrt(np.maximum(float(v_now) ** 2 + 2 * a_acc * (st - st[0]), 0.))
    hit = float((v > cap + 1e-9).mean())
    route['speeds'] = np.minimum(v, cap)
    return hit


def path_heights(xy, fallback_z):
    """Sensed ground height along a route, from the CURRENT frame's grid; points with no valid cell take the
    nearest valid height along the route, and a route with no valid cell at all takes fallback_z (the vehicle's
    own measured contact height). Replaces the heightmap lookup the frozen driver builder used."""
    xy = np.asarray(xy, float)
    z = V2.sample(V2.G['z'], xy[:, 0], xy[:, 1])
    cov = V2.sample(V2.G['cover'].astype(np.float32), xy[:, 0], xy[:, 1])
    ok = np.isfinite(z) & (cov > 0.999)
    if not ok.any():
        return np.full(len(xy), float(fallback_z)), 0.0
    idx = np.arange(len(xy))
    z = np.interp(idx, idx[ok], z[ok])
    return z, float(ok.mean())


def fallback_base(pose, goal, radii=(8.5, 8.2, 8.05)):
    """A valid route shape to the SAME waypoint when the frozen builder has none.

    `gen_planner.base_route` tries the frozen Hermite, wider/tighter Hermite starts, tight arcs of radius 12/10/9
    the short way round and 12/10/9/8.5 the long way. Near the arena edge a right-angle-plus turn needs the short
    way at a radius the builder never tries: at one measured waypoint every listed shape failed on `arena` and
    radius 8.5 short-way was valid (curvature 0.1176 /m, still inside the 0.125 /m limit). This fills that gap
    without touching the frozen builder that earlier results used.
    """
    pose = np.asarray(pose, float)
    for cfg, _ in passes_for(pose):
        for R in radii:
            for lw in (False, True):
                r = P._arc_line(pose, goal, R, long_way=lw)
                if r is not None and P.safe_validate(r, [], cfg, pose)['valid']:
                    r['meta'] = dict(r.get('meta', {}), candidate='fallback_arc', arc_radius_m=R, long_way=lw,
                                     arena_half_extent_m=cfg.arena_half_extent_m)
                    return r
    return None


def any_base(pose, goal, cfg=None):
    """The frozen builder's route if it validates, else the tight-arc fallback, else None."""
    pose = np.asarray(pose, float)
    r = P.base_route(pose, goal)
    if P.safe_validate(r, [], cfg or P.CFG, pose)['valid']:
        return r
    return fallback_base(pose, goal)


def straight_ahead(pose, distances=(12., 10., 8.)):
    """Last resort: drive forward. A vehicle that cannot turn towards anything can still go straight, and one
    replanning period later it is somewhere else."""
    pose = np.asarray(pose, float)
    fwd = np.array([math.cos(pose[2]), math.sin(pose[2])])
    for cfg, half in passes_for(pose):
        for d in distances:
            q = pose[:2] + d * fwd
            if np.max(np.abs(q)) > half:
                continue
            r = P.base_route(pose, q)
            if P.safe_validate(r, [], cfg, pose)['valid']:
                r['meta'] = dict(r.get('meta', {}), candidate='straight_ahead')
                return q, r
    return None, None


def surrogate_goal(pose, goal, radii=(25., 20., 15., 12., 10.), max_turn_deg=180., step_deg=10.):
    """A reachable stand-in for a waypoint the vehicle cannot turn towards.

    The route validator enforces a 0.125 /m curvature limit and the arena bounds. A vehicle that arrives at a
    waypoint pointing ~110 deg away from the next one, near the arena edge, has no valid route to it at all: the
    frozen Hermite shape, the wider/tighter starts and the tight-arc fallbacks are all rejected, and the planner
    returns nothing. Rather than end the mission on a geometry technicality, the runner aims at the closest
    reachable point to the real waypoint and replans from there; the waypoint itself is unchanged, and reaching it
    is still what counts.
    """
    pose = np.asarray(pose, float); goal = np.asarray(goal, float)
    for cfg, half in passes_for(pose):
        best = None
        for d in radii:
            for dth in np.arange(0., max_turn_deg + 1e-9, step_deg):
                for sgn in ((1,) if dth in (0., 180.) else (1, -1)):
                    th = pose[2] + math.radians(sgn * dth)
                    q = pose[:2] + d * np.array([math.cos(th), math.sin(th)])
                    if np.max(np.abs(q)) > half:
                        continue
                    if any_base(pose, q, cfg) is None:
                        continue
                    score = float(np.linalg.norm(q - goal))
                    if best is None or score < best[0]:
                        best = (score, q)
        if best is not None:
            return best[1]
    return None


def trim_route(route, pose, min_len_m=8.0):
    """The part of a route still ahead of the vehicle, re-stationed, so the route the vehicle is already on can be
    scored against fresh candidates instead of being discarded by construction."""
    wp = np.asarray(route['waypoints'], float); sp = np.asarray(route['speeds'], float)
    st = np.asarray(route['stations'], float)
    i = int(np.argmin(np.linalg.norm(wp - np.asarray(pose[:2])[None], axis=1)))
    if len(wp) - i < 4 or st[-1] - st[i] < min_len_m:
        return None
    return {'waypoints': wp[i:], 'speeds': sp[i:], 'stations': st[i:] - st[i],
            'headings': np.asarray(route.get('headings', np.zeros(len(wp))), float)[i:],
            'meta': {'candidate': 'current'}}


class Navigator:
    """Frozen ensemble + the live corridor pipeline. One instance per simulation."""

    def __init__(self, pattern, margin_m=MARGIN_M, n_cand=P.N_CAND, device='cpu', mask=True,
                 reachable=False, camera=None, sense_radius_m=0.0):
        self.model = P.GridRiskModel(pattern, device=device)
        self.margin_m, self.n_cand, self.mask, self.reachable = margin_m, n_cand, mask, reachable
        self.sense_radius_m = float(sense_radius_m)
        self.camera = dict(camera or CAMERA)
        self.grid = None
        self.channels = self.model.channels

    # ---- sensing -------------------------------------------------------------------------------------------
    def observe(self, rgb, depth, pose=None):
        t0 = time.perf_counter()
        self.grid = M.grid_from_arrays(depth, rgb, self.camera)
        if self.sense_radius_m > 0 and pose is not None:
            # stand-in for a limited-range sensor: keep only what lies within sense_radius_m of the vehicle, and
            # mark the rest exactly as unobserved ground (cover 0), not as flat or safe
            n, mpp, half = self.grid['meta']['n'], self.grid['meta']['mpp'], self.grid['meta']['half_extent_m']
            xs = -half + (np.arange(n) + 0.5) * mpp
            X, Y = np.meshgrid(xs, xs)
            far = (X - pose[0]) ** 2 + (Y - pose[1]) ** 2 > self.sense_radius_m ** 2
            self.grid['cover'] = np.where(far, 0., self.grid['cover']).astype(np.float32)
            for k in ('z', 'range_m', 'sec'):
                self.grid[k] = np.where(far, np.nan, self.grid[k]).astype(np.float32)
        V2.set_grid(self.grid)
        cov = self.grid['cover']
        return {'backproject_s': time.perf_counter() - t0, 'coverage_fraction': float((cov > 0).mean()),
                'valid_depth_fraction': float(np.mean(np.isfinite(depth) & (depth > 0)
                                                      & (depth < self.camera['max_depth_m'] - 1e-6)))}

    # ---- planning ------------------------------------------------------------------------------------------
    def fixed_pool(self, base, pose, rng, speed):
        """Candidates at a constant commanded speed around a given shape, plus the shape itself.

        Used only for the fallback shapes. A tight rescue arc (8 m radius, the curvature limit) validates as a
        route but the path follower cannot hold it at 4 m/s: the steering saturates, the vehicle runs 6 m wide and
        off the terrain. Slowing the command to 2 m/s makes the same arc trackable. Nothing on the normal path is
        affected."""
        out = [dict(base, speeds=P.constant_speed(np.asarray(base['stations'], float), speed),
                    meta=dict(base.get('meta', {}), candidate='fallback_base'))]
        tries = 6 * self.n_cand
        for _ in range(tries):
            if len(out) >= self.n_cand:
                break
            r = S.sample_one(base, rng, lat_sigma=5.0, sp_sigma=0.0, base_speed=speed)
            if P.safe_validate(r, [], P.CFG, np.asarray(pose, float))['valid']:
                out.append(r)
        return out, tries

    def decide(self, pose, goal, rng, v_now=0.0, fixed_speed=None, current=None, switch_margin=0.0, base=None,
               pick='model'):
        pose = np.asarray(pose, float); goal = np.asarray(goal, float)
        t = {}
        t0 = time.perf_counter()
        with plan_bound(pose) as half:
            base = P.base_route(pose, goal) if base is None else base
            if fixed_speed is None:
                cands, tries = P.proposal_pool(base, pose, rng, n=self.n_cand)
            else:
                cands, tries = self.fixed_pool(base, pose, rng, fixed_speed)
        t['propose_s'] = time.perf_counter() - t0
        info = {'n_candidates': len(cands), 'tries': tries, 'plan_half_m': half,
                'base_start_tangent_scale': base['meta'].get('start_tangent_scale'),
                'base_arc_radius_m': base['meta'].get('arc_radius_m'),
                'base_kind': base['meta'].get('candidate', 'frozen')}
        cur_idx = None
        if current is not None:
            trimmed = trim_route(current, pose)
            if trimmed is not None:
                cur_idx = len(cands); cands = list(cands) + [trimmed]
        info['current_in_pool'] = cur_idx is not None
        if not cands:
            return None, {**info, 'reason': 'no valid candidate', 'timing': t}
        if self.reachable:
            t0 = time.perf_counter()
            hits = [reachable_speed(r, v_now) for r in cands]
            t['speed_clip_s'] = time.perf_counter() - t0
            info['speed_clip_fraction'] = float(np.mean(hits))
        t0 = time.perf_counter()
        exclude = VC.exclusion_mask(pose, self.margin_m) if self.mask else None
        t['mask_s'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        if self.mask:
            X, L, inval = corridors12_batch(cands, exclude)
        else:
            X, L, inval = [], [], []
            for r in cands:
                x, l = V2.tensor12(np.asarray(r['waypoints']), np.asarray(r['speeds']), np.asarray(r['stations']))
                inval.append(float(1 - x[4].mean())); X.append(x); L.append(l)
            X = np.stack(X).astype(np.float32); L = np.asarray(L, np.float32)
        t['corridor_s'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        z, pr = self.model.score(X, P.geom_ctx(pose[:2], goal, pose[2], L))
        t['score_s'] = time.perf_counter() - t0
        i = int(rng.integers(len(cands))) if pick == 'random' else int(np.argmin(z))
        kept = False
        if cur_idx is not None and z[i] > z[cur_idx] - switch_margin:
            i, kept = cur_idx, True
        info['kept_current'] = kept
        if cur_idx is not None:
            info['current_logit'] = float(z[cur_idx]); info['best_logit'] = float(z.min())
        r = cands[i]
        info.update(index=i, logit=float(z[i]), risk=float(pr[i]), risk_median=float(np.median(pr)),
                    invalid_fraction=float(np.mean(inval)), invalid_fraction_picked=float(inval[i]),
                    mean_speed=float(np.asarray(r['speeds'])[1:-1].mean()),
                    length_m=float(np.asarray(r['stations'])[-1]), timing=t,
                    plan_s=float(sum(t.values())))
        return r, info


def mask_leak(pose, margin_m, tmap, radius_m=8.0, threshold_m=0.6):
    """Evaluation-only check that the exclusion zone still covers the vehicle in the CURRENT frame: how far
    outside the zone does a cell sit whose sensed height is more than `threshold_m` above the simulator's terrain?
    Uses the privileged heightmap purely as a reference; nothing here feeds the planner."""
    n, mpp, half = V2.G['n'], V2.G['mpp'], V2.G['half']
    xs = -half + (np.arange(n) + 0.5) * mpp
    X, Y = np.meshgrid(xs, xs)
    d2 = (X - pose[0]) ** 2 + (Y - pose[1]) ** 2
    near = (d2 <= radius_m ** 2) & (V2.G['cover'] > 0) & np.isfinite(V2.G['z'])
    if not near.any():
        return {'cells': 0, 'max_height_m': 0.0, 'max_distance_m': 0.0}
    zx, zy = X[near], Y[near]
    ref = tmap.height(zx, zy)
    above = V2.G['z'][near] - ref
    ex = VC.exclusion_mask(pose, margin_m)[near]
    bad = (~ex) & (above > threshold_m)
    if not bad.any():
        return {'cells': 0, 'max_height_m': float(above[~ex].max()) if (~ex).any() else 0.0,
                'max_distance_m': 0.0}
    c, s = np.cos(pose[2]), np.sin(pose[2])
    dx, dy = zx[bad] - pose[0], zy[bad] - pose[1]
    along = np.abs(dx * c + dy * s) - VC.HALF_LENGTH_M
    across = np.abs(-dx * s + dy * c) - VC.HALF_WIDTH_M_VEH
    dist = np.maximum(np.maximum(along, 0), np.maximum(across, 0))
    return {'cells': int(bad.sum()), 'max_height_m': float(above[bad].max()),
            'max_distance_m': float(dist.max())}


def corridors12_batch(cands, exclude=None):
    """All 256 candidates' 12-channel corridors in one pass over the grid.

    Identical output to vehicle_corridor.tensor12_excluded applied candidate by candidate (checked to 0 on stored
    frames); it exists only because the per-candidate Python loop is a third of the planner's latency, and latency
    is one of this milestone's reported numbers.
    """
    ns, nl = V2.N_STATION, V2.N_LATERAL
    n = len(cands)
    GX = np.empty((n, ns, nl)); GY = np.empty((n, ns, nl))
    SP = np.empty((n, ns)); LEN = np.empty(n); DS = np.empty(n)
    for i, r in enumerate(cands):
        wp = np.asarray(r['waypoints'], float); sp = np.asarray(r['speeds'], float)
        gx, gy, grid = V2.corridor_points(wp, np.asarray(r['stations'], float))
        GX[i], GY[i] = gx, gy
        s_ref = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
        SP[i] = np.interp(np.linspace(0, s_ref[-1], ns), s_ref, sp)
        LEN[i] = float(grid[-1] - grid[0])
        DS[i] = max(LEN[i] / (ns - 1), 1e-3)
    cover = V2.G['cover'].astype(np.float32)
    if exclude is not None:
        cover = np.where(exclude, 0.0, cover)
    inside = (np.abs(GX) < V2.G['half'] - 1e-6) & (np.abs(GY) < V2.G['half'] - 1e-6)
    cov = V2.sample(cover, GX, GY)
    zs = V2.sample(V2.G['z'], GX, GY)
    valid = inside & (cov > 0.999) & np.isfinite(zs)
    cnt = valid.sum(2)
    has = cnt >= VC.MIN_VALID_PER_STATION
    ref = np.where(has.any(1), has.argmax(1), -1)
    rows = np.arange(n)
    rng_g = V2.sample(V2.G['range_m'], GX, GY)

    def reference(field):
        out = np.empty(n)
        vr = valid[rows, ref]; fr = field[rows, ref]
        with np.errstate(invalid='ignore'):
            out = np.where(vr.any(1), np.nansum(np.where(vr, fr, 0.), 1) / np.maximum(vr.sum(1), 1), np.nan)
        anyv = valid.any((1, 2))
        allmean = np.where(anyv, np.nansum(np.where(valid, field, 0.), (1, 2)) / np.maximum(valid.sum((1, 2)), 1), np.nan)
        return np.where(ref >= 0, out, allmean)

    z0 = reference(zs); r0 = reference(rng_g)
    z0 = np.where(np.isfinite(z0), z0, 0.0)
    r0 = np.where(np.isfinite(r0), r0, V2.G['cam_h'])
    zf = np.where(valid, zs, z0[:, None, None])
    ga = np.clip(np.gradient(zf, axis=1) / DS[:, None, None], -2, 2)
    gc = np.clip(np.gradient(zf, 2 * V2.HALF_WIDTH_M / (nl - 1), axis=2), -2, 2)
    sec = V2.sample(V2.G['sec'], GX, GY)
    rngm = np.where(valid, rng_g, V2.G['cam_h']); sec = np.where(valid, sec, 1.0)
    rgb = V2.sample(V2.G['rgb'], GX, GY)
    X = np.stack([zf - z0[:, None, None], ga, gc, np.repeat(SP[:, :, None], nl, 2), valid.astype(float),
                  rngm, rngm - r0[:, None, None], sec - 1.0, rgb[0], rgb[1], rgb[2],
                  np.clip(cov, 0, 8) / 8.0], 1).astype(np.float32)
    return X, LEN.astype(np.float32), 1.0 - valid.mean((1, 2))
