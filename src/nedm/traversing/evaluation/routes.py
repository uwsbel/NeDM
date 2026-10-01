"""Candidate routes of the traversing planners (numpy only): route family, validator, base and straight routes, the
static planner map with batched corridors, the route context, the decision history window and the route hash.

A port of the experiment branch at 901d6c9, bitwise equal to it (test_routes.py; goldens written by the original code):
  family      theta = (a1..a3 | dv1..dv4). Lateral offset sum_j a_j sin(j pi f), a_j capped at 0.55*0.125*L^2/(j pi)^2,
              clipped at +-10 m; free-end smoothstep speed knots dv_k (+-4 m/s) on the base speeds; then speeds clipped
              to [0.5, 6], the terminal cone sqrt(2*2*(L - s)) and the 1.5 m/s^2 forward / 2.0 m/s^2 backward passes
              (f104_n2_sampler.py:35-93, f104_n2_iter.py:20-84). A fixed-speed family (fixed2) has theta in R^3; its
              prior draws still consume 4 speed normals (sd 0).
  validator   gen_planner.safe_validate with gen_planner.CFG (fdm_mppi.py:54-168). M1's arena bound (37 m, widened)
              and its reversal rejection (nav_online.py:55-111) are arguments here, never module state.
  base_route  gen_planner.base_route (gen_planner.py:45-121); it equals route_00 on all 3,250 released cases.
  StaticMap   static_map_v1 + f104_n2_iter.corridors_batch; the scorer rounds corridors to float16, not this module.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nedm.traversing.training.state import ACT_DIM, OBSERVABLE_COLS, Z1_DIM

LAT_SIGMA, LAT_CLIP, SP_SIGMA, SP_CLIP = 5.0, 10.0, 1.5, 4.0
MODES, KNOTS, KAPPA_MAX, BUDGET = 3, 4, 0.125, 0.55
PRIOR_SD = np.array([LAT_SIGMA / j for j in range(1, MODES + 1)] + [SP_SIGMA] * KNOTS)   # (5, 2.5, 5/3, 1.5 x 4)
A_ACC, A_DEC, V_MIN, V_MAX = 1.5, 2.0, 0.5, 6.0
ANCHOR_OFFSETS, ANCHOR_SPEEDS = (0.0, -4.0, 4.0), (2.0, 4.0, 6.0)
ARENA_HALF_M, SPEED_MAX, HALF_LENGTH_M, HALF_WIDTH_M, MARGIN_M, PATH_STEP_M = 40.0, 6.0, 2.6, 1.3, 0.1, 0.25   # CFG
N_STATION, N_LATERAL, CORRIDOR_HALF_M, HIST_T = 96, 32, 6.0, 40
HIST_DIM = len(OBSERVABLE_COLS) + ACT_DIM               # 15 = 12 observable state columns + the 3 actions
KEYS = ('waypoints', 'speeds', 'stations', 'headings')
ENDS_TOL_M = 0.25                                       # route start / end vs case start / goal


# ------------------------------------------------------------------------------------------------------- validator
def _curvature_max(p):                                  # planner_s._curvature_max
    ab, bc, ac = p[1:-1] - p[:-2], p[2:] - p[1:-1], p[2:] - p[:-2]
    cross = np.abs(ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0])
    denom = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ac, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = np.where(denom > 1e-9, 2.0 * cross / denom, 0.0)
    return float(kappa.max()) if len(kappa) else 0.0


def _valid(route, pose, bound):
    """fdm_mppi.validate_reference(route, [], CFG, pose)['valid']; raises ValueError on a degenerate route."""
    xy, speed = np.asarray(route['waypoints'], np.float64), np.asarray(route['speeds'], np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or speed.shape != (len(xy),):
        raise ValueError('reference requires >=3 XY waypoints and one speed per waypoint')
    if not np.isfinite(xy).all() or not np.isfinite(speed).all():
        raise ValueError('nonfinite reference')
    station = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
    if np.any(np.diff(station) <= 1e-8):
        raise ValueError('duplicate consecutive waypoints')
    s0 = route.get('meta', {}).get('fdm_station')       # set on an unstripped route_00: validated from that station
    i = (min(max(0, int(np.searchsorted(station, s0)) - 1), len(xy) - 2) if s0 is not None
         else min(int(np.argmin(np.linalg.norm(xy - pose[:2], axis=1))), len(xy) - 2))
    xy, speed, station = xy[i:], speed[i:], station[i:]
    acc = np.diff(speed ** 2) / (2. * np.maximum(np.diff(station), 1e-8))
    if (_curvature_max(xy) > KAPPA_MAX + 1e-6 or speed.min() < 0.0 - 1e-6 or speed.max() > SPEED_MAX + 1e-6
            or acc.max(initial=0.) > A_ACC + 1e-6 or acc.min(initial=0.) < -A_DEC - 1e-6):
        return False
    heading = np.unwrap(np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0])))
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]              # planner_s._resample
    t = np.linspace(0.0, s[-1], max(2, int(round(s[-1] / PATH_STEP_M)) + 1))
    dense = np.stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])], 1)
    heading = np.interp(np.r_[0., np.linalg.norm(np.diff(dense, axis=0), axis=1).cumsum()], station - station[0], heading)
    tangent = np.stack((np.cos(heading), np.sin(heading)), axis=1)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    hl, hw = HALF_LENGTH_M + MARGIN_M, HALF_WIDTH_M + MARGIN_M
    corners = np.concatenate([dense + a * hl * tangent + b * hw * normal for a in (-1., 1.) for b in (-1., 1.)])
    return not (np.abs(corners) > bound).any()


def validate(route, pose, *, bound=ARENA_HALF_M, reversal_deg=None):
    """gen_planner.safe_validate(route, [], CFG, pose)['valid']: a degenerate route (e.g. folded by its lateral offset)
    is rejected, not raised. M1 passes bound=widened_bound(pose) and reversal_deg=45, which first rejects any route
    turning more than that between consecutive segments (nav_online.safe_validate_no_reversal)."""
    pose = np.asarray(pose, np.float64)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError(f'validate: pose must be a finite (x, y, yaw), got {pose}')
    if reversal_deg is not None and max_step_turn_deg(route) > reversal_deg:
        return False
    try:
        return _valid(route, pose, float(bound))
    except ValueError:
        return False


def max_step_turn_deg(route):
    """Largest heading change between consecutive non-degenerate segments (nav_online.py:69-75)."""
    d = np.diff(np.asarray(route['waypoints'], float), axis=0)
    d = d[np.linalg.norm(d, axis=1) > 1e-6]
    if len(d) < 2:
        return 0.0
    h = np.arctan2(d[:, 1], d[:, 0])
    return float(np.degrees(np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi).max()))


def widened_bound(pose, half=37.0, limit=45.0):
    """M1 planning bound (nav_online.plan_bound): `half`, widened to the vehicle's radius + 2.5 m, at most `limit`."""
    return min(max(half, float(np.max(np.abs(np.asarray(pose, float)[:2]))) + 2.5), limit)


# ---------------------------------------------------------------------------------------------------- route family
def _check_theta(theta, fixed_speed):
    th, dim = np.asarray(theta, float), MODES + (0 if fixed_speed is not None else KNOTS)
    if th.shape != (dim,):
        raise ValueError(f'theta shape {th.shape} != ({dim},) for fixed_speed={fixed_speed}')
    return th


def _base_arrays(base):
    xy, station = np.asarray(base['waypoints'], float), np.asarray(base['stations'], float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or station.shape != (len(xy),):
        raise ValueError(f'base route needs (n, 2) waypoints and n stations, got {xy.shape} and {station.shape}')
    L = float(station[-1] - station[0])
    if not L > 0:
        raise ValueError(f'base route length {L} is not positive')
    return xy, np.clip((station - station[0]) / max(station[-1] - station[0], 1e-6), 0, 1), L


def caps(L):
    """Per-mode amplitude caps: the validator's curvature budget."""
    return BUDGET * KAPPA_MAX * L ** 2 / (np.arange(1, MODES + 1) * np.pi) ** 2


def project(theta, L, fixed_speed=None):
    """theta clipped onto the caps (a_j) and +-4 m/s (dv_k)."""
    th = _check_theta(theta, fixed_speed).copy()
    th[:MODES] = np.clip(th[:MODES], -caps(L), caps(L))
    if fixed_speed is None:
        th[MODES:] = np.clip(th[MODES:], -SP_CLIP, SP_CLIP)
    return th


def _shape(xy, speed, lat, dv):
    """f104_n2_sampler.shape: offset the base polyline, then the speed limits in the recorded order."""
    t = np.gradient(xy, axis=0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    pts = xy + lat[:, None] * np.stack([-t[:, 1], t[:, 0]], 1)
    st = np.r_[0.0, np.linalg.norm(np.diff(pts, axis=0), axis=1).cumsum()]
    v = np.clip(speed + dv, V_MIN, V_MAX)
    v = np.minimum(v, np.sqrt(2 * A_DEC * np.maximum(st[-1] - st, 0)))
    for j in range(1, len(v)):                        # Python min/max over numpy scalars, as recorded
        v[j] = min(v[j], np.sqrt(max(v[j-1] ** 2 + 2 * A_ACC * (st[j] - st[j-1]), 0)))
    for j in range(len(v) - 2, -1, -1):
        v[j] = min(v[j], np.sqrt(max(v[j+1] ** 2 + 2 * A_DEC * (st[j+1] - st[j]), 0)))
    return {'waypoints': pts, 'speeds': v, 'stations': st,
            'headings': np.arctan2(np.gradient(pts[:, 1]), np.gradient(pts[:, 0])), 'meta': {}}


def _speed_knots(f, vals):
    """f104_n2_sampler.speed_knots: smoothstep between free-end knots at f = 0, 1/3, 2/3, 1."""
    kx = np.linspace(0.0, 1.0, KNOTS)
    seg = np.clip(np.searchsorted(kx, f, side='right') - 1, 0, KNOTS - 2)
    u = (f - kx[seg]) / (kx[seg + 1] - kx[seg]); w = u * u * (3 - 2 * u)
    return vals[seg] + (vals[seg + 1] - vals[seg]) * w


def from_params(base, theta, fixed_speed=None):
    """theta -> route (f104_n2_iter.from_params); the clipped theta is kept in meta['theta']."""
    xy, f, L = _base_arrays(base)
    theta = _check_theta(theta, fixed_speed)
    a = np.clip(theta[:MODES], -caps(L), caps(L))
    lat = np.clip(sum(a[j] * np.sin((j + 1) * np.pi * f) for j in range(MODES)), -LAT_CLIP, LAT_CLIP)
    if fixed_speed is None:
        dvk = np.clip(theta[MODES:], -SP_CLIP, SP_CLIP)
        speed, dv, th = np.asarray(base['speeds'], float), _speed_knots(f, dvk), np.r_[a, dvk]
    else:
        speed, dv, th = np.full(len(xy), float(fixed_speed)), _speed_knots(f, np.zeros(KNOTS)), a
    r = _shape(xy, speed, lat, dv)
    r['meta'] = {'candidate': 'n2_iter', 'theta': th.tolist(), 'max_lateral_m': float(np.abs(lat).max()),
                 'mean_speed_mps': float(r['speeds'][1:-1].mean())}
    return r


def draw(rng, L, fixed_speed=None, mu=None, sd=None):
    """One projected theta (f104_n2_iter.draw_prior). The prior (mu = sd = None) consumes 3 lateral normals, then 4
    speed normals (sd 0 on a fixed-speed family), exactly like f104_n2_sampler.sample_one; else one N(mu, sd) draw."""
    if mu is None and sd is None:
        a = rng.normal(0, LAT_SIGMA, MODES) / np.arange(1, MODES + 1)
        dv = rng.normal(0, 0.0 if fixed_speed is not None else SP_SIGMA, KNOTS)
        return project(a if fixed_speed is not None else np.r_[a, dv], L, fixed_speed)
    if mu is None or sd is None:
        raise ValueError('draw needs both mu and sd, or neither (the prior)')
    return project(rng.normal(_check_theta(mu, fixed_speed), _check_theta(sd, fixed_speed)), L, fixed_speed)


def anchor(base, offset, cruise):
    """A designed route: constant cruise speed, lateral offset * sin^2(pi f), the family's speed limits."""
    xy, f, _ = _base_arrays(base)
    r = _shape(xy, np.full(len(xy), float(cruise)), offset * np.sin(np.pi * f) ** 2, np.zeros(len(xy)))
    r['meta'] = {'candidate': 'n2_anchor', 'lateral_offset_m': float(offset), 'cruise_speed_mps': float(cruise),
                 'max_lateral_m': abs(float(offset)), 'mean_speed_mps': float(r['speeds'][1:-1].mean())}
    return r


def family_anchors(base, speeds=ANCHOR_SPEEDS):
    """The anchors in pool order (offsets outer, speeds inner); a fixed-speed family passes speeds=(v,)."""
    return [anchor(base, off, v) for off in ANCHOR_OFFSETS for v in speeds]


def straight(base, pose, cruise=6.0, valid=validate):
    """The straight baseline (ag_picks.straight_route): the offset-0 anchor at `cruise` and its index in the one-shot
    pool (valid anchors lead it), or (None, None) when it is invalid. Uses no rng."""
    if float(cruise) not in ANCHOR_SPEEDS:
        raise ValueError(f'straight cruise {cruise} m/s is not an anchor speed {ANCHOR_SPEEDS}')
    anc = [anchor(base, 0.0, v) for v in ANCHOR_SPEEDS[:ANCHOR_SPEEDS.index(float(cruise)) + 1]]
    ok = [bool(valid(r, pose)) for r in anc]
    return (anc[-1], sum(ok[:-1])) if ok[-1] else (None, None)


# ------------------------------------------------------------------------------------------------------ base route
def constant_speed(stations, v):
    """v with the 2 m/s^2 terminal cone (gen_planner.constant_speed)."""
    st = np.asarray(stations, float)
    return np.minimum(np.full(len(st), float(v)), np.sqrt(4 * np.maximum(st[-1] - st, 0)))


def _at_2mps(xy, meta):
    st = np.r_[0., np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
    return {'waypoints': xy, 'speeds': constant_speed(st, 2.0), 'stations': st,
            'headings': np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0])), 'meta': meta}


def _hermite_xy(pose, goal, scale, step_m=0.5):
    """Cubic Hermite from the pose heading (tangent scale * |goal - pose|) to the goal (gen_planner._hermite)."""
    delta = goal - pose[:2]; length = float(np.linalg.norm(delta))
    t = np.linspace(0., 1., max(33, int(np.ceil(length / step_m)) + 1))[:, None]
    t0 = scale * length * np.array([np.cos(pose[2]), np.sin(pose[2])])
    xy = (2*t**3 - 3*t**2 + 1) * pose[:2] + (t**3 - 2*t**2 + t) * t0 + (-2*t**3 + 3*t**2) * goal + (t**3 - t**2) * delta
    return xy, delta, length, t


def arc_line(pose, goal, radius, long_way=False, step_m=0.5):
    """Turn on a circle of `radius` until facing the goal, then straight, at 2 m/s (gen_planner._arc_line); None when
    the goal is inside the circle. long_way turns away from the goal side."""
    p, th, g = np.asarray(pose[:2], float), float(pose[2]), np.asarray(goal, float)
    fwd = np.array([np.cos(th), np.sin(th)]); left = np.array([-fwd[1], fwd[0]])
    side = (1.0 if fwd[0] * (g - p)[1] - fwd[1] * (g - p)[0] >= 0 else -1.0) * (-1.0 if long_way else 1.0)   # np.cross
    c = p + side * radius * left
    cg = g - c; dist = float(np.linalg.norm(cg))
    if dist <= radius * 1.05:
        return None
    phi0 = math.atan2(p[1] - c[1], p[0] - c[0])
    sweep = ((math.atan2(cg[1], cg[0]) - side * math.acos(radius / dist) - phi0) * side) % (2 * math.pi)
    ang = phi0 + side * np.linspace(0, sweep, max(2, int(math.ceil(radius * sweep / step_m)) + 1))
    arc = c[None] + radius * np.stack([np.cos(ang), np.sin(ang)], 1)
    tp = arc[-1]
    n_line = max(2, int(math.ceil(float(np.linalg.norm(g - tp)) / step_m)) + 1)
    xy = np.concatenate([arc, (tp[None] + (g - tp)[None] * np.linspace(0, 1, n_line)[:, None])[1:]])
    st = np.r_[0., np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
    if np.any(np.diff(st) <= 1e-6):
        xy = xy[np.r_[True, np.diff(st) > 1e-6]]
    return _at_2mps(xy, {'start_tangent_scale': None, 'arc_radius_m': radius})


def base_route(pose, goal, valid=validate):
    """A route from a pose (x, y, yaw) to a goal at 2 m/s (gen_planner.base_route): the frozen generator's route_00
    (fdm_diverse_planner.propose_route_families, offset 0) if valid, else the first valid of the Hermite start scales
    1.5, 2, 0.7, 2.5, arcs of radius 12, 10, 9 and long-way arcs of 12, 10, 9, 8.5 m; else the first shape again (a
    moving decision asserts validity). M1 passes its bound and reversal check through `valid`."""
    pose, goal = np.asarray(pose, float), np.asarray(goal, float)
    if pose.shape != (3,) or goal.shape != (2,):
        raise ValueError(f'base_route: pose {pose.shape} must be (x, y, yaw) and goal {goal.shape} (x, y)')
    xy, delta, length, t = _hermite_xy(pose, goal, 1.0)
    if length < 1e-6:
        raise ValueError(f'base_route: goal {goal} coincides with the pose {pose}')
    normal = np.array([-(delta / length)[1], (delta / length)[0]])
    first = _at_2mps(xy + 0.0 * np.sin(np.pi * t) ** 2 * normal, {'start_tangent_scale': 1.0})   # the 0-offset family
    if valid(first, pose):
        return first
    options = ([_at_2mps(_hermite_xy(pose, goal, sc)[0], {'start_tangent_scale': sc}) for sc in (1.5, 2.0, 0.7, 2.5)]
               + [arc_line(pose, goal, R) for R in (12.0, 10.0, 9.0)]
               + [arc_line(pose, goal, R, long_way=True) for R in (12.0, 10.0, 9.0, 8.5)])
    return next((r for r in options if r is not None and valid(r, pose)), first)


# ---------------------------------------------------------------------------------------------- map and corridors
@dataclass(frozen=True, eq=False)
class StaticMap:
    """An arena's static overhead RGB-D planner map (static_map_v1/observation.{json,npz}; f104_n2_dataset.init_map):
    rgbd (4, n, n) f32 (read-only) with channel 3 = elevation / elev_scale (-2 off the rendered terrain), mpp =
    2 h tan(hfov / 2) / n metres per pixel, centre pixel ctr = (n - 1) / 2; observation.npz sha256-checked."""
    path: Path
    meta: dict
    sha256: str
    rgbd: np.ndarray
    elev_scale: float
    mpp: float
    ctr: float

    @classmethod
    def load(cls, path):
        """path: the folder holding observation.json/npz (Task.map)."""
        p = Path(path)
        if not (p / 'observation.json').is_file() or not (p / 'observation.npz').is_file():
            raise FileNotFoundError(f'{path}: no static_map_v1 observation.json/npz here')
        meta = json.loads((p / 'observation.json').read_text())
        raw = (p / 'observation.npz').read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != meta.get('observation_sha256'):
            raise ValueError(f'{p}/observation.npz sha256 {sha[:12]} != observation_sha256 {meta.get("observation_sha256")}')
        with np.load(io.BytesIO(raw)) as z:
            rgbd = z['rgbd'].astype(np.float32)
        if rgbd.ndim != 3 or rgbd.shape[0] != 4 or rgbd.shape[1] != rgbd.shape[2] or rgbd.shape[1] < 2:
            raise ValueError(f'{p}: rgbd shape {rgbd.shape}, expected (4, n, n)')
        rgbd.setflags(write=False)
        cam, n = meta['camera'], rgbd.shape[1]
        return cls(p, meta, sha, rgbd, float(cam['elevation_scale_m']),
                   float((2 * cam['cam_height_m'] * np.tan(cam['hfov_rad'] / 2)) / n), (n - 1) / 2.0)

    def sample(self, x, y):
        """Bilinear channel 3 at world (x, y) and whether all 4 pixels are on the terrain (f104_n2_dataset.sample_map)."""
        row, col = self.ctr - np.asarray(y) / self.mpp, self.ctr + np.asarray(x) / self.mpp
        n, R = self.rgbd.shape[1], self.rgbd[3]
        r0 = np.clip(np.floor(row).astype(int), 0, n - 2); c0 = np.clip(np.floor(col).astype(int), 0, n - 2)
        fr = np.clip(row - r0, 0, 1); fc = np.clip(col - c0, 0, 1)
        p00, p01, p10, p11 = R[r0, c0], R[r0, c0 + 1], R[r0 + 1, c0], R[r0 + 1, c0 + 1]
        valid = np.min(np.stack([p00, p01, p10, p11]), axis=0) > -1.999
        return p00 * (1 - fr) * (1 - fc) + p01 * (1 - fr) * fc + p10 * fr * (1 - fc) + p11 * fr * fc, valid

    def corridors(self, routes):
        """X (n, 5, 96, 32) f32 and L (n,) f32 (f104_n2_iter.corridors_batch): 96 stations evenly spaced along the
        route x 32 lateral samples over +-6 m; channels elevation - e0, grade and cross slope (clipped +-2), commanded
        speed along the waypoint arclength, valid. Invalid cells hold e0 = the centre cell of station 0, else the mean
        of station 0's valid cells; a route whose first station is entirely off the map raises (the original gave NaN)."""
        if not len(routes):
            raise ValueError('corridors of an empty route list')
        ns, nl, n = N_STATION, N_LATERAL, len(routes)
        GX, GY, V, LEN = np.empty((n, ns, nl)), np.empty((n, ns, nl)), np.empty((n, ns)), np.empty(n)
        off = np.linspace(-CORRIDOR_HALF_M, CORRIDOR_HALF_M, nl)
        for i, r in enumerate(routes):
            wp, sp, s = np.asarray(r['waypoints']), np.asarray(r['speeds']), np.asarray(r['stations'], float)
            s_ref = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
            if s.size != len(wp) or not np.all(np.diff(s) > 0):      # f104_n2_dataset.resample_route
                s = s_ref
            grid = np.linspace(s[0], s[-1], ns)
            pts = np.stack([np.interp(grid, s, wp[:, 0]), np.interp(grid, s, wp[:, 1])], 1)
            d = np.gradient(pts, axis=0); tn = np.linalg.norm(d, axis=1, keepdims=True); tn[tn < 1e-9] = 1e-9
            tang = d / tn; norm = np.stack([-tang[:, 1], tang[:, 0]], 1)
            GX[i] = pts[:, 0:1] + norm[:, 0:1] * off[None, :]; GY[i] = pts[:, 1:2] + norm[:, 1:2] * off[None, :]
            V[i] = np.interp(np.linspace(0, s_ref[-1], ns), s_ref, sp)
            LEN[i] = float(grid[-1] - grid[0])
        h, vld = self.sample(GX, GY)
        elev = np.where(vld, h * self.elev_scale, np.nan)
        e0 = elev[:, 0, nl // 2].copy()
        for i in np.flatnonzero(~np.isfinite(e0)):
            if not np.isfinite(elev[i, 0]).any():
                raise ValueError(f'route {i}: its first station is entirely off the map {self.path}')
            e0[i] = np.nanmean(elev[i, 0])
        fill = np.where(np.isfinite(elev), elev, e0[:, None, None])
        dss = np.maximum(LEN / (ns - 1), 1e-3)
        ga = np.clip(np.gradient(fill, axis=1) / dss[:, None, None], -2, 2)
        gc = np.clip(np.gradient(fill, 2 * CORRIDOR_HALF_M / (nl - 1), axis=2), -2, 2)
        X = np.stack([fill - e0[:, None, None], ga, gc, np.repeat(V[:, :, None], nl, 2), vld.astype(float)], 1)
        return X.astype(np.float32), LEN.astype(np.float32)


def geom5(pose, goal, L):
    """Route context (n, 5) f32 = [goal dx, goal dy, |goal - pose|, pose yaw, route length] (gen_planner.geom_ctx)."""
    rel = np.asarray(goal, float) - np.asarray(pose, float)[:2]
    return np.stack([[rel[0], rel[1], np.linalg.norm(rel), float(pose[2]), l] for l in L]).astype(np.float32)


# ------------------------------------------------------------------------------------------ history, hashes, files
def history_window(state, action, k, T=HIST_T, terminal_state=None):
    """Decision window at frame k of a recording (ga_approach.history_window): row t (j = k - T + 1 + t) =
    [state[j, observable cols] | action[j - 1]] f32, valid for 1 <= j <= k. A recording of exactly k rows (the pass-1
    prefix of every released decision) needs its terminal_state as row k, passed here or stacked onto `state`; a
    longer recording uses its row k (ci_a5data.py:452-456, 638). -> hist (T, 15) f32, hmask (T,) bool; the encoder
    appends the mask as channel 16."""
    st, ac, k = np.asarray(state, np.float32), np.asarray(action, np.float32), int(k)
    if st.ndim != 2 or st.shape[1] != Z1_DIM or ac.ndim != 2 or ac.shape[1] != ACT_DIM or k < 0:
        raise ValueError(f'history_window: state {st.shape}, action {ac.shape}, k {k}')
    if k > 0 and len(st) == k:
        if terminal_state is None:
            raise ValueError(f'the recording has exactly k = {k} rows: its terminal_state is row k')
        st = np.vstack([st, np.asarray(terminal_state, np.float32).reshape(1, Z1_DIM)])
    if k > 0 and (len(st) <= k or len(ac) < k):     # the original silently masked row k (min(k, n - 1))
        raise ValueError(f'{len(st)} state / {len(ac)} action rows: no decision window at frame {k}')
    no = len(OBSERVABLE_COLS)
    hist, mask = np.zeros((T, HIST_DIM), np.float32), np.zeros(T, bool)
    for t in range(T):
        j = k - T + 1 + t
        if 1 <= j <= k:
            hist[t, :no], hist[t, no:], mask[t] = st[j, list(OBSERVABLE_COLS)], ac[j - 1], True
    return hist, mask


def route_json(r) -> dict:
    """A route as written to JSON: the KEYS as float lists."""
    return {k: np.asarray(r[k], float).tolist() for k in KEYS}


def route_sha256(r):
    """Content hash of a route as written (f104_n2_iter.route_sha256)."""
    return hashlib.sha256(json.dumps(route_json(r)).encode()).hexdigest()


def ends_within(route, start, goal, tol=ENDS_TOL_M) -> bool:
    """The route starts within `tol` of `start` (None: not checked) and ends within `tol` of `goal`
    (the collectors' contract, crm_collect.py:199-200)."""
    xy = np.asarray(route['waypoints'], float)
    return bool((start is None or np.linalg.norm(xy[0] - np.asarray(start, float)) <= tol)
                and np.linalg.norm(xy[-1] - np.asarray(goal, float)) <= tol)


def route_time(r):
    """Commanded route time: sum ds / v_mid with v_mid floored at 0.3 m/s (f104_n2_iter.route_time)."""
    st, v = np.asarray(r['stations'], float), np.asarray(r['speeds'], float)
    return float((np.diff(st) / np.maximum(0.5 * (v[1:] + v[:-1]), 0.3)).sum())


def load_route(src):
    """A route JSON (path or parsed dict) as float64 arrays with its meta cleared (planner_arms.load_case): a planner
    base (route_00, whose meta.fdm_station the validator would otherwise read) or a given route."""
    d = json.loads(Path(src).read_text()) if isinstance(src, (str, Path)) else src
    missing = [k for k in KEYS if k not in d]
    if missing:
        raise ValueError(f'{src if isinstance(src, (str, Path)) else "route"}: route lacks {missing}')
    return {**{k: np.asarray(d[k], float) for k in KEYS}, 'meta': {}}
