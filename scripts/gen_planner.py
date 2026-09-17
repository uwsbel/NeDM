"""The deployed planner as one importable module: terrain map -> candidate routes -> risk -> pick.

Used by the generalisation test (picks made offline, driven later) and by the multi-goal mission runner (replans
online inside Chrono at every goal). Nothing here is new modelling; it packages the night-2 pieces:
  proposal   scripts/f104_n2_sampler.py (256 candidates: 9 designed-route anchors + wide sine-basis samples)
  corridor   scripts/f104_n2_dataset.station_tensor (96 stations x 32 lateral samples)
  model      night2_v1/final/N2_s*.pt (5-seed ensemble, geometry-only context)
  hand rule  gen_v1/hand_rule.json (non-learned terrain+speed baseline, fit on the same f104 training routes)

Map source: the arena heightmap (TerrainMap), encoded the way the model's static overhead depth map is encoded.
On f104 the two agree to 0.035 m mean height error and 0.985 logit correlation, and heightmaps exist for every
arena, so the same code path serves f104 and the new arenas.
"""
import glob, json, math, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# On the cluster the planner runs inside the frozen collector's process: GEN_SRC points at that source tree's src
# (its route-family and validator code is identical to this worktree's), GEN_MODELS / GEN_RULE at the copies.
sys.path.insert(0, os.environ.get('GEN_SRC', str(ROOT / 'src'))); sys.path.insert(0, str(HERE))
import f104_n2_dataset as DS
import f104_n2_sampler as S
from nedm.traverse.terrain import TerrainMap
from nedm.traverse.fdm_mppi import MPPIConfig, validate_reference
from nedm.traverse.fdm_diverse_planner import propose_route_families
from gen_terrain_features import FEATURES, route_features

CFG = MPPIConfig(max_speed_mps=6., min_speed_mps=0.0, max_curvature_inv_m=.125, arena_half_extent_m=40.)
ELEV_SCALE = 10.0
N_CAND = 256


def set_map(arena_dir):
    """Point the corridor sampler at an arena's heightmap (row 0 = +y, as in the static depth map)."""
    tm = TerrainMap.from_dir(Path(arena_dir))
    h = np.flipud(tm.height_grid).astype(np.float32)
    n = h.shape[0]
    R = np.zeros((4, n, n), np.float32); R[3] = h / ELEV_SCALE
    DS.G.update(rgbd=R, elev_scale=ELEV_SCALE, npx=n, mpp=tm.size_m / n, ctr=(n - 1) / 2.0)
    return tm


def constant_speed(stations, v):
    stations = np.asarray(stations, float)
    return np.minimum(np.full(len(stations), float(v)), np.sqrt(4 * np.maximum(stations[-1] - stations, 0)))


def _hermite(pose, goal, scale, step_m=.5):
    pose, goal = np.asarray(pose, float), np.asarray(goal, float)
    delta = goal - pose[:2]; length = float(np.linalg.norm(delta))
    t = np.linspace(0., 1., max(33, int(np.ceil(length / step_m)) + 1))[:, None]
    t0 = scale * length * np.array([np.cos(pose[2]), np.sin(pose[2])])
    xy = (2*t**3 - 3*t**2 + 1) * pose[:2] + (t**3 - 2*t**2 + t) * t0 + (-2*t**3 + 3*t**2) * goal + (t**3 - t**2) * delta
    st = np.r_[0., np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
    hd = np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0]))
    return {'waypoints': xy, 'speeds': constant_speed(st, 2.), 'stations': st, 'headings': hd, 'meta': {'start_tangent_scale': scale}}


def _arc_line(pose, goal, radius, step_m=.5, long_way=False):
    """Turn on a circle of the given radius until facing the goal, then drive straight (Dubins CS).
    long_way=True turns away from the goal side (a wider loop), for starts facing out of the arena."""
    p = np.asarray(pose[:2], float); th = float(pose[2]); g = np.asarray(goal, float)
    fwd = np.array([np.cos(th), np.sin(th)]); left = np.array([-fwd[1], fwd[0]])
    side = 1.0 if np.cross(fwd, g - p) >= 0 else -1.0
    if long_way:
        side = -side
    c = p + side * radius * left
    cg = g - c; dist = float(np.linalg.norm(cg))
    if dist <= radius * 1.05:
        return None
    phi0 = math.atan2(p[1] - c[1], p[0] - c[0])
    # tangent point: angle of c->g minus/plus acos(R/d)
    base = math.atan2(cg[1], cg[0]); off = math.acos(radius / dist)
    phi_t = base - side * off
    sweep = (phi_t - phi0) * side
    sweep = sweep % (2 * math.pi)
    n_arc = max(2, int(math.ceil(radius * sweep / step_m)) + 1)
    ang = phi0 + side * np.linspace(0, sweep, n_arc)
    arc = c[None] + radius * np.stack([np.cos(ang), np.sin(ang)], 1)
    tp = arc[-1]; L = float(np.linalg.norm(g - tp))
    n_line = max(2, int(math.ceil(L / step_m)) + 1)
    line = tp[None] + (g - tp)[None] * np.linspace(0, 1, n_line)[:, None]
    xy = np.concatenate([arc, line[1:]])
    st = np.r_[0., np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
    if np.any(np.diff(st) <= 1e-6):
        keep = np.r_[True, np.diff(st) > 1e-6]; xy = xy[keep]; st = np.r_[0., np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
    hd = np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0]))
    return {'waypoints': xy, 'speeds': constant_speed(st, 2.), 'stations': st, 'headings': hd,
            'meta': {'start_tangent_scale': None, 'arc_radius_m': radius}}


def safe_validate(route, obstacles, cfg, anchor):
    """validate_reference, but a degenerate route (e.g. folded by a lateral offset) is rejected instead of raising."""
    try:
        return validate_reference(route, obstacles, cfg, anchor)
    except ValueError as e:
        return {'valid': False, 'reasons': [str(e)]}


def base_route(pose, goal, scales=(1.0, 1.5, 2.0, 0.7, 2.5), radii=(12.0, 10.0, 9.0)):
    """Straight route from a pose (x, y, yaw) to a goal at 2 m/s.

    First choice is exactly the frozen generator's route_00 (Hermite, propose_route_families). A mission leg can
    start with the vehicle facing well away from the next goal, where that shape breaks the curvature limit; then
    wider/tighter Hermite starts and finally a tight-arc-then-straight route are tried in a fixed order, and the
    first shape the validator accepts is used. Single start/goal tests never reach the fallbacks (their starts face
    the goal).
    """
    r = propose_route_families(list(pose), goal, speeds=[2.], offsets=[0.], step_m=.5)[0]
    r['speeds'] = constant_speed(r['stations'], 2.)
    first = {k: np.asarray(r[k], float) for k in ('waypoints', 'speeds', 'stations', 'headings')} | {'meta': {'start_tangent_scale': 1.0}}
    if safe_validate(first, [], CFG, np.asarray(pose, float))['valid']:
        return first
    options = ([_hermite(pose, goal, sc) for sc in scales[1:]] + [_arc_line(pose, goal, R) for R in radii]
               + [_arc_line(pose, goal, R, long_way=True) for R in radii + (8.5,)])
    for cand in options:
        if cand is not None and safe_validate(cand, [], CFG, np.asarray(pose, float))['valid']:
            return cand
    return first


def proposal_pool(base, pose, rng, n=N_CAND):
    """Night-2 proposal: anchors first, then wide random samples, rejection on the route validator."""
    cands, tries = S.propose(base, np.asarray(pose, float), rng, n=n, validate=safe_validate, cfg=CFG)
    return cands, tries


def fixed2_pool(base, pose, rng, n=N_CAND):
    """Geometry-only candidates at a constant 2 m/s (the speed-constrained regime)."""
    out = []
    for _ in range(6 * n):
        if len(out) >= n: break
        r = S.sample_one(base, rng, lat_sigma=5.0, sp_sigma=0.0, base_speed=2.0)
        if safe_validate(r, [], CFG, np.asarray(pose, float))['valid']:
            out.append(r)
    return out


def corridors(cands):
    X, L = [], []
    for r in cands:
        x, l = DS.station_tensor(np.asarray(r['waypoints']), np.asarray(r['speeds']), np.asarray(r['stations']))
        X.append(x); L.append(l)
    return np.stack(X).astype(np.float32), np.asarray(L, np.float32)


def geom_ctx(start_xy, goal_xy, start_yaw, L):
    rel = np.asarray(goal_xy, float) - np.asarray(start_xy, float)
    return np.stack([[rel[0], rel[1], np.linalg.norm(rel), float(start_yaw), l] for l in L]).astype(np.float32)


class RiskModel:
    """Deployed 5-seed ensemble. score() returns (mean route logit, P(unsafe))."""
    def __init__(self, pattern=None, device=None):
        import torch
        from gen_riskmodel import Net, route_logit
        self.torch, self.route_logit = torch, route_logit
        pattern = pattern or os.environ.get('GEN_MODELS') or str(ROOT / 'artifacts/traverse/fdm_f104_50h_20260909/night2_v1/final/N2_s*.pt')
        self.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.members = []
        for p in sorted(glob.glob(pattern)):
            ck = torch.load(p, map_location=self.dev, weights_only=False)
            m = Net(ck['cin'], ck['nctx'], arch=ck['arch'], layers=ck.get('layers', 2)).to(self.dev)
            m.load_state_dict(ck['state']); m.eval(); self.members.append((m, ck))
        if not self.members:
            raise FileNotFoundError(pattern)

    def score(self, X, ctx5, bs=256):
        torch = self.torch; zs = []
        for m, ck in self.members:
            nm = ck['norm']; out = []
            for i in range(0, len(X), bs):
                x = X[i:i + bs].astype(np.float32).copy()
                x[:, :4] = (x[:, :4] - nm['mu'][None, :, None, None]) / nm['sd'][None, :, None, None]
                x = np.concatenate([x, np.ones((len(x), 1, x.shape[2], x.shape[3]), np.float32)], 1)
                c = (ctx5[i:i + bs] - ck['ctx_mu']) / ck['ctx_sd']
                with torch.no_grad():
                    out.append(self.route_logit(m(torch.tensor(x, device=self.dev),
                                                  torch.tensor(c, dtype=torch.float32, device=self.dev))).double().cpu().numpy())
            zs.append(np.concatenate(out))
        z = np.mean(zs, 0)
        return z, 1 - np.exp(-np.exp(z))


class HandRule:
    """Non-learned baseline: linear terrain+speed score, higher = riskier."""
    def __init__(self, path=None):
        path = path or os.environ.get('GEN_RULE') or ROOT / 'artifacts/traverse/fdm_f104_50h_20260909/gen_v1/hand_rule.json'
        d = json.load(open(path))
        assert d['features'] == FEATURES
        self.mu, self.sd, self.w = (np.asarray(d[k], float) for k in ('mu', 'sd', 'w'))

    def features(self, X, L):
        return np.stack([route_features(X[i], float(L[i])) for i in range(len(X))])

    def score(self, F):
        return ((F - self.mu) / self.sd) @ self.w


def anchor_index(cands, offset, cruise):
    for i, r in enumerate(cands):
        m = r.get('meta', {})
        if m.get('candidate') == 'n2_anchor' and abs(m.get('lateral_offset_m', 99) - offset) < 1e-6 \
                and abs(m.get('cruise_speed_mps', -1) - cruise) < 1e-6:
            return i
    return None


def plan(pose, goal, rng, model=None, rule=None, mode='n2'):
    """One planning call. mode: 'n2' | 's' (sensor model, needs set_sensor_map) | 'rule' | 'straight6' | 'straight2'."""
    base = base_route(pose, goal)
    cands, tries = proposal_pool(base, pose, rng)
    info = {'mode': mode, 'n_candidates': len(cands), 'tries': tries,
            'base_start_tangent_scale': base['meta'].get('start_tangent_scale'), 'base_arc_radius_m': base['meta'].get('arc_radius_m')}
    if not cands:
        return None, {**info, 'reason': 'no valid candidate'}
    if mode in ('straight6', 'straight2'):
        i = anchor_index(cands, 0.0, 6.0 if mode == 'straight6' else 2.0)
        if i is None:
            return None, {**info, 'reason': 'straight anchor failed validation'}
        return cands[i], {**info, 'index': i}
    if mode == 's':
        X10, L = corridors10(cands)
        z, pr = model.score(X10, geom_ctx(pose[:2], goal, pose[2], L))
        i = int(np.argmin(z)); info.update(index=i, risk=float(pr[i]), logit=float(z[i]), risk_median=float(np.median(pr)))
        r = cands[i]
        info.update(mean_speed=float(np.asarray(r['speeds'])[1:-1].mean()), length_m=float(np.asarray(r['stations'])[-1]))
        return r, info
    X, L = corridors(cands)
    if mode == 'n2':
        z, p = model.score(X, geom_ctx(pose[:2], goal, pose[2], L))
        i = int(np.argmin(z)); info.update(index=i, risk=float(p[i]), logit=float(z[i]),
                                          risk_median=float(np.median(p)))
    else:
        s = rule.score(rule.features(X, L))
        i = int(np.argmin(s)); info.update(index=i, rule_score=float(s[i]))
    r = cands[i]
    info.update(mean_speed=float(np.asarray(r['speeds'])[1:-1].mean()), length_m=float(np.asarray(r['stations'])[-1]))
    return r, info


# ---------------------------------------------------------------------------------------------------------------
# sensor_v1: corridors from an arena's captured overhead RGB-D image (no heightmap), for both the current model
# (height channels computed from that image) and sensor-channel models.
# ---------------------------------------------------------------------------------------------------------------
SENSOR = {'on': False}


def set_sensor_map(mapdir):
    """Use the captured RGB-D image at mapdir (observation.npz/json) for every corridor from now on."""
    import sensor_dataset as SD
    SD.init_map(str(mapdir))
    DS.G.update(rgbd=SD.G['rgbd'], elev_scale=SD.G['elev_scale'], npx=SD.G['npx'], mpp=SD.G['mpp'], ctr=SD.G['ctr'])
    SENSOR.update(on=True, SD=SD)


def corridors10(cands):
    """Ten-channel sensor corridors (sensor_dataset.X10 layout) and route lengths."""
    SD = SENSOR['SD']; X, L = [], []
    for r in cands:
        x, l = SD.tensor10(np.asarray(r['waypoints']), np.asarray(r['speeds']), np.asarray(r['stations']))
        X.append(x); L.append(l)
    return np.stack(X).astype(np.float32), np.asarray(L, np.float32)


class SensorRiskModel:
    """Ensemble trained by scripts/sensor_train.py on a chosen channel set. score(X10, ctx5) -> (logit, P)."""
    X10 = ['elev_rel', 'grade', 'cross', 'speed', 'valid', 'depth_rel', 'ray_sec1', 'R', 'G', 'B']

    def __init__(self, pattern, device=None):
        import torch
        from gen_riskmodel import Net, route_logit
        self.torch, self.route_logit = torch, route_logit
        self.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.members = []
        for p in sorted(glob.glob(pattern)):
            ck = torch.load(p, map_location=self.dev, weights_only=False)
            m = Net(ck['cin'], ck['nctx'], arch=ck['arch'], layers=ck.get('layers', 2)).to(self.dev)
            m.load_state_dict(ck['state']); m.eval(); self.members.append((m, ck))
        if not self.members:
            raise FileNotFoundError(pattern)
        self.channels = self.members[0][1]['channels']

    def score(self, X10, ctx5, bs=256):
        torch = self.torch; zs = []
        for m, ck in self.members:
            sel = [self.X10.index(c) for c in ck['channels']]; nm = ck['norm']; cont = nm['cont_index']; out = []
            for i in range(0, len(X10), bs):
                x = X10[i:i + bs][:, sel].astype(np.float32).copy()
                x[:, cont] = (x[:, cont] - nm['mu'][None, :, None, None]) / nm['sd'][None, :, None, None]
                x = np.concatenate([x, np.ones((len(x), 1, x.shape[2], x.shape[3]), np.float32)], 1)
                c = (ctx5[i:i + bs] - ck['ctx_mu']) / ck['ctx_sd']
                with torch.no_grad():
                    out.append(self.route_logit(m(torch.tensor(x, device=self.dev),
                                                  torch.tensor(c, dtype=torch.float32, device=self.dev))).double().cpu().numpy())
            zs.append(np.concatenate(out))
        z = np.mean(zs, 0)
        return z, 1 - np.exp(-np.exp(z))


# ---------------------------------------------------------------------------------------------------------------
# sensor_v2: corridors from the back-projected world grid (scripts/sensor_map_v2.py), for matched height/depth models.
# ---------------------------------------------------------------------------------------------------------------
def set_grid_map(griddir):
    """Use a v2 world grid (depth back-projected with the camera intrinsics) for corridors from now on."""
    import sensor_dataset_v2 as V2
    V2.init_grid(str(griddir))
    SENSOR.update(on=True, V2=V2)


def corridors12(cands):
    V2 = SENSOR['V2']; X, L = [], []
    for r in cands:
        x, l = V2.tensor12(np.asarray(r['waypoints']), np.asarray(r['speeds']), np.asarray(r['stations']))
        X.append(x); L.append(l)
    return np.stack(X).astype(np.float32), np.asarray(L, np.float32)


class GridRiskModel(SensorRiskModel):
    """Ensemble trained by scripts/sensor_train_v2.py on v2 corridors (channel names in sensor_dataset_v2.CHANNELS)."""
    X10 = ['z_rel', 'grade', 'cross', 'speed', 'valid', 'range_abs', 'range_rel', 'sec1', 'R', 'G', 'B', 'cover1']
