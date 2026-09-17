"""Night-2 MPPI proposal. Fixes the two limits that caused last night's 4 closed-loop failures.

Night-1 proposal: 3 knots, lateral sigma 2.6 m (clip +-6), speed deltas multiplied by a sin^2 envelope that is
ZERO at both ends -- so every candidate approached the goal at the base 2 m/s, and a goal on a hill or past a
crater could not be reached with momentum. Lateral spread was also only +-3-4 m in practice.

Night-2 proposal:
  lateral : smooth sine basis sum_j a_j sin(j*pi*f), j = 1..3 (zero at both ends, so the start pose and the
            goal stay pinned). Amplitudes are drawn N(0, sigma/j) and then scaled to the curvature budget
            a_j <= 0.55 * kappa_max * L^2 / (j*pi)^2, which is what the route validator actually enforces.
            CORRECTION (audited 2026-09-12): an earlier draft of this docstring claimed the night-1 knot bumps
            were "rejected 98% of the time". Wrong -- measured night-1 acceptance is 48.8% on average (min 12.7%).
            What limited night 1 was its +-6 m lateral CLIP and the speed envelope, not a rejection collapse.
            Curvature is nonetheless the only rejection reason for both proposals (100% of rejections).
  speed   : 4 knots, sigma 1.5 m/s, clip +-4, NO end envelope. Only two physical caps remain:
            the terminal deceleration cone v <= sqrt(2*a*(L-s)) (a = 2 m/s^2, same as the designed routes)
            and the forward/backward acceleration projection.
  anchors : the 9 designed-style routes (offsets 0, -4, +4 m x 2, 4, 6 m/s) are always in the candidate set,
            so the planner can always fall back on a fast straight line.
"""
import numpy as np

A_ACC, A_DEC = 1.5, 2.0          # m/s^2, matches MPPIConfig defaults used in the campaign
V_MIN, V_MAX = 0.5, 6.0


def smooth_interp(f, k, vals):
    """C1 knot interpolation (smoothstep), zero at both ends."""
    kx = np.linspace(0.0, 1.0, k + 2); kv = np.r_[0.0, vals, 0.0]
    seg = np.clip(np.searchsorted(kx, f, side='right') - 1, 0, k)
    u = (f - kx[seg]) / (kx[seg + 1] - kx[seg]); w = u * u * (3 - 2 * u)
    return kv[seg] + (kv[seg + 1] - kv[seg]) * w


def speed_knots(f, k, vals):
    """Speed profile knots that are NOT forced to zero at the ends (free end speed)."""
    kx = np.linspace(0.0, 1.0, k)
    seg = np.clip(np.searchsorted(kx, f, side='right') - 1, 0, k - 2)
    u = (f - kx[seg]) / (kx[seg + 1] - kx[seg]); w = u * u * (3 - 2 * u)
    return vals[seg] + (vals[seg + 1] - vals[seg]) * w


def shape(xy, station, speed, lat, dv):
    pts = xy.copy()
    t = np.gradient(xy, axis=0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    pts = xy + lat[:, None] * np.stack([-t[:, 1], t[:, 0]], 1)
    st = np.r_[0.0, np.linalg.norm(np.diff(pts, axis=0), axis=1).cumsum()]
    v = np.clip(speed + dv, V_MIN, V_MAX)
    v = np.minimum(v, np.sqrt(2 * A_DEC * np.maximum(st[-1] - st, 0)))      # terminal deceleration cone
    for j in range(1, len(v)):
        v[j] = min(v[j], np.sqrt(max(v[j-1] ** 2 + 2 * A_ACC * (st[j] - st[j-1]), 0)))
    for j in range(len(v) - 2, -1, -1):
        v[j] = min(v[j], np.sqrt(max(v[j+1] ** 2 + 2 * A_DEC * (st[j+1] - st[j]), 0)))
    hd = np.arctan2(np.gradient(pts[:, 1]), np.gradient(pts[:, 0]))
    return {'waypoints': pts, 'speeds': v, 'stations': st, 'headings': hd, 'meta': {}}


def lateral_profile(f, L, rng, sigma=5.0, modes=3, kappa_max=0.125, budget=0.55):
    """Curvature-safe lateral offset: sum_j a_j sin(j pi f), |a_j| capped by the validator's curvature limit."""
    a = rng.normal(0, sigma, modes) / np.arange(1, modes + 1)
    cap = budget * kappa_max * L ** 2 / (np.arange(1, modes + 1) * np.pi) ** 2
    a = np.clip(a, -cap, cap)
    return sum(a[j] * np.sin((j + 1) * np.pi * f) for j in range(modes)), a


def sample_one(base, rng, knots=4, lat_sigma=5.0, lat_clip=10.0, sp_sigma=1.5, sp_clip=4.0, base_speed=None,
               modes=3, kappa_max=0.125):
    xy = np.asarray(base['waypoints'], float); station = np.asarray(base['stations'], float)
    speed = np.asarray(base['speeds'], float) if base_speed is None else np.full(len(xy), float(base_speed))
    f = np.clip((station - station[0]) / max(station[-1] - station[0], 1e-6), 0, 1)
    L = float(station[-1] - station[0])
    lat, _ = lateral_profile(f, L, rng, sigma=lat_sigma, modes=modes, kappa_max=kappa_max)
    lat = np.clip(lat, -lat_clip, lat_clip)
    dv = speed_knots(f, knots, np.clip(rng.normal(0, sp_sigma, knots), -sp_clip, sp_clip))
    r = shape(xy, station, speed, lat, dv)
    r['meta'] = {'candidate': 'n2_wide', 'max_lateral_m': float(np.abs(lat).max()),
                 'mean_speed_mps': float(r['speeds'][1:-1].mean())}
    return r


def anchors(base, offsets=(0.0, -4.0, 4.0), speeds=(2.0, 4.0, 6.0)):
    """The designed-route family: constant speed, fixed lateral offset, terminal deceleration cone."""
    xy = np.asarray(base['waypoints'], float); station = np.asarray(base['stations'], float)
    f = np.clip((station - station[0]) / max(station[-1] - station[0], 1e-6), 0, 1)
    out = []
    for off in offsets:
        lat = off * np.sin(np.pi * f) ** 2
        for v in speeds:
            r = shape(xy, station, np.full(len(xy), float(v)), lat, np.zeros(len(xy)))
            r['meta'] = {'candidate': 'n2_anchor', 'lateral_offset_m': float(off), 'cruise_speed_mps': float(v),
                         'max_lateral_m': abs(float(off)), 'mean_speed_mps': float(r['speeds'][1:-1].mean())}
            out.append(r)
    return out


def propose(base, anchor_pose, rng, n=256, validate=None, cfg=None, **kw):
    """n valid candidates: the 9 anchors first, then wide random samples (rejection on the route validator)."""
    out = []
    for r in anchors(base):
        if validate is None or validate(r, [], cfg, anchor_pose)['valid']:
            out.append(r)
    tries = 0
    while len(out) < n and tries < 8 * n:
        tries += 1
        r = sample_one(base, rng, **kw)
        if validate is None or validate(r, [], cfg, anchor_pose)['valid']:
            out.append(r)
    return out, tries
