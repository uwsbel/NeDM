"""Build the f104 fixed-arena planner dataset.

One row per candidate route (= one collected episode). Inputs are strictly
pre-drive: the static RGB-D terrain map, the start state, and the planned
reference route. Labels come from the recorded outcome.
"""
import json, os, glob, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = '/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_f104_50h_20260909'
RUNS = ROOT + '/production_v2/runs'
MAPD = ROOT + '/static_map_v1'
OUT  = ROOT + '/planner_dataset_v1.npz'

N_STATION = 96          # samples along the route
N_LATERAL = 32          # samples across the corridor
HALF_WIDTH_M = 6.0      # corridor half-width

_cam = json.load(open(MAPD + '/observation.json'))['camera']
_obs = np.load(MAPD + '/observation.npz')
RGBD = _obs['rgbd'].astype(np.float32)            # (4,512,512) ch3 = elev/10, -2 sentinel
ELEV_SCALE = float(_cam['elevation_scale_m'])
NPX = RGBD.shape[1]
MPP = (2 * _cam['cam_height_m'] * np.tan(_cam['hfov_rad'] / 2)) / NPX
CTR = (NPX - 1) / 2.0


def world_to_px(x, y):
    """Solved and validated against privileged heights: rmse 0.033 m."""
    row = CTR - np.asarray(y) / MPP
    col = CTR + np.asarray(x) / MPP
    return row, col


def sample_map(x, y):
    """Bilinear sample of the 4 RGB-D channels. Returns (4,...) and a valid mask."""
    row, col = world_to_px(x, y)
    r0 = np.clip(np.floor(row).astype(int), 0, NPX - 2)
    c0 = np.clip(np.floor(col).astype(int), 0, NPX - 2)
    fr = np.clip(row - r0, 0, 1)[None]
    fc = np.clip(col - c0, 0, 1)[None]
    p00 = RGBD[:, r0, c0]; p01 = RGBD[:, r0, c0 + 1]
    p10 = RGBD[:, r0 + 1, c0]; p11 = RGBD[:, r0 + 1, c0 + 1]
    valid = np.min(np.stack([p00[3], p01[3], p10[3], p11[3]]), axis=0) > -1.999
    out = (p00 * (1 - fr) * (1 - fc) + p01 * (1 - fr) * fc
           + p10 * fr * (1 - fc) + p11 * fr * fc)
    return out, valid


def resample_route(wp, stations, n):
    """Arc-length resample the reference polyline to n points."""
    s = np.asarray(stations, dtype=np.float64)
    if s.size != len(wp) or not np.all(np.diff(s) > 0):
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
        s = d
    grid = np.linspace(s[0], s[-1], n)
    return np.stack([np.interp(grid, s, wp[:, 0]), np.interp(grid, s, wp[:, 1])], 1), grid


def one(ep):
    try:
        oc = json.load(open(ep + '/outcome.json'))
        cs = json.load(open(ep + '/case.json'))
        cr = np.load(ep + '/command_reference.npz')
        an = np.load(ep + '/anchor_state.npz', allow_pickle=True)
    except Exception as e:
        return None

    wp = np.asarray(cr['reference_waypoints'], dtype=np.float64)
    if wp.ndim != 2 or wp.shape[0] < 3:
        return None
    pts, grid = resample_route(wp, cr['reference_stations'], N_STATION)

    # local tangent / normal per station
    d = np.gradient(pts, axis=0)
    tn = np.linalg.norm(d, axis=1, keepdims=True); tn[tn < 1e-9] = 1e-9
    tang = d / tn
    norm = np.stack([-tang[:, 1], tang[:, 0]], 1)

    # corridor sample grid: (N_STATION, N_LATERAL)
    off = np.linspace(-HALF_WIDTH_M, HALF_WIDTH_M, N_LATERAL)
    gx = pts[:, 0:1] + norm[:, 0:1] * off[None, :]
    gy = pts[:, 1:2] + norm[:, 1:2] * off[None, :]
    patch, vmask = sample_map(gx, gy)                     # (4,S,L), (S,L)
    patch = patch.astype(np.float32)
    elev = patch[3] * ELEV_SCALE
    elev = np.where(vmask, elev, np.nan)

    # ---- terrain-profile scalar features (the heuristic baseline lives here) ----
    ds = float(np.mean(np.diff(grid))) if grid.size > 1 else 1.0
    centre = elev[:, N_LATERAL // 2]
    with np.errstate(invalid='ignore'):
        grade = np.gradient(np.nan_to_num(centre, nan=np.nanmean(centre)), max(ds, 1e-6))
        cross = (elev[:, -1] - elev[:, 0]) / (2 * HALF_WIDTH_M)
        rough = np.nanstd(elev, axis=1)
    fin = lambda a: np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    grade, cross, rough = fin(grade), fin(cross), fin(rough)

    speeds = np.asarray(cr['reference_speeds'], dtype=np.float64)
    hdg = np.unwrap(np.asarray(cr['reference_headings'], dtype=np.float64))
    route_len = float(np.sum(np.linalg.norm(np.diff(wp, axis=0), axis=1)))
    curv = np.abs(np.gradient(np.unwrap(np.arctan2(tang[:, 1], tang[:, 0])), max(ds, 1e-6)))

    scal = dict(
        route_len_m=route_len,
        straight_dist_m=float(np.linalg.norm(wp[-1] - wp[0])),
        tortuosity=float(route_len / max(np.linalg.norm(wp[-1] - wp[0]), 1e-6)),
        max_abs_grade=float(np.max(np.abs(grade))),
        mean_abs_grade=float(np.mean(np.abs(grade))),
        p95_abs_grade=float(np.percentile(np.abs(grade), 95)),
        max_abs_cross=float(np.max(np.abs(cross))),
        mean_abs_cross=float(np.mean(np.abs(cross))),
        max_rough=float(np.max(rough)),
        mean_rough=float(np.mean(rough)),
        elev_gain_m=float(np.sum(np.clip(np.diff(np.nan_to_num(centre)), 0, None))),
        elev_range_m=float(np.nanmax(centre) - np.nanmin(centre)),
        max_curv=float(np.max(curv)),
        mean_curv=float(np.mean(curv)),
        total_heading_change=float(np.abs(hdg[-1] - hdg[0])) if hdg.size else 0.0,
        mean_ref_speed=float(np.mean(speeds)) if speeds.size else 0.0,
        min_ref_speed=float(np.min(speeds)) if speeds.size else 0.0,
        invalid_frac=float(1.0 - vmask.mean()),
    )

    st = np.asarray(an['state'], dtype=np.float32)
    hist = np.asarray(an['history'], dtype=np.float32)
    lay = cs.get('layout', {})
    status = oc.get('status', 'unknown')

    return dict(
        ep=os.path.basename(ep),
        group=cs.get('id', ''),
        split=cs.get('split', ''),
        stratum=cs.get('evaluation_stratum', ''),
        status=status,
        failure=int(status != 'goal_reached'),
        blockage=int(status == 'prolonged_blockage_terminated'),
        stall_flag=int(bool(oc.get('bounded_blockage_v1')) or bool(oc.get('sustained_near_stop'))),
        near_stop_s=float(oc.get('longest_consecutive_effortful_near_zero_speed_s') or 0.0),
        goal_time_s=float(oc.get('goal_time_s') or oc.get('elapsed_s') or 0.0),
        elapsed_s=float(oc.get('elapsed_s') or 0.0),
        init_goal_dist=float(oc.get('initial_goal_distance_m') or 0.0),
        work_kj=float(oc.get('positive_work_kj') or 0.0),
        start_xy=np.asarray(lay.get('start_xy', [0, 0]), dtype=np.float32),
        start_yaw=np.float32(lay.get('start_yaw', 0.0)),
        goal_xy=np.asarray(cs.get('goal_xy', [0, 0]), dtype=np.float32),
        patch=patch, state0=st, hist=hist,
        route=pts.astype(np.float32),
        scal=np.array([scal[k] for k in SCAL_KEYS], dtype=np.float32),
    )


SCAL_KEYS = ['route_len_m', 'straight_dist_m', 'tortuosity', 'max_abs_grade', 'mean_abs_grade',
             'p95_abs_grade', 'max_abs_cross', 'mean_abs_cross', 'max_rough', 'mean_rough',
             'elev_gain_m', 'elev_range_m', 'max_curv', 'mean_curv', 'total_heading_change',
             'mean_ref_speed', 'min_ref_speed', 'invalid_frac']

if __name__ == '__main__':
    eps = sorted(glob.glob(RUNS + '/*'))
    if len(sys.argv) > 1:
        eps = eps[:int(sys.argv[1])]
    print(f'extracting {len(eps)} episodes ...', flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=14) as ex:
        for i, r in enumerate(ex.map(one, eps, chunksize=32)):
            if r is not None:
                rows.append(r)
            if (i + 1) % 2000 == 0:
                print(f'  {i+1}/{len(eps)}', flush=True)
    print(f'kept {len(rows)}/{len(eps)}', flush=True)
    stack = lambda k: np.stack([r[k] for r in rows])
    arr = lambda k, dt: np.array([r[k] for r in rows], dtype=dt)
    np.savez_compressed(
        OUT,
        ep=arr('ep', object), group=arr('group', object), split=arr('split', object),
        stratum=arr('stratum', object), status=arr('status', object),
        failure=arr('failure', np.int8), blockage=arr('blockage', np.int8),
        stall_flag=arr('stall_flag', np.int8), near_stop_s=arr('near_stop_s', np.float32),
        goal_time_s=arr('goal_time_s', np.float32), elapsed_s=arr('elapsed_s', np.float32),
        init_goal_dist=arr('init_goal_dist', np.float32), work_kj=arr('work_kj', np.float32),
        start_xy=stack('start_xy'), start_yaw=arr('start_yaw', np.float32),
        goal_xy=stack('goal_xy'), patch=stack('patch'), state0=stack('state0'),
        hist=stack('hist'), route=stack('route'), scal=stack('scal'),
        scal_keys=np.array(SCAL_KEYS, dtype=object),
    )
    print('wrote', OUT, os.path.getsize(OUT) / 1e6, 'MB')
