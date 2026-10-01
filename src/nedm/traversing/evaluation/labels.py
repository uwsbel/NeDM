"""Offline outcome labels of the traversing evaluation, and the cell of an unlabelled drive per results table.

Pure functions of a drive's stored 50 ms arrays (numpy only). Every rule compares in the dtype the study compared in
(spec 0.11): rolling back and the mission slide on the recorded float32 vx and throttle (vx == f32(-0.1) is not
backwards, throttle == f32(0.3) is not effortful), the belly run after astype(float64), the tracker metrics in float64.
  point_labels     f104_n2_analyze.py:16-24 through ga_analyze.py:36-41 (fewer than 22 frames -> unsafe)
  belly_flag       ov_eval_index.py:77-88 (clearance < -0.05 m for more than 1.0 s in a row, i.e. 21 frames)
  tracker_metrics  gb_track_analyze.py:53-104 (unsafe = status substring; a timeout is F)
  mission_labels   nav_runner.py:482-511 slide rule; nav_analyze.py:49 (unsafe = a slide or not complete)
Codes (traversing/results/README.md): S goal, safe; s goal with an event; U not reached, unsafe; F not reached, safe
(tracker timeouts only); '-' not run. Label kinds: rollback | rollback_belly | goal_belly | tracker | mission.
The schemas of the traversing/results CSVs and their row builders live with the tests (tests/.../evaluation/tables.py).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .config import DT, LABELS

S0 = 20
BACK_VX, EFFORT, MIN_VX = np.float32(-0.10), np.float32(0.3), np.float32(-0.30)   # the study's float32 compares
DRIVEN = ('goal_reached', 'timeout', 'rollover', 'terrain_bounds_exit', 'prolonged_blockage_terminated',
          'soil_breakthrough_terminated')                   # gb_track_analyze.KNOWN_STATUSES
MISSION = ('mission_complete', 'rollover', 'terrain_bounds_exit', 'prolonged_blockage_terminated', 'timeout',
           'mission_timeout', 'no_route', 'no_route_leg0', 'no_route_reroute')     # nav_runner.py:345-434
UNSAFE_SUBSTR = ('rollover', 'breakthrough', 'blockage', 'off_route', 'bounds_exit')    # gb_track_analyze.py:34
UNLABELLED = ('crash', 'launch_failed')     # no drive to label: ON_MISSING decides the cell
# The cell of an unlabelled drive per traversing/results table, '-' (left out) unless listed. ga_analyze.py:209 keeps only
# groups with every arm OF ONE ANALYSIS; the M2 columns come from overlapping analyses (A0A3, A5, s2, s4), so no
# per-column rule equals each of them: 'drop_pair' blanks the whole row, stricter than the study (no released drive
# crashed). ov_unseen_analyze.py:194-199 counts a twice-failed drive as not reached safely.
ON_MISSING = {'m2_shared_risk_soil': 'drop_pair', 'm2_shared_risk_rigid': 'drop_pair', 'm4_polaris_unseen_soil': 'U'}


def _code(fail, unsafe):
    return ('U' if unsafe else 'F') if fail else ('s' if unsafe else 'S')


def _status(status, allowed):
    if status not in allowed:
        raise ValueError(f'status {status!r} is not one of {allowed}')


def _recorded(state, action):
    s, a = np.asarray(state), np.asarray(action)
    if s.dtype != np.float32 or a.dtype != np.float32:     # upcasting would move the float32 thresholds
        raise TypeError(f'state/action must be the recorded float32 arrays, got {s.dtype}/{a.dtype}')
    if s.ndim != 2 or s.shape[1] != 17 or a.shape != (len(s), 3):
        raise ValueError(f'state {s.shape} / action {a.shape}: expected (n, 17) / (n, 3)')
    return s, s[:, 0], a[:, 1]


def point_labels(state, action, status):
    """fail / unsafe / back_s / min_vx / max_tilt of a single-goal drive. From frame 20: back_s = 0.05 x frames with
    vx < -0.10 under throttle > 0.3; unsafe unless the goal was reached with back_s < 0.05 and min vx > -0.30."""
    s, vx, thr = _recorded(state, action)
    _status(status, DRIVEN)
    fail = status != 'goal_reached'
    if len(s) < 22:                                         # the ga_analyze.safe_labels guard
        return dict(fail=int(fail), unsafe=1, back_s=0.0, min_vx=0.0, max_tilt=0.0)
    vx, thr = vx[S0:], thr[S0:]
    back = ((vx < BACK_VX) & (thr > EFFORT)).sum() * DT
    clean = not fail and back < 0.05 and vx.min() > MIN_VX
    return dict(fail=int(fail), unsafe=int(not clean), back_s=float(back), min_vx=float(vx.min()),
                max_tilt=float(np.degrees(np.abs(s[S0:, 2:4])).max()))


def belly_flag(clearance):
    """1 if the lowest hull point stayed more than 0.05 m under the undisturbed surface for more than 1 s in a row."""
    deep = np.asarray(clearance).astype(float) < -0.05     # float64 compare, as recorded; NaN is not deep
    if deep.ndim != 1:
        raise ValueError(f'belly clearance must be 1-D, got shape {deep.shape}')
    edge = np.diff(np.r_[0, deep.astype(np.int8), 0])
    run = int((np.flatnonzero(edge == -1) - np.flatnonzero(edge == 1)).max(initial=0))
    return int(run * DT > 1.0)                              # 20 frames = 1.0 s is not a flag


def tracker_metrics(pose, state, action, reference_waypoints, desired_speed, status):
    """Cross-track of every reference waypoint to the polyline of the recorded frame-start poses (no terminal pose),
    capped at 6 m, 5 % Winsorised mean (np.percentile bounds); |vx - desired| from frame min(20, n-1); mean |diff|
    of the action over frames and channels. All in float64."""
    _status(status, DRIVEN)
    pose, vx, act = np.asarray(pose, float), np.asarray(state, float)[:, 0], np.asarray(action, float)
    st, n = np.asarray(reference_waypoints, float), len(pose)
    if not (n and pose.shape == (n, 3) and vx.shape == (n,) and act.shape == (n, 3) and st.ndim == 2 and len(st)
            and st.shape[1] == 2):                          # an empty reference would give nan, not a metric
        raise ValueError(f'tracker arrays: pose {pose.shape}, vx {vx.shape}, action {act.shape}, reference {st.shape}')
    xy = pose[:, :2]
    if n < 2:
        d = np.linalg.norm(st - xy[0], axis=-1)
    else:                                                   # point-to-segment, as station_xtrack
        a, ab = xy[:-1], xy[1:] - xy[:-1]
        ab2 = (ab ** 2).sum(1)
        pa = st[:, None, :] - a[None, :, :]
        t = (pa * ab[None, :, :]).sum(-1) / np.where(ab2 > 0, ab2, 1.0)
        t = np.clip(np.where(ab2[None, :] > 0, t, 0.0), 0.0, 1.0)
        d = np.linalg.norm(pa - t[..., None] * ab[None, :, :], axis=-1).min(1)
    xt = np.minimum(d, 6.0)
    xt = np.clip(xt, np.percentile(xt, 5.0), np.percentile(xt, 95.0))
    vdes = np.asarray(desired_speed, float)[:n]
    k0, m = min(S0, n - 1), min(n, len(vdes))
    serr = np.abs(vx[k0:m] - vdes[k0:m])
    da = np.abs(np.diff(act, axis=0))
    fail, unsafe = int(status != 'goal_reached'), int(any(s in status for s in UNSAFE_SUBSTR))
    return dict(status=status, completed=1 - fail, unsafe=unsafe, code=_code(fail, unsafe),
                xtrack_station_winsor_mean_m=float(xt.mean()),
                speed_abs_err_mean_mps=float(serr.mean()) if serr.size else float('nan'),
                mean_abs_action_change=float(da.mean()) if da.size else float('nan'))


def mission_labels(state, action, status, goals_reached, n_goals):
    """slide = frame >= 20 and ((vx < -0.10 and throttle > 0.3) or vx < -0.30), all strict (at exactly f32(-0.30) the
    point label's min rule disagrees); elapsed_s = frames x 0.05, a Python float product (settle excluded)."""
    s, vx, thr = _recorded(state, action)
    _status(status, MISSION)
    if not 0 <= goals_reached <= n_goals or n_goals < 1 or (goals_reached == n_goals) != (status == 'mission_complete'):
        raise ValueError(f'{status} with {goals_reached}/{n_goals} waypoints')
    back = (vx < BACK_VX) & (thr > EFFORT)
    slide = int(((back | (vx < MIN_VX)) & (np.arange(len(vx)) >= S0)).any())
    fail = int(status != 'mission_complete')
    return dict(status=status, waypoints_total=int(n_goals), waypoints_reached=int(goals_reached), backward_slide=slide,
                back_s=float(back[S0:].sum() * DT), elapsed_s=len(vx) * DT, code=_code(fail, fail or slide))


def load_drive(folder, label):
    """What `label` reads from a run folder. A class run: record.json (a mission record also carries goals_reached and
    n_goals) + trajectory.npz (+ route.json for the tracker). A released drive: outcome.json or mission_outcome.json +
    trajectory.npz (+ command_reference.npz for the tracker). Belly labels: vehicle_extra.npz. Missing files raise.
    mission_outcome_sha256 is the sha256 of the outcome file read (record.json for a class run)."""
    p = Path(folder)
    ours = (p / 'record.json').exists()
    name = 'record.json' if ours else 'mission_outcome.json' if label == 'mission' else 'outcome.json'
    raw = (p / name).read_bytes()
    o = json.loads(raw)
    d = dict(status=o['status'], positive_work_kj=o.get('positive_work_kj'))
    if label == 'mission':
        d.update(goals_reached=o['goals_reached'], n_goals=o['n_goals'],
                 mission_outcome_sha256=hashlib.sha256(raw).hexdigest())
    elif ours:
        d['near_stop_fired'] = o['near_stop_fired']
    else:                                                   # released: rule off -> no near_stop_rule block
        rule = o.get('near_stop_rule') or (o.get('ext') or {}).get('near_stop_rule')
        d['near_stop_fired'] = None if rule is None else bool(rule['fired'])
    if d['status'] in UNLABELLED or (d['status'] == 'no_route' and label != 'mission'):
        return d                                            # never driven: no arrays
    with np.load(p / 'trajectory.npz') as z:
        d.update(state=z['state'], action=z['action'], pose=z['pose'])
        if label == 'tracker' and ours:
            d.update(desired_speed_mps=z['desired_speed_mps'],
                     reference_waypoints=json.loads((p / 'route.json').read_text())['waypoints'])
    if label == 'tracker' and not ours:
        with np.load(p / 'command_reference.npz') as c:
            d.update(desired_speed_mps=c['desired_speed_mps'], reference_waypoints=c['reference_waypoints'])
    if label.endswith('_belly'):
        with np.load(p / 'vehicle_extra.npz') as v:
            d['belly_clearance_min_m'] = v['belly_clearance_min_m']
    return d


def drive_labels(src, label):
    """Every number of one drive under `label` plus 'label' and 'code'. `src`: a run folder, a mapping with
    load_drive's keys, or a Record (its fields, arrays and extras' arrays)."""
    if label not in LABELS:
        raise ValueError(f'label {label!r} is not one of {LABELS}')
    d = load_drive(src, label) if isinstance(src, (str, Path)) else src if isinstance(src, Mapping) else \
        {**vars(src), **src.arrays, **{k: v for x in src.extras.values() for k, v in x.items()}}
    s = d['status']
    if s in UNLABELLED or (s == 'no_route' and label == 'tracker'):
        raise ValueError(f'status {s!r} carries no {label} label (ON_MISSING decides the cell)')
    if s == 'no_route' and label != 'mission':             # the planner found no valid route: never driven
        return dict(label=label, status=s, fail=1, unsafe=1, code='U')
    if label == 'mission':                                  # a missing input is a KeyError naming it
        out = mission_labels(d['state'], d['action'], s, d['goals_reached'], d['n_goals'])
        out['mission_outcome_sha256'] = d['mission_outcome_sha256']
    elif label == 'tracker':
        out = tracker_metrics(d['pose'], d['state'], d['action'], d['reference_waypoints'], d['desired_speed_mps'], s)
        ns = d['near_stop_fired']
        out.update(positive_work_kj=d['positive_work_kj'], near_stop_40s_fired='na' if ns is None else int(ns))
    elif label == 'goal_belly':                             # smoke: goal + belly only, rolling back not computed
        _status(s, DRIVEN)
        out = dict(status=s, fail=int(s != 'goal_reached'), belly_flag=belly_flag(d['belly_clearance_min_m']))
        out['code'] = _code(out['fail'], out['fail'] or out['belly_flag'])
    else:
        out = dict(status=s, **point_labels(d['state'], d['action'], s))
        if label == 'rollback_belly':                       # ov_eval_index.py:153-154: unsafe_belly = unsafe or flag
            out['belly_flag'] = belly_flag(d['belly_clearance_min_m'])
        out['code'] = _code(out['fail'], out['unsafe'] or out.get('belly_flag', 0))
    return dict(out, label=label)
