#!/usr/bin/env python
"""A5 pass 1 (PLAN A5, generalist_20260921): the common 3 s approach route per suite group, the CRM pass-1 task
rows, and the analysis of the recorded pass-1 state at frame 60.

Stages (``--stage``):

  routes      For each of the 800 suite groups build two candidates: (1) the straight start-goal line at a constant
              3 m/s (waypoints every 0.5 m, constant heading, stations from the geometry, last waypoint exactly the
              goal) and (2) the group's ``route_00`` with every speed replaced by 3 m/s.  Both are checked with the
              reference contract (``nedm.traverse.fdm_diverse_planner.check_reference_contract``) and the planner's
              validator (``gen_planner.safe_validate`` with ``gen_planner.CFG``, anchored at the layout pose), the way
              ``gc_control`` does, plus the collectors' start/goal 0.25 m rule.  The chosen approach is the candidate
              with the lower mean |grade| over its first 12 m (heights from the Chrono-frame v2 grid via
              ``gb_crop.sample_height``, grade from height differences at 0.5 m steps along the route); the straight
              line wins ties and ``route_00`` is the fallback when the straight line fails validation.  Writes
              ``a5/approach/<group>.json`` (collector route format, meta {approach, mean_grade_12m_deg,
              max_grade_12m_deg, ...}) and ``a5/approach_index.json`` with the per-group choice, both candidates'
              grades and validation verdicts, and a histogram summary (share of groups whose 12 m max |grade| exceeds
              12 and 17 degrees, overall and per stratum).

  tasks-crm   CRM pass-1 task rows {id: <g>__pass1, group, case, route, run, tier, episode_seed, extra:
              ['--horizon-s', '3']} with paths relative to CRM_ROOT (the suite lives under generalist/suite/, the
              approach routes under generalist/a5/approach/) -> ``a5/tasks_pass1_crm.json``; prints the rsync
              commands that would ship the routes (they are NOT run).  ``crm_worker.py`` appends ``extra`` after its
              own ``--horizon-s 120``, so the later value (3 s = 60 frames) wins in argparse.

  analyze     ``--runs DIR [DIR ...]``: from every run dir with a ``trajectory.npz`` compute the frame-60 state
              (vx, pose; row 60 when the recording has >= 61 rows, else the collector's ``terminal_state`` /
              ``terminal_pose`` which is the state at the start of interval 60 of a 60-frame recording), the CRM
              sinkage increase from ``crm_extra.npz`` when present (per wheel: tyre radius minus the spindle height
              above the arena TerrainMap at the reconstructed spindle xy, exactly the collector's own
              ``max_wheel_sinkage`` formula; increase = frame k minus frame 0, k = min(60, last recorded row)) and
              the moving criterion (vx > 1 m/s and sinkage increase < 0.1 m).  Writes ``pass1_state.json`` and
              ``poses.json`` in the ``ga_planner.py --poses`` format ({group: {run, frame: 60}} for recordings with
              a row 60; recordings with exactly 60 rows get {pose, history: <npz>, frame: 60} with the (40, 15)
              window built the ``ga_planner.history_from_trajectory`` way from the rows plus the terminal state,
              because ``ga_planner`` indexes ``pose[60]`` directly).  Several ``--runs`` dirs (one per world, named
              by ``--worlds``) give ``poses_<world>.json`` each and an analysis set = moving in every world.

  selftest-analyze  Runs ``analyze`` on 5 recorded CRM episodes (``crm_f104_v1/collect_v1/runs``, distinct groups)
              as pass-1 stand-ins, and on 60-row truncated copies of the same episodes, and checks the two agree
              (pose, vx, history window against ``ga_planner.history_from_trajectory``, ``ga_planner.load_history``
              reads the written npz).  Results in ``a5/selftest/analyze/RESULTS.json``.

Run from the repo root: ``PYTHONPATH=src:scripts CUDA_VISIBLE_DEVICES= python scripts/ga_approach.py --stage ...``.
Nothing here submits cluster jobs or runs Chrono.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, shutil, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(HERE))

K = ROOT / 'artifacts/traverse/generalist_20260921'
SUITE = K / 'A_adapt/suite'
A5 = K / 'A_adapt/a5'
GRID = ROOT / 'artifacts/traverse/crm_f104_v1/grids/arena_f104_50h_v1/grid.npz'
ARENA = ROOT / 'assets/traverse/arena_f104_50h_v1'
STANDIN_RUNS = ROOT / 'artifacts/traverse/crm_f104_v1/collect_v1/runs'
CRM_ROOT = '/work1/dannegrut/harry/experiments/crm_f104_20260916'
SUITE_REL, A5_REL = 'generalist/suite', 'generalist/a5'          # under CRM_ROOT (NOTES_ga_suite shipping block)

APPROACH_SPEED_MPS = 3.0
STEP_M = 0.5
GRADE_LEN_M = 12.0
FRAME = 60                       # the pass-1 decision frame (3 s at 50 ms)
DT = 0.05
VX_MOVING_MPS = 1.0
SINK_INC_MAX_M = 0.1
HIST_STATE_COLS = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]      # ga_planner.HIST_STATE_COLS (PLAN conventions)
HIST_T, HIST_DIM = 40, 15
# Spindle centre in the chassis reference frame: HMMWV_Full suspension locations (+-1.688965, 0, 0) plus the
# double-wishbone spindle offsets (front -0.040, rear +0.036; +-0.910 lateral; -0.026 vertical), from
# share/chrono/data/vehicle/hmmwv/{vehicle/HMMWV_Vehicle.json, suspension/HMMWV_DoubleWishbone{Front,Rear}.json}.
WHEEL_OFFSETS = {'fl': (1.688965 - 0.040, 0.910, -0.026), 'fr': (1.688965 - 0.040, -0.910, -0.026),
                 'rl': (-1.688965 + 0.036, 0.910, -0.026), 'rr': (-1.688965 + 0.036, -0.910, -0.026)}


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=_jsonable)


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    raise TypeError(f'not JSON serialisable: {type(v).__name__}')


def load_suite():
    d = json.load(open(SUITE / 'suite.json'))
    assert d['n_groups'] == len(d['groups']) == 800, d['n_groups']
    return d


def load_case(g):
    case = json.load(open(SUITE / 'cases' / f'{g}.json'))
    lay = case['layout']
    pose = np.array([lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']], float)
    goal = np.asarray(case['goal_xy'], float)
    rp = SUITE / 'cases' / 'routes' / g / 'route_00.json'
    r00 = json.load(open(rp))
    return case, pose, goal, r00, rp


# ------------------------------------------------------------------------------------------------------ routes
def straight_route(start_xy, goal_xy, v=APPROACH_SPEED_MPS, step_m=STEP_M):
    """Straight start->goal line: waypoints every ~0.5 m (last one exactly the goal), constant heading and speed."""
    s, g = np.asarray(start_xy, float), np.asarray(goal_xy, float)
    L = float(np.linalg.norm(g - s))
    n = max(3, int(math.ceil(L / step_m)) + 1)
    xy = s[None] + (g - s)[None] * np.linspace(0., 1., n)[:, None]
    xy[0], xy[-1] = s, g
    st = np.r_[0., np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
    hd = np.full(n, math.atan2(g[1] - s[1], g[0] - s[0]))
    return {'waypoints': xy, 'speeds': np.full(n, float(v)), 'stations': st, 'headings': hd, 'meta': {}}


def route00_at_speed(r00, v=APPROACH_SPEED_MPS):
    r = {k: np.asarray(r00[k], float) for k in ('waypoints', 'speeds', 'stations', 'headings')}
    r['speeds'] = np.full(len(r['waypoints']), float(v))
    r['meta'] = {}
    return r


def validate_route(route, pose, goal):
    """Reasons the collectors or the planner would reject the route (empty list = valid).

    Contract: ``fdm_diverse_planner.check_reference_contract`` (the original; ``gc_control`` carries a copy).
    Validator: ``gen_planner.safe_validate(route, [], gen_planner.CFG, anchor=pose)`` (curvature 0.125 1/m, speeds in
    [0, 6], accel 1.5 / decel 2.0 m/s^2, swept footprint inside +-40 m).  Collector rule: first waypoint within 0.25 m
    of the start, last within 0.25 m of the goal (``crm_collect.py`` / ``traverse_fdm_rgbd_diverse_chrono.py``).
    """
    from nedm.traverse.fdm_diverse_planner import check_reference_contract
    import gen_planner as GP
    reasons = []
    try:
        check_reference_contract(route)
    except ValueError as e:
        reasons.append(f'contract: {e}')
    v = GP.safe_validate(route, [], GP.CFG, np.asarray(pose, float))
    reasons += [f'validator: {r}' for r in v.get('reasons', [])] if not v['valid'] else []
    xy = np.asarray(route['waypoints'], float)
    if np.linalg.norm(xy[0] - np.asarray(pose, float)[:2]) > .25:
        reasons.append('collector: first waypoint > 0.25 m from the start')
    if np.linalg.norm(xy[-1] - np.asarray(goal, float)) > .25:
        reasons.append('collector: last waypoint > 0.25 m from the goal')
    return reasons


def gc_agrees(route, pose, reasons):
    """The same verdict from gc_control's torch-free copies (contract + planner_validator), as its self-test asserts."""
    import gc_control as GC
    ok_gc = True
    try:
        GC.check_reference_contract(route)
    except ValueError:
        ok_gc = False
    ok_gc = ok_gc and GC.planner_validator(route, pose)
    ok_here = not any(r.startswith(('contract', 'validator')) for r in reasons)
    return ok_gc == ok_here


def grade_profile(grid, route, length_m=GRADE_LEN_M, step_m=STEP_M):
    """Grades (deg, uphill positive) at 0.5 m steps along the first 12 m of the route from v2-grid heights."""
    import gb_crop
    st = np.asarray(route['stations'], float); wp = np.asarray(route['waypoints'], float)
    s = np.arange(0., min(length_m, float(st[-1])) + 1e-9, step_m)
    xy = np.stack([np.interp(s, st, wp[:, 0]), np.interp(s, st, wp[:, 1])], -1)
    h = gb_crop.sample_height(grid, xy)
    g = np.degrees(np.arctan2(np.diff(h), np.diff(s)))
    return dict(mean_abs_deg=float(np.abs(g).mean()), max_abs_deg=float(np.abs(g).max()), mean_signed_deg=float(g.mean()),
                rise_m=float(h[-1] - h[0]), n_segments=int(len(g)), grades_deg=[round(float(x), 3) for x in g])


def stage_routes(a):
    import gb_crop, gc_control as GC
    grid = gb_crop.load_grid(str(a.grid))
    suite = load_suite()
    out_dir = a.a5 / 'approach'; out_dir.mkdir(parents=True, exist_ok=True)
    idx, n_gc_disagree = {}, 0
    t0 = time.time()
    for e in suite['groups']:
        g = e['group']
        case, pose, goal, r00, rp = load_case(g)
        assert case['id'] == g
        cands = {'straight': straight_route(pose[:2], goal), 'route00': route00_at_speed(r00)}
        info = {}
        for name, r in cands.items():
            reasons = validate_route(r, pose, goal)
            if not gc_agrees(r, pose, reasons):
                n_gc_disagree += 1
            gp = grade_profile(grid, r)
            info[name] = dict(valid=not reasons, reasons=reasons, n_waypoints=int(len(r['waypoints'])),
                              length_m=float(r['stations'][-1]), **gp)
        # geometry check: the straight line vs route_00's polyline (max lateral distance of route_00 from the line)
        u = (goal - pose[:2]) / np.linalg.norm(goal - pose[:2]); nrm = np.array([-u[1], u[0]])
        lat = float(np.abs((np.asarray(r00['waypoints'], float) - pose[:2]) @ nrm).max())
        yaw_line = math.atan2(*(goal - pose[:2])[::-1])
        dyaw = math.degrees(math.atan2(math.sin(yaw_line - pose[2]), math.cos(yaw_line - pose[2])))
        if info['straight']['valid'] and (not info['route00']['valid']
                                          or info['straight']['mean_abs_deg'] <= info['route00']['mean_abs_deg'] + 1e-9):
            choice, why = 'straight', ('route00_invalid' if not info['route00']['valid'] else
                                       'tie' if abs(info['straight']['mean_abs_deg'] - info['route00']['mean_abs_deg']) <= 1e-9 else 'lower_grade')
        elif info['straight']['valid']:
            choice, why = 'route00', 'lower_grade'
        else:
            choice, why = 'route00', ('straight_invalid' if info['route00']['valid'] else 'both_invalid_fallback')
        r = cands[choice]
        r['meta'] = dict(candidate='a5_pass1_approach', family='ga_approach_v1', approach=choice, choice_reason=why,
                         mean_grade_12m_deg=info[choice]['mean_abs_deg'], max_grade_12m_deg=info[choice]['max_abs_deg'],
                         mean_signed_grade_12m_deg=info[choice]['mean_signed_deg'], grade_length_m=GRADE_LEN_M,
                         cruise_speed_mps=APPROACH_SPEED_MPS, scene_id=g, group=g, stratum=e['stratum'],
                         evaluation_stratum=e['evaluation_stratum'], straight_valid=info['straight']['valid'],
                         route00_valid=info['route00']['valid'], route00_file_sha256=sha256_file(rp),
                         route00_max_lateral_from_line_m=lat, start_yaw_minus_line_heading_deg=dyaw,
                         grid=os.path.relpath(grid['path'], ROOT), horizon_s=FRAME * DT)
        rj = GC.route_to_json(r)
        rj['meta']['route_sha256'] = GC.route_sha256(r)
        jdump(rj, out_dir / f'{g}.json')
        idx[g] = dict(approach=choice, choice_reason=why, mean_grade_12m_deg=info[choice]['mean_abs_deg'],
                      max_grade_12m_deg=info[choice]['max_abs_deg'], mean_signed_grade_12m_deg=info[choice]['mean_signed_deg'],
                      stratum=e['stratum'], evaluation_stratum=e['evaluation_stratum'], tier=int(suite['groups'].index(e)),
                      route_sha256=rj['meta']['route_sha256'], route00_max_lateral_from_line_m=lat,
                      start_yaw_minus_line_heading_deg=dyaw, straight=info['straight'], route00=info['route00'])
    summ = summarise(idx)
    summ['gc_control_verdict_disagreements'] = n_gc_disagree
    summ['wall_s'] = round(time.time() - t0, 1)
    index = dict(schema=1, created=now(), stage='routes', n_groups=len(idx), grid=os.path.relpath(grid['path'], ROOT),
                 grid_sha256=sha256_file(grid['path']), suite_sha256=sha256_file(SUITE / 'suite.json'),
                 approach_speed_mps=APPROACH_SPEED_MPS, step_m=STEP_M, grade_length_m=GRADE_LEN_M,
                 rule='straight if valid and mean|grade| over the first 12 m <= route_00 at 3 m/s (ties -> straight); '
                      'else route_00 at 3 m/s (also when the straight line fails validation)',
                 summary=summ, groups=idx)
    jdump(index, a.a5 / 'approach_index.json')
    print(json.dumps({k: v for k, v in index.items() if k != 'groups'}, indent=1))
    print(f'wrote {len(idx)} routes to {out_dir} and {a.a5 / "approach_index.json"}')


def summarise(idx):
    def share(vals, thr):
        vals = np.asarray(vals, float)
        return dict(n=int((vals > thr).sum()), of=int(len(vals)), share=float((vals > thr).mean()) if len(vals) else None)
    out = dict(choice_counts={c: sum(1 for v in idx.values() if v['approach'] == c) for c in ('straight', 'route00')},
               choice_reasons={}, straight_invalid=sum(1 for v in idx.values() if not v['straight']['valid']),
               route00_invalid=sum(1 for v in idx.values() if not v['route00']['valid']),
               invalid_reasons={}, geometry_identical_groups=sum(1 for v in idx.values() if v['route00_max_lateral_from_line_m'] < 0.01),
               max_start_yaw_minus_line_heading_deg=float(max(abs(v['start_yaw_minus_line_heading_deg']) for v in idx.values())))
    for v in idx.values():
        out['choice_reasons'][v['choice_reason']] = out['choice_reasons'].get(v['choice_reason'], 0) + 1
        for c in ('straight', 'route00'):
            for r in v[c]['reasons']:
                out['invalid_reasons'][f'{c}: {r}'] = out['invalid_reasons'].get(f'{c}: {r}', 0) + 1
    bins = [0, 5, 10, 12, 17, 25, 90]
    for label, sel in (('all', lambda v: True), ('fresh', lambda v: v['stratum'] == 'fresh'), ('reused', lambda v: v['stratum'] == 'reused')):
        vs = [v for v in idx.values() if sel(v)]
        for which, key in (('chosen', None), ('straight', 'straight'), ('route00', 'route00')):
            mx = [v['max_grade_12m_deg'] if key is None else v[key]['max_abs_deg'] for v in vs]
            mn = [v['mean_grade_12m_deg'] if key is None else v[key]['mean_abs_deg'] for v in vs]
            sg = [v['mean_signed_grade_12m_deg'] if key is None else v[key]['mean_signed_deg'] for v in vs]
            out[f'{label}/{which}'] = dict(max_abs_gt_12deg=share(mx, 12.), max_abs_gt_17deg=share(mx, 17.),
                                           mean_uphill_gt_8deg=share(sg, 8.), mean_abs_gt_8deg=share(mn, 8.),
                                           max_abs_hist={f'{bins[i]}-{bins[i+1]}': int(((np.asarray(mx) >= bins[i]) & (np.asarray(mx) < bins[i+1])).sum()) for i in range(len(bins) - 1)},
                                           mean_abs_deg_p50=float(np.median(mn)) if mn else None, max_abs_deg_p50=float(np.median(mx)) if mx else None)
    out['plan_review_reference_reused_straight_at_start_yaw'] = dict(max_abs_gt_12deg='80/200', max_abs_gt_17deg='53/200', mean_uphill_gt_8deg='31/200')
    return out


# --------------------------------------------------------------------------------------------------- tasks-crm
def stage_tasks_crm(a):
    import gc_control as GC
    suite = load_suite()
    index = json.load(open(a.a5 / 'approach_index.json'))
    rows = []
    for gi, e in enumerate(suite['groups']):
        g = e['group']; rid = f'{g}__pass1'
        rp = a.a5 / 'approach' / f'{g}.json'
        assert rp.exists(), rp
        rj = json.load(open(rp))
        sha = GC.route_sha256(rj)
        assert sha == index['groups'][g]['route_sha256'] == rj['meta']['route_sha256'], g
        assert (SUITE / 'cases' / f'{g}.json').exists()
        assert abs(float(a.horizon_s) / DT - round(float(a.horizon_s) / DT)) < 1e-9, 'horizon must be a 50 ms multiple'
        rows.append(dict(id=rid, group=g, case=f'{SUITE_REL}/cases/{g}.json', route=f'{A5_REL}/approach/{g}.json', run=True,
                         tier=gi, episode_seed=int(hashlib.md5(rid.encode()).hexdigest()[:8], 16),
                         extra=['--horizon-s', f'{float(a.horizon_s):g}'], sha256=sha, arms=['pass1'], aliases=[],
                         stratum=e['stratum'], evaluation_stratum=e['evaluation_stratum'], approach=rj['meta']['approach'],
                         world='crm'))
    assert len({r['id'] for r in rows}) == len(rows) == 800
    jdump(rows, a.a5 / 'tasks_pass1_crm.json')
    frames = int(round(float(a.horizon_s) / DT))
    print(f'wrote {len(rows)} rows to {a.a5 / "tasks_pass1_crm.json"} (horizon {float(a.horizon_s):g} s = {frames} frames, '
          f'{sum(r["approach"] == "straight" for r in rows)} straight / {sum(r["approach"] == "route00" for r in rows)} route00)')
    ship = [f'# ship the A5 pass-1 approach routes and task rows to the CRM root (NOT run by ga_approach.py)',
            f'C={CRM_ROOT}', f'A5={a.a5}',
            f'ssh amd "mkdir -p $C/{A5_REL}/approach"',
            f'rsync -az $A5/approach/ amd:$C/{A5_REL}/approach/',
            f'rsync -az $A5/tasks_pass1_crm.json $A5/approach_index.json amd:$C/{A5_REL}/',
            f'# the suite cases must already be at $C/{SUITE_REL}/cases/ (NOTES_ga_suite shipping block); check:',
            f'ssh amd "ls $C/{SUITE_REL}/cases | wc -l; ls $C/{A5_REL}/approach | wc -l"',
            f'# launch (unmodified collector, one sbatch per GPU partition; not run here):',
            f'# bash $C/source/scripts/crm_launch.sh $C/{A5_REL}/tasks_pass1_crm.json $C/{A5_REL}/pass1_out configs/crm_main.json 2']
    print('\n'.join(ship))
    (a.a5 / 'SHIP_pass1_crm.txt').write_text('\n'.join(ship) + '\n')


# ----------------------------------------------------------------------------------------------------- analyze
def history_window(state, action, k, T=HIST_T):
    """``ga_planner.history_from_trajectory`` on plain arrays: row t (j = k-T+1+t) = [state[j, 12 cols] | action[j-1]],
    valid iff 1 <= j <= min(k, len(state)-1)."""
    st = np.asarray(state, np.float32); ac = np.asarray(action, np.float32); n = len(st)
    hist = np.zeros((T, HIST_DIM), np.float32); mask = np.zeros(T, bool)
    for t in range(T):
        j = k - T + 1 + t
        if 1 <= j <= min(k, n - 1):
            hist[t, :12] = st[j, HIST_STATE_COLS]; hist[t, 12:] = ac[j - 1]; mask[t] = True
    return hist, mask


def quat_rotate_xy(q, v):
    """World xy of a chassis-frame vector v under the chassis quaternion q (n, 4) (e0 scalar first, Chrono order)."""
    e0, e1, e2, e3 = (q[:, i].astype(np.float64) for i in range(4))
    x, y, z = v
    r00 = 1 - 2 * (e2 * e2 + e3 * e3); r01 = 2 * (e1 * e2 - e0 * e3); r02 = 2 * (e1 * e3 + e0 * e2)
    r10 = 2 * (e1 * e2 + e0 * e3); r11 = 1 - 2 * (e1 * e1 + e3 * e3); r12 = 2 * (e2 * e3 - e0 * e1)
    return np.stack([r00 * x + r01 * y + r02 * z, r10 * x + r11 * y + r12 * z], -1)


def wheel_sinkage(extra, pose, tm):
    """(n, 4) tyre radius minus spindle height above the TerrainMap at the reconstructed spindle xy (the collector's
    ``max_wheel_sinkage`` formula, crm_collect.py:285-286, with the spindle xy from pose + quaternion + offsets)."""
    order = [str(w).replace('tire_', '') for w in extra['wheel_order']]
    q = np.asarray(extra['quat'], np.float64); n = len(q)
    out = np.zeros((n, 4))
    for i, w in enumerate(order):
        xy = np.asarray(pose[:n, :2], np.float64) + quat_rotate_xy(q, WHEEL_OFFSETS[w])
        gz = np.asarray(tm.height(xy[:, 0], xy[:, 1]), np.float64)
        out[:, i] = float(extra['tire_radius_m'][i]) - (np.asarray(extra['spindle_z_m'][:, i], np.float64) - gz)
    return out


def analyze_run(run_dir, tm, k=FRAME):
    run_dir = Path(run_dir)
    rec = dict(run=str(run_dir.resolve()), moving=False, reasons=[])
    tp = run_dir / 'trajectory.npz'
    if not tp.exists():
        rec['reasons'].append('missing_trajectory'); return None, rec
    traj = np.load(tp, allow_pickle=True)
    outc = json.load(open(run_dir / 'outcome.json')) if (run_dir / 'outcome.json').exists() else {}
    group = outc.get('case_id') or (json.load(open(run_dir / 'case.json'))['id'] if (run_dir / 'case.json').exists()
                                    else run_dir.name.split('__')[0])
    st, ac, po = np.asarray(traj['state'], np.float32), np.asarray(traj['action'], np.float32), np.asarray(traj['pose'], np.float64)
    n = len(po)
    rec.update(group=group, n_frames=int(n), status=outc.get('status'), elapsed_s=outc.get('elapsed_s'), frame=k)
    if n > k:
        s_k, p_k, src = st[k], po[k], 'row'
        hist, hmask = history_window(st, ac, k)
    elif n == k and 'terminal_state' in traj.files and 'terminal_pose' in traj.files:
        s_k, p_k, src = np.asarray(traj['terminal_state'], np.float32), np.asarray(traj['terminal_pose'], np.float64), 'terminal'
        hist, hmask = history_window(np.vstack([st, s_k[None]]), ac, k)
    else:
        rec['frame_source'] = 'absent'; rec['reasons'].append(f'short_recording_{n}_rows')
        return group, rec
    rec.update(frame_source=src, vx=float(s_k[0]), vy=float(s_k[1]), yaw_rate=float(s_k[6]), roll=float(s_k[2]), pitch=float(s_k[3]),
               pose=[float(v) for v in p_k], hist_valid_steps=int(hmask.sum()),
               displacement_from_start_m=float(np.linalg.norm(p_k[:2] - po[0, :2])))
    ep = run_dir / 'crm_extra.npz'
    if ep.exists():
        extra = np.load(ep, allow_pickle=True)
        ke = min(k, len(extra['quat']) - 1)
        s = wheel_sinkage(extra, po, tm)
        proxy = (np.asarray(extra['tire_radius_m'])[None, :] - (np.asarray(extra['spindle_z_m']) - np.asarray(extra['bmp_ground_z_m'])[:, None])).mean(1)
        rec.update(sink_frame=int(ke), sink_inc_max_m=float((s[ke] - s[0]).max()), sink_inc_mean_m=float((s[ke] - s[0]).mean()),
                   sink_inc_per_wheel_m=[float(v) for v in s[ke] - s[0]], sink_frame0_per_wheel_m=[float(v) for v in s[0]],
                   sink_proxy_chassis_inc_m=float(proxy[ke] - proxy[0]), sink_source='crm_extra')
        sinking = rec['sink_inc_max_m'] >= SINK_INC_MAX_M
    else:
        rec.update(sink_source=None); sinking = False
    slow = rec['vx'] <= VX_MOVING_MPS
    if slow:
        rec['reasons'].append(f'vx_{rec["vx"]:.2f}_le_{VX_MOVING_MPS:g}')
    if sinking:
        rec['reasons'].append(f'sinkage_increase_{rec["sink_inc_max_m"]:.3f}_ge_{SINK_INC_MAX_M:g}')
    rec['moving'] = not slow and not sinking
    return group, (rec, hist, hmask)


def stage_analyze(a, quiet=False):
    from nedm.traverse.terrain import TerrainMap
    tm = TerrainMap.from_dir(Path(a.arena))
    worlds = a.worlds or ['crm', 'rigid', 'w3', 'w4'][:len(a.runs)]
    assert len(worlds) == len(a.runs), (worlds, a.runs)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    state = dict(schema=1, created=now(), stage='analyze', frame=FRAME, arena=os.path.relpath(Path(a.arena).resolve(), ROOT),
                 criterion=dict(vx_gt_mps=VX_MOVING_MPS, sinkage_increase_lt_m=SINK_INC_MAX_M,
                                sinkage='max over wheels of [tyre radius - (spindle_z - TerrainMap height at the spindle xy)] '
                                        'at frame min(60, last row) minus frame 0; applied when crm_extra.npz exists'),
                 worlds={}, analysis_set=[], excluded={})
    per_world_moving = []
    for world, rdir in zip(worlds, a.runs):
        rdir = Path(rdir)
        dirs = sorted(p for p in rdir.iterdir() if p.is_dir() and (p / 'trajectory.npz').exists()) if rdir.is_dir() else []
        if a.limit:
            dirs = dirs[:a.limit]
        groups, dup, hist_dir = {}, [], out / f'hist_{world}'
        poses = {}
        for d in dirs:
            g, res = analyze_run(d, tm)
            if g is None:
                continue
            if g in groups:
                dup.append(str(d)); continue
            if isinstance(res, tuple):
                rec, hist, hmask = res
            else:
                rec = res
            groups[g] = rec
            if rec['moving']:
                if rec['frame_source'] == 'row':
                    poses[g] = dict(run=rec['run'], frame=FRAME)
                else:  # 60-row recording: ga_planner indexes pose[60], so give the pose and the window explicitly
                    hist_dir.mkdir(parents=True, exist_ok=True)
                    hp = hist_dir / f'{g}.npz'
                    np.savez_compressed(hp, hist=hist, hmask=hmask, group=np.asarray(g), frame=np.int32(FRAME), source=np.asarray('rows+terminal'))
                    poses[g] = dict(pose=rec['pose'], history=str(hp.resolve()), frame=FRAME, pass1_run=rec['run'], pass1_source='terminal')
        n_mov = sum(r['moving'] for r in groups.values())
        reasons = {}
        for r in groups.values():
            for x in r['reasons']:
                key = x.split('_')[0] + ('_' + x.split('_')[1] if x.startswith('sinkage') or x.startswith('short') else '')
                reasons[key] = reasons.get(key, 0) + 1
        state['worlds'][world] = dict(runs_dir=str(rdir.resolve()), n_runs=len(dirs), n_groups=len(groups), n_moving=n_mov,
                                      n_excluded=len(groups) - n_mov, exclusion_reasons=reasons, duplicate_group_runs=dup,
                                      frame_source_counts={s: sum(1 for r in groups.values() if r.get('frame_source') == s) for s in ('row', 'terminal', 'absent')},
                                      vx_p50=float(np.median([r['vx'] for r in groups.values() if 'vx' in r])) if any('vx' in r for r in groups.values()) else None,
                                      groups=groups)
        per_world_moving.append({g for g, r in groups.items() if r['moving']})
        jdump(poses, out / f'poses_{world}.json')
        state['worlds'][world]['poses_file'] = str(out / f'poses_{world}.json')
    common = set.intersection(*per_world_moving) if per_world_moving else set()
    all_groups = set.union(*[set(state['worlds'][w]['groups']) for w in worlds]) if worlds else set()
    state['analysis_set'] = sorted(common)
    state['n_analysis_set'] = len(common)
    state['excluded'] = dict(n=len(all_groups - common), groups=sorted(all_groups - common))
    # poses.json = the first world's poses restricted to the analysis set (groups moving in every world given)
    first = json.load(open(out / f'poses_{worlds[0]}.json'))
    jdump({g: first[g] for g in sorted(common) if g in first}, out / 'poses.json')
    for w in worlds:
        pw = json.load(open(out / f'poses_{w}.json'))
        jdump({g: pw[g] for g in sorted(common) if g in pw}, out / f'poses_{w}.json')
    jdump(state, out / 'pass1_state.json')
    if not quiet:
        print(json.dumps({k: (v if k != 'worlds' else {w: {kk: vv for kk, vv in d.items() if kk != 'groups'} for w, d in v.items()})
                          for k, v in state.items() if k != 'excluded'}, indent=1))
        print(f'wrote {out / "pass1_state.json"} and {out / "poses.json"} ({len(common)} groups in the analysis set)')
    return state


# ---------------------------------------------------------------------------------------------- selftest-analyze
def stage_selftest_analyze(a):
    import ga_planner as GA
    sdir = a.a5 / 'selftest' / 'analyze'
    if sdir.exists():
        shutil.rmtree(sdir)
    full, cut = sdir / 'runs_full', sdir / 'runs_cut60'
    full.mkdir(parents=True); cut.mkdir(parents=True)
    chosen, seen = [], set()
    for d in sorted(STANDIN_RUNS.iterdir()):
        if len(chosen) >= 5:
            break
        if not (d / 'trajectory.npz').exists() or not (d / 'crm_extra.npz').exists() or not (d / 'outcome.json').exists():
            continue
        g = json.load(open(d / 'outcome.json')).get('case_id')
        if g in seen or len(np.load(d / 'trajectory.npz')['pose']) < FRAME + 1:
            continue
        seen.add(g); chosen.append(d)
    assert len(chosen) == 5, chosen
    for d in chosen:
        shutil.copytree(d, full / d.name)
        z = np.load(d / 'trajectory.npz', allow_pickle=True); e = np.load(d / 'crm_extra.npz', allow_pickle=True)
        c = cut / d.name; c.mkdir()
        shutil.copy(d / 'outcome.json', c / 'outcome.json')
        tz = {k: z[k] for k in z.files}
        tz.update(state=z['state'][:FRAME], action=z['action'][:FRAME], pose=z['pose'][:FRAME], terminal_state=z['state'][FRAME],
                  terminal_pose=z['pose'][FRAME], power_kw=z['power_kw'][:FRAME], parked=z['parked'][:FRAME])
        np.savez_compressed(c / 'trajectory.npz', **tz)
        ez = {k: (e[k][:FRAME] if e[k].ndim and len(e[k]) == len(z['pose']) else e[k]) for k in e.files}
        np.savez_compressed(c / 'crm_extra.npz', **ez)
    res = dict(created=now(), standins=[d.name for d in chosen], checks={})
    ns_full = argparse.Namespace(runs=[str(full)], worlds=['crm'], out=str(sdir / 'out_full'), arena=a.arena, limit=0, a5=a.a5)
    ns_cut = argparse.Namespace(runs=[str(cut)], worlds=['crm'], out=str(sdir / 'out_cut60'), arena=a.arena, limit=0, a5=a.a5)
    sf, sc = stage_analyze(ns_full, quiet=True), stage_analyze(ns_cut, quiet=True)
    gf, gc = sf['worlds']['crm']['groups'], sc['worlds']['crm']['groups']
    res['checks']['five_groups_each'] = (len(gf) == 5 and len(gc) == 5)
    res['checks']['frame_sources'] = dict(full=sf['worlds']['crm']['frame_source_counts'], cut=sc['worlds']['crm']['frame_source_counts'])
    res['checks']['moving_agree'] = all(gf[g]['moving'] == gc[g]['moving'] for g in gf)
    res['checks']['pose_max_abs_diff'] = float(max(np.abs(np.asarray(gf[g]['pose']) - np.asarray(gc[g]['pose'])).max() for g in gf))
    res['checks']['vx_max_abs_diff'] = float(max(abs(gf[g]['vx'] - gc[g]['vx']) for g in gf))
    res['checks']['sink_inc_max_abs_diff_m'] = float(max(abs(gf[g]['sink_inc_max_m'] - gc[g]['sink_inc_max_m']) for g in gf))
    res['checks']['sink_frames'] = {g: (gf[g]['sink_frame'], gc[g]['sink_frame']) for g in gf}
    # history windows: the cut copies' explicit npz vs ga_planner.history_from_trajectory on the full recording at 60
    pf = json.load(open(sdir / 'out_full' / 'poses.json')); pc = json.load(open(sdir / 'out_cut60' / 'poses.json'))
    res['checks']['poses_full_format'] = all(set(v) == {'run', 'frame'} and v['frame'] == FRAME for v in pf.values())
    res['checks']['poses_cut_format'] = all({'pose', 'history', 'frame'} <= set(v) and 'run' not in v for v in pc.values())
    res['checks']['poses_same_groups'] = sorted(pf) == sorted(pc)
    hd = []
    for g, v in pc.items():
        traj = np.load(Path(pf[g]['run']) / 'trajectory.npz')
        h_ref, m_ref = GA.history_from_trajectory(traj, FRAME, HIST_T)
        h_np, m_np, how = GA.load_history(v['history'], g, FRAME, HIST_T)
        hd.append(dict(group=g, max_abs_diff=float(np.abs(h_ref - h_np).max()), mask_equal=bool((m_ref == m_np).all()),
                       valid_steps=int(m_np.sum()), loader=how, pose_equal=bool(np.allclose(v['pose'], traj['pose'][FRAME]))))
    res['checks']['history_vs_ga_planner'] = hd
    res['checks']['history_ok'] = all(d['max_abs_diff'] == 0.0 and d['mask_equal'] and d['valid_steps'] == HIST_T and d['pose_equal'] for d in hd)
    # the per-run records for the report
    res['records'] = {g: {k: gf[g].get(k) for k in ('n_frames', 'status', 'frame_source', 'vx', 'sink_inc_max_m', 'sink_inc_mean_m', 'sink_proxy_chassis_inc_m', 'moving', 'reasons')} for g in gf}
    res['records_cut60'] = {g: {k: gc[g].get(k) for k in ('n_frames', 'frame_source', 'vx', 'sink_frame', 'sink_inc_max_m', 'moving', 'reasons')} for g in gc}
    res['all_ok'] = all(v for k, v in res['checks'].items() if isinstance(v, bool))
    jdump(res, sdir / 'RESULTS.json')
    print(json.dumps(res, indent=1))
    print('SELFTEST', 'OK' if res['all_ok'] else 'FAILED', '->', sdir / 'RESULTS.json')
    return res['all_ok']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['routes', 'tasks-crm', 'analyze', 'selftest-analyze'])
    ap.add_argument('--a5', type=Path, default=A5, help='output root (approach/, approach_index.json, tasks_pass1_crm.json)')
    ap.add_argument('--grid', default=str(GRID), help='Chrono-frame v2 grid for the route grades')
    ap.add_argument('--arena', default=str(ARENA), help='arena dir (TerrainMap) for the sinkage reconstruction')
    ap.add_argument('--horizon-s', default='3', help='pass-1 horizon written into extra (50 ms multiple)')
    ap.add_argument('--runs', nargs='+', help='analyze: dir(s) of pass-1 run dirs (one per world)')
    ap.add_argument('--worlds', nargs='*', help='analyze: world label per --runs dir (default crm, rigid)')
    ap.add_argument('--out', default=str(A5 / 'pass1'), help='analyze: output dir for pass1_state.json / poses.json')
    ap.add_argument('--limit', type=int, default=0, help='analyze: only the first N run dirs per world (0 = all)')
    a = ap.parse_args(argv)
    if a.stage == 'routes':
        stage_routes(a)
    elif a.stage == 'tasks-crm':
        stage_tasks_crm(a)
    elif a.stage == 'analyze':
        assert a.runs, '--runs is required'
        stage_analyze(a)
    elif a.stage == 'selftest-analyze':
        ok = stage_selftest_analyze(a)
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
