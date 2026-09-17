"""Read-only current-cohort physical support, including schema-safe goals."""
from pathlib import Path
from datetime import datetime, timezone
import collections
import hashlib
import json
import math
import sys

import numpy as np

base = Path('/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909')
root = base/'full_cohort_v2'
source = base/'snapshots/campaign_v2'
manifest_path = source/'artifacts/traverse/fdm_diverse_v1_20260909/cases/cases.json'
manifest = json.loads(manifest_path.read_text())
records = [r for r in manifest['records'] if r['split'] in ('train', 'val')]


def bounded_first(poses, actions, parked, goal, radius):
    n = len(actions)
    if n < 40:
        return None
    # Exact same 40 intervals / 41 measured endpoints as the training schema.
    # Axis-aligned window bounds only accelerate the pairwise-diameter test:
    # large axis span proves failure; a small box diagonal proves success.
    excluded = parked | np.maximum.accumulate(np.linalg.norm(poses[:, :2]-goal, axis=1) <= radius)
    effort = np.convolve((actions[:, 1] > .3).astype(np.int32), np.ones(40, np.int32), 'valid') == 40
    allowed = np.convolve((~excluded).astype(np.int32), np.ones(41, np.int32), 'valid') == 41
    windows = np.lib.stride_tricks.sliding_window_view(poses[:, :2], 41, axis=0)
    spans = windows.max(axis=-1)-windows.min(axis=-1)
    eligible = np.flatnonzero(effort & allowed & (spans*spans <= .25**2+1e-12).all(axis=1))
    for begin in eligible:
        if np.square(spans[begin]).sum() <= .25**2+1e-12:
            return int(begin+40)
        xy = poses[begin:begin+41, :2].astype(np.float64)
        if np.square(xy[:, None]-xy[None, :]).sum(axis=-1).max() <= .25**2+1e-12:
            return int(begin+40)
    return None


all_routes, scenes = [], []
bounded_verification_samples = []
for record in records:
    case = json.loads((manifest_path.parent/record['case']).read_text())
    goal = np.asarray(case['goal_xy'], dtype=np.float64)
    radius = float(case.get('goal_radius_m', 2.5))
    scene_rows = []
    for index, relative in enumerate(record['routes']):
        route_path = manifest_path.parent/relative
        route = json.loads(route_path.read_text())
        route_meta = route['meta']
        directory = root/'raw'/record['scene_id']/route_path.stem
        row = {'scene_id': record['scene_id'], 'split': record['split'], 'family': record['family'],
               'route_index': index, 'route_name': route_path.stem,
               'offset_m': route_meta['lateral_offset_m'], 'cruise_speed_mps': route_meta['cruise_speed_mps'],
               'completed': False}
        if not (directory/'batch_complete.json').exists():
            scene_rows.append(row)
            all_routes.append(row)
            continue
        marker = json.loads((directory/'batch_complete.json').read_text())
        task = marker['contract']['task']
        assert task['case_sha256'] == record['case_sha256'] and task['route_sha256'] == record['route_sha256'][index]
        assert task['split'] == record['split'] and task['horizon_s'] == 180
        outcome = json.loads((directory/'outcome.json').read_text())
        with np.load(directory/'trajectory.npz', allow_pickle=False) as raw:
            poses = np.vstack([raw['pose'], raw['terminal_pose']])
            actions = raw['action']
            parked = np.r_[raw['parked'], raw['terminal_parked']].astype(bool)
            dt = float(raw['dt_s'])
        with np.load(directory/'rich_intervals.npz', allow_pickle=False) as intervals:
            asset = intervals['max_asset_contact_max_resultant_n']
            chassis = intervals['max_chassis_contact_resultant_n']
            rolls = intervals['max_abs_roll_rad']
            pitches = intervals['max_abs_pitch_rad']
            work = intervals['engine_interface_positive_work_kj']
            assert all(np.isfinite(v).all() for v in (asset, chassis, rolls, pitches, work))
            maxima = [float(np.max(v)) for v in (asset, chassis, rolls, pitches)]
            total_work = float(np.sum(work, dtype=np.float64))
        bounded_endpoint = bounded_first(poses, actions, parked, goal, radius)
        first_by_kind = (not bounded_verification_samples or
                         not any((v['first'] is None) == (bounded_endpoint is None) for v in bounded_verification_samples))
        if first_by_kind or len(bounded_verification_samples) < 5:
            sys.path.insert(0, str(source/'src'))
            from nedm.traverse.fdm_bounded_targets import bounded_motion_endpoints
            exact = np.flatnonzero(bounded_motion_endpoints(poses, actions, parked, goal_xy=goal, goal_radius_m=radius))
            expected = int(exact[0]) if len(exact) else None
            assert expected == bounded_endpoint, (record['scene_id'], index, expected, bounded_endpoint)
            bounded_verification_samples.append({'scene_id': record['scene_id'], 'route_index': index, 'first': expected})
        contact = maxima[0] > 1 or maxima[1] > 1
        attitude_risk = maxima[2] > math.radians(60) or maxima[3] > math.radians(60)
        reached = bool(outcome['goal_reached'])
        safe = reached and not contact and bounded_endpoint is None and not attitude_risk
        row.update(completed=True, status=outcome['status'], goal_reached=reached, schema_safe_goal=safe,
                   goal_time_s=outcome['goal_time_s'], elapsed_s=outcome['elapsed_s'],
                   positive_engine_interface_work_kj=total_work,
                   asset_contact=maxima[0] > 1, chassis_contact=maxima[1] > 1,
                   max_asset_contact_n=maxima[0], max_chassis_contact_n=maxima[1],
                   max_abs_roll_deg=math.degrees(maxima[2]), max_abs_pitch_deg=math.degrees(maxima[3]),
                   solver_attitude_risk=attitude_risk,
                   bounded_motion=bounded_endpoint is not None,
                   first_bounded_motion_endpoint_s=bounded_endpoint*.05 if bounded_endpoint is not None else None,
                   strict_near_stop=bool(outcome['sustained_near_stop']),
                   completion_sha256=hashlib.sha256((directory/'batch_complete.json').read_bytes()).hexdigest())
        scene_rows.append(row)
        all_routes.append(row)
    safe_rows = [r for r in scene_rows if r.get('schema_safe_goal')]
    scenes.append({'scene_id': record['scene_id'], 'split': record['split'], 'family': record['family'],
                   'declared_routes': len(scene_rows), 'completed_routes': sum(r['completed'] for r in scene_rows),
                   'goal_completions': sum(r.get('goal_reached', False) for r in scene_rows),
                   'schema_safe_completions': len(safe_rows),
                   'safe_controls': [{k: r[k] for k in ('route_index', 'route_name', 'offset_m', 'cruise_speed_mps',
                                                        'goal_time_s', 'positive_engine_interface_work_kj')} for r in safe_rows],
                   'fastest_safe': min(safe_rows, key=lambda r: r['goal_time_s']) if safe_rows else None,
                   'least_work_safe': min(safe_rows, key=lambda r: r['positive_engine_interface_work_kj']) if safe_rows else None,
                   'routes': scene_rows})
by_family = {}
for family in sorted(set(r['family'] for r in records)):
    rows = [r for r in all_routes if r['family'] == family and r['completed']]
    by_family[family] = {'completed': len(rows), 'goals': sum(r['goal_reached'] for r in rows),
                         'schema_safe_goals': sum(r['schema_safe_goal'] for r in rows),
                         'safe_by_speed': dict(collections.Counter(str(r['cruise_speed_mps']) for r in rows if r['schema_safe_goal']))}
report = {'schema': 'fdm_diverse_actual_route_support_v1', 'observed_utc': datetime.now(timezone.utc).isoformat(),
          'job_id': '412066', 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
          'source_snapshot': str(source), 'declared_routes': len(all_routes),
          'completed_routes': sum(r['completed'] for r in all_routes),
          'schema_safe_definition': 'Goal reached; no asset OR chassis interval resultant >1 N; no two-second bounded-motion event with every throttle>0.3 and no parking/goal-arrival; no solver-step abs roll OR pitch>60 degrees. Strict near-stop is reported separately.',
          'work_definition': 'Sum positive engine output torque times transmission motorshaft feedback speed, at every synchronized physics step; mechanical interface work, not fuel consumption.',
          'bounded_acceleration_equivalence_samples': bounded_verification_samples,
          'scene_count': len(scenes), 'by_family': by_family, 'scenes': scenes,
          'validation_scenes_without_observed_safe_control': [s['scene_id'] for s in scenes if s['split'] == 'val' and not s['schema_safe_completions']],
          'selection_boundary': 'All 450 train/validation routes remain in the pack. This report characterizes measured control support; it does not remove or replace outcomes. Missing rows mean collection pending, not failure.'}
print(json.dumps(report, indent=2, allow_nan=False))
