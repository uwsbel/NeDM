#!/usr/bin/env python3
"""Post-hoc physical intervention on an already recorded launch reference.

Import frozen online physics unchanged. Replay the recorded decision or replace
only its reference with the recorded selected parent family. This is a causal
diagnostic, never a new protected planner score or a model improvement.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def archive_parity(actual, original):
    checks = {}
    for filename in ('trajectory.npz', 'rich_telemetry.npz', 'rich_intervals.npz', 'anchor_state.npz'):
        with np.load(actual / filename, allow_pickle=False) as a, np.load(original / filename, allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise ValueError(f'Archive keys differ: {filename}')
            for key in a.files:
                av, bv = a[key], b[key]
                checks[f'{filename}:{key}'] = bool(av.dtype == bv.dtype and av.shape == bv.shape
                    and av.tobytes() == bv.tobytes())
    return {'passed': all(checks.values()), 'array_count': len(checks), 'arrays': checks}


def run(a):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Physical diagnostic must run in an AMD Slurm allocation')
    protocol = json.loads(a.protocol.read_text())
    if protocol['scope'] != 'post_hoc_refinement_counterfactual':
        raise ValueError('Wrong diagnostic scope')
    if sha(__file__) != protocol['wrapper_sha256']:
        raise ValueError('Diagnostic wrapper changed after declaration')
    source = Path(protocol['source_root'])
    if sha(source / 'source_manifest.json') != protocol['source_manifest_sha256']:
        raise ValueError('Frozen source manifest changed')
    item = next(x for x in protocol['tasks'] if x['id'] == a.task_id)
    original = Path(item['original_root'])
    old_protocol = json.loads((original / 'online_protocol.json').read_text())
    for rel, expected in old_protocol['source_sha256'].items():
        if sha(source / rel) != expected:
            raise ValueError(f'Frozen source changed: {rel}')
    for rel, expected in item['original_sha256'].items():
        if sha(original / rel) != expected:
            raise ValueError(f'Original evidence changed: {rel}')
    arguments = item['arguments']
    if arguments.get('planning-mode') != 'plan_once' or arguments.get('video-fps', 0) != 0:
        raise ValueError('Require an unchanged single-launch headless comparison')
    if sha(arguments['checkpoint']) != old_protocol['model_checkpoint_sha256']:
        raise ValueError('Checkpoint mismatch')
    if sha(arguments['scene-observation']) != old_protocol['scene_observation_sha256']:
        raise ValueError('Observation mismatch')
    record = json.loads((original / 'decisions/decision_00000.json').read_text())
    if record['frame'] != 0 or record['paused']:
        raise ValueError('Require an executable launch decision')
    packet = copy.deepcopy(record['decision'])
    parent_index = packet['selected_family_index']
    if item['mode'] in ('parent_family', 'best_unrefined'):
        if item['mode'] == 'best_unrefined':
            best = min((x for x in packet['family_scores'] if x['cost'] is not None),
                       key=lambda x: x['cost'])
            parent_index = best['family_index']
        parent = copy.deepcopy(record['candidate_references'][parent_index])
        score = next(x for x in packet['family_scores'] if x['family_index'] == parent_index)
        if not score['allowed_by_predicted_risk'] or score['cost'] is None:
            raise ValueError('Selected parent was not admitted by the frozen scorer')
        packet['route'] = parent
        packet['cost'] = score['cost']
        packet['selected_family_index'] = parent_index
        packet['refinements'] = []
        packet['model_evaluations'] = 0
        packet['claim'] = f"Post-hoc intervention: recorded {item['mode']} reference; no new model inference"
    elif item['mode'] != 'replay_selected':
        raise ValueError('Unknown intervention')
    calls = []
    case = json.loads(Path(arguments['case']).read_text())
    observation = np.load(arguments['scene-observation'], allow_pickle=False)
    expected_rgbd = observation['rgbd'].copy()
    # The frozen online runner supplies the case's float64 goal. The sensor
    # archive holds a rounded float32 copy only for launch-registration checks.
    expected_goal = np.asarray(case['goal_xy'], dtype=float)
    observation.close()
    sys.path.insert(0, str(source / 'src'))
    sys.path.insert(0, str(source / 'scripts'))
    import nedm.traverse.fdm_diverse_planner as planner
    real_planner = planner.plan_rgbd_routes

    def recorded_launch(model, rgbd, history, pose, goal, **kwargs):
        if calls or kwargs['elapsed_s'] != 0.:
            raise ValueError('Unexpected later planning call')
        if hashlib.sha256(np.asarray(history).tobytes()).hexdigest() != record['history_sha256']:
            raise ValueError('Causal launch history differs')
        if not np.array_equal(pose, record['anchor_pose']):
            raise ValueError('Launch pose differs')
        if not np.array_equal(rgbd, expected_rgbd) or not np.array_equal(goal, expected_goal):
            raise ValueError('Observation or goal differs')
        families = kwargs['families']
        if len(families) != len(record['candidate_references']):
            raise ValueError('Launch family count differs')
        for actual, saved in zip(families, record['candidate_references']):
            for field in ('waypoints', 'stations', 'headings', 'speeds'):
                if not np.array_equal(actual[field], saved[field]):
                    raise ValueError(f'Launch family differs: {field}')
        calls.append({'frame': 0, 'mode': item['mode'], 'parent_index': parent_index,
                      'launch_state_history_and_families_exact': True})
        return copy.deepcopy(packet)

    planner.plan_rgbd_routes = recorded_launch
    script = source / 'scripts/traverse_fdm_rgbd_diverse_online.py'
    spec = importlib.util.spec_from_file_location('frozen_online_refinement_diagnostic', script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    out = Path(item['out'])
    if out.exists():
        raise ValueError('Use a new physical output directory')
    argv = [str(script), '--out', str(out), '--chrono-data', a.chrono_data]
    for key, value in arguments.items():
        argv.append('--' + key)
        argv.extend(str(v) for v in (value if isinstance(value, list) else [value]))
    previous_argv = sys.argv
    started = time.time()
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = previous_argv
        planner.plan_rgbd_routes = real_planner
    if len(calls) != 1:
        raise ValueError('Exactly one recorded launch must be used')
    actual_protocol = json.loads((out / 'online_protocol.json').read_text())
    if actual_protocol != old_protocol:
        raise ValueError('Frozen online/controller/model protocol changed')
    actual_runtime = json.loads((out / 'simulation_provenance.json').read_text())
    old_runtime = json.loads((original / 'simulation_provenance.json').read_text())
    runtime_keys = ('case_sha256', 'arena_meta_sha256', 'arena_bmp_sha256', 'source_sha256',
                    'runtime_sha256', 'script_sha256', 'physics_dt_s', 'camera', 'driver',
                    'terrain_texture_sha256')
    if any(actual_runtime[k] != old_runtime[k] for k in runtime_keys):
        raise ValueError('Frozen physical runtime, scene or controller differs')
    decision = json.loads((out / 'decisions/decision_00000.json').read_text())
    if decision['decision'] != packet:
        raise ValueError('Recorded intervention differs from requested decision')
    outcome = json.loads((out / 'outcome.json').read_text())
    old_outcome = json.loads((original / 'outcome.json').read_text())
    physical_keys = ('status', 'schema_safe_goal_reached', 'elapsed_s', 'goal_progress_m',
                     'positive_work_kj', 'schema_contact', 'schema_rollover', 'bounded_blockage_v1',
                     'max_abs_roll_deg', 'max_abs_pitch_deg', 'planning_abstentions')
    parity = archive_parity(out, original) if item['mode'] == 'replay_selected' else None
    if parity is not None and (not parity['passed'] or any(outcome[k] != old_outcome[k] for k in physical_keys)):
        raise ValueError('Selected-reference physical replay failed exact parity')
    result = {'scope': protocol['scope'], 'task_id': item['id'], 'mode': item['mode'],
              'original_trial_id': item['original_trial_id'], 'protocol_sha256': sha(a.protocol),
              'wrapper_sha256': sha(__file__), 'source_manifest_sha256': protocol['source_manifest_sha256'],
              'slurm_job_id': os.environ['SLURM_JOB_ID'], 'calls': calls, 'runtime_exact': True,
              'online_protocol_exact': True, 'requested_decision_exact': True, 'parity': parity,
              'outcome': {k: outcome[k] for k in physical_keys},
              'artifact_sha256': {f: sha(out / f) for f in ('outcome.json', 'trajectory.npz',
                  'rich_telemetry.npz', 'rich_intervals.npz', 'anchor_state.npz',
                  'decisions/decision_00000.json', 'online_protocol.json', 'simulation_provenance.json')},
              'wall_s': time.time() - started}
    write(out / 'counterfactual_audit.json', result)
    print(json.dumps({k: result[k] for k in ('task_id', 'mode', 'outcome', 'wall_s')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--chrono-data', required=True)
    run(parser.parse_args())
