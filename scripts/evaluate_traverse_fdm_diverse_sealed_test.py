#!/usr/bin/env python3
"""Evaluate the predeclared protected cohort only after an explicit final freeze.

No optimizer is created. Test labels remain labels, never forward inputs.
The guard and --help use only the standard library and do not open test paths.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = Path('/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909')
CONTROLS = ('normal', 'shuffle', 'blank')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require_hash(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or sha(path) != expected:
        raise ValueError(f'Frozen hash mismatch: {path}')


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_manifest(path, expected):
    require_hash(path, expected)
    manifest = read(path)
    for relative, digest in manifest['files'].items():
        target = (Path(path).parent / relative).resolve()
        if not target.is_relative_to(Path(path).parent.resolve()):
            raise ValueError('Source manifest escapes its immutable snapshot')
        require_hash(target, digest)
    return manifest


def validate_freeze(path, digest, supplement_path, supplement_sha256):
    """This function must finish before anything in the protected cohort is read."""
    require_hash(path, digest)
    freeze = read(path)
    if freeze.get('schema') != 'fdm_protected_test_freeze_v1' or freeze.get('test_unseal_authorized') is not True:
        raise ValueError('Protected test remains sealed: an explicitly authorized final freeze is required')
    if not freeze.get('frozen_utc') or len(freeze.get('test_scene_ids', [])) != 6:
        raise ValueError('Final freeze must declare time and all six test scenes')
    require_hash(supplement_path, supplement_sha256)
    supplement = read(supplement_path)
    if supplement.get('schema') != 'fdm_protected_offline_freeze_v1' or supplement.get('core_freeze_sha256') != digest:
        raise ValueError('Offline supplement must bind the immutable authorized core freeze')
    offline = supplement['offline_evaluation']
    freeze = {**freeze, 'offline_evaluation': offline}
    source = Path(offline['source_manifest_file'])
    if source.resolve().parent != ROOT.resolve():
        raise ValueError('Run this evaluator from the source snapshot pinned by the freeze')
    code = source_manifest(source, offline['source_manifest_sha256'])
    require_hash(freeze['tasks_file'], freeze['tasks_sha256'])
    train_pack = Path(offline['training_pack_path'])
    require_hash(train_pack / 'manifest.json', offline['training_pack_manifest_sha256'])
    if offline['training_pack_manifest_sha256'] != freeze['training_pack_manifest_sha256']:
        raise ValueError('Online and offline training normalization provenance differ')
    train = read(train_pack / 'manifest.json')
    if set(train['splits']) != {'train', 'val'} or train['horizon_steps'] != 60 or train['output_dt_s'] != .2:
        raise ValueError('Require the frozen full H60 train/validation pack')
    require_hash(train_pack / 'normalization.json', train['normalization_sha256'])
    if set(freeze['test_scene_ids']) & set(train['scene_splits']):
        raise ValueError('Test scene declaration overlaps training or validation')
    packing = CAMPAIGN / 'snapshots/packing_v3/source_manifest.json'
    source_manifest(packing, offline['packing_source_manifest_sha256'])
    models = [(arm, Path(offline[f'{arm}_checkpoint_path'])) for arm in ('rgbd', 'blank')]
    for arm, checkpoint in models:
        if str(checkpoint) != freeze['primary'][f'{arm}_checkpoint_path']:
            raise ValueError('Offline model differs from the model chosen before test unsealing')
        require_hash(checkpoint, freeze['checkpoint_files'][str(checkpoint)])
        if checkpoint.name != freeze['primary']['checkpoint_kind'] + '.pt':
            raise ValueError('RGBD and matched blank must use the frozen best/last checkpoint kind')
        if not isinstance(offline['checkpoint_steps'].get(str(checkpoint)), int):
            raise ValueError('Every model must have a predeclared checkpoint step')
    return freeze, offline, code, train, models


def prepare(freeze, offline, pack, workers):
    # Authorization and all model/source hashes have already passed. Only here
    # do we first open the protected collection declaration and its payloads.
    declaration = Path(freeze['protected_collection_manifest_file'])
    require_hash(declaration, freeze['protected_collection_manifest_sha256'])
    if pack.exists():
        raise ValueError('Preparation needs a fresh output; existing test packs are not overwritten')
    collection_source = CAMPAIGN / 'snapshots/campaign_v2'
    command = [sys.executable, str(CAMPAIGN / 'snapshots/packing_v3/scripts/traverse_fdm_rgbd_diverse_prepare.py'),
        '--cohort-manifest', str(declaration),
        '--raw-root', str(CAMPAIGN / 'protected_test_cohort_v2/raw'),
        '--observation-root', str(CAMPAIGN / 'protected_test_cohort_v2/observations'),
        '--source-snapshot-root', str(collection_source), '--arena-root', str(collection_source),
        '--out', str(pack), '--horizon', '60', '--output-dt', '.2', '--anchor-stride', '20',
        '--workers', str(workers), '--sealed-test', '--normalization-from', offline['training_pack_path']]
    subprocess.run(command, check=True)


def load_test_pack(pack, freeze, offline, train):
    import numpy as np
    manifest = read(pack / 'manifest.json')
    if manifest['cohort_state'] != 'complete' or set(manifest['splits']) != {'test'}:
        raise ValueError('Evaluation requires an explicitly named complete test-only pack')
    entry = manifest['splits']['test']
    if entry['episodes'] != 90 or entry['scenes'] != 6 or set(entry['scene_ids']) != set(freeze['test_scene_ids']):
        raise ValueError('Protected cohort must retain all 90 declared references and all six scenes')
    if manifest['normalization_from']['manifest_sha256'] != offline['training_pack_manifest_sha256']:
        raise ValueError('Test preparation did not use the frozen training normalization')
    require_hash(pack / 'normalization.json', train['normalization_sha256'])
    if manifest['normalization_sha256'] != train['normalization_sha256']:
        raise ValueError('Test normalization hash differs from training')
    for key in ('horizon_steps', 'output_dt_s', 'event_schema', 'bounded_motion_version', 'source_sha256', 'runtime_sha256', 'camera'):
        if manifest[key] != train[key]:
            raise ValueError(f'Test/train collection or target contract differs: {key}')
    for name, key in (('test.npz', 'sha256'), ('test_rgbd.npy', 'rgbd_sha256'), ('test_episodes.json', 'episodes_sha256')):
        require_hash(pack / name, entry[key])
    with np.load(pack / 'test.npz', allow_pickle=False) as file:
        data = {key: file[key].copy() for key in file.files}
    episodes = read(pack / 'test_episodes.json')
    if len(episodes) != 90 or Counter(r['scene_id'] for r in episodes) != Counter({scene: 15 for scene in entry['scene_ids']}):
        raise ValueError('Episode list does not contain exactly 15 references per scene')
    if any(r['split'] != 'test' for r in episodes) or int((data['anchor'] == 0).sum()) != 90:
        raise ValueError('Episode split/anchor-zero coverage changed')
    if len(data['anchor']) != entry['windows']:
        raise ValueError('Window count differs from sealed pack manifest')
    for key, shape in entry['shapes'].items():
        if list(data[key].shape) != shape or not np.isfinite(data[key]).all():
            raise ValueError(f'Invalid test tensor {key}')
    return manifest, data, episodes


def actual_reference_support(episodes, freeze):
    """All 90 physical outcomes, with a side/speed fixed before unsealing."""
    import numpy as np
    from nedm.traverse.fdm_diverse_targets import prepare_episode_labels
    baseline = freeze['offline_evaluation']['fixed_reference_baseline']
    if baseline != {'lateral_offset_m': -44.0, 'cruise_speed_mps': 6.0, 'expected_route_index': 11}:
        raise ValueError('Baseline must be the predeclared -44 m, 6 m/s reference (index 11)')
    rows = []
    for episode in episodes:
        source = CAMPAIGN / 'protected_test_cohort_v2/raw' / episode['scene_id'] / Path(episode['source']).name
        for name, digest in episode['source_sha256'].items():
            require_hash(source / name, digest)
        with np.load(source / 'trajectory.npz', allow_pickle=False) as file:
            raw = {key: file[key].copy() for key in file.files}
        with np.load(source / 'rich_intervals.npz', allow_pickle=False) as file:
            rich = {key: file[key].copy() for key in file.files}
        metadata, outcome = read(source / 'collection_meta.json'), read(source / 'outcome.json')
        pose = np.vstack([raw['pose'], raw['terminal_pose']])
        state = np.vstack([raw['state'], raw['terminal_state']])
        parked = np.r_[raw['parked'], raw['terminal_parked']].astype(bool)
        labels = prepare_episode_labels(pose, state, raw['action'], parked, rich,
            goal_xy=np.asarray(metadata['route']['waypoints'][-1]), goal_radius_m=outcome['goal_radius_m'])
        asset = float(np.max(rich['max_asset_contact_max_resultant_n']))
        chassis = float(np.max(rich['max_chassis_contact_resultant_n']))
        roll = float(np.degrees(np.max(rich['max_abs_roll_rad'])))
        pitch = float(np.degrees(np.max(rich['max_abs_pitch_rad'])))
        bounded = bool(np.any(labels['bounded_endpoints']))
        route = metadata['route']['meta']
        route_index = int(Path(episode['source']).name.rsplit('_', 1)[1])
        selected_baseline = route['lateral_offset_m'] == -44.0 and route['cruise_speed_mps'] == 6.0
        if selected_baseline != (route_index == 11):
            raise ValueError('Frozen baseline index and exact physical route metadata disagree')
        reached = bool(outcome['goal_reached'])
        rows.append({'episode_id': episode['id'], 'scene_id': episode['scene_id'], 'family': episode['family'],
            'route_index': route_index, 'lateral_offset_m': route['lateral_offset_m'], 'cruise_speed_mps': route['cruise_speed_mps'],
            'predeclared_baseline': selected_baseline, 'status': outcome['status'], 'goal_reached': reached,
            'schema_safe_goal': reached and asset <= 1 and chassis <= 1 and not bounded and roll <= 60 and pitch <= 60,
            'goal_time_s': outcome['goal_time_s'], 'elapsed_s': outcome['elapsed_s'],
            'positive_engine_interface_work_kj': float(np.sum(rich['engine_interface_positive_work_kj'], dtype=np.float64)),
            'max_asset_contact_n': asset, 'max_chassis_contact_n': chassis, 'max_abs_roll_deg': roll, 'max_abs_pitch_deg': pitch,
            'bounded_motion': bounded, 'sustained_stall': bool(np.any(labels['sustained_endpoints'])),
            'source_sha256': episode['source_sha256']})
    scenes = {}
    for scene in sorted(freeze['test_scene_ids']):
        cohort = [r for r in rows if r['scene_id'] == scene]
        safe = [r for r in cohort if r['schema_safe_goal']]
        baseline_rows = [r for r in cohort if r['predeclared_baseline']]
        if len(cohort) != 15 or len(baseline_rows) != 1:
            raise ValueError('Missing reference or predeclared baseline')
        scenes[scene] = {'references': 15, 'goals': sum(r['goal_reached'] for r in cohort), 'safe_goals': len(safe),
            'predeclared_baseline': baseline_rows[0],
            'hindsight_fastest_safe': min(safe, key=lambda r: r['goal_time_s']) if safe else None,
            'hindsight_least_work_safe': min(safe, key=lambda r: r['positive_engine_interface_work_kj']) if safe else None}
    return {'schema': 'fdm_protected_reference_support_v1', 'routes': rows, 'scenes': scenes,
        'baseline_safe_goals': sum(s['predeclared_baseline']['schema_safe_goal'] for s in scenes.values()),
        'hindsight_feasible_scenes': sum(s['safe_goals'] > 0 for s in scenes.values()),
        'scope': 'All 90 references retained. Hindsight minima describe physical feasibility only; they are not deployed selections or MPPI results.',
        'safe_definition': 'Goal reached, asset and chassis interval contact <=1 N, no bounded two-second motion/effort event, solver absolute roll/pitch <=60 degrees.',
        'work_definition': 'Positive engine-interface mechanical work over recorded simulation; not fuel consumption.'}


def evaluate(pack, out, freeze, offline, code, train, models, args):
    import numpy as np
    import torch
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
    from traverse_fdm_rgbd_diverse_train import Images, evaluate_rgbd, load_low_dim, select_progress_target, json_clean
    from traverse_fdm_rgbd_diverse_report import causal_groups, select_targets, summarize
    from nedm.traverse.fdm_diverse_model import load_rgbd_checkpoint
    manifest, full_data, episodes = load_test_pack(pack, freeze, offline, train)
    data = load_low_dim(pack / 'test.npz')
    groups, strata = causal_groups(full_data, episodes, CAMPAIGN / 'protected_test_cohort_v2/raw')
    if strata['unavailable_episodes'] or strata['known_windows'] != len(data['anchor']):
        raise ValueError('Every reference must have verified raw causal labels')
    if out.exists():
        raise ValueError('Evaluation needs a fresh report directory')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested AMD GPU inference is unavailable')
    images = Images(pack / 'test_rgbd.npy', len(data['anchor']), device, image_index=data['image_index'], preload=device.type == 'cuda')
    report = {'schema': 'fdm_protected_offline_evaluation_v1', 'split': 'test', 'freeze_sha256': args.freeze_sha256,
        'offline_freeze_sha256': args.offline_freeze_sha256,
        'frozen_utc': freeze['frozen_utc'], 'test_pack_manifest_sha256': sha(pack / 'manifest.json'),
        'training_pack_manifest_sha256': offline['training_pack_manifest_sha256'], 'scene_count': 6, 'episode_count': 90,
        'window_count': len(data['anchor']), 'selection': 'All 90 references and uniform one-second anchors; no outcome filtering.',
        'inference_inputs': ['history', 'commands', 'global_features', 'nominal_pose', 'single_scene_global_rgbd'],
        'image_precision': 'Frozen float16 training-cache encoding, converted to float32 for inference; not the online float32-map deployment path.',
        'limits': 'Offline forecast evaluation and fixed-reference physical support. Neither proves receding-horizon MPPI completion.',
        'controls': list(CONTROLS), 'shuffle_seed': 20260908, 'thresholds': {'contact': .35, 'rollover': .35, 'bounded_motion': .5},
        'strata': strata, 'models': {}, 'host': socket.gethostname(), 'job_id': os.environ.get('SLURM_JOB_ID'),
        'device': str(device), 'torch_version': torch.__version__, 'hip_version': torch.version.hip, 'source_manifest_sha256': offline['source_manifest_sha256']}
    seeds, kinds = set(), set()
    for arm, path in models:
        model, checkpoint = load_rgbd_checkpoint(path, device)
        if model.config.arm != arm or model.config.horizon != 60 or model.config.progress_event_definition != 'bounded_motion':
            raise ValueError('Checkpoint arm/horizon/target differs from frozen declaration')
        if checkpoint['step'] != offline['checkpoint_steps'][str(path)] or checkpoint['provenance']['data']['manifest.json'] != offline['training_pack_manifest_sha256']:
            raise ValueError('Checkpoint step or training provenance mismatch')
        for relative, digest in checkpoint['provenance']['code'].items():
            if code['files'].get(relative) != digest:
                raise ValueError(f'Evaluation model/helper source differs from training: {relative}')
        seeds.add(checkpoint['args']['seed']); kinds.add(path.name)
        weights = {key: checkpoint['args'][flag] for key, flag in (
            ('xy', 'xy_weight'), ('yaw', 'yaw_weight'), ('work', 'work_weight'), ('events', 'event_weight'), ('attitude', 'attitude_weight'))}
        selected = select_progress_target(data, model.config.progress_event_definition)
        reporting_data = select_targets(full_data, model.config.progress_event_definition)
        supported = model.supported_events.cpu().numpy()
        result = {'checkpoint': str(path), 'checkpoint_sha256': freeze['checkpoint_files'][str(path)],
            'step': checkpoint['step'], 'seed': checkpoint['args']['seed'], 'arm': arm, 'normalization_source': 'Checkpoint training normalization, never refitted',
            'loss_weights': weights, 'supported_events': supported.tolist(), 'controls': {}}
        for control in CONTROLS:
            metrics, prediction = evaluate_rgbd(model, selected, images, args.batch, device, weights, control, 20260908)
            prediction_path = out / f'{arm}_test_predictions_{control}.npz'
            np.savez(prediction_path, **prediction)
            result['controls'][control] = {'metrics': metrics,
                'horizons': summarize(prediction, reporting_data, groups, manifest, supported, manifest['splits']['test']['scene_ids']),
                'prediction_file': prediction_path.name, 'prediction_sha256': sha(prediction_path)}
            print(json.dumps({'arm': arm, 'control': control, 'windows': len(data['anchor']), 'seconds': metrics['evaluation_seconds']}), flush=True)
            del prediction
        report['models'][arm] = result
        dump(out / f'{arm}_metrics.json', json_clean(result))
        del model, checkpoint
    if len(seeds) != 1 or len(kinds) != 1:
        raise ValueError('RGBD and blank must be matched seed/checkpoint kind')
    support = actual_reference_support(episodes, freeze)
    dump(out / 'fixed_reference_support.json', json_clean(support))
    report['fixed_reference_support'] = {'file': 'fixed_reference_support.json', 'sha256': sha(out / 'fixed_reference_support.json'),
        'baseline_safe_goals': support['baseline_safe_goals'], 'hindsight_feasible_scenes': support['hindsight_feasible_scenes']}
    imported = {str(Path(module.__file__).resolve().relative_to(ROOT)): sha(module.__file__)
        for module in sys.modules.values() if getattr(module, '__file__', None) and Path(module.__file__).suffix == '.py'
        and Path(module.__file__).resolve().is_relative_to(ROOT)}
    if any(code['files'].get(name) != digest for name, digest in imported.items()):
        raise ValueError('Unfrozen imported evaluation source')
    report['imported_source_sha256'] = imported
    dump(out / 'report.json', json_clean(report))
    lines = ['# Protected offline FDM evaluation', '',
        'All 90 fixed references on six sealed scenes were evaluated after the final freeze. These are offline forecast metrics, not closed-loop MPPI outcomes.', '',
        '| Model | Step | 12 s anchor-zero FDE (m) | 12 s work MAE (kJ) |', '|---|---:|---:|---:|']
    for arm, result in report['models'].items():
        entry = result['controls']['normal']['horizons']['12s']['groups']['anchor0']
        lines.append(f"| {arm} | {result['step']} | {entry['motion']['endpoint_fde_m']:.3f} | {entry['work_kj']['mae']:.3f} |")
    lines += ['', f"The predeclared -44 m, 6 m/s reference completes safely on {support['baseline_safe_goals']}/6 scenes. Hindsight over all 15 references finds a safe control on {support['hindsight_feasible_scenes']}/6 scenes; that is a feasibility bound, not a deployed policy.", '',
        'Full per-scene, causal-prefix, event-calibration and image-control metrics are in report.json. Mechanical work is not fuel energy. Overlapping windows are not independent samples.']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    dump(out / 'complete.json', {'complete': True, 'freeze_sha256': args.freeze_sha256,
        'offline_freeze_sha256': args.offline_freeze_sha256,
        'report_sha256': sha(out / 'report.json'), 'support_sha256': sha(out / 'fixed_reference_support.json'),
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'models': 2, 'controls_each': 3, 'episodes': 90, 'scenes': 6})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--offline-freeze', type=Path, required=True)
    parser.add_argument('--offline-freeze-sha256', required=True)
    parser.add_argument('--pack-out', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('preflight', 'prepare', 'evaluate', 'all'), default='all')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()
    freeze, offline, code, train, models = validate_freeze(args.freeze, args.freeze_sha256, args.offline_freeze, args.offline_freeze_sha256)
    # Preflight validates source/model/train provenance; never opens test paths.
    if args.stage == 'preflight':
        print(json.dumps({'authorized_freeze_valid': True, 'test_payload_opened': False}))
        return
    if args.stage in ('prepare', 'all'):
        prepare(freeze, offline, args.pack_out, args.workers)
    if args.stage in ('evaluate', 'all'):
        evaluate(args.pack_out, args.out, freeze, offline, code, train, models, args)


if __name__ == '__main__':
    main()
