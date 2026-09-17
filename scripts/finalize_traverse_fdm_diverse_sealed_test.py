#!/usr/bin/env python3
"""Finalize verified frozen test exports without rerunning inference or training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from evaluate_traverse_fdm_diverse_sealed_test import (
    validate_freeze, source_manifest, require_hash, sha, read, dump,
    load_test_pack, actual_reference_support, CAMPAIGN, CONTROLS)


def source_audit(code):
    """All pinned first-party files stay strict; synthetic external files are logged."""
    imported, ignored = {}, []
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename or Path(filename).suffix != '.py':
            continue
        path = Path(filename)
        first_party = name.startswith(('nedm', 'traverse_fdm', 'evaluate_traverse', 'finalize_traverse'))
        if not path.is_file():
            if first_party or (path.is_absolute() and path.resolve().is_relative_to(ROOT)):
                raise ValueError(f'Missing real first-party module: {name}: {filename}')
            # Torch creates these synthetic modules via exec(compile(...)).
            if (name, filename) in {('torch.ops', '_ops.py'), ('torch.classes', '_classes.py')}:
                ignored.append({'module': name, 'advertised_file': filename, 'reason': 'Nonexistent relative synthetic external PyTorch source'})
                continue
            raise ValueError(f'Unexpected missing external source, not a declared synthetic module: {name}: {filename}')
        resolved = path.resolve()
        if resolved.is_relative_to(ROOT):
            relative = str(resolved.relative_to(ROOT))
            digest = sha(resolved)
            if code['files'].get(relative) != digest:
                raise ValueError(f'Unpinned real first-party source: {relative}')
            imported[relative] = digest
    return imported, ignored


def verify_numeric_summary(actual, expected):
    """Count/schema exactness, with explicit float32 SIMD rounding allowances."""
    differences = []
    def compare(a, b, path=''):
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                raise ValueError(f'Summary keys changed: {path}')
            for key in a:
                compare(a[key], b[key], path + '/' + str(key))
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                raise ValueError(f'Summary list length changed: {path}')
            for index, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path + '/' + str(index))
        elif a == b:
            return
        elif isinstance(a, float) and isinstance(b, float):
            angular = 'attitude_deg' in path or 'yaw_error_deg' in path
            atol = 2e-5 if angular else 2e-7
            if not math.isclose(a, b, rel_tol=2e-6, abs_tol=atol):
                raise ValueError(f'Summary value changed beyond numerical precision: {path}: {a} vs {b}')
            differences.append({'path': path, 'absolute_delta': abs(a-b), 'angular_degrees': angular})
        else:
            raise ValueError(f'Summary discrete value changed: {path}: {a} vs {b}')
    compare(actual, expected)
    return {'all_counts_schema_and_discrete_values_exact': True, 'all_floats_within_tolerance': True,
        'floating_values_not_bit_identical': len(differences),
        'max_absolute_delta_angular_degrees': max((d['absolute_delta'] for d in differences if d['angular_degrees']), default=0.),
        'max_absolute_delta_other_metrics': max((d['absolute_delta'] for d in differences if not d['angular_degrees']), default=0.),
        'tolerance': {'relative': 2e-6, 'absolute_angular_degrees': 2e-5, 'absolute_other': 2e-7},
        'example_float_differences': differences[:20]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--offline-freeze', type=Path, required=True)
    parser.add_argument('--offline-freeze-sha256', required=True)
    parser.add_argument('--pack-out', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    freeze, offline, code, train, models = validate_freeze(args.freeze, args.freeze_sha256, args.offline_freeze, args.offline_freeze_sha256)
    export = offline['prediction_export']
    require_hash(export['manifest_file'], export['manifest_sha256'])
    exports = read(export['manifest_file'])
    if exports['core_freeze_sha256'] != args.freeze_sha256 or exports['producer_job_id'] != '412126':
        raise ValueError('Export provenance does not bind the frozen inference run')
    source_manifest(exports['producer_source_manifest_file'], exports['producer_source_manifest_sha256'])
    require_hash(exports['producer_supplement_file'], exports['producer_supplement_sha256'])
    require_hash(exports['producer_log_file'], exports['producer_log_sha256'])
    producer_freeze = read(exports['producer_supplement_file'])
    if producer_freeze['core_freeze_sha256'] != args.freeze_sha256:
        raise ValueError('Producer supplement core-freeze mismatch')
    directory = Path(exports['root'])
    for relative, digest in exports['files'].items():
        require_hash(directory / relative, digest)
    # Imports occur after authorization; no checkpoint forward or optimizer is used.
    import numpy as np
    from traverse_fdm_rgbd_diverse_report import causal_groups, summarize, select_targets
    from traverse_fdm_train import json_clean
    manifest, data, episodes = load_test_pack(args.pack_out, freeze, offline, train)
    if sha(args.pack_out / 'manifest.json') != exports['test_pack_manifest_sha256']:
        raise ValueError('Export/test-pack hash mismatch')
    groups, strata = causal_groups(data, episodes, CAMPAIGN / 'protected_test_cohort_v2/raw')
    if strata['unavailable_episodes'] or strata['known_windows'] != len(data['anchor']):
        raise ValueError('Causal source coverage incomplete')
    if args.out.exists():
        raise ValueError('Use a fresh finalized output')
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'fdm_protected_offline_evaluation_v1', 'split': 'test',
        'freeze_sha256': args.freeze_sha256, 'offline_freeze_sha256': args.offline_freeze_sha256,
        'frozen_utc': freeze['frozen_utc'], 'test_pack_manifest_sha256': sha(args.pack_out / 'manifest.json'),
        'training_pack_manifest_sha256': offline['training_pack_manifest_sha256'],
        'scene_count': 6, 'episode_count': 90, 'window_count': len(data['anchor']),
        'selection': 'All 90 references and uniform one-second anchors; no outcome filtering.',
        'inference_inputs': ['history', 'commands', 'global_features', 'nominal_pose', 'single_scene_global_rgbd'],
        'image_precision': 'Same float16 cache encoding as training, promoted to float32; online native float32-map input is a separate path.',
        'limits': 'Offline forecasts and fixed-reference physical support do not prove closed-loop MPPI completion.',
        'controls': list(CONTROLS), 'shuffle_seed': 20260908,
        'thresholds': {'contact': .35, 'rollover': .35, 'bounded_motion': .5},
        'strata': strata, 'models': {}, 'inference_device': exports['inference_device'],
        'producer_job_id': exports['producer_job_id'], 'finalization_job_id': os.environ.get('SLURM_JOB_ID'),
        'export_manifest_sha256': export['manifest_sha256'], 'source_manifest_sha256': offline['source_manifest_sha256'],
        'finalization': 'Prediction checksums verified; all horizon/causal/scene summaries recomputed from saved outputs with explicit floating-point precision audit; raw physical support independently recomputed. No inference or optimizer updates.'}
    checked = []
    for arm, checkpoint in models:
        result = read(directory / f'{arm}_metrics.json')
        if result['checkpoint'] != str(checkpoint) or result['checkpoint_sha256'] != freeze['checkpoint_files'][str(checkpoint)] or result['step'] != 5000 or result['seed'] != 11:
            raise ValueError('Saved export is not the predeclared matched LAST5000 checkpoint')
        if set(result['controls']) != set(CONTROLS):
            raise ValueError('Required image controls missing')
        for control in CONTROLS:
            entry = result['controls'][control]
            file = directory / entry['prediction_file']
            require_hash(file, entry['prediction_sha256'])
            with np.load(file, allow_pickle=False) as archive:
                prediction = {key: archive[key].copy() for key in archive.files}
            actual = json_clean(summarize(prediction, select_targets(data, 'bounded_motion'), groups, manifest,
                np.asarray(result['supported_events']), manifest['splits']['test']['scene_ids']))
            comparison = verify_numeric_summary(actual, entry['horizons'])
            entry['prediction_file'] = str(file)
            checked.append({'arm': arm, 'control': control, 'prediction_sha256': sha(file), 'summary_comparison': comparison})
            del prediction
        report['models'][arm] = result
    support = actual_reference_support(episodes, freeze)
    support_comparison = verify_numeric_summary(json_clean(support), read(directory / 'fixed_reference_support.json'))
    dump(args.out / 'fixed_reference_support.json', json_clean(support))
    report['fixed_reference_support'] = {'file': 'fixed_reference_support.json', 'sha256': sha(args.out / 'fixed_reference_support.json'),
        'baseline_safe_goals': support['baseline_safe_goals'], 'hindsight_feasible_scenes': support['hindsight_feasible_scenes']}
    imported, ignored = source_audit(code)
    report['imported_source_sha256'] = imported
    report['ignored_synthetic_or_external_source_files'] = ignored
    report['prediction_checks'] = checked
    report['raw_support_recomputation'] = support_comparison
    dump(args.out / 'report.json', json_clean(report))
    lines = ['# Protected offline FDM evaluation', '',
        'All 90 fixed references on six protected scenes were evaluated after the final freeze. Saved predictions were checksum-verified; all summaries were independently recomputed within recorded floating-point precision. Inference was not repeated.', '',
        '| Model | Step | 12 s anchor-zero FDE (m) | 12 s work MAE (kJ) |', '|---|---:|---:|---:|']
    for arm, result in report['models'].items():
        entry = result['controls']['normal']['horizons']['12s']['groups']['anchor0']
        lines.append(f"| {arm} | {result['step']} | {entry['motion']['endpoint_fde_m']:.3f} | {entry['work_kj']['mae']:.3f} |")
    lines += ['', f"The predeclared -44 m, 6 m/s reference completes safely on {support['baseline_safe_goals']}/6 scenes. Hindsight over all15 references finds safe controls on {support['hindsight_feasible_scenes']}/6 scenes; this is feasibility, not a deployed policy.", '',
        'Full event calibration, per-scene, causal-prefix and image controls are in report.json. Mechanical work is not fuel energy; overlapping windows are not independent samples. Offline forecasts do not establish receding-horizon MPPI completion.']
    (args.out / 'report.md').write_text('\n'.join(lines) + '\n')
    dump(args.out / 'complete.json', {'complete': True, 'freeze_sha256': args.freeze_sha256,
        'offline_freeze_sha256': args.offline_freeze_sha256, 'report_sha256': sha(args.out / 'report.json'),
        'support_sha256': sha(args.out / 'fixed_reference_support.json'), 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'predictions_recomputed': False, 'summary_recomputations_verified': 6, 'raw_support_recomputed_verified': True,
        'models': 2, 'controls_each': 3, 'episodes': 90, 'scenes': 6})
    print(json.dumps({'complete': True, 'out': str(args.out), 'episodes': 90, 'scenes': 6, 'recomputed_summaries': 6}), flush=True)


if __name__ == '__main__':
    main()
