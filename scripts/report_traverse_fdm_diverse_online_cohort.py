#!/usr/bin/env python3
"""Audit every declared online trial and apply a preregistered validation gate.

No training, replanning, forecast replay or protected test access. Failed,
missing, timed-out and abstaining trials remain in unconditional denominators.
Time/work fractions are computed only for paired verified schema-safe goals.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ('outcome.json', 'online_protocol.json', 'online_planning_summary.json',
            'anchor_equality.json', 'trajectory.npz', 'rich_telemetry.npz', 'rich_intervals.npz')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def load(path):
    with np.load(path, allow_pickle=False) as values:
        return {k: values[k].copy() for k in values.files}


def json_clean(value):
    if isinstance(value, dict): return {str(k):json_clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [json_clean(v) for v in value]
    if isinstance(value, np.ndarray): return json_clean(value.tolist())
    if isinstance(value, (float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(json_clean(value), indent=2, allow_nan=False)+'\n')


def mapped(path, mappings):
    path = Path(path)
    if path.exists(): return path
    for old, new in mappings:
        if path.is_relative_to(old): return new/path.relative_to(old)
    return path


def require_hash(path, expected, description):
    if sha(path) != expected: raise ValueError(f'{description} checksum mismatch: {path}')


def require(condition, text):
    if not condition: raise ValueError(text)


def expected_weight(arguments, mappings):
    if arguments.get('cost-mode', 'energy_risk') == 'time_risk': return 0.
    if arguments.get('cost-config'):
        return float(read(mapped(arguments['cost-config'], mappings)).get('energy_weight_s_per_kj', .02))
    return .02


def audit_rejections(folder, planning, protocol, *, verify_decisions=False):
    """Retain summary counters; optionally cross-check every saved rejection list."""
    supported = 'invalid_fresh_candidates' in protocol.get('candidate_policy', {})
    if not supported:
        return {'rejected_fresh_candidates':None, 'decisions_with_rejected_fresh':None,
                'max_rejected_fresh_per_decision':None, 'rejection_evidence':'Not recorded by this source version'}
    limit = len(protocol['candidate_speeds_mps'])*len(protocol['candidate_offsets_m'])
    counts = [row.get('rejected_candidate_count') for row in planning]
    require(all(type(count) is int and 0 <= count <= limit for count in counts), 'Invalid or missing fresh-reference rejection counter')
    result = {'rejected_fresh_candidates':sum(counts), 'decisions_with_rejected_fresh':sum(count > 0 for count in counts),
              'max_rejected_fresh_per_decision':max(counts, default=0),
              'rejection_evidence':'Checksum-verified planning summary counters', 'rejection_details':{}}
    if verify_decisions:
        files = sorted((folder/'decisions').glob('decision_*.json'))
        require(len(files) == len(planning), 'Saved decision file count differs from planning summary')
        digest = hashlib.sha256()
        for index, (path, summary, count) in enumerate(zip(files, planning, counts)):
            require(path.name == f'decision_{index:05d}.json', 'Saved decision sequence has missing or extra indices')
            decision = read(path)
            require(decision['frame'] == summary['frame'] and decision['time_s'] == summary['time_s'], 'Decision rejection anchor differs from planning summary')
            rejected = decision['rejected_candidate_references']
            require(len(rejected) == count, 'Saved rejected references differ from checksum-verified summary counter')
            origins = [entry['candidate_origin'] for entry in rejected]
            require(len(origins) == len(set(origins)), 'A fresh candidate is counted as rejected more than once')
            for entry in rejected:
                require(entry['candidate_origin'] in {f'fresh_family_{i:02d}' for i in range(limit)}, 'Rejection unexpectedly applies to an active/original reference')
                require(entry['reason'] == 'reference_contract', 'Unexpected fresh-reference rejection reason')
                require(len(entry['reference_sha256']) == 64 and all(c in '0123456789abcdef' for c in entry['reference_sha256']), 'Missing rejected-reference geometry hash')
                detail = entry['detail']; result['rejection_details'][detail] = result['rejection_details'].get(detail, 0)+1
            digest.update(path.name.encode()+b'\0'+sha(path).encode()+b'\n')
        result.update(rejection_evidence='Every saved decision rejection list agrees with checksum-verified summary counters',
                      decision_files_sha256=digest.hexdigest())
    return result


def verify_checkpoint_kind(path, step, status, declared_model, frozen):
    """Keep validation-best and fixed-budget checkpoints separate and explicit."""
    kind = declared_model.get('checkpoint_kind', 'best')
    require(kind in ('best','last'), 'Unknown declared checkpoint kind')
    require(Path(path).name == kind+'.pt', 'Checkpoint filename differs from declared best/last kind')
    require(frozen.get('checkpoint_kind', kind) == kind, 'Frozen task checkpoint kind differs from selection spec')
    budget = int(declared_model['training_update_budget'])
    require(status['state'] == 'complete' and int(status['step']) == budget, 'Predeclared training update budget is not complete')
    expected_step = int(status['best_step']) if kind == 'best' else budget
    require(int(step) == expected_step and int(step) <= budget, 'Checkpoint step disagrees with declared best/last rule and completed training status')
    return kind


def audit_trial(task, result, args, mappings):
    name = task['id']; declared = task['arguments']; folder = args.batch/name
    row = {'trial_id':name, 'scene_id':Path(declared['case']).stem,
           'declared_horizon_s':float(declared.get('horizon-s', 180.)),
           'energy_weight_s_per_kj':None, 'process_exit_code':None if result is None else result.get('exit_code'),
           'artifact_valid':False, 'verified_schema_safe_goal':False, 'goal_reached':False,
           'status':'missing', 'errors':[]}
    authorized = getattr(args, 'authorized_test_scene_ids', frozenset())
    if (task.get('split') == 'test' or '_test_' in declared['case'] or '_test_' in name) and row['scene_id'] not in authorized:
        row['status']='unauthorized';row['errors'].append('Protected declared task requires a verified concrete-freeze gate before any artifact read')
        return row
    try: row['energy_weight_s_per_kj'] = expected_weight(declared, mappings)
    except (OSError,ValueError,KeyError) as error: row['errors'].append(str(error))
    protocol = outcome = None
    try:
        case_path = mapped(declared['case'], mappings); case = read(case_path)
        require(case['split'] in ('train','val') or (case['split'] == 'test' and row['scene_id'] in authorized),
                'Protected test cases require a separate verified concrete-freeze gate')
        require(result is not None, 'Declared task has no batch result')
        require(read(args.batch/(name+'.result.json')) == result, 'Per-task result disagrees with batch result')
        if 'log_sha256' in result:
            require_hash(args.batch/(name+'.log'), result['log_sha256'], 'Trial log')
        require(result.get('exit_code') == 0, 'Subprocess did not complete successfully; retained as failed trial')
        require(set(result.get('sha256', {})) >= set(REQUIRED), 'Successful result lacks mandatory artifact checksums')
        for filename in REQUIRED:
            require_hash(folder/filename, result['sha256'][filename], filename)
        protocol, outcome = read(folder/'online_protocol.json'), read(folder/'outcome.json')
        planning_mode = protocol.get('planning_mode', 'receding')
        require(planning_mode in ('receding','plan_once'), 'Unknown executed planning mode')
        require(planning_mode == declared.get('planning-mode', 'receding'), 'Executed planning mode differs from declared task')
        require(outcome.get('planning_mode', 'receding') == planning_mode, 'Outcome planning mode differs from protocol')
        row['planning_mode'] = planning_mode
        row.update({k:outcome.get(k) for k in ('case_id','status','goal_reached','goal_time_s','elapsed_s','goal_progress_m',
            'positive_work_kj','schema_contact','schema_rollover','bounded_blockage_v1','planning_decisions','planning_abstentions',
            'max_abs_roll_deg','max_abs_pitch_deg','planning_wall_s')})
        row['schema_safe_goal_reported'] = bool(outcome['schema_safe_goal_reached'])
        require(row['scene_id'] == outcome['case_id'], 'Declared scene and outcome differ')
        require(task.get('scene_id', row['scene_id']) == row['scene_id'], 'Task scene metadata disagrees with case')
        require(task.get('energy_weight_s_per_kj', row['energy_weight_s_per_kj']) == row['energy_weight_s_per_kj'], 'Task energy metadata disagrees with declared command')
        if 'outcome' in result:
            require(all(outcome[k] == value for k,value in result['outcome'].items()), 'Batch outcome summary differs from actual outcome')
        weight = float(protocol['cost_config']['energy_weight_s_per_kj'])
        require(weight == row['energy_weight_s_per_kj'], 'Executed energy coefficient differs from declared task')
        row['checkpoint_sha256'] = protocol['model_checkpoint_sha256']; row['checkpoint_step'] = int(protocol['checkpoint_step'])
        row['checkpoint_path'] = str(mapped(declared['checkpoint'], mappings))
        require_hash(row['checkpoint_path'], protocol['model_checkpoint_sha256'], 'Model checkpoint')
        row['split'] = case['split']; row['family'] = case.get('family', 'unreported')
        require_hash(case_path, outcome['case_sha256'], 'Case')
        observation_path = mapped(declared['scene-observation'], mappings)
        require_hash(observation_path, protocol['scene_observation_sha256'], 'Fixed observation')
        observation = load(observation_path)
        equality = read(folder/'anchor_equality.json')
        require(equality['matched'] and equality['observation_sha256'] == protocol['scene_observation_sha256'], 'Physical launch equality failed')
        for source, digest in protocol['source_sha256'].items():
            require_hash(args.code_root/source, digest, 'Online source')
        physical = load(folder/'trajectory.npz'); rich = load(folder/'rich_telemetry.npz'); intervals = load(folder/'rich_intervals.npz')
        n = len(physical['state']); pose = np.vstack([physical['pose'],physical['terminal_pose']])
        require(n == outcome['frames'] and len(rich['time_s']) == n+1 and len(intervals['duration_s']) == n, 'Measured sample/interval counts disagree')
        require(np.isfinite(rich['time_s']).all() and (np.diff(rich['time_s']) > 0.).all(), 'Invalid actual timestamps')
        require(np.allclose(np.diff(rich['time_s']), intervals['duration_s'], atol=1e-8, rtol=0.), 'Actual intervals do not cover adjacent sample times')
        require(np.allclose(np.diff(rich['time_s']), .05, atol=1e-8, rtol=0.), 'Unexpected telemetry cadence')
        require(np.isclose(rich['time_s'][-1], outcome['elapsed_s'], atol=1e-7, rtol=0.), 'Final elapsed time differs from measured endpoint')
        require(outcome['elapsed_s'] <= row['declared_horizon_s']+1e-7, 'Run exceeds declared physical horizon')
        require(np.array_equal(pose, np.column_stack([rich[k] for k in ('pos_world_x_m','pos_world_y_m','yaw_rad')])), 'Rich pose differs from core physical telemetry')
        require(np.allclose(physical['state'][0], observation['state'], atol=1e-5, rtol=0.), 'Initial state differs from scored fixed observation')
        require(np.allclose(physical['pose'][0], observation['pose'], atol=2e-6, rtol=0.), 'Initial pose differs from scored fixed observation')
        work = intervals['engine_interface_positive_work_kj']
        require(np.isfinite(work).all() and (work >= 0.).all(), 'Missing or negative positive-work measurement')
        require(np.allclose(work, physical['positive_work_kj_per_interval'], atol=1e-8, rtol=1e-8), 'Core/rich interval work differs')
        require(np.isclose(work.sum(), outcome['positive_work_kj'], atol=1e-6, rtol=1e-8), 'Outcome work differs from measured sum')
        require(bool((intervals['solver_step_work'] == 1).all()), 'Missing solver-rate work coverage')
        require(bool((intervals['post_step_risk_sample_count'] > 0).all()), 'Missing solver-rate physical-risk coverage')
        contact = np.maximum(intervals['max_chassis_contact_resultant_n'], intervals['max_asset_contact_max_resultant_n'])
        peaks = np.column_stack([intervals[f'max_abs_{axis}_rad'] for axis in ('roll','pitch')])
        require(np.isfinite(contact).all() and np.isfinite(peaks).all(), 'Incomplete contact/attitude evidence')
        schema_contact = bool((contact > 1.).any()); rollover = bool((np.degrees(peaks) > 60.).any())
        require(schema_contact == outcome['schema_contact'] and rollover == outcome['schema_rollover'], 'Outcome contact/rollover differs from measured interval schema')
        bounded = []
        for start in range(max(0,n-40+1)):
            if physical['parked'][start:start+40].any() or not (physical['action'][start:start+40,1] > .3).all(): continue
            points = pose[start:start+41,:2]
            if np.square(points[:,None]-points[None,:]).sum(-1).max() <= .25**2: bounded.append(start)
        require(bool(bounded) == outcome['bounded_blockage_v1'], 'Bounded blockage differs from measured pose/action criterion')
        goal = np.asarray(case['goal_xy']); radius = float(case['goal_radius_m'])
        goal_reached = bool(outcome['goal_reached'])
        if goal_reached:
            require(outcome['status'] == 'goal_reached' and outcome['goal_time_s'] is not None, 'Inconsistent full-goal completion')
            require(np.linalg.norm(pose[-1,:2]-goal) <= radius+1e-7, 'Reported goal lacks a physical endpoint in supplied goal radius')
            require(np.isclose(outcome['goal_time_s'], outcome['elapsed_s']), 'Full-goal time differs from actual termination')
        else:
            require(outcome['goal_time_s'] is None, 'Failure has an invented goal time')
        safe = goal_reached and not schema_contact and not rollover and not bounded
        require(safe == bool(outcome['schema_safe_goal_reached']), 'Schema-safe goal field disagrees with measured consequences')
        planning_document = read(folder/'online_planning_summary.json')
        require(planning_document.get('planning_mode', 'receding') == planning_mode, 'Planning summary mode differs from protocol')
        planning = planning_document['decisions']
        require(len(planning) == outcome['planning_decisions'], 'Missing/extra planning summaries')
        require(sum(bool(p['abstained']) for p in planning) == outcome['planning_abstentions'], 'Abstention count differs from all decisions')
        frames = [p['frame'] for p in planning]
        require(frames == sorted(set(frames)) and all(0 <= f < n for f in frames), 'Planning anchors are duplicated, unordered or future')
        if planning_mode == 'plan_once':
            require(frames == [0], 'Plan-once trial must contain exactly one decision at launch')
        row.update(audit_rejections(folder, planning, protocol, verify_decisions=args.verify_decision_rejections))
        row['planning_abstention_fraction'] = outcome['planning_abstentions']/max(1,len(planning))
        if 'command_planner_paused' in rich:
            paused = rich['command_planner_paused'][:-1]
            require(np.isfinite(paused).all(), 'Missing commanded pause state')
            row['commanded_zero_speed_pause_s'] = float(np.dot(paused, intervals['duration_s']))
        row['source_runtime_signature'] = canonical({'source':protocol['source_sha256'], 'runtime':read(folder/'simulation_provenance.json')['runtime_sha256']})
        costs = dict(protocol['cost_config']); costs.pop('energy_weight_s_per_kj')
        row['matched_settings_signature'] = canonical({'checkpoint':protocol['model_checkpoint_sha256'], 'costs_except_energy':costs,
            'mppi':protocol['mppi_config'], 'native_controller':protocol['native_controller'], 'replan_period_s':protocol['replan_period_s'],
            'seed':protocol['seed'], 'image_intervention':protocol['image_intervention'], 'speeds':protocol['candidate_speeds_mps'],
            'offsets':protocol['candidate_offsets_m'], 'candidate_policy':protocol.get('candidate_policy'), 'planning_mode':planning_mode,
            'horizon_s':row['declared_horizon_s']})
        row['observation_sha256'] = protocol['scene_observation_sha256']
        row['artifact_valid'] = True; row['verified_schema_safe_goal'] = safe
    except (OSError,ValueError,KeyError,IndexError,AssertionError) as error:
        row['errors'].append(str(error))
        if row['status']=='missing' and result is not None:
            row['status']='process_failed' if result.get('exit_code')!=0 else 'invalid_artifacts'
    return row


def summarize(rows):
    weights = sorted(set(r['energy_weight_s_per_kj'] for r in rows if r['energy_weight_s_per_kj'] is not None))
    result = []
    for weight in weights:
        arm = [r for r in rows if r['energy_weight_s_per_kj'] == weight]
        result.append({'energy_weight_s_per_kj':weight, 'declared_trials':len(arm),
            'process_successes':sum(r['process_exit_code'] == 0 for r in arm), 'valid_artifacts':sum(r['artifact_valid'] for r in arm),
            'schema_safe_full_goals':sum(r['verified_schema_safe_goal'] for r in arm),
            'unconditional_safe_success_fraction':sum(r['verified_schema_safe_goal'] for r in arm)/len(arm),
            'timeouts':sum(r['status'] == 'timeout' for r in arm),
            'trials_with_abstentions':sum((r.get('planning_abstentions') or 0)>0 for r in arm),
            'collision_trials':sum(bool(r.get('schema_contact')) for r in arm),
            'blockage_trials':sum(bool(r.get('bounded_blockage_v1')) for r in arm),
            'rejected_fresh_candidates_recorded':sum(r.get('rejected_fresh_candidates') or 0 for r in arm),
            'decisions_with_rejected_fresh_recorded':sum(r.get('decisions_with_rejected_fresh') or 0 for r in arm),
            'trials_with_verified_rejection_counters':sum(r.get('rejected_fresh_candidates') is not None for r in arm)})
    return result


def paired_tradeoffs(rows, baseline=0.):
    pairs = []
    for candidate in rows:
        if candidate['energy_weight_s_per_kj'] in (None, baseline): continue
        bases = [r for r in rows if r['scene_id'] == candidate['scene_id'] and r['energy_weight_s_per_kj'] == baseline]
        row = {'scene_id':candidate['scene_id'], 'candidate_weight_s_per_kj':candidate['energy_weight_s_per_kj'],
               'candidate_trial_id':candidate['trial_id'], 'eligible':False, 'reason':'No unique declared baseline',
               'work_saving_fraction':None, 'time_increase_fraction':None}
        if len(bases) == 1:
            base = bases[0]; row['baseline_trial_id'] = base['trial_id']
            matched = (base.get('matched_settings_signature') is not None and base.get('matched_settings_signature') == candidate.get('matched_settings_signature')
                       and base.get('source_runtime_signature') == candidate.get('source_runtime_signature')
                       and base.get('observation_sha256') == candidate.get('observation_sha256'))
            if not matched: row['reason'] = 'Compared non-energy settings/runtime/observation differ or are unverified'
            elif not (base['verified_schema_safe_goal'] and candidate['verified_schema_safe_goal']): row['reason'] = 'Both trials did not reach verified schema-safe full goals'
            elif not (base['positive_work_kj'] > 0 and base['goal_time_s'] > 0): row['reason'] = 'Baseline denominators are not positive'
            else:
                row.update(eligible=True, reason=None, work_saving_fraction=1-candidate['positive_work_kj']/base['positive_work_kj'],
                           time_increase_fraction=candidate['goal_time_s']/base['goal_time_s']-1)
        pairs.append(row)
    return pairs


def select_energy(rows, spec, validity_errors, *, metadata_verified=False):
    errors = list(validity_errors); weights = spec['energy_weights_s_per_kj']; scenes = sorted(set(r['scene_id'] for r in rows))
    if len(rows) != spec['expected_trial_count']: errors.append('Declared validation grid trial count is not 24')
    if len(scenes) != spec['expected_scene_count']: errors.append('Validation grid scene count is not six')
    for scene in scenes:
        values = [r['energy_weight_s_per_kj'] for r in rows if r['scene_id'] == scene]
        if sorted(values, key=lambda v:-1 if v is None else v) != weights: errors.append(f'{scene}: weights are missing, duplicated or unexpected')
    if any(r.get('split') != 'val' for r in rows): errors.append('Validation selection contains a non-validation or unknown scene')
    if any(not r['artifact_valid'] for r in rows): errors.append('Validation grid has missing/invalid artifacts or failed execution')
    if len(set(r.get('matched_settings_signature') for r in rows)) != 1: errors.append('Non-energy settings are missing or differ across validation grid')
    if len(set(r.get('source_runtime_signature') for r in rows)) != 1: errors.append('Source/runtime evidence is missing or differs across validation grid')
    if not metadata_verified: errors.append('Preregistered model/budget metadata not verified')
    pairs = paired_tradeoffs(rows, spec['baseline_weight_s_per_kj'])
    arms = summarize(rows); counts = {r['energy_weight_s_per_kj']:r['schema_safe_full_goals'] for r in arms}
    evaluations = []
    for weight in weights[1:]:
        eligible = [r for r in pairs if r['candidate_weight_s_per_kj'] == weight and r['eligible']]
        work = float(np.mean([r['work_saving_fraction'] for r in eligible])) if eligible else None
        time = float(np.mean([r['time_increase_fraction'] for r in eligible])) if eligible else None
        gates = {'at_least_three_safe_pairs':len(eligible) >= spec['minimum_schema_safe_scene_pairs'],
                 'mean_work_saving_at_least_five_percent':work is not None and work >= spec['minimum_mean_work_saving_fraction']-1e-12,
                 'mean_time_increase_at_most_twentyfive_percent':time is not None and time <= spec['maximum_mean_time_increase_fraction']+1e-12,
                 'no_unconditional_success_count_drop':counts.get(weight,0) >= counts.get(0.,0)}
        evaluations.append({'energy_weight_s_per_kj':weight,'paired_safe_scenes':len(eligible),'mean_work_saving_fraction':work,
            'mean_time_increase_fraction':time,'unconditional_safe_goal_count':counts.get(weight,0),'gates':gates,'qualifies':all(gates.values())})
    if errors:
        return {'selection_state':'blocked_incomplete_or_invalid_grid','selected_weight_s_per_kj':None,'gate_passed':False,
                'freeze_allowed':False,'errors':sorted(set(errors)),'candidate_evaluations':evaluations,'paired_scene_tradeoffs':pairs}
    qualifying = [r['energy_weight_s_per_kj'] for r in evaluations if r['qualifies']]
    return {'selection_state':'qualified' if qualifying else 'complete_valid_grid_no_qualifier_default_retained',
            'selected_weight_s_per_kj':min(qualifying) if qualifying else .02,'gate_passed':bool(qualifying),'freeze_allowed':True,
            'baseline_unconditional_safe_goal_count':counts.get(0.,0),'candidate_evaluations':evaluations,'paired_scene_tradeoffs':pairs,
            'claim':'Predeclared gate satisfied' if qualifying else 'Default retained; no energy-efficiency or optimum claim'}


def write_report(out, rows, summary, pairs, audit, selection, *, plots=True):
    fields = ['trial_id','scene_id','planning_mode','energy_weight_s_per_kj','process_exit_code','artifact_valid','verified_schema_safe_goal','status',
              'elapsed_s','goal_time_s','goal_progress_m','positive_work_kj','schema_contact','schema_rollover','bounded_blockage_v1',
              'planning_decisions','planning_abstentions','planning_abstention_fraction','commanded_zero_speed_pause_s',
              'rejected_fresh_candidates','decisions_with_rejected_fresh','max_rejected_fresh_per_decision']
    with (out/'all_trials.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows: writer.writerow({k:row.get(k) for k in fields})
    text = ['# Declared online cohort report', '',
            f"Every declared trial is counted: **{len(rows)} trials**, **{sum(r['verified_schema_safe_goal'] for r in rows)} verified schema-safe full goals**.",
            'Process completion and physical success are separate. Failed, missing, timed-out and abstaining trials remain in denominators.', '',
            '| Energy weight (s/kJ) | Safe full goals / all trials | Valid artifacts | Timeouts | Trials with abstentions |',
            '|---:|---:|---:|---:|---:|']
    for a in summary:
        text.append(f"| {a['energy_weight_s_per_kj']:g} | {a['schema_safe_full_goals']} / {a['declared_trials']} | {a['valid_artifacts']} | {a['timeouts']} | {a['trials_with_abstentions']} |")
    text += ['', '## Every individual trial', '', '| Scene | Weight | Process exit | Physical status | Safe goal | Progress (m) | Work consumed (kJ) | Abstentions |', '|---|---:|---:|---|---|---:|---:|---:|']
    number = lambda v: 'unavailable' if v is None else f'{v:.2f}'
    for r in rows:
        text.append(f"| {r['scene_id']} | {r['energy_weight_s_per_kj']} | {r['process_exit_code']} | {r['status']} | {r['verified_schema_safe_goal']} | {number(r.get('goal_progress_m'))} | {number(r.get('positive_work_kj'))} | {r.get('planning_abstentions','unavailable')} |")
    text += ['', 'Individual failed-run work is descriptive consumption, not evidence of an efficiency improvement.', '', '## Paired safe-goal tradeoffs', '']
    eligible = [p for p in pairs if p['eligible']]
    if not eligible: text.append('**No eligible paired safe full goals.** Time/work tradeoff fractions are unavailable; shorter or failed runs cannot be called more efficient.')
    else:
        text += ['| Scene | Weight | Work saving | Time increase |','|---|---:|---:|---:|']
        for p in eligible: text.append(f"| {p['scene_id']} | {p['candidate_weight_s_per_kj']:g} | {p['work_saving_fraction']:.1%} | {p['time_increase_fraction']:.1%} |")
    text += ['', '## Selection status', '', f"`{selection['selection_state']}`. Selected weight: `{selection.get('selected_weight_s_per_kj')}`; gate passed: `{selection.get('gate_passed',False)}`.",
             'The full preregistered validation gate requires at least three safe pairs out of six scenes, mean work saving at least 5%, mean time increase at most 25%, and no unconditional safe-success count drop. Incomplete or invalid grids block selection and freeze.', '',
            'This report does not open protected test scenes or establish generalization from a pilot.', '']
    if selection.get('training_completion_evidence'):
        evidence = selection['training_completion_evidence']
        text += [f"Declared checkpoint kind: **{evidence['checkpoint_kind']}**; chosen step **{evidence['checkpoint_step']}** from completed update budget **{evidence['completed_step']}**. Planning mode: **{selection['planning_mode']}**. This declaration is specific to this grid and does not replace another grid's declared primary model.", '']
    if audit['errors']: text += ['Audit issues:']+['- '+e for e in audit['errors']]+['']
    if any(r.get('rejected_fresh_candidates') is not None for r in rows):
        text += ['## Rejected fresh reference counters', '',
                 'Counts concern invalid fresh geometry proposals, not predicted or measured terrain risk. Missing counter evidence stays unavailable.', '',
                 '| Energy weight | Rejected fresh proposals | Decisions with rejections | Trials with verified counters |',
                 '|---:|---:|---:|---:|']
        for a in summary:
            text.append(f"| {a['energy_weight_s_per_kj']:g} | {a['rejected_fresh_candidates_recorded']} | {a['decisions_with_rejected_fresh_recorded']} | {a['trials_with_verified_rejection_counters']} / {a['declared_trials']} |")
        text.append('')
    (out/'report.md').write_text('\n'.join(text))
    if not plots:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(13,max(5,.31*len(rows))),layout='constrained')
    x=np.arange(len(summary)); values=[a['unconditional_safe_success_fraction'] for a in summary]
    axes[0].bar(x,values,color='#2477aa');axes[0].set_ylim(0,1.12);axes[0].set_xticks(x,[f"{a['energy_weight_s_per_kj']:g}" for a in summary])
    for i,a in enumerate(summary):axes[0].text(i,values[i]+.03,f"{a['schema_safe_full_goals']}/{a['declared_trials']}",ha='center')
    axes[0].set_title('Unconditional schema-safe full-goal success');axes[0].set_ylabel('Fraction of every declared trial');axes[0].set_xlabel('Energy weight (s/kJ)')
    positions=np.arange(len(rows));axes[1].barh(positions,[np.nan if r.get('goal_progress_m') is None else r['goal_progress_m'] for r in rows],color=['#208b65' if r['verified_schema_safe_goal'] else '#b65b52' for r in rows])
    for i,row in enumerate(rows):
        if row.get('goal_progress_m') is None:
            axes[1].text(0,i,'unavailable: incomplete rollout',va='center',fontsize=7,color='#666666')
    axes[1].set_yticks(positions,[r['trial_id'].replace('diverse_v1_val_','') for r in rows],fontsize=8);axes[1].set_xlabel('Measured net goal progress (m)')
    axes[1].set_title('Every trial: red is not a verified safe full goal')
    fig.savefig(out/'cohort_overview.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True);parser.add_argument('--tasks',type=Path,required=True)
    parser.add_argument('--code-root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--path-map',action='append',default=[],help='REMOTE_PREFIX=LOCAL_PREFIX for mirrored immutable inputs')
    parser.add_argument('--select-validation-energy',action='store_true')
    parser.add_argument('--no-plots',action='store_true',help='Write the complete audited JSON, CSV and Markdown without importing matplotlib')
    parser.add_argument('--verify-decision-rejections',action='store_true',help='Cross-check every saved rejected-reference list against the hashed planning summary; run near the data to avoid large transfers')
    parser.add_argument('--selection-spec',type=Path,default=ROOT/'artifacts/traverse/fdm_diverse_v1_20260909/online_tasks/energy_selection_spec_v1.json')
    args=parser.parse_args();mappings=[]
    for value in args.path_map:
        old,sep,new=value.partition('=')
        if not sep:parser.error('--path-map requires REMOTE_PREFIX=LOCAL_PREFIX')
        mappings.append((Path(old),Path(new).resolve()))
    args.batch=args.batch.resolve();args.code_root=args.code_root.resolve()
    tasks_document=read(args.tasks)
    tasks=tasks_document['tasks'];names=[t['id'] for t in tasks]
    if any(t.get('split')=='test' or '_test_' in t['arguments']['case'] or '_test_' in t['id'] for t in tasks):parser.error('Protected test tasks are forbidden')
    if any(read(mapped(t['arguments']['case'],mappings)).get('split') not in ('train','val') for t in tasks):parser.error('All validation-reporter cases must declare train/val before any outcome read')
    declaration=read(args.batch/'declaration.json');batch=read(args.batch/'batch_result.json')
    if args.out.exists() and any(args.out.iterdir()):parser.error('Use a new empty report directory')
    args.out.mkdir(parents=True,exist_ok=True);errors=[]
    def check(condition,message):
        if not condition:errors.append(message)
    check(len(names)==len(set(names)),'Task declaration contains duplicate IDs')
    check(tasks==declaration['tasks'],'Provided tasks differ from recorded declaration')
    check(sha(args.tasks)==declaration['tasks_sha256'],'Task file checksum differs from recorded declaration')
    check(sha(args.code_root/'source_manifest.json')==declaration['source_manifest_sha256'],'Source manifest checksum differs')
    if 'source_manifest_sha256' in tasks_document:
        check(tasks_document['source_manifest_sha256']==declaration['source_manifest_sha256'],'Executed snapshot differs from predeclared task-file snapshot')
    check(sha(args.code_root/'scripts/traverse_fdm_rgbd_diverse_online_batch.py')==declaration['batch_script_sha256'],'Batch runner checksum differs')
    results=batch['results'];result_names=[r['id'] for r in results]
    check(len(results)==len(set(result_names)),'Batch result contains duplicate IDs')
    check(set(result_names)==set(names),'Batch results omit declared IDs or include undeclared IDs')
    check(batch['declared']==len(tasks) and batch['finished']==len(results)==len(tasks),'Declared/finished trial counts disagree')
    check(set(batch['failures'])==set(r['id'] for r in results if r.get('exit_code')!=0),'Batch failure list disagrees with per-trial exits')
    lookup={r['id']:r for r in results};rows=[audit_trial(t,lookup.get(t['id']),args,mappings) for t in tasks]
    errors += [f"{r['trial_id']}: {e}" for r in rows for e in r['errors']]
    spec=read(args.selection_spec);summary=summarize(rows);pairs=paired_tradeoffs(rows)
    selection={'selection_state':'not_requested_diagnostic_only','selected_weight_s_per_kj':None,'gate_passed':False,'freeze_allowed':False}
    if args.select_validation_energy:
        metadata_verified=False;training_evidence=None
        try:
            require(tasks_document.get('energy_selection_spec_sha256')==sha(args.selection_spec),'Task file does not bind the exact preregistered selection spec')
            checkpoints=set(r.get('checkpoint_path') for r in rows)
            require(len(checkpoints)==1 and None not in checkpoints,'Grid does not use a single verifiable checkpoint')
            import torch
            checkpoint=torch.load(next(iter(checkpoints)),map_location='cpu',weights_only=False)
            training_args=checkpoint['args'];config=checkpoint['model_config'];declared_model=spec['model_declaration']
            require(float(config['horizon'])*float(config['dt'])==declared_model['forecast_horizon_s'],'Wrong preregistered model horizon')
            require(int(training_args['seed'])==declared_model['training_seed'],'Wrong preregistered training seed')
            require(int(training_args['steps'])==declared_model['training_update_budget'],'Wrong preregistered training update budget')
            status_path=Path(next(iter(checkpoints))).parent/'status.json'
            training_status=read(status_path)
            frozen=tasks_document['frozen_training']
            checkpoint_kind=verify_checkpoint_kind(next(iter(checkpoints)),checkpoint['step'],training_status,declared_model,frozen)
            planning_mode=spec.get('planning_mode','receding')
            require(planning_mode in ('receding','plan_once'), 'Unknown selection-spec planning mode')
            require(all(row.get('planning_mode') == planning_mode for row in rows),'Grid planning mode differs from selection spec')
            require(sha(next(iter(checkpoints)))==frozen['checkpoint_sha256'],'Checkpoint differs from predeclared frozen training selection')
            require(int(checkpoint['step'])==int(frozen['checkpoint_step']),'Checkpoint step differs from predeclared selection')
            require(int(training_status['step'])==int(frozen['training_completed_step']),'Training completion differs from predeclared frozen status')
            require(sha(status_path)==frozen['status_sha256'],'Training status checksum differs from predeclared frozen status')
            require(checkpoint['provenance']['data']['manifest.json']==frozen['train_val_pack_manifest_sha256'],'Checkpoint data manifest differs from predeclared train/validation pack')
            training_evidence={'status_path':str(status_path),'status_sha256':sha(status_path),
                'state':training_status['state'],'completed_step':training_status['step'],'best_step':training_status['best_step'],
                'checkpoint_kind':checkpoint_kind,'checkpoint_step':int(checkpoint['step']),
                'checkpoint_sha256':sha(next(iter(checkpoints))),'training_seed':training_args['seed'],
                'declared_update_budget':training_args['steps'],'model_horizon_s':float(config['horizon'])*float(config['dt'])}
            metadata_verified=True
        except (OSError,ValueError,KeyError,TypeError) as error:errors.append(str(error))
        selection=select_energy(rows,spec,errors,metadata_verified=metadata_verified)
        selection['training_completion_evidence']=training_evidence
        selection['planning_mode']=spec.get('planning_mode','receding')
        selection['model_declaration']=spec['model_declaration']
    audit={'errors':errors,'declared_trial_count':len(tasks),'reported_trial_count':len(rows),'all_declared_trials_retained':True,
           'task_file_sha256':sha(args.tasks),'selection_spec_sha256':sha(args.selection_spec),
           'source_manifest_sha256':sha(args.code_root/'source_manifest.json'),'batch_result_sha256':sha(args.batch/'batch_result.json'),
           'declaration_sha256':sha(args.batch/'declaration.json'),'slurm_job_id':declaration.get('slurm_job_id')}
    dump(args.out/'cohort_report.json',{'audit':audit,'unconditional_arms':summary,'all_trials':rows,'paired_safe_tradeoffs':pairs,
         'energy_selection':selection,'script_sha256':sha(__file__),'interpretation':'Unconditional success first; no failed-run efficiency claims; pilot report does not select a weight'})
    write_report(args.out,rows,summary,pairs,audit,selection,plots=not args.no_plots)
    print(json.dumps({'out':str(args.out.resolve()),'trials':len(rows),'schema_safe_full_goals':sum(r['verified_schema_safe_goal'] for r in rows),
                      'audit_errors':len(errors),'selection':selection['selection_state']}))
    return 2 if errors or selection['selection_state']=='blocked_incomplete_or_invalid_grid' else 0


if __name__=='__main__':raise SystemExit(main())
