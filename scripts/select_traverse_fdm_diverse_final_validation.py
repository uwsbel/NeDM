#!/usr/bin/env python3
"""Apply a declared configuration ranking to three audited validation grids.

This writes a validation-selection result, never a concrete test freeze.
No images, physics rollouts, checkpoint tensors or protected test files are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from report_traverse_fdm_diverse_online_cohort import read, sha, require, dump, select_energy


def rank_candidates(candidates):
    return sorted(candidates, key=lambda row:(-row['selected_weight_safe_goals'],
        -row['zero_weight_safe_goals'], -int(row['energy_gate_passed']), row['preference_order']))


def audit_cohort(configuration, tasks_path, report_path, spec_path):
    tasks, report, spec = read(tasks_path), read(report_path), read(spec_path)
    audit = report['audit']; rows = report['all_trials']; previous = report['energy_selection']
    require(not audit['errors'], 'Source cohort audit contains errors')
    require(audit['all_declared_trials_retained'] and audit['declared_trial_count'] == audit['reported_trial_count'] == 24, 'Source cohort omitted trials')
    require(sha(tasks_path) == audit['task_file_sha256'], 'Task manifest differs from audited cohort')
    require(sha(spec_path) == audit['selection_spec_sha256'] == tasks['energy_selection_spec_sha256'], 'Energy spec differs from task/audit declaration')
    require(tasks['source_manifest_sha256'] == audit['source_manifest_sha256'], 'Audited source differs from task declaration')
    require(Path(tasks_path).name == configuration['grid'], 'Wrong declared grid for this configuration')
    mode = configuration['planning_mode']; kind = configuration['checkpoint_kind']
    require(spec.get('planning_mode','receding') == mode, 'Configuration planning mode differs from energy spec')
    require(spec['model_declaration'].get('checkpoint_kind','best') == kind, 'Configuration checkpoint kind differs from energy spec')
    require(all(row.get('planning_mode','receding') == mode for row in rows), 'Mixed or incorrect executed planning modes')
    evidence = previous['training_completion_evidence']; frozen = tasks['frozen_training']
    require(evidence is not None and evidence['state'] == 'complete', 'Missing completed-training audit evidence')
    require(evidence.get('checkpoint_kind','best') == kind, 'Audited checkpoint kind differs from configuration')
    require(evidence['checkpoint_sha256'] == frozen['checkpoint_sha256'], 'Audited checkpoint differs from frozen task')
    require(evidence['status_sha256'] == frozen['status_sha256'], 'Audited training status differs from frozen task')
    require(evidence['completed_step'] == frozen['training_completed_step'] == 5000, 'Incomplete declared training budget')
    require(all(row.get('checkpoint_sha256') == frozen['checkpoint_sha256'] and row.get('checkpoint_step') == frozen['checkpoint_step'] for row in rows), 'Mixed or incorrect checkpoint in source cohort')
    require(frozen['checkpoint_step'] == (evidence['best_step'] if kind == 'best' else 5000), 'Checkpoint selection kind has incorrect update step')
    selection = select_energy(rows, spec, [], metadata_verified=True)
    require(selection['freeze_allowed'], 'Source energy grid is incomplete or invalid')
    for field in ('selection_state','selected_weight_s_per_kj','gate_passed','freeze_allowed'):
        require(selection[field] == previous[field], 'Recomputed energy gate disagrees with source report')
    weight = selection['selected_weight_s_per_kj']
    selected_rows = [row for row in rows if row['energy_weight_s_per_kj'] == weight]
    zero_rows = [row for row in rows if row['energy_weight_s_per_kj'] == 0.]
    require(len(selected_rows) == len(zero_rows) == 6, 'Configuration denominator is not six scenes')
    return {**configuration, 'selected_weight_s_per_kj':weight,
            'selected_weight_safe_goals':sum(row['verified_schema_safe_goal'] for row in selected_rows),
            'zero_weight_safe_goals':sum(row['verified_schema_safe_goal'] for row in zero_rows),
            'declared_scenes':6, 'energy_gate_passed':selection['gate_passed'],
            'checkpoint_sha256':frozen['checkpoint_sha256'], 'checkpoint_step':frozen['checkpoint_step'],
            'source_manifest_sha256':audit['source_manifest_sha256'],
            'task_file_sha256':sha(tasks_path), 'cohort_report_sha256':sha(report_path),
            'energy_spec_sha256':sha(spec_path), 'energy_gate':selection}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rule',required=True,type=Path)
    parser.add_argument('--rule-sha256',required=True)
    parser.add_argument('--cohort',nargs=3,action='append',required=True,metavar=('TASKS','REPORT','SPEC'))
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    require(sha(args.rule) == args.rule_sha256, 'Final selection rule checksum differs from preregistration')
    rule = read(args.rule)
    require(rule['version'] == 'final_validation_selection_rule_v1', 'Unknown final configuration rule')
    declared = rule['eligible_configurations']
    require(len(declared) == 3 and len(set(c['grid'] for c in declared)) == 3, 'Expected three unique declared configurations')
    require(len(args.cohort) == 3, 'All three complete validation cohorts are required before selection')
    inputs = {Path(values[0]).name:values for values in args.cohort}
    require(set(inputs) == set(c['grid'] for c in declared), 'Provided grids do not exactly match preregistered configurations')
    require(not args.out.exists() or not any(args.out.iterdir()), 'Use a new empty selection directory')
    args.out.mkdir(parents=True,exist_ok=True)
    candidates, errors = [], []
    for configuration in declared:
        try:
            candidates.append(audit_cohort(configuration, *map(Path,inputs[configuration['grid']])))
        except (ValueError,KeyError,OSError,TypeError) as error:
            errors.append(configuration['grid']+': '+str(error))
    ranked = rank_candidates(candidates) if not errors else []
    result = {'rule_sha256':sha(args.rule), 'script_sha256':sha(__file__), 'errors':errors,
              'selection_state':'validation_selected' if ranked else 'blocked_incomplete_or_invalid_cohorts',
              'concrete_freeze_written':False, 'concrete_freeze_required_before_test':True,
              'protected_test_opened':False, 'candidates':candidates, 'ranked_configurations':ranked,
              'selected_configuration':ranked[0] if ranked else None,
              'interpretation':'Selection uses whole-scene safe-goal counts and the predeclared tie breaks; it does not establish held-out test performance or replace reporting of the original primary baseline.'}
    dump(args.out/'selection.json',result)
    lines = ['# Final configuration selection from validation', '',
             'All three configurations must have complete, audited 24-trial grids. No protected test data is read and no test freeze is written.', '',
             '| Planning mode | Checkpoint | Selected energy weight | Safe goals at selected weight | Safe goals at zero weight | Energy gate |',
             '|---|---|---:|---:|---:|---|']
    for row in candidates:
        lines.append(f"| {row['planning_mode']} | {row['checkpoint_kind']} step {row['checkpoint_step']} | {row['selected_weight_s_per_kj']} | {row['selected_weight_safe_goals']} / 6 | {row['zero_weight_safe_goals']} / 6 | {row['energy_gate_passed']} |")
    if ranked:
        winner = ranked[0]
        lines += ['',f"Validation-selected configuration: **{winner['planning_mode']}, {winner['checkpoint_kind']} step {winner['checkpoint_step']}, energy weight {winner['selected_weight_s_per_kj']} s/kJ**.",
                  'A concrete checksum-bound test freeze is still required. Original best-checkpoint receding planning remains a reported baseline.']
    if errors: lines += ['', 'Selection blocked:']+['- '+error for error in errors]
    (args.out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'out':str(args.out),'state':result['selection_state'], 'errors':errors,
                      'selected':None if not ranked else {key:ranked[0][key] for key in ('planning_mode','checkpoint_kind','selected_weight_s_per_kj','selected_weight_safe_goals')}}))
    return 2 if errors else 0


if __name__ == '__main__': raise SystemExit(main())
