#!/usr/bin/env python3
"""Report a protected 30-trial cohort only after an exact concrete-freeze gate.

The required caller-supplied freeze checksum is an authorization boundary.
No protected case, observation or outcome is decoded before the complete
freeze/task/source/model/cost/observation membership checks succeed.
There is no protected-test parameter selection or model selection in this tool.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import json

from report_traverse_fdm_diverse_online_cohort import (
    sha, read, require, require_hash, mapped, dump, audit_trial, paired_tradeoffs, expected_weight)

ARMS = ('selected_rgbd_time','selected_rgbd_energy','matched_blank_time',
        'matched_blank_energy','original_best_receding_time')


def verify_freeze(args, mappings):
    """Gate uses declarations and byte hashes; no protected outcomes are read."""
    require_hash(args.freeze,args.freeze_sha256,'Explicitly authorized concrete freeze')
    freeze = read(args.freeze)
    require(freeze['schema'] == 'fdm_protected_test_freeze_v1','Unknown concrete-freeze schema')
    require(freeze.get('test_unseal_authorized') is True,'Concrete freeze does not authorize protected evaluation')
    require(freeze['test_outcomes_opened_before_freeze'] is False,'Freeze does not attest protected outcome seal')
    tasks_path = mapped(freeze['tasks_file'],mappings)
    require(args.tasks.resolve() == tasks_path.resolve(),'Provided task path differs from authorized freeze')
    require_hash(tasks_path,freeze['tasks_sha256'],'Frozen protected tasks')
    require_hash(args.code_root/'source_manifest.json',freeze['source_manifest_sha256'],'Frozen source manifest')
    source = read(args.code_root/'source_manifest.json')['files']
    document = read(tasks_path)
    require(document['source_manifest_sha256'] == freeze['source_manifest_sha256'],'Tasks disagree with frozen source')
    require_hash(mapped(freeze['protected_collection_manifest_file'],mappings),freeze['protected_collection_manifest_sha256'],'Original sealed collection manifest')
    require_hash(args.selection_rule,freeze['selection_rule_sha256'],'Predeclared final validation-selection rule')
    selection=None
    if 'validation_selection_file' in freeze:
        selection_path=mapped(freeze['validation_selection_file'],mappings)
        require_hash(selection_path,freeze['validation_selection_sha256'],'Frozen validation selection')
        selection=read(selection_path)
        known={row['grid']:row['cohort_report_sha256'] for row in selection['candidates']}
        for key,digest in freeze['validation_report_sha256'].items():
            if Path(key).is_absolute():require_hash(mapped(key,mappings),digest,'Frozen validation cohort evidence')
            else:require(known.get(key)==digest,'Frozen cohort hash differs from checksum-bound validation selection')
    else:
        for path,digest in freeze['validation_report_sha256'].items():
            require(Path(path).is_absolute(),'Relative validation evidence requires a checksum-bound selection file')
            require_hash(mapped(path,mappings),digest,'Frozen validation selection/report evidence')
    tasks = document['tasks']; scenes = freeze['test_scene_ids']
    require(len(scenes) == len(set(scenes)) == 6,'Freeze must specify exactly six unique protected scenes')
    require(set(freeze['arms']) == set(ARMS) and len(freeze['arms']) == 5,'Freeze must specify exactly the five declared arms')
    require(len(tasks) == 30 and len({task['id'] for task in tasks}) == 30,'Protected declaration must retain all 30 unique trials')
    membership = [(task['scene_id'],task['arm']) for task in tasks]
    require(len(set(membership)) == 30 and set(membership) == {(scene,arm) for scene in scenes for arm in ARMS},'Protected task scene/arm membership differs from freeze')
    for field,argument in [('checkpoint_files','checkpoint'),('cost_files','cost-config'),('observation_files','scene-observation')]:
        referenced = {task['arguments'][argument] for task in tasks}
        require(referenced == set(freeze[field]),f'Frozen {field} do not exactly cover declared task inputs')
        for path,digest in freeze[field].items():
            require_hash(mapped(path,mappings),digest,'Frozen '+field)
    primary = freeze['primary']
    require(primary['planning_mode'] in ('receding','plan_once') and primary['checkpoint_kind'] in ('best','last'),'Unknown frozen primary configuration')
    checkpoints_by_arm={arm:{task['arguments']['checkpoint'] for task in tasks if task['arm']==arm} for arm in ARMS}
    require(all(len(paths)==1 for paths in checkpoints_by_arm.values()),'A protected arm mixes checkpoints across scenes')
    require(checkpoints_by_arm['selected_rgbd_time']==checkpoints_by_arm['selected_rgbd_energy'],'RGBD time/energy arms use different checkpoints')
    require(checkpoints_by_arm['matched_blank_time']==checkpoints_by_arm['matched_blank_energy'],'Blank time/energy arms use different checkpoints')
    if 'rgbd_checkpoint_path' in primary:require(checkpoints_by_arm['selected_rgbd_time']=={primary['rgbd_checkpoint_path']},'Executed RGBD checkpoint differs from frozen primary path')
    if 'blank_checkpoint_path' in primary:require(checkpoints_by_arm['matched_blank_time']=={primary['blank_checkpoint_path']},'Executed blank checkpoint differs from frozen primary path')
    if selection is not None:
        winner=selection['selected_configuration']
        require(not selection['errors'] and selection['selection_state']=='validation_selected','Frozen validation selection is incomplete')
        require(winner['planning_mode']==primary['planning_mode'] and winner['checkpoint_kind']==primary['checkpoint_kind']
                and winner['selected_weight_s_per_kj']==primary['energy_weight_s_per_kj'],'Concrete primary differs from the declared validation selection')
    for task in tasks:
        arguments=task['arguments']; arm=task['arm']
        require(task.get('split') == 'test','Protected task has incorrect declared split')
        require(Path(arguments['case']).stem == task['scene_id'],'Protected case path does not identify its declared scene')
        case_path=mapped(arguments['case'],mappings)
        relative=str(case_path.relative_to(args.code_root))
        require(relative in source,'Protected case is absent from the immutable source manifest')
        require_hash(case_path,source[relative],'Frozen protected case bytes')
        kind='best' if arm=='original_best_receding_time' else primary['checkpoint_kind']
        mode='receding' if arm=='original_best_receding_time' else primary['planning_mode']
        weight=primary['energy_weight_s_per_kj'] if arm.endswith('_energy') else 0.
        require(Path(arguments['checkpoint']).name == kind+'.pt','Arm checkpoint kind differs from concrete freeze')
        require(arguments.get('planning-mode','receding') == mode,'Arm planning mode differs from concrete freeze')
        require(expected_weight(arguments,mappings) == weight,'Arm energy weight differs from concrete freeze')
        require(task.get('energy_weight_s_per_kj',weight) == weight,'Arm energy metadata disagrees with frozen configuration')
    return freeze,document


def validate_training(freeze,tasks,mappings):
    import torch
    evidence={}
    for path,digest in freeze['checkpoint_files'].items():
        checkpoint=torch.load(mapped(path,mappings),map_location='cpu',weights_only=False)
        config=checkpoint['model_config']; arguments=checkpoint['args']
        require(config['horizon']*config['dt']==12. and arguments['seed']==11 and arguments['steps']==5000,'Protected model violates matched horizon/seed/training budget')
        require(checkpoint['provenance']['data']['manifest.json']==freeze['training_pack_manifest_sha256'],'Protected model training pack differs from freeze')
        status=read(mapped(path,mappings).parent/'status.json')
        require(status['state']=='complete' and status['step']==5000,'Protected checkpoint training run is incomplete')
        kind=Path(path).stem
        require(checkpoint['step']==(status['best_step'] if kind=='best' else 5000),'Protected checkpoint update step violates best/last kind')
        for task in tasks:
            if task['arguments']['checkpoint'] == path:
                expected='blank' if task['arm'].startswith('matched_blank_') else 'rgbd'
                require(config['arm']==arguments['arm']==expected,'Protected arm is not backed by the declared RGBD/blank trained model')
        evidence[path]={'checkpoint_sha256':digest,'step':checkpoint['step'],'kind':kind,
                        'trained_arm':config['arm'],'status_sha256':sha(mapped(path,mappings).parent/'status.json')}
    return evidence


def arm_summary(rows):
    result=[]
    for arm in ARMS:
        values=[row for row in rows if row['arm']==arm]
        result.append({'arm':arm,'declared_trials':len(values),'verified_artifacts':sum(r['artifact_valid'] for r in values),
            'schema_safe_full_goals':sum(r['verified_schema_safe_goal'] for r in values),
            'full_goals_reported':sum(bool(r.get('goal_reached')) for r in values),
            'timeouts':sum(r['status']=='timeout' for r in values),'process_failures':sum(r['process_exit_code']!=0 for r in values),
            'contact_trials':sum(bool(r.get('schema_contact')) for r in values),
            'blockage_trials':sum(bool(r.get('bounded_blockage_v1')) for r in values),
            'trials_with_abstentions':sum((r.get('planning_abstentions') or 0)>0 for r in values),
            'rejected_fresh_candidates':sum(r.get('rejected_fresh_candidates') or 0 for r in values)})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True);parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--tasks',type=Path,required=True);parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--code-root',type=Path,required=True);parser.add_argument('--selection-rule',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--path-map',action='append',default=[])
    parser.add_argument('--verify-decision-rejections',action='store_true')
    args=parser.parse_args();args.code_root=args.code_root.resolve();args.batch=args.batch.resolve()
    mappings=[]
    for mapping in args.path_map:
        old,sep,new=mapping.partition('=');require(bool(sep),'Path mapping requires OLD=NEW')
        mappings.append((Path(old),Path(new).resolve()))
    # This entire concrete-freeze gate precedes every protected result read.
    freeze,document=verify_freeze(args,mappings)
    training=validate_training(freeze,document['tasks'],mappings)
    args.authorized_test_scene_ids=frozenset(freeze['test_scene_ids'])
    declaration=read(args.batch/'declaration.json');batch=read(args.batch/'batch_result.json');tasks=document['tasks']
    require(declaration['tasks']==tasks and declaration['tasks_sha256']==freeze['tasks_sha256'],'Executed task declaration differs from concrete freeze')
    require(declaration['source_manifest_sha256']==freeze['source_manifest_sha256'],'Executed source differs from concrete freeze')
    require_hash(args.code_root/'scripts/traverse_fdm_rgbd_diverse_online_batch.py',declaration['batch_script_sha256'],'Protected batch runner')
    frozen_time=datetime.fromisoformat(freeze['frozen_utc'].replace('Z','+00:00')).timestamp()
    require(declaration['started_unix'] >= frozen_time,'Protected execution began before the concrete freeze')
    require(not args.out.exists() or not any(args.out.iterdir()),'Use a new empty protected report directory')
    args.out.mkdir(parents=True,exist_ok=True)
    errors=[];results=batch['results'];names=[row['id'] for row in results]
    if len(results)!=len(set(names)) or set(names)!={task['id'] for task in tasks}:errors.append('Missing, extra or duplicate declared results')
    if batch['declared']!=30 or batch['finished']!=len(results) or len(results)!=30:errors.append('Declared/finished counts differ from full 30-trial cohort')
    if set(batch['failures'])!={row['id'] for row in results if row.get('exit_code')!=0}:errors.append('Batch failure accounting disagrees with per-trial exits')
    lookup={row['id']:row for row in results};rows=[]
    for task in tasks:
        row=audit_trial(task,lookup.get(task['id']),args,mappings);row['arm']=task['arm'];rows.append(row)
        errors += [row['trial_id']+': '+error for error in row['errors']]
        if row['artifact_valid'] and row.get('split')!='test':errors.append(row['trial_id']+': measured case split is not protected test')
    matched=[]
    for row in rows:
        if row['arm']!='original_best_receding_time' and row['artifact_valid']:
            signature=json.loads(row['matched_settings_signature']);signature.pop('checkpoint')
            matched.append(json.dumps(signature,sort_keys=True))
    if len(set(matched))>1:errors.append('Selected RGBD and matched blank arms differ in non-model/non-energy settings')
    summaries=arm_summary(rows)
    pairs={name:paired_tradeoffs([row for row in rows if row['arm'] in arm_pair]) for name,arm_pair in {
        'selected_rgbd_energy_vs_time':('selected_rgbd_time','selected_rgbd_energy'),
        'matched_blank_energy_vs_time':('matched_blank_time','matched_blank_energy')}.items()}
    result={'audit':{'freeze_sha256':sha(args.freeze),'task_file_sha256':sha(args.tasks),
        'source_manifest_sha256':freeze['source_manifest_sha256'],'batch_result_sha256':sha(args.batch/'batch_result.json'),
        'declared_trial_count':30,'reported_trial_count':len(rows),'all_declared_trials_retained':True,'errors':errors,
        'slurm_job_id':declaration.get('slurm_job_id')},'script_sha256':sha(__file__),
        'shared_audit_script_sha256':sha(Path(__file__).with_name('report_traverse_fdm_diverse_online_cohort.py')),
        'frozen_primary':freeze['primary'],'training_evidence':training,'unconditional_arms':summaries,'all_trials':rows,
        'paired_safe_tradeoffs':pairs,'parameter_selection_performed':False,
        'interpretation':'All frozen arms and scenes retained; descriptive protected evaluation only. Within-modality time/work pairs require both full safe goals. No test-based model, coefficient or planning-mode selection.'}
    dump(args.out/'cohort_report.json',result)
    fields=['trial_id','scene_id','arm','planning_mode','energy_weight_s_per_kj','artifact_valid','process_exit_code',
            'status','verified_schema_safe_goal','elapsed_s','goal_time_s','goal_progress_m','positive_work_kj',
            'schema_contact','schema_rollover','bounded_blockage_v1','planning_abstentions','rejected_fresh_candidates']
    with (args.out/'all_trials.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({field:row.get(field) for field in fields})
    lines=['# Frozen protected cohort: all 30 trials','',f"Concrete freeze: `{sha(args.freeze)}`.",'',
        '| Arm | Safe full goals / all trials | Complete verified artifacts | Timeouts | Process failures |',
        '|---|---:|---:|---:|---:|']
    for row in summaries:lines.append(f"| {row['arm']} | {row['schema_safe_full_goals']} / {row['declared_trials']} | {row['verified_artifacts']} | {row['timeouts']} | {row['process_failures']} |")
    for name,values in pairs.items():
        lines += ['',f'## {name}','', '| Scene | Eligible safe pair | Work saving | Time increase | Reason if unavailable |','|---|---|---:|---:|---|']
        for row in values:
            number=lambda value:'unavailable' if value is None else f'{value:.2%}'
            lines.append(f"| {row['scene_id']} | {row['eligible']} | {number(row['work_saving_fraction'])} | {number(row['time_increase_fraction'])} | {row['reason'] or ''} |")
    lines += ['','No parameters or model choices were selected from this protected evaluation. Original best-checkpoint receding planning is a separate diagnostic baseline. All individual trials are retained in the adjacent CSV/JSON.']
    if errors:lines += ['','Audit issues:']+['- '+error for error in errors]
    (args.out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'out':str(args.out),'errors':errors,'unconditional_arms':summaries}))
    return 2 if errors else 0


if __name__=='__main__':raise SystemExit(main())
