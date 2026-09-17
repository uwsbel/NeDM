#!/usr/bin/env python3
"""CPU arithmetic tests for the preregistered cohort selection, no physics/NN."""
from pathlib import Path
from copy import deepcopy
import json
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from report_traverse_fdm_diverse_online_cohort import select_energy, paired_tradeoffs, verify_checkpoint_kind, audit_rejections


def fixture():
    rows=[]
    for scene in range(6):
        for weight in (0.,.02,.2,.5):
            rows.append({'trial_id':f'scene{scene}_w{weight}', 'scene_id':f'scene{scene}', 'split':'val',
                'energy_weight_s_per_kj':weight, 'process_exit_code':0, 'artifact_valid':True,
                'verified_schema_safe_goal':False, 'positive_work_kj':100., 'goal_time_s':100.,
                'status':'timeout', 'matched_settings_signature':'fixed', 'source_runtime_signature':'fixed',
                'observation_sha256':f'image{scene}', 'planning_abstentions':0})
    return rows


def goals(rows,weight,scenes,work=90.,time=110.):
    for row in rows:
        if row['energy_weight_s_per_kj']==weight and int(row['scene_id'][5:]) in scenes:
            row.update(verified_schema_safe_goal=True,status='goal_reached',positive_work_kj=work,goal_time_s=time)


def main():
    spec=json.loads((ROOT/'artifacts/traverse/fdm_diverse_v1_20260909/online_tasks/energy_selection_spec_v1.json').read_text())
    rows=fixture()
    result=select_energy(rows,spec,[],metadata_verified=True)
    assert result['freeze_allowed'] and not result['gate_passed'] and result['selected_weight_s_per_kj']==.02
    assert all(p['work_saving_fraction'] is None and p['time_increase_fraction'] is None for p in paired_tradeoffs(rows))
    # Two favorable pairs cannot qualify, even with no unconditional loss.
    goals(rows,0.,range(2),100.,100.);goals(rows,.02,range(2))
    result=select_energy(rows,spec,[],metadata_verified=True)
    assert not result['gate_passed'] and not result['candidate_evaluations'][0]['gates']['at_least_three_safe_pairs']
    # Exactly three safe pairs, inclusive mean boundaries, smallest qualifier.
    rows=fixture();goals(rows,0.,range(3),100.,100.);goals(rows,.02,range(3),95.,125.);goals(rows,.2,range(3),80.,110.)
    result=select_energy(rows,spec,[],metadata_verified=True)
    assert result['gate_passed'] and result['selected_weight_s_per_kj']==.02
    # A better conditional subset cannot hide a lost unconditional success.
    goals(rows,0.,[3],100.,100.)
    result=select_energy(rows,spec,[],metadata_verified=True)
    assert not result['gate_passed'] and not result['candidate_evaluations'][0]['gates']['no_unconditional_success_count_drop']
    # Means weight scenes equally, not by work or travel distance.
    rows=fixture();goals(rows,0.,range(3),100.,100.);goals(rows,.2,range(3),90.,110.)
    chosen=[r for r in rows if r['energy_weight_s_per_kj']==.2 and r['verified_schema_safe_goal']]
    for row,work,time in zip(chosen,(100.,90.,95.),(150.,110.,115.)):
        row.update(positive_work_kj=work,goal_time_s=time)
    result=select_energy(rows,spec,[],metadata_verified=True)
    assert result['selected_weight_s_per_kj']==.2 and result['gate_passed']
    # Incomplete, duplicate, corrupt and mismatched settings must block freeze.
    for bad,errors,verified in ((rows[:-1],[],True),(rows+[deepcopy(rows[0])],[],True),(rows,['checksum mismatch'],True),(rows,[],False)):
        value=select_energy(bad,spec,errors,metadata_verified=verified)
        assert value['selected_weight_s_per_kj'] is None and not value['freeze_allowed']
    mismatch=deepcopy(rows);mismatch[1]['matched_settings_signature']='different'
    result=select_energy(mismatch,spec,[],metadata_verified=True)
    assert not result['freeze_allowed'] and result['selected_weight_s_per_kj'] is None
    # Validation-best and fixed-budget checkpoints have different status rules.
    status={'state':'complete','step':5000,'best_step':4000}
    declaration={'training_update_budget':5000}
    assert verify_checkpoint_kind('best.pt',4000,status,declaration,{})=='best'
    fixed={**declaration,'checkpoint_kind':'last'}
    assert verify_checkpoint_kind('last.pt',5000,status,fixed,{'checkpoint_kind':'last'})=='last'
    for arguments in [
        ('last.pt',5000,status,declaration,{}),
        ('best.pt',5000,status,declaration,{}),
        ('last.pt',4000,status,fixed,{}),
        ('last.pt',5000,{**status,'state':'running'},fixed,{}),
        ('last.pt',5000,status,fixed,{'checkpoint_kind':'best'}),
    ]:
        try: verify_checkpoint_kind(*arguments)
        except ValueError: pass
        else: raise AssertionError('Checkpoint kind/status mismatch was accepted')
    # Different checkpoints or planning modes cannot pool into one energy grid.
    for field in ('checkpoint','planning_mode'):
        mixed=deepcopy(rows)
        for index,row in enumerate(mixed):
            row['matched_settings_signature']=json.dumps({field:'primary' if index<12 else 'alternative'})
        result=select_energy(mixed,spec,[],metadata_verified=True)
        assert not result['freeze_allowed'] and result['selected_weight_s_per_kj'] is None
    # Rejection counts are checked against actual saved lists, not inferred from success.
    with tempfile.TemporaryDirectory() as temporary:
        folder=Path(temporary);(folder/'decisions').mkdir()
        protocol={'candidate_policy':{'invalid_fresh_candidates':'log'},'candidate_speeds_mps':[2,4,6],'candidate_offsets_m':[0,-22,22,-44,44]}
        planning=[{'frame':0,'time_s':0.,'rejected_candidate_count':1}]
        rejection={'candidate_origin':'fresh_family_00','reference_sha256':'a'*64,'reason':'reference_contract','detail':'cusp'}
        path=folder/'decisions/decision_00000.json'
        path.write_text(json.dumps({'frame':0,'time_s':0.,'rejected_candidate_references':[rejection]}))
        checked=audit_rejections(folder,planning,protocol,verify_decisions=True)
        assert checked['rejected_fresh_candidates']==1 and checked['rejection_details']=={'cusp':1}
        planning[0]['rejected_candidate_count']=0
        try: audit_rejections(folder,planning,protocol,verify_decisions=True)
        except ValueError: pass
        else: raise AssertionError('Corrupt rejection summary was accepted')
    print('PASS: all-trial failure accounting, safe-pair-only fractions, minimum three pairs, inclusive thresholds, equal-scene means, no success drop, smallest qualifying weight, explicit default, and invalid-grid freeze block')
    print('PASS: distinct best/last checkpoint status rules, mixed checkpoint/planning-mode rejection, and exact saved rejection-counter consistency')


if __name__=='__main__':main()
