#!/usr/bin/env python3
"""CPU fake-file checks; never uses actual protected cases or observations."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import json

from report_traverse_fdm_diverse_protected_cohort import verify_freeze, ARMS
from report_traverse_fdm_diverse_online_cohort import sha, audit_trial


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return str(path)


def main():
    blocked=audit_trial({'id':'fake_test_case','split':'test','arguments':{'case':'/nonexistent/fake_test_case.json'}},
                        {'exit_code':0},SimpleNamespace(batch=Path('/nonexistent')),[])
    assert blocked['status']=='unauthorized' and not blocked['artifact_valid']
    with tempfile.TemporaryDirectory(prefix='fdm_fake_freeze_') as temporary:
        root=Path(temporary);source=root/'source';source.mkdir()
        checkpoints={}
        for name in ('rgbd/last.pt','rgbd/best.pt','blank/last.pt'):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'FAKE checkpoint bytes; never deserialize')
            checkpoints[str(path)]=sha(path)
        time=write(root/'time.json',{'energy_weight_s_per_kj':0.})
        energy=write(root/'energy.json',{'energy_weight_s_per_kj':.02})
        costs={path:sha(path) for path in (time,energy)}
        scenes=[f'fake_test_scene_{i}' for i in range(6)];files={};observations={};tasks=[]
        for scene in scenes:
            case=source/(scene+'.json');case.write_bytes(b'FAKE opaque case; intentionally not JSON')
            files[case.name]=sha(case)
            observation=root/(scene+'.npz');observation.write_bytes(b'FAKE opaque observation; intentionally not NumPy')
            observations[str(observation)]=sha(observation)
            for arm in ARMS:
                checkpoint=root/('rgbd/best.pt' if arm=='original_best_receding_time' else 'blank/last.pt' if arm.startswith('matched_blank') else 'rgbd/last.pt')
                tasks.append({'id':scene+'_'+arm,'scene_id':scene,'split':'test','arm':arm,
                    'arguments':{'case':str(case),'checkpoint':str(checkpoint),'scene-observation':str(observation),
                        'cost-mode':'energy_risk' if arm.endswith('_energy') else 'time_risk',
                        'cost-config':energy if arm.endswith('_energy') else time,
                        'planning-mode':'receding' if arm=='original_best_receding_time' else 'plan_once'}})
        manifest=source/'source_manifest.json';write(manifest,{'files':files})
        taskfile=root/'tasks.json';write(taskfile,{'tasks':tasks,'source_manifest_sha256':sha(manifest)})
        collection=root/'collection.bin';collection.write_bytes(b'FAKE sealed collection manifest')
        rule=root/'rule.bin';rule.write_bytes(b'FAKE prior rule')
        report=root/'report.bin';report.write_bytes(b'FAKE prior validation evidence')
        freeze={'schema':'fdm_protected_test_freeze_v1','test_unseal_authorized':True,'test_outcomes_opened_before_freeze':False,
            'tasks_file':str(taskfile),'tasks_sha256':sha(taskfile),'source_manifest_sha256':sha(manifest),
            'protected_collection_manifest_file':str(collection),'protected_collection_manifest_sha256':sha(collection),
            'selection_rule_sha256':sha(rule),'validation_report_sha256':{str(report):sha(report)},
            'test_scene_ids':scenes,'arms':list(ARMS),'checkpoint_files':checkpoints,'cost_files':costs,
            'observation_files':observations,'primary':{'planning_mode':'plan_once','checkpoint_kind':'last','energy_weight_s_per_kj':.02}}
        frozen=root/'freeze.json';write(frozen,freeze)
        args=SimpleNamespace(freeze=frozen,freeze_sha256=sha(frozen),tasks=taskfile,code_root=source,selection_rule=rule)
        checked,document=verify_freeze(args,[])
        assert len(document['tasks'])==30 and checked['test_scene_ids']==scenes
        # Real observation/case parsers would fail on opaque bytes; successful gate
        # proves only declared bytes and metadata are inspected in this stage.
        def reject():
            try:verify_freeze(args,[])
            except ValueError:return
            raise AssertionError('Unbound protected inputs were accepted')
        args.freeze_sha256='0'*64;reject();args.freeze_sha256=sha(frozen)
        observation=Path(next(iter(observations)));saved=observation.read_bytes();observation.write_bytes(b'changed');reject();observation.write_bytes(saved)
        # Even a self-consistent rehashed task file cannot duplicate a scene/arm.
        taskcopy=json.loads(taskfile.read_text());taskcopy['tasks'][0]['arm']=taskcopy['tasks'][1]['arm']
        write(taskfile,taskcopy);freeze['tasks_sha256']=sha(taskfile);write(frozen,freeze);args.freeze_sha256=sha(frozen);reject()
    print('PASS fake-data freeze gate: exact authorization hash, complete 6x5 membership, source/models/costs/observations bound before decoding, and tampering rejection')


if __name__=='__main__':main()
