#!/usr/bin/env python3
"""Capture a near-asset launch or execute a frozen 12 s diagnostic reference.

This wrapper imports immutable online_v9 collection/physics modules. It never
trains a model, searches candidates, filters predicted-risky routes, edits a
scene, changes native PID gains, or invents post-termination measurements.
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

ARRAY_FIELDS = ('waypoints', 'stations', 'headings', 'speeds')
ARRAY_HASH_CONVENTION = 'sha256(concat(field_utf8 + NUL + little_endian_int64_shape + little_endian_float64_C_bytes), fields=waypoints,stations,headings,speeds)'
ARTIFACTS = ('trajectory.npz', 'rich_intervals.npz', 'rich_telemetry.npz', 'anchor_state.npz',
             'outcome.json', 'online_protocol.json', 'simulation_provenance.json',
             'anchor_equality.json', 'online_planning_summary.json', 'decisions/decision_00000.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def require(condition, message):
    if not condition: raise ValueError(message)


def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    return value


def dump(path, value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')


def checked(path, expected, label):
    path = Path(path).resolve()
    require(sha(path)==expected, f'{label} checksum mismatch: {path}')
    return path


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key:archive[key].copy() for key in archive.files}


def reference(path):
    value = read(path)
    route = value['route'] if 'route' in value else value
    require(isinstance(route,dict), 'Require an explicit executable reference')
    arrays = {key:np.asarray(route[key],dtype=np.float64) for key in ARRAY_FIELDS}
    n = len(arrays['waypoints'])
    require(n>=3 and arrays['waypoints'].shape==(n,2), 'Invalid reference waypoints')
    require(all(arrays[k].shape==(n,) for k in ARRAY_FIELDS[1:]), 'Reference arrays disagree in length')
    require(all(np.isfinite(v).all() for v in arrays.values()), 'Nonfinite reference')
    require((np.diff(arrays['stations'])>0).all(), 'Nonincreasing reference stations')
    require((arrays['speeds']>=0).all() and (arrays['speeds']<=6).all(), 'Reference speed outside frozen forward-command range')
    return route


def reference_arrays_sha256(route):
    h = hashlib.sha256()
    for key in ARRAY_FIELDS:
        value = np.asarray(route[key], dtype='<f8', order='C')
        h.update(key.encode()+b'\0')
        h.update(np.asarray(value.shape, dtype='<i8').tobytes())
        h.update(value.tobytes(order='C'))
    return h.hexdigest()


def exact_arrays(a, b):
    return {key:np.array_equal(np.asarray(a[key]),np.asarray(b[key])) for key in ARRAY_FIELDS}


def import_frozen(source, filename, name):
    sys.path.insert(0,str(source/'src'))
    sys.path.insert(0,str(source/'scripts'))
    spec = importlib.util.spec_from_file_location(name, source/'scripts'/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_common(args):
    require(bool(os.environ.get('SLURM_JOB_ID')), 'Physics and rendering must run in an AMD Slurm allocation')
    checked(args.manifest,args.manifest_sha256,'Frozen diagnostic manifest')
    spec = read(args.manifest)
    require(float(spec.get('horizon_s',12.))==12., 'This diagnostic has a fixed 12 s horizon')
    require(float(spec.get('dt',.2))==.2, 'This diagnostic requires the frozen 0.2 s forecast grid')
    checked(__file__,spec['wrapper_sha256'],'Declared physical wrapper')
    source = Path(spec['source_root']).resolve()
    checked(source/'source_manifest.json',spec['source_manifest_sha256'],'Frozen source manifest')
    inventory = read(source/'source_manifest.json')['files']
    for relative,digest in inventory.items(): checked(source/relative,digest,'Frozen source file')
    case_file = checked(spec['case'],spec['case_sha256'],'Near-start case')
    case = read(case_file)
    require(case['id']==spec['start_id'], 'Declared start differs from case identity')
    require(float(case.get('horizon_s',12.))==12., 'Near-start case horizon differs from declared 12 s')
    if spec.get('runtime_fingerprint'):
        fingerprint = checked(spec['runtime_fingerprint'],spec['runtime_fingerprint_sha256'],'Runtime inventory')
        for path,digest in read(fingerprint)['runtime_sha256'].items(): checked(path,digest,'Runtime file')
        os.environ['FDM_RUNTIME_FINGERPRINT']=str(fingerprint)
    require(not args.out.exists(), 'Use a new output directory; existing artifacts are preserved')
    return spec,source,case_file,case


def observe(args, spec, source, case_file, case):
    observer_source=Path(spec.get('observer_source_root',source)).resolve()
    if observer_source!=source:
        checked(observer_source/'source_manifest.json',spec['observer_source_manifest_sha256'],'Frozen observer manifest')
        for relative,digest in read(observer_source/'source_manifest.json')['files'].items():
            checked(observer_source/relative,digest,'Frozen observer source')
    require((observer_source/'scripts/traverse_fdm_rgbd_diverse_chrono.py').exists(),
        'The online snapshot omits the standalone collector; declare observer_source_root and its manifest SHA (campaign_v2).')
    module = import_frozen(observer_source,'traverse_fdm_rgbd_diverse_chrono.py','near_assets_frozen_collector')
    original = module.write_observation
    captured = {}

    def measured_observation(out,scene,state,pose,goal,goal_radius,camera,case_id):
        vehicle = scene.hmmwv.GetVehicle()
        chassis = scene.hmmwv.GetChassis().GetBody()
        captured.update({'case_id':case_id,'native_time_s':float(scene.system.GetChTime()),
            'chassis_contact_resultant_n':float(chassis.GetContactForce().Length()),
            'asset_contact_resultant_n':[float(body.GetContactForce().Length()) for _,body in scene.asset_bodies],
            'roll_rad':float(vehicle.GetRoll()),'pitch_rad':float(vehicle.GetPitch()),
            'scope':'Current simulator launch validation only, not model inputs or future labels'})
        return original(out,scene,state,pose,goal,goal_radius,camera,case_id)

    module.write_observation = measured_observation
    previous = sys.argv
    started=time.time()
    try:
        sys.argv=[str(module.__file__),'observe','--case',str(case_file),'--out',str(args.out),
            '--chrono-data',args.chrono_data,'--horizon-s','12']
        module.main()
    finally:
        sys.argv=previous;module.write_observation=original
    require(bool(captured), 'Observation hook did not capture the measured launch')
    anchor=load_npz(args.out/'anchor_state.npz'); image=load_npz(args.out/'observation.npz')
    checks={key:np.array_equal(anchor[key],image[key]) for key in ('state','pose','history')}
    require(all(checks.values()), 'Observed state, native pose or causal history differs from native anchor')
    require(np.array_equal(anchor['goal_xy'],np.asarray(case['goal_xy'],np.float64)), 'Native anchor goal differs from exact case goal')
    provenance=read(args.out/'simulation_provenance.json')
    dump(args.out/'near_assets_observation.json',{'schema':'fdm_near_assets_h12_observation_v1',
        'scope':'Diagnostic near-start capture; no prediction or traversal outcome selection',
        'start_id':spec['start_id'],'manifest_sha256':sha(args.manifest),'wrapper_sha256':sha(__file__),
        'source_manifest_sha256':spec['source_manifest_sha256'],'case_sha256':sha(case_file),
        'observer_source_root':str(observer_source),'observer_source_manifest_sha256':sha(observer_source/'source_manifest.json'),
        'observation_sha256':sha(args.out/'observation.npz'),'anchor_state_sha256':sha(args.out/'anchor_state.npz'),
        'launch_checks':checks,'anchor_exact':True,'launch_validation':captured,
        'anchor_pose':anchor['pose'],'history_sha256':hashlib.sha256(anchor['history'].tobytes()).hexdigest(),
        'case_goal_xy_float64':anchor['goal_xy'],'observation_goal_xy_float32':image['goal_xy'],
        'source_sha256':provenance['source_sha256'],'runtime_sha256':provenance['runtime_sha256'],
        'wall_s':time.time()-started,'slurm_job_id':os.environ['SLURM_JOB_ID'],
        'artifacts_sha256':{name:sha(args.out/name) for name in ('observation.npz','observation.json','anchor_state.npz','simulation_provenance.json','rgb.png')}})


def rollout(args,spec,source,case_file,case):
    for pathkey,hashkey in [('checkpoint','checkpoint_sha256'),('observation_npz','observation_sha256'),
        ('observation_json','observation_json_sha256'),('predictions_npz','predictions_sha256'),
        ('candidates_json','candidates_sha256'),('selection_json','selection_sha256'),
        ('cost_config_file','cost_config_sha256'),
        ('observation_provenance_json','observation_provenance_sha256')]: checked(spec[pathkey],spec[hashkey],pathkey)
    require(read(spec['cost_config_file'])==spec['cost_config'], 'Cost file differs from declared configuration')
    entries=spec['routes']
    require(len(entries)==4 and len({r['id'] for r in entries})==4,'Exactly four distinct frozen diagnostic references are required')
    matches=[entry for entry in entries if entry['id']==args.candidate_id]
    require(len(matches)==1,'Candidate is absent or duplicated in frozen manifest')
    entry=matches[0]
    require(Path(entry['actual_dir']).resolve()==args.out.resolve(),'Output path differs from frozen candidate declaration')
    ref_file=checked(entry['reference_json'],entry['reference_sha256'],'Chosen diagnostic reference')
    route=reference(ref_file)
    index=int(entry['prediction_index']);require(index>=0,'Prediction index must be nonnegative')
    observation_file=Path(spec['observation_npz'])
    image=load_npz(observation_file)
    native_file=Path(spec.get('anchor_state_npz',observation_file.parent/'anchor_state.npz'))
    if spec.get('anchor_state_sha256'): checked(native_file,spec['anchor_state_sha256'],'Native observation anchor')
    anchor=load_npz(native_file)
    require(all(np.array_equal(anchor[k],image[k]) for k in ('state','pose','history')),'Observation native anchor differs from scored input arrays')
    exact_goal=np.asarray(case['goal_xy'],np.float64)
    require(np.array_equal(anchor['goal_xy'],exact_goal),'Native scored anchor goal differs from physical case goal')
    for field,expected in [('anchor_pose',anchor['pose']),('goal_xy',exact_goal)]:
        if field in spec: require(np.array_equal(spec[field],expected),'Manifest '+field+' differs from exact anchor')
    history_hash=hashlib.sha256(anchor['history'].tobytes()).hexdigest()
    if spec.get('history_sha256'): require(spec['history_sha256']==history_hash,'Manifest causal history differs')
    selection=read(spec['selection_json'])
    require(selection['physical_results_used_for_selection'] is False,'Selection used physical outcomes')
    require(spec['prediction_frozen_before_physical_execution'] is True,'Prediction must be frozen before physical execution')
    require(selection['prediction_outputs_sha256']==spec['predictions_sha256'],'Selection binds different forecasts')
    require(selection['history_sha256']==history_hash,'Scored launch history differs from native anchor')
    for key,expected in [('anchor_pose',anchor['pose']),('anchor_state17',anchor['state']),('goal_xy',exact_goal)]:
        require(np.array_equal(selection[key],expected),'Scored '+key+' differs from exact physical launch')
    selected=[row for row in selection['selected'] if row['id']==entry['id']]
    require(len(selected)==1 and selected[0]['prediction_index']==index and selected[0]['candidate_id']==entry['candidate_id'],
        'Reporting route does not match the frozen selected candidate index')
    candidates=read(spec['candidates_json'])['candidates']
    scored=[row for row in candidates if row['prediction_index']==index]
    require(len(scored)==1 and scored[0]['id']==entry['candidate_id'],'Forecast index does not identify its frozen candidate')
    require(scored[0]['route']==route,'Executed reference differs from the full scored reference, including metadata')
    with np.load(spec['predictions_npz'],allow_pickle=False) as forecast:
        for key,width in [('trajectory',4),('work',1),('event_probability',3),('attitude',2)]:
            values=forecast[key]
            require(values.shape==(len(candidates),60,width),'Unexpected frozen '+key+' forecast shape')
            require(np.isfinite(values[index]).all(),'Selected forecast contains nonfinite values')
    obs_provenance=read(spec['observation_provenance_json'])
    require(obs_provenance['runtime_sha256']==spec['runtime_sha256'],'Observed runtime differs from frozen reporting manifest')
    for path,digest in obs_provenance['runtime_sha256'].items(): checked(path,digest,'Observed runtime dependency')
    import nedm.traverse.fdm_diverse_planner as planner
    real_planner=planner.plan_rgbd_routes
    calls=[]
    packet={'route':copy.deepcopy(route),'cost':None,'selected_family_index':index,
        'abstained':False,'goal_reached':False,'family_scores':[],'checks':[],
        'refinements':[],'model_evaluations':0,
        'claim':'Explicit frozen candidate execution, including predicted-risky candidates; no new neural inference or replanning'}

    def recorded_reference(model,rgbd,history,pose,goal,**kwargs):
        require(not calls and kwargs['elapsed_s']==0.,'Unexpected later planning call')
        require(model.config.horizon==60 and np.isclose(model.config.dt,.2),'Checkpoint does not supply the declared 12 s forecast')
        equality={'rgbd':np.array_equal(rgbd,image['rgbd']),
            'history':np.array_equal(history,anchor['history']),
            'pose':np.array_equal(pose,anchor['pose']),
            'case_goal':np.array_equal(goal,exact_goal)}
        require(all(equality.values()),'Physical launch differs from the frozen prediction anchor: '+str(equality))
        calls.append({'time_s':0.,'candidate_id':entry['id'],'prediction_index':index,'checks':equality,
            'history_sha256':hashlib.sha256(np.asarray(history).tobytes()).hexdigest()})
        return copy.deepcopy(packet)

    planner.plan_rgbd_routes=recorded_reference
    module=import_frozen(source,'traverse_fdm_rgbd_diverse_online.py','near_assets_frozen_online')
    argv=[str(module.__file__),'--case',str(case_file),'--checkpoint',spec['checkpoint'],
        '--scene-observation',str(observation_file),'--out',str(args.out),'--chrono-data',args.chrono_data,
        '--horizon-s','12','--planning-device','cpu','--planning-mode','plan_once',
        '--replan-period-s','1','--cost-mode',spec.get('cost_mode','energy_risk'),
        '--cost-config',spec['cost_config_file'],'--mppi-samples','64','--mppi-iterations','2',
        '--seed',str(spec.get('seed',11)),'--video-fps',str(args.video_fps)]
    for key in ('speeds','offsets'):
        if key in spec: argv+=['--'+key]+[str(v) for v in spec[key]]
    previous=sys.argv;started=time.time()
    try:
        sys.argv=argv;module.main()
    finally:
        sys.argv=previous;planner.plan_rgbd_routes=real_planner
    require(len(calls)==1,'Require exactly one recorded reference at launch')
    actual_anchor=load_npz(args.out/'anchor_state.npz')
    anchor_checks={key:np.array_equal(actual_anchor[key],anchor[key]) for key in ('state','pose','history','goal_xy')}
    require(all(anchor_checks.values()),'Actual native state/pose/history/goal differs from scored anchor')
    saved=read(args.out/'decisions/decision_00000.json')
    require(saved['decision']==packet,'The executed decision packet differs from frozen reference')
    require(saved['history_sha256']==history_hash,'Recorded history digest differs')
    outcome=read(args.out/'outcome.json')
    route_checks=exact_arrays(outcome['final_reference'],route)
    require(all(route_checks.values()),'Actual final reference arrays differ from scored reference')
    online=read(args.out/'online_protocol.json');runtime=read(args.out/'simulation_provenance.json')
    require(online['planning_mode']=='plan_once' and outcome['planning_decisions']==1,'Unexpected replanning')
    require(online['model_checkpoint_sha256']==spec['checkpoint_sha256'],'Executed checkpoint differs')
    require(online['scene_observation_sha256']==spec['observation_sha256'],'Executed observation differs')
    require(online['cost_config']==spec['cost_config'],'Executed cost metadata differs from frozen planner')
    require(online['mppi_config']==spec['mppi_config'],'Executed MPPI metadata differs from frozen planner')
    require(online['candidate_speeds_mps']==spec['speeds'] and online['candidate_offsets_m']==spec['offsets'],
        'Inherited geometric proposal metadata differs from frozen planner')
    require(runtime['case_sha256']==spec['case_sha256'],'Executed scene differs')
    for key in ('case_sha256','arena_meta_sha256','arena_bmp_sha256','runtime_sha256','physics_dt_s','camera'):
        require(runtime[key]==obs_provenance[key], 'Observation/execution runtime mismatch: '+key)
    # The capture and execution entry points differ; overlapping physics modules must agree.
    overlap=set(runtime['source_sha256']) & set(obs_provenance['source_sha256'])
    require(all(runtime['source_sha256'][k]==obs_provenance['source_sha256'][k] for k in overlap),'Common observation/execution source differs')
    require(0<float(outcome['elapsed_s'])<=12.0000001,'Invalid actual horizon')
    full=bool(np.isclose(outcome['elapsed_s'],12.,atol=1e-7,rtol=0))
    require(full or outcome['status'] in ('goal_reached','rollover'),'Unexplained early termination')
    physical=load_npz(args.out/'trajectory.npz');rich=load_npz(args.out/'rich_telemetry.npz');intervals=load_npz(args.out/'rich_intervals.npz')
    require(len(physical['state'])==outcome['frames']==len(intervals['duration_s']),'Measured frame count mismatch')
    require(len(rich['time_s'])==outcome['frames']+1,'Missing real terminal telemetry')
    require(np.isclose(rich['time_s'][-1],outcome['elapsed_s'],atol=1e-7,rtol=0),'Terminal timestamp mismatch')
    artifacts={name:sha(args.out/name) for name in ARTIFACTS}
    for name in ('frame_metadata.json','frame_times_s.npy','video_camera.json'):
        if (args.out/name).exists(): artifacts[name]=sha(args.out/name)
    result={'schema':'fdm_near_assets_h12_execution_v1','scope':'Selected diagnostic reference execution, not a protected score or fresh model decision',
        'start_id':spec['start_id'],'candidate_id':entry['id'],'prediction_index':index,
        'reference_json':str(ref_file),'reference_sha256':entry['reference_sha256'],
        'reporting_manifest_sha256':sha(args.manifest),'predictions_sha256':spec['predictions_sha256'],
        'candidates_sha256':spec['candidates_sha256'],'selection_sha256':spec['selection_sha256'],
        'observation_sha256':spec['observation_sha256'],'checkpoint_sha256':spec['checkpoint_sha256'],
        'source_manifest_sha256':spec['source_manifest_sha256'],'wrapper_sha256':sha(__file__),
        'case_sha256':spec['case_sha256'],'cost_config_sha256':spec['cost_config_sha256'],
        'anchor_state_sha256':sha(native_file),'observation_provenance_sha256':spec['observation_provenance_sha256'],
        'horizon_s':12.,'forecast_dt_s':.2,'actual_elapsed_s':outcome['elapsed_s'],
        'full_horizon_observed':full,'terminal_reason':outcome['status'],
        'terminal_semantics':'Frozen online_v9 stops at 12 s cap, goal arrival or rollover. No continuation, padding or fabricated state is appended after termination.',
        'executed_reference_arrays_sha256':reference_arrays_sha256(route),
        'reference_arrays_hash_convention':ARRAY_HASH_CONVENTION,'reference_arrays_exact':route_checks,
        'anchor_exact':True,'anchor_checks':anchor_checks,'calls':calls,
        'history_sha256':history_hash,'anchor_pose':anchor['pose'],
        'case_goal_xy_float64':exact_goal,'observation_goal_xy_float32':image['goal_xy'],
        'observation_goal_rounding_difference_m':np.asarray(image['goal_xy'],float)-exact_goal,
        'source_sha256':runtime['source_sha256'],'runtime_sha256':runtime['runtime_sha256'],
        'common_observation_source_checked':sorted(overlap),'online_protocol_role':'Inherited frozen execution configuration; one supplied decision is substituted by this hash-bound wrapper.',
        'risk_gates_enforced_during_execution':False,
        'risk_gate_scope':'Every declared candidate is intentionally executed, including predicted-risky references, to measure consequences of the same forecast/reference.',
        'new_model_inference':False,'replanning':False,'video_fps':args.video_fps,
        'video_scope':'Frames from this same physical run' if args.video_fps else 'No video requested',
        'artifacts_sha256':artifacts,'wall_s':time.time()-started,'slurm_job_id':os.environ['SLURM_JOB_ID']}
    dump(args.out/'near_assets_execution.json',result)
    print(json.dumps({'candidate_id':entry['id'],'elapsed_s':outcome['elapsed_s'],'terminal_reason':outcome['status'],
        'anchor_exact':True,'reference_exact':True,'out':str(args.out)}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('observe','rollout'))
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--candidate-id')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--chrono-data',required=True)
    p.add_argument('--video-fps',type=float,default=0.)
    args=p.parse_args()
    if args.phase=='rollout' and not args.candidate_id: p.error('rollout requires --candidate-id')
    if args.phase=='observe' and args.video_fps: p.error('observe captures the global image only')
    if not np.isfinite(args.video_fps) or args.video_fps<0: p.error('Video rate must be finite and nonnegative')
    spec,source,case_file,case=validate_common(args)
    sys.path.insert(0,str(source/'src'));sys.path.insert(0,str(source/'scripts'))
    (observe if args.phase=='observe' else rollout)(args,spec,source,case_file,case)


if __name__=='__main__': main()
