#!/usr/bin/env python3
"""Freeze a twelve-second RGB-D/MPPI candidate comparison before Chrono outcomes.

Only a measured observation, causal launch history, supplied goal, geometric
proposal settings and the frozen checkpoint enter planning. The request names
scene files for downstream provenance; this process never reads their contents.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def route_fingerprint(route):
    h = hashlib.sha256()
    for key in ('waypoints', 'stations', 'headings', 'speeds'):
        a = np.asarray(route[key], dtype='<f8')
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    h.update(np.asarray([route.get('meta', {}).get('fdm_station', 0.)], dtype='<f8').tobytes())
    return h.hexdigest()


def geometric_separation(a, b):
    """Maximum separation at 61 paired fractions of each first <=72m arc."""
    points=[]
    for route in (a, b):
        ss=np.asarray(route['stations'],float); xy=np.asarray(route['waypoints'],float)
        query=np.linspace(ss[0],min(ss[0]+72.,ss[-1]),61)
        points.append(np.stack([np.interp(query,ss,xy[:,k]) for k in range(2)],axis=-1))
    return float(np.linalg.norm(points[0]-points[1],axis=-1).max())


def run(args):
    request=json.loads(args.request.read_text())
    source=Path(request['source_root'])
    assert sha(source/'source_manifest.json')==request['source_manifest_sha256']
    inventory=json.loads((source/'source_manifest.json').read_text())
    file_table=inventory.get('files', inventory.get('file_sha256', {}))
    source_verified=0
    for rel, expected in file_table.items():
        if rel.endswith('.py'):
            assert sha(source/rel)==expected, rel
            source_verified+=1
    sys.path.insert(0,str(source/'src'))
    import torch
    import nedm.traverse.fdm_diverse_planner as planner
    from nedm.traverse.fdm_diverse_model import load_rgbd_checkpoint
    from nedm.traverse.fdm_mppi import MPPIConfig
    from nedm.traverse.fdm_online_candidates import POLICY
    torch.set_num_threads(args.threads)
    assert sha(request['checkpoint'])==request['checkpoint_sha256']
    model, checkpoint=load_rgbd_checkpoint(request['checkpoint'], device='cpu')
    assert model.config.horizon==60 and np.isclose(model.config.dt,.2)
    observation_path=Path(request['observation_npz'])
    observation_meta_path=observation_path.with_suffix('.json')
    meta=json.loads(observation_meta_path.read_text())
    assert sha(observation_path)==meta['observation_sha256']
    with np.load(observation_path,allow_pickle=False) as obs:
        rgbd=obs['rgbd'].copy(); history=obs['history'].copy(); pose=obs['pose'].copy()
        state=obs['state'].copy(); stored_goal=obs['goal_xy'].copy()
    goal=np.asarray(request['goal_xy'],dtype=float)
    assert np.array_equal(goal.astype(np.float32),stored_goal)
    assert abs(float(state[0]))<.3, 'Vehicle not settled at launch'
    provenance_path=observation_path.parent/'simulation_provenance.json'
    provenance=json.loads(provenance_path.read_text())
    assert provenance['case_sha256']==request['case_sha256']
    costs=planner.RGBDCostConfig(**request['cost_config'])
    cfg=MPPIConfig(samples=64,iterations=2,max_speed_mps=6.,arena_half_extent_m=120.,
        knots=5,lateral_sigma_m=.6,max_lateral_m=2.,path_step_m=.5,
        max_curvature_inv_m=POLICY['max_reference_curvature_inv_m'])
    original=planner.RGBDReferenceScorer
    rows=[]; outputs=[]; by_fingerprint={}; instances=[]
    class RecordingScorer(original):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw); instances.append(self)
        def predict(self,routes):
            result=super().predict(routes)
            for i, route in enumerate(routes):
                fp=route_fingerprint(route)
                if fp not in by_fingerprint:
                    index=len(rows); by_fingerprint[fp]=index
                    rows.append({'id':f'candidate_{index:05d}','prediction_index':index,
                                 'route':copy.deepcopy(route),'route_sha256':fp})
                    outputs.append({k:v[i].copy() for k,v in result.items()})
            return result
    families=planner.propose_route_families(pose,goal,speeds=request['speeds'],offsets=request['offsets'],step_m=.5)
    planner.RGBDReferenceScorer=RecordingScorer
    started=time.time()
    try:
        decision=planner.plan_rgbd_routes(model,rgbd,history,pose,goal,
            seed=request['seed'],families=families,mppi_config=cfg,cost_config=costs)
    finally:
        planner.RGBDReferenceScorer=original
    if not outputs:
        raise ValueError('No kinematically valid candidates; retain input request and review proposal geometry')
    predictions={k:np.stack([o[k] for o in outputs]) for k in outputs[0]}
    components=instances[0].cost_breakdown(predictions)
    risks=np.max(np.stack([components['contact_probability']/costs.max_contact_probability,
        components['low_progress_probability']/costs.max_low_progress_probability,
        components['rollover_probability']/costs.max_rollover_probability,
        components['predicted_peak_roll_deg']/costs.hard_roll_deg,
        components['predicted_peak_pitch_deg']/costs.hard_pitch_deg]),axis=0)
    for i,row in enumerate(rows):
        row['components']={k:v[i] for k,v in components.items()}
        row['normalized_risk']=float(risks[i])
    threshold=2.
    def pick(order, count, excluded):
        chosen=[]
        for i in order:
            i=int(i)
            if i in excluded or any(geometric_separation(rows[i]['route'], rows[j]['route'])<threshold for j in chosen):
                continue
            chosen.append(i)
            if len(chosen)==count:
                return chosen
        return chosen
    finite=[i for i in np.argsort(components['cost'],kind='stable') if np.isfinite(components['cost'][i])]
    preferred=pick(finite,2,set())
    fallback=False
    if len(preferred)<2:
        fallback=True
        # An abstaining model cannot supply two accepted routes. Keep the
        # promised four physical diagnostics, explicitly label rejected picks.
        order=finite+[int(i) for i in np.argsort(components['unfiltered_cost'],kind='stable') if int(i) not in finite]
        preferred=pick(order,2,set())
    risky=pick(np.argsort(-risks,kind='stable'),2,set(preferred))
    if len(preferred)!=2 or len(risky)!=2:
        raise ValueError('Fewer than four candidate references satisfying declared group distinctness')
    out=args.out
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new prediction output directory')
    out.mkdir(parents=True,exist_ok=True)
    prediction_path=out/'predictions.npz'
    np.savez_compressed(prediction_path,**predictions)
    candidate_path=out/'candidates.json'
    dump(candidate_path,{'schema':'fdm_near_assets_candidates_v1','candidates':rows})
    dump(out/'mppi_decision.json',decision)
    selected=[]; manifest_routes=[]
    for group, indices in [('preferred',preferred),('risky',risky)]:
        for rank, index in enumerate(indices,1):
            cid=f'{group}_{rank}'
            accepted=bool(components['allowed_by_predicted_risk'][index])
            row={'id':cid,'candidate_id':rows[index]['id'],'prediction_index':index,
                 'predicted_group':group,'exceeds_risk_gate':not accepted,
                 'normalized_risk':float(risks[index]),
                 'components':{k:v[index] for k,v in components.items()}}
            label=f'Preferred {rank}' if group=='preferred' else f'Higher predicted risk {rank}'
            if group=='preferred' and not accepted: label += ' (rejected fallback)'
            if group=='risky' and accepted: label += ' (below rejection threshold)'
            ref=out/'references'/f'{cid}.json'; dump(ref,rows[index]['route'])
            manifest_routes.append({'id':cid,'candidate_id':rows[index]['id'],'label':label,'predicted_group':group,
                'prediction_index':index,'reference_json':str(ref),'reference_sha256':sha(ref),
                'actual_dir':str(Path(request['actual_root'])/cid)})
            selected.append(row)
    selection={
        'schema':'fdm_near_assets_selection_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'selection_rule':'v1: two ascending finite model-cost picks; two descending normalized-risk picks, excluding preferred IDs; greedy geometric distinctness within each group. If fewer than two accepted geometries, retain two lowest unfiltered-cost alternatives explicitly marked rejected. Never use physical future outcomes.',
        'normalized_risk_definition':'max(contact_probability/0.35,low_progress_probability/0.5,rollover_probability/0.35,predicted_peak_roll_deg/55,predicted_peak_pitch_deg/55), using frozen scorer eligible prefixes',
        'geometric_distinctness_definition':'max Euclidean XY separation at61 equal fractions along each first min(72m,route_length) arc; each pair within the same group must reach threshold',
        'geometric_distinctness_threshold_m':threshold,'selected':selected,
        'accepted_candidate_count':int(np.isfinite(components['cost']).sum()),'total_unique_candidates':len(rows),
        'preferred_fallback_required':fallback,'planner_abstained':bool(decision.get('abstained',False)),
        'mppi_winner_prediction_index':None if decision.get('route') is None else by_fingerprint[route_fingerprint(decision['route'])],
        'anchor_pose':pose,'anchor_state17':state,'history_sha256':hashlib.sha256(history.tobytes()).hexdigest(),
        'goal_xy':goal,'source_python_files_verified':source_verified,
        'planning_wall_s':time.time()-started,'planner_script_sha256':sha(__file__),
        'request_sha256':sha(args.request),'prediction_outputs_sha256':sha(prediction_path),
        'observed_settle_proprioception_only_gate':'abs(body vx)<0.3m/s; exact physical start parity independently enforced during execution',
        'physical_results_used_for_selection':False}
    selection_path=out/'selection.json';dump(selection_path,selection)
    manifest={
        'schema':'fdm_near_assets_reporting_v1','scene_id':request['scene_id'],'start_id':request['start_id'],
        'source_root':str(source),'source_manifest_sha256':request['source_manifest_sha256'],
        'checkpoint':request['checkpoint'],'checkpoint_sha256':request['checkpoint_sha256'],
        'checkpoint_step':checkpoint['step'],'observation_npz':str(observation_path),'observation_sha256':sha(observation_path),
        'observation_json':str(observation_meta_path),'observation_json_sha256':sha(observation_meta_path),
        'observation_provenance_json':str(provenance_path),'observation_provenance_sha256':sha(provenance_path),
        'runtime_sha256':provenance['runtime_sha256'],
        'predictions_npz':str(prediction_path),'predictions_sha256':sha(prediction_path),
        'candidates_json':str(candidate_path),'candidates_sha256':sha(candidate_path),
        'selection_json':str(selection_path),'selection_sha256':sha(selection_path),
        'case':request['case'],'case_sha256':request['case_sha256'],
        'cost_config':asdict(costs),'cost_config_file':request['cost_config_file'],'cost_config_sha256':request['cost_config_sha256'],
        'horizon_s':12.,'dt':.2,'supported_events':model.supported_events.detach().cpu().numpy(),
        'mppi_config':asdict(cfg),'seed':request['seed'],'speeds':request['speeds'],'offsets':request['offsets'],
        'routes':manifest_routes,'scope':'Twelve-second fixed-reference diagnostic at new geometry-selected validation starts; no whole-route success claim',
        'planning_input_boundary':'Measured global RGB-D, measured launch history/pose, supplied goal, geometric candidate references; no terrain/asset geometry loaded by planning process',
        'prediction_frozen_before_physical_execution':True}
    dump(out/'reporting_manifest.json',manifest)
    print(json.dumps(clean({'scene':request['scene_id'],'candidates':len(rows),'accepted':selection['accepted_candidate_count'],
        'fallback':fallback,'selected':selected,'out':str(out),'wall_s':selection['planning_wall_s']})))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=4)
    run(p.parse_args())
