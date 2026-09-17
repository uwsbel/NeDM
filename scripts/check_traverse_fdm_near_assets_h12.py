#!/usr/bin/env python3
"""CPU-only packet guard checks; fake physics sentinel performs no simulation."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import numpy as np

path=Path(__file__).with_name('traverse_fdm_near_assets_h12.py')
loader=importlib.util.spec_from_file_location('near_wrapper_check',path)
wrapper=importlib.util.module_from_spec(loader);loader.loader.exec_module(wrapper)

class ReachedPhysics(Exception):pass


def write(path,value):path.write_text(json.dumps(value)+'\n')


with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);obs=root/'obs';obs.mkdir()
    state=np.zeros(17,np.float32);pose=np.array([1.123456789,2.,.1],np.float64)
    history=np.zeros((16,24),np.float32);goal=np.array([84.123456789,7.123456789],np.float64)
    np.savez(obs/'observation.npz',rgbd=np.zeros((4,512,512),np.float32),state=state,pose=pose,history=history,goal_xy=goal.astype(np.float32))
    np.savez(obs/'anchor_state.npz',state=state,pose=pose,history=history,goal_xy=goal)
    write(obs/'observation.json',{});write(obs/'simulation_provenance.json',{'runtime_sha256':{}})
    checkpoint=root/'checkpoint.pt';checkpoint.write_bytes(b'no neural checkpoint loaded by fixture')
    cost=root/'cost.json';write(cost,{'energy_weight_s_per_kj':.02})
    route={'waypoints':[[1.,2.],[42.,4.],[84.123456789,7.123456789]],'stations':[0.,41.04875,83.22],
        'headings':[.1,.1,.1],'speeds':[4.,4.,0.],'meta':{'fdm_station':0.}}
    ref=root/'route.json';write(ref,route)
    candidates=root/'candidates.json';write(candidates,{'candidates':[{'id':'candidate_0','prediction_index':0,'route':route}]})
    predictions=root/'predictions.npz';np.savez(predictions,trajectory=np.zeros((1,60,4),np.float32),work=np.zeros((1,60,1),np.float32),event_probability=np.ones((1,60,3),np.float32),attitude=np.zeros((1,60,2),np.float32))
    selection=root/'selection.json'
    picked=[{'id':cid,'prediction_index':0,'candidate_id':'candidate_0'} for cid in ('preferred_1','preferred_2','risky_1','risky_2')]
    selection_data={'physical_results_used_for_selection':False,'prediction_outputs_sha256':wrapper.sha(predictions),
        'history_sha256':wrapper.hashlib.sha256(history.tobytes()).hexdigest(),'anchor_pose':pose.tolist(),
        'anchor_state17':state.tolist(),'goal_xy':goal.tolist(),'selected':picked}
    write(selection,selection_data)
    spec={'checkpoint':str(checkpoint),'checkpoint_sha256':wrapper.sha(checkpoint),
        'observation_npz':str(obs/'observation.npz'),'observation_sha256':wrapper.sha(obs/'observation.npz'),
        'observation_json':str(obs/'observation.json'),'observation_json_sha256':wrapper.sha(obs/'observation.json'),
        'observation_provenance_json':str(obs/'simulation_provenance.json'),'observation_provenance_sha256':wrapper.sha(obs/'simulation_provenance.json'),
        'predictions_npz':str(predictions),'predictions_sha256':wrapper.sha(predictions),
        'candidates_json':str(candidates),'candidates_sha256':wrapper.sha(candidates),
        'selection_json':str(selection),'selection_sha256':wrapper.sha(selection),
        'cost_config_file':str(cost),'cost_config_sha256':wrapper.sha(cost),'cost_config':{'energy_weight_s_per_kj':.02},
        'runtime_sha256':{},'prediction_frozen_before_physical_execution':True,
        'routes':[{'id':x['id'],'candidate_id':'candidate_0','prediction_index':0,'reference_json':str(ref),
                   'reference_sha256':wrapper.sha(ref),'actual_dir':str(root/x['id'])} for x in picked]}
    args=SimpleNamespace(candidate_id='risky_1',out=root/'risky_1',chrono_data='/unused',video_fps=0.,manifest=root/'manifest.json')
    case={'goal_xy':goal.tolist()}
    planner=ModuleType('nedm.traverse.fdm_diverse_planner');planner.plan_rgbd_routes=lambda *a,**k:None
    packages={name:ModuleType(name) for name in ('nedm','nedm.traverse')}
    packages['nedm'].traverse=packages['nedm.traverse'];packages['nedm.traverse'].fdm_diverse_planner=planner
    for name,module in {**packages,'nedm.traverse.fdm_diverse_planner':planner}.items():sys.modules[name]=module
    fake=SimpleNamespace(__file__='/unused/frozen.py')
    def fake_main():
        decision=planner.plan_rgbd_routes(SimpleNamespace(config=SimpleNamespace(horizon=60,dt=.2)),
            np.zeros((4,512,512),np.float32),history,pose,goal,elapsed_s=0.)
        assert decision['abstained'] is False and decision['route']==route and decision['model_evaluations']==0
        raise ReachedPhysics('Fully predicted-risky candidate reaches explicit execution unchanged')
    fake.main=fake_main;wrapper.import_frozen=lambda *a,**k:fake
    try:wrapper.rollout(args,spec,root,root/'case.json',case)
    except ReachedPhysics:pass
    else:raise AssertionError('Expected fake physics sentinel')
    bad=copy.deepcopy(spec);bad['routes'][2]['prediction_index']=1
    try:wrapper.rollout(args,bad,root,root/'case.json',case)
    except ValueError as error:assert 'index' in str(error)
    else:raise AssertionError('Mismatched forecast index accepted')
    changed=copy.deepcopy(selection_data);changed['goal_xy']=goal.astype(np.float32).astype(float).tolist()
    write(selection,changed);bad=copy.deepcopy(spec);bad['selection_sha256']=wrapper.sha(selection)
    try:wrapper.rollout(args,bad,root,root/'case.json',case)
    except ValueError as error:assert 'goal_xy' in str(error)
    else:raise AssertionError('Rounded scored goal accepted as exact native goal')
    altered=copy.deepcopy(route);altered['speeds'][1]+=0.01
    assert wrapper.reference_arrays_sha256(route)!=wrapper.reference_arrays_sha256(altered)
    assert not all(wrapper.exact_arrays(route,altered).values())
print('PASS: risky route executes without filtering; mismatched prediction index and rounded goal reject before physics; speed changes alter exact-reference evidence. No physics or inference ran.')
