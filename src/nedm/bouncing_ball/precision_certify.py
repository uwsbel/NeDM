"""Frozen-model paired certification on a freshly collected held-out cohort."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from nedm.bouncing_ball.collection import STATE_FIELDS, atomic_json
from nedm.bouncing_ball.model import load_model
from nedm.bouncing_ball.precision import prepare
from nedm.bouncing_ball.training import evaluate, quantiles


def raw_trace(task):
    index, episode = task
    path=Path(episode['raw_data_root'])/episode['csv_path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=episode['csv_sha256']: raise RuntimeError('raw hash mismatch')
    with path.open() as stream:
        rows=list(csv.DictReader(stream))
    end=episode['model_duration_s']
    rows=[r for r in rows if float(r['time_s'])<=end+1e-10]
    return index,np.array([float(r['time_s']) for r in rows]),np.array([[float(r[f]) for f in STATE_FIELDS] for r in rows])


@torch.no_grad()
def event_trajectory(model, initial, times):
    """Evaluate the learned event law at dense times, without interpolation.

    This helper is for the certified ground-first two-impact launch domain.
    Native recursive rollouts are checked against it separately.
    """
    tg=model.impact_time(initial,0)
    ground=model.flight(initial,tg)
    ground=torch.cat((ground[:,:2],ground[:,2:]+model.impulse(ground if model.feature_dim==5 else ground[:,2:],0)),-1)
    tw=model.impact_time(ground,1)
    wall=model.flight(ground,tw)
    wall=torch.cat((wall[:,:2],wall[:,2:]+model.impulse(wall if model.feature_dim==5 else wall[:,2:],1)),-1)
    first=model.flight(initial[:,None,:],times)
    middle=model.flight(ground[:,None,:],(times-tg[:,None]).clamp_min(0))
    last=model.flight(wall[:,None,:],(times-tg[:,None]-tw[:,None]).clamp_min(0))
    result=torch.where((times>=tg[:,None])[...,None],middle,first)
    result=torch.where((times>=tg[:,None]+tw[:,None])[...,None],last,result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('AMD GPU allocation required')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    choice=json.loads(args.selection.read_text())
    chosen_hash=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    assert choice['checkpoint_sha256']==chosen_hash and choice['selection_split']=='validation'
    index,states,lengths,splits,contacts=prepare(args.data,'cuda')
    model,metadata=load_model(args.checkpoint,'cuda')
    model.double()
    native_dt=model.dt
    baseline,_=load_model(args.baseline,'cuda')
    baseline.double()
    for predictor in (model, baseline):
        for parameter in predictor.parameters(): parameter.requires_grad_(False)
    assert abs(baseline.dt-index['model_dt_s'])<1e-12
    limits=dict(position_rmse_p95_m=.003,endpoint_error_p95_m=.006,velocity_mae_p95_mps=.02,
                spin_mae_p95_radps=.1,contact_time_error_p95_s=.02,contact_order_fraction=1.,max_penetration_p95_m=.002)
    model.dt=index['model_dt_s']
    comparisons={}
    for name,split in [('validation',1),('fresh_test',2)]:
        ids=torch.nonzero(splits==split).flatten()
        old=evaluate(baseline,states,lengths,contacts,ids,limits)
        new=evaluate(model,states,lengths,contacts,ids,limits)
        comparisons[name]={'baseline':old,'candidate':new,'trajectory_p95_reduction_factor':old['position_rmse_m']['p95']/new['position_rmse_m']['p95'],
                           'endpoint_p95_reduction_factor':old['endpoint_error_m']['p95']/new['endpoint_error_m']['p95']}
        print(json.dumps({'cohort':name,'trajectory_p95_mm':new['position_rmse_m']['p95']*1000,'endpoint_p95_mm':new['endpoint_error_m']['p95']*1000,
                          'trajectory_gain':comparisons[name]['trajectory_p95_reduction_factor'],'contact_order':new['contact_order_fraction']}),flush=True)
    # Same initial conditions: compare the true 50 ms model to the dense solve.
    ids=torch.nonzero(splits==2).flatten()
    initial=states[ids,0]
    model.dt=native_dt
    native=model.rollout(initial,round(1.7/native_dt))
    native_times=torch.arange(native.shape[1],device='cuda',dtype=torch.float64)[None,:]*native_dt
    direct=event_trajectory(model,initial,native_times)
    agreement=float((native-direct).abs().max())
    assert agreement<1e-7,agreement
    # Raw 0.5 ms data remain independent; never train/select using these traces.
    tasks=[(int(i),index['episodes'][int(i)]) for i in ids.cpu().tolist()]
    dense_entries=[]
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,times,truth in pool.map(raw_trace,tasks,chunksize=8):
            tensor=torch.tensor(truth[None,0],dtype=torch.float64,device='cuda')
            grid=torch.tensor(times[None],dtype=torch.float64,device='cuda')
            pred=event_trajectory(model,tensor,grid)[0].cpu().numpy()
            error=np.linalg.norm(pred[:,:2]-truth[:,:2],axis=-1)
            dense_entries.append({'index':i,'position_rmse_m':float(np.sqrt(np.mean(error**2))),'endpoint_error_m':float(error[-1]),
                                  'maximum_position_error_m':float(error.max())})
    dense={k:quantiles([e[k] for e in dense_entries]) for k in ['position_rmse_m','endpoint_error_m','maximum_position_error_m']}
    dense.update(episodes=len(dense_entries),sample_step_s=.0005,per_episode=dense_entries,native_dense_state_max_difference=agreement)
    # Save candidate and metadata without changing its dt or neural weights.
    shutil.copyfile(args.checkpoint,args.output/'best.pt')
    report={'complete':True,'selected_before_test':choice,'checkpoint_sha256':chosen_hash,'native_dt_s':native_dt,'matched_comparison_dt_s':index['model_dt_s'],
            'comparisons':comparisons,'dense_fresh_test':dense,'data_sha256':index['model_data_sha256'],
            'fresh_seed':json.loads((Path(index['source_campaigns'][-1]['root'])/'campaign_index.json').read_text())['campaign']['seed'],'runtime':{'host':platform.node(),'job_id':os.environ.get('SLURM_JOB_ID'),
            'torch_version':torch.__version__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'one_order_passed':comparisons['fresh_test']['trajectory_p95_reduction_factor']>=10 and comparisons['fresh_test']['endpoint_p95_reduction_factor']>=10,
            'two_orders_passed':comparisons['fresh_test']['trajectory_p95_reduction_factor']>=100 and comparisons['fresh_test']['endpoint_p95_reduction_factor']>=100}
    atomic_json(args.output/'certification.json',report)
    print(json.dumps({'dense':{k:v for k,v in dense.items() if k!='per_episode'},'one_order_passed':report['one_order_passed'],'two_orders_passed':report['two_orders_passed']},indent=2),flush=True)


if __name__=='__main__':main()
