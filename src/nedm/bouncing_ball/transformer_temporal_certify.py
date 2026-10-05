"""Freeze validation selections, then certify once on a new Chrono cohort."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import numpy as np
import torch
from nedm.bouncing_ball.collection import atomic_json
from nedm.bouncing_ball.model import load_model
from nedm.bouncing_ball.transformer_contact_eval import data_packet,free_metrics,teacher_metrics,launch_gradient_check
from nedm.bouncing_ball.precision_certify import raw_trace
from nedm.bouncing_ball.training import quantiles


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['select','certify'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--names',nargs='+');p.add_argument('--device',default='cuda')
    args=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(1)
    data=data_packet(args.data,args.device)
    if args.mode=='select':
        candidates=[];ids=torch.nonzero(data['splits']==1).flatten()
        for name in args.names:
            folder=args.root/'runs'/name;done=json.loads((folder/'complete.json').read_text())
            assert done['complete'] and not done['smoke'] and not done['test_used_for_selection']
            source=folder/'best.pt';digest=hashlib.sha256(source.read_bytes()).hexdigest()
            assert digest==done['checkpoint_sha256']
            model,meta=load_model(source,args.device);metrics=free_metrics(model,data,ids)
            if abs(metrics['selection_score']-done['validation_score'])>1e-8:
                raise RuntimeError('Checkpoint loading changed saved validation behavior')
            zero_projections=all(float(b.attn.c_proj.weight.norm())==0 and float(b.mlp.c_proj.weight.norm())==0 for b in model.backbone.blocks)
            dest=args.root/'frozen'/name;dest.mkdir(parents=True,exist_ok=False);shutil.copyfile(source,dest/'best.pt')
            candidates.append(dict(name=name,path=str(dest/'best.pt'),sha256=digest,validation=metrics,
                model_config=model.config,stage=meta['stage'],update=meta['update'],
                functional_core='data_fitted_affine_control' if model.affine_preserving and zero_projections else 'trained_transformer'))
        trained=[c for c in candidates if c['functional_core']=='trained_transformer']
        if not trained:raise ValueError('A trained Transformer is required for primary model selection')
        chosen=min(trained,key=lambda row:row['validation']['selection_score'])['name']
        atomic_json(args.root/'selection.json',dict(candidates=candidates,chosen=chosen,
            overall_accuracy_control=min(candidates,key=lambda row:row['validation']['selection_score'])['name'],
            primary_selection_requires_trained_transformer=True,test_used_for_selection=False,
            selection_split='validation',data_sha256=data['index']['model_data_sha256']))
        print(json.dumps(dict(chosen=chosen,frozen=len(candidates))),flush=True);return
    selection=json.loads((args.root/'selection.json').read_text());assert not selection['test_used_for_selection']
    ids=torch.nonzero(data['splits']==2).flatten();assert len(ids)==900
    output=args.root/'certification';output.mkdir(parents=True,exist_ok=False)
    baseline=Path('/work1/dannegrut/harry/experiments')
    candidates=list(selection['candidates'])
    for name,path,note in (
        ('prior_fullstate_117mm',baseline/'ball_transformer_contact_20261001T192800Z/frozen/transformer_state5_binary_refined/best.pt','Previous pure residual Transformer; full-state gate; bounce input vx,vz'),
        ('historical_transformer_v2',baseline/'ball_span_v1_20260930/runs/nrd_v2_certified/best.pt','Transformer with analytical flight/contact timing/geometry'),
        ('analytical_mlp_reference',baseline/'ball_precision_20261001/runs/certified_v2/best.pt','Analytical flight/contact timing plus learned MLP response, no Transformer')):
        candidates.append(dict(name=name,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),note=note))
    result=dict(fresh_seed=202610015,episodes=len(ids),dt_s=data['dt'],models={},chosen_before_test=selection['chosen'],
                selection_before_test=True,test_used_for_selection=False,data_sha256=data['index']['model_data_sha256'])
    for candidate in candidates:
        source=Path(candidate['path']);assert hashlib.sha256(source.read_bytes()).hexdigest()==candidate['sha256']
        model,meta=load_model(source,args.device);model.double()
        native_model_dt=model.dt
        if candidate['name']=='analytical_mlp_reference':model.dt=data['dt']
        metrics=free_metrics(model,data,ids,output/(candidate['name']+'_traces.npz'))
        row=dict(checkpoint_sha256=candidate['sha256'],model_config=model.config,free_rollout=metrics,note=candidate.get('note'),
                 trained_model_dt_s=native_model_dt,evaluation_dt_s=model.dt)
        if hasattr(model,'contact_enabled'):
            row['teacher_forced']=teacher_metrics(model,data,ids)
            row['launch_gradients']=launch_gradient_check(model,[[4.3,-10.2],[5.5,-9.75],[6.8,-9.2]])
        result['models'][candidate['name']]=row
        print(json.dumps(dict(name=candidate['name'],p95_trajectory_mm=1000*metrics['position_rmse_m']['p95'],
            median_trajectory_mm=1000*metrics['position_rmse_m']['median'],p95_endpoint_mm=1000*metrics['endpoint_error_m']['p95'],
            order=metrics['contact_order_fraction'])),flush=True)
    # Uniform interpolation between native samples, never analytical timing.
    chosen=selection['chosen'];trace=dict(np.load(output/(chosen+'_traces.npz')));raw=[]
    for row,idx in enumerate(ids.cpu().tolist()):
        _,times,truth=raw_trace((idx,data['index']['episodes'][idx]));n=int(trace['lengths'][row])
        native=np.arange(n)*data['dt'];keep=times<=native[-1]+1e-10;s=trace['predicted'][row,:n]
        pred=np.stack([np.interp(times[keep],native,s[:,k]) for k in (0,1)],-1)
        distance=np.linalg.norm(pred-truth[keep,:2],axis=-1)
        raw.append(dict(index=idx,position_rmse_m=float(np.sqrt((distance**2).mean())),maximum_position_error_m=float(distance.max())))
    result['selected_raw_interpolation']=dict(model=chosen,sample_dt_s=.0005,method='linear interpolation of native positions; no analytic contact reconstruction',
        position_rmse_m=quantiles([r['position_rmse_m'] for r in raw]),maximum_position_error_m=quantiles([r['maximum_position_error_m'] for r in raw]),per_episode=raw)
    atomic_json(output/'paired_results.json',result)
    print(json.dumps(dict(complete=True,output=str(output/'paired_results.json'))),flush=True)


if __name__=='__main__':main()
