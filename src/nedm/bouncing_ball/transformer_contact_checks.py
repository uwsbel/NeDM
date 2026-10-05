"""Actual checkpoint loading and validation-only branch diagnostics on AMD."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from nedm.bouncing_ball.collection import atomic_json
from nedm.bouncing_ball.model import load_model
from nedm.bouncing_ball.transformer_contact_eval import data_packet
from nedm.bouncing_ball.training import quantiles
from nedm.core.training.model_transformer import ContinuousTransformer


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--data',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run model checks on AMD compute node')
    torch.set_num_threads(1)
    selection=json.loads((args.root/'selection.json').read_text());rows=[]
    for role in ['transformer_only','literal_primary','fullstate_gate','surface_modes']:
        name=selection['chosen'][role];record=next(r for r in selection['candidates'] if r['name']==name)
        source=Path(record['checkpoint']);assert hashlib.sha256(source.read_bytes()).hexdigest()==record['sha256']
        model,metadata=load_model(source,'cpu')
        assert isinstance(model.backbone,ContinuousTransformer)
        assert model.backbone.config.input_dim==5
        assert not any(hasattr(model,k) for k in ['gravity','floor_center_z','wall_center_x','physics_dt'])
        action=torch.tensor([[5.5,-9.75]],dtype=torch.float32,requires_grad=True)
        initial=torch.cat((torch.zeros_like(action[:,:1]),torch.ones_like(action[:,:1]),action,torch.zeros_like(action[:,:1])),-1)
        predicted=model.rollout(initial,170);loss=predicted[:,-1,:2].square().sum()
        gradient,=torch.autograd.grad(loss,action)
        with torch.no_grad():reference=model.rollout(initial.detach().double(),170)
        rows.append(dict(role=role,name=name,shape=list(predicted.shape),backbone=type(model.backbone).__name__,
            finite=bool(torch.isfinite(predicted).all()),finite_launch_gradient=bool(torch.isfinite(gradient).all()),
            float32_to_float64_state_difference=float((reference-predicted.detach()).abs().max()),
            checkpoint_sha256=record['sha256'],gate_input_dim=model.gate_input_dim,bounce_input_dim=2))
    packet=data_packet(args.data,'cpu',1);ids=torch.nonzero(packet['splits']==1).flatten();diagnostics={}
    with torch.no_grad():
        for role in ['literal_primary','fullstate_gate']:
            name=selection['chosen'][role];record=next(r for r in selection['candidates'] if r['name']==name)
            model,_=load_model(record['checkpoint'],'cpu');entries=[]
            for batch in ids.split(256):
                current=packet['states'][batch,0];predictions=[current];gates=[];responses=[]
                steps=int(packet['lengths'][batch].max())-1
                for step in range(steps):
                    following,info=model.details(current)
                    gates.append(info['gates'].sum(-1));responses.append((following-current-info['delta1'])[:,2:4].norm(dim=-1))
                    predictions.append(following);current=following
                predicted=torch.stack(predictions,1).numpy();gate=torch.stack(gates,1).numpy();response=torch.stack(responses,1).numpy()
                for row,idx in enumerate(batch.tolist()):
                    n=int(packet['lengths'][idx]);truth=packet['states'][idx,:n].numpy()
                    labels=packet['contacts'][idx,:n-1].numpy().sum(-1)>.5
                    first=int(np.flatnonzero(labels)[0])+1
                    distance=np.linalg.norm(predicted[row,:n,:2]-truth[:,:2],axis=-1)
                    bad=np.flatnonzero(distance>.01)
                    entries.append(dict(index=idx,pre_first_contact_max_error_m=float(distance[:first].max()),
                        first_10mm_error_time_s=float(bad[0]*model.dt) if len(bad) else None,
                        expected_first_contact_time_s=first*model.dt,
                        gate_on_steps=int((gate[row,:n-1]>0).sum()),
                        false_on_steps=int(((gate[row,:n-1]>0)&~labels).sum()),
                        missed_label_steps=int(((gate[row,:n-1]==0)&labels).sum()),
                        false_on_mean_velocity_correction_mps=float(response[row,:n-1][(gate[row,:n-1]>0)&~labels].mean()) if ((gate[row,:n-1]>0)&~labels).any() else 0.))
            diagnostics[role]=dict(name=name,episodes=len(entries),
                pre_first_contact_max_error_m=quantiles([r['pre_first_contact_max_error_m'] for r in entries]),
                gate_on_steps=quantiles([r['gate_on_steps'] for r in entries]),
                total_false_on_steps=sum(r['false_on_steps'] for r in entries),total_missed_label_steps=sum(r['missed_label_steps'] for r in entries),
                first_10mm_error_before_first_contact=sum(r['first_10mm_error_time_s'] is not None and r['first_10mm_error_time_s']<r['expected_first_contact_time_s'] for r in entries),
                per_episode=entries)
    output=args.root/'diagnostics';output.mkdir(exist_ok=True)
    atomic_json(output/'loading_and_branches.json',dict(public_loader=rows,validation_only_branch_diagnostics=diagnostics,job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(dict(public_loader=rows,diagnostics={k:{j:v for j,v in r.items() if j!='per_episode'} for k,r in diagnostics.items()}),indent=2),flush=True)


if __name__=='__main__':main()
