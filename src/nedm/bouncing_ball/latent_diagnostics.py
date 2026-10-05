"""Post-selection learned-switch gradient stress and public checkpoint loading."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import torch
from nedm.bouncing_ball.collection import atomic_json
from nedm.bouncing_ball.model import load_model
from nedm.bouncing_ball.latent_evaluation import transition_data,launch_gradient_check


@torch.no_grad()
def decisions(model,actions):
    state=torch.cat((torch.zeros_like(actions[:,:1]),torch.ones_like(actions[:,:1]),actions,torch.zeros_like(actions[:,:1])),-1)
    sequence=[];phases=[]
    for _ in range(round(1.7/model.dt)):
        state, details=model.details(state)
        sequence.append(details["logits"]>=0)
        codes=(details["raw_phase"]<=0).to(torch.int64)+2*(details["raw_phase"]>=1).to(torch.int64)
        phases.append(torch.where(details["logits"]>=0,codes,torch.full_like(codes,-1)))
    return torch.stack(sequence,1),torch.stack(phases,1)


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("AMD compute allocation required")
    torch.set_num_threads(1)
    selection=json.loads((args.root/"selection.json").read_text())
    packet=transition_data(args.data,"cpu",5)
    val=torch.nonzero(packet["splits"]==1).flatten()
    broad=val[torch.linspace(0,len(val)-1,64).round().long()]
    wall=packet["contacts"][val,:,1].argmax(1)
    phase=packet["phases"][val,wall,1]
    margin=torch.minimum(phase,packet["dt"]-phase).abs()
    edges=val[margin.argsort()[:24]]
    chosen=torch.cat((broad,edges)).unique(sorted=True)
    launches=packet["states"][chosen,0,2:4].tolist()
    report={"selection_frozen_before_audit":True,"validation_only":True,"probe_episodes":len(chosen),
        "probe_definition":"64 span samples plus24 validation launches whose TRUE wall impact is closest to an interval boundary; truth chooses stress cases only, never inference input",
        "job_id":os.environ["SLURM_JOB_ID"],"host":platform.node(),"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"models":{}}
    for name,entry in selection["chosen"].items():
        model,metadata=load_model(entry["checkpoint"],"cpu")
        assert hashlib.sha256(Path(entry["checkpoint"]).read_bytes()).hexdigest()==entry["checkpoint_sha256"]
        gradient=launch_gradient_check(model,launches)
        action=torch.tensor(launches,dtype=torch.float64)
        base,base_phase=decisions(model,action)
        changes=[]
        for epsilon in (1e-5,1e-4,1e-3):
            different=torch.zeros(len(action),dtype=torch.bool);phase_changed=torch.zeros_like(different)
            for k in (0,1):
                offset=torch.zeros_like(action);offset[:,k]=epsilon
                positive,positive_phase=decisions(model,action+offset)
                negative,negative_phase=decisions(model,action-offset)
                different |= ((positive!=base)|(negative!=base)).any((1,2))
                phase_changed |= ((positive_phase!=base_phase)|(negative_phase!=base_phase)).any((1,2))
            changes.append({"epsilon":epsilon,"contact_sequence_changed_episodes":int(different.sum()),
                "active_phase_clip_branch_changed_episodes":int(phase_changed.sum()) if model.config["stage"]=="response" else None})
        launch=torch.tensor([[5.5,-9.75]],dtype=torch.float32,requires_grad=True)
        initial=torch.cat((torch.zeros(1,1),torch.ones(1,1),launch,torch.zeros(1,1)),-1)
        trajectory=model.rollout(initial,round(1.7/model.dt))
        grad=torch.autograd.grad(trajectory[:,-1,:2].sum(),launch)[0]
        with torch.no_grad(): double=model.rollout(initial.detach().double(),round(1.7/model.dt))
        result={"checkpoint_sha256":entry["checkpoint_sha256"],"gradient_stress":gradient,"contact_sequence_changes":changes,
            "public_loading":{"loaded_via":"nedm.bouncing_ball.model.load_model","float32_launch_gradient_finite":bool(torch.isfinite(grad).all()),
                "trajectory_shape":list(trajectory.shape),"float32_vs_float64_max_state_difference":float((trajectory.detach()-double).abs().max()),
                "separate_bounce_experts":2 if hasattr(model,"responses") else 0,
                "analytical_geometry_attributes_present":any(hasattr(model,k) for k in ("floor_center_z","wall_center_x","impact_time"))}}
        report["models"][name]=result
        print(json.dumps({"model":name,"gradient_fd":gradient["finite_differences"],"changed_contact_sequences":changes,"public_loading":result["public_loading"]}),flush=True)
    folder=args.output or args.root/"diagnostics";folder.mkdir(parents=True,exist_ok=False)
    atomic_json(folder/"loading_and_gradients.json",report)


if __name__=="__main__":main()
