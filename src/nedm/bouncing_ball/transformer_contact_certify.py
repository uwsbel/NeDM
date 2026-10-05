"""Freeze validation-selected checkpoints and compare on a fresh Chrono cohort."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import shutil
from pathlib import Path
import numpy as np
import torch
from nedm.bouncing_ball.collection import atomic_json
from nedm.bouncing_ball.model import load_model
from nedm.bouncing_ball.transformer_contact import load_transformer_contact
from nedm.bouncing_ball.transformer_contact_eval import data_packet,free_metrics,teacher_metrics,launch_gradient_check
from nedm.bouncing_ball.precision_certify import raw_trace
from nedm.bouncing_ball.training import quantiles


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("select","certify"))
    p.add_argument("--root",type=Path,required=True);p.add_argument("--data",type=Path,required=True)
    p.add_argument("--names",nargs="+");p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("Run on AMD compute nodes")
    torch.set_num_threads(1)
    packet=data_packet(args.data,args.device,1)
    if args.mode=="select":
        candidates=[]
        val=torch.nonzero(packet["splits"]==1).flatten()
        for name in args.names:
            root=args.root/"runs"/name
            done=json.loads((root/"complete.json").read_text())
            assert done["complete"] and not done["smoke"] and not done["test_used_for_selection"]
            metadata=torch.load(root/"best.pt",map_location="cpu",weights_only=False)
            model,_=load_transformer_contact(root/"best.pt",args.device)
            measured=free_metrics(model,packet,val)
            destination=args.root/"frozen"/name
            destination.mkdir(parents=True,exist_ok=False)
            shutil.copyfile(root/"best.pt",destination/"best.pt")
            candidates.append(dict(name=name,checkpoint=str(destination/"best.pt"),sha256=done["checkpoint_sha256"],
                validation=measured,training_selection_validation=metadata["validation"],model_config=metadata["model_config"],stage=metadata["stage"],update=metadata["update"]))
        groups={"transformer_only":["transformer_only","transformer_only_refined"],
            "literal_primary":["transformer_xz_binary","transformer_xz_binary_refined"],
            "surface_modes":["transformer_xz_modes","transformer_xz_modes_refined"],
            "fullstate_gate":["transformer_state5_binary","transformer_state5_binary_refined"],
            "soft_gate":["transformer_xz_soft"]}
        chosen={group:min([c for c in candidates if c["name"] in names],key=lambda c:c["validation"]["selection_score"])["name"] for group,names in groups.items()}
        atomic_json(args.root/"selection.json",dict(candidates=candidates,chosen=chosen,test_used_for_selection=False,
            selection_split="validation",data_sha256=packet["index"]["model_data_sha256"],job_id=os.environ["SLURM_JOB_ID"]))
        print(json.dumps(dict(frozen=[c["name"] for c in candidates],test_used_for_selection=False)),flush=True)
        return
    selection=json.loads((args.root/"selection.json").read_text())
    assert not selection["test_used_for_selection"]
    ids=torch.nonzero(packet["splits"]==2).flatten();assert len(ids)==900
    output=args.root/"certification";output.mkdir(parents=True,exist_ok=False)
    report=dict(fresh_test_episodes=len(ids),fresh_seed=202610014,dt_s=packet["dt"],
        data_sha256=packet["index"]["model_data_sha256"],selection_before_test=True,models={},
        chosen_before_test=selection["chosen"],job_id=os.environ["SLURM_JOB_ID"],host=platform.node(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for candidate in selection["candidates"]:
        source=Path(candidate["checkpoint"]);assert hashlib.sha256(source.read_bytes()).hexdigest()==candidate["sha256"]
        model,_=load_transformer_contact(source,args.device)
        metrics=free_metrics(model,packet,ids,output/(candidate["name"]+"_traces.npz"))
        teacher=teacher_metrics(model,packet,ids)
        gradients=launch_gradient_check(model,[[4.3,-10.2],[5.5,-9.75],[6.8,-9.2]])
        report["models"][candidate["name"]]=dict(checkpoint_sha256=candidate["sha256"],model_config=model.config,
            free_rollout=metrics,teacher_forced=teacher,launch_gradients=gradients)
        print(json.dumps(dict(name=candidate["name"],p95_trajectory_mm=metrics["position_rmse_m"]["p95"]*1000,
            p95_endpoint_mm=metrics["endpoint_error_m"]["p95"]*1000,order=metrics["contact_order_fraction"])),flush=True)
    old=Path('/work1/dannegrut/harry/experiments/ball_span_v1_20260930/runs')
    for name,folder in (("historical_v1","nrd_v1"),("historical_v1_refined","nrd_v1_refine"),("historical_v2_certified","nrd_v2_certified")):
        source=old/folder/"best.pt";model,_=load_model(source,args.device);model.double()
        assert abs(model.dt-packet["dt"])<1e-12
        metrics=free_metrics(model,packet,ids,output/(name+"_traces.npz"))
        report["models"][name]=dict(checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            note="historical Transformer with known flight/geometry priors; not a pure residual-only control",
            model_config=model.config,free_rollout=metrics)
        print(json.dumps(dict(name=name,p95_trajectory_mm=metrics["position_rmse_m"]["p95"]*1000,
            p95_endpoint_mm=metrics["endpoint_error_m"]["p95"]*1000,order=metrics["contact_order_fraction"])),flush=True)
    # Same linear native-position interpolation for every model: no invented
    # analytic contact timing. Raw within-step comparisons remain separate.
    raw={name:[] for name in report["models"]}
    traces={name:dict(np.load(output/(name+"_traces.npz"))) for name in raw}
    for row,idx in enumerate(ids.cpu().tolist()):
        _,times,truth=raw_trace((idx,packet["index"]["episodes"][idx]))
        for name,trace in traces.items():
            n=int(trace["lengths"][row]);s=trace["predicted"][row,:n]
            native=np.arange(n)*packet["dt"];keep=times<=native[-1]+1e-10
            prediction=np.stack([np.interp(times[keep],native,s[:,k]) for k in (0,1)],-1)
            distance=np.linalg.norm(prediction-truth[keep,:2],axis=-1)
            raw[name].append(dict(index=idx,position_rmse_m=float(np.sqrt((distance**2).mean())),
                endpoint_error_m=float(distance[-1]),maximum_position_error_m=float(distance.max())))
    report["raw_position_interpolation"]={name:dict(sample_dt_s=.0005,method="linear interpolation of native 10 ms positions, identical for all models, no analytical event reconstruction",
        **{key:quantiles([r[key] for r in rows]) for key in ("position_rmse_m","endpoint_error_m","maximum_position_error_m")},per_episode=rows) for name,rows in raw.items()}
    atomic_json(output/"paired_results.json",report)
    print(json.dumps(dict(complete=True,output=str(output/"paired_results.json"))),flush=True)


if __name__=="__main__":main()
