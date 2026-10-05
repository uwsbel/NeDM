"""Freeze candidates using validation, then paired evaluation on new Chrono data."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import shutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch
from nedm.bouncing_ball.collection import atomic_json
from nedm.bouncing_ball.model import load_model
from nedm.bouncing_ball.latent_model import load_latent
from nedm.bouncing_ball.latent_evaluation import transition_data,teacher_metrics,free_metrics,launch_gradient_check,learned_dense_trajectory
from nedm.bouncing_ball.precision_certify import raw_trace
from nedm.bouncing_ball.training import quantiles


class BaselineAdapter:
    """Preserve baseline native updates while comparing on a common grid."""
    def __init__(self,model,dt):
        self.model=model
        self.dt=dt
        self.stride=round(dt/model.dt)
        assert abs(self.stride*model.dt-dt)<1e-12

    def details(self,state):
        y=self.model.rollout(state,self.stride)[:,-1]
        signatures=torch.stack(((state[:,3]<0)&(y[:,3]>0),(state[:,2]>0)&(y[:,2]<0)),-1)
        logits=torch.where(signatures,torch.full_like(signatures,100.,dtype=state.dtype),torch.full_like(signatures,-100.,dtype=state.dtype))
        return y,{"logits":logits,"phase_s":torch.full_like(logits,.5*self.dt)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("select","certify"))
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--names",nargs="+")
    p.add_argument("--baseline",type=Path)
    p.add_argument("--precision",type=Path)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"): raise RuntimeError("Run evaluation on AMD compute nodes")
    torch.set_num_threads(1)
    packet=transition_data(args.data,args.device,5)
    val=torch.nonzero(packet["splits"]==1).flatten()
    if args.mode=="select":
        results=[]
        for name in args.names:
            completion=json.loads((args.root/"runs"/name/"complete.json").read_text())
            assert completion["complete"] and not completion["smoke"]
            model, metadata=load_latent(args.root/"runs"/name/"best.pt",args.device)
            modes=[("native",model.config.get("learned_phase",True),model.config.get("gate","hard"))]
            if model.config["stage"]=="response" and model.config.get("learned_phase",True) and model.config["contact_input_dim"]==5:
                modes += [("soft",True,"soft"),("midpoint",False,"hard")]
            for suffix,phase,gate in modes:
                candidate=copy.deepcopy(model)
                # Config and dt are saved with the chosen inference behavior.
                candidate.config={**candidate.config,"learned_phase":phase,"gate":gate}
                teacher=teacher_metrics(candidate,packet,val)
                free=free_metrics(candidate,packet,val)
                source=args.root/"runs"/name/"best.pt"
                entry={"name":name+"_"+suffix,"source_checkpoint":str(source),
                    "source_checkpoint_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"inference":candidate.config,
                    "teacher_forced":teacher,"free_rollout":free,"selection_score":free["selection_score"]}
                results.append(entry)
                print(json.dumps({"name":entry["name"],"trajectory_p95_mm":free["position_rmse_m"]["p95"]*1000,
                    "endpoint_p95_mm":free["endpoint_error_m"]["p95"]*1000,"order":free["contact_order_fraction"],
                    "f1":[c["f1"] for c in teacher["contact"]]}),flush=True)
        groups={"stage1":[r for r in results if r["inference"]["stage"]=="response"],
                "stage2":[r for r in results if r["inference"]["stage"]!="response"]}
        chosen={}
        for group,items in groups.items():
            best=min(items,key=lambda r:r["selection_score"])
            folder=args.root/"frozen"/group
            folder.mkdir(parents=True,exist_ok=False)
            original=torch.load(best["source_checkpoint"],map_location="cpu",weights_only=False)
            original["model_config"]=best["inference"]
            original["selection_note"]={"validation_only":True,"inference_ablation":best["name"]}
            torch.save(original,folder/"best.pt")
            chosen[group]={"candidate":best["name"],"checkpoint":str(folder/"best.pt"),
                "checkpoint_sha256":hashlib.sha256((folder/"best.pt").read_bytes()).hexdigest(),"validation":best}
        report={"selection_split":"validation","test_data_used":False,"chosen":chosen,"candidates":results,
            "data_sha256":packet["index"]["model_data_sha256"],"job_id":os.environ["SLURM_JOB_ID"],
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        atomic_json(args.root/"selection.json",report)
    else:
        choice=json.loads((args.root/"selection.json").read_text())
        assert choice["selection_split"]=="validation" and not choice["test_data_used"]
        ids=torch.nonzero(packet["splits"]==2).flatten()
        assert len(ids)==900
        output=args.root/"certification"
        output.mkdir(parents=True,exist_ok=False)
        report={"fresh_test_episodes":len(ids),"fresh_seed":202610013,"model_dt_s":packet["dt"],
            "data_sha256":packet["index"]["model_data_sha256"],"selected_before_test":choice["chosen"],
            "job_id":os.environ["SLURM_JOB_ID"],"host":platform.node(),"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"models":{}}
        # Certify every predeclared ablation; do not select from test results.
        # Frozen winners are annotated, all negatives remain in the report.
        for candidate in choice["candidates"]:
            model,_=load_latent(candidate["source_checkpoint"],args.device)
            model.config=candidate["inference"]
            teacher=teacher_metrics(model,packet,ids)
            free=free_metrics(model,packet,ids)
            gradient=launch_gradient_check(model,[[4.3,-10.2],[5.5,-9.75],[6.8,-9.2]])
            report["models"][candidate["name"]]={"teacher_forced":teacher,"free_rollout":free,"launch_gradients":gradient,
                "source_checkpoint_sha256":candidate["source_checkpoint_sha256"]}
            print(json.dumps({"test_candidate":candidate["name"],"trajectory_p95_mm":free["position_rmse_m"]["p95"]*1000,
                              "endpoint_p95_mm":free["endpoint_error_m"]["p95"]*1000,"order":free["contact_order_fraction"]}),flush=True)
        for name,path in (("original_transformer",args.baseline),("analytical_contact_precision",args.precision)):
            model,_=load_model(path,args.device)
            model.double()
            free=free_metrics(BaselineAdapter(model,packet["dt"]),packet,ids)
            free["contact_from_predicted_states_note"]="kinematic signatures for baseline adapter, not a learned latent classification score"
            report["models"][name]={"free_rollout":free,"checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
            print(json.dumps({"baseline":name,"trajectory_p95_mm":free["position_rmse_m"]["p95"]*1000,
                "endpoint_p95_mm":free["endpoint_error_m"]["p95"]*1000,"order":free["contact_order_fraction"]}),flush=True)
        chosen=choice["chosen"]["stage1"]
        model,_=load_latent(chosen["checkpoint"],args.device)
        assert hashlib.sha256(Path(chosen["checkpoint"]).read_bytes()).hexdigest()==chosen["checkpoint_sha256"]
        maximum_duration=max(packet["index"]["episodes"][int(i)]["model_duration_s"] for i in ids.cpu().tolist())
        steps=math.ceil(maximum_duration/model.dt-1e-10)
        dense=learned_dense_trajectory(model,packet["states"][ids,0],steps,100).cpu().numpy()
        with torch.no_grad(): native=model.rollout(packet["states"][ids,0],steps).cpu().numpy()
        agreement=float(abs(dense[:,::100]-native).max())
        assert agreement<1e-8
        row={int(index):i for i,index in enumerate(ids.cpu().tolist())}
        tasks=[(int(i),packet["index"]["episodes"][int(i)]) for i in ids.cpu().tolist()]
        dense_entries=[]
        with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context("spawn")) as pool:
            for i,times,truth in pool.map(raw_trace,tasks,chunksize=8):
                samples=(times/.0005).round().astype("int64")
                prediction=dense[row[i],samples]
                distance=((prediction[:,:2]-truth[:,:2])**2).sum(-1)**.5
                dense_entries.append({"index":i,"position_rmse_m":float((distance**2).mean()**.5),
                    "endpoint_error_m":float(distance[-1]),"maximum_position_error_m":float(distance.max())})
        report["raw_dense_stage1"]={"candidate":chosen["candidate"],"sample_dt_s":.0005,"episodes":len(dense_entries),
            "native_dense_state_max_difference":agreement,"inference":"learned interval contact logits and phase, no analytical contact detector",
            **{k:quantiles([e[k] for e in dense_entries]) for k in ("position_rmse_m","endpoint_error_m","maximum_position_error_m")},
            "per_episode":dense_entries}
        atomic_json(output/"paired_results.json",report)
        print(json.dumps({"complete":True,"output":str(output/"paired_results.json")}),flush=True)


if __name__=="__main__":main()
