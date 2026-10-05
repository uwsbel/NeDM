"""Five-target launch optimization through frozen state-only contact NRD."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from nedm.bouncing_ball.collection import atomic_json
from nedm.bouncing_ball.latent_model import load_latent
from nedm.bouncing_ball.latent_evaluation import launch_gradient_check
from nedm.bouncing_ball.optimize_launch import optimize,benchmark_targets,check_loss_gradients


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"): raise RuntimeError("AMD allocation required")
    torch.set_num_threads(1)
    model,metadata=load_latent(args.checkpoint,"cpu")
    config=json.loads(args.config.read_text())
    index=json.loads((args.data/"campaign_index.json").read_text())
    physics=index["config"]
    steps=round(config["terminal_time_s"]/model.dt)
    specifications=benchmark_targets(args.data,config,steps)
    targets=torch.tensor([s["xz_m"] for s in specifications],dtype=torch.float64)
    starts=targets.new_tensor(config["initial_velocity_mps"]).expand_as(targets).clone()
    bounds=tuple(targets.new_tensor([physics["launch"]["vx_range_mps"][k],physics["launch"]["vz_range_mps"][k]]) for k in (0,1))
    args.output.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter()
    with (args.output/"history.jsonl").open("w") as stream:
        def progress(entry):
            stream.write(json.dumps(entry)+"\n");stream.flush()
            if entry["iteration"]%10==0:print(json.dumps({"iteration":entry["iteration"],"distance_m":entry["distance_m"]}),flush=True)
        # No geometric contact decision, timing, surface threshold or filter
        # is consulted in this optimizer or model rollout.
        result=optimize(model,targets,starts,bounds,steps=steps,iterations=config["iterations"],
            learning_rate=config["learning_rate"],backtracks=config["backtracks"],max_step=config["max_step_mps"],
            tolerance=config["nrd_tolerance_m"],require_two_impacts=False,progress=progress)
    restarts=[]
    for i in range(len(targets)):
        if result["loss"][i] <= config["nrd_tolerance_m"]**2: continue
        seeds=targets.new_tensor(config.get("restart_velocities_mps",[]))
        if seeds.numel()==0: continue
        alternate=optimize(model,targets[i:i+1].expand_as(seeds),seeds,bounds,steps=steps,iterations=config["iterations"],
            learning_rate=config["learning_rate"],backtracks=config["backtracks"],max_step=config["max_step_mps"],
            tolerance=config["nrd_tolerance_m"],require_two_impacts=False,stop_when_any=True)
        selected=int(alternate["loss"].argmin())
        improved=bool(alternate["loss"][selected]<result["loss"][i])
        restarts.append({"case":specifications[i]["name"],"fixed_starts_mps":seeds.tolist(),"selected":selected,
            "final_distances_m":alternate["loss"].sqrt().tolist(),"improved":improved,"history":alternate["history"]})
        if not improved:continue
        offset=result["history"][-1]["iteration"]+1
        for row in alternate["history"]:
            merged={"iteration":offset+row["iteration"],"restart_case":specifications[i]["name"]}
            for key in ("loss_m2","distance_m","velocity_mps","gradient","learning_rate"):
                merged[key]=list(result["history"][-1][key]);merged[key][i]=row[key][selected]
            result["history"].append(merged)
        for iteration,snapshot in alternate["snapshots"]:
            combined=result["after"].numpy().copy();combined[i]=snapshot[selected]
            result["snapshots"].append((offset+iteration,combined))
        result["velocity"][i]=alternate["velocity"][selected]
        result["after"][i]=alternate["after"][selected]
        result["loss"][i]=alternate["loss"][selected]
    previous,following=result["after"][:,:-1],result["after"][:,1:]
    ground=(previous[:,:,3]<0)&(following[:,:,3]>0)
    wall=(previous[:,:,2]>0)&(following[:,:,2]<0)
    valid=(ground.sum(1)==1)&(wall.sum(1)==1)&(ground.to(torch.int64).argmax(1)<wall.to(torch.int64).argmax(1))
    cases=[]
    for i,spec in enumerate(specifications):
        cases.append({**spec,"initial_velocity_mps":starts[i].tolist(),"optimized_velocity_mps":result["velocity"][i].tolist(),
            "initial_nrd_endpoint_m":result["before"][i,-1,:2].tolist(),"optimized_nrd_endpoint_m":result["after"][i,-1,:2].tolist(),
            "initial_nrd_distance_m":result["history"][0]["distance_m"][i],
            "optimized_nrd_distance_m":float(result["loss"][i].sqrt()),"nrd_two_impacts":bool(valid[i])})
    report={"config":config,"physics_config":physics,"cases":cases,
        "objective":"L=(x(T)-target_x)^2+(z(T)-target_z)^2","optimizer":"projected gradient with exact-model backtracking and predeclared fixed restarts; no Chrono feedback",
        "checkpoint":str(args.checkpoint),"checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "model_config":model.config,"weights_frozen":all(not p.requires_grad for p in model.parameters()),"dtype":"float64",
        "target_data_sha256":index["model_data_sha256"],"host":platform.node(),"job_id":os.environ["SLURM_JOB_ID"],
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"iterations":result["history"][-1]["iteration"],
        "elapsed_s":time.perf_counter()-began,"nrd_passed":all(c["nrd_two_impacts"] and c["optimized_nrd_distance_m"]<=config["nrd_tolerance_m"] for c in cases),
        "no_analytical_contact_filter":True}
    atomic_json(args.output/"optimization.json",report)
    atomic_json(args.output/"restarts.json",restarts)
    atomic_json(args.output/"history.json",result["history"])
    atomic_json(args.output/"gradient_checks.json",{"initial":check_loss_gradients(model,starts,targets,steps,(0.,1.),0.),
        "optimized":check_loss_gradients(model,result["velocity"],targets,steps,(0.,1.),0.)})
    np.savez_compressed(args.output/"trajectories.npz",before=result["before"].cpu().numpy(),after=result["after"].cpu().numpy(),
        time_s=np.arange(steps+1)*model.dt,targets=targets.cpu().numpy(),
        snapshot_iterations=[s[0] for s in result["snapshots"]],snapshots=np.stack([s[1] for s in result["snapshots"]]))
    print(json.dumps({"nrd_passed":report["nrd_passed"],"distances_m":[c["optimized_nrd_distance_m"] for c in cases],"iterations":report["iterations"]}),flush=True)


if __name__=="__main__":main()
