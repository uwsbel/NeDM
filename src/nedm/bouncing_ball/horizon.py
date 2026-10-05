"""Diagnose end-of-episode contacts and prepare a pre-impact model horizon."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from nedm.bouncing_ball.collection import atomic_json
from nedm.bouncing_ball.model import load_model
from nedm.bouncing_ball.training import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-data", type=Path)
    parser.add_argument("--margin-s", type=float, default=.02)
    args = parser.parse_args()
    torch.set_num_threads(4)
    index = json.loads((args.data / "campaign_index.json").read_text())
    data = dict(np.load(args.data / "model_data.npz"))
    model, checkpoint = load_model(args.checkpoint,"cuda")
    states = torch.as_tensor(data["states"],device="cuda")
    contacts = torch.as_tensor(data["contacts"],device="cuda")
    lengths = torch.as_tensor(data["lengths"],device="cuda")
    val = torch.as_tensor(np.flatnonzero(data["splits"]==1),device="cuda")
    limits = checkpoint["runtime"]["config"]["evaluation"]
    ends = np.array([next(e["time_s"] for e in item["events"] if e["kind"]=="next_ground_contact") for item in index["episodes"]])
    full = evaluate(model,states,lengths,contacts,val,limits)
    failures = []
    for entry in full["per_episode"]:
        if entry["contact_order"]:
            continue
        i, n = entry["index"], int(data["lengths"][entry["index"]])
        with torch.no_grad():
            prediction = model.rollout(states[i:i+1,0],n-1)[0].cpu().numpy()
        ground = (np.flatnonzero((prediction[:-1,3]<0)&(prediction[1:,3]>=0))+1)*model.dt
        wall = (np.flatnonzero((prediction[:-1,2]>0)&(prediction[1:,2]<=0))+1)*model.dt
        failures.append({"index": i, "ground_times_s": ground.tolist(), "wall_times_s": wall.tolist(),
                         "true_next_ground_s": float(ends[i]), "model_end_s": (n-1)*model.dt})
    variants = {}
    for margin in sorted({.02,.05,args.margin_s}):
        trimmed = np.minimum(data["lengths"],np.floor((ends-margin)/model.dt+1e-9).astype(int)+1)
        result = evaluate(model,states,torch.as_tensor(trimmed,device="cuda"),contacts,val,limits)
        variants[str(margin)] = {key:value for key,value in result.items() if key!="per_episode"}
    atomic_json(args.report,{"validation_only": True,"full": {k:v for k,v in full.items() if k!="per_episode"},
                             "contact_failures": failures,"margin_evaluations": variants})
    if args.output_data:
        output = args.output_data
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(output)
        output.mkdir(parents=True,exist_ok=True)
        data["lengths"] = np.minimum(data["lengths"],np.floor((ends-args.margin_s)/model.dt+1e-9).astype(int)+1)
        maximum = int(data["lengths"].max())
        data["states"] = data["states"][:,:maximum].copy()
        data["contacts"] = data["contacts"][:,:maximum-1].copy()
        for i,n in enumerate(data["lengths"]):
            data["states"][i,n:] = data["states"][i,n-1]
            data["contacts"][i,n-1:] = 0
            item = index["episodes"][i]
            wall_time = next(e["time_s"] for e in item["events"] if e["kind"]=="wall_release")
            if (n-1)*model.dt < wall_time+.15:
                raise ValueError("margin removed required post-wall motion")
            item["model_samples"] = int(n)
            item["model_duration_s"] = (int(n)-1)*model.dt
        np.savez_compressed(output / "model_data.npz",**data)
        index["raw_data_root"] = str(args.data.resolve())
        index["source_model_data_sha256"] = index["model_data_sha256"]
        index["model_data_sha256"] = hashlib.sha256((output / "model_data.npz").read_bytes()).hexdigest()
        index["terminal_margin_s"] = args.margin_s
        index["model_duration_range_s"] = [float((data["lengths"].min()-1)*model.dt),float((data["lengths"].max()-1)*model.dt)]
        atomic_json(output / "campaign_index.json",index)
    print(json.dumps({"contact_failures":len(failures),"margins":variants},indent=2),flush=True)


if __name__ == "__main__":
    main()
