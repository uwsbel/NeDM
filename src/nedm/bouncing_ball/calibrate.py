"""Select smooth contact localization width using validation rollouts only."""
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
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--widths",type=float,nargs="+",default=[1e-4,3e-5,1e-5,3e-6])
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    data = np.load(args.data / "model_data.npz")
    states, lengths, contacts = [torch.as_tensor(data[key],device="cuda") for key in ["states","lengths","contacts"]]
    indices = torch.as_tensor(np.flatnonzero(data["splits"]==1),device="cuda")
    model, checkpoint = load_model(args.checkpoint,"cuda")
    limits = checkpoint["runtime"]["config"]["evaluation"]
    entries, best = [], float("inf")
    for width in args.widths:
        model.width = width
        model.config["gate_width_m"] = width
        result = evaluate(model,states,lengths,contacts,indices,limits)
        compact = {key:value for key,value in result.items() if key!="per_episode"}
        entries.append({"gate_width_m":width,"validation":compact})
        # Penalize large tail failures as well as typical rollout error.
        score = result["selection_score"] + result["position_rmse_m"]["max"]*.1 + result["endpoint_error_m"]["max"]*.1
        if score <= best:  # Equal validation fits prefer the sharper instantaneous contact.
            best = score
            candidate = {**checkpoint,"model_config":dict(model.config),"validation":result,
                         "contact_calibration":{"validation_only":True,"original_checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                                                "gate_width_m":width,"selection_score":score}}
            torch.save(candidate,args.output_dir / "best.pt")
            atomic_json(args.output_dir / "validation.json",result)
        print(json.dumps({"gate_width_m":width,"position_p95_m":result["position_rmse_m"]["p95"],
                          "position_max_m":result["position_rmse_m"]["max"],"endpoint_max_m":result["endpoint_error_m"]["max"],
                          "contact_order_fraction":result["contact_order_fraction"],"passed":result["passed"]}),flush=True)
    atomic_json(args.output_dir / "calibration.json",{"validation_only":True,"candidates":entries})


if __name__ == "__main__":main()
