"""Preserve train/validation episodes and add a fresh sealed test cohort."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from nedm.bouncing_ball.collection import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary",type=Path,required=True)
    parser.add_argument("--heldout",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--margin-s",type=float,default=.02)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise FileExistsError(args.output_dir)
    trajectories,labels,episodes,sources = [],[],[],[]
    for source, keep in [(args.primary,[0,1]),(args.heldout,[2])]:
        index = json.loads((source / "campaign_index.json").read_text())
        if not index["complete"] or hashlib.sha256((source / "model_data.npz").read_bytes()).hexdigest()!=index["model_data_sha256"]:
            raise ValueError("invalid source campaign")
        sources.append({"root":str(source.resolve()),"model_data_sha256":index["model_data_sha256"],"episodes":index["episode_count"]})
        data = dict(np.load(source / "model_data.npz"))
        for i in np.flatnonzero(np.isin(data["splits"],keep)):
            item = dict(index["episodes"][i])
            end = next(e["time_s"] for e in item["events"] if e["kind"]=="next_ground_contact")
            n = min(int(data["lengths"][i]),int(np.floor((end-args.margin_s)/index["model_dt_s"]+1e-9))+1)
            wall = next(e["time_s"] for e in item["events"] if e["kind"]=="wall_release")
            if (n-1)*index["model_dt_s"]<wall+.15:raise ValueError("insufficient post-wall horizon")
            trajectories.append(data["states"][i,:n])
            labels.append(data["contacts"][i,:n-1])
            item.update(raw_data_root=str(source.resolve()),model_samples=n,model_duration_s=(n-1)*index["model_dt_s"])
            episodes.append(item)
    if len({e["episode_id"] for e in episodes})!=len(episodes):raise ValueError("episode leakage")
    counts = Counter((tuple(e["launch_cell"]),e["split"]) for e in episodes)
    primary_index = json.loads((args.primary / "campaign_index.json").read_text())
    for ix in range(primary_index["campaign"]["grid"][0]):
        for iz in range(primary_index["campaign"]["grid"][1]):
            for split,count in primary_index["campaign"]["episodes_per_cell"].items():
                if counts[((ix,iz),split)]!=count:raise ValueError("cell coverage quota mismatch")
    maximum = max(len(state) for state in trajectories)
    states = np.zeros((len(episodes),maximum,5),np.float32)
    contacts = np.zeros((len(episodes),maximum-1,2),np.float32)
    for i,(state,contact) in enumerate(zip(trajectories,labels,strict=True)):
        states[i,:len(state)]=state;states[i,len(state):]=state[-1]
        contacts[i,:len(contact)]=contact
    args.output_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output_dir / "model_data.npz",states=states,contacts=contacts,
                        lengths=np.array([len(s) for s in trajectories]),
                        splits=np.array([{"train":0,"val":1,"test":2}[e["split"]] for e in episodes]),
                        cells=np.array([e["launch_cell"] for e in episodes]),
                        launches=np.array([[e["launch"]["vx_mps"],e["launch"]["vz_mps"]] for e in episodes]))
    report = {**primary_index,"episodes":episodes,"episode_count":len(episodes),"source_campaigns":sources,
              "terminal_margin_s":args.margin_s,"test_policy":f"fresh seed {json.loads((args.heldout / 'campaign_index.json').read_text())['campaign']['seed']}; original test is diagnostic only",
              "model_duration_range_s":[min(e["model_duration_s"] for e in episodes),max(e["model_duration_s"] for e in episodes)],
              "model_data_sha256":hashlib.sha256((args.output_dir / "model_data.npz").read_bytes()).hexdigest()}
    atomic_json(args.output_dir / "campaign_index.json",report)
    print(json.dumps({"episodes":len(episodes),"split_counts":dict(Counter(e["split"] for e in episodes)),
                      "model_duration_range_s":report["model_duration_range_s"],"terminal_margin_s":args.margin_s}),flush=True)


if __name__ == "__main__":main()
