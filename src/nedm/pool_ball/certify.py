"""Evaluate frozen checkpoints on the sealed cohort and on validation, side by side.

Run only after every model is frozen. The sealed cohort is used for nothing
else before this step. Selection stays the validation choice recorded here
(lowest validation score), whatever the sealed numbers say.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch

from nedm.pool_ball.campaign import atomic_json
from nedm.pool_ball.evaluate import free_metrics, load_data
from nedm.pool_ball.model import load_pool


def summarise(report):
    keys = ("b_rmse_m", "b_at_target_m", "eligible_b_at_target_m", "a_rmse_m", "b_end_m")
    out = {k: {q: 1000 * report[k][q] for q in ("median", "p95", "max")} for k in keys}
    out.update(episodes=report["episodes"], eligible_episodes=report["eligible_episodes"],
               event_ok=report["event_ok_fraction"], eligible_event_ok=report["eligible_event_ok_fraction"],
               finite=report["finite_fraction"], b_at_target_over_10mm=report["b_at_target_over_10mm"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--sealed-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("AMD compute nodes only")
    torch.set_num_threads(1)
    validation_data = load_data(args.train_data, "cuda")
    sealed = load_data(args.sealed_data, "cuda")
    val = torch.nonzero(validation_data["splits"] == 1).flatten()
    test = torch.nonzero(sealed["splits"] == 2).flatten()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = {}
    for run in args.runs:
        checkpoint = run / "best.pt"
        if not checkpoint.exists():
            rows[run.name] = {"missing": True}
            continue
        model, packet = load_pool(checkpoint, "cuda")
        v = free_metrics(model, validation_data, val)
        s = free_metrics(model, sealed, test, save_traces=args.output / f"{run.name}_sealed_traces.npz")
        atomic_json(args.output / f"{run.name}_sealed.json", s)
        rows[run.name] = {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                          "stage": packet["stage"], "update": packet["update"], "model_config": packet["model_config"],
                          "validation_score": v["selection_score"], "validation": summarise(v), "sealed": summarise(s)}
        print(json.dumps({run.name: {"val_score": v["selection_score"], "sealed_b_target_p95_mm": rows[run.name]["sealed"]["b_at_target_m"]["p95"],
                                      "sealed_eligible_p95_mm": rows[run.name]["sealed"]["eligible_b_at_target_m"]["p95"]}}), flush=True)
    present = {k: r for k, r in rows.items() if not r.get("missing")}
    selected = min(present, key=lambda k: present[k]["validation_score"]) if present else None
    atomic_json(args.output / "certification.json", {"rows": rows, "selected_by_validation": selected,
                                                     "sealed_data_sha256": sealed["index"]["model_data_sha256"],
                                                     "host": platform.node(), "job_id": os.environ["SLURM_JOB_ID"]})
    print(json.dumps({"selected_by_validation": selected}))


if __name__ == "__main__":
    main()
