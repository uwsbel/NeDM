"""Exploratory out-of-range cohorts: position error of every arm on one unified-format cohort.

Unified models get the cohort's scene (fixed-body geometry) through
model_v2.retarget; earlier models cannot take a scene and run as trained.
Metrics (target body, 10 ms grid, from the first state): path RMSE (pool: to
t = 2 s; ball: whole valid episode), error at the target time, end error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from nedm.contact_nrd.adapters import BallView, PoolView
from nedm.contact_nrd.evaluate import load_data, quantiles
from nedm.contact_nrd.model_v2 import load_any, retarget


@torch.no_grad()
def rollout_positions(kind, model, initial, steps):
    """initial [N, M, 9] unified -> target-body positions [N, steps+1, 3] (unified frame)."""
    if kind == "unified":
        return model.rollout(initial, steps)
    if kind == "pool":
        s14 = PoolView.to_pool(initial)
        out = model.rollout(s14, steps)
        u = torch.zeros(*out.shape[:-1], 2, 9, dtype=out.dtype, device=out.device)
        for b in (0, 1):
            u[..., b, 0:2] = out[..., 7 * b:7 * b + 2]
        return u
    s5 = BallView.to_ball(initial)
    out = model.rollout(s5, steps)
    u = torch.zeros(*out.shape[:-1], 1, 9, dtype=out.dtype, device=out.device)
    u[..., 0, 0], u[..., 0, 2] = out[..., 0], out[..., 1]
    return u


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--unified", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", nargs="*", default=[], help="name=checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID")
    torch.set_num_threads(1)
    data = load_data(args.data, "cuda")
    system = data["system"]
    name = system["name"]
    stride, target = data["stride"], data["target_step"]
    body = system["target_body"]
    ids = torch.arange(len(data["episodes"]), device="cuda")
    arms = []
    for run in args.unified:
        model, _ = load_any(run / "best.pt", "cuda")
        retarget(model, system)
        arms.append((run.name, "unified", model, run / "best.pt"))
    for item in args.reference:
        label, path = item.split("=", 1)
        if name == "pool":
            from nedm.pool_ball.model import load_pool
            model = load_pool(Path(path), "cuda")[0]
        else:
            from nedm.bouncing_ball.model import load_model
            model = load_model(Path(path), "cuda")[0].double()
            if abs(model.dt - data["dt"]) > 1e-12:
                model.dt = data["dt"]
        arms.append((label, name, model, Path(path)))
    rows = {}
    for label, kind, model, path in arms:
        per = []
        for chosen in ids.split(256):
            truth = data["states"][chosen][:, ::stride]
            pred = rollout_positions(kind, model, truth[:, 0], truth.shape[1] - 1)
            slot = data_slot(system, body)
            p = pred[..., slot, :3].cpu().numpy()
            t = truth[..., slot, :3].cpu().numpy()
            for row, idx in enumerate(chosen.tolist()):
                n = (int(data["lengths"][idx]) - 1) // stride + 1
                plane = [0, 1] if name == "pool" else [0, 2]  # the motion plane of each system
                err = np.linalg.norm((p[row, :n] - t[row, :n])[:, plane], axis=-1)
                if not np.isfinite(err).all():
                    per.append({"finite": False})
                    continue
                upto = target + 1 if name == "pool" else n
                per.append({"finite": True, "rmse_m": float(np.sqrt(np.mean(err[:upto] ** 2))),
                            "at_target_m": float(err[min(target, n - 1)]), "end_m": float(err[-1])})
        finite = [e for e in per if e["finite"]]
        rows[label] = {"kind": kind, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "finite": len(finite) / len(per),
                       **{k + "_mm": {q: 1e3 * v for q, v in quantiles([e[k] for e in finite]).items()} for k in ("rmse_m", "at_target_m", "end_m")}}
        print(json.dumps({label: rows[label]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"data": str(args.data), "data_sha256": data["index"]["data_sha256"], "system": name,
                                       "episodes": len(ids), "scene": system["bodies"], "rows": rows}, indent=1))


def data_slot(system, body):
    moving = [k for k, b in enumerate(system["bodies"]) if b["moving"]]
    return moving.index(body)


if __name__ == "__main__":
    main()
