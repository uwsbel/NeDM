"""Teacher-forced one-step contact error per pair (validation only).

For every pair, validation windows in which that pair is the only one in contact:
the model's next state with the true switches (gates = labels) against Chrono.
Errors of the affected bodies, split into position, velocity and spin; plus the
same for windows with no contact (core only). Works for version-2 and version-3
checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from nedm.contact_nrd.evaluate import load_data, window_labels
from nedm.contact_nrd.model_v2 import load_any


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="name=path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-pair", type=int, default=4000)
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID")
    data = load_data(args.data, "cuda")
    states, stride, lengths, splits = data["states"], data["stride"], data["lengths"], data["splits"]
    labels = window_labels(data["contacts"], stride)
    n_win = labels.shape[1]
    vwin = (torch.arange(n_win, device="cuda")[None, :] < (lengths - stride)[:, None]) & (splits == 1)[:, None]
    system = data["system"]
    pairs = system["pairs"]
    moving = [k for k, b in enumerate(system["bodies"]) if b["moving"]]
    gen = torch.Generator(device="cuda").manual_seed(7)
    single = labels.sum(-1) == 1
    sets = {}
    for p, (i, j) in enumerate(pairs):
        w = torch.nonzero(vwin & single & labels[..., p])
        if len(w):
            w = w[torch.randperm(len(w), device="cuda", generator=gen)[: args.per_pair]]
        bodies = [moving.index(i)] + ([moving.index(j)] if j in moving else [])
        sets[f"{system['bodies'][i]['name']}-{system['bodies'][j]['name']}"] = (w, bodies)
    w = torch.nonzero(vwin & ~labels.any(-1))
    sets["no contact"] = (w[torch.randperm(len(w), device="cuda", generator=gen)[: args.per_pair]], list(range(len(moving))))
    out = {}
    for item in args.checkpoints:
        name, path = item.split("=", 1)
        model, _ = load_any(Path(path), "cuda")
        h_len = max(getattr(model, "k", 1), model.context)
        rows = {}
        for key, (w, bodies) in sets.items():
            if not len(w):
                continue
            t = w[:, 1, None] + torch.arange(1 - h_len, 1, device="cuda") * stride
            t = torch.where(t < 0, w[:, 1, None] % stride, t)
            hist = states[w[:, 0, None], t]
            s, y = states[w[:, 0], w[:, 1]], states[w[:, 0], w[:, 1] + stride]
            g = labels[w[:, 0], w[:, 1]].to(s.dtype)
            pred = torch.cat([model.details(a, b, c)[0] for a, b, c in zip(s.split(2048), hist.split(2048), g.split(2048))])
            err = (pred - y)[:, bodies]
            rows[key] = {"windows": len(w),
                         "position_mm": {"rms": 1e3 * float(err[..., 0:3].norm(dim=-1).square().mean().sqrt()),
                                         "p95": 1e3 * float(torch.quantile(err[..., 0:3].norm(dim=-1).flatten().float(), 0.95))},
                         "velocity_mm_s": {"rms": 1e3 * float(err[..., 3:6].norm(dim=-1).square().mean().sqrt()),
                                           "p95": 1e3 * float(torch.quantile(err[..., 3:6].norm(dim=-1).flatten().float(), 0.95))},
                         "spin_rad_s": {"rms": float(err[..., 6:9].norm(dim=-1).square().mean().sqrt()),
                                        "p95": float(torch.quantile(err[..., 6:9].norm(dim=-1).flatten().float(), 0.95))}}
        out[name] = rows
        print(json.dumps({name: {k: (round(v["position_mm"]["rms"], 4), round(v["velocity_mm_s"]["rms"], 2), round(v["spin_rad_s"]["rms"], 3))
                                 for k, v in rows.items()}}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
