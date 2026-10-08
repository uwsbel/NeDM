"""Score trained contact NRD models on a test directory (every episode in it).

From the repo root (after ``python -m nedm.contact_nrd.download --part test models``):

    PYTHONPATH=src python -m nedm.contact_nrd.test --data artifacts/contact_nrd/pool/test \\
        --model seed61=artifacts/contact_nrd/pool/models/seed61.pt seed62=artifacts/contact_nrd/pool/models/seed62.pt

PATH is a checkpoint file or a run directory with best.pt. Each episode is a free rollout from its first recorded
state with its recorded commands (data.free_metrics). One line per model, each error as median / p95: the target
body's position error at the target time, its orientation error (only if it has a quaternion) and its path RMS error
up to the target time; then the fraction of episodes whose contact events match the recorded ones and the fraction
of finite rollouts.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from nedm.contact_nrd.data import channel_index, file_sha256, free_metrics, load_data
from nedm.contact_nrd.model import load_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="test directory (system.json, unified_data.npz)")
    parser.add_argument("--model", nargs="+", required=True, metavar="NAME=PATH", help="checkpoint or run directory")
    parser.add_argument("--output", type=Path, help="JSON file with the summary and the per-episode results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    models = {}
    for item in args.model:
        name, path = item.split("=", 1)
        path = Path(path) / "best.pt" if Path(path).is_dir() else Path(path)
        models[name] = (path, load_model(path, args.device)[0])
    data = load_data(args.data, args.device, model_step_s=next(iter(models.values()))[1].dt)
    system = data["system"]
    has_quat = bool(channel_index(system, system["bodies"][system["target_body"]]["name"], "quaternion"))
    ids = torch.arange(len(data["splits"]), device=args.device)
    results = {}
    for name, (path, model) in models.items():
        m = free_metrics(model, data, ids)
        line = (f"{name}: target position {1000 * m['target_pos_m']['median']:.3f} / {1000 * m['target_pos_m']['p95']:.3f} mm")
        if has_quat:
            line += (f", target angle {math.degrees(m['target_angle_rad']['median']):.3f} / "
                     f"{math.degrees(m['target_angle_rad']['p95']):.3f} deg")
        line += (f", path RMS {1000 * m['path_rmse_m']['median']:.3f} / {1000 * m['path_rmse_m']['p95']:.3f} mm"
                 f" (median / p95), events ok {m['event_ok_fraction']:.4f}, finite {m['finite_fraction']:.4f}"
                 f" ({m['episodes']} episodes)")
        print(line, flush=True)
        results[name] = dict(checkpoint=str(path), checkpoint_sha256=file_sha256(path),
                             **{k: m[k] for k in ("target_pos_m", "target_angle_rad", "path_rmse_m", "event_ok_fraction",
                                                  "finite_fraction", "episodes", "per_episode")})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(data=str(args.data), data_sha256=file_sha256(args.data / "unified_data.npz"),
                                               target_step=data["target_step"], dt_s=data["dt"], models=results), indent=1))


if __name__ == "__main__":
    main()
