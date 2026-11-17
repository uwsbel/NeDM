"""Score models of the architecture ablation (3T, 3M, 1T, 1M) on a test directory (every episode in it).

From the repo root:

    PYTHONPATH=src python -m nedm.contact_nrd.arch_ablation.test --data <case>/test --output <results>/test_scores.json \\
        --model 3t=<case>/models/seed61.pt 3m=<runs>/pool_3m_s61 1t=<runs>/pool_1t_s61 1m=<runs>/pool_1m_s61

PATH is a checkpoint file or a run directory with best.pt; a released contact NRD checkpoint is a 3T. Every model is
scored by ``nedm.contact_nrd.data.evaluate`` (the contact NRD's test: free rollouts from the first recorded state, or
the task metrics of the data's ``evaluation`` block). Two scores per model:

* ``motion_score``: the selection score without its contact-event terms. A one-network model predicts no contacts, so
  this is the score all four are compared on.
* ``selection_score``: the contact NRD's own score. For a 1T or 1M it is not given: its contact switches are all off.

The output JSON holds each model's report without the per-episode rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nedm.contact_nrd.arch_ablation import checkpoints, common, one_net
from nedm.contact_nrd.data import evaluate, file_sha256, load_data, step_average_rates


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="test directory (system.json, unified_data.npz)")
    parser.add_argument("--model", nargs="+", required=True, metavar="NAME=PATH", help="checkpoint or run directory")
    parser.add_argument("--output", type=Path, help="JSON file with every model's report")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    models = {}
    for item in args.model:
        name, path = item.split("=", 1)
        path = Path(path) / "best.pt" if Path(path).is_dir() else Path(path)
        models[name] = (path, *checkpoints.load_model(path, args.device))
    data = load_data(args.data, args.device, model_step_s=next(iter(models.values()))[1].dt)
    if "evaluation" in data["system"]:   # as nedm.contact_nrd.test: task metrics compare step-average rates
        kinds = {tuple(sorted(m.config["integrate"]["kinds"])) for _, m, _ in models.values()}
        if len(kinds) != 1:
            raise ValueError("the models integrate different body kinds; score them separately")
        step_average_rates(data, set(kinds.pop()))
    ids = torch.arange(len(data["splits"]), device=args.device)
    results = {}
    for name, (path, model, packet) in models.items():
        arch = checkpoints.arch_of(packet)
        one = arch in ("1t", "1m")
        report = (one_net.motion_evaluate(evaluate, model) if one else evaluate)(model, data, ids)
        report.pop("per_episode", None)
        trained = model.one if one else model
        results[name] = dict(arch=arch, parameters=common.count(trained), checkpoint=str(path),
                             checkpoint_sha256=file_sha256(path), motion_score=common.motion_score(report),
                             **{**report, "selection_score": None if one else report["selection_score"]})
        official = "n/a" if one else f"{report['selection_score']:.4f}"
        print(f"{name} ({arch}, {results[name]['parameters']} parameters): motion score "
              f"{results[name]['motion_score']:.4f}, selection score {official}, finite {report['finite_fraction']:.4f}", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(data=str(args.data), data_sha256=file_sha256(args.data / "unified_data.npz"),
                                               target_step=data["target_step"], dt_s=data["dt"], models=results), indent=1, default=str))


if __name__ == "__main__":
    main()
