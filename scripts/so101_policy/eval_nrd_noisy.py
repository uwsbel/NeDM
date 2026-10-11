"""Grasp NRD accuracy on policy-like commands (NRD data round of the SO101 policy study): the 'before' and 'after'
numbers of a fine-tune.

    PYTHONPATH=src python scripts/so101_policy/eval_nrd_noisy.py --data ~/so101_policy/data/noisy_v1 \\
        --model s61=~/so101_policy/ckpt/nrd_final_s61.pt s62=~/so101_policy/ckpt/nrd_final_s62.pt \\
        --out ~/so101_policy/runs/nrd_round/before_noisy_v1_val.json [--split 1] [--families ...] [--batch B]

Truth = the MuJoCo record (the collector's episodes; recorded commands). Episodes: one split (default 1 = validation;
never the sealed test, split 2: refused) of the task families (select: true; or --families). Each model runs the
evaluation of the data's system.json (data.evaluate, configs/contact_nrd/evaluation/README.md): a free rollout from
the first recorded state with the recorded commands (cube position error at the target time 4 s and at the last
valid record), 1 s windows from recorded 4-state histories every 0.5 s (cube position and angle, arm joint RMS), the
outcome flags. Lift flag (evaluation rule): some step with cube z > 25 mm, both jaw switches on and the table switch
off; recorded from the record and its contact labels, predicted from the free rollout and the model's switches (a
non-finite rollout counts as wrong). With two or more models: the ensemble table (lift flag wrong in each model; wrong
and the first two models disagree; both wrong with the same answer). Groups: all, each family, the OU noise std
bins of the noisy families (variant "s<std mrad>t<tau cs>j<jaw offset mrad>": 5-10, 10-15, 15-20 mrad).

Output JSON: per model the summary of each group (errors in mm and degrees, p50 / p95; lift flags), the ensemble
table and per-episode lift flags. A compact table is printed.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

from nedm.contact_nrd.data import add_derived, add_relative, evaluate, file_rows, file_sha256, load_data, step_average_rates
from nedm.contact_nrd.model import load_model


def pick(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or d.get(k) is None:
            return None
        d = d[k]
    return d


def brief(r):
    """Headline numbers of one evaluate summary (one group): mm, degrees, rad, counts."""
    def q(x, k=1e3):
        return None if x is None else [round(k * x["median"], 3), round(k * x["p95"], 3)]
    lift = pick(r, "free", "flags", "lifted") or {}
    return dict(episodes=r["episodes"], score=r.get("selection_score"), finite=r.get("finite_fraction"),
                free_4s_cube_mm=q(pick(r, "free", "target", "cube", "pos_m")),
                free_end_cube_mm=q(pick(r, "free", "final", "cube", "pos_m")),
                free_4s_cube_deg=q(pick(r, "free", "target", "cube", "angle_rad"), 180 / math.pi),
                win1s_cube_mm=q(pick(r, "windows", "1.0", "cube", "pos_m")),
                win1s_cube_deg=q(pick(r, "windows", "1.0", "cube", "angle_rad"), 180 / math.pi),
                win1s_arm_joint_rms_mrad=q(pick(r, "windows", "1.0", "arm", "joint_rms_rad")),
                lift=dict(recorded=lift.get("recorded"), predicted=lift.get("predicted"), missed=lift.get("missed"),
                          false=lift.get("false"), nonfinite=lift.get("nonfinite"), disagreement=lift.get("disagreement")))


def noise_bin(variant):
    m = re.match(r"s(\d+)t", str(variant))
    if not m:
        return None
    s = int(m.group(1))
    return "std_05_10mrad" if s < 10 else "std_10_15mrad" if s < 15 else "std_15_20mrad"


def ensemble(flags, groups):
    """flags {model: (recorded [E] bool, predicted [E] bool or None for non-finite)} -> per group counts."""
    names = list(flags)
    rec = flags[names[0]][0]
    wrong = {n: np.array([p is None or bool(p) != bool(r) for r, p in zip(rec, flags[n][1])]) for n in names}
    pred = {n: np.array([bool(p) if p is not None else False for p in flags[n][1]]) for n in names}
    out = {}
    for g, sel in groups.items():
        d = {"episodes": int(sel.sum()), "recorded_lifts": int(rec[sel].sum())}
        for n in names:
            d[n] = dict(wrong=int((wrong[n] & sel).sum()), false_lift=int((pred[n] & ~rec & sel).sum()),
                        missed_lift=int((~pred[n] & rec & sel).sum()),
                        wrong_pct=round(100 * float((wrong[n] & sel).sum()) / max(1, int(sel.sum())), 2))
        if len(names) >= 2:
            a, b = names[:2]
            disagree = pred[a] != pred[b]
            d["seeds_disagree"] = int((disagree & sel).sum())
            for n in (a, b):
                d[n]["wrong_and_seeds_disagree"] = int((wrong[n] & disagree & sel).sum())
            d["both_wrong_same_answer"] = int((wrong[a] & wrong[b] & ~disagree & sel).sum())
            d["either_wrong"] = int(((wrong[a] | wrong[b]) & sel).sum())
        out[g] = d
    return out


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", nargs="+", required=True, metavar="NAME=PATH", help="checkpoint or run directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", type=int, default=1, help="0 train or 1 validation (2 = sealed test: refused)")
    parser.add_argument("--families", nargs="+", help="default: every family with select: true")
    parser.add_argument("--batch", type=int, help="rollouts per GPU batch (default: the evaluation block's; memory only)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.split not in (0, 1):
        raise SystemExit("only split 0 or 1: the sealed test is not for this script")
    t0 = time.perf_counter()
    models = {}
    for item in args.model:
        name, path = item.split("=", 1)
        path = Path(path).expanduser()
        path = path / "best.pt" if path.is_dir() else path
        models[name] = (path, load_model(path, args.device)[0])
    first = next(iter(models.values()))[1]
    for _, m in models.values():
        if json.dumps(m.config, sort_keys=True) != json.dumps(first.config, sort_keys=True):
            raise SystemExit("the models have different configs (state channels); score them separately")
    data = load_data(args.data.expanduser(), args.device, model_step_s=first.dt, core_only="drop")
    if "evaluation" not in data["system"]:
        raise SystemExit("the data need an evaluation block in system.json")
    if args.batch:
        data["system"]["evaluation"]["batch"] = args.batch
    step_average_rates(data, set(first.config["integrate"]["kinds"]))
    add_derived(data, first.config.get("derived"))
    add_relative(data, first.config.get("relative"))
    fams = data["system"]["families"]
    names = [f["name"] for f in fams]
    chosen = args.families or [f["name"] for f in fams if f["select"]]
    unknown = sorted(set(chosen) - set(names))
    if unknown:
        raise SystemExit(f"unknown families {unknown}; the data have {names}")
    keep = torch.tensor([n in chosen for n in names], device=args.device)
    idx = torch.nonzero((data["splits"] == args.split) & keep[data["family"]]).flatten()
    rows = data["file_index"][idx].cpu().numpy() if "file_index" in data else idx.cpu().numpy()
    with np.load(args.data.expanduser() / "unified_data.npz") as f:
        variant = f["variant"][rows] if "variant" in f.files else np.array([""] * len(rows))
    famrow = data["family"][idx].cpu().numpy()
    groups = {"all": np.ones(len(idx), bool)}
    groups.update({names[k]: famrow == k for k in np.unique(famrow)})
    bins = np.array([noise_bin(v) for v in variant], dtype=object)
    for b in ("std_05_10mrad", "std_10_15mrad", "std_15_20mrad"):
        if (bins == b).any():
            groups[b] = bins == b
    print(f"{args.data}: split {args.split}, {len(idx)} episodes, families {chosen}", flush=True)
    out = dict(data=str(args.data), data_sha256=file_sha256(args.data.expanduser() / "unified_data.npz"), split=args.split,
               families=chosen, episodes=len(idx), models={}, groups={k: int(v.sum()) for k, v in groups.items()})
    flags = {}
    for name, (path, model) in models.items():
        t = time.perf_counter()
        rep = file_rows(evaluate(model, data, idx, by_family=True), data)
        per = rep["per_episode"]
        rec = np.array([e["free"]["flags"]["lifted"]["recorded"] for e in per])
        pred = [e["free"]["flags"]["lifted"]["predicted"] for e in per]
        flags[name] = (rec, pred)
        summary = {"all": brief(rep), **{k: brief(v) for k, v in rep.get("families", {}).items()}}
        out["models"][name] = dict(checkpoint=str(path), checkpoint_sha256=file_sha256(path), summary=summary,
                                   selected_score=pick(rep, "selected", "selection_score"),
                                   seconds=round(time.perf_counter() - t, 1),
                                   per_episode=[dict(row=int(e["index"]), family=e.get("family"),
                                                     lifted_recorded=bool(e["free"]["flags"]["lifted"]["recorded"]),
                                                     lifted_predicted=e["free"]["flags"]["lifted"]["predicted"],
                                                     free_4s_cube_mm=None if e["free"]["target"]["cube"]["pos_m"] is None
                                                     else round(1e3 * e["free"]["target"]["cube"]["pos_m"], 3))
                                                for e in per])
        print(f"{name}: {time.perf_counter() - t:.0f} s", flush=True)
        for g, s in summary.items():
            print(f"  {name} {g:18s} n {s['episodes']:5d}  1 s cube p50/p95 {s['win1s_cube_mm']} mm  "
                  f"4 s {s['free_4s_cube_mm']} mm  end {s['free_end_cube_mm']} mm  lift rec/pred {s['lift']['recorded']}/"
                  f"{s['lift']['predicted']} missed {s['lift']['missed']} false {s['lift']['false']}  score {s['score']}",
                  flush=True)
    out["ensemble"] = ensemble(flags, groups)
    for g, d in out["ensemble"].items():
        print(f"  ensemble {g:18s} " + json.dumps({k: v for k, v in d.items()}), flush=True)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
