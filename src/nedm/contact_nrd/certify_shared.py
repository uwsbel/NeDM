"""Score frozen unified checkpoints and the frozen earlier models on one fresh
cohort with the earlier study's own evaluator, then give paired verdicts.

Pool: nedm.pool_ball.evaluate.free_metrics on the pool-format cohort.
Ball: nedm.bouncing_ball.transformer_contact_eval.free_metrics on the merged
certification packet (split 2 = the fresh cohort).
Verdict (pre-registered): per statistic (median, p95) of the primary metric,
r = unified / reference with a paired bootstrap 95 % interval (10,000
resamples of episodes, same resample for both arms): beats if upper < 1,
matches if upper <= 1.25, worse if lower > 1.25, else inconclusive. Counts use
an exact McNemar test. Any non-finite rollout fails the arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from nedm.contact_nrd.adapters import BallView, PoolView
from nedm.contact_nrd.model_v2 import load_any, retarget
from nedm.contact_nrd.systems import ball_system, pool_system

PRIMARY = {"pool": "b_at_target_m", "ball": "position_rmse_m"}
SECONDARY = {"pool": ("b_rmse_m", "b_end_m", "a_rmse_m"), "ball": ("endpoint_error_m", "maximum_position_error_m")}
EVENT = {"pool": "event_ok", "ball": "contact_order"}
TAIL = {"pool": "b_at_target_m", "ball": "endpoint_error_m"}


def bootstrap_ratio(a, b, stat, rng, n=10000):
    ratios = np.empty(n)
    for k in range(n):
        i = rng.integers(0, len(a), len(a))
        ratios[k] = stat(a[i]) / stat(b[i])
    return float(stat(a) / stat(b)), float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def verdict(lo, hi):
    if hi < 1.0:
        return "beats"
    if hi <= 1.25:
        return "matches"
    if lo > 1.25:
        return "worse"
    return "inconclusive"


def mcnemar(x, y):
    """x, y boolean failure indicators per episode; exact two-sided p-value."""
    b, c = int((x & ~y).sum()), int((~x & y).sum())
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def p95(x):
    return np.quantile(x, 0.95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("pool", "ball"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--unified", type=Path, nargs="+", required=True, help="unified run directories (best.pt); name=dir also accepted")
    parser.add_argument("--reference", nargs="*", default=[], help="name=checkpoint")
    parser.add_argument("--ensemble", nargs="*", default=[], help="name=run_dir1,run_dir2,... (averaged unified models)")
    parser.add_argument("--headline", required=True, help="unified run name used for verdicts")
    parser.add_argument("--comparison", required=True, help="reference name for the matching claim")
    parser.add_argument("--retarget", action="store_true", help="give unified models the cohort's fixed-body geometry")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID")
    torch.set_num_threads(1)
    if args.system == "pool":
        from nedm.pool_ball.evaluate import free_metrics, load_data
        from nedm.pool_ball.model import load_pool
        data = load_data(args.data, "cuda")
        sha = data["index"]["model_data_sha256"]

        def load_reference(path):
            return load_pool(path, "cuda")[0]
    else:
        from nedm.bouncing_ball.model import load_model
        from nedm.bouncing_ball.transformer_contact_eval import data_packet, free_metrics
        data = data_packet(args.data, "cuda")
        sha = data["index"]["model_data_sha256"]

        def load_reference(path):
            model = load_model(path, "cuda")[0].double()
            if abs(model.dt - data["dt"]) > 1e-12:
                model.dt = data["dt"]  # timestep-flexible analytic reference, as in its certification
            return model
    ids = torch.nonzero(data["splits"] == 2).flatten()
    cache = {round(data["dt"], 9): data}

    def data_for(dt):
        """The cohort on a model's own step (10 ms or 20 ms); per-episode results stay aligned by episode."""
        key = round(dt, 9)
        if key not in cache:
            if args.system == "pool":
                cache[key] = load_data(args.data, "cuda", model_step_s=dt)
            else:
                cache[key] = data_packet(args.data, "cuda", stride=round(dt / data["dt"]))
        return cache[key]
    arms = {}
    for item in args.unified:
        label, run = (str(item).split("=", 1) if "=" in str(item) else (None, str(item)))
        run = Path(run)
        model, packet = load_any(run / "best.pt", "cuda")
        if args.retarget:
            retarget(model, (pool_system if args.system == "pool" else ball_system)(data["index"]["config"]))
        view = (PoolView if args.system == "pool" else BallView)(model)
        arms[label or run.name] = (view, run / "best.pt")
    for item in args.reference:
        name, path = item.split("=", 1)
        arms[name] = (load_reference(Path(path)), Path(path))
    from nedm.contact_nrd.ensemble import load_ensemble
    for item in args.ensemble:
        name, runs = item.split("=", 1)
        runs = runs.split(",")
        model = load_ensemble(runs, "cuda")
        arms[name] = ((PoolView if args.system == "pool" else BallView)(model), Path(runs[0]) / "best.pt")
    rows, per = {}, {}
    for name, (model, path) in arms.items():
        m = free_metrics(model, data_for(model.dt), ids)
        per[name] = m["per_episode"]
        keys = (PRIMARY[args.system], *SECONDARY[args.system])
        rows[name] = {"checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      **{k + "_mm": {q: 1e3 * m[k][q] for q in ("median", "p95", "max")} for k in keys},
                      "finite": m["finite_fraction"],
                      "events_ok": m.get("event_ok_fraction", m.get("contact_order_fraction"))}
        if args.system == "pool":
            rows[name].update(eligible_b_at_target_mm={q: 1e3 * m["eligible_b_at_target_m"][q] for q in ("median", "p95", "max")},
                              b_at_target_over_10mm=m["b_at_target_over_10mm"], eligible_episodes=m["eligible_episodes"],
                              eligible_events_ok=m["eligible_event_ok_fraction"])
        else:
            rows[name]["endpoint_over_10mm"] = sum(e["endpoint_error_m"] > 0.01 for e in m["per_episode"] if e["finite"])
        print(json.dumps({name: rows[name]}), flush=True)
    rng = np.random.default_rng(20261002)
    head = per[args.headline]
    verdicts = {}
    for name in arms:
        if name == args.headline:
            continue
        ref = per[name]
        both = [k for k in range(len(head)) if head[k]["finite"] and ref[k]["finite"]]
        out = {"episodes": len(both), "unified_finite": rows[args.headline]["finite"], "reference_finite": rows[name]["finite"],
               "comparison": name == args.comparison}
        for key in (PRIMARY[args.system], *SECONDARY[args.system]):
            a = np.array([head[k][key] for k in both])
            b = np.array([ref[k][key] for k in both])
            out[key] = {}
            for stat_name, stat in (("median", np.median), ("p95", p95)):
                r, lo, hi = bootstrap_ratio(a, b, stat, rng)
                out[key][stat_name] = {"unified_mm": 1e3 * float(stat(a)), "reference_mm": 1e3 * float(stat(b)),
                                       "ratio": r, "ci95": [lo, hi], "verdict": verdict(lo, hi)}
            if key != PRIMARY[args.system]:
                out[key]["note"] = "secondary, descriptive"
        ev = EVENT[args.system]
        ue = np.array([not head[k].get(ev, False) for k in range(len(head))])
        re_ = np.array([not ref[k].get(ev, False) for k in range(len(head))])
        out["event_failures"] = {"unified": int(ue.sum()), "reference": int(re_.sum()), "mcnemar_p": mcnemar(ue, re_)}
        tail = TAIL[args.system]
        ut = np.array([(not head[k]["finite"]) or head[k][tail] > 0.01 for k in range(len(head))])
        rt = np.array([(not ref[k]["finite"]) or ref[k][tail] > 0.01 for k in range(len(head))])
        out["over_10mm"] = {"metric": tail, "unified": int(ut.sum()), "reference": int(rt.sum()), "mcnemar_p": mcnemar(ut, rt)}
        verdicts[name] = out
        print(json.dumps({"vs": name, "primary": out[PRIMARY[args.system]], "events": out["event_failures"], "over_10mm": out["over_10mm"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    slim = {name: [{k: v for k, v in e.items() if not isinstance(v, (list, tuple))} for e in entries] for name, entries in per.items()}
    args.output.write_text(json.dumps({"system": args.system, "data": str(args.data), "data_sha256": sha, "episodes": len(ids),
                                       "retarget": args.retarget,
                                       "headline": args.headline, "comparison": args.comparison, "rows": rows,
                                       "verdicts": verdicts, "per_episode": slim}, indent=1))


if __name__ == "__main__":
    main()
