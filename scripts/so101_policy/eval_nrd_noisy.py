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

Before / after table of a fine-tune (no GPU; reads output JSONs of this script, same data and split in each):

    python scripts/so101_policy/eval_nrd_noisy.py --compare before=before_noisy_v1_val.json after=after_noisy_v1_val.json \\
        --out compare_noisy_v1_val.json

One row per (group, model): episodes, 1 s window cube position p95 (mm, recorded 4-state histories every 0.5 s), free
rollout cube position p95 at 4 s (mm), lift flag wrong (% of the group's episodes; missed / false). Noise-bin groups
have no window statistics (per-episode 1 s errors are not stored): their 4 s p95 comes from the per-episode values.
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
    def q(x, k=1e3):   # [median, p95] scaled; None where a family has no such statistic (e.g. probes without windows)
        if x is None:
            return None
        return [None if x.get(s) is None else round(k * x[s], 3) for s in ("median", "p95")]
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


def sign_test(fixed, broken):
    """Exact two-sided binomial (McNemar) p of fixed vs broken discordant pairs."""
    n, k = fixed + broken, min(fixed, broken)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(items, out, gate=1.10):
    """Before / after table (module docstring) of output JSONs given as LABEL=PATH -> rows (also written to out).
    With exactly two files (before, after) also, per group and model name in both: the forgetting gate (after <= gate x
    before for the 1 s and 4 s cube p95, lift-wrong count <= max(gate x before, before + 1)) and the paired lift flags
    (episodes fixed: wrong before, right after; broken: the reverse; exact sign-test p)."""
    files = {}
    for item in items:
        label, path = item.split("=", 1)
        files[label] = json.loads(Path(path).expanduser().read_text())
    ref = next(iter(files.values()))
    for label, f in files.items():
        if (f["data_sha256"], f["split"], f["families"]) != (ref["data_sha256"], ref["split"], ref["families"]):
            raise SystemExit(f"{label}: other data, split or families than the first file")
    rows = []
    for g in ref["groups"]:
        for label, f in files.items():
            for name, m in f["models"].items():
                s = m["summary"].get(g)
                e = f["ensemble"].get(g, {}).get(name, {})
                if s is None:   # a noise bin: 4 s p95 from the per-episode values of that bin's rows
                    if g not in f["ensemble"]:
                        continue
                    rows_g = set(f.get("group_rows", {}).get(g, []))
                    vals = [p["free_4s_cube_mm"] for p in m["per_episode"] if p["row"] in rows_g and p["free_4s_cube_mm"] is not None]
                    s = dict(episodes=f["ensemble"][g]["episodes"], win1s_cube_mm=None,
                             free_4s_cube_mm=[None, round(float(np.percentile(vals, 95)), 3)] if vals else None)
                rows.append(dict(group=g, label=label, name=name, model=f"{label}:{name}", episodes=s["episodes"],
                                 win1s_cube_p95_mm=(s.get("win1s_cube_mm") or [None, None])[1],
                                 free_4s_cube_p95_mm=(s.get("free_4s_cube_mm") or [None, None])[1],
                                 lift_wrong=e.get("wrong"), lift_wrong_pct=e.get("wrong_pct"), lift_missed=e.get("missed_lift"),
                                 lift_false=e.get("false_lift"), recorded_lifts=f["ensemble"].get(g, {}).get("recorded_lifts")))
    fmt = lambda x, w: f"{x:>{w}}" if x is not None else f"{'-':>{w}}"
    print(f"{'group':18s} {'model':16s} {'n':>5s} {'1s p95 mm':>10s} {'4s p95 mm':>10s} {'lift wrong %':>12s} {'missed':>7s} {'false':>6s}")
    for r in rows:
        print(f"{r['group']:18s} {r['model']:16s} {r['episodes']:5d} {fmt(r['win1s_cube_p95_mm'], 10)} {fmt(r['free_4s_cube_p95_mm'], 10)} "
              f"{fmt(r['lift_wrong_pct'], 12)} {fmt(r['lift_missed'], 7)} {fmt(r['lift_false'], 6)}")
    pairs = []
    if len(files) == 2:
        (lb, fb), (la, fa) = files.items()
        for g in ref["groups"]:
            sel = None if g == "all" else set(ref.get("group_rows", {}).get(g, []))
            for name in fb["models"]:
                if name not in fa["models"]:
                    continue
                rb = next((r for r in rows if r["group"] == g and r["label"] == lb and r["name"] == name), None)
                ra = next((r for r in rows if r["group"] == g and r["label"] == la and r["name"] == name), None)
                if rb is None or ra is None:
                    continue
                wrong = lambda p: p["lifted_predicted"] is None or bool(p["lifted_predicted"]) != bool(p["lifted_recorded"])
                wb = {p["row"]: wrong(p) for p in fb["models"][name]["per_episode"] if sel is None or p["row"] in sel}
                wa = {p["row"]: wrong(p) for p in fa["models"][name]["per_episode"] if sel is None or p["row"] in sel}
                fixed = sum(wb[k] and not wa[k] for k in wb if k in wa)
                broken = sum(wa[k] and not wb[k] for k in wb if k in wa)
                checks = {}
                for k in ("win1s_cube_p95_mm", "free_4s_cube_p95_mm"):
                    if rb[k] is not None and ra[k] is not None:
                        checks[k] = ra[k] <= gate * rb[k]
                if rb["lift_wrong"] is not None and ra["lift_wrong"] is not None:
                    checks["lift_wrong"] = ra["lift_wrong"] <= max(gate * rb["lift_wrong"], rb["lift_wrong"] + 1)
                pairs.append(dict(group=g, model=name, lift_fixed=fixed, lift_broken=broken, sign_test_p=round(sign_test(fixed, broken), 6),
                                  gate=gate, gate_checks=checks, gate_pass=all(checks.values())))
        print(f"\n{'group':18s} {'model':6s} {'lift fixed':>10s} {'broken':>7s} {'sign p':>9s}  gate x{gate} (1 s p95, 4 s p95, lift wrong)")
        for p in pairs:
            c = p["gate_checks"]
            print(f"{p['group']:18s} {p['model']:6s} {p['lift_fixed']:10d} {p['lift_broken']:7d} {p['sign_test_p']:9.4g}  "
                  f"{'PASS' if p['gate_pass'] else 'FAIL'} {c}")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(files={k: str(v) for k, v in (i.split("=", 1) for i in items)}, rows=rows, paired=pairs), indent=1))
    return rows, pairs


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", nargs="+", metavar="LABEL=JSON", help="before / after table of output JSONs (no evaluation)")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model", nargs="+", metavar="NAME=PATH", help="checkpoint or run directory")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gate", type=float, default=1.10, help="--compare: forgetting gate factor (after <= gate x before)")
    parser.add_argument("--split", type=int, default=1, help="0 train or 1 validation (2 = sealed test: refused)")
    parser.add_argument("--families", nargs="+", help="default: every family with select: true")
    parser.add_argument("--batch", type=int, help="rollouts per GPU batch (default: the evaluation block's; memory only)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.compare:
        compare(args.compare, args.out, args.gate)
        return
    if args.data is None or not args.model or args.out is None:
        parser.error("--data, --model and --out are required (or --compare)")
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
               families=chosen, episodes=len(idx), models={}, groups={k: int(v.sum()) for k, v in groups.items()},
               group_rows={k: rows[v].tolist() for k, v in groups.items() if k != "all"})
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
                                                     free_4s_cube_mm=None if pick(e, "free", "target", "cube", "pos_m") is None
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
