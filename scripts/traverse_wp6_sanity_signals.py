#!/usr/bin/env python
"""Do the planner's sanity signals identify the picks that go wrong in Chrono? (plan §24 step 3)

For every driven pick the planner recorded: the two scorers' disagreement on energy, the other scorer's
acceptance, the imagined roll / pitch / wheel-load extremes, the imagined time. A pick "went wrong" when Chrono
did not complete it, stalled, made contact, or its Chrono cost exceeded the planner's own predicted cost by more
than ``--miss`` (default 25 %). Each signal is ranked and scored by AUC for that outcome, with the precision of
its top decile, so we measure which signals are worth acting on rather than assume it.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("batch", help="Chrono batch dir (rows.jsonl)")
ap.add_argument("--results", required=True, help="planner results.json")
ap.add_argument("--miss", type=float, default=0.25, help="Chrono cost over predicted by more than this = mis-prediction")
ap.add_argument("--picks", nargs="*", default=None, help="restrict to these pick names (default: all with predictions)")
args = ap.parse_args()
rows = defaultdict(dict)
for l in Path(args.batch, "rows.jsonl").read_text().splitlines():
    if not l.strip().startswith("{"):
        continue
    r = json.loads(l)
    for n in r["candidate"].split("+"):
        rows[n][r["key"]] = r
pred = {r["key"]: r for r in json.loads(Path(args.results).read_text())}
cost = lambda r: r["time_s"] + r["energy_kj"] / 10
X, y, tags = defaultdict(list), [], []
for name, rs in rows.items():
    if args.picks and name not in args.picks:
        continue
    for k, r in rs.items():
        p = pred.get(k, {})
        if p.get(f"{name}_cost") is None and p.get(f"{name}_time") is None:
            continue
        wrong = (not r["completed"]) or r.get("stalled", False) or r["contact"]
        if r["completed"] and p.get(f"{name}_cost") is not None:
            wrong = wrong or cost(r) > (1 + args.miss) * p[f"{name}_cost"]
        y.append(int(wrong)); tags.append((name, k))
        X["|disagree|"].append(abs(p.get(f"{name}_disagree", np.nan)))
        X["-disagree (wm below geo)"].append(-p.get(f"{name}_disagree", np.nan))
        X["other scorer rejects"].append(float(not p.get(f"{name}_ok_geo", True)) if name.startswith("wm") else float(not p.get(f"{name}_ok_wm", True)))
        X["pred max roll deg"].append(p.get(f"{name}_pred_max_roll_deg", np.nan))
        X["pred max pitch deg"].append(p.get(f"{name}_pred_max_pitch_deg", np.nan))
        X["-pred min tire load"].append(-p.get(f"{name}_pred_min_fz_n", np.nan))
        X["imagined time"].append(p.get(f"{name}_time_wm", p.get(f"{name}_time", np.nan)))
        X["-energy floor margin"].append(-(p.get(f"{name}_energy", np.nan) - p.get(f"{name}_energy_floor", np.nan)))
y = np.array(y)
print(f"{len(y)} driven picks, {y.sum()} went wrong ({100 * y.mean():.0f} %): not completed / stalled / contact / Chrono cost > {1 + args.miss:.2f} x predicted")
if y.sum() == 0 or y.sum() == len(y):
    print("no contrast between outcomes; nothing to rank"); raise SystemExit
def auc(score, label):
    m = np.isfinite(score); s, l = score[m], label[m]
    if l.sum() == 0 or l.sum() == len(l):
        return float("nan"), int(m.sum())
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # tie-aware average ranks
    for v in np.unique(s):
        idx = np.nonzero(s == v)[0]
        ranks[idx] = ranks[idx].mean()
    n1, n0 = l.sum(), (1 - l).sum()
    return float((ranks[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), int(m.sum())
print(f"{'signal (higher = more suspicious)':34s} {'AUC':>5s} {'n':>4s} {'top-10% precision':>17s} {'base rate':>9s}")
for name, v in X.items():
    v = np.array(v, float)
    a, n = auc(v, y)
    m = np.isfinite(v)
    k = max(1, int(0.1 * m.sum()))
    top = np.argsort(-v[m])[:k]
    prec = y[m][top].mean() if k else float("nan")
    print(f"{name:34s} {a:5.2f} {n:4d} {prec:17.2f} {y[m].mean():9.2f}")
wrong = [(t, ) for t, w in zip(tags, y) if w]
print("picks that went wrong:", ", ".join(f"{n}@{k[-16:]}" for (n, k), in wrong) if wrong else "none")
