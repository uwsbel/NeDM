"""Final-pose score of Chrono policy checks (eval_so101_planar_chrono.py output dirs), per arm, goal class and start group,
with paired arms (spec SPEC_rt_v1.md sections 1 and 7; pre-registered before any new result).

python scripts/so101_push_rl/score_eval.py <eval dir> [<eval dir> ...] [--ids eval_ids_val.json] [--baseline A] \
    [--tols 10:3,5:3,15:3,25:3,10:5] [--out <prefix>]

Several dirs (e.g. the --shard parts of one check) are pooled; a (task, arm) pair may appear only once. Per task and arm
(arrays/task<k>.npz <arm>_states [H+1, 23] on the 20 ms grid, S[0] = the start frame j_a, S[H] = the horizon end; T xy =
cols 10:12, quaternion (w, x, y, z) 13:17, planar velocity 17:19, yaw rate 22; validity = not invalid_any of episodes.jsonl:
arm-table, non-finger link contact, T height / tilt, keep-out):
  position error |goal xy - xy(H)| (mm), yaw error |wrap(goal yaw - yaw(H))| (deg);
  settled: T planar speed < --v-tol (5 mm/s) and |yaw rate| < --w-tol-deg (5 deg/s) on the last --settle-steps (5)
  states S[H-4 .. H] (step-average rates over 20 ms each: the last 0.1 s);
  success at P mm / Y deg = valid and settled and pos < P and yaw < Y (the first --tols entry is primary: 10 mm / 3 deg);
  T moved |xy(H) - xy(0)| (mm), T turned yaw(H) - yaw(0) (deg, unwrapped); goal displacement / yaw change from S[0].
Rows with an error (no states) are counted as errors, not scored. The script checks its own final errors and yaw change
against the dir's episodes.jsonl (final_pos_mm, final_yaw_deg, yaw_change_deg) and warns on any difference.

Classes: --ids (class -> rows, class -> {group: rows}, optionally under a "classes" key; other keys are ignored; rows of
the run that are in no class go to "unlisted", listed rows that were not run are counted as "missing") or the task level
(LEVEL_CLASS; other levels -> class L<level>, group "all"). Groups: trained / untrained starts. Decide on the trained-start
rows; the untrained rows are the generalisation report; the test split is run once at the end.
Per arm x class x group: n, success at every tolerance (primary with a Wilson 95 % CI), invalid and settled share,
median / p90 of the position and yaw errors, medians of T moved / turned, horizons; by turning sign (goal yaw change
> +1 deg / < -1 deg / else 0) and per level. Baseline (--baseline, default the first policy of meta.json) vs every other
arm: exact two-sided McNemar test of the primary success on the tasks both arms ran without error.
Output: markdown on stdout; --out writes <out>.md and <out>.json (settings, dirs, arms, cells, paired, by_level,
consistency, per_task).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

LEVEL_CLASS = {   # level -> (class, start group), SPEC_rt_v1 sections 1 and 7
    30: ("ST", "trained"), 40: ("ST", "untrained"), 46: ("ST", "untrained"),
    31: ("LT1", "trained"), 41: ("LT1", "untrained"), 47: ("LT1", "untrained"),
    32: ("LT2", "trained"), 42: ("LT2", "untrained"),
    33: ("SR", "trained"), 43: ("SR", "untrained"),
    10: ("LR", "trained"), 11: ("LR", "trained"), 14: ("LR", "trained"), 15: ("LR", "trained"),
    20: ("LR", "untrained"), 21: ("LR", "untrained"), 45: ("LR", "untrained"),
    34: ("CB", "trained"), 44: ("CB", "untrained"),
}
CLASS_ORDER = ("ST", "LT1", "LT2", "SR", "LR", "CB")
GROUP_ORDER = ("trained", "untrained", "all")
INVALID_KEYS = ("arm_table", "other_links", "t_bad", "keepout")
Z95 = 1.959963984540054


def yaw_of(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def wilson(k, n, z=Z95):
    """Wilson score interval of k successes in n trials -> (lo, hi); (nan, nan) for n = 0."""
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p-value of b and c discordant pairs (binomial, p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def parse_tols(text):
    tols = [tuple(float(x) for x in t.split(":")) for t in text.split(",") if t.strip()]
    assert tols and all(len(t) == 2 for t in tols), f"--tols wants pos_mm:yaw_deg,... ({text})"
    return tols


def tol_key(t):
    return f"{t[0]:g}/{t[1]:g}"


def score_states(S, goal, settle_steps=5, v_tol=0.005, w_tol_deg=5.0):
    """Final-pose quantities of one episode: S [H+1, 23] (S[0] = start frame), goal (x, y, yaw)."""
    S, g = np.asarray(S, float), np.asarray(goal, float)
    H = len(S) - 1
    xy, yaw = S[:, 10:12], np.unwrap(yaw_of(S[:, 13:17]))
    tail = S[max(0, H - settle_steps + 1):]
    settled = bool((np.linalg.norm(tail[:, 17:19], axis=1) < v_tol).all() and (np.abs(tail[:, 22]) < math.radians(w_tol_deg)).all())
    return dict(H=H, pos_mm=1e3 * float(np.linalg.norm(g[:2] - xy[H])), yaw_deg=math.degrees(abs(float(wrap(g[2] - yaw[H])))),
                settled=settled, moved_mm=1e3 * float(np.linalg.norm(xy[H] - xy[0])), turned_deg=math.degrees(float(yaw[H] - yaw[0])),
                goal_disp_mm=1e3 * float(np.linalg.norm(g[:2] - xy[0])), goal_dyaw_deg=math.degrees(float(wrap(g[2] - yaw[0]))))


def load_ids(path):
    """eval_ids json -> {task: [(class, group or None), ...]}."""
    d = json.loads(Path(path).read_text())
    if isinstance(d.get("classes"), dict):
        d = d["classes"]
    rows = lambda v: isinstance(v, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in v)
    out = {}
    for c, v in d.items():
        if rows(v):
            for t in v:
                out.setdefault(int(t), []).append((c, None))
        elif isinstance(v, dict):
            for g, w in v.items():
                if rows(w):
                    for t in w:
                        out.setdefault(int(t), []).append((c, g))
    return out


def load_dirs(dirs):
    """-> rows {(task, arm): episodes.jsonl row + _dir}, metas [per dir], arms (meta order, then the rest sorted)."""
    rows, metas, arms = {}, [], []
    for d in map(Path, dirs):
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        metas.append(dict(dir=str(d), node=meta.get("node"), time=meta.get("time"), tasks=meta.get("tasks"), stage_env=meta.get("stage_env"),
                          policies=meta.get("policies"), baseline=meta.get("baseline"), items=meta.get("items")))
        for name in [p[0] for p in meta.get("policies") or []] + [f"planar_{b}" for b in meta.get("baseline") or []]:
            if name not in arms:
                arms.append(name)
        with open(d / "episodes.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (int(r["task"]), r["controller"])
                if key in rows:
                    raise SystemExit(f"task {key[0]} arm {key[1]} appears twice ({rows[key]['_dir']}, {d})")
                rows[key] = dict(r, _dir=str(d))
    arms += sorted({a for _, a in rows} - set(arms))
    return rows, metas, arms


def score_rows(rows, tols, settle_steps, v_tol, w_tol_deg):
    """-> per-task records {(task, arm): rec} and the consistency check against the EVL fields."""
    recs, cache, diff = {}, {}, dict(pos_mm=0.0, yaw_deg=0.0, turned_deg=0.0, n=0)
    for (task, arm), r in sorted(rows.items()):
        rec = dict(task=task, arm=arm, level=int(r["level"]), start=int(r["start"]), goal=list(map(float, r["goal"])))
        f = Path(r["_dir"]) / "arrays" / f"task{task:05d}.npz"
        if "error" not in r:
            if f not in cache:
                cache.clear()
                cache[f] = np.load(f) if f.exists() else {}
            z = cache[f]
            if f"{arm}_states" not in getattr(z, "files", z):
                rec["error"] = f"no {arm}_states in {f}"
        else:
            rec["error"] = str(r["error"])[:200]
        if "error" not in rec:
            rec.update(score_states(z[f"{arm}_states"], r["goal"], settle_steps, v_tol, w_tol_deg))
            rec["valid"] = not bool(r["invalid_any"])
            rec["invalid_reasons"] = [k for k in INVALID_KEYS if r.get(k)]
            rec["success"] = {tol_key(t): bool(rec["valid"] and rec["settled"] and rec["pos_mm"] < t[0] and rec["yaw_deg"] < t[1]) for t in tols}
            for k, ek in (("pos_mm", "final_pos_mm"), ("yaw_deg", "final_yaw_deg"), ("turned_deg", "yaw_change_deg")):
                if ek in r:
                    diff[k] = max(diff[k], abs(rec[k] - float(r[ek])))
            diff["n"] += 1
        recs[(task, arm)] = rec
    return recs, diff


def cells_of(task, level, ids):
    if ids is None:
        return [LEVEL_CLASS.get(level, (f"L{level}", "all"))]
    if task not in ids:
        return [("unlisted", LEVEL_CLASS.get(level, (None, "all"))[1])]
    return [(c, g if g is not None else LEVEL_CLASS.get(level, (None, "all"))[1]) for c, g in ids[task]]


def sign_of(dyaw_deg):
    return "+" if dyaw_deg > 1.0 else ("-" if dyaw_deg < -1.0 else "0")


def stats(recs, tols, missing=0):
    ok = [r for r in recs if "error" not in r]
    n = len(ok)
    q = lambda k, p: float(np.quantile([r[k] for r in ok], p)) if n else float("nan")
    k1 = sum(r["success"][tol_key(tols[0])] for r in ok)
    out = dict(n=n, errors=len(recs) - n, missing=missing,
               success={tol_key(t): int(sum(r["success"][tol_key(t)] for r in ok)) for t in tols},
               rate={tol_key(t): (sum(r["success"][tol_key(t)] for r in ok) / n if n else float("nan")) for t in tols},
               primary=tol_key(tols[0]), primary_ci95=list(wilson(k1, n)),
               invalid=float(np.mean([not r["valid"] for r in ok])) if n else float("nan"),
               invalid_reasons={k: int(sum(k in r["invalid_reasons"] for r in ok)) for k in INVALID_KEYS},
               settled=float(np.mean([r["settled"] for r in ok])) if n else float("nan"),
               pos_mm=dict(median=q("pos_mm", .5), p90=q("pos_mm", .9)), yaw_deg=dict(median=q("yaw_deg", .5), p90=q("yaw_deg", .9)),
               moved_mm_median=q("moved_mm", .5), turned_deg_median=q("turned_deg", .5),
               abs_turned_deg_median=float(np.median([abs(r["turned_deg"]) for r in ok])) if n else float("nan"),
               goal_disp_mm_median=q("goal_disp_mm", .5), abs_goal_dyaw_deg_median=float(np.median([abs(r["goal_dyaw_deg"]) for r in ok])) if n else float("nan"),
               horizons=sorted({r["H"] for r in ok}))
    return out


def summarise(recs, arms, ids, tols, baseline):
    """-> cells {class: {group: {arm: stats, by_sign}}}, paired {arm: {class: {group: ...}}}, by_level {level: {arm: stats}},
    listed tasks that no arm ran."""
    tasks = sorted({t for t, _ in recs})
    level = {t: recs[(t, a)]["level"] for t, a in recs}
    members = {}                                                   # (class, group) -> tasks
    for t in tasks:
        for cg in cells_of(t, level[t], ids):
            members.setdefault(cg, []).append(t)
    listed, not_run = {}, set()                                    # pre-registered tasks per cell; listed but never run
    for t, cl in (ids or {}).items():
        for c, g in cl:
            if g is None and t not in level:                       # flat ids format: the group comes from the level
                not_run.add(t)
                continue
            listed.setdefault((c, g if g is not None else LEVEL_CLASS.get(level[t], (None, "all"))[1]), set()).add(t)
            if t not in level:
                not_run.add(t)
    for cg in listed:
        members.setdefault(cg, [])
    order = lambda cg: (CLASS_ORDER.index(cg[0]) if cg[0] in CLASS_ORDER else len(CLASS_ORDER), cg[0],
                        GROUP_ORDER.index(cg[1]) if cg[1] in GROUP_ORDER else len(GROUP_ORDER), str(cg[1]))
    cells, paired = {}, {}
    for cg in sorted(members, key=order):
        c, g = cg
        ts = members[cg]
        for a in arms:
            R = [recs[(t, a)] for t in ts if (t, a) in recs]
            miss = len(listed.get(cg, set()) - {t for t in ts if (t, a) in recs})
            st = stats(R, tols, miss)
            st["by_sign"] = {s: stats([r for r in R if "error" not in r and sign_of(r["goal_dyaw_deg"]) == s], tols)
                             for s in ("+", "-", "0") if any("error" not in r and sign_of(r["goal_dyaw_deg"]) == s for r in R)}
            st["max_abs_goal_dyaw_deg"] = max([abs(r["goal_dyaw_deg"]) for r in R if "error" not in r], default=float("nan"))
            cells.setdefault(c, {}).setdefault(g, {})[a] = st
        if baseline is None:
            continue
        for a in arms:
            if a == baseline:
                continue
            both = [t for t in ts if "error" not in recs.get((t, baseline), {"error": 1}) and "error" not in recs.get((t, a), {"error": 1})]
            sb = [recs[(t, baseline)]["success"][tol_key(tols[0])] for t in both]
            sa = [recs[(t, a)]["success"][tol_key(tols[0])] for t in both]
            b = sum(x and not y for x, y in zip(sb, sa))
            cc = sum(y and not x for x, y in zip(sb, sa))
            paired.setdefault(a, {}).setdefault(c, {})[g] = dict(pairs=len(both), baseline=int(sum(sb)), arm=int(sum(sa)), baseline_only=int(b),
                                                                arm_only=int(cc), diff=(cc - b) / len(both) if both else float("nan"),
                                                                p_mcnemar=mcnemar_exact(int(b), int(cc)))
    by_level = {}
    for lv in sorted(set(level.values())):
        for a in arms:
            R = [recs[(t, a)] for t in tasks if level[t] == lv and (t, a) in recs]
            if R:
                by_level.setdefault(lv, {})[a] = stats(R, tols)
    return cells, paired, by_level, sorted(not_run)


def pct(x):
    return "-" if x != x else f"{100 * x:.0f} %"


def mm(x):
    return f"{x['median']:.1f} / {x['p90']:.1f}"


def markdown(res):
    tols, arms, base = res["settings"]["tols"], res["arms"], res["baseline"]
    keys = [tol_key(t) for t in tols]
    k0 = keys[0]
    nodes = ", ".join(sorted({str(m["node"]) for m in res["dirs"]}))
    stage = "; ".join(sorted({str(m["stage_env"]) for m in res["dirs"]}))
    L = [f"# Chrono final-pose score ({', '.join(m['dir'] for m in res['dirs'])})", "",
         f"Success = final pose at the horizon end, settled over the last {res['settings']['settle_steps']} steps (0.1 s), valid; "
         f"primary {k0} (mm / deg). Decide on the trained-start rows; untrained rows = generalisation.",
         f"Arms: {', '.join(arms)}; baseline {base}. Nodes: {nodes}. Stage env: {stage}."]
    if res["not_run"]:
        L.append(f"WARNING: {len(res['not_run'])} listed tasks were not run: {res['not_run'][:20]}")
    head = (f"| Class | Group | Arm | n | {k0} [95 % CI] | " + " | ".join(keys[1:]) + " | invalid | settled | pos med / p90 (mm) | "
            "yaw med / p90 (deg) | moved med (mm) | abs turned med (deg) | goal med (mm / abs deg) | H |")
    L += ["", head, "|" + "---|" * (head.count("|") - 1)]
    for c, groups in res["cells"].items():
        for g, per in groups.items():
            for a in arms:
                s = per[a]
                lo, hi = s["primary_ci95"]
                n = str(s["n"]) + (f" (+{s['errors']} err)" if s["errors"] else "") + (f" ({s['missing']} missing)" if s["missing"] else "")
                L.append(f"| {c} | {g} | {a} | {n} | {s['success'][k0]}/{s['n']} = {pct(s['rate'][k0])} [{pct(lo)}, {pct(hi)}] | "
                         + " | ".join(pct(s["rate"][k]) for k in keys[1:])
                         + f" | {pct(s['invalid'])} | {pct(s['settled'])} | {mm(s['pos_mm'])} | {mm(s['yaw_deg'])} | "
                         f"{s['moved_mm_median']:.1f} | {s['abs_turned_deg_median']:.1f} | {s['goal_disp_mm_median']:.1f} / "
                         f"{s['abs_goal_dyaw_deg_median']:.1f} | {','.join(map(str, s['horizons']))} |")
    for a, per_c in res["paired"].items():
        L += ["", f"Paired {base} vs {a} on {k0} (exact two-sided McNemar on the tasks both arms ran without error):", "",
              f"| Class | Group | pairs | {base} | {a} | {base} only | {a} only | diff | p |", "|---|---|---|---|---|---|---|---|---|"]
        for c, per_g in per_c.items():
            for g, p in per_g.items():
                diff = "-" if p["diff"] != p["diff"] else f"{100 * p['diff']:+.0f} pts"
                L.append(f"| {c} | {g} | {p['pairs']} | {p['baseline']} | {p['arm']} | {p['baseline_only']} | {p['arm_only']} | {diff} | "
                         f"{p['p_mcnemar']:.3g} |")
    rows = [(c, g, a, sg, st) for c, groups in res["cells"].items() for g, per in groups.items() for a in arms
            if per[a]["max_abs_goal_dyaw_deg"] > 7.5 for sg, st in per[a]["by_sign"].items()]
    if rows:
        L += ["", f"By turning sign of the goal (cells with goal yaw changes above 7.5 deg), {k0}:", "",
              "| Class | Group | Arm | sign | n | success | pos med (mm) | yaw med (deg) | turned med (deg) |", "|---|---|---|---|---|---|---|---|---|"]
        L += [f"| {c} | {g} | {a} | {sg} | {st['n']} | {st['success'][k0]} ({pct(st['rate'][k0])}) | {st['pos_mm']['median']:.1f} | "
              f"{st['yaw_deg']['median']:.1f} | {st['turned_deg_median']:.1f} |" for c, g, a, sg, st in rows]
    L += ["", f"By level, {k0}:", "", "| Level | Arm | n | success | invalid | pos med (mm) | yaw med (deg) |", "|---|---|---|---|---|---|---|"]
    for lv, per in res["by_level"].items():
        for a, st in per.items():
            n = str(st["n"]) + (f" (+{st['errors']} err)" if st["errors"] else "")
            L.append(f"| {lv} | {a} | {n} | {st['success'][k0]} ({pct(st['rate'][k0])}) | {pct(st['invalid'])} | "
                     f"{st['pos_mm']['median']:.1f} | {st['yaw_deg']['median']:.1f} |")
    cons = res["consistency"]
    L += ["", f"Check against episodes.jsonl ({cons['n']} episodes): max |diff| final pos {cons['pos_mm']:.2g} mm, final yaw "
              f"{cons['yaw_deg']:.2g} deg, yaw change {cons['turned_deg']:.2g} deg."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="eval_so101_planar_chrono.py output dirs (pooled)")
    ap.add_argument("--ids", default=None, help="eval_ids json (class -> rows or class -> {group: rows}); default: classes by level")
    ap.add_argument("--baseline", default=None, help="arm compared with every other arm (default: the first policy of meta.json)")
    ap.add_argument("--tols", default="10:3,5:3,15:3,25:3,10:5", help="pos_mm:yaw_deg list; the first is the primary rule")
    ap.add_argument("--settle-steps", type=int, default=5, help="20 ms states at the end that must be slow (5 = 0.1 s)")
    ap.add_argument("--v-tol", type=float, default=0.005, help="m/s, T planar speed")
    ap.add_argument("--w-tol-deg", type=float, default=5.0, help="deg/s, T yaw rate")
    ap.add_argument("--out", default=None, help="write <out>.md and <out>.json")
    a = ap.parse_args(argv)
    tols = parse_tols(a.tols)
    rows, metas, arms = load_dirs(a.dirs)
    if not rows:
        raise SystemExit("no episodes")
    base = a.baseline if a.baseline is not None else arms[0]
    if base not in arms:
        raise SystemExit(f"baseline {base} not among the arms {arms}")
    ids = load_ids(a.ids) if a.ids else None
    recs, diff = score_rows(rows, tols, a.settle_steps, a.v_tol, a.w_tol_deg)
    cells, paired, by_level, not_run = summarise(recs, arms, ids, tols, base if len(arms) > 1 else None)
    res = dict(settings=dict(tols=tols, primary=tol_key(tols[0]), settle_steps=a.settle_steps, v_tol_mps=a.v_tol, w_tol_deg=a.w_tol_deg,
                             ids=a.ids, rule="valid and settled and pos < P mm and |yaw| < Y deg at S[H]"),
               dirs=metas, arms=arms, baseline=base, cells=cells, paired=paired, by_level=by_level, not_run=not_run, consistency=diff,
               per_task=list(recs.values()))
    md = markdown(res)
    print(md)
    if max(diff["pos_mm"], diff["yaw_deg"], diff["turned_deg"]) > 1e-6:
        print(f"WARNING: final errors differ from episodes.jsonl: {diff}", file=sys.stderr)
    if len({m["node"] for m in metas}) > 1:
        node = {m["dir"]: m["node"] for m in metas}
        mixed = sorted({t for t, _ in rows if len({node[r["_dir"]] for (u, _), r in rows.items() if u == t}) > 1})
        print(f"WARNING: dirs from different nodes {sorted({str(m['node']) for m in metas})}: pooled rates mix nodes"
              + (f"; {len(mixed)} tasks have arms from different nodes, so their pairs mix nodes too (first: {mixed[:10]})"
                 if mixed else " (every task's arms come from one node)"), file=sys.stderr)
    if len({str(m["tasks"]) for m in metas}) > 1:
        print(f"WARNING: dirs evaluated different task tables {sorted({str(m['tasks']) for m in metas})}: task ids are rows of "
              "each table, so pooled classes may mix tables", file=sys.stderr)
    for m in metas:                                       # a dir whose job stopped early (items = planned task rows)
        got = len({t for (t, _), r in rows.items() if r["_dir"] == m["dir"]})
        if m["items"] is not None and got < int(m["items"]):
            print(f"WARNING: {m['dir']} has {got} of its {m['items']} planned task rows (truncated run?)", file=sys.stderr)
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_name(out.name + ".md").write_text(md)
        out.with_name(out.name + ".json").write_text(json.dumps(res, indent=1, default=str))
        print("wrote", out.with_name(out.name + ".md"), out.with_name(out.name + ".json"))
    return res


if __name__ == "__main__":
    main()
