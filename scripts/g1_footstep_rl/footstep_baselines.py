"""The H-FS baseline on footstep courses in CPU MuJoCo: HOMIE + a footstep-to-speed controller
(footsteps_mujoco.HfsController, decisions M1-M2 of its docstring; "spec N" refers to the internal design spec of the
study, not released). Scorer, scenes and statistics: nedm.g1_footstep_rl.footsteps_mujoco; eval directories in the
format of eval_footsteps_mujoco.py (its report, paired McNemar and video tools work on them).

Subcommands (outputs in --out, default artifacts/g1_footstep_rl/baselines):
  tune      H-FS grid g_v {0.8, 1.0, 1.2} x lookahead {1, 2} x k_y {0, 1, 2} (spec: 18) x v_min {0.2, 0.3} (M2) = 36
            settings on tuning seeds 0-15 of F1, F2, MN (1,728 episodes) -> tune.json, tune_rows.json, best.json. Pick:
            F2 hit@5 of all planned targets (unreached targets count as misses, so a setting that fails early is not
            rewarded) + MN clean completion; ties -> F1 hit@5 of all targets
  eval      H-FS with best.json (or --params) on dev seeds (new cells 1000-1063, old 100-163) -> <out>/HFS_best/
            (eval directory) + the kill test of spec 13 -> kill_test.json
  report    BASELINES.md from the outputs above

    cd <repository>; PYTHONPATH=src python -B scripts/g1_footstep_rl/footstep_baselines.py tune --workers 8
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_footsteps_mujoco as E  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "artifacts/g1_footstep_rl/baselines"
GRID = dict(g_v=(0.8, 1.0, 1.2), lookahead=(1, 2), k_y=(0.0, 1.0, 2.0), v_min=(0.2, 0.3))
TUNE_CELLS = ("F1", "F2", "MN")
EVAL_CELLS = ("F1", "F2", "F3", "MP", "MN", "ES", "L1", "L2")


def grid():
    keys = list(GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*(GRID[k] for k in keys))]


def _job(job):
    import torch
    torch.set_num_threads(1)
    from nedm.g1_footstep_rl import footsteps_mujoco as FM
    course, variant = FM.course_for(job["level"], job["seed"])
    if course is None:
        return dict(level=job["level"], seed=job["seed"], outcome="no_plan", setting=job["setting"]), []
    ctrl = FM.make_controller(job["spec"], None, course)
    row, ep, env = FM.run_course_episode(course, ctrl)
    row["setting"] = job["setting"]
    steps = [dict({k: v for k, v in r.items() if k != "place"}, level=row["level"], seed=row["seed"],
                  setting=job["setting"]) for r in (ep.progress.metrics()["rows"] if ep.progress else [])]
    return row, steps


def cmd_tune(a):
    import multiprocessing as mp
    from nedm.g1_footstep_rl import footsteps_mujoco as FM
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = grid()
    seeds = list(range(*__import__("nedm.g1_footstep_rl.footsteps_task", fromlist=["SEEDS"]).SEEDS["hfs_tune"]))
    jobs = [dict(spec=dict(kind="hfs", params=p), level=lv, seed=s, setting=i)
            for i, p in enumerate(settings) for lv in TUNE_CELLS for s in seeds]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(_job, jobs, chunksize=4)
    wall = time.time() - t0
    rows = [r for r, _ in res]
    steps = [s for _, ss in res for s in ss]
    table = []
    for i, p in enumerate(settings):
        d = dict(setting=i, params=p)
        for lv in TUNE_CELLS:
            c = FM.cell_summary([r for r in rows if r["setting"] == i and r["level"] == lv],
                                [s for s in steps if s["setting"] == i and s["level"] == lv], n_boot=200)
            d[lv] = dict(n=c["n"], completed=c["completed"]["rate"], clean_completion=c["clean_completion"]["rate"],
                         hit5=c["hit5"]["rate"], hit5_all=c["hit5_all"]["rate"], err_p50=c["err_p50"]["value"],
                         outcomes=c["outcomes"])
        d["score"] = d["F2"]["hit5_all"] + d["MN"]["clean_completion"]
        d["tie"] = d["F1"]["hit5_all"]
        table.append(d)
    best = max(table, key=lambda d: (d["score"], d["tie"]))
    E.jdump(out / "tune.json", E._clean(dict(wall_s=round(wall, 1), seeds=[seeds[0], seeds[-1]], cells=TUNE_CELLS,
                                               grid=GRID, host=__import__("socket").gethostname(), table=table,
                                               rule="max F2 hit5 of all targets + MN clean completion; tie F1")))
    E.jdump(out / "tune_rows.json", E._clean(rows))
    E.jdump(out / "best.json", E._clean(dict(params=best["params"], setting=best["setting"], score=best["score"])))
    print(f"[tune] {len(jobs)} episodes in {wall:.0f} s; best {best['params']} score {best['score']:.3f}")
    for d in sorted(table, key=lambda d: -d["score"])[:8]:
        print(f"  {d['params']}: F1 compl {d['F1']['completed']:.2f} hit5_all {d['F1']['hit5_all']:.2f} | F2 compl "
              f"{d['F2']['completed']:.2f} hit5 {d['F2']['hit5']:.2f} hit5_all {d['F2']['hit5_all']:.2f} | MN clean "
              f"{d['MN']['clean_completion']:.2f}")


def best_params(a):
    if getattr(a, "params", None):
        return json.loads(a.params)
    p = Path(a.out) / "best.json"
    if not p.exists():
        raise SystemExit(f"{p} missing: run tune first or give --params")
    return json.loads(p.read_text())["params"]


def cmd_eval(a):
    params = best_params(a)
    out = Path(a.out) / "HFS_best"
    spec = dict(kind="hfs", params=params)
    E.run_eval(spec, a.levels, ["dev"], out, a.workers, plots=a.plots, extra_meta=dict(note="H-FS best of the tune"))
    sm = E.report(out)
    kill(Path(a.out), sm)


def kill(out, sm):
    """Spec 13 kill test: H-FS F2 hit5 >= 80 % and completion >= 70 % -> the flat demo is not a headline; keep the
    edge stop (ES) as a demo only if H-FS ES completion < 50 %."""
    c = sm["cells"]
    res = {}
    if "F2" in c and c["F2"].get("n"):
        h, comp = c["F2"]["hit5"]["rate"], c["F2"]["completed"]["rate"]
        res["F2"] = dict(hit5=h, completion=comp, flat_demo_killed=bool(h >= 0.80 and comp >= 0.70))
    if "ES" in c and c["ES"].get("n"):
        comp = c["ES"]["completed"]["rate"]
        res["ES"] = dict(completion=comp, keep_edge_stop_demo=bool(comp < 0.50))
    E.jdump(out / "kill_test.json", E._clean(res))
    print("[kill test]", json.dumps(E._clean(res)))
    return res


def cmd_report(a):
    out = Path(a.out)
    L = ["# HOMIE baselines on footstep courses (MuJoCo)", ""]
    if (out / "tune.json").exists():
        t = json.loads((out / "tune.json").read_text())
        b = json.loads((out / "best.json").read_text())
        L += [f"H-FS tuning: {len(t['table'])} settings x {', '.join(t['cells'])} x seeds {t['seeds'][0]}-{t['seeds'][1]} "
              f"({t['wall_s']:.0f} s on {t['host']}). Rule: {t['rule']}. Best: `{json.dumps(b['params'])}`.", "",
              "| g_v | look | k_y | v_min | F1 completion | F1 hit@5 all | F2 completion | F2 hit@5 (step) | F2 hit@5 all | MN clean completion | Score |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        f = lambda v: "n/a" if v is None else f"{v:.2f}"
        for d in sorted(t["table"], key=lambda d: -d["score"]):
            p = d["params"]
            L.append(f"| {p['g_v']:g} | {p['lookahead']} | {p['k_y']:g} | {p['v_min']:g} | {f(d['F1']['completed'])} | "
                     f"{f(d['F1']['hit5_all'])} | {f(d['F2']['completed'])} | {f(d['F2']['hit5'])} | "
                     f"{f(d['F2']['hit5_all'])} | {f(d['MN']['clean_completion'])} | {d['score']:.2f} |")
    for name in ("HFS_best",):
        p = out / name / "SUMMARY.md"
        if p.exists():
            L += ["", f"## {name}", ""] + p.read_text().splitlines()[2:]
    if (out / "kill_test.json").exists():
        L += ["", "Kill test (spec 13): `" + (out / "kill_test.json").read_text().strip() + "`"]
    (out / "BASELINES.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["tune", "eval", "report"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--levels", nargs="+", default=None)
    ap.add_argument("--params", default=None, help="H-FS parameters as JSON (default: <out>/best.json)")
    ap.add_argument("--plots", type=int, default=0)
    a = ap.parse_args()
    if a.levels is None:
        a.levels = list(EVAL_CELLS)
    {"tune": cmd_tune, "eval": cmd_eval, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
