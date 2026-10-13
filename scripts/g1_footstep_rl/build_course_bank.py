"""Build the footstep course banks (spec section 4.7 of the internal design spec, not released; numpy, footsteps_task).

- Training bank: 25,000 courses per training level (F0-F3, P1-P3, ES); seed of course i of a level =
  10,000 + LEVEL_CODES index x 100,000 + i (footsteps_task.train_seed).
- Validation bank: 128 courses per validation level (the training levels + MP, MN), seeds 9,000-9,127.
- bank_report.json: per bank and level the distributions of N, L, dL, W, plank length, gap, footprints per plank,
  shares (stop, cliff, turns, lanes, half widths, L > 0.32 m), tries per course, the section 2.2 reference numbers
  for comparison, the build time and the file sizes.

Every course is drawn with footsteps_task.sample_canonical(level, level_rng(level, seed), seed) (course frame:
origin = start foot midpoint, heading 0) and checked with check_course (with its program) before it is stored; the
build stops if any course breaks a rule. The float32 round trip of the stored arrays is checked (largest error in
the report).

    PYTHONPATH=src python -B scripts/g1_footstep_rl/build_course_bank.py              # full banks (~1-2 min, 8 workers)
    PYTHONPATH=src python -B scripts/g1_footstep_rl/build_course_bank.py --max-per-level 500 --out /tmp/courses  # smoke

The released banks are g1_footstep_rl/rl_inputs/courses/ on the Hub; the default output folder
artifacts/g1_footstep_rl/rebuilt/courses keeps a rebuild apart from them.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True

from nedm.g1_footstep_rl import footsteps_task as FT  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "artifacts/g1_footstep_rl/rebuilt/courses"
TRAIN_NAME, VAL_NAME = "course_bank_v4_train.npz", "course_bank_v4_val.npz"
# spec section 2.2, check 2 (prototype generator, through-like endings, 4,000 courses per level)
SPEC_CHECK2 = {
    "P1": dict(plank_p5_50_95=[0.26, 0.50, 0.76], gap_p5_50_95=[0.032, 0.050, 0.067], L_p50_95=[0.27, 0.34],
               share_L_over_0p32=0.22, per_plank_1_2_3=[0.23, 0.55, 0.22]),
    "P2": dict(plank_p5_50_95=[0.25, 0.44, 0.73], gap_p5_50_95=[0.033, 0.064, 0.088], L_p50_95=[0.29, 0.34],
               share_L_over_0p32=0.23, per_plank_1_2_3=[0.36, 0.45, 0.19]),
    "P3": dict(plank_p5_50_95=[0.24, 0.26, 0.70], gap_p5_50_95=[0.054, 0.089, 0.118], L_p50_95=[0.30, 0.36],
               share_L_over_0p32=0.36, per_plank_1_2_3=[0.53, 0.34, 0.13]),
    "MP/MN": dict(plank_p5_50_95=[0.24, 0.41, 0.71], gap_p5_50_95=[0.035, 0.078, 0.117], L_p50_95=[0.28, 0.36],
                  share_L_over_0p32=0.27, per_plank_1_2_3=[0.44, 0.39, 0.18]),
}


def course_stats(c):
    """Per-course numbers for the report (program step lengths, geometric widths, planks)."""
    L = np.asarray(c.program["L"], np.float64)
    _, W = FT.step_vectors(c)
    walk = np.ones(c.n, bool)
    walk[0] = False
    if c.end == "stop":
        walk[-1] = False                                  # the closing step (L = 0)
    st = dict(n=c.n, stop=c.end == "stop", cliff=c.cliff, hw=c.half_width, tries=int(c.program["tries"]),
              L=L[walk].tolist(), dL=np.diff(L)[1:].tolist(), W=W[1:].tolist(), d0=float(L[0]),
              dpsi_max=float(np.abs(c.program.get("dpsi", np.zeros(1))).max()),
              lane_max=float(np.abs(c.program.get("lane", np.zeros(1))).max()),
              turn=bool(c.program.get("turn", False)), lanes=bool(c.program.get("lanes", False)))
    if c.is_planks:
        iv = c.intervals
        k = c.target_planks()
        cross = np.flatnonzero(np.diff(k) > 0) + 1
        st.update(plank=(iv[1:-1, 1] - iv[1:-1, 0]).tolist(), gap=(iv[1:, 0] - iv[:-1, 1]).tolist(),
                  per_plank=np.bincount(k, minlength=len(iv))[1:-1].tolist(), L_cross=L[cross].tolist(),
                  n_planks=len(iv) - 2, final_len=float(iv[-1, 1] - iv[-1, 0]))
    return st


def work(job):
    """job: (level, seeds) -> (bank arrays, per-course stats, largest float32 round-trip error)."""
    level, seeds = job
    courses, stats = [], []
    for seed in seeds:
        c = FT.sample_canonical(level, FT.level_rng(level, seed), seed)
        bad = FT.check_course(c)
        if bad:
            raise RuntimeError(f"{level} seed {seed}: {bad}")
        courses.append(c)
        stats.append(course_stats(c))
    arrays = FT.courses_to_arrays(courses)
    err = 0.0
    for i, c in enumerate(courses):
        b = FT.course_from_arrays(arrays, i)
        t, ce = c.local()
        err = max(err, float(np.abs(b.targets - t).max()), float(np.abs(b.centre - ce).max()))
        if c.is_planks:
            err = max(err, float(np.abs(b.intervals - c.intervals).max()))
    return arrays, stats, err


def pct(x, q=(1, 5, 50, 95, 99)):
    x = np.asarray(x, np.float64)
    if not len(x):
        return None
    return dict(zip([f"p{v}" for v in q], np.round(np.percentile(x, q), 4).tolist()),
                min=round(float(x.min()), 4), max=round(float(x.max()), 4), mean=round(float(x.mean()), 4))


def summarise(stats):
    cat = lambda k: [v for s in stats for v in s.get(k, [])]
    one = lambda k: [s[k] for s in stats if k in s]
    L = np.asarray(cat("L"))
    out = dict(courses=len(stats), N=pct(one("n")), L=pct(L), dL=pct(cat("dL")), W=pct(cat("W")), d0=pct(one("d0")),
               share_L_over_0p32=round(float((L > 0.32).mean()), 4), share_stop=round(float(np.mean(one("stop"))), 4),
               share_cliff=round(float(np.mean(one("cliff"))), 4), tries=pct(one("tries")),
               share_turn=round(float(np.mean(one("turn"))), 4), share_lanes=round(float(np.mean(one("lanes"))), 4),
               dpsi_abs_max=round(float(np.max(one("dpsi_max"))), 4), lane_abs_max=round(float(np.max(one("lane_max"))), 4))
    hw = np.asarray(one("hw"))
    out["half_width_shares"] = {f"{v:g}": round(float((np.abs(hw - v) < 1e-6).mean()), 4) for v in np.unique(hw)}
    if any("plank" in s for s in stats):
        per = np.asarray(cat("per_plank"))
        out.update(plank_len=pct(cat("plank")), gap=pct(cat("gap")), L_crossing=pct(cat("L_cross")),
                   n_planks=pct(one("n_planks")), final_platform_len=pct(one("final_len")),
                   footprints_per_plank_share={str(k): round(float((per == k).mean()), 4) for k in (1, 2, 3)})
    return out


def build(levels, seeds_of, workers, chunk=500):
    jobs = []
    for lv in levels:
        seeds = seeds_of(lv)
        jobs += [(lv, seeds[i:i + chunk]) for i in range(0, len(seeds), chunk)]
    if workers <= 1:
        res = [work(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(work, jobs, chunksize=1)
    arrays = {k: np.concatenate([r[0][k] for r in res]) for k in FT.BANK_KEYS}
    stats = {lv: [] for lv in levels}
    for (lv, _), r in zip(jobs, res):
        stats[lv] += r[1]
    return arrays, {lv: summarise(s) for lv, s in stats.items()}, max(r[2] for r in res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--per-level", type=int, default=FT.SEEDS["train_per_level"])
    ap.add_argument("--max-per-level", type=int, default=0, help="cap the training courses per level (smoke tests)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per = min(a.per_level, a.max_per_level) if a.max_per_level else a.per_level
    t0 = time.time()
    train, train_rep, train_err = build(FT.TRAIN_LEVELS, lambda lv: [FT.train_seed(lv, i) for i in range(per)],
                                        a.workers)
    t1 = time.time()
    val, val_rep, val_err = build(FT.VAL_LEVELS, lambda lv: list(range(*FT.SEEDS["val"])), a.workers, chunk=128)
    t2 = time.time()
    meta = dict(version="v4", spec="g1_footstep_rl/README.md",
                frame="course frame: origin = start foot midpoint, x = start pelvis heading; float32",
                level_codes=list(FT.LEVEL_CODES), end_codes=FT.END_CODES, nmax=FT.NMAX, pmax=FT.PMAX,
                pad="targets/centre rows past n repeat the last row; interval rows past n_intervals = (1e9, -1e9)",
                generator="footsteps_task.sample_canonical(level, level_rng(level, seed), seed)", host=socket.gethostname())
    paths = dict(train=out / TRAIN_NAME, val=out / VAL_NAME)
    FT.save_bank(paths["train"], train, dict(meta, bank="train", seeds="10,000 + level index x 100,000 + i",
                                             per_level=per, levels=list(FT.TRAIN_LEVELS)))
    FT.save_bank(paths["val"], val, dict(meta, bank="val", seeds="9,000-9,127 per level", per_level=128,
                                         levels=list(FT.VAL_LEVELS)))
    t3 = time.time()
    tl = time.time()
    back = FT.load_bank(paths["train"])
    load_s = time.time() - tl
    assert len(back["n"]) == len(train["n"]) and np.array_equal(back["seed"], train["seed"])
    rep = dict(spec_check2_reference=SPEC_CHECK2, train=train_rep, val=val_rep,
               files={k: dict(path=str(p), mb=round(p.stat().st_size / 2 ** 20, 2), courses=int(len(b["n"])))
                      for (k, p), b in zip(paths.items(), (train, val))},
               float32_round_trip_max_err_m=max(train_err, val_err),
               seconds=dict(train=round(t1 - t0, 1), val=round(t2 - t1, 1), save=round(t3 - t2, 1),
                            load_train=round(load_s, 1)),
               workers=a.workers, host=socket.gethostname(), argv=sys.argv,
               note="L = program step lengths of walking steps (target 1 on, without the closing step); dL from "
                    "step 2 on incl. the closing step; W = geometric width of target 1 on; plank_len = interior "
                    "planks; footprints_per_plank = interior planks.")
    (out / "bank_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(dict(files=rep["files"], seconds=rep["seconds"], err=rep["float32_round_trip_max_err_m"]),
                     indent=1))
    for lv, r in train_rep.items():
        extra = ""
        if "plank_len" in r:
            extra = (f" plank p5/50/95 {r['plank_len']['p5']}/{r['plank_len']['p50']}/{r['plank_len']['p95']}"
                     f" gap {r['gap']['p5']}/{r['gap']['p50']}/{r['gap']['p95']}"
                     f" per-plank {r['footprints_per_plank_share']}")
        print(f"{lv}: N {r['N']['min']}-{r['N']['max']} L p50/p95 {r['L']['p50']}/{r['L']['p95']} "
              f">0.32 {r['share_L_over_0p32']} stop {r['share_stop']} tries max {r['tries']['max']}{extra}")


if __name__ == "__main__":
    main()
