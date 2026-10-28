"""Release sim check, v12 (CHECK_SPEC.md section 3): per-episode and per-start metrics of check_eval.py runs.

    python -B check_analyze.py TAG=runs/x.npz[:kind] ... --out out/analysis_<name>.json
kind: H (hold starts, tasks/h68.npz; rows % 68 = start, rows // 68 = draw) or J (jit224; the cube side of each row
from the task metadata, Addendum B6: the file changes side in blocks of 8, 112 rows cube left L->R, 112 cube right
R->L). The J task metrics (grasp / lift / place / near place / carry stop) are those of sim_eps.py (this folder;
the plane = the run's own plane). Ported from the study's ~/so101_r2s_claude/v12/check/lib/check_analyze.py: file
paths from this repository and its download folder; --h68 names the H68 metadata (was <check folder>/tasks/h68.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "artifacts" / "so101_pick_place"
DT = 0.02
LOOSE, LOOSE6 = 10.0, 6.0
OPEN_CMD = 20.0
FAR_MM = 40.0
LIFT_MM = 25.0
WIN_S = 6.0
_K = {}


def kin():
    if "k" not in _K:
        import torch
        from nedm.so101_policy.kin import SO101Kin
        k = SO101Kin()
        _K["k"] = lambda q: tuple(x.numpy() for x in k.fk(torch.as_tensor(q)))
    return _K["k"]


def hold_start(jaw_deg, n=15):
    ok = (jaw_deg >= 6) & (jaw_deg <= 13)
    run = np.convolve(ok.astype(int), np.ones(n, int), "valid") >= n
    h = np.flatnonzero(run)
    return int(h[0]) if len(h) else None


def planned_jaw(chunk, k, n_dec, lo=0.0, hi=1.5):
    dstep = np.arange(n_dec) * 5 - 1
    m = (dstep >= k + lo / DT) & (dstep < k + hi / DT)
    if not m.any():
        return float("nan")
    return float(np.degrees(chunk[:n_dec][m, :5, 5].mean(1)).mean())


def ep_metrics(q, cube, ex, chunk, n, goal_xy):
    q, cube = q[:n].astype(np.float64), cube[:n].astype(np.float64)
    cmd = np.degrees(ex[:, :, 5].reshape(-1)[:n].astype(np.float64))
    jaw = np.degrees(q[:, 5])
    P, R = kin()(q)
    gz = P[:, 2] * 1e3
    k = hold_start(jaw)
    if k is None:
        return dict(hold=False, grasp_release=False, release_robot=False)
    up = np.flatnonzero(gz[k:] > gz[k] + LIFT_MM)
    kl = k + int(up[0]) if len(up) else n
    kw = min(kl, k + int(round(WIN_S / DT)), n)
    dgoal = np.linalg.norm(cube[:, :2] - np.asarray(goal_xy)[None], axis=1) * 1e3
    hit = np.flatnonzero((cmd[k:kw] > OPEN_CMD) & (dgoal[k:kw] > FAR_MM))
    op_r = np.flatnonzero(cmd[k:kl] > OPEN_CMD)
    nd = int(np.ceil(n / 5))
    out = dict(hold=True, k=int(k), t_hold=round(k * DT, 2), lift=bool(len(up)),
               t_lift=round(float(kl - k) * DT, 2) if len(up) else None,
               plan=planned_jaw(chunk, k, nd), plan_0_1=planned_jaw(chunk, k, nd, 0.0, 1.0),
               grasp_release=bool(len(hit)), t_grel=round(float(hit[0]) * DT, 2) if len(hit) else None,
               release_robot=bool(len(op_r)), t_rel_robot=round(float(op_r[0]) * DT, 2) if len(op_r) else None,
               cmd_max_win=float(cmd[k:kw].max()), jaw_max_win=float(jaw[k:kw].max()),
               goal_mm_hold=float(dgoal[k]))
    if len(hit):
        t = k + int(hit[0])
        j = min(t + 25, n - 1)
        out["dropped"] = bool(np.linalg.norm(P[j] - cube[j, :3]) * 1e3 > 25.0)
    s0, s1 = k + 5, min(k + 55, n)
    if s1 > s0:
        loc = np.einsum("nji,nj->ni", R[s0:s1], cube[s0:s1, :3] - P[s0:s1]) * 1e3      # R^T (c - p), mm
        out["seat_mm"] = np.round(loc.mean(0), 2).tolist()
        out["gp_above_cube_mm"] = round(float((P[s0:s1, 2] - cube[s0:s1, 2]).mean() * 1e3), 2)
        out["closed"] = bool(np.median(np.linalg.norm(P[s0:s1] - cube[s0:s1, :3], axis=1)) * 1e3 < 25.0)
    return out


def run_episodes(f):
    z = np.load(f, allow_pickle=True)
    sq, sc, ex, ch, L = z["state_q"], z["state_cube"], z["rec_exec"], z["rec_chunk"], z["steps"]
    goal = z["goal_xy"]
    rows = []
    for i in range(sq.shape[0]):
        n = int(min(L[i], sq.shape[1]))
        r = ep_metrics(sq[i], sc[i], ex[i], ch[i], n, goal[i])
        r.update(ep=i, end=int(z["end"][i]), steps=n, lenient=bool(z["lenient"][i]), seed=int(z["seeds"][i]))
        rows.append(r)
    info = json.loads(str(z["check"])) if "check" in z else {}
    return rows, info


def per_start(rows, N):
    out = []
    for s in range(N):
        rs = [r for r in rows if r["ep"] % N == s]
        pl = [r["plan"] for r in rs if r["hold"] and np.isfinite(r["plan"])]
        out.append(dict(row=s, draws=len(rs), holds=sum(r["hold"] for r in rs),
                        plan=float(np.median(pl)) if pl else None,
                        grasp_release=sum(r["grasp_release"] for r in rs),
                        release_robot=sum(r["release_robot"] for r in rs),
                        seat_mm=(np.round(np.median([r["seat_mm"] for r in rs if "seat_mm" in r], 0), 2).tolist()
                                 if any("seat_mm" in r for r in rs) else None),
                        gp_above_cube_mm=(float(np.median([r["gp_above_cube_mm"] for r in rs if "gp_above_cube_mm" in r]))
                                          if any("gp_above_cube_mm" in r for r in rs) else None)))
    return out


def counts(starts, rows, sel=None):
    st = [s for s in starts if sel is None or s["row"] in sel]
    rs = [r for r in rows if sel is None or r["row"] in sel]
    pl = np.array([s["plan"] for s in st if s["plan"] is not None])
    gr_rows = sorted(int(s["row"]) for s in st if s["grasp_release"] > 0)      # Addendum B3: distinct starts
    return dict(starts=len(st), starts_hold=int(len(pl)), loose=int((pl > LOOSE).sum()), loose6=int((pl > LOOSE6).sum()),
                starts_grasp_release=len(gr_rows), grasp_release_starts=gr_rows,
                plan_mean=round(float(pl.mean()), 3) if len(pl) else None,
                plan_p50=round(float(np.median(pl)), 3) if len(pl) else None,
                episodes=len(rs), ep_hold=sum(r["hold"] for r in rs),
                grasp_release=sum(r["grasp_release"] for r in rs), release_robot=sum(r["release_robot"] for r in rs),
                ep_loose=sum(bool(r["hold"] and r["plan"] > LOOSE) for r in rs))


J224_TASKS = str(DATA / "rl_inputs/check_tasks/jit224.npz")
J224_META = str(DATA / "rl_inputs/check_tasks/all280.json")   # = sim_eps.META


def j224_side_rows():
    """Addendum B6: the rows of jit224 with the cube left (direction L->R) and right (R->L), from the task metadata
    (row b -> the file's seed of row b -> its metadata entry). A re-seeded repeat keeps the file's row order."""
    meta = json.loads(Path(J224_META).read_text())
    by = {meta["seed0"] + m["row"]: m for m in meta["tasks"]}
    seeds = np.load(J224_TASKS)["seeds"].astype(int)
    d = [by[int(x)]["direction"] for x in seeds]
    left = {b for b, x in enumerate(d) if x == "L->R"}
    right = {b for b, x in enumerate(d) if x == "R->L"}
    if len(left) + len(right) != len(seeds):
        raise SystemExit("jit224: a row with no L->R / R->L direction")
    return left, right


def j_task(f, plane, info):
    """sim_eps.py task metrics per direction (rows 0-111 L->R, 112-223 R->L); seeds mapped back by row."""
    from nedm.so101_policy import task as _T  # noqa: F401  (this checkout's task module first)
    sys.path.insert(0, str(HERE))
    import sim_eps as SE
    meta = json.loads(SE.META.read_text())
    z = np.load(f, allow_pickle=True)
    seeds = z["seeds"].astype(int)
    rows_meta = [dict(m, seed=meta["seed0"] + m["row"]) for m in meta["tasks"]]
    by = {m["seed"]: m for m in rows_meta}
    t224 = np.load(J224_TASKS)["seeds"].astype(int)
    if not set(seeds) <= set(by):          # a re-seeded repeat: map row b -> the file's seed of row b
        if len(seeds) != len(t224):
            raise SystemExit(f"{f}: {len(seeds)} rows, not the 224 of jit224")
        rows_meta = [dict(by[int(t224[b])], seed=int(seeds[b])) for b in range(len(seeds))]
    eps = SE.episodes(f, plane, rows_meta)
    out = {}
    for d, kk in (("L->R", "lr"), ("R->L", "rl")):
        rs = [x for x in eps if x["direction"] == d]
        out[kk] = dict(n=len(rs), grasp=sum(x["grasp"] for x in rs), lift=sum(x["lift"] for x in rs),
                       place=sum(x["place"] for x in rs), near_place=sum(x["near_place"] for x in rs),
                       carry_stop=sum(x["carry_stop"] for x in rs), stall=sum(x["stall"] for x in rs),
                       far_stop=sum(any(s["to_goal_mm"] > 30.0 for s in x["stops"]) for x in rs))
    out["all"] = {k: out["lr"][k] + out["rl"][k] for k in out["lr"]}
    out["per_episode"] = [dict(ep=x["ep"], grasp=x["grasp"], place=x["place"], carry_stop=x["carry_stop"],
                               near_place=x["near_place"], end=x["end"]) for x in eps]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="TAG=file.npz:H|J")
    p.add_argument("--out", required=True)
    p.add_argument("--h68", default=str(DATA / "rl_inputs/check_tasks/h68.json"), help="H68 start metadata")
    a = p.parse_args()
    out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    h68 = json.loads(Path(a.h68).read_text())["holds"]
    sets = {"H30": set(range(30)), "H38": set(range(30, len(h68)))}
    for spec in a.runs:
        tag, rest = spec.split("=", 1)
        f, kind = rest.rsplit(":", 1)
        rows, info = run_episodes(f)
        r = dict(file=f, kind=kind, info=info)
        if kind == "H":
            N = len(h68)
            for x in rows:
                x["row"], x["draw"] = x["ep"] % N, x["ep"] // N
            st = per_start(rows, N)
            r["starts"] = st
            r["counts"] = {"H68": counts(st, rows), **{k: counts(st, rows, v) for k, v in sets.items()}}
            for side in ("L", "R"):
                sel = {m["row"] for m in h68 if m["side"] == side}
                r["counts"][f"H68_{side}"] = counts(st, rows, sel)
        else:
            N = len(rows)
            for x in rows:
                x["row"], x["draw"] = x["ep"], 0
            st = per_start(rows, N)
            if N != 224:
                raise SystemExit(f"{f}: {N} rows, not the 224 of jit224")
            jl, jr = j224_side_rows()                      # Addendum B6 (was rows 0-111 / 112-223: wrong)
            r["counts"] = {"J224": counts(st, rows), "J_L": counts(st, rows, jl), "J_R": counts(st, rows, jr)}
            plane = info.get("plane")
            r["task"] = j_task(f, tuple(plane) if plane else None, info)
        r["episodes"] = rows
        out[tag] = r
        c = r["counts"]
        main_key = "H68" if kind == "H" else "J224"
        cc = c[main_key]
        extra = (f" | task L->R g {r['task']['lr']['grasp']} p {r['task']['lr']['place']} stop {r['task']['lr']['carry_stop']}"
                 f" R->L g {r['task']['rl']['grasp']} p {r['task']['rl']['place']} stop {r['task']['rl']['carry_stop']}"
                 if kind == "J" else "")
        print(f"{tag:26s} {main_key}: holds {cc['starts_hold']}/{cc['starts']} starts, plan mean {cc['plan_mean']} p50 "
              f"{cc['plan_p50']}, loose {cc['loose']} (>6: {cc['loose6']}), grasp releases {cc['grasp_release']}/"
              f"{cc['episodes']} (robot rule {cc['release_robot']}){extra}", flush=True)
    Path(a.out).write_text(json.dumps(out, default=float))


if __name__ == "__main__":
    main()
