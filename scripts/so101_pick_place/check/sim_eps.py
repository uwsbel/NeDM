"""Per-episode outcomes of the sim reproduction runs (eval_rec.py npz files) with the robot's carry-stop rule.

Per episode (sim world b; heights above the PHYSICS table: the calib plane in "real" runs, z = 0 in "nominal" runs):
  grasp      the cube held (jaw 3-15 deg and the cube centre within 25 mm of the grasp point) for >= 1 s, or lifted
  lift       the cube bottom >= 20 mm above the table at any step (analyze_grasp.py LIFT rule)
  place      sim lenient success (task rule)
  near_place robot-like place: lenient, or lifted and then the jaw opened (> 0.19 rad) with the cube centre <= 30 mm
             (xy) from the goal centre (a drop from a hover above the plate counts; the sim's drop rules
             release_high / unsupported end such episodes before the cube lands)
  goal_hover_s  time held with the grasp point within 30 mm (xy) of the goal centre
  carry stop the robot rule (robot_eps.py): >= 2 s of consecutive decisions with the cube held and the planned motion
             (max over the 5 arm joints of |last chunk step - q at the decision|) < 0.5 deg
  stall      >= 2 s of consecutive decisions with the cube held and the grasp point moving < 5 mm in the next 2 s
             (any plan; includes a plan that pushes into the floor or the table)
  end        the sim end code (task.ENDS)
usage: sim_eps.py OUT.json TAG=run.npz[:plane] ...   (plane = 'real' for the calib plane, else nominal)

Ported from the study's ~/so101_r2s_claude/v11/fullcheck/lib/sim_eps.py: code, kin points and task metadata come from
this repository and its download folder (the study read hw_lib.py of commit ddb08332d; its tcp() is the same code).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[3]                 # the repository root
sys.path.insert(0, str(WT / "src"))
sys.path.insert(0, str(WT / "scripts/so101_policy/real2sim/hw"))
import hw_lib as H  # noqa: E402
from nedm.so101_policy import task as T  # noqa: E402

KIN = str(WT / "artifacts/so101_pick_place/rl_inputs/geometry/kin_points_v1.npz")
PLANE = (-0.00468, -0.03641, 0.01836)
META = WT / "artifacts/so101_pick_place/rl_inputs/check_tasks/all280.json"   # task metadata of jit224 (direction per seed)
DT = 0.02
MOVE_DEG, STOP_S, STALL_MM = 0.5, 2.0, 5.0
HELD_JAW = (3.0, 15.0)
HELD_D = 0.025
LIFT, DOWN, OPEN, DROP_R = 0.02, 0.005, 0.19, 0.030
_L = None


def layer():
    global _L
    if _L is None:
        _L = H.ActionLayer(KIN)
    return _L


def runs(mask, min_n):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_n:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def episodes(npz, plane, meta_rows):
    z = np.load(npz, allow_pickle=True)
    q = z["state_q"].astype(np.float64)
    cube = z["state_cube"].astype(np.float64)
    steps = z["steps"].astype(int)
    length = z["length"].astype(int)
    goal = z["goal_xy"].astype(np.float64)
    le = z["lenient"].astype(bool)
    end = z["end"].astype(int)
    rq, rc = z["rec_q"].astype(np.float64), z["rec_chunk"].astype(np.float64)
    n, Tm = q.shape[:2]
    seeds = z["seeds"]
    by_seed = {m["seed"]: m for m in meta_rows}
    tcp = layer().tcp(q.reshape(-1, 6)).reshape(n, Tm, 3)
    per = 5 * DT
    nstop = int(round(STOP_S / per))
    out = []
    for b in range(n):
        k = max(1, min(int(steps[b]), Tm))
        c = cube[b, :k, :3]
        fl = np.zeros(k) if plane is None else plane[0] + plane[1] * c[:, 0] + plane[2] * c[:, 1]
        h = c[:, 2] - T.CUBE_HALF - fl
        jaw = np.degrees(q[b, :k, 5])
        dist = np.linalg.norm(tcp[b, :k] - c, axis=1)
        held = (jaw >= HELD_JAW[0]) & (jaw <= HELD_JAW[1]) & (dist < HELD_D)
        up = np.flatnonzero(h >= LIFT)
        k_lift = int(up[0]) if len(up) else None
        held_runs = runs(held, int(1.0 / DT))
        grasp = bool(held_runs) or k_lift is not None
        rel = None
        if k_lift is not None:
            gone = np.flatnonzero((np.arange(k) > k_lift) & ((np.radians(jaw) > OPEN) | (h < DOWN)))
            if len(gone):
                g = int(gone[0])
                rel = dict(t=round((g + 1) * DT, 2), dxy_mm=round(float(np.linalg.norm(c[g, :2] - goal[b])) * 1e3, 1),
                           h_mm=round(float(h[g]) * 1e3, 1))
        relj = None
        if k_lift is not None:
            op = np.flatnonzero((np.arange(k) > k_lift) & (np.radians(jaw) > OPEN))
            if len(op):
                g = int(op[0])
                relj = dict(t=round((g + 1) * DT, 2), dxy_mm=round(float(np.linalg.norm(c[g, :2] - goal[b])) * 1e3, 1),
                            h_mm=round(float(h[g]) * 1e3, 1))
        near_goal = np.linalg.norm(tcp[b, :k, :2] - goal[b], axis=1) < DROP_R
        goal_hover_s = round(float((held & near_goal).sum() * DT), 2)
        # decisions: d starts at step 5 d (state index 5 d - 1; d = 0: the start state ~ index 0)
        D = min(int(length[b]), rq.shape[1])
        si = np.clip(5 * np.arange(D) - 1, 0, k - 1)
        mot = np.degrees(np.abs(rc[b, :D, -1, :5] - rq[b, :D, :5])).max(1)
        hd = held[si]
        stops = []
        for i0, i1 in runs(hd & (mot < MOVE_DEG), nstop):
            ss = si[i0:i1]
            stops.append(dict(t0=round(float(si[i0] * DT), 2), dur_s=round((i1 - i0) * per, 2),
                              motion_p50_deg=round(float(np.median(mot[i0:i1])), 2),
                              tcp_z_mm=round(float(np.median(tcp[b, ss, 2])) * 1e3, 1),
                              cube_h_mm=round(float(np.median(h[ss])) * 1e3, 1),
                              to_goal_mm=round(float(np.median(np.linalg.norm(tcp[b, ss, :2] - goal[b], axis=1))) * 1e3, 1)))
        # stall: held and the grasp point moves < 5 mm over the next 2 s (100 steps)
        fut = np.clip(si + int(STOP_S / DT), 0, k - 1)
        moved = np.linalg.norm(tcp[b, fut] - tcp[b, si], axis=1) * 1e3
        ok_f = (si + int(STOP_S / DT)) <= k - 1
        stalls = []
        for i0, i1 in runs(hd & ok_f & (moved < STALL_MM), 1):
            ss = si[i0:i1]
            stalls.append(dict(t0=round(float(si[i0] * DT), 2), dur_s=round((i1 - i0) * per + STOP_S, 2),
                               motion_p50_deg=round(float(np.median(mot[i0:i1])), 2),
                               tcp_z_mm=round(float(np.median(tcp[b, ss, 2])) * 1e3, 1),
                               cube_h_mm=round(float(np.median(h[ss])) * 1e3, 1),
                               to_goal_mm=round(float(np.median(np.linalg.norm(tcp[b, ss, :2] - goal[b], axis=1))) * 1e3, 1)))
        m = by_seed[int(seeds[b])]
        out.append(dict(ep=b, seed=int(seeds[b]), set=m["set"], layout=m["layout"], jitter=m["jitter"],
                        direction=m["direction"], robot_episode=m["episode"], robot_student=m["student"],
                        grasp=grasp, lift=k_lift is not None, place=bool(le[b]),
                        near_place=bool(le[b] or (relj is not None and relj["dxy_mm"] <= DROP_R * 1e3)), release=rel,
                        release_jaw=relj, goal_hover_s=goal_hover_s,
                        carry_stop=bool(stops), stops=stops, stall=bool(stalls), stalls=stalls,
                        end=T.ENDS[int(end[b])], t_end=round(k * DT, 2), held_s=round(float(held.sum() * DT), 1),
                        jaw_min_deg=round(float(jaw.min()), 1)))
    return out


def summary(rows):
    def cnt(rs):
        return dict(n=len(rs), grasp=sum(r["grasp"] for r in rs), lift=sum(r["lift"] for r in rs),
                    place=sum(r["place"] for r in rs), near_place=sum(r["near_place"] for r in rs),
                    carry_stop=sum(r["carry_stop"] for r in rs), stall=sum(r["stall"] for r in rs))
    s = {}
    for st in sorted({r["set"] for r in rows}):
        rs = [r for r in rows if r["set"] == st]
        s[st] = {d: cnt([r for r in rs if r["direction"] == d]) for d in ("R->L", "L->R")}
        ends = {}
        for r in rs:
            ends[r["end"]] = ends.get(r["end"], 0) + 1
        s[st]["ends"] = ends
    return s


def main():
    meta = json.loads(META.read_text())
    rows_meta = [dict(m, seed=meta["seed0"] + m["row"]) for m in meta["tasks"]]
    res = {}
    for spec in sys.argv[2:]:
        tag, f = spec.split("=", 1)
        pl = None
        if f.endswith(":real"):
            f, pl = f[:-5], PLANE
        elif f.endswith(":realnoy"):
            f, pl = f[:-8], (PLANE[0], PLANE[1], 0.0)
        rows = episodes(f, pl, rows_meta)
        s = summary(rows)
        res[tag] = dict(file=f, plane=pl, summary=s, episodes=rows)
        for st, v in s.items():
            line = " | ".join(f"{d}: n {v[d]['n']} g {v[d]['grasp']} l {v[d]['lift']} p {v[d]['place']} "
                              f"np {v[d]['near_place']} stop {v[d]['carry_stop']} stall {v[d]['stall']}"
                              for d in ("R->L", "L->R"))
            print(f"{tag:28s} {st:9s} {line} | ends {v['ends']}")
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
