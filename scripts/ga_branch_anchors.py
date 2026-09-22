#!/usr/bin/env python
"""Moving-prefix branch anchors for PLAN A4 (stage ``select`` only).

Chooses 800 anchors PER WORLD from the recorded f104 episodes (CRM ``crm_f104_v1/collect_v1/runs``; rigid
``fdm_f104_50h_20260909/production_v3/runs`` + ``production_v4/runs``), the same (episode, frame) in both worlds where
possible, and writes everything a later continuation-sampling stage needs (that stage is NOT here: it waits for the
pending gc_control fix).

Anchor classes (an anchor = a recorded episode cut at frame F, 50 ms frames):
  clean_moving  60 %  F in {40, 80, 120} (2/4/6 s); at F the vehicle moves (vx > 1 m/s), is < 1 m from the route, is not
                      parked, has >= 12 m of route left, and the prefix is clean (no stall onset and no rollback before F).
  low_progress  40 %  the episode has a stall onset = first run of 20 consecutive frames with |vx| < 0.3 and throttle > 0.3
                      searched from frame 20 (the night-2 labeller's rule); F in [onset-20, onset+10], F = onset-10 when
                      possible; CRM anchors are pre-screened so the mean spindle-height drop below the BMP ground between
                      F-20 and F is < 0.1 m (crm_extra.npz: spindle_z_m - bmp_ground_z_m, mean over the 4 wheels); rigid
                      keeps the same F rule without the screen. Not parked and >= 5 m of route left (so a continuation exists).
Splits come from the cache manifest (the twin group split): 700 anchors from train groups, 100 from held-out groups
(50 val + 50 test, each kept with its group's split). Stratified over groups (at most one anchor per episode and world,
at most 2 per group and world, groups without an anchor are served first), deterministic seed. Paired candidates
(episode feasible for the class in BOTH worlds) are used first; per-world fill from single-world candidates only when
the paired pool runs out (reported). Planner-suite ids/groups are asserted absent.

Outputs in --out:
  anchors_<world>.json   list of anchor records (episode, world, group, split, class, F, onset, recorded pose/vx at F,
                         goal, remaining distance, route file + sha256, run dir, sinkage numbers, pairing flags, ...)
  hist_<world>.npz       hist (n,40,15) f32 = [state[F-39+t][cols 0-6,11-15], action[F-40+t]], hmask (n,40) bool
                         (True where the action frame F-40+t >= 0: F < 40 is partially masked), anchor_id (n,), F (n,)
  scan_<world>.json      per-episode scan (reused on re-runs unless --rescan)
  summary.json           the counts table, F distribution, survivor counts, onset-to-termination margins, pairing.
CPU only, process pool <= 6.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import f104_n2_dataset as DS  # noqa: E402  (first_run, project: the night-2 labeller's definitions, numpy only)

K = REPO / "artifacts/traverse/generalist_20260921"
DEFAULT_MANIFEST = K / "B_tracker/cache_v1/cache_manifest.json"
DEFAULT_OUT = K / "A_adapt/a4/anchors"
CRM_ROOT = REPO / "artifacts/traverse/crm_f104_v1/collect_v1/runs"
RIGID_ROOTS = [REPO / "artifacts/traverse/fdm_f104_50h_20260909/production_v3/runs",
               REPO / "artifacts/traverse/fdm_f104_50h_20260909/production_v4/runs"]
ROUTES_DESIGNED = REPO / "artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases/routes"
ROUTES_ONPOLICY = REPO / "artifacts/traverse/fdm_f104_50h_20260909/cases_night2_onpolicy/routes"
BLACKLIST = re.compile(r"^(f104_crm_eval_group_|f104_g1_test_group_|f104_pair_group_)")
TRAIN_GROUP = re.compile(r"^f104_v2_group_\d+$")

WORLDS = ("crm", "rigid")
HIST_STATE_COLS = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]   # deployable columns (PLAN conventions)
HIST_T = 40
DT, S0 = 0.05, 20                       # frame period; labeller search start (frame 20 = 1 s)
CM_FRAMES = (40, 80, 120)               # clean-moving cuts (2 / 4 / 6 s)
CM_VX, CM_DEV, CM_REM = 1.0, 1.0, 12.0
STALL_VX, STALL_THR, STALL_RUN = 0.3, 0.3, 20
LP_LO, LP_HI, LP_PREF = -20, 10, -10    # low-progress window and preferred offset relative to the onset
SINK_WIN, SINK_MAX = 20, 0.1            # CRM screen: drop of the mean spindle height over the last 20 frames < 0.1 m
LP_MIN_REM = 5.0
CLASSES = ("low_progress", "clean_moving")   # scarcer class is served first


# ----------------------------------------------------------------------------------------------------- helpers
def run_dir(world: str, episode: str) -> Path | None:
    roots = [CRM_ROOT] if world == "crm" else RIGID_ROOTS
    for r in roots:
        if (r / episode).is_dir():
            return r / episode
    return None


def route_file(episode: str, group: str) -> tuple[Path, str]:
    """The route json the recorded episode drove (verified against command_reference.npz in the extraction pass)."""
    kind = episode[len(group) + 1:]                 # 'route_03' or 'op_05'
    if kind.startswith("route_"):
        return ROUTES_DESIGNED / group / f"{kind}.json", "designed"
    if kind.startswith("op_"):
        return ROUTES_ONPOLICY / group / f"{kind}.json", "on_policy"
    raise ValueError(f"unknown route kind in episode id {episode!r}")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def pct(v, qs=(10, 50, 90)):
    v = np.asarray(v, float)
    if v.size == 0:
        return {f"p{q}": None for q in qs} | {"n": 0}
    return {f"p{q}": round(float(np.percentile(v, q)), 3) for q in qs} | {"n": int(v.size), "mean": round(float(v.mean()), 3)}


# ----------------------------------------------------------------------------------------------------- pass 1: scan
def scan_one(job: dict) -> dict:
    """Per-episode feasibility of every candidate cut (both classes). Never raises: returns {'error': ...}."""
    world, episode = job["world"], job["episode"]
    src = run_dir(world, episode)
    out = {"episode": episode, "world": world}
    if src is None:
        return out | {"error": "run dir missing"}
    try:
        with np.load(src / "trajectory.npz", allow_pickle=False) as z:
            state = np.asarray(z["state"], np.float32); act = np.asarray(z["action"], np.float32)
            pose = np.asarray(z["pose"], np.float64); parked = np.asarray(z["parked"], bool)
            assert state.shape[1] == 17 and act.shape[1] == 3 and abs(float(z["dt_s"]) - DT) < 1e-9
        with np.load(src / "command_reference.npz", allow_pickle=False) as r:
            wp = np.asarray(r["reference_waypoints"], np.float64)
        n = len(state)
        vx = state[:, 0].astype(np.float64); thr = act[:, 1].astype(np.float64)
        s, dev, L = DS.project(pose[:, :2], wp)
        smax = np.maximum.accumulate(s)
        onset = DS.first_run((np.abs(vx) < STALL_VX) & (thr > STALL_THR), STALL_RUN, S0)
        rb = DS.first_run(((vx < -0.10) & (thr > STALL_THR)) | (vx < -0.30), 1, S0)
        out.update(n=int(n), route_len_m=float(L), onset=None if onset is None else int(onset),
                   rollback=None if rb is None else int(rb))
        cm = []
        for F in CM_FRAMES:
            if F >= n - 1:
                continue
            ok = (vx[F] > CM_VX and dev[F] < CM_DEV and (L - smax[F]) >= CM_REM and not parked[F]
                  and (onset is None or onset > F) and (rb is None or rb > F))
            if ok:
                cm.append(int(F))
        out["cm"] = cm
        out["lp"] = []; out["lp_window"] = []
        if onset is not None:
            out["margin_frames"] = int(n - 1 - onset)
            h = None
            if world == "crm":
                with np.load(src / "crm_extra.npz", allow_pickle=False) as e:
                    h = (np.asarray(e["spindle_z_m"], np.float64) - np.asarray(e["bmp_ground_z_m"], np.float64)[:, None]).mean(1)
                assert len(h) == n
            for F in range(max(1, onset + LP_LO), min(onset + LP_HI, n - 2) + 1):
                drop = None if h is None else float(h[max(0, F - SINK_WIN)] - h[F])
                ok = (not parked[F]) and (L - smax[F]) >= LP_MIN_REM and (drop is None or drop < SINK_MAX)
                out["lp_window"].append([int(F), bool(ok), None if drop is None else round(drop, 4)])
                if ok:
                    out["lp"].append(int(F))
        return out
    except Exception as exc:  # noqa: BLE001
        return out | {"error": f"{type(exc).__name__}: {exc}"}


def lp_preferred(row: dict, feasible: list[int] | None = None) -> int | None:
    """onset-10 when feasible, else the feasible F nearest to onset-10 (earlier one on ties)."""
    fs = row["lp"] if feasible is None else feasible
    if not fs:
        return None
    pref = row["onset"] + LP_PREF
    return min(fs, key=lambda F: (abs(F - pref), F))


# ----------------------------------------------------------------------------------------------------- selection
def select(scan: dict, man: dict, quotas: dict, rng: np.random.Generator, lp_same_frame: bool = True) -> tuple[dict, dict]:
    """Joint selection over both worlds. Returns (picks per world, bookkeeping).

    lp_same_frame: for a low-progress pair whose feasible windows overlap, use one common F (the one closest to both
    worlds' onset-10; this can move a cut up to 10 frames past the onset); False = every world cuts at its own onset-10."""
    split_of = {e: man["split_of"][f"{e}@crm"] for e in scan["crm"]}
    group_of = {e: man["group_of"][f"{e}@crm"] for e in scan["crm"]}
    for e in scan["rigid"]:
        split_of.setdefault(e, man["split_of"][f"{e}@rigid"]); group_of.setdefault(e, man["group_of"][f"{e}@rigid"])
    used = {w: set() for w in WORLDS}; gcount = {w: Counter() for w in WORLDS}
    picks = {w: [] for w in WORLDS}; fcount = {w: Counter() for w in WORLDS}
    book = defaultdict(dict)

    def feasible(world, ep, cls):
        row = scan[world].get(ep)
        if row is None or "error" in row:
            return []
        return row["cm"] if cls == "clean_moving" else row["lp"]

    def choose_cm_F(common, worlds):
        # balance the F distribution: lowest running count over the worlds that receive the anchor, random tie-break
        counts = [sum(fcount[w][F] for w in worlds) for F in common]
        best = [F for F, c in zip(common, counts) if c == min(counts)]
        return int(best[rng.integers(len(best))])

    def take(world, ep, cls, F, paired, same_frame):
        picks[world].append(dict(episode=ep, world=world, group=group_of[ep], split=split_of[ep], cls=cls, F=int(F),
                                 paired=bool(paired), same_frame=bool(same_frame)))
        used[world].add(ep); gcount[world][group_of[ep]] += 1
        if cls == "clean_moving":
            fcount[world][int(F)] += 1

    def round_robin(cands: dict, quota: int, worlds: tuple, place) -> int:
        """cands: group -> episodes (already shuffled). Serve groups with the fewest anchors first; cap 2 per group/world."""
        taken = 0
        groups = sorted(cands); rng.shuffle(groups)
        for level in (0, 1):
            progress = True
            while taken < quota and progress:
                progress = False
                for g in groups:
                    if taken >= quota:
                        break
                    if max(gcount[w][g] for w in worlds) != level:
                        continue
                    while cands[g]:
                        ep = cands[g].pop()
                        if any(ep in used[w] for w in worlds):
                            continue
                        place(ep); taken += 1; progress = True
                        break
        return taken

    for split in ("train", "val", "test"):
        eps_split = sorted(e for e in set(scan["crm"]) | set(scan["rigid"]) if split_of[e] == split)
        for cls in CLASSES:
            quota = quotas[split][cls]
            # ---- paired candidates: feasible for this class in both worlds (clean-moving: a common F)
            cands = defaultdict(list)
            for ep in eps_split:
                fc, fr = feasible("crm", ep, cls), feasible("rigid", ep, cls)
                if not fc or not fr:
                    continue
                if cls == "clean_moving" and not set(fc) & set(fr):
                    continue
                cands[group_of[ep]].append(ep)
            for g in cands:
                rng.shuffle(cands[g])
            n_pair_cands = sum(len(v) for v in cands.values())

            def place_pair(ep, cls=cls):
                rc, rr = scan["crm"][ep], scan["rigid"][ep]
                if cls == "clean_moving":
                    F = choose_cm_F(sorted(set(rc["cm"]) & set(rr["cm"])), WORLDS)
                    take("crm", ep, cls, F, True, True); take("rigid", ep, cls, F, True, True)
                else:
                    common = sorted(set(rc["lp"]) & set(rr["lp"])) if lp_same_frame else []
                    if common:
                        pc, pr = rc["onset"] + LP_PREF, rr["onset"] + LP_PREF
                        F = min(common, key=lambda F: (abs(F - pc) + abs(F - pr), F))
                        take("crm", ep, cls, F, True, True); take("rigid", ep, cls, F, True, True)
                    else:
                        take("crm", ep, cls, lp_preferred(rc), True, False)
                        take("rigid", ep, cls, lp_preferred(rr), True, False)

            n_paired = round_robin(cands, quota, WORLDS, place_pair)
            book[split][cls] = dict(quota=quota, paired_candidates=n_pair_cands, paired_taken=n_paired, fill={})
            # ---- per-world fill from single-world candidates
            for w in WORLDS:
                need = quota - n_paired
                if need <= 0:
                    book[split][cls]["fill"][w] = dict(candidates=0, taken=0)
                    continue
                cands_w = defaultdict(list)
                for ep in eps_split:
                    if ep in used[w] or not feasible(w, ep, cls):
                        continue
                    cands_w[group_of[ep]].append(ep)
                for g in cands_w:
                    rng.shuffle(cands_w[g])
                n_c = sum(len(v) for v in cands_w.values())

                def place_one(ep, w=w, cls=cls):
                    row = scan[w][ep]
                    F = choose_cm_F(row["cm"], (w,)) if cls == "clean_moving" else lp_preferred(row)
                    take(w, ep, cls, F, False, False)

                n_f = round_robin(cands_w, need, (w,), place_one)
                book[split][cls]["fill"][w] = dict(candidates=n_c, taken=n_f)
    return picks, dict(book)


# ----------------------------------------------------------------------------------------------------- pass 2: extract
def extract_one(job: dict) -> dict:
    world, ep, F, group = job["world"], job["episode"], int(job["F"]), job["group"]
    src = run_dir(world, ep)
    with np.load(src / "trajectory.npz", allow_pickle=False) as z:
        state = np.asarray(z["state"], np.float32); act = np.asarray(z["action"], np.float32)
        pose = np.asarray(z["pose"], np.float64); parked = np.asarray(z["parked"], bool)
    with np.load(src / "command_reference.npz", allow_pickle=False) as r:
        wp = np.asarray(r["reference_waypoints"], np.float64); sp = np.asarray(r["reference_speeds"], np.float64)
        desired = np.asarray(r["desired_speed_mps"], np.float64)
    case = json.loads((src / "case.json").read_text()); outc = json.loads((src / "outcome.json").read_text())
    n = len(state)
    assert 0 < F < n - 1, (ep, world, F, n)
    s, dev, L = DS.project(pose[:, :2], wp); smax = np.maximum.accumulate(s)
    # history window (ga_build_mixed convention): hist[t] = [state[F-39+t][cols], action[F-40+t]], masked where the action frame < 0
    t = np.arange(HIST_T); si = F - (HIST_T - 1) + t; ai = F - HIST_T + t; ok = ai >= 0
    hist = np.zeros((HIST_T, 15), np.float32)
    hist[ok] = np.concatenate([state[si[ok]][:, HIST_STATE_COLS], act[ai[ok]]], 1)
    # route file that was driven (verified) + sha
    rf, kind = route_file(ep, group)
    match = False; sha = None
    if rf.exists():
        rj = json.load(open(rf))
        match = (np.abs(np.asarray(rj["waypoints"], float) - wp).max() < 1e-9
                 and np.abs(np.asarray(rj["speeds"], float) - sp).max() < 1e-9)
        sha = sha256_file(rf)
    rec_sha = outc.get("route_sha256")
    vx = float(state[F, 0]); thr = float(act[F, 1])
    slow = (np.abs(state[:, 0]) < STALL_VX) & (act[:, 1] > STALL_THR)
    stalled_F = bool(F - STALL_RUN + 1 >= 0 and slow[F - STALL_RUN + 1:F + 1].all())   # F ends a 20-frame stall run
    rec = dict(anchor_id=f"{ep}@{F}@{world}", episode=ep, world=world, group=group, split=job["split"], cls=job["cls"],
               F=F, t_F_s=round(F * DT, 3), n_frames=int(n), status=outc["status"], goal_reached=bool(outc["goal_reached"]),
               onset_frame=job.get("onset"), F_minus_onset=None if job.get("onset") is None else int(F - job["onset"]),
               onset_to_end_frames=job.get("margin_frames"),
               onset_to_end_s=None if job.get("margin_frames") is None else round(job["margin_frames"] * DT, 3),
               pose_F=[float(pose[F, 0]), float(pose[F, 1]), float(pose[F, 2])], vx_F=vx, throttle_F=thr, brake_F=float(act[F, 2]),
               steer_F=float(act[F, 0]), desired_speed_F=float(desired[F]), parked_F=bool(parked[F]),
               slow_under_throttle_F=bool(slow[F]), in_stall_run_at_F=stalled_F,
               station_F_m=float(smax[F]), lateral_dev_F_m=float(dev[F]), route_len_m=float(L), remaining_m=float(L - smax[F]),
               goal_xy=[float(case["goal_xy"][0]), float(case["goal_xy"][1])], goal_radius_m=float(case["goal_radius_m"]),
               goal_dist_F_m=float(np.hypot(case["goal_xy"][0] - pose[F, 0], case["goal_xy"][1] - pose[F, 1])),
               start_xy=[float(v) for v in case["layout"]["start_xy"]],
               route_file=rel(rf), route_kind=kind, route_file_matches_recording=bool(match), route_sha256=sha,
               recorded_route_sha256=rec_sha, command_reference=rel(src / "command_reference.npz"), run_dir=rel(src),
               paired=job["paired"], same_frame=job["same_frame"], hist_valid_steps=int(ok.sum()),
               sinkage_drop_m=None, spindle_height_F_m=None)
    if world == "crm":
        with np.load(src / "crm_extra.npz", allow_pickle=False) as e:
            h = (np.asarray(e["spindle_z_m"], np.float64) - np.asarray(e["bmp_ground_z_m"], np.float64)[:, None]).mean(1)
        rec["sinkage_drop_m"] = round(float(h[max(0, F - SINK_WIN)] - h[F]), 4)
        rec["spindle_height_F_m"] = round(float(h[F]), 4)
    return dict(rec=rec, hist=hist, hmask=ok)


# ----------------------------------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["select", "continuations"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-train", type=int, default=700)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--clean-frac", type=float, default=0.6)
    ap.add_argument("--lp-same-frame", choices=["prefer", "never"], default="prefer",
                    help="low-progress pairs: 'prefer' one common F when the windows overlap (default), 'never' = own onset-10 per world")
    ap.add_argument("--rescan", action="store_true", help="ignore scan_<world>.json in --out")
    ap.add_argument("--limit", type=int, default=0, help="scan only the first N manifest episodes per world (smoke)")
    a = ap.parse_args(argv)
    if a.stage != "select":
        raise SystemExit("stage 'continuations' is a later stage (after the pending gc_control fix); only 'select' is implemented")
    workers = max(1, min(6, a.workers))
    out = a.out; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    man = json.loads(a.manifest.read_text())
    assert man.get("schema") == 3, "cache manifest schema != 3"
    eps = {w: sorted(k[:-len(w) - 1] for k in man["episodes"] if k.endswith(f"@{w}")) for w in WORLDS}
    if a.limit:
        eps = {w: v[:a.limit] for w, v in eps.items()}
    for w in WORLDS:
        for e in eps[w]:
            g = man["group_of"][f"{e}@{w}"]
            assert not BLACKLIST.match(e) and not BLACKLIST.match(g), f"planner-suite id/group in the manifest: {e} / {g}"
            assert TRAIN_GROUP.match(g), f"unexpected group id {g}"

    # ---- pass 1: scan every episode of both worlds (cached)
    scan = {}
    for w in WORLDS:
        cache = out / f"scan_{w}.json"
        if cache.exists() and not a.rescan and not a.limit:
            scan[w] = json.loads(cache.read_text())
            if set(scan[w]) != set(eps[w]):
                print(f"[{w}] cached scan covers {len(scan[w])} episodes != manifest {len(eps[w])}: rescanning", flush=True)
                scan[w] = None
        else:
            scan[w] = None
        if scan[w] is None:
            jobs = [dict(world=w, episode=e) for e in eps[w]]
            rows = {}
            with Pool(workers) as pool:
                for i, r in enumerate(pool.imap_unordered(scan_one, jobs, chunksize=32)):
                    rows[r["episode"]] = r
                    if (i + 1) % 5000 == 0:
                        print(f"[{w}] scanned {i + 1}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
            scan[w] = rows
            if not a.limit:
                cache.write_text(json.dumps(rows))
        errs = [e for e, r in scan[w].items() if "error" in r]
        print(f"[{w}] scan: {len(scan[w])} episodes, {len(errs)} errors ({time.time() - t0:.0f} s)", flush=True)
        if errs:
            print("   first errors:", [(e, scan[w][e]["error"]) for e in errs[:3]], flush=True)

    # ---- survivor statistics (before selection)
    split_of = lambda e, w: man["split_of"][f"{e}@{w}"]  # noqa: E731
    surv = {}
    for w in WORLDS:
        st = {}
        for split in ("train", "val", "test"):
            rows = [r for e, r in scan[w].items() if "error" not in r and split_of(e, w) == split]
            with_onset = [r for r in rows if r["onset"] is not None]
            pref_ok = [r for r in with_onset if (r["onset"] + LP_PREF) in r["lp"]]
            any_ok = [r for r in with_onset if r["lp"]]
            cm_any = [r for r in rows if r["cm"]]
            st[split] = dict(episodes=len(rows), with_stall_onset=len(with_onset),
                             lp_survivors_at_onset_minus_10=len(pref_ok), lp_survivors_any_F_in_window=len(any_ok),
                             lp_groups_with_survivor=len({man["group_of"][f"{r['episode']}@{w}"] for r in any_ok}),
                             onset_s=pct([r["onset"] * DT for r in with_onset]),
                             onset_to_end_s_all=pct([r["margin_frames"] * DT for r in with_onset]),
                             onset_to_end_s_survivors=pct([r["margin_frames"] * DT for r in any_ok]),
                             cm_episodes_any_F=len(cm_any),
                             cm_episodes_per_F={str(F): sum(1 for r in rows if F in r["cm"]) for F in CM_FRAMES},
                             cm_groups_with_candidate=len({man["group_of"][f"{r['episode']}@{w}"] for r in cm_any}))
            if w == "crm":
                drops = [x[2] for r in with_onset for x in r["lp_window"] if x[0] == r["onset"] + LP_PREF and x[2] is not None]
                st[split]["sinkage_drop_at_onset_minus_10_m"] = pct(drops)
        surv[w] = st
    # paired pools
    both = sorted(set(scan["crm"]) & set(scan["rigid"]))
    pair = {}
    for split in ("train", "val", "test"):
        e_s = [e for e in both if split_of(e, "crm") == split and "error" not in scan["crm"][e] and "error" not in scan["rigid"][e]]
        pair[split] = dict(
            episodes_in_both_worlds=len(e_s),
            clean_moving_common_F=sum(1 for e in e_s if set(scan["crm"][e]["cm"]) & set(scan["rigid"][e]["cm"])),
            low_progress_both=sum(1 for e in e_s if scan["crm"][e]["lp"] and scan["rigid"][e]["lp"]),
            low_progress_both_common_F=sum(1 for e in e_s if set(scan["crm"][e]["lp"]) & set(scan["rigid"][e]["lp"])),
            stall_onset_both=sum(1 for e in e_s if scan["crm"][e]["onset"] is not None and scan["rigid"][e]["onset"] is not None),
            stall_onset_crm_only=sum(1 for e in e_s if scan["crm"][e]["onset"] is not None and scan["rigid"][e]["onset"] is None),
            stall_onset_rigid_only=sum(1 for e in e_s if scan["crm"][e]["onset"] is None and scan["rigid"][e]["onset"] is not None))

    # ---- selection
    def q(n):
        c = int(round(a.clean_frac * n)); return {"clean_moving": c, "low_progress": n - c}
    quotas = {"train": q(a.n_train), "val": q(a.n_val), "test": q(a.n_test)}
    rng = np.random.default_rng(a.seed)
    picks, book = select(scan, man, quotas, rng, lp_same_frame=(a.lp_same_frame == "prefer"))
    for w in WORLDS:
        for p in picks[w]:
            r = scan[w][p["episode"]]
            p.update(onset=r["onset"], margin_frames=r.get("margin_frames"))
        # constraints
        c_ep = Counter(p["episode"] for p in picks[w]); assert max(c_ep.values(), default=0) <= 1, "episode used twice"
        c_g = Counter(p["group"] for p in picks[w]); assert max(c_g.values(), default=0) <= 2, "group used > 2 times"
        for p in picks[w]:
            assert not BLACKLIST.match(p["episode"]) and not BLACKLIST.match(p["group"])
            assert p["split"] == man["split_of"][f"{p['episode']}@{w}"]
            if p["cls"] == "clean_moving":
                assert p["F"] in CM_FRAMES
            else:
                assert p["onset"] is not None and LP_LO <= p["F"] - p["onset"] <= LP_HI, p

    # ---- pass 2: extract records + history windows
    anchors = {}; hist_arrays = {}
    for w in WORLDS:
        jobs = sorted(picks[w], key=lambda p: (p["split"], p["cls"], p["episode"]))
        with Pool(workers) as pool:
            res = pool.map(extract_one, jobs, chunksize=8)
        recs = [r["rec"] for r in res]
        for i, r in enumerate(recs):
            r["hist_index"] = i
        anchors[w] = recs
        hist_arrays[w] = dict(hist=np.stack([r["hist"] for r in res]) if res else np.zeros((0, HIST_T, 15), np.float32),
                              hmask=np.stack([r["hmask"] for r in res]) if res else np.zeros((0, HIST_T), bool),
                              anchor_id=np.array([r["anchor_id"] for r in recs]), episode=np.array([r["episode"] for r in recs]),
                              F=np.array([r["F"] for r in recs], np.int64), hist_cols=np.asarray(HIST_STATE_COLS, np.int16))
        # checks: history last step equals the state at F; mask count = min(F, 40); CRM screen; CM rules
        H, M = hist_arrays[w]["hist"], hist_arrays[w]["hmask"]
        assert np.all(M.sum(1) == np.minimum(hist_arrays[w]["F"], HIST_T)), "hmask count != min(F, 40)"
        for r, h, m in zip(recs, H, M):
            assert abs(float(h[-1, 0]) - r["vx_F"]) < 1e-6 and m[-1], "hist last step != state at F"
            if r["cls"] == "clean_moving":
                assert r["vx_F"] > CM_VX and r["lateral_dev_F_m"] < CM_DEV and r["remaining_m"] >= CM_REM and not r["parked_F"], r["anchor_id"]
            else:
                assert not r["parked_F"] and r["remaining_m"] >= LP_MIN_REM, r["anchor_id"]
                if w == "crm":
                    assert r["sinkage_drop_m"] < SINK_MAX, r["anchor_id"]
        np.savez_compressed(out / f"hist_{w}.npz", **hist_arrays[w])
        json.dump(recs, open(out / f"anchors_{w}.json", "w"), indent=1)

    # ---- summary
    def table(w):
        t = {}
        for split in ("train", "val", "test"):
            for cls in CLASSES:
                rs = [r for r in anchors[w] if r["split"] == split and r["cls"] == cls]
                t[f"{split}/{cls}"] = dict(n=len(rs), quota=quotas[split][cls], paired=sum(r["paired"] for r in rs),
                                          same_frame=sum(r["same_frame"] for r in rs), groups=len({r["group"] for r in rs}))
        return t
    summary = dict(
        written=time.strftime("%Y-%m-%d %H:%M:%S"), wall_s=round(time.time() - t0, 1), seed=a.seed, workers=workers,
        lp_same_frame=a.lp_same_frame,
        quotas=quotas, rules=dict(clean_moving=dict(F=list(CM_FRAMES), vx_gt=CM_VX, lateral_dev_lt=CM_DEV, remaining_ge=CM_REM,
                                                    not_parked=True, no_stall_onset_or_rollback_before_F=True),
                                  low_progress=dict(onset=f"first run of {STALL_RUN} frames |vx|<{STALL_VX} & throttle>{STALL_THR}, searched from frame {S0}",
                                                    F_window=[LP_LO, LP_HI], F_preferred=LP_PREF, crm_screen=f"mean spindle height drop over {SINK_WIN} frames < {SINK_MAX} m",
                                                    not_parked=True, remaining_ge=LP_MIN_REM),
                                  per_episode_max=1, per_group_max=2, paired_first=True),
        sources=dict(manifest=rel(a.manifest), crm=rel(CRM_ROOT), rigid=[rel(r) for r in RIGID_ROOTS],
                     routes_designed=rel(ROUTES_DESIGNED), routes_onpolicy=rel(ROUTES_ONPOLICY)),
        counts={w: table(w) for w in WORLDS},
        totals={w: dict(n=len(anchors[w]), per_split=dict(Counter(r["split"] for r in anchors[w])),
                        per_class=dict(Counter(r["cls"] for r in anchors[w])), groups=len({r["group"] for r in anchors[w]}),
                        paired=sum(r["paired"] for r in anchors[w]), same_frame=sum(r["same_frame"] for r in anchors[w]),
                        route_file_matches=sum(r["route_file_matches_recording"] for r in anchors[w]),
                        recorded_sha_matches=sum(1 for r in anchors[w] if r["recorded_route_sha256"] in (None, r["route_sha256"])),
                        recorded_sha_present=sum(1 for r in anchors[w] if r["recorded_route_sha256"]),
                        status=dict(Counter(r["status"] for r in anchors[w]))) for w in WORLDS},
        F_distribution={w: dict(clean_moving=dict(Counter(str(r["F"]) for r in anchors[w] if r["cls"] == "clean_moving")),
                                low_progress_F_minus_onset=dict(sorted(Counter(str(r["F_minus_onset"]) for r in anchors[w] if r["cls"] == "low_progress").items(), key=lambda kv: int(kv[0]))),
                                low_progress_t_F_s=pct([r["t_F_s"] for r in anchors[w] if r["cls"] == "low_progress"]),
                                hist_fully_valid=sum(1 for r in anchors[w] if r["hist_valid_steps"] == HIST_T)) for w in WORLDS},
        selected_low_progress={w: dict(onset_to_end_s=pct([r["onset_to_end_s"] for r in anchors[w] if r["cls"] == "low_progress"]),
                                       sinkage_drop_m=pct([r["sinkage_drop_m"] for r in anchors[w] if r["cls"] == "low_progress" and r["sinkage_drop_m"] is not None]),
                                       vx_F=pct([r["vx_F"] for r in anchors[w] if r["cls"] == "low_progress"]),
                                       in_stall_run_at_F=sum(r["in_stall_run_at_F"] for r in anchors[w] if r["cls"] == "low_progress"),
                                       goal_reached_recorded=sum(r["goal_reached"] for r in anchors[w] if r["cls"] == "low_progress")) for w in WORLDS},
        selected_clean_moving={w: dict(vx_F=pct([r["vx_F"] for r in anchors[w] if r["cls"] == "clean_moving"]),
                                       remaining_m=pct([r["remaining_m"] for r in anchors[w] if r["cls"] == "clean_moving"]),
                                       goal_reached_recorded=sum(r["goal_reached"] for r in anchors[w] if r["cls"] == "clean_moving"),
                                       sinkage_drop_m=pct([r["sinkage_drop_m"] for r in anchors[w] if r["cls"] == "clean_moving" and r["sinkage_drop_m"] is not None])) for w in WORLDS},
        survivors=surv, paired_pools=pair, selection=book,
        scan_errors={w: sum(1 for r in scan[w].values() if "error" in r) for w in WORLDS})
    json.dump(summary, open(out / "summary.json", "w"), indent=1)

    # ---- printed table
    print(f"\nAnchors written to {out}  (seed {a.seed}, {summary['wall_s']} s)")
    print(f"{'world':6} {'split':6} {'class':13} {'n':>4} {'quota':>5} {'paired':>6} {'sameF':>5} {'groups':>6}")
    for w in WORLDS:
        for k, v in summary["counts"][w].items():
            split, cls = k.split("/")
            print(f"{w:6} {split:6} {cls:13} {v['n']:4d} {v['quota']:5d} {v['paired']:6d} {v['same_frame']:5d} {v['groups']:6d}")
        tot = summary["totals"][w]
        print(f"{w:6} total {tot['n']} anchors in {tot['groups']} groups; route file matches recording {tot['route_file_matches']}/{tot['n']}; "
              f"recorded outcome status {tot['status']}")
    for w in WORLDS:
        fd = summary["F_distribution"][w]
        print(f"[{w}] clean-moving F: {fd['clean_moving']}   low-progress F-onset: {fd['low_progress_F_minus_onset']}   "
              f"windows fully valid (F >= 40): {fd['hist_fully_valid']}/{len(anchors[w])}")
    for w in WORLDS:
        for split in ("train", "val", "test"):
            s = surv[w][split]
            extra = f"  sinkage drop at onset-10 p50/p90 {s['sinkage_drop_at_onset_minus_10_m']['p50']}/{s['sinkage_drop_at_onset_minus_10_m']['p90']} m" if w == "crm" else ""
            print(f"[{w}/{split}] episodes {s['episodes']}: with stall onset {s['with_stall_onset']} -> survivors at onset-10 "
                  f"{s['lp_survivors_at_onset_minus_10']}, any F in window {s['lp_survivors_any_F_in_window']} ({s['lp_groups_with_survivor']} groups); "
                  f"onset-to-end s p10/50/90 all {s['onset_to_end_s_all']['p10']}/{s['onset_to_end_s_all']['p50']}/{s['onset_to_end_s_all']['p90']}, "
                  f"survivors {s['onset_to_end_s_survivors']['p10']}/{s['onset_to_end_s_survivors']['p50']}/{s['onset_to_end_s_survivors']['p90']}; "
                  f"clean-moving candidates {s['cm_episodes_any_F']} ({s['cm_groups_with_candidate']} groups) per F {s['cm_episodes_per_F']}{extra}")
    for split in ("train", "val", "test"):
        print(f"[pairs/{split}] {pair[split]}")
    for w in WORLDS:
        sl = summary["selected_low_progress"][w]
        print(f"[{w}] selected low-progress: onset-to-end s p10/50/90 {sl['onset_to_end_s']['p10']}/{sl['onset_to_end_s']['p50']}/{sl['onset_to_end_s']['p90']}, "
              f"vx at F p10/50/90 {sl['vx_F']['p10']}/{sl['vx_F']['p50']}/{sl['vx_F']['p90']}, inside a stall run at F {sl['in_stall_run_at_F']}, "
              f"sinkage drop p50/p90 {sl['sinkage_drop_m'].get('p50')}/{sl['sinkage_drop_m'].get('p90')} m")
    short = [(w, k, v) for w in WORLDS for k, v in summary["counts"][w].items() if v["n"] < v["quota"]]
    if short:
        print("SHORTFALL:", short)
    print(f"selection bookkeeping: {json.dumps(book)}")


if __name__ == "__main__":
    main()
