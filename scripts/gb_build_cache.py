#!/usr/bin/env python
"""Schema-3 dynamics cache for milestone B: CRM + rigid f104 recordings -> one per-episode npz each + manifest.

Sources (read only): CRM ``crm_f104_v1/collect_v1/runs`` (domain 1) and rigid ``fdm_f104_50h_20260909/production_v3/runs``
+ ``production_v4/runs`` (domain 0). Every run dir holds ``trajectory.npz`` (state (T,17) in the ``tire_normal_force_omega_pt``
order, action (T,3) [steer, throttle, brake], pose (T,3), power_kw (T,), parked (T,)), ``command_reference.npz`` (per-interval
``desired_speed_mps`` + the reference route) , ``outcome.json`` (status) and ``case.json`` (group split as declared).

Per episode ``<id>@crm.npz`` / ``<id>@rigid.npz`` (ids collide across the two worlds):
  z1 (T,17) f32, act (T,3) f32, pose (T,3) f32, power (T,1) f32 kW, desired_speed (T,) f32, parked (T,) bool,
  stalled (T,) bool  = frame inside a run of >= 20 consecutive frames with |vx| < 0.3 m/s and throttle > 0.3 (contract),
  hold_ok (T,) bool  = |act[k+1]-act[k]| <= 0.1 on every channel and no throttle/brake flip between k and k+1
                       (flip = throttle>0 at k and brake>0 at k+1, or the reverse). hold_ok[k] is True only when frame k+1
                       is in the cache: the last kept frame is always False (also for cut episodes, whose recording continues),
  route_waypoints (N,2), route_speeds (N,), route_headings (N,), route_stations (N,),
  domain (0 rigid / 1 crm), group, status, episode_id, source_dir, n_recorded (frames before the cut), cut (bool).
Episodes longer than ``--max-frames`` are cut to that length (hold_ok/stalled computed on the full recording first, then
cut; the cut and the dropped frames are counted per status in the report).

``cache_manifest.json``: schema 3, episodes (sorted keys), domain_of, group_of, split_of, status_of, n_frames_of, plus a
``report`` block (counts per domain/split/status, hold_ok retention per regime, held-out groups, cuts, exclusions, size).
Split = the twin group split (``twin_crm.npz`` group/split arrays) [PLAN R1/R12]. A group absent there is accepted as
'train' only with ``--case-split-fallback`` (the default, as the module brief asks) AND only if it is an ``f104_v2_group_*``
group whose case.json says train; every such episode is counted and its group listed in the report
(``case_split_fallback_train``) so the deviation from "the twin split is the only split" is never silent. With
``--no-case-split-fallback`` such episodes are excluded and counted. Planner-suite groups (``f104_crm_eval_group_*``,
``f104_g1_test_group_*``, ``f104_pair_group_*``) are asserted absent.
"""
from __future__ import annotations

import argparse, json, os, re, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CRM = REPO / "artifacts/traverse/crm_f104_v1/collect_v1/runs"
DEFAULT_RIGID = [REPO / "artifacts/traverse/fdm_f104_50h_20260909/production_v3/runs",
                 REPO / "artifacts/traverse/fdm_f104_50h_20260909/production_v4/runs"]
DEFAULT_TWIN = REPO / "artifacts/traverse/crm_night2_v1/datasets/twin_crm.npz"
DEFAULT_QA = REPO / "artifacts/traverse/crm_f104_v1/collect_v1/qa.json"
DEFAULT_OUT = REPO / "artifacts/traverse/generalist_20260921/B_tracker/cache_v1"
BLACKLIST = re.compile(r"^(f104_crm_eval_group_|f104_g1_test_group_|f104_pair_group_)")
TRAIN_GROUP = re.compile(r"^f104_v2_group_\d+$")
DOMAIN_NAME = {0: "rigid", 1: "crm"}
HOLD_TOL, STALL_VX, STALL_THR, STALL_RUN = 0.1, 0.3, 0.3, 20


def group_of_id(episode_id: str) -> str:
    """'f104_v2_group_0000_route_03' -> 'f104_v2_group_0000' (id = <group>_<kind>_<nn>)."""
    m = re.match(r"^(.*_group_\d+)_[a-z]+_\d+(__[A-Za-z0-9]+)?$", episode_id)   # optional __<tag> suffix (hold-mode / harvest runs)
    if not m:
        raise ValueError(f"cannot parse group from episode id {episode_id!r}")
    return m.group(1)


def stalled_mask(z1: np.ndarray, act: np.ndarray, run: int = STALL_RUN) -> np.ndarray:
    """(T,) bool: frames inside a run of >= ``run`` consecutive frames with |vx| < 0.3 and throttle > 0.3."""
    slow = (np.abs(z1[:, 0]) < STALL_VX) & (act[:, 1] > STALL_THR)
    out = np.zeros(len(slow), bool)
    start = None
    for k, s in enumerate(np.r_[slow, False]):
        if s and start is None:
            start = k
        elif not s and start is not None:
            if k - start >= run:
                out[start:k] = True
            start = None
    return out


def hold_ok_mask(act: np.ndarray, tol: float = HOLD_TOL) -> tuple[np.ndarray, np.ndarray]:
    """(hold_ok (T,), flip (T,)) for transitions k -> k+1; the last frame has no successor (both False)."""
    T = len(act)
    hold, flip = np.zeros(T, bool), np.zeros(T, bool)
    if T >= 2:
        a0, a1 = act[:-1], act[1:]
        small = np.all(np.abs(a1 - a0) <= tol + 1e-7, axis=1)
        flip[:-1] = ((a0[:, 1] > 0) & (a1[:, 2] > 0)) | ((a0[:, 2] > 0) & (a1[:, 1] > 0))
        hold[:-1] = small & ~flip[:-1]
    return hold, flip


def brake_onset_mask(act: np.ndarray) -> np.ndarray:
    """(T,) bool: first frame of every braking run (brake > 0 here, <= 0 at the previous frame); frame 0 never counts."""
    on = act[:, 2] > 0
    out = np.zeros(len(act), bool)
    out[1:] = on[1:] & ~on[:-1]
    return out


def convert_one(job: dict) -> dict:
    """Read one run dir, write one cache npz, return its manifest row (never raises for a bad run: returns error)."""
    src, key, domain, out_dir, max_frames = Path(job["src"]), job["key"], job["domain"], Path(job["out"]), job["max_frames"]
    try:
        with np.load(src / "trajectory.npz", allow_pickle=False) as t:
            z1 = np.asarray(t["state"], np.float32); act = np.asarray(t["action"], np.float32)
            pose = np.asarray(t["pose"], np.float32); power = np.asarray(t["power_kw"], np.float32)[:, None]
            parked = np.asarray(t["parked"], bool)
            assert list(t["state_fields"])[0] == "vel_body_x_mps" and z1.shape[1] == 17, "unexpected state preset"
            assert abs(float(t["dt_s"]) - 0.05) < 1e-9
        with np.load(src / "command_reference.npz", allow_pickle=False) as c:
            desired = np.asarray(c["desired_speed_mps"], np.float32)
            route = {f"route_{k}": np.asarray(c[f"reference_{k}"]) for k in ("waypoints", "speeds", "headings", "stations")}
        outcome = json.loads((src / "outcome.json").read_text()); status = outcome["status"]
        held_mode = outcome.get("mode") in ("pid_held", "pid_perturbed", "policy")   # ext collectors: every recorded action IS the held triple
        T = len(z1)
        assert act.shape == (T, 3) and pose.shape == (T, 3) and power.shape == (T, 1) and parked.shape == (T,), "row count mismatch"
        assert desired.shape == (T,), f"desired_speed length {desired.shape} != {T}"
        if T < 2:
            return {"key": key, "error": f"only {T} frame(s)"}
        stalled = stalled_mask(z1, act)
        hold, flip = hold_ok_mask(act)
        if held_mode:   # hold-mode recordings (PLAN v2 B3): the transition k -> k+1 is a true zero-order hold whatever the step size
            hold = np.ones(T, bool); hold[-1] = False
        onset = brake_onset_mask(act)
        cut = T > max_frames
        n = min(T, max_frames)
        sl = slice(0, n)
        hold_kept = hold[sl].copy(); hold_kept[-1] = False  # hold_ok[k] True only when frame k+1 is in the cache
        np.savez_compressed(out_dir / f"{key}.npz", z1=z1[sl], act=act[sl], pose=pose[sl], power=power[sl],
                            desired_speed=desired[sl], parked=parked[sl], stalled=stalled[sl], hold_ok=hold_kept,
                            domain=np.int64(domain), group=np.array(job["group"]), status=np.array(status),
                            episode_id=np.array(job["id"]), source_dir=np.array(str(src)), n_recorded=np.int64(T),
                            cut=np.bool_(cut), **route)
        # retention bookkeeping over the KEPT frames that have a successor (k < n-1)
        valid = np.zeros(n, bool); valid[:-1] = True
        reg = {"stalled": stalled[sl] & valid, "moving": ~stalled[sl] & valid, "brake_onset": onset[sl] & valid,
               "pre_brake_onset": np.r_[onset[1:n], False] & valid, "all": valid}
        counts = {r: [int(hold_kept[m].sum()), int(m.sum())] for r, m in reg.items()}
        return {"key": key, "id": job["id"], "domain": domain, "group": job["group"], "status": status, "n_frames": n,
                "n_recorded": T, "cut": cut, "retention": counts, "n_flip": int(flip[sl][valid].sum()),
                "n_stalled": int(stalled[sl].sum()), "n_parked_stalled": int((stalled[sl] & parked[sl]).sum()),
                "n_stalled_dropped": int(stalled[n:].sum()),
                "bytes": os.path.getsize(out_dir / f"{key}.npz")}
    except Exception as exc:  # noqa: BLE001
        return {"key": key, "error": f"{type(exc).__name__}: {exc}"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crm-runs", nargs="+", default=[str(DEFAULT_CRM)])
    ap.add_argument("--rigid-runs", nargs="+", default=[str(p) for p in DEFAULT_RIGID])
    ap.add_argument("--twin", default=str(DEFAULT_TWIN), help="twin_crm.npz with group + split arrays (the only split)")
    ap.add_argument("--qa", default=str(DEFAULT_QA), help="collect_v1 qa.json; its flagged ids are excluded (use '' to keep)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="self-test: at most this many episodes per source (0 = all)")
    ap.add_argument("--case-split-fallback", action=argparse.BooleanOptionalAction, default=True,
                    help="accept an f104_v2_group_* group absent from the twin file as 'train' when its case.json says so (counted and listed in the report)")
    args = ap.parse_args(argv)
    args.workers = max(1, min(8, args.workers))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with np.load(args.twin, allow_pickle=True) as tw:
        pairs = sorted(set(zip(tw["group"].tolist(), tw["split"].tolist())))
    split_of_group = dict(pairs)
    assert len(split_of_group) == len(pairs), "a twin group carries two splits"
    twin_counts = {s: sum(1 for g in split_of_group.values() if g == s) for s in ("train", "val", "test")}
    flagged = set()
    if args.qa and Path(args.qa).exists():
        flagged = {f["id"] for f in json.loads(Path(args.qa).read_text()).get("flagged_ids", [])}

    jobs, excluded = [], {"blacklisted": [], "unknown_group": [], "qa_flagged": [], "no_files": []}
    fallback_train = []  # episodes whose 'train' split came from case.json, not from the twin file
    sources = [(1, Path(p)) for p in args.crm_runs] + [(0, Path(p)) for p in args.rigid_runs]
    for domain, root in sources:
        ids = sorted(os.listdir(root))
        if args.limit:
            ids = ids[: args.limit]
        for eid in ids:
            src = root / eid
            if not all((src / f).exists() for f in ("trajectory.npz", "command_reference.npz", "outcome.json", "case.json")):
                excluded["no_files"].append(f"{eid}@{DOMAIN_NAME[domain]}"); continue
            group = group_of_id(eid)
            if BLACKLIST.match(eid) or BLACKLIST.match(group):
                excluded["blacklisted"].append(f"{eid}@{DOMAIN_NAME[domain]}"); continue
            if domain == 1 and eid in flagged:
                excluded["qa_flagged"].append(f"{eid}@crm"); continue
            split = split_of_group.get(group)
            if split is None:
                case_split = json.loads((src / "case.json").read_text()).get("split")
                if args.case_split_fallback and TRAIN_GROUP.match(group) and case_split == "train":
                    split = "train"; fallback_train.append(f"{eid}@{DOMAIN_NAME[domain]}")
                else:
                    excluded["unknown_group"].append(f"{eid}@{DOMAIN_NAME[domain]}"); continue
            jobs.append({"src": str(src), "id": eid, "key": f"{eid}@{DOMAIN_NAME[domain]}", "domain": domain,
                         "group": group, "split": split, "out": str(out), "max_frames": args.max_frames})
    assert not excluded["blacklisted"], f"planner-suite episodes present in the sources: {excluded['blacklisted'][:5]}"
    print(f"{len(jobs)} episodes queued ({sum(1 for j in jobs if j['domain'] == 1)} crm, {sum(1 for j in jobs if j['domain'] == 0)} rigid); "
          f"excluded: { {k: len(v) for k, v in excluded.items()} }", flush=True)
    if fallback_train:
        print(f"WARNING: {len(fallback_train)} episodes from {len({group_of_id(k.split('@')[0]) for k in fallback_train})} groups absent from the twin "
              f"split file were accepted as 'train' from their case.json (--case-split-fallback); they are listed in the report", flush=True)

    rows, errors = [], []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(convert_one, jobs, chunksize=16), 1):
            (errors if "error" in r else rows).append(r)
            if i % 5000 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done, {len(errors)} errors, {time.time() - t0:.0f}s", flush=True)
    split_by_key = {j["key"]: j["split"] for j in jobs}
    rows.sort(key=lambda r: r["key"])
    episodes = [r["key"] for r in rows]
    domain_of = {r["key"]: int(r["domain"]) for r in rows}
    group_of = {r["key"]: r["group"] for r in rows}
    split_of = {k: split_by_key[k] for k in episodes}
    status_of = {r["key"]: r["status"] for r in rows}
    n_frames_of = {r["key"]: int(r["n_frames"]) for r in rows}
    for k in episodes:  # the blacklist assertion, on the written set, by id and by group
        assert not BLACKLIST.match(k) and not BLACKLIST.match(group_of[k]), k

    def agg(sel):
        rs = [r for r in rows if sel(r)]
        ret = {}
        for reg in ("all", "stalled", "moving", "brake_onset", "pre_brake_onset"):
            ok = sum(r["retention"][reg][0] for r in rs); n = sum(r["retention"][reg][1] for r in rs)
            ret[reg] = {"hold_ok": ok, "frames": n, "retention": (ok / n) if n else None}
        n_tr = sum(r["retention"]["all"][1] for r in rs)
        return {"episodes": len(rs), "frames": sum(r["n_frames"] for r in rs), "frames_recorded": sum(r["n_recorded"] for r in rs),
                "cut": sum(1 for r in rs if r["cut"]), "cut_by_status": {s: sum(1 for r in rs if r["cut"] and r["status"] == s) for s in sorted({r["status"] for r in rs if r["cut"]})},
                "frames_dropped_by_cut": sum(r["n_recorded"] - r["n_frames"] for r in rs),
                "stalled_frames_dropped_by_cut": sum(r["n_stalled_dropped"] for r in rs), "stalled_frames": sum(r["n_stalled"] for r in rs),
                "stalled_frames_while_parked": sum(r["n_parked_stalled"] for r in rs),
                "flip_transitions": sum(r["n_flip"] for r in rs), "flip_rate": (sum(r["n_flip"] for r in rs) / n_tr) if n_tr else None,
                "hold_ok_retention": ret,
                "by_status": {s: sum(1 for r in rs if r["status"] == s) for s in sorted({r["status"] for r in rs})},
                "by_split": {s: sum(1 for r in rs if split_of[r["key"]] == s) for s in ("train", "val", "test")},
                "groups": len({r["group"] for r in rs}),
                "heldout_groups_present": {s: len({r["group"] for r in rs if split_of[r["key"]] == s}) for s in ("val", "test")}}

    report = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": round(time.time() - t0, 1), "workers": args.workers,
              "max_frames": args.max_frames, "sources": {DOMAIN_NAME[d]: [str(p) for dd, p in sources if dd == d] for d in (0, 1)},
              "twin_split": str(args.twin), "twin_groups_by_split": twin_counts, "qa_flagged_excluded": sorted(excluded["qa_flagged"]),
              "case_split_fallback": {"enabled": bool(args.case_split_fallback), "episodes": len(fallback_train),
                                      "groups": sorted({group_of_id(k.split("@")[0]) for k in fallback_train}), "first_episodes": sorted(fallback_train)[:50]},
              "excluded_counts": {k: len(v) for k, v in excluded.items()}, "excluded_unknown_group": sorted(excluded["unknown_group"])[:50],
              "errors": errors, "total_bytes": sum(r["bytes"] for r in rows), "total_gib": round(sum(r["bytes"] for r in rows) / 2**30, 3),
              "all": agg(lambda r: True), "crm": agg(lambda r: r["domain"] == 1), "rigid": agg(lambda r: r["domain"] == 0),
              "definitions": {"stalled": f"frame inside a run of >= {STALL_RUN} consecutive frames with |vx| < {STALL_VX} and throttle > {STALL_THR}",
                              "hold_ok": f"|act[k+1]-act[k]| <= {HOLD_TOL} on all 3 channels and no throttle/brake flip; True only when frame k+1 is in the cache (last kept frame False, also for cut episodes)",
                              "flip": "throttle>0 at k and brake>0 at k+1, or brake>0 at k and throttle>0 at k+1",
                              "brake_onset": "frame k with brake>0 and brake<=0 at k-1 (retention of the transition k->k+1)",
                              "pre_brake_onset": "frame k-1 before a brake onset (the transition that switches the brake on)",
                              "moving": "not stalled", "retention": "hold_ok transitions / transitions with a successor frame in the regime"}}
    manifest = {"schema": 3, "episodes": episodes, "domain_of": domain_of, "group_of": group_of, "split_of": split_of,
                "status_of": status_of, "n_frames_of": n_frames_of, "domain_names": DOMAIN_NAME, "definitions": report["definitions"],
                "split_source": f"twin group split ({Path(args.twin).name}); case.json fallback accepted {len(fallback_train)} episodes as train",
                "z1_preset": "tire_normal_force_omega_pt", "act_columns": ["steer", "throttle", "brake"], "dt_s": 0.05,
                "blacklist": BLACKLIST.pattern, "report": report}
    (out / "cache_manifest.json").write_text(json.dumps(manifest))
    (out / "build_report.json").write_text(json.dumps(report, indent=1))
    for name in ("all", "crm", "rigid"):
        a = report[name]
        ret = {k: (None if v["retention"] is None else round(v["retention"], 4)) for k, v in a["hold_ok_retention"].items()}
        print(f"{name}: {a['episodes']} episodes, {a['frames']} frames ({a['cut']} cut at {args.max_frames}), splits {a['by_split']}, "
              f"held-out groups {a['heldout_groups_present']}, statuses {a['by_status']}\n    hold_ok retention {ret}, "
              f"flip rate {a['flip_rate']:.4f}, stalled frames {a['stalled_frames']} (parked {a['stalled_frames_while_parked']})", flush=True)
    print(f"cut at {args.max_frames}: all {report['all']['cut_by_status']} ({report['all']['frames_dropped_by_cut']} frames dropped, "
          f"{report['all']['stalled_frames_dropped_by_cut']} of them stalled); case.json split fallback used for {len(fallback_train)} episodes")
    print(f"{len(episodes)} episodes -> {out} ({report['total_gib']} GiB, {len(errors)} errors, {report['wall_s']} s)")


if __name__ == "__main__":
    main()
