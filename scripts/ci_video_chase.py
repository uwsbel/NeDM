#!/usr/bin/env python3
"""3D chase-camera videos of manifest arms (CRM-improvement study), re-simulated locally.

Arms come from ``artifacts/traverse/crm_improve_20260922/videos/manifest.json`` (case file, approach route, branch
frame, branch route, recorded run directory).  Nothing in the physics is changed or copied:

  soil   ``scripts/crm_collect_ext.py --mode branch`` is called through its own ``main(argv, scene_hook, frame_hook)``
         (the optional visual hooks it inherited from ``crm_collect.py``), with the recorded drive's arguments:
         --case, --route <approach>, --branch-frame F, --branch-route <branch>, --crm-config crm_main.json,
         --horizon-s 120.
  rigid  ``scripts/gen_collect_ext.py --mode branch --local`` (same arguments, horizon 120 s).  Its frame observer
         factory ``gen_collect.make_observer`` is wrapped at run time so the unchanged telemetry observer still runs
         and the chase camera renders after it on every recorded frame.

The hooks only add visual shapes to the fixed ground body, a sensor manager and one camera (no bodies, no contacts)
and read positions.  ``--no-render`` runs the same hooks without any rendering, so a camera-free re-drive can be
compared with the filmed one.

Camera: an OptiX RGB camera attached to the ground body and re-posed before every render 12 m behind and 5 m above
the vehicle, following its position and (lightly smoothed) heading.  Markers: the straight approach line (light grey
spheres, shown until the decision frame), the planner's route (orange spheres, shown from the decision frame on:
visibility is switched and the render scene rebuilt once at that frame), the goal circle (violet).

Soil: the sensor module cannot draw SPH particles from Python (``ChSensorManager.AttachFsiSphSystem`` exists in this
build but its options struct is not wrapped, and the default options draw nothing).  The ground is therefore a
visual-only mesh rebuilt from the simulated particle positions: one vertex per 16 cm column (2 x 2 particle
columns) at the arena heightmap height plus the displacement of that column's top since set-up (top = 4th-highest
particle, so up to three thrown particles do not raise it).  Untouched ground is therefore the smooth heightmap (the
8 cm particle layers would otherwise draw as terraces; the initial column tops match the heightmap to 2.5 cm rms),
and every rut, berm or dug-in wheel is the particles' own displacement, at 16 cm resolution.  Columns within 5 m of
the vehicle are refreshed every frame; particles further away are outside the solver's active domain.

Subcommands (drives need the local Chrono build; CRM drives one at a time under ``flock /tmp/luffy_crm.lock``).
``--case`` is any manifest key (soil, rigid, soil2, ...); the world (soil or rigid) comes from the manifest entry.

  PYTHONPATH=/home/harry/chrono/build/bin /usr/bin/python3.12 scripts/ci_video_chase.py drive --case rigid --arm HnG
  flock /tmp/luffy_crm.lock env PYTHONPATH=/home/harry/chrono/build/bin /usr/bin/python3.12 \
      scripts/ci_video_chase.py drive --case soil --arm HnG [--no-render]
  python scripts/ci_video_chase.py repro                         # -> videos/chase_reproduction.json
  python scripts/ci_video_chase.py compose --case soil --arm HnG # -> videos/chase_soil_HnG.mp4
  python scripts/ci_video_chase.py side --case soil              # -> videos/chase_soil_side_by_side.mp4

Choosing a further soil case (videos/chase_work/screen/run_screen.sh drives the screen):

  python scripts/ci_video_chase.py soil-candidates --out <candidates.json> --exclude <groups already filmed>
  python scripts/ci_video_chase.py --manifest <screen manifest> add-case --key g0244 --group f104_pair_group_0244
  ... --manifest <screen manifest> --work <screen folder> drive --case g0244 --arm H3s --no-render   (all six arms)
  python scripts/ci_video_chase.py --manifest <screen manifest> --work <screen folder> screen --candidates <file>
  python scripts/ci_video_chase.py add-case --key soil2 --group <chosen group>   # new key in videos/manifest.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT / "src"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VID = ROOT / "artifacts/traverse/crm_improve_20260922/videos"
WORK = VID / "chase_work"                # --work overrides (screening drives)
MANIFEST = VID / "manifest.json"         # --manifest overrides (screening manifests)
K1 = ROOT / "artifacts/traverse/generalist_20260921/A_adapt"   # the 3 s baseline drives, suite cases, approach routes
K2 = ROOT / "artifacts/traverse/crm_improve_20260922"          # the 1 s / 0.5 s drives
CRM_CONFIG = ROOT / "artifacts/traverse/crm_f104_v1/configs/crm_main.json"
CRM_CHRONO_DATA = ROOT / "chrono/data"            # the night-1 soil demo's data dir (HMMWV files identical to the build's)
RIGID_CHRONO_DATA = Path("/home/harry/chrono/data")  # the local nav re-runs' data dir
DT = 0.05
FPS = 20
W, H = 1280, 720
HFOV = math.radians(62.0)
BACK_M, UP_M = 12.0, 5.0
TOP_K = 4
BOG_LIMIT_M = 0.24 + 0.06   # crm_collect(_ext): soil depth + breakthrough margin, held for 0.25 s -> bogged down
SOIL_RGB = (0.50, 0.40, 0.28)
FILMED = {"crm": ["H3s", "L1_X", "HnG"], "rigid": ["H3s", "L0p5_X", "HnG"]}   # filmed arms per world
WORLD_TITLE = {"crm": "Soft soil (deformable ground)", "rigid": "Rigid ground"}
# the six manifest arms (arm, on-screen label, approach s, decision frame), in manifest order
ARM_SPECS = [("H3s", "CNN-GRU (earlier training), decides after a 3 s approach (old protocol)", 3.0, 60),
             ("L1_Hn", "CNN-GRU, decides after 1 s", 1.0, 20),
             ("L1_X", "Transformer, decides after 1 s", 1.0, 20),
             ("L0p5_Hn", "CNN-GRU, decides after 0.5 s", 0.5, 10),
             ("L0p5_X", "Transformer, decides after 0.5 s", 0.5, 10),
             ("HnG", "CNN-GRU + gradient refinement, decides after 0.5 s (final)", 0.5, 10)]
# extra manifest fields of the 3 s arm: it drives the previous study's CNN-GRU (K1 H_deploy ensemble, trained on
# mixed_reanchor.npz only: 105,193 fit rows), every other CNN-GRU arm the retrained one (K2 deploy_a1_haux_gru:
# mixed_reanchor_plus_branch_both.npz = the same rows + 4,614 branch-drive rows, short_anchor.npz = decision frames
# 10/20/30, anchor_k60.npz = decision frame 60; 217,814 fit rows, 2.07 x)
H3S_EXTRA = {"short_label": "CNN-GRU (earlier), 3 s approach (old)",
             "footnote": "Earlier training: the previous study's CNN-GRU; the other CNN-GRU planners use the retrained "
                         "model, trained on about twice the data."}
# how the side-by-side subtitle names each arm
WHO = {"H3s": "the old 3 s protocol", "L1_Hn": "the 1 s CNN-GRU", "L1_X": "the 1 s transformer",
       "L0p5_Hn": "the 0.5 s CNN-GRU", "L0p5_X": "the 0.5 s transformer", "HnG": "the final planner"}
STUCK_MPS, STUCK_AFTER_S = 0.3, 2.0   # the 'stuck' phase line: below 0.3 m/s (the no-progress rules' "stopped") for 2 s

# dataviz palette (light surface)
SURF, INK, INK2, GRID = (252, 252, 251), (11, 11, 11), (82, 81, 78), (228, 227, 223)
BLUE, ORANGE, AQUA, VIOLET = (42, 120, 214), (235, 104, 52), (27, 175, 122), (74, 58, 167)
RED = (208, 59, 59)   # status: critical (the top-down videos' failure colour)
APPROACH_RGB = (0.93, 0.93, 0.90)


def rgbf(c):
    return tuple(v / 255.0 for v in c)


def read(p):
    return json.loads(Path(p).read_text())


def dump(p, v):
    Path(p).write_text(json.dumps(v, indent=1, allow_nan=False) + "\n")


def case_entry(case_key):
    man = read(MANIFEST)
    if case_key not in man:
        raise SystemExit(f"case {case_key} not in {MANIFEST} (have {', '.join(man)})")
    return man[case_key]


def arm_entry(case_key, arm):
    c = case_entry(case_key)
    for a in c["arms"]:
        if a["arm"] == arm:
            return c, a
    raise SystemExit(f"arm {arm} not in manifest case {case_key}")


def is_soil(case_key):
    return case_entry(case_key)["world"] == "crm"


def filmed_arms(case_key):
    return FILMED[case_entry(case_key)["world"]]


def case_title(case_key):
    return WORLD_TITLE[case_entry(case_key)["world"]]


def fmt_t(t):
    """Times are multiples of the 50 ms step: two decimals with one trailing zero dropped (12.85, 12.7, 34.0), exactly
    as the top-down videos print them."""
    s = f"{round(float(t) / DT) * DT:.2f}"
    return s[:-1] if s.endswith("0") else s


def stuck_onset(vx):
    """Start (s) of the final stretch below 0.3 m/s, the same rule as the top-down videos' 'stuck from'."""
    i = len(vx)
    while i > 0 and abs(float(vx[i - 1])) < STUCK_MPS:
        i -= 1
    return i * DT


def speed_txt(v, width=4):
    """Forward speed to one decimal, never '-0.0' (|v| < 0.05 prints as 0.0, as in the top-down videos)."""
    v = 0.0 if abs(v) < 0.05 else float(v)
    return f"{v:{width}.1f}"


def work_dir(case_key, arm):
    return WORK / f"{case_key}_{arm}"


def file_sha256(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def soil_case_entry(group):
    """Manifest entry for one soil (CRM) group, built the way the 'soil' entry was: recorded run of every arm through
    the run indexes (a drive whose branch route duplicates another arm's was driven once, under the index's
    'driven_as' name), the arm's branch route from its named picks, the approach route of its protocol.  The route
    files are checked against the file hashes the collector wrote into the recorded outcome."""
    iH = read(K1 / "a5/run_index_crm_pass2.json")
    i2 = read(K2 / "s2/idx/run_index_crm.json")
    i4 = read(K2 / "s4/run_index_s4.json")
    case_path = K1 / f"suite/cases/{group}.json"
    case = read(case_path)
    arms = []
    for arm, label, s, F in ARM_SPECS:
        if arm == "H3s":
            run = K1 / "a5/crm_pass2_runs" / iH[f"{group}__H_B"]["driven_as"]
            br = K1 / f"a5/picks_crm_H_named/routes/{group}__H_B.json"
            ap = K1 / f"a5/approach/{group}.json"
        elif arm == "HnG":
            run = K2 / "s4/runs" / i4[f"{group}__L0p5_HnG_G"]["driven_as"]
            br = K2 / f"s4/idx/picks_crm_L0p5_HnG_named/routes/{group}__L0p5_HnG_G.json"
            ap = K2 / f"a5data/approach_suite/{group}.json"
        else:
            run = K2 / "s2/runs_crm" / i2[f"{group}__{arm}_B"]["driven_as"]
            br = K2 / f"s2/idx/picks_crm_{arm}_named/routes/{group}__{arm}_B.json"
            ap = K2 / f"a5data/approach_suite/{group}.json"
        o = read(run / "outcome.json")
        blk = o["branch"]
        if o.get("case_id") != group or int(blk["branch_frame"]) != F:
            raise SystemExit(f"{group} {arm}: recorded run {run} is not this group/decision frame")
        if file_sha256(br) != blk["branch_route_sha256"] or file_sha256(ap) != blk["prefix_route_sha256"]:
            raise SystemExit(f"{group} {arm}: route files differ from the ones the recorded drive used")
        arms.append({"arm": arm, "label": label, **(H3S_EXTRA if arm == "H3s" else {}), "approach_s": s,
                     "branch_frame": F, "run_dir": str(run.relative_to(ROOT)),
                     "branch_route": str(br), "approach_route": str(ap), "status": o["status"],
                     "elapsed_s": o["elapsed_s"], "have_traj": (run / "trajectory.npz").exists()})
    return {"world": "crm", "group": group, "case": str(case_path), "arena": str((ROOT / case["arena"]).resolve()),
            "arms": arms}


def soil_candidates(args):
    """Rank soil groups whose RECORDED drives show the study's contrast: the old 3 s approach fails while the 1 s
    transformer and the final 0.5 s gradient-refined CNN-GRU both reach the goal.

    tier 0: the 1 s CNN-GRU also fails and both plain 0.5 s arms reach the goal; tier 1: the 1 s CNN-GRU reaches the
    goal, both 0.5 s arms too; tier 2: the 1 s CNN-GRU fails, a 0.5 s arm fails too; tier 3: the rest.
    Within a tier, groups rank by the success margin: the lowest forward speed any successful arm drove from 0.5 s
    after its decision to its end (a success that never slowed down is the least likely to flip on another GPU).
    Also reported: the failure margin, the shortest time any failed arm spent below 1 m/s after its decision (a long
    stall before bogging down is a clear failure), and the group already filmed (excluded with --exclude).

    Revised order (written as 'revised_order' after the first two screened groups, 0356 and 0244; the third, 0440,
    was already being driven, so the order was followed from the fourth screened group on): a bogged-down failure's
    end time is set by when the digging wheels cross the sinkage limit, which the local re-drives of the first two
    groups moved by -2.8 to +15.6 s (with 0440 as well: -2.8 to +23.6 s), whereas a drive stuck from early on that
    never bogs down ends by the blockage rule at a fixed time (34.00 s).  So: tier-1 groups whose failed old-protocol drive ended by the blockage rule, success margin at least
    1 m/s, ordered by that drive's recorded sinkage margin to the bogged-down limit (largest first)."""
    i2 = read(K2 / "s2/idx/run_index_crm.json")
    groups = sorted({v["group"] for v in i2.values()})
    iH, i4 = read(K1 / "a5/run_index_crm_pass2.json"), read(K2 / "s4/run_index_s4.json")
    F = {arm: f for arm, _, _, f in ARM_SPECS}
    rows = []
    for g in groups:
        runs = {"H3s": K1 / "a5/crm_pass2_runs" / iH[f"{g}__H_B"]["driven_as"],
                "HnG": K2 / "s4/runs" / i4[f"{g}__L0p5_HnG_G"]["driven_as"]}
        for arm in ("L1_Hn", "L1_X", "L0p5_Hn", "L0p5_X"):
            runs[arm] = K2 / "s2/runs_crm" / i2[f"{g}__{arm}_B"]["driven_as"]
        st = {a: read(r / "outcome.json") for a, r in runs.items()}
        ok = {a: st[a]["status"] == "goal_reached" for a in runs}
        if ok["H3s"] or not ok["L1_X"] or not ok["HnG"] or g in (args.exclude or []):
            continue
        succ_v, fail_stall, arms = [], [], {}
        for a in F:
            vx = np.load(runs[a] / "trajectory.npz")["state"][:, 0].astype(float)
            post = vx[F[a] + 10:]
            slow = np.abs(vx) < 0.3
            i = len(vx)
            while i > 0 and slow[i - 1]:
                i -= 1
            arms[a] = {"status": st[a]["status"], "elapsed_s": st[a]["elapsed_s"], "run_dir": str(runs[a].relative_to(ROOT)),
                       "max_wheel_sinkage_below_bmp_m": st[a]["crm"]["max_wheel_sinkage_below_bmp_m"],
                       "below_0p3mps_to_the_end_from_s": i * DT,
                       "min_speed_after_decision_mps": float(post.min()),
                       "time_below_1mps_after_decision_s": float((post < 1.0).sum() * DT),
                       "final_goal_distance_m": st[a]["final_goal_distance_m"]}
            (succ_v if ok[a] else fail_stall).append(arms[a]["min_speed_after_decision_mps"] if ok[a]
                                                    else arms[a]["time_below_1mps_after_decision_s"])
        both05 = ok["L0p5_Hn"] and ok["L0p5_X"]
        tier = (0 if both05 else 2) if not ok["L1_Hn"] else (1 if both05 else 3)
        rows.append({"group": g, "tier": tier, "success_margin_mps": min(succ_v), "failure_stall_s": min(fail_stall),
                     "arms": arms})
    rows.sort(key=lambda r: (r["tier"], -r["success_margin_mps"], r["group"]))
    paras = soil_candidates.__doc__.split("\n\n")
    rev = [r for r in rows if r["tier"] == 1 and r["arms"]["H3s"]["status"] == "prolonged_blockage_terminated"
           and r["success_margin_mps"] >= 1.0]
    rev.sort(key=lambda r: (r["arms"]["H3s"]["max_wheel_sinkage_below_bmp_m"], r["group"]))
    out = {"schema": "ci_soil_candidates_v1", "rule": paras[0].strip(),
           "ranking": " ".join(paras[1].split()), "revised_order_rule": " ".join(paras[2].split()),
           "bogged_down_limit_m": BOG_LIMIT_M, "excluded": args.exclude or [],
           "n_groups_scanned": len(groups), "n_candidates": len(rows),
           "tier_counts": {t: sum(r["tier"] == t for r in rows) for t in range(4)},
           "revised_order": [{"group": r["group"], "rank": rows.index(r) + 1,
                              "old_protocol_max_sinkage_m": r["arms"]["H3s"]["max_wheel_sinkage_below_bmp_m"],
                              "success_margin_mps": r["success_margin_mps"]} for r in rev],
           "candidates": rows}
    dump(Path(args.out), out)
    for k, r in enumerate(rows[:args.show]):
        a = r["arms"]
        print(f"{k + 1:3d} tier {r['tier']} {r['group']}  success margin {r['success_margin_mps']:.2f} m/s  "
              f"failure stall {r['failure_stall_s']:.1f} s  " +
              " ".join(f"{x}:{'ok' if a[x]['status'] == 'goal_reached' else 'FAIL'}{a[x]['elapsed_s']:.2f}" for x in F))
    print("revised order:", [(r["group"], r["rank"], round(r["old_protocol_max_sinkage_m"], 3)) for r in out["revised_order"]])
    print(out["tier_counts"], "->", args.out)


def screen(args):
    """Camera-free local re-drives of every arm of every case in the (screening) manifest vs the recorded drives."""
    man = read(MANIFEST)
    cand = read(args.candidates) if args.candidates else None
    rank = {r["group"]: (k + 1, r) for k, r in enumerate(cand["candidates"])} if cand else {}
    groups = []
    for key in man:
        C = man[key]
        film = filmed_arms(key)
        arms = []
        for a in C["arms"]:
            wd = work_dir(key, a["arm"])
            x = {"arm": a["arm"], "label": a["label"], "filmed_arm": a["arm"] in film, "recorded_run_dir": a["run_dir"]}
            if not (wd / "run_norender/outcome.json").exists():
                x["error"] = "not driven"
            else:
                c = compare_runs(ROOT / a["run_dir"], wd / "run_norender", int(a["branch_frame"]))
                x.update({"local_run_dir": str((wd / "run_norender").relative_to(ROOT)), "recorded": c["recorded"],
                          "local": c["local"], "same_status": c["same_status"], "elapsed_diff_s": c["elapsed_diff_s"],
                          "max_position_diff_m": c["max_position_diff_m"], "first_frame_over_0p1m": c["first_frame_over_0p1m"],
                          "terminal_position_diff_m": c["terminal_position_diff_m"],
                          "matches": bool(c["same_status"] and abs(c["elapsed_diff_s"]) <= args.tol_s + 1e-9)})
            arms.append(x)
        done = all("error" not in x for x in arms)
        g = {"key": key, "group": C["group"], "complete": done,
             "all_six_match": bool(done and all(x["matches"] for x in arms)),
             "filmed_three_match": bool(all(x.get("matches", False) for x in arms if x["filmed_arm"])),
             "max_abs_elapsed_diff_s": max((abs(x["elapsed_diff_s"]) for x in arms if "error" not in x), default=None),
             "arms": arms}
        if C["group"] in rank:
            g["candidate_rank"], r = rank[C["group"]]
            g.update({"tier": r["tier"], "success_margin_mps": r["success_margin_mps"], "failure_stall_s": r["failure_stall_s"]})
        groups.append(g)
    out = {"schema": "ci_soil_group_screen_v1", "written": time.strftime("%Y-%m-%d %H:%M"),
           "what": "Soil groups screened for the second filmed soil case: camera-free local re-drives (this workstation, "
                   "soil solver on CUDA) of all six arms, one at a time under the CRM lock, against the recorded cluster "
                   "drives. A group qualifies when every local outcome equals the recorded one and every elapsed time "
                   f"is within {args.tol_s:g} s; screened in candidate order, stopping at the first that qualifies.",
           "tol_s": args.tol_s, "candidates_file": str(Path(args.candidates).resolve().relative_to(ROOT)) if args.candidates else None,
           "candidate_rule": cand["rule"] if cand else None, "candidate_ranking": cand["ranking"] if cand else None,
           "revised_order_rule": cand.get("revised_order_rule") if cand else None,
           "screen_order": [g["group"] for g in groups], "order_note": args.note, "groups": groups}
    dump(WORK / "screen.json", out)
    for g in groups:
        print(f"{g['key']} {g['group']} rank {g.get('candidate_rank')} all six {g['all_six_match']} "
              f"filmed three {g['filmed_three_match']}")
        for x in g["arms"]:
            if "error" in x:
                print(f"    {x['arm']:8s} {x['error']}")
            else:
                print(f"    {x['arm']:8s} rec {x['recorded']['status']:30s} {x['recorded']['elapsed_s']:6.2f}  "
                      f"loc {x['local']['status']:30s} {x['local']['elapsed_s']:6.2f}  d {x['elapsed_diff_s']:+.2f} s  "
                      f"maxd {x['max_position_diff_m']:.2f} m  {'MATCH' if x['matches'] else 'differs'}")


def add_case(args):
    """Add one soil group to a manifest under a new key (existing keys are never changed)."""
    path = MANIFEST
    man = read(path) if path.exists() else {}
    if args.key in man:
        raise SystemExit(f"{args.key} already in {path}")
    man[args.key] = soil_case_entry(args.group)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(path, man)
    print(path, args.key, args.group, [(a["arm"], a["status"], round(a["elapsed_s"], 2)) for a in man[args.key]["arms"]])


def outcome_text(status, elapsed, stuck_from=None):
    """End-of-drive wording, identical to the top-down videos' for the three outcomes they show."""
    t = fmt_t(elapsed)
    if status == "goal_reached":
        return f"goal reached in {t} s"
    if status == "soil_breakthrough_terminated":
        return f"bogged down at {t} s: wheels dug through the soil"
    if status == "prolonged_blockage_terminated":
        since = f" from {fmt_t(stuck_from)} s" if stuck_from is not None else ""
        return f"stuck{since}; drive ended at {t} s (no progress)"
    if status == "rollover":
        return f"rolled over at {t} s"
    if status == "terrain_bounds_exit":
        return f"left the arena at {t} s"
    if status == "timeout":
        return f"did not reach the goal within {elapsed:.0f} s"
    return f"{status.replace('_', ' ')} at {t} s"


def brief_outcome(status, elapsed, stuck_from=None):
    """Short form for running text (header notes, subtitles)."""
    if status == "soil_breakthrough_terminated":
        return f"bogged down at {fmt_t(elapsed)} s"
    if status == "prolonged_blockage_terminated":
        since = f" from {fmt_t(stuck_from)} s" if stuck_from is not None else ""
        return f"stuck{since}, drive ended at {fmt_t(elapsed)} s"
    return outcome_text(status, elapsed, stuck_from)


def outcome_verb(status, elapsed, stuck_from=None, past=False):
    """'bogs down at 11.15 s' / 'reached the goal in 12.85 s' ... for sentences about one drive."""
    t = fmt_t(elapsed)
    if status == "goal_reached":
        return f"{'reached' if past else 'reaches'} the goal in {t} s"
    if status == "soil_breakthrough_terminated":
        return f"{'bogged' if past else 'bogs'} down at {t} s"
    if status == "prolonged_blockage_terminated":
        since = f" from {fmt_t(stuck_from)} s" if stuck_from is not None else ""
        return f"{'got' if past else 'gets'} stuck{since} (drive ended at {t} s)"
    return f"{'ended' if past else 'ends'} as follows: {outcome_text(status, elapsed, stuck_from)}"


def gap_phrase(e):
    """' and 4.1 m further back along the same path' when a same-outcome replay ends more than 1 m from the recorded
    end point (empty otherwise)."""
    if not e.get("same_status") or float(e["terminal_position_diff_m"]) <= 1.0:
        return ""
    if float(e["terminal_distance_to_recorded_path_m"]) < 0.5:
        # both end points lie on the recorded path: the gap is their distance, the direction from the arc length
        # (the arc length itself also counts the small back-and-forth of spinning wheels, so it is not the gap)
        g = float(e["terminal_along_recorded_path_diff_m"])
        return (f" and {float(e['terminal_position_diff_m']):.1f} m {'further back' if g < 0 else 'further ahead'} "
                "along the same path")
    return f" and {float(e['terminal_position_diff_m']):.1f} m from the recorded end point"


# ============================================================================================ rendering (drive side)
class Chase:
    """Markers, optional soil surface and the chase camera.  Visual shapes on the fixed ground body only."""

    def __init__(self, chrono, system, ground, tmap, case, approach, branch, branch_frame, frames_dir, render,
                 hmmwv, soil_terrain=None, soil_depth_m=0.24, soil_spacing_m=0.08):
        self.c, self.system, self.ground, self.tmap, self.case = chrono, system, ground, tmap, case
        self.approach, self.branch, self.F = approach, branch, int(branch_frame)
        self.frames_dir, self.render, self.hmmwv = Path(frames_dir), bool(render), hmmwv
        self.soil, self.depth, self.spacing = soil_terrain, float(soil_depth_m), float(soil_spacing_m)
        self.rows, self.timing, self.meta = [], [], {}
        self.yaw = None
        self.swapped = False
        self.dump_at = set()

    # ---- markers
    def _sphere(self, x, y, r, rgb, dz, emissive=0.55, visible=True):
        c = self.c
        sph = c.ChVisualShapeSphere(r)
        m = c.ChVisualMaterial()
        m.SetDiffuseColor(c.ChColor(*rgb))
        m.SetEmissiveColor(c.ChColor(*[emissive * v for v in rgb]))
        sph.SetMaterial(0, m)
        sph.SetVisible(visible)
        z = float(self.tmap.height(float(x), float(y))) + dz
        self.ground.AddVisualShape(sph, c.ChFramed(c.ChVector3d(float(x), float(y), z), c.QUNIT))
        return sph

    def _route(self, route, every_m, r, rgb, visible):
        shapes, last = [], -1e9
        for (x, y), s in zip(route["waypoints"], route["stations"]):
            if s - last >= every_m:
                shapes.append(self._sphere(x, y, r, rgb, 0.3, visible=visible))
                last = s
        return shapes

    def setup(self):
        t0 = time.time()
        self.approach_shapes = self._route(self.approach, 2.0, 0.11, APPROACH_RGB, True)
        self.branch_shapes = self._route(self.branch, 1.5, 0.14, rgbf(ORANGE), False)
        gx, gy = self.case["goal_xy"]
        rad = float(self.case.get("goal_radius_m", 2.5))
        for k in range(28):
            a = 2 * math.pi * k / 28
            self._sphere(gx + rad * math.cos(a), gy + rad * math.sin(a), 0.2, rgbf(VIOLET), 0.4, emissive=0.8)
        if self.soil is not None and self.render:
            self._soil_init()
        if self.render:
            import pychrono.sensor as sens
            from nedm.traverse.scene import SKY_RGB
            c = self.c
            self.manager = sens.ChSensorManager(self.system)
            self.manager.scene.SetAmbientLight(c.ChVector3f(0.35, 0.35, 0.38))
            self.manager.scene.AddDirectionalLight(c.ChColor(1.0, 0.95, 0.85), math.radians(40.0), math.radians(120.0))
            bg = sens.Background()
            bg.mode = sens.BackgroundMode_SOLID_COLOR
            bg.color_zenith = c.ChVector3f(*SKY_RGB)
            self.manager.scene.SetBackground(bg)
            self.cam = sens.ChCameraSensor(self.ground, 1000.0, c.ChFramed(c.ChVector3d(0, 0, 100.0), c.QuatFromAngleY(0.3)),
                                           W, H, HFOV, 2)
            self.cam.SetName("chase")
            self.cam.SetLag(0.0)
            self.cam.SetCollectionWindow(0.0)
            self.cam.PushFilter(sens.ChFilterRGBA8Access())
            self.manager.AddSensor(self.cam)
            self.last_launch = None
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.meta["setup_s"] = time.time() - t0

    # ---- soil surface from the particles
    def _particles(self):
        """Soil (fluid) markers only: the position array holds the fluid markers first, then the boundary (floor and
        side-wall) markers; the split is checked against the heightmap at set-up."""
        P = np.asarray(self.fluid.GetParticlePositionsNumpy(), np.float64).reshape(-1, 3)
        return P[:self.n_fluid]

    def _tops(self, P):
        """(cell ids, 4th-highest z) per occupied column (the lowest if it holds fewer): an untouched 2 x 2 column's
        top layer has four particles, and up to three thrown particles cannot raise it."""
        ix = np.floor((P[:, 0] - self.x0) / self.cell).astype(np.int64)
        iy = np.floor((P[:, 1] - self.y0) / self.cell).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        cid, z = iy[ok] * self.nx + ix[ok], P[ok, 2]
        order = np.lexsort((z, cid))
        cid, z = cid[order], z[order]
        uniq, start, count = np.unique(cid, return_index=True, return_counts=True)
        pick = start + np.maximum(count - TOP_K, 0)
        return uniq, z[pick]

    def _soil_init(self):
        c = self.c
        t0 = time.time()
        self.fluid = self.soil.GetFluidSystemSPH()
        self.n_fluid = int(self.fluid.GetNumFluidMarkers())
        P_all = np.asarray(self.fluid.GetParticlePositionsNumpy(), np.float64).reshape(-1, 3)
        P = P_all[:self.n_fluid]
        rest = P_all[self.n_fluid:]
        sp = self.spacing
        self.cell = 2 * sp
        self.x0, self.y0 = float(P[:, 0].min()) - sp / 2, float(P[:, 1].min()) - sp / 2
        self.nx = int(math.ceil((float(P[:, 0].max()) - self.x0) / self.cell))
        self.ny = int(math.ceil((float(P[:, 1].max()) - self.y0) / self.cell))
        cid, top = self._tops(P)
        Z = np.full(self.nx * self.ny, np.nan)
        Z[cid] = top
        self.xc = self.x0 + (np.arange(self.nx) + 0.5) * self.cell
        self.yc = self.y0 + (np.arange(self.ny) + 0.5) * self.cell
        XX, YY = np.meshgrid(self.xc, self.yc)
        bmp = self.tmap.height(XX.ravel(), YY.ravel())
        filled = np.isfinite(Z)
        self.offset = float(np.median(bmp[filled] - Z[filled]))
        resid = Z[filled] + self.offset - bmp[filled]
        # drawn surface = heightmap + (current column top - initial column top): smooth where untouched (the 8 cm
        # particle layers would otherwise draw as terraces), and every rut or berm is the particles' own displacement
        self.top0 = np.where(filled, Z, bmp - self.offset)
        self.B = bmp
        Z = bmp.copy()
        self.Z = Z
        # shading normals: from a lightly smoothed heightmap (the 8-bit BMP steps would otherwise show as stripes)
        # plus the unsmoothed gradient of the particle displacement, so ruts shade at full resolution
        Bs = bmp.reshape(self.ny, self.nx)
        k = np.array([1, 4, 6, 4, 1], float) / 16
        for _ in range(2):
            Bs = np.apply_along_axis(lambda r: np.convolve(np.pad(r, 2, mode="edge"), k, "valid"), 1, Bs)
            Bs = np.apply_along_axis(lambda r: np.convolve(np.pad(r, 2, mode="edge"), k, "valid"), 0, Bs)
        gy, gx = np.gradient(Bs, self.cell)
        self.gBx, self.gBy = gx.ravel(), gy.ravel()
        N = self._normals(self.gBx, self.gBy)
        self.meta["soil_surface"] = {
            "markers_total": int(len(P_all)), "fluid_markers_used": int(len(P)),
            "fluid_xyz_min": P.min(0).tolist(), "fluid_xyz_max": P.max(0).tolist(),
            "other_markers_xyz_min": rest.min(0).tolist() if len(rest) else None,
            "other_markers_xyz_max": rest.max(0).tolist() if len(rest) else None,
            "cell_m": self.cell, "grid": [self.nx, self.ny], "x0": self.x0, "y0": self.y0,
            "offset_to_heightmap_m": self.offset, "empty_columns_filled_from_heightmap": int((~filled).sum()),
            "untouched_surface_vs_heightmap_rmse_m": float(np.sqrt(np.mean(resid ** 2))),
            "untouched_surface_vs_heightmap_p99_abs_m": float(np.percentile(np.abs(resid), 99)),
            "rule": "drawn height = heightmap + (4th-highest particle of the 2x2-particle column now - the same at set-up); columns within 5 m of the chassis refreshed every frame; the rmse/p99 lines compare the raw initial column tops (+ median offset) with the heightmap"}
        mesh = c.ChTriangleMeshConnected()
        verts = mesh.GetCoordsVertices()
        V = c.ChVector3d
        for k in range(self.nx * self.ny):
            verts.push_back(V(float(XX.flat[k]), float(YY.flat[k]), float(Z[k])))
        normals = mesh.GetCoordsNormals()
        for k in range(self.nx * self.ny):
            normals.push_back(V(float(N[k, 0]), float(N[k, 1]), float(N[k, 2])))
        faces, nfaces = mesh.GetIndicesVertices(), mesh.GetIndicesNormals()
        V3 = c.ChVector3i
        nx = self.nx
        for j in range(self.ny - 1):
            for i in range(nx - 1):
                a = j * nx + i
                t1, t2 = V3(a, a + 1, a + nx + 1), V3(a, a + nx + 1, a + nx)
                faces.push_back(t1)
                faces.push_back(t2)
                nfaces.push_back(t1)
                nfaces.push_back(t2)
        self.verts, self.norms = mesh.GetCoordsVertices(), mesh.GetCoordsNormals()
        shape = c.ChVisualShapeTriangleMesh()
        shape.SetMesh(mesh)
        shape.SetMutable(True)
        mat = c.ChVisualMaterial()
        mat.SetDiffuseColor(c.ChColor(*SOIL_RGB))
        mat.SetRoughness(0.9)
        shape.AddMaterial(mat) if not shape.GetNumMaterials() else shape.SetMaterial(0, mat)
        self.ground.AddVisualShape(shape, c.ChFramed(c.VNULL, c.QUNIT))
        self.soil_shape, self.soil_mesh = shape, mesh
        self.nmoved = np.zeros(self.nx * self.ny, bool)
        self.meta["soil_surface"]["build_s"] = time.time() - t0

    @staticmethod
    def _normals(gx, gy):
        n = np.stack([-gx, -gy, np.ones_like(gx)], 1)
        return n / np.linalg.norm(n, axis=1, keepdims=True)

    def _soil_update(self, x, y):
        box = 5.0
        P = self._particles()
        m = (np.abs(P[:, 0] - x) < box + self.cell) & (np.abs(P[:, 1] - y) < box + self.cell)
        i0 = max(int((x - box - self.x0) / self.cell), 0)
        i1 = min(int((x + box - self.x0) / self.cell) + 1, self.nx)
        j0 = max(int((y - box - self.y0) / self.cell), 0)
        j1 = min(int((y + box - self.y0) / self.cell) + 1, self.ny)
        if i1 <= i0 or j1 <= j0:
            return 0
        cid, top = self._tops(P[m])
        J, I = np.meshgrid(np.arange(j0, j1), np.arange(i0, i1), indexing="ij")
        cells = (J * self.nx + I).ravel()
        new_top = self.top0[cells] - self.depth    # a column with no particle left: the bare floor
        pos = np.searchsorted(cid, cells)
        pos = np.clip(pos, 0, max(len(cid) - 1, 0))
        hit = (len(cid) > 0) & (cid[pos] == cells) if len(cid) else np.zeros(len(cells), bool)
        new_top = np.where(hit, top[pos], new_top)
        new = self.B[cells] + (new_top - self.top0[cells])
        changed = np.nonzero(np.abs(new - self.Z[cells]) > 1e-4)[0]
        V = self.c.ChVector3d
        for k in changed:
            cell = int(cells[k])
            self.verts[cell] = V(float(self.xc[cell % self.nx]), float(self.yc[cell // self.nx]), float(new[k]))
        self.Z[cells] = new
        if len(changed):
            # normals of the box (one cell of margin is inside the 5 m box already): smoothed-map slope + displacement slope
            Dg = (self.Z - self.B).reshape(self.ny, self.nx)[max(j0 - 1, 0):min(j1 + 1, self.ny), max(i0 - 1, 0):min(i1 + 1, self.nx)]
            gy, gx = np.gradient(Dg, self.cell)
            jj, ii = np.meshgrid(np.arange(max(j0 - 1, 0), min(j1 + 1, self.ny)), np.arange(max(i0 - 1, 0), min(i1 + 1, self.nx)), indexing="ij")
            cc = (jj * self.nx + ii).ravel()
            N = self._normals(self.gBx[cc] + gx.ravel(), self.gBy[cc] + gy.ravel())
            moved = np.abs(gx.ravel()) + np.abs(gy.ravel()) > 1e-6
            for k in np.nonzero(moved | self.nmoved[cc])[0]:
                self.norms[int(cc[k])] = V(float(N[k, 0]), float(N[k, 1]), float(N[k, 2]))
            self.nmoved[cc] = moved
        return int(len(changed))

    def _take(self, timeout_s):
        """The render launched by this Update: the first buffer whose launch count exceeds the previous one (the
        count starts at the camera's first due time, not at 1, because the manager is created after the settle)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            buf = self.cam.GetMostRecentRGBA8Buffer()
            if buf.HasData() and (self.last_launch is None or buf.LaunchedCount > self.last_launch):
                return np.ascontiguousarray(buf.GetRGBA8Data()[::-1, :, :3]), int(buf.LaunchedCount)
            time.sleep(0.0005)
        raise RuntimeError("chase camera frame never arrived")

    # ---- one recorded frame
    def frame(self, frame, pose, cz, extra):
        t0 = time.time()
        row = {"frame": int(frame), "t": frame * DT, "x": float(pose[0]), "y": float(pose[1]), "yaw": float(pose[2]),
               "z": float(cz), **extra}
        self.rows.append(row)
        if not self.render:
            return
        c = self.c
        if frame >= self.F and not self.swapped:
            for s in self.approach_shapes:
                s.SetVisible(False)
            for s in self.branch_shapes:
                s.SetVisible(True)
            self.manager.ReconstructScenes()
            self.swapped = True
            self.meta["scene_rebuilt_at_frame"] = int(frame)
        n_changed = self._soil_update(float(pose[0]), float(pose[1])) if self.soil is not None else 0
        if self.soil is not None and frame in self.dump_at:
            P = self._particles()
            m = (np.abs(P[:, 0] - pose[0]) < 8) & (np.abs(P[:, 1] - pose[1]) < 8)
            np.savez_compressed(self.frames_dir.parent / f"particles_{frame:05d}.npz", P=P[m].astype(np.float32),
                                pose=np.asarray(pose), cz=cz)
        yaw = float(pose[2]) if self.yaw is None else self.yaw
        yaw += 0.35 * ((float(pose[2]) - yaw + math.pi) % (2 * math.pi) - math.pi)
        self.yaw = yaw
        cx, cy = pose[0] - BACK_M * math.cos(yaw), pose[1] - BACK_M * math.sin(yaw)
        lim = lambda q: float(np.clip(q, -39.9, 39.9))
        ground = max(float(self.tmap.height(lim(cx), lim(cy))), float(self.tmap.height(lim(pose[0]), lim(pose[1]))))
        z = max(cz + UP_M, ground + 3.5)
        pitch = math.atan2(z - (cz + 0.5), BACK_M)
        self.cam.SetOffsetPose(c.ChFramed(c.ChVector3d(float(cx), float(cy), float(z)),
                                          c.QuatFromAngleZ(yaw) * c.QuatFromAngleY(pitch)))
        self.manager.Update()
        rgb, launch = self._take(120.0)
        if self.last_launch is not None and launch != self.last_launch + 1:
            self.meta.setdefault("launch_gaps", []).append([int(frame), int(self.last_launch), int(launch)])
        self.last_launch = launch
        from PIL import Image
        Image.fromarray(rgb).save(self.frames_dir / f"f_{frame:05d}.jpg", quality=92)
        self.timing.append({"frame": int(frame), "render_s": time.time() - t0, "soil_vertices_changed": n_changed})


def sinkage_extra(vehicle, tmap, wheel_specs, radii):
    worst = -1e9
    for name, axle, side in wheel_specs:
        p = vehicle.GetSpindlePos(axle, side)
        worst = max(worst, radii[name] - (float(p.z) - float(tmap.height(float(p.x), float(p.y)))))
    return float(worst)


# ============================================================================================ drive
def drive(args):
    case_key, arm = args.case, args.arm
    c, a = arm_entry(case_key, arm)
    wd = work_dir(case_key, arm)
    run_dir = wd / ("run_norender" if args.no_render else "run")
    frames_dir = wd / "frames"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    if not args.no_render:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True)
    approach = read(a["approach_route"])
    branch = read(a["branch_route"])
    F = int(a["branch_frame"])
    state = {}
    started = time.time()

    if is_soil(case_key):
        import crm_collect_ext as CX
        cfg = read(CRM_CONFIG)

        def scene_hook(chrono, veh, hmmwv, system, terrain, tmap, route, case):
            from nedm.hmmwv_data import WHEEL_SPECS
            hmmwv.SetChassisVisualizationType(chrono.VisualizationType_MESH)
            hmmwv.SetWheelVisualizationType(chrono.VisualizationType_MESH)
            hmmwv.SetTireVisualizationType(chrono.VisualizationType_MESH)
            ch = Chase(chrono, system, terrain.GetGroundBody(), tmap, case, route, branch, F, frames_dir,
                       not args.no_render, hmmwv, soil_terrain=terrain, soil_depth_m=cfg["depth_m"],
                       soil_spacing_m=cfg["spacing_m"])
            ch.dump_at = set(args.dump_at or [])
            ch.setup()
            if ch.dump_at:
                P0 = ch._particles() if ch.soil is not None and ch.render else None
                if P0 is not None:
                    np.savez_compressed(frames_dir.parent / "particles_initial.npz", P=P0.astype(np.float32))
            vehicle = hmmwv.GetVehicle()
            radii = {n: float(vehicle.GetTire(ax, sd).GetRadius()) for n, ax, sd in WHEEL_SPECS}
            state.update(chase=ch, tmap=tmap, vehicle=vehicle, specs=WHEEL_SPECS, radii=radii)

        def frame_hook(frame, row, state=None, pose=None, action=None, desired_speed=None, _s=state):
            slip = max(abs(float(row[f"{n}_slip_ratio"])) for n in ("tire_fl", "tire_fr", "tire_rl", "tire_rr"))
            extra = {"vx": float(state[0]), "steer": float(action[0]), "throttle": float(action[1]),
                     "brake": float(action[2]), "desired": float(desired_speed), "max_slip": slip,
                     "sinkage_m": sinkage_extra(_s["vehicle"], _s["tmap"], _s["specs"], _s["radii"])}
            _s["chase"].frame(frame, pose, float(row["pos_z_m"]), extra)
            if args.max_frames and frame + 1 >= args.max_frames:
                raise SystemExit(f"--max-frames {args.max_frames} reached")

        argv = ["--source-root", str(ROOT), "--case", c["case"], "--route", a["approach_route"], "--out", str(run_dir),
                "--chrono-data", str(CRM_CHRONO_DATA), "--crm-config", str(CRM_CONFIG), "--horizon-s", "120",
                "--mode", "branch", "--branch-frame", str(F), "--branch-route", a["branch_route"]]
        try:
            CX.main(argv, scene_hook=scene_hook, frame_hook=frame_hook)
        except SystemExit as e:
            if not args.max_frames:
                raise
            print(e)
    else:
        import gen_collect
        import gen_collect_ext as GX
        from nedm.traverse.terrain import TerrainMap
        tmap = TerrainMap.from_dir((ROOT / read(c["case"])["arena"]).resolve())
        case = read(c["case"])
        original_make_observer = gen_collect.make_observer

        class ChaseObserver:
            def __init__(self, inner):
                self.inner = inner

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def on_frame(self, scene, frame, state17, pose3, action3, *, command_context=None):
                r = self.inner.on_frame(scene, frame, state17, pose3, action3, command_context=command_context)
                if "chase" not in state:
                    import pychrono as chrono
                    ch = Chase(chrono, scene.system, scene.patch_body, tmap, case, approach, branch, F, frames_dir,
                               not args.no_render, scene.hmmwv)
                    ch.setup()
                    state["chase"] = ch
                cz = float(scene.hmmwv.GetChassis().GetBody().GetPos().z)
                cc = command_context or {}
                extra = {"vx": float(state17[0]), "steer": float(action3[0]), "throttle": float(action3[1]),
                         "brake": float(action3[2]), "desired": float(cc.get("desired_speed_mps", float("nan")))}
                state["chase"].frame(frame, pose3, cz, extra)
                if args.max_frames and frame + 1 >= args.max_frames:
                    raise SystemExit(f"--max-frames {args.max_frames} reached")
                return r

        gen_collect.make_observer = lambda out, case_path: ChaseObserver(original_make_observer(out, case_path))
        argv = ["--source-root", str(ROOT), "--case", c["case"], "--route", a["approach_route"], "--out", str(run_dir),
                "--chrono-data", str(RIGID_CHRONO_DATA), "--horizon-s", "120", "--mode", "branch",
                "--branch-frame", str(F), "--branch-route", a["branch_route"], "--local"]
        gargs = GX.parser().parse_args(argv)
        GX.validate_mode_args(gargs)
        try:
            GX.run_episode(gargs)
        except SystemExit as e:
            if not args.max_frames:
                raise
            print(e)

    ch = state.get("chase")
    import pychrono
    meta = {"schema": "ci_chase_drive_v1", "case": case_key, "arm": arm, "label": a["label"], "render": not args.no_render,
            "run_dir": str(run_dir.relative_to(ROOT)), "branch_frame": F, "wall_s": time.time() - started,
            "host": os.uname().nodename, "python": sys.executable, "pychrono": pychrono.__file__,
            "chrono_data": str(CRM_CHRONO_DATA if is_soil(case_key) else RIGID_CHRONO_DATA),
            "max_frames": args.max_frames, "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
            **(ch.meta if ch else {})}
    if ch and ch.timing:
        rs = np.asarray([t["render_s"] for t in ch.timing])
        meta["render_s_mean"], meta["render_s_max"] = float(rs.mean()), float(rs.max())
    tag = "_norender" if args.no_render else ""
    dump(wd / f"drive{tag}.json", meta)
    if ch:
        dump(wd / f"overlay{tag}.json", ch.rows)
        if ch.timing:
            dump(wd / "render_timing.json", ch.timing)
    out = run_dir / "outcome.json"
    if out.exists():
        o = read(out)
        print(json.dumps({"case": case_key, "arm": arm, "status": o["status"], "elapsed_s": o["elapsed_s"],
                          "recorded": [a["status"], a["elapsed_s"]], "wall_s": meta["wall_s"]}))


# ============================================================================================ reproduction check
def to_polyline(P, Q):
    """Distance of each point of P (n, 2) to the polyline through Q (m, 2), and the arc length along Q of the
    closest point."""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    if len(Q) == 1:
        return np.linalg.norm(P - Q[0], axis=1), np.zeros(len(P))
    A, AB = Q[:-1], np.diff(Q, axis=0)
    L2 = np.maximum((AB ** 2).sum(1), 1e-18)
    u = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0.0, 1.0)
    D = np.linalg.norm(P[:, None, :] - (A[None] + u[..., None] * AB[None]), axis=2)
    j = D.argmin(1)
    i = np.arange(len(P))
    seg = np.sqrt((AB ** 2).sum(1))
    station = np.concatenate([[0.0], np.cumsum(seg)])[j] + u[i, j] * seg[j]
    return D[i, j], station


def compare_runs(rec_dir, loc_dir, F):
    ro, lo = read(Path(rec_dir) / "outcome.json"), read(Path(loc_dir) / "outcome.json")
    rz, lz = np.load(Path(rec_dir) / "trajectory.npz"), np.load(Path(loc_dir) / "trajectory.npz")
    rp, lp = np.asarray(rz["pose"])[:, :2], np.asarray(lz["pose"])[:, :2]
    n = min(len(rp), len(lp))
    d = np.linalg.norm(rp[:n] - lp[:n], axis=1)
    rt, lt = np.asarray(rz["terminal_pose"])[:2], np.asarray(lz["terminal_pose"])[:2]
    if len(rp) == len(lp) and abs(float(lo["elapsed_s"]) - float(ro["elapsed_s"])) < 1e-6:
        # equal end times: the two terminal poses are one more pair at equal times (index n = the terminal frame), so
        # the max below is the largest distance at equal times over the whole drive, end point included
        d = np.append(d, np.linalg.norm(rt - lt))
    over = np.nonzero(d > 0.1)[0]
    rpath, lpath = np.vstack([rp, rt[None]]), np.vstack([lp, lt[None]])   # the driven paths, terminal pose included
    dpath, spath = to_polyline(lpath, rpath)
    rvx = np.concatenate([np.asarray(rz["state"])[:, 0], [np.asarray(rz["terminal_state"])[0]]])
    lvx = np.concatenate([np.asarray(lz["state"])[:, 0], [np.asarray(lz["terminal_state"])[0]]])
    stuck = "prolonged_blockage_terminated"
    return {
        "recorded": {"status": ro["status"], "elapsed_s": ro["elapsed_s"], "frames": int(len(rp))},
        "local": {"status": lo["status"], "elapsed_s": lo["elapsed_s"], "frames": int(len(lp)), "wall_s": lo.get("wall_s")},
        "same_status": ro["status"] == lo["status"],
        "elapsed_diff_s": round(float(lo["elapsed_s"]) - float(ro["elapsed_s"]), 6),
        "max_position_diff_m": float(d.max()),
        "max_position_diff_frame": int(d.argmax()),
        "position_diff_at_decision_m": float(d[F]) if F < n else None,
        "mean_position_diff_m": float(d.mean()),
        "first_frame_over_0p1m": int(over[0]) if len(over) else None,
        "terminal_position_diff_m": float(np.linalg.norm(rt - lt)),
        "compared_frames": int(n),
        # geometric deviation regardless of timing: each local position's distance to the recorded path line (the
        # segments between the recorded 50 ms positions, terminal pose included)
        "max_distance_to_recorded_path_m": float(dpath.max()),
        "terminal_distance_to_recorded_path_m": float(dpath[-1]),
        # where the local drive ended along the recorded path, minus the recorded path's length (negative = behind)
        "terminal_along_recorded_path_diff_m": float(spath[-1] - np.linalg.norm(np.diff(rpath, axis=0), axis=1).sum()),
        "recorded_stuck_from_s": stuck_onset(rvx) if ro["status"] == stuck else None,
        "local_stuck_from_s": stuck_onset(lvx) if lo["status"] == stuck else None,
        "identical_trajectory": bool(len(rp) == len(lp) and float(d.max()) == 0.0),
    }


def repro(args):
    entries, others = [], []
    man = read(MANIFEST)
    for case_key in man:
        arms = filmed_arms(case_key)
        if not any(work_dir(case_key, a["arm"]).exists() for a in man[case_key]["arms"]):
            continue   # no local re-drive of this case yet
        for a in man[case_key]["arms"]:   # arms without a video: camera-free local re-drive only, if one was made
            wd = work_dir(case_key, a["arm"])
            if a["arm"] in arms or not (wd / "run_norender/outcome.json").exists():
                continue
            o = {"case": case_key, "group": man[case_key]["group"], "arm": a["arm"], "label": a["label"], "filmed": False,
                 "local_run_dir": str((wd / "run_norender").relative_to(ROOT)),
                 **compare_runs(ROOT / a["run_dir"], wd / "run_norender", int(a["branch_frame"]))}
            o["differs"] = not (o["same_status"] and abs(o["elapsed_diff_s"]) < 1e-6)
            others.append(o)
        for arm in arms:
            c, a = arm_entry(case_key, arm)
            wd = work_dir(case_key, arm)
            e = {"case": case_key, "group": c["group"], "arm": arm, "label": a["label"], "filmed": True,
                 "recorded_run_dir": a["run_dir"], "branch_frame": a["branch_frame"],
                 "local_run_dir": str((wd / "run").relative_to(ROOT))}
            if not (wd / "run/outcome.json").exists():
                e["error"] = "no local filmed re-drive"
                entries.append(e)
                continue
            cmp_ = compare_runs(ROOT / a["run_dir"], wd / "run", int(a["branch_frame"]))
            e.update(cmp_)
            e["differs"] = not (cmp_["same_status"] and abs(cmp_["elapsed_diff_s"]) < 1e-6)
            rec_txt = outcome_text(cmp_["recorded"]["status"], cmp_["recorded"]["elapsed_s"], cmp_["recorded_stuck_from_s"])
            loc_txt = outcome_text(cmp_["local"]["status"], cmp_["local"]["elapsed_s"], cmp_["local_stuck_from_s"])
            md = cmp_["max_position_diff_m"]
            # "within": rounded up, so the stated distance is an upper bound (the largest distance at equal times)
            dist = f"{math.ceil(100 * md - 1e-9):.0f} cm" if md < 1.0 else f"{math.ceil(10 * md - 1e-9) / 10:.1f} m"
            rs, ls_ = cmp_["recorded_stuck_from_s"], cmp_["local_stuck_from_s"]
            if not e["differs"] and rs is not None and ls_ is not None and fmt_t(rs) != fmt_t(ls_):
                # same outcome and end time, but the stuck stretch starts at another time here: name both times, so
                # the footer agrees with the header's and end card's "stuck from" (the replay's own)
                same = (f" (stuck from {fmt_t(rs)} s there, {fmt_t(ls_)} s here; drive ended at "
                        f"{fmt_t(cmp_['recorded']['elapsed_s'])} s, no progress).")
            else:
                same = f": {rec_txt}."
            e["overlay_note"] = ("3D replay, re-simulated on a different computer. Original recorded run: "
                                 f"{rec_txt}." if e["differs"] else
                                 "3D replay, re-simulated on a different computer. Same outcome and end time as the "
                                 f"original recorded run{same} Path within {dist} of it.")
            e["local_outcome_text"], e["recorded_outcome_text"] = loc_txt, rec_txt
            e["header_note"] = recorded_note(e)
            if (wd / "run_norender/outcome.json").exists():
                nr = compare_runs(wd / "run_norender", wd / "run", int(a["branch_frame"]))
                e["filmed_vs_camera_free_local"] = {k: nr[k] for k in ("same_status", "elapsed_diff_s", "max_position_diff_m",
                                                                         "terminal_position_diff_m", "identical_trajectory")}
                e["filmed_vs_camera_free_local"]["camera_free"] = nr["recorded"]
            if (wd / "drive.json").exists():
                dj = read(wd / "drive.json")
                e["drive"] = {k: dj.get(k) for k in ("wall_s", "render_s_mean", "host", "pychrono", "chrono_data", "soil_surface",
                                                     "scene_rebuilt_at_frame")}
            entries.append(e)
    out = {"schema": "ci_chase_reproduction_v1", "written": time.strftime("%Y-%m-%d %H:%M"),
           "what": "Local re-drives (luffy, RTX 5090; soil solver on CUDA, rigid physics on this CPU) of the recorded "
                   "cluster drives (AMD); on screen they are called 3D replays re-simulated on a different computer. "
                   "Per drive: status, elapsed time, position difference at equal times (per recorded 50 ms frame, "
                   "plus the terminal poses when both drives end at the same time; also reflects speed differences), "
                   "max_distance_to_recorded_path_m = the largest distance of the "
                   "local positions from the recorded path line (the segments between the recorded 50 ms positions, "
                   "terminal pose included; geometry only), terminal_distance_to_recorded_path_m and "
                   "terminal_along_recorded_path_diff_m = where the local drive ended relative to the recorded path "
                   "(arc length of its closest point minus the recorded path length; negative = short of the "
                   "recorded end), and the start of the final stretch below 0.3 m/s for drives stopped by the "
                   "no-progress rule (recorded_ / local_stuck_from_s). 'differs' = status or elapsed time differs; "
                   "then the video header states the recorded outcome (header_note).",
           "entries": entries,
           "other_arms_camera_free": others}
    if (WORK / "screen/screen.json").exists():   # the soil-group screen that picked the second soil case
        out["soil_group_screen"] = read(WORK / "screen/screen.json")
    dump(VID / "chase_reproduction.json", out)
    for o in others:
        print(f"  (no video) {o['case']:5s} {o['arm']:7s} rec {o['recorded']['status']:30s} {o['recorded']['elapsed_s']:6.2f}  "
              f"loc {o['local']['status']:30s} {o['local']['elapsed_s']:6.2f}  maxd {o['max_position_diff_m']:.3f} m")
    for e in entries:
        if "error" in e:
            print(e["case"], e["arm"], e["error"])
        else:
            print(f"{e['case']:5s} {e['arm']:7s} rec {e['recorded']['status']:30s} {e['recorded']['elapsed_s']:6.2f}  "
                  f"loc {e['local']['status']:30s} {e['local']['elapsed_s']:6.2f}  maxd {e['max_position_diff_m']:.3f} m  "
                  f"dF {e['position_diff_at_decision_m']}  differs {e['differs']}"
                  + (f"  cam-vs-nocam maxd {e['filmed_vs_camera_free_local']['max_position_diff_m']:.4f}"
                     if "filmed_vs_camera_free_local" in e else ""))


# ============================================================================================ overlay / compose
def font(size, bold=False):
    from PIL import ImageFont
    f = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(f, size)


class Minimap:
    """Top view (x right, y up) around the start, goal and routes: hillshaded heightmap in greys, the straight
    approach (dashed, secondary ink), the planner's route (orange, from the decision), the driven track (blue), the
    vehicle (ink) and the goal circle (violet)."""

    def __init__(self, case, approach, branch, size_px, rows_list):
        from PIL import Image
        from nedm.traverse.terrain import TerrainMap
        tmap = TerrainMap.from_dir((ROOT / case["arena"]).resolve())
        pts = [np.asarray(approach["waypoints"])[:, :2], np.asarray(branch["waypoints"])[:, :2],
               np.asarray([case["layout"]["start_xy"], case["goal_xy"]])]
        pts += [np.asarray([[r["x"], r["y"]] for r in rows]) for rows in rows_list if rows]
        P = np.concatenate(pts)
        lo, hi = P.min(0) - 6.0, P.max(0) + 6.0
        span = float(max(hi - lo))
        mid = (lo + hi) / 2
        self.x0, self.y1, self.span, self.px = mid[0] - span / 2, mid[1] + span / 2, span, size_px
        n = size_px
        xs = self.x0 + (np.arange(n) + 0.5) / n * span
        ys = self.y1 - (np.arange(n) + 0.5) / n * span
        XX, YY = np.meshgrid(xs, ys)
        h = tmap.height(XX, YY)
        gy, gx = np.gradient(h, -span / n, span / n)       # image rows run toward -y
        az, el = math.radians(135.0), math.radians(45.0)   # light from the north-west
        L = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        shade = np.clip((-gx * L[0] - gy * L[1] + L[2]) / np.sqrt(gx ** 2 + gy ** 2 + 1), 0, 1)
        base = 0.48 + 0.47 * shade
        img = (np.stack([base] * 3, -1) * np.array(SURF) / 255.0 * 255).clip(0, 255).astype(np.uint8)
        self.base = Image.fromarray(img)
        self.case, self.approach, self.branch = case, approach, branch

    def xy(self, x, y):
        return ((x - self.x0) / self.span * self.px, (self.y1 - y) / self.span * self.px)

    def draw(self, rows_upto, decided, scale=1.0):
        from PIL import ImageDraw
        im = self.base.copy()
        d = ImageDraw.Draw(im)
        w = max(1, int(round(2 * scale)))
        ap = [self.xy(x, y) for x, y in np.asarray(self.approach["waypoints"])[:, :2]]
        for k in range(len(ap) - 1):   # dashed straight approach
            if (k // 3) % 2 == 0:
                d.line([ap[k], ap[k + 1]], fill=INK2, width=w)
        if decided:
            d.line([self.xy(x, y) for x, y in np.asarray(self.branch["waypoints"])[:, :2]], fill=ORANGE, width=w + 1)
        gx, gy = self.xy(*self.case["goal_xy"])
        r = float(self.case.get("goal_radius_m", 2.5)) / self.span * self.px
        d.ellipse([gx - r, gy - r, gx + r, gy + r], outline=VIOLET, width=w + 1)
        tr = [self.xy(rw["x"], rw["y"]) for rw in rows_upto]
        if len(tr) > 1:
            d.line(tr, fill=BLUE, width=w + 1)
        if tr:
            x, y = tr[-1]
            rr = 4 * scale
            d.ellipse([x - rr - 1.5, y - rr - 1.5, x + rr + 1.5, y + rr + 1.5], fill=SURF)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=INK)
        d.rectangle([0, 0, self.px - 1, self.px - 1], outline=GRID, width=1)
        return im


def legend(d, x, y, f, scale=1.0):
    items = [("driven", BLUE, "solid"), ("planner's route", ORANGE, "solid"), ("straight approach", INK2, "dash"),
             ("goal", VIOLET, "ring")]
    for k, (txt, col, kind) in enumerate(items):
        yy = y + k * 17 * scale
        if kind == "solid":
            d.line([x, yy + 7 * scale, x + 18 * scale, yy + 7 * scale], fill=col, width=int(3 * scale))
        elif kind == "dash":
            d.line([x, yy + 7 * scale, x + 6 * scale, yy + 7 * scale], fill=col, width=int(2 * scale))
            d.line([x + 11 * scale, yy + 7 * scale, x + 18 * scale, yy + 7 * scale], fill=col, width=int(2 * scale))
        else:
            d.ellipse([x + 4 * scale, yy + 1 * scale, x + 14 * scale, yy + 11 * scale], outline=col, width=int(2 * scale))
        d.text((x + 24 * scale, yy), txt, fill=INK, font=f)


def load_arm_bundle(case_key, arm, rep):
    c, a = arm_entry(case_key, arm)
    wd = work_dir(case_key, arm)
    rows = read(wd / "overlay.json")
    frames = sorted((wd / "frames").glob("f_*.jpg"))
    if len(frames) != len(rows):
        raise SystemExit(f"{case_key} {arm}: {len(frames)} frames vs {len(rows)} overlay rows")
    e = next((x for x in rep["entries"] if x["case"] == case_key and x["arm"] == arm), None)
    if e is None or "error" in e:
        raise SystemExit(f"{case_key} {arm}: run `repro` first")
    return {"case": read(c["case"]), "entry": a, "rows": rows, "frames": frames, "rep": e,
            "approach": read(a["approach_route"]), "branch": read(a["branch_route"]), "F": int(a["branch_frame"]),
            "approach_s": float(a["approach_s"])}


def phase_text(b, k):
    F, s = b["F"], b["approach_s"]
    rows = b["rows"]
    j = k
    while j >= 0 and abs(rows[j]["vx"]) < STUCK_MPS:
        j -= 1
    j += 1                                             # first frame of the current stretch below 0.3 m/s
    if j <= k and rows[k]["t"] - rows[j]["t"] >= STUCK_AFTER_S - 1e-9:
        thr = float(np.mean([r["throttle"] for r in rows[j:k + 1]]))
        moved = math.hypot(rows[k]["x"] - rows[j]["x"], rows[k]["y"] - rows[j]["y"])
        return (f"Stuck since {fmt_t(rows[j]['t'])} s: {'full throttle, ' if thr >= 0.9 else ''}"
                f"{'no progress' if moved < 0.5 else 'barely moving'}"), RED
    if k < F:
        return f"Approach: driving straight at the goal; the planner decides at {s:g} s", INK2
    return f"Planner in control since {s:g} s, following its chosen route", ORANGE


def rounded(d, box, fill, r=10, outline=None):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline)


def ffmpeg(frames_glob_dir, pattern, out, fps=FPS):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_glob_dir / pattern),
           "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)


def recorded_note(e):
    """Header line for a local re-drive that differs from the recorded drive (None if it does not): how it differs,
    in plain words (the viewer is not told about machines beyond 'a different computer')."""
    if not e["differs"]:
        return None
    rec, loc = e["recorded"], e["local"]
    txt = (f"Original recorded run: {brief_outcome(rec['status'], rec['elapsed_s'], e.get('recorded_stuck_from_s'))}; "
           "this 3D replay was re-simulated on a different computer and ")
    if e["same_status"]:
        dt = float(e["elapsed_diff_s"])
        when = f"{fmt_t(abs(dt))} s {'later' if dt > 0 else 'earlier'}" if abs(dt) > 1e-9 else "at the same time"
        return txt + f"ends the same way, {when}{gap_phrase(e)}"
    return txt + instead_phrase(loc["status"], loc["elapsed_s"], e.get("local_stuck_from_s"))


def instead_phrase(status, elapsed, stuck_from=None):
    """'bogs down instead, at 11.15 s' for a replay whose outcome is not the recorded one."""
    t = fmt_t(elapsed)
    if status == "soil_breakthrough_terminated":
        return f"bogs down instead, at {t} s"
    if status == "goal_reached":
        return f"reaches the goal instead, in {t} s"
    if status == "prolonged_blockage_terminated":
        since = f", from {fmt_t(stuck_from)} s" if stuck_from is not None else ""
        return f"gets stuck instead{since} (drive ended at {t} s)"
    return f"ends differently: {outcome_text(status, elapsed, stuck_from)}"


def cap(t):
    """First letter upper case, the rest untouched (str.capitalize would lower 'CNN-GRU')."""
    return t[:1].upper() + t[1:]


NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def side_summary(case_key, B):
    """Subtitle of a side-by-side video: how each replay compares with its recording, and where the recordings are."""
    opp = [b for b in B if not b["rep"]["same_status"]]
    rest = [b for b in B if b["rep"]["same_status"]]
    parts = []
    for b in opp:
        e = b["rep"]
        parts.append(cap(f"{WHO[b['entry']['arm']]} "
                         f"{outcome_verb(e['local']['status'], e['local']['elapsed_s'], e.get('local_stuck_from_s'))} "
                         "in this replay but "
                         f"{outcome_verb(e['recorded']['status'], e['recorded']['elapsed_s'], e.get('recorded_stuck_from_s'), past=True)}"
                         " in the recording."))
    if rest:
        if all(not b["rep"]["differs"] for b in rest):
            md = [float(b["rep"]["max_position_diff_m"]) for b in rest]
            cm = sorted(math.ceil(100 * m - 1e-9) for m in md)
            span = f"{cm[0]}-{cm[-1]} cm" if cm[0] != cm[-1] else f"{cm[0]} cm"
            who = (f"All {NUM.get(len(rest), len(rest))} replays" if not opp else
                   f"The other {NUM.get(len(rest), len(rest))}" if len(rest) > 1 else cap(WHO[rest[0]['entry']['arm']]))
            parts.append(f"{who} end the same way as recorded, at the same times, on paths within {span} of the "
                         "recorded ones." if max(md) < 1.0 else
                         f"{who} end the same way as recorded, at the same times.")
        else:
            bits = []
            for b in rest:
                e = b["rep"]
                dt = float(e["elapsed_diff_s"])
                when = (f"{fmt_t(abs(dt))} s {'later' if dt > 0 else 'earlier'}" if abs(dt) > 1e-9 else
                        f"at the same {fmt_t(e['local']['elapsed_s'])} s")
                bits.append(f"{WHO[b['entry']['arm']]} {when}{gap_phrase(e)}")
            who = (f"The other {NUM.get(len(rest), len(rest))}" if opp and len(rest) > 1 else
                   "The replays" if not opp else "The other one")
            parts.append(f"{who} end the same way as recorded: " + ", ".join(bits) + ".")
    parts.append(f"The recorded drives are in compare_{case_key}.mp4.")
    return " ".join(parts)


def compose(args):
    from PIL import Image, ImageDraw
    rep = read(VID / "chase_reproduction.json")
    b = load_arm_bundle(args.case, args.arm, rep)
    e, rows = b["rep"], b["rows"]
    tmp = work_dir(args.case, args.arm) / "composed"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    mm = Minimap(b["case"], b["approach"], b["branch"], 230, [rows])
    f_title, f_body, f_small, f_big = font(24, True), font(18), font(15), font(34, True)
    soil = is_soil(args.case)
    loc_status, loc_elapsed = e["local"]["status"], e["local"]["elapsed_s"]
    if e.get("header_note", recorded_note(e)) != recorded_note(e):
        raise SystemExit("chase_reproduction.json is stale: run `repro` first")
    n = len(rows)
    hold = int(3.0 * FPS)
    note = recorded_note(e)
    note_col = RED if not e.get("same_status", True) else INK2   # red only for a different outcome
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    note_lines = wrap(probe, note, f_body, 960) if note else []
    # the arm's footnote (the 3 s arm: which training run it drives) sits under the title it explains
    fn_lines = wrap(probe, b["entry"]["footnote"], f_small, 960) if b["entry"].get("footnote") else []
    dy = 20 * len(fn_lines) + (6 if fn_lines else 0)
    head_bot = 150 + dy + 28 * len(note_lines)     # the header grows by one line per footnote / note line
    ban0 = head_bot + 20                           # the decision banner sits below the header
    # a drive stopped by the no-progress rule: when that rule can fire at the earliest (collector's stop policy)
    req = work_dir(args.case, args.arm) / "run/collection_request.json"
    earliest = (read(req).get("stop_policy") or {}).get("earliest_possible_stop_s") if req.exists() else None
    for k in range(n + hold):
        kk = min(k, n - 1)
        r = rows[kk]
        im = Image.open(b["frames"][kk]).convert("RGB")
        d = ImageDraw.Draw(im, "RGBA")
        # header panel
        t_show = r["t"] if k < n else loc_elapsed
        # the clock and the rest are drawn at fixed positions, so the readout does not shift when the clock's width
        # changes (12.7 s / 12.75 s)
        t_part = f"time {fmt_t(t_show)} s"
        rest = f"speed {speed_txt(r['vx'])} m/s"
        if soil:
            rest += f"     wheel sinkage {max(r['sinkage_m'], 0.0):.2f} m (bogged-down limit {BOG_LIMIT_M:.2f} m)"
        x_rest = 32 + d.textlength("time 88.88 s     ", font=f_body)
        line = "time 88.88 s     " + rest                    # widest clock, for the panel width
        ptxt, pcol = phase_text(b, kk)
        if ptxt.startswith("Stuck since") and loc_status == "prolonged_blockage_terminated" and earliest:
            ptxt += f"; such a drive is stopped at {fmt_t(earliest)} s at the earliest"
        sub = f"{case_title(args.case)}, same start and goal for every planner"
        pw = max(d.textlength(b["entry"]["label"], font=f_title), d.textlength(sub, font=f_body),
                 d.textlength(line, font=f_body), 14 + d.textlength(ptxt, font=f_body),
                 max((d.textlength(t, font=f_small) for t in fn_lines), default=0),
                 max((14 + d.textlength(t, font=f_body) for t in note_lines), default=0), 600) + 36
        rounded(d, [16, 14, 16 + pw, head_bot], SURF + (232,))
        d.text((32, 24), b["entry"]["label"], fill=INK, font=f_title)
        d.text((32, 58), sub, fill=INK2, font=f_body)
        for i, t in enumerate(fn_lines):
            d.text((32, 86 + 20 * i), t, fill=INK2, font=f_small)
        d.text((32, 86 + dy), t_part, fill=INK, font=f_body)
        d.text((x_rest, 86 + dy), rest, fill=INK, font=f_body)
        d.rectangle([32, 120 + dy, 38, 140 + dy], fill=pcol)
        d.text((46, 118 + dy), ptxt, fill=INK, font=f_body)
        for i, t in enumerate(note_lines):     # a re-drive that differs from its recording says so on every frame
            if i == 0:
                d.rectangle([32, 148 + dy, 38, 168 + dy], fill=note_col)
            d.text((46, 146 + dy + 28 * i), t, fill=INK, font=f_body)
        # decision banner for 1.5 s
        if b["F"] <= kk < b["F"] + int(1.5 * FPS) and k < n:
            txt = f"Planner takes over at {b['approach_s']:g} s"
            tw = d.textlength(txt, font=f_big)
            x0 = (W - tw) / 2 - 24
            rounded(d, [x0, ban0, x0 + tw + 48, ban0 + 58], SURF + (240,))
            d.rectangle([x0, ban0, x0 + 8, ban0 + 58], fill=ORANGE)
            d.text((x0 + 30, ban0 + 8), txt, fill=INK, font=f_big)
        # minimap + legend
        mimg = mm.draw(rows[:kk + 1], kk >= b["F"])
        mx, my = W - 230 - 20, H - 230 - 20 - 78
        rounded(d, [mx - 10, my - 10, W - 10, H - 12], SURF + (232,))
        im.paste(mimg, (mx, my))
        legend(d, mx + 4, my + 236, f_small)
        # footer
        fwid = mx - 10 - 16 - 32 - 12                                          # stays left of the minimap panel
        foot = [] if note else wrap(d, e["overlay_note"], f_small, fwid)   # a differing replay says it in the header
        if soil:
            foot += wrap(d, "Soil surface drawn from the simulated soil particles (16 cm columns). Sinkage = how far "
                            "the deepest wheel has sunk below the untouched soil surface. A drive counts as bogged down "
                            f"when a wheel stays more than {BOG_LIMIT_M:.2f} m down (through the 0.24 m soil layer) "
                            "for 0.25 s.", f_small, fwid)
        fh = 22 * len(foot) + 12
        fw = max(d.textlength(t, font=f_small) for t in foot) + 32
        rounded(d, [16, H - 14 - fh, 16 + fw, H - 14], SURF + (225,))
        for i, t in enumerate(foot):
            d.text((32, H - 14 - fh + 7 + 22 * i), t, fill=INK2, font=f_small)
        # end card
        if k >= n:
            lines = [(cap(e["local_outcome_text"]), f_big, INK, 48)]
            if e["differs"] or e["recorded_outcome_text"] != e["local_outcome_text"]:
                lines.append(("Original recorded run: " + e["recorded_outcome_text"], f_body, INK2, 30))
            tw = max(d.textlength(t, font=f) for t, f, _, _ in lines)
            bh = 30 + sum(h for *_, h in lines)
            x0, y0 = (W - tw) / 2 - 30, H / 2 - bh / 2 - 40
            rounded(d, [x0, y0, x0 + tw + 60, y0 + bh], SURF + (244,), r=14)
            yy = y0 + 18
            for t, f, col, h in lines:
                d.text((x0 + 30, yy), t, fill=col, font=f)
                yy += h
        im.save(tmp / f"c_{k:05d}.jpg", quality=92)
    out = VID / f"chase_{args.case}_{args.arm}.mp4"
    ffmpeg(tmp, "c_%05d.jpg", out)
    Image.open(tmp / f"c_{n + hold - 1:05d}.jpg").save(work_dir(args.case, args.arm) / "last_composed.jpg", quality=90)
    Image.open(tmp / f"c_{min(b['F'] + 5, n - 1):05d}.jpg").save(work_dir(args.case, args.arm) / "decision_composed.jpg", quality=90)
    shutil.rmtree(tmp)
    print(out, f"{(n + hold) / FPS:.2f} s", f"{out.stat().st_size / 1e6:.1f} MB")


def side(args):
    from PIL import Image, ImageDraw
    rep = read(VID / "chase_reproduction.json")
    arms = args.arms or filmed_arms(args.case)
    B = [load_arm_bundle(args.case, arm, rep) for arm in arms]
    for b in B:
        if b["rep"].get("header_note", recorded_note(b["rep"])) != recorded_note(b["rep"]):
            raise SystemExit("chase_reproduction.json is stale: run `repro` first")
    # subtitle lines: how every replay compares with its recording, naming the top-down video with the recordings
    sub2 = side_summary(args.case, B)
    sub2_col = RED if any(not b["rep"]["same_status"] for b in B) else INK2   # red only for a different outcome
    PW, PH = 640, 360
    f_title, f_sub, f_lab, f_body, f_small, f_mid = font(24, True), font(16), font(17, True), font(15), font(13), font(22, True)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    sub2_lines = wrap(probe, sub2, f_sub, PW * len(B) - 44)
    TOP = 64 + 22 * len(sub2_lines)
    tw_ = PW - 14 - 178
    need = 0                                         # the info area holds label, phase (<= 2 lines), note, footnote
    for b in B:
        h = 10 + 22 * len(wrap(probe, b["entry"]["label"], f_lab, tw_)) + 6 + 2 * 19 + 8
        h += 17 * len(wrap(probe, b["rep"]["overlay_note"], f_small, tw_))
        if b["entry"].get("footnote"):
            h += 6 + 17 * len(wrap(probe, b["entry"]["footnote"], f_small, tw_))
        need = max(need, h + 10)
    INFO = max(200, need)
    CW, CH = PW * len(B), TOP + PH + INFO
    CH += CH % 2                                     # even height for yuv420p
    mms = [Minimap(b["case"], b["approach"], b["branch"], 150, [b["rows"]]) for b in B]
    tmp = WORK / f"side_{args.case}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    n_max = max(len(b["rows"]) for b in B)
    hold = int(3.0 * FPS)
    soil = is_soil(args.case)
    for k in range(n_max + hold):
        canvas = Image.new("RGB", (CW, CH), SURF)
        d = ImageDraw.Draw(canvas, "RGBA")
        t_now = min(k, n_max) * DT
        d.text((16, 10), f"{case_title(args.case)}: three planners, same start and goal, on one clock", fill=INK, font=f_title)
        sub = ("3D replays, re-simulated on a different computer from the original recorded drives; time "
               f"{fmt_t(t_now)} s.")
        if soil:
            sub += "  Soil surface drawn from the simulated soil particles."
        d.text((16, 40), sub, fill=INK2, font=f_sub)
        for j, line in enumerate(sub2_lines):
            if j == 0:
                d.rectangle([16, 64, 21, 80], fill=sub2_col)
            d.text((28, 62 + 22 * j), line, fill=INK, font=f_sub)
        for i, b in enumerate(B):
            rows, n = b["rows"], len(b["rows"])
            kk = min(k, n - 1)
            done = k >= n
            r = rows[kk]
            x0 = i * PW
            im = Image.open(b["frames"][kk]).convert("RGB").resize((PW, PH), Image.LANCZOS)
            pd = ImageDraw.Draw(im, "RGBA")
            t_part = f"{fmt_t(r['t'] if not done else b['rep']['local']['elapsed_s'])} s"
            txt = f"{speed_txt(r['vx'])} m/s"
            if soil:
                txt += f"   sinkage {max(r['sinkage_m'], 0):.2f} m (limit {BOG_LIMIT_M:.2f})"
            x_rest = 16 + pd.textlength("88.88 s   ", font=f_body)   # fixed position: the clock's width varies
            rounded(pd, [8, 8, x_rest + 8 + pd.textlength(txt, font=f_body), 36], SURF + (225,), r=6)
            pd.text((16, 12), t_part, fill=INK, font=f_body)
            pd.text((x_rest, 12), txt, fill=INK, font=f_body)
            if b["F"] <= kk < b["F"] + int(1.5 * FPS) and not done:
                t2 = f"Planner takes over at {b['approach_s']:g} s"
                tw = pd.textlength(t2, font=f_mid)
                rounded(pd, [(PW - tw) / 2 - 16, 52, (PW + tw) / 2 + 16, 90], SURF + (240,), r=8)
                pd.rectangle([(PW - tw) / 2 - 16, 52, (PW - tw) / 2 - 10, 90], fill=ORANGE)
                pd.text(((PW - tw) / 2 + 2, 58), t2, fill=INK, font=f_mid)
            if done:
                t2 = cap(b["rep"]["local_outcome_text"])
                lines = [(t, f_mid, INK, 30) for t in wrap_clauses(pd, t2, f_mid, PW - 80)]
                if b["rep"]["differs"] or b["rep"]["recorded_outcome_text"] != b["rep"]["local_outcome_text"]:
                    # the recorded outcome on the card itself, not only in the note below
                    lines += [(t, f_body, INK2, 22) for t in wrap_clauses(pd, b["rep"]["recorded_outcome_text"], f_body,
                                                                           PW - 80, prefix="Original recorded run: ")]
                tw = max(pd.textlength(t, font=f) for t, f, _, _ in lines)
                hh = sum(h for *_, h in lines)
                rounded(pd, [(PW - tw) / 2 - 18, PH / 2 - hh / 2 - 12, (PW + tw) / 2 + 18, PH / 2 + hh / 2 + 10], SURF + (240,), r=10)
                yy = PH / 2 - hh / 2 - 2
                for t, f, col, h in lines:
                    pd.text(((PW - pd.textlength(t, font=f)) / 2, yy), t, fill=col, font=f)
                    yy += h
            canvas.paste(im, (x0, TOP))
            if i:
                d.line([x0, TOP, x0, CH], fill=SURF, width=4)
            # info area
            y = TOP + PH + 10
            for line in wrap(d, b["entry"]["label"], f_lab, tw_):
                d.text((x0 + 14, y), line, fill=INK, font=f_lab)
                y += 22
            y += 6
            ptxt, pcol = phase_text(b, kk)
            if done:
                ptxt, pcol = cap(b["rep"]["local_outcome_text"]), INK
            for j, line in enumerate(wrap(d, ptxt, f_body, tw_ - 12)):
                if j == 0:
                    d.rectangle([x0 + 14, y + 2, x0 + 19, y + 18], fill=pcol)
                d.text((x0 + 26, y), line, fill=INK, font=f_body)
                y += 19
            y += 8
            for line in wrap(d, b["rep"]["overlay_note"], f_small, tw_):
                d.text((x0 + 14, y), line, fill=INK2, font=f_small)
                y += 17
            if b["entry"].get("footnote"):
                y += 6
                for line in wrap(d, b["entry"]["footnote"], f_small, tw_):
                    d.text((x0 + 14, y), line, fill=INK2, font=f_small)
                    y += 17
            mimg = mms[i].draw(rows[:kk + 1], kk >= b["F"], scale=0.8)
            canvas.paste(mimg, (x0 + PW - 160, TOP + PH + 8))
        # legend of the minimaps, in the title bar
        legend_x = CW - 560
        items = [("driven", BLUE), ("planner's route", ORANGE), ("straight approach", INK2), ("goal", VIOLET)]
        xx = legend_x
        for txt, col in items:
            if txt == "goal":
                d.ellipse([xx, 20, xx + 12, 32], outline=col, width=2)
            elif txt == "straight approach":
                d.line([xx, 26, xx + 6, 26], fill=col, width=2)
                d.line([xx + 10, 26, xx + 16, 26], fill=col, width=2)
            else:
                d.line([xx, 26, xx + 16, 26], fill=col, width=3)
            d.text((xx + 22, 18), txt, fill=INK, font=f_body)
            xx += 30 + d.textlength(txt, font=f_body) + 14
        canvas.save(tmp / f"s_{k:05d}.jpg", quality=92)
    out = VID / f"chase_{args.case}_side_by_side.mp4"
    ffmpeg(tmp, "s_%05d.jpg", out)
    Image.open(tmp / f"s_{n_max + hold - 1:05d}.jpg").save(WORK / f"side_{args.case}_last.jpg", quality=90)
    Image.open(tmp / f"s_{min(40, n_max - 1):05d}.jpg").save(WORK / f"side_{args.case}_t2s.jpg", quality=90)
    shutil.rmtree(tmp)
    print(out, f"{(n_max + hold) / FPS:.2f} s", f"{out.stat().st_size / 1e6:.1f} MB")


def wrap_clauses(d, text, f, width, prefix=""):
    """Like wrap, but a text that needs more than one line breaks after its '; ' or ': ' first (the end cards:
    'Stuck from 5.1 s;' / 'drive ended at 34.0 s (no progress)'); a prefix stays on the first line."""
    if d.textlength(prefix + text, font=f) <= width:
        return [prefix + text]
    parts = [q for q in re.split(r"(?<=[;:]) ", text) if q]
    parts[0] = prefix + parts[0]
    return [line for q in parts for line in wrap(d, q, f, width)]


def wrap(d, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if d.textlength(t, font=f) <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines


def main():
    global MANIFEST, WORK
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", help=f"manifest file (default {MANIFEST.relative_to(ROOT)})")
    p.add_argument("--work", help=f"work folder for drives and frames (default {WORK.relative_to(ROOT)})")
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("drive")
    q.add_argument("--case", required=True, help="manifest key (soil, rigid, soil2, ...)")
    q.add_argument("--arm", required=True)
    q.add_argument("--no-render", action="store_true")
    q.add_argument("--max-frames", type=int, default=0, help="smoke test: stop after N recorded frames")
    q.add_argument("--dump-at", type=int, nargs="*", help="soil: save the soil particles near the vehicle at these frames (surface-rule check)")
    sp.add_parser("repro")
    q = sp.add_parser("compose")
    q.add_argument("--case", required=True)
    q.add_argument("--arm", required=True)
    q = sp.add_parser("side")
    q.add_argument("--case", required=True)
    q.add_argument("--arms", nargs="*")
    q = sp.add_parser("soil-candidates", help="rank soil groups by their recorded outcomes (no driving)")
    q.add_argument("--out", required=True)
    q.add_argument("--exclude", nargs="*")
    q.add_argument("--show", type=int, default=20)
    q = sp.add_parser("add-case", help="add a soil group to the manifest under a new key")
    q.add_argument("--key", required=True)
    q.add_argument("--group", required=True)
    q = sp.add_parser("screen", help="compare camera-free local re-drives of every manifest arm with the recorded drives")
    q.add_argument("--candidates")
    q.add_argument("--tol-s", type=float, default=1.0)
    q.add_argument("--note", help="how the screening order was chosen (stored in screen.json)")
    a = p.parse_args()
    if a.manifest:
        MANIFEST = Path(a.manifest).resolve()
    if a.work:
        WORK = Path(a.work).resolve()
    {"drive": drive, "repro": repro, "compose": compose, "side": side, "soil-candidates": soil_candidates,
     "add-case": add_case, "screen": screen}[a.cmd](a)


if __name__ == "__main__":
    main()
