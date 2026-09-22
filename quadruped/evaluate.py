#!/usr/bin/env python3
"""Score a policy in Chrono, and check it is operating inside its model's data.

Two questions, and the second is the one the previous study never asked.

  DID IT TRACK BETTER? Mean absolute command-tracking error per channel, over episodes
  BOTH arms completed. Emitted in the record shape crm_verdict.py consumes, so the paired
  comparison and its cross-host and cross-build refusals come for free.

  WAS IT ENTITLED TO? A fine-tuned policy visits (state, action) pairs its model may not
  have been fit on, and a model asked to predict there is extrapolating -- which is
  exactly where an optimiser finds structure the model invented. Gate 4 scores the visited
  points against the training corpus and says so. A tracking gain earned outside the data
  is not a result.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from lib import provenance as PROV          # noqa: E402
from lib.coverage import Reference, verdict as cov_verdict   # noqa: E402


def episode(chrono, policy_path, urdf, cfg, terrain_kind, command, seconds,
            warmup_s, spacing, step, soil, patch_x, patch_y, depth, spawn_offset=0.0):
    """Run one episode; return per-step tracking and the visited (state, action) rows.

    `command` is either a fixed (vx, vy, wz) or a schedule f(t) -> (vx, vy, wz), the same
    form the collector's command families produce. Tracking error is measured against the
    command ACTIVE at each step, which the policy is also given at each control step.
    """
    sched = command if callable(command) else (lambda t, _c=tuple(command): _c)
    from nedm.quadruped.constants import STAND_ACTION
    from quadruped.lib.policy import Go2Policy
    from quadruped.params import transforms as T
    sys.path.insert(0, str(HERE))
    from collect import (build_scene, plan_path, EDGE_MARGIN_M,   # noqa: PLC0415
                         MAX_PATCH_X, MAX_PATCH_Y)

    # THE EVALUATION BED IS SIZED THE SAME WAY THE COLLECTION BED IS, from the planned
    # path widened for yaw drift, not from `|vx| * duration`.
    #
    # Two reasons, and the second is the one that matters. The straight-line estimate is
    # simply wrong -- this policy turns at up to 0.31 rad/s with the yaw command at zero,
    # so a "straight" 6 s episode walks an arc. And an evaluation that truncates is an
    # evaluation biased by truncation: episodes that drift most get cut shortest, so the
    # surviving comparison is drawn from the better-behaved half of each arm's behaviour.
    # That would flatter whichever policy drifts more, which is exactly the axis
    # fine-tuning is expected to change.
    xlo, xhi, ylo, yhi = plan_path(sched, seconds, warmup_s)
    px = min(MAX_PATCH_X, max(patch_x, (xhi - xlo) + 2 * (EDGE_MARGIN_M + 0.2)))
    py = min(MAX_PATCH_Y, max(patch_y, (yhi - ylo) + 2 * (EDGE_MARGIN_M + 0.2)))
    # The replicate perturbation: where on the particle lattice the run begins. The bed is
    # widened by the offset so the shifted path still clears its edges.
    px = min(MAX_PATCH_X, px + 2 * abs(spawn_offset))
    spawn = (-0.5 * (xlo + xhi) + spawn_offset, -0.5 * (ylo + yhi))
    system, robot, terrain, soil_top, dt = build_scene(
        chrono, terrain_kind, urdf, spacing, step, soil, px, py, depth,
        travel_m=max(xhi - xlo, yhi - ylo), spawn_xy=spawn,
        span_xy=(xhi - xlo + 2 * abs(spawn_offset), yhi - ylo))

    pol = Go2Policy(policy_path, cfg=cfg)
    pol.command = np.asarray(sched(0.0), dtype=np.float32)

    for _ in range(int(warmup_s / dt)):
        robot.actuate(STAND_ACTION)
        robot.apply_pd()
        (terrain or system).DoStepDynamics(dt)

    every = max(1, int(round(0.02 / dt)))
    n = int(seconds / dt)
    err = {"vx": [], "vy": [], "wz": []}
    visited, min_z = [], float("inf")
    cmd = np.asarray(sched(0.0), dtype=np.float32)
    for i in range(n):
        if i % every == 0:
            cmd = np.asarray(sched(i * dt), dtype=np.float32)
            pol.command = cmd
            robot.actuate(pol.act(robot))
        robot.apply_pd()
        (terrain or system).DoStepDynamics(dt)
        b = robot.base()
        p, v, r = b.GetPos(), b.GetPosDt(), b.GetRot()
        if not math.isfinite(p.z):
            return None
        min_z = min(min_z, p.z)
        R = T.quat_to_rot(r.e0, r.e1, r.e2, r.e3)
        vb = R.T @ np.array([v.x, v.y, v.z])
        w = b.GetAngVelLocal()
        err["vx"].append(abs(vb[0] - cmd[0]))
        err["vy"].append(abs(vb[1] - cmd[1]))
        err["wz"].append(abs(w.z - cmd[2]))
        if i % every == 0:
            visited.append(np.concatenate([robot.joint_pos(), robot.joint_vel(),
                                           [vb[0], vb[1], vb[2], w.x, w.y, w.z, p.z],
                                           robot.target]))
    s = int(0.5 / dt)
    return {
        "mae_vx": float(np.mean(err["vx"][s:])),
        "mae_vy": float(np.mean(err["vy"][s:])),
        "mae_wz": float(np.mean(err["wz"][s:])),
        "min_z_m": float(min_z),
        "completed": 1,
        # UPRIGHT IS RELATIVE TO THE SURFACE, not to zero. The old 0.15 m floor was below
        # the 0.20 m soil top, so a robot whose base had sunk BENEATH the surface still
        # scored upright. On CRM the trunk is not coupled to the soil at all, so a fallen
        # robot does not rest on the bed, it descends through it -- a base below the
        # surface is not a low stance, it is a fall in progress.
        "upright": 1 if min_z > soil_top + 0.10 else 0,
        "visited": np.asarray(visited),
    }


def corpus_matrix(corpus: Path):
    """The (state, action) the corpus occupies, in the same column order episode() emits.

    STREAMED, one file at a time, parsing only the columns the reference uses. The first
    version read every row of every segment into dicts of strings -- 2 M rows x 195
    columns of Python str on the v2 corpus, ~30 GB -- to keep 36 numeric columns. On a
    30 GB workstation that was fatal: the kernel OOM-killed two evaluations on a3, and sbel
    appears to have locked up and rebooted mid-evaluation. The matrix is identical (same
    files in the same order, same columns, missing and empty values as NaN, same filter),
    so Gate 4 results do not change; only the peak memory does.
    """
    files = sorted(glob.glob(str(corpus / "episodes" / "*.csv")))
    if not files:
        raise SystemExit(f"no episodes under {corpus}")
    with open(files[0], newline="") as fh:
        c = next(csv.reader(fh))
    jp = [x for x in c if x.startswith("joint_") and x.endswith("_pos_rad")]
    jv = [x for x in c if x.startswith("joint_") and x.endswith("_vel_radps")]
    base = ["vel_body_x_mps", "vel_body_y_mps", "vel_body_z_mps",
            "roll_rate_radps", "ang_vel_body_y_radps", "yaw_rate_radps", "pos_z_m"]
    act = [x for x in c if x.startswith("joint_") and x.endswith("_target_rad")]
    cols = jp + jv + base + act
    nan = float("nan")
    blocks = []
    for f in files:
        with open(f, newline="") as fh:
            rd = csv.reader(fh)
            hdr = next(rd)
            pos = {k: i for i, k in enumerate(hdr)}
            ix = [pos.get(k, -1) for k in cols]
            blk = [[(float(row[i]) if (i >= 0 and i < len(row) and row[i] != "") else nan)
                    for i in ix] for row in rd]
        if blk:
            blocks.append(np.asarray(blk, dtype=np.float64))
    M = np.concatenate(blocks, axis=0)
    return M[np.isfinite(M).all(1)], cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--corpus", required=True, help="training corpus, for Gate 4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--terrain", choices=["rigid", "crm"], default="crm")
    ap.add_argument("--episodes", type=int, default=16,
                    help="CRM tracking has a standard deviation of about 5.7 points "
                         "(docs/EVALUATION.md), so 4 episodes resolves only ~5.7 points "
                         "at 2 se and 16 resolves ~2.9. The old default of 4 could not "
                         "see most of the effects this study reports.")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--warmup-s", type=float, default=1.0)
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--paths", type=int, default=0,
                    help="PATH MODE: this many held-out command schedules PER FAMILY, drawn "
                         "from the collector's own generator (lib/commands.py) with "
                         "evaluation-only seeds, instead of the fixed straight-line command. "
                         "The fine-tune trains on all ten families; scoring only straight "
                         "walking would test a sliver of what it was tuned for.")
    ap.add_argument("--families", default="all",
                    help="comma-separated command families for --paths, or 'all'")
    ap.add_argument("--path-seed", type=int, default=777_000_000,
                    help="seed base for path mode; far from every collection seed "
                         "(20260921 + 1000*shard + episode), so no path was seen in training")
    ap.add_argument("--spawn-spread", type=float, default=1.0,
                    help="replicates are spread over +/- this many metres of spawn "
                         "position. 1.0 is the validated perturbation; 0.25 understated "
                         "the variance and manufactured significance.")
    ap.add_argument("--max-ood", type=float, default=0.05)
    ap.add_argument("--spacing", type=float, default=0.02)
    ap.add_argument("--step", type=float, default=5e-4)
    ap.add_argument("--soil", default="soft")
    ap.add_argument("--patch-x", type=float, default=8.0)
    ap.add_argument("--patch-y", type=float, default=4.0)
    ap.add_argument("--depth", type=float, default=0.20)
    a = ap.parse_args()

    import pychrono as chrono
    import yaml
    cfg = yaml.safe_load((HERE / "params" / "policy.yaml").read_text())

    recs, vis = [], []
    # REPLICATES ARE THE SAME CONDITION AT DIFFERENT LATTICE POSITIONS, not different
    # commands. The old scheme varied the command linearly with the episode index,
    # vx * (1 + 0.15 * (k - (n-1)/2)), which was tuned for n = 4 and gave 0.39-0.61 m/s.
    # Raising the default to 16 episodes for statistical power stretched the same formula
    # to -0.06 .. 1.06 m/s -- a robot commanded BACKWARDS averaged into a forward-tracking
    # score, and one commanded twice as fast as intended. The episodes were no longer
    # replicates of anything.
    #
    # Now every episode runs the same command, and the replicate perturbation is the spawn
    # offset over +/-1.0 m: the perturbation docs/EVALUATION.md validated, which spans the
    # CRM tracking variance (sd about 5.7 points) instead of understating it the way the
    # +/-0.25 m spread did. The case list is a deterministic function of the arguments, so
    # a base policy and a fine-tuned one evaluated with the same flags see identical cases
    # and the comparison is paired episode for episode.
    cases = []   # (family, schedule, seconds, spawn_offset, descriptor)
    if a.paths > 0:
        from quadruped.lib import commands as CMD  # noqa: PLC0415
        sys.path.insert(0, str(HERE))
        from collect import plan_path, MAX_PATCH_X, MAX_PATCH_Y, EDGE_MARGIN_M  # noqa
        exc = yaml.safe_load((HERE / "params" / "excitation.yaml").read_text())
        ranges = {k: tuple(v) for k, v in exc["commands"]["ranges"].items()}
        fams = list(CMD.FAMILIES) if a.families == "all" else a.families.split(",")
        for fi, fam in enumerate(fams):
            for j in range(a.paths):
                rng = np.random.default_rng(a.path_seed + 1000 * fi + j)
                prm = CMD.draw_params(fam, rng, ranges)
                sec = a.seconds
                # A path the largest bed cannot hold is SHORTENED for every arm alike, not
                # truncated by the bed edge in whichever arm drifts furthest.
                while True:
                    sch = CMD.schedule(fam, prm, sec,
                                       np.random.default_rng(a.path_seed + 1000 * fi + j + 1))
                    xlo, xhi, ylo, yhi = plan_path(sch, sec, a.warmup_s)
                    fits = ((xhi - xlo) + 2 * (EDGE_MARGIN_M + 0.2) + 2 * a.spawn_spread
                            <= MAX_PATCH_X and (yhi - ylo) + 2 * (EDGE_MARGIN_M + 0.2)
                            <= MAX_PATCH_Y)
                    if fits or sec <= 6.0:
                        break
                    sec -= 1.0
                # Spawn spread across a family's paths, so lattice position is not
                # confounded with path.
                off = float(np.linspace(-a.spawn_spread, a.spawn_spread, a.paths)[j]
                            if a.paths > 1 else 0.0)
                cases.append((fam, sch, sec, off, {k: round(v, 3) for k, v in prm.items()}))
        print(f"path mode: {len(cases)} paths, {a.paths} per family over {len(fams)} "
              f"families, seed {a.path_seed}")
    else:
        # REPLICATES ARE THE SAME CONDITION AT DIFFERENT LATTICE POSITIONS, not different
        # commands (see git history: a command varied with the episode index once put a
        # robot commanded backwards into a forward-tracking score). The replicate
        # perturbation is the spawn offset over +/-1.0 m.
        offsets = np.linspace(-a.spawn_spread, a.spawn_spread, a.episodes)
        for k in range(a.episodes):
            cases.append(("fixed", (a.vx, 0.0, 0.0), a.seconds, float(offsets[k]),
                          {"vx": a.vx}))

    for k, (fam, cmd, sec, off, desc) in enumerate(cases):
        r = episode(chrono, Path(a.policy), Path(a.urdf), cfg, a.terrain,
                    cmd, sec, a.warmup_s, a.spacing, a.step,
                    a.soil, a.patch_x, a.patch_y, a.depth, spawn_offset=off)
        base_rec = {"episode_id": f"{a.label}_{k:03d}", "family": fam, "params": desc,
                    "seconds": sec, "spawn_offset": off,
                    "chrono_md5": PROV.chrono_provenance()["md5"]}
        if r is None:
            recs.append({**base_rec, "completed": 0})
            print(f"  ep {k}  {fam:<11s}  DIVERGED")
            continue
        v = r.pop("visited")
        vis.append(v)
        r.update(base_rec)
        if fam == "fixed":
            r["cmd_vx"] = a.vx
        recs.append(r)
        print(f"  ep {k}  {fam:<11s} {sec:4.1f}s  mae_vx {r['mae_vx']:.4f}  "
              f"mae_vy {r['mae_vy']:.4f}  mae_wz {r['mae_wz']:.4f}  min_z {r['min_z_m']:.3f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(recs, indent=1) + "\n")

    print()
    if not vis:
        print("GATE 4  skipped: no episode produced states")
        return 1
    V = np.vstack(vis)
    M, cols = corpus_matrix(Path(a.corpus))
    ref = Reference(M, rng=np.random.default_rng(0), names=cols)
    sc = ref.score(V)
    ok, msg = cov_verdict(sc, max_ood=a.max_ood)
    print(f"GATE 4  visited {sc['n']} points against a {len(M)}-row corpus")
    print(f"        outside the corpus region : {100 * sc['ood_fraction']:.1f}%")
    print(f"        distance ratio            : {sc['dist_ratio']:.2f}")
    print(f"        channels extrapolated     : {sc['channels_extrapolated']} of {len(cols)}")
    print(f"        worst channel             : {sc['worst_channel']} "
          f"({100 * sc['worst_channel_outside']:.1f}% beyond range)")
    print(f"        -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"\n  {msg}")

    man = PROV.manifest("result", REPO,
                        inputs=[{"kind": "corpus", "path": str(a.corpus)},
                                {"kind": "policy", "path": str(a.policy)}],
                        outputs=[{"path": str(out)}],
                        metric_defs={"mae": "v1", "coverage_knn": "v1"},
                        extra={"label": a.label, "terrain": a.terrain,
                               "coverage": sc, "coverage_pass": bool(ok)},
                        notes=(f"{len(cases)} paths ({a.paths}/family, seed "
                               f"{a.path_seed})" if a.paths else f"{a.episodes} episodes"))
    PROV.write(out.parent / f"{a.label}_manifest.json", man)
    print(f"\nwrote {out} and {out.parent / (a.label + '_manifest.json')}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
