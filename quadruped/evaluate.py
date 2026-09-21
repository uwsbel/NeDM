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
            warmup_s, spacing, step, soil, patch_x, patch_y, depth):
    """Run one episode; return per-step tracking and the visited (state, action) rows."""
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
    xlo, xhi, ylo, yhi = plan_path(lambda t: tuple(command), seconds, warmup_s)
    px = min(MAX_PATCH_X, max(patch_x, (xhi - xlo) + 2 * (EDGE_MARGIN_M + 0.2)))
    py = min(MAX_PATCH_Y, max(patch_y, (yhi - ylo) + 2 * (EDGE_MARGIN_M + 0.2)))
    spawn = (-0.5 * (xlo + xhi), -0.5 * (ylo + yhi))
    system, robot, terrain, soil_top, dt = build_scene(
        chrono, terrain_kind, urdf, spacing, step, soil, px, py, depth,
        travel_m=max(xhi - xlo, yhi - ylo), spawn_xy=spawn,
        span_xy=(xhi - xlo, yhi - ylo))

    pol = Go2Policy(policy_path, cfg=cfg)
    pol.command = np.asarray(command, dtype=np.float32)

    for _ in range(int(warmup_s / dt)):
        robot.actuate(STAND_ACTION)
        robot.apply_pd()
        (terrain or system).DoStepDynamics(dt)

    every = max(1, int(round(0.02 / dt)))
    n = int(seconds / dt)
    err = {"vx": [], "vy": [], "wz": []}
    visited, min_z = [], float("inf")
    for i in range(n):
        if i % every == 0:
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
        err["vx"].append(abs(vb[0] - command[0]))
        err["vy"].append(abs(vb[1] - command[1]))
        err["wz"].append(abs(w.z - command[2]))
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
    """The (state, action) the corpus occupies, in the same column order episode() emits."""
    rows = []
    for f in sorted(glob.glob(str(corpus / "episodes" / "*.csv"))):
        rows.extend(list(csv.DictReader(open(f))))
    if not rows:
        raise SystemExit(f"no episodes under {corpus}")
    c = rows[0].keys()
    jp = [x for x in c if x.startswith("joint_") and x.endswith("_pos_rad")]
    jv = [x for x in c if x.startswith("joint_") and x.endswith("_vel_radps")]
    base = ["vel_body_x_mps", "vel_body_y_mps", "vel_body_z_mps",
            "roll_rate_radps", "ang_vel_body_y_radps", "yaw_rate_radps", "pos_z_m"]
    act = [x for x in c if x.startswith("joint_") and x.endswith("_target_rad")]
    cols = jp + jv + base + act
    M = np.array([[float(r.get(k, "nan") or "nan") for k in cols] for r in rows])
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
    for k in range(a.episodes):
        vx = a.vx * (1.0 + 0.15 * (k - (a.episodes - 1) / 2))
        r = episode(chrono, Path(a.policy), Path(a.urdf), cfg, a.terrain,
                    [vx, 0.0, 0.0], a.seconds, a.warmup_s, a.spacing, a.step,
                    a.soil, a.patch_x, a.patch_y, a.depth)
        if r is None:
            recs.append({"episode_id": f"{a.label}_{k:03d}", "completed": 0,
                         "chrono_md5": PROV.chrono_provenance()["md5"]})
            print(f"  ep {k}  DIVERGED")
            continue
        v = r.pop("visited")
        vis.append(v)
        r.update({"episode_id": f"{a.label}_{k:03d}", "cmd_vx": vx,
                  "chrono_md5": PROV.chrono_provenance()["md5"]})
        recs.append(r)
        print(f"  ep {k}  cmd {vx:+.2f}  mae_vx {r['mae_vx']:.4f}  "
              f"min_z {r['min_z_m']:.3f}")

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
                        notes=f"{a.episodes} episodes")
    PROV.write(out.parent / f"{a.label}_manifest.json", man)
    print(f"\nwrote {out} and {out.parent / (a.label + '_manifest.json')}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
