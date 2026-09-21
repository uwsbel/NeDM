#!/usr/bin/env python3
"""Collect one corpus. One command, one manifest, one Chrono build.

Structure mirrors the plan: preflight, then per episode -- randomise the initial state,
warm up on the stand pose, run under the policy with excitation, gate every row for
validity, split the episode at its pushes, and write the surviving segments.

What this deliberately does NOT do is delete rows in place. Training windows are built
inside an episode as `length - sequence_length + 1`, so removing rows from the middle
silently produces windows that span a temporal jump. Episodes are split into segments,
each written as its own file with its own id.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from quadruped.lib import commands as CMD           # noqa: E402
from quadruped.lib import excitation as EX          # noqa: E402
from quadruped.lib import provenance as PROV        # noqa: E402
from quadruped.lib import validity as VAL           # noqa: E402
from quadruped.lib.policy import Go2Policy          # noqa: E402


def load_params():
    p = HERE / "params"
    return (yaml.safe_load((p / "excitation.yaml").read_text()),
            yaml.safe_load((p / "presets.yaml").read_text()),
            yaml.safe_load((p / "policy.yaml").read_text()))


def build_scene(chrono, kind, urdf, spacing, step, soil, patch_x, patch_y, depth,
                travel_m=0.0, spawn_xy=(0.0, 0.0)):
    """Returns (system, robot, terrain, soil_top, dt). See walk_check.py for the
    provenance of every constant here; each one was established by a failure."""
    from nedm.quadruped.robot import Go2Robot
    from nedm.quadruped.terrain import build_crm, build_rigid_ground, measure_leg_reach
    import types

    chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0025)
    chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0025)

    system = chrono.ChSystemSMC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    system.GetSolver().AsIterative().SetMaxIterations(100)

    cwd = os.getcwd()
    rigid = kind == "rigid"
    if rigid:
        # THE GROUND MUST OUTLAST THE EPISODE. The default 10 m box is centred at the
        # origin, so it runs out at +/-5 m: at 0.5 m/s a 20 s episode travels 10 m and
        # walks straight off the edge. Measured, that reads as mean_z -4.7 -- free fall --
        # and in a collector it shows up as every episode truncating near the same row,
        # which looks like an unstable policy rather than a finite floor.
        size_m = max(10.0, 2.0 * (travel_m + 2.0))
        build_rigid_ground(chrono, system, size_m=size_m)
        soil_top = 0.05
        dt = 1.0 / 400.0
    else:
        # Same constraint on soil: the patch is centred, so it reaches +/- patch_x/2.
        need = 2.0 * (travel_m + 0.5)
        if patch_x < need:
            raise SystemExit(
                f"patch_x {patch_x} m cannot hold {travel_m:.1f} m of travel: the patch is "
                f"centred, so it reaches +/-{patch_x / 2:.1f} m. Need >= {need:.1f} m, or a "
                f"shorter episode, or a slower command.")
        soil_top = depth
        dt = 4 * step

    os.chdir(urdf.parent)
    try:
        leg_reach = measure_leg_reach(chrono, urdf)
    finally:
        os.chdir(cwd)

    # Clears the FULLY EXTENDED leg: the parser starts every joint at zero.
    spawn_z = soil_top + (0.02 if rigid else 2.0 * spacing) + leg_reach
    init = chrono.ChFramed(chrono.ChVector3d(spawn_xy[0], spawn_xy[1], spawn_z),
                           chrono.ChQuaterniond(1, 0, 0, 0))

    os.chdir(urdf.parent)
    try:
        robot = Go2Robot(system, urdf, init, actuation="torque")
    finally:
        os.chdir(cwd)

    terrain = None
    if not rigid:
        import pychrono.fsi as fsi
        import pychrono.vehicle as veh
        a = types.SimpleNamespace(spacing=spacing, step=step, soil=soil, patch_x=patch_x,
                                  patch_y=patch_y, depth=depth, soil_bottom=0.0,
                                  artificial_viscosity=2.0, no_calf_fsi=False,
                                  check_embedded=False, soil_young=None, soil_cohesion=None)
        os.chdir(urdf.parent)
        try:
            terrain, _ = build_crm(chrono, fsi, veh, system, robot, a)
        finally:
            os.chdir(cwd)
    return system, robot, terrain, soil_top, dt


def run_episode(chrono, ep_index, seed, args, exc, pol_cfg, urdf):
    """One episode. Returns (segments, verdict, meta)."""
    from nedm.quadruped.constants import STAND_ACTION
    from nedm.quadruped.dataset import capture_row, csv_field_names

    rng = np.random.default_rng(seed)

    # THE COMMAND IS DRAWN FIRST, because it sets how far the robot travels and therefore
    # how long the episode may be and where it must start. Drawing it after the scene was
    # built meant the duration limit was computed from the CLI default rather than from
    # the command actually used, so every episode ran the full 20 s regardless of speed.
    ep_cfg0 = exc["episode"]
    _dur_req = args.duration_s or ep_cfg0["duration_s"]
    if args.family == "fixed":
        fam, fam_p = "fixed", {}
        sched_fn = lambda t: (args.vx, args.vy, args.wz)  # noqa: E731
    else:
        fam = args.family
        _ranges = {k: tuple(v) for k, v in exc["commands"]["ranges"].items()}
        # RESAMPLED MID-EPISODE, matching training. The policy was trained with the
        # command redrawn every 10 s, so a step change partway through is the nominal
        # case it was optimised against, and holding one command for a whole episode is
        # the off-nominal one. One draw per window, so the draw count is fixed by the
        # duration rather than by the values drawn.
        _every = float(exc["commands"].get("resample_s") or _dur_req)
        _n_win = max(1, int(math.ceil(_dur_req / _every)))
        _subs = []
        for _w in range(_n_win):
            _p = CMD.draw_params(fam, rng, _ranges)
            _subs.append(CMD.schedule(fam, _p, _every, rng))
        fam_p = {"windows": _n_win, "resample_s": _every}

        def sched_fn(t, _subs=_subs, _every=_every, _n=_n_win):
            w = min(int(t / _every), _n - 1)
            return _subs[w](t - w * _every)
    # Peak demanded speed over the whole schedule, not the value at t=0: vel_step and
    # stop_and_go change command mid-episode and weave sweeps yaw continuously.
    _probe = [sched_fn(t) for t in np.linspace(0.0, _dur_req, 64)]
    pk_vx = max(abs(c[0]) for c in _probe)
    pk_vy = max(abs(c[1]) for c in _probe)
    # THE PATCH SETS THE EPISODE, NOT THE OTHER WAY ROUND. Duration is not a free
    # parameter: the robot runs off the bed after a fixed distance, so fix the TRAVEL
    # budget and let duration follow from the command. A 20 s episode at 1.5 m/s needs
    # 30 m of soil; the same patch holds a 20 s episode at 0.4 m/s comfortably.
    #
    # Spawning at the FAR END rather than the centre doubles the usable patch, since a
    # centred robot can only use half of it in the direction it is going.
    ep_cfg = exc["episode"]
    margin = 0.5
    bud_x, bud_y = args.patch_x - 2 * margin, args.patch_y - 2 * margin
    # Expected achieved speed, from the measured CRM envelope: roughly 0.75-0.88 of
    # command, so 0.8 is used rather than the command itself, which would over-reserve.
    exp_vx, exp_vy = 0.8 * pk_vx, 0.8 * pk_vy
    t_lim = min(bud_x / exp_vx if exp_vx > 1e-6 else 1e9,
                bud_y / exp_vy if exp_vy > 1e-6 else 1e9)
    _dur = max(min(_dur_req, t_lim), ep_cfg["min_duration_s"])
    _travel = exp_vx * (_dur + args.warmup_s)
    _s0 = sched_fn(0.0)
    sx = -np.sign(_s0[0]) * (args.patch_x / 2 - margin) if pk_vx > 1e-6 else 0.0
    sy = -np.sign(_s0[1]) * (args.patch_y / 2 - margin) if pk_vy > 1e-6 else 0.0
    if args.terrain == "rigid":
        sx = sy = 0.0        # the rigid floor is sized to the travel instead
    system, robot, terrain, soil_top, dt = build_scene(
        chrono, args.terrain, urdf, args.spacing, args.step, args.soil,
        args.patch_x, args.patch_y, args.depth, travel_m=_travel,
        spawn_xy=(float(sx), float(sy)))

    pol = Go2Policy(args.policy, cfg=pol_cfg)

    ep = exc["episode"]
    rec_dt = 1.0 / ep["record_hz"]
    ctrl_dt = 1.0 / ep["control_hz"]
    dur = _dur

    ai = exc["action_injection"]
    if args.sigma is not None:
        # Still draw, so the stream position is identical whether or not sigma is pinned.
        EX.sample_sigma(rng, ai["sigma_rad"]["low"], ai["sigma_rad"]["high"])
        sigma = float(args.sigma)
    else:
        sigma = EX.sample_sigma(rng, ai["sigma_rad"]["low"], ai["sigma_rad"]["high"])
    ou = EX.OUActionNoise(sigma_rad=sigma, tau_s=ai["tau_s"], dt=ctrl_dt,
                          rng=rng) if ai["enabled"] else None

    pu = exc["push"]
    seg_cfg = pu["segmentation"]
    # A schedule that cannot fit raises, so ask for only as many pushes as the (possibly
    # shortened) episode can hold rather than letting a fast episode fail outright.
    rows_avail = int(dur / rec_dt)
    drop = int(math.ceil(pu["duration_s"] / rec_dt)) + 2 * seg_cfg["guard_steps"] + 1
    fits = 0
    while ((fits + 2) * seg_cfg["min_segment_rows"] + (fits + 1) * drop) <= rows_avail:
        fits += 1
    n_push = min(args.pushes, fits)

    sched = None
    if pu["enabled"] and n_push:
        sched = EX.PushSchedule(duration_s=dur, events_per_episode=n_push, rng=rng,
                                warmup_s=1.0, push_duration_s=pu["duration_s"],
                                guard_steps=seg_cfg["guard_steps"],
                                min_segment_rows=seg_cfg["min_segment_rows"], dt=rec_dt)

    cmd = np.array(sched_fn(0.0), dtype=np.float32)
    pol.command = cmd

    for _ in range(int(args.warmup_s / dt)):
        robot.actuate(STAND_ACTION)
        robot.apply_pd()
        (terrain or system).DoStepDynamics(dt)

    n = int(dur / dt)
    every_ctrl = max(1, int(round(ctrl_dt / dt)))
    every_rec = max(1, int(round(rec_dt / dt)))
    rows = []
    t = 0.0
    acc = None
    push_vec = np.zeros(6)
    base = robot.base()
    if sched is not None:
        acc = base.AddAccumulator()

    for i in range(n):
        t = i * dt
        if i % every_ctrl == 0:
            cmd = np.asarray(sched_fn(t), dtype=np.float32)
            pol.command = cmd
            a = pol.act(robot)
            if ou is not None:
                a = a + pol.sign * ou.step().astype(np.float32)
            robot.actuate(a)
        robot.apply_pd()

        if sched is not None:
            base.EmptyAccumulator(acc)
            if sched.active(t):
                if not push_vec.any():
                    f = EX.sample_push(rng, pu["magnitude_n"]["low"], pu["magnitude_n"]["high"])
                    push_vec = np.concatenate([f, np.zeros(3)])
                base.AccumulateForce(acc, chrono.ChVector3d(*push_vec[:3]),
                                     base.GetPos(), False)
            else:
                push_vec = np.zeros(6)

        (terrain or system).DoStepDynamics(dt)

        if i % every_rec == 0:
            rows.append(capture_row(
                chrono=chrono, robot=robot, terrain=terrain, soil_top_m=soil_top,
                action=robot.target, command=cmd,
                soil_z=[float("nan")] * 4, soil_ctrl=float("nan"),
                scenario_name=args.family, scenario_family=args.family,
                episode_id=f"{args.corpus}_{ep_index:04d}", split="train",
                sample_index=len(rows), time_s=t, perturb=push_vec,
                gravity=[0.0, 0.0, -9.81]))

    jp = [f for f in csv_field_names() if f.startswith("joint_") and f.endswith("_pos_rad")]
    kept, tail, verdict = VAL.truncate(rows, jp, dt_s=rec_dt)

    # Segment AFTER truncation, against the length actually written.
    if sched is not None and kept:
        segs = sched.segments(duration_s=len(kept) * rec_dt, guard_steps=seg_cfg["guard_steps"],
                              dt=rec_dt)
    else:
        segs = [(0, len(kept))]
    # A segment shorter than the training window yields no windows at all, so writing it
    # costs a file and buys nothing. These appear when truncation lands mid-segment and
    # leaves a stub: measured, a 10-row tail from a 2,000-row episode that truncated at
    # 1,336. Dropped here and counted, so the loss is visible in the manifest rather than
    # showing up later as a corpus that fails its own window gate.
    min_rows = seg_cfg["min_segment_rows"]
    out = [kept[a:b] for a, b in segs if b - a >= min_rows]
    dropped_short = sum(1 for a, b in segs if 0 < b - a < min_rows)
    return out, verdict, {"sigma_rad": float(sigma), "n_rows": len(rows),
                          "kept": len(kept), "segments": len(out),
                          "family": fam, "family_params": fam_p,
                          "duration_s": round(dur, 2), "pushes": n_push,
                          "dropped_short_segments": dropped_short,
                          "events": (sched.event_times() if sched else [])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--terrain", choices=["rigid", "crm"], default="crm")
    ap.add_argument("--duration-s", type=float, default=None)
    ap.add_argument("--warmup-s", type=float, default=1.0)
    ap.add_argument("--pushes", type=int, default=2)
    ap.add_argument("--family", default=None,
                    help="pin one family, or 'fixed' to use --vx/--vy/--wz verbatim. "
                         "Default draws a balanced assignment across all families.")
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spacing", type=float, default=0.02)
    ap.add_argument("--step", type=float, default=5e-4)
    ap.add_argument("--soil", default="soft")
    ap.add_argument("--patch-x", type=float, default=8.0)
    ap.add_argument("--patch-y", type=float, default=4.0)
    ap.add_argument("--depth", type=float, default=0.20)
    ap.add_argument("--sigma", type=float, default=None,
                    help="pin the OU injection sigma instead of drawing it; the "
                         "calibration sweep uses this to index its x-axis")
    ap.add_argument("--skip-doctor", action="store_true")
    a = ap.parse_args()

    if not a.skip_doctor:
        from quadruped.doctor import require, Failed
        try:
            require("collect")
        except Failed as e:
            print(f"doctor: FAIL  {e}", file=sys.stderr)
            return 1

    exc, presets, pol_cfg = load_params()
    if pol_cfg.get("sign", {}).get("value") is None:
        print("FATAL: params/policy.yaml sign is unset; run establish_sign.py --write",
              file=sys.stderr)
        return 1

    import pychrono as chrono
    out = Path(a.out) / a.corpus
    (out / "episodes").mkdir(parents=True, exist_ok=True)

    from nedm.quadruped.dataset import csv_field_names
    fields = csv_field_names()

    t0 = time.time()
    verdicts, written, total_rows, short = [], 0, 0, 0
    fam_rng = np.random.default_rng(a.seed)
    fams = ([a.family] * a.episodes if a.family
            else CMD.stratified_families(a.episodes, fam_rng,
                                         exc["commands"]["families"]))
    fam_counts = {}
    for k in range(a.episodes):
        a.family = fams[k]
        fam_counts[fams[k]] = fam_counts.get(fams[k], 0) + 1
        segs, verdict, meta = run_episode(chrono, k, a.seed + k, a, exc, pol_cfg,
                                          Path(a.urdf))
        verdicts.append(verdict)
        short += meta["dropped_short_segments"]
        for j, seg in enumerate(segs):
            eid = f"{a.corpus}_{k:04d}_s{j}"
            with open(out / "episodes" / f"{eid}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(seg)
            written += 1
            total_rows += len(seg)
        status = "ok" if verdict.ok else f"truncated@{verdict.row}({verdict.check})"
        print(f"  ep {k:3d}  {meta['family']:<12s} {meta['duration_s']:5.1f}s "
              f"push {meta['pushes']}  sigma {meta['sigma_rad']:.3f}  "
              f"rows {meta['n_rows']} -> kept {meta['kept']}  "
              f"seg {meta['segments']}  {status}", flush=True)

    summary = VAL.summarise(verdicts)
    man = PROV.manifest(
        "corpus", REPO,
        outputs=[{"path": str(out), "segments": written, "rows": total_rows}],
        metric_defs={"validity": "v1", "sampler": EX.SAMPLER_VERSION},
        extra={"corpus": a.corpus, "terrain": a.terrain, "command": [a.vx, a.vy, a.wz],
               "episodes": a.episodes, "segments": written, "rows": total_rows,
               "failures": summary, "family_balance": fam_counts,
               "dropped_short_segments": short,
               "command_ranges": exc["commands"]["ranges"],
               "excitation": {"action_injection": exc["action_injection"],
                              "push": {"enabled": bool(a.pushes),
                                       "events_per_episode": a.pushes,
                                       "direction": exc["push"]["direction"]}},
               "wall_clock_s": round(time.time() - t0, 1)},
        notes=f"{a.episodes} episodes on {a.terrain}")
    PROV.write(out / "manifest.json", man)
    print(f"\n{written} segments, {total_rows} rows, {summary['truncated']} truncated "
          f"({summary['rate']:.0%})")
    print(f"wall clock {man['wall_clock_s']} s -> {out}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
