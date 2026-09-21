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
from quadruped.params import transforms as TR       # noqa: E402
from nedm.quadruped.terrain import (assert_spawn_on_patch,   # noqa: E402
                                    crm_patch_bounds)

LEGS = ("rr", "rl", "fr", "fl")

# How far inside the near edge the robot starts, and how much bed must remain ahead of
# it at the end. The start inset is the larger of the two because commands resample
# mid-episode and stop_and_go and weave can reverse, so the robot needs room behind it.
START_INSET_M = 1.0
EDGE_MARGIN_M = 0.5
_SLACK_M = 0.2   # the grown bed must clear the same check it was sized against

# Ceiling on how far the bed may grow. Patch length is nearly free per step, but it
# is not free in memory or build time, and an unbounded bed would let one mistyped
# command allocate tens of millions of particles. 30 m holds a 20 s episode at the
# top of the vx range; beyond that the command is slowed instead.
MAX_PATCH_X = 30.0
MAX_PATCH_Y = 12.0

# The real ceiling is GPU memory, not length. Per-step cost barely depends on particle
# count once an active domain is on (docs/COST.md), so length is cheap to BUY but not
# cheap to STORE: 26.9 x 12.0 x 0.2 m at 0.02 m spacing is 8.1 M particles, against the
# 1.77 M the cost benchmark actually exercised. A budget expressed in particles says what
# is really constrained; the length caps above only bound one axis each and miss the
# product. Over budget, the bed is shrunk and the command slowed to match.
MAX_PARTICLES = 4_000_000

# Unplanned yaw rate, measured rather than assumed. On rigid ground with the yaw command
# held at zero the policy turns at +0.126 rad/s; in the first CRM corpus a `constant`
# episode drifted +0.124 rad/s and a `lateral` one +0.309. 0.25 sits above the typical
# case and below the worst.
#
# RAISED FROM 0.25 AFTER MEASURING IT. The 60-episode pilot truncated 6 of its first 30
# episodes on `off_bed`, all of them in families that change command mid-episode --
# `random`, `vel_step`, `yaw_step` -- and four of those six still kept 91-95% of their
# rows. Marginal misses, not gross ones, which is what a slightly-too-small allowance
# looks like. 0.35 covers the 0.309 rad/s worst case actually observed rather than sitting
# just under it.
#
# The cost of being generous here is small and the cost of being tight is not: truncation
# is correlated with drift, so the episodes cut shortest are the ones that drifted most,
# and a corpus trimmed that way over-represents well-behaved locomotion.
YAW_DRIFT_RADPS = 0.35


def plan_path(sched_fn, dur_s, warmup_s, speed_factor=0.8, dt=0.05,
              drift_radps=YAW_DRIFT_RADPS):
    """Dead-reckon where the commanded episode actually goes, relative to the spawn.

    Returns (xmin, xmax, ymin, ymax) of the path, widened for yaw drift.

    THE COMMANDED YAW IS NOT THE ACHIEVED YAW. Measured on rigid ground, the terrain this
    policy was trained on, with the yaw command held at zero: the robot turns at
    +0.126 rad/s. In the first CRM corpus the same thing showed up as +0.124 rad/s on a
    `constant` episode and +0.309 rad/s on a `lateral` one -- 69 and 213 degrees of
    unplanned heading change. A straight command does not walk a straight line, it walks
    an arc, and the direction is not predictable from the command.

    That is a real property of the policy rather than an integration fault: yaw tracking
    is left-right symmetric (+1.0 -> +0.941, -1.0 -> -0.935) and the offset collapses from
    0.126 at zero command to 0.003 at unit command, which is a deadband, not an asymmetry.

    So the path is integrated THREE times -- at the commanded yaw rate and at that rate
    plus and minus the drift -- and the union of the three bounding boxes is returned. A
    disc of radius equal to the path length would also be safe and is what the geometry
    strictly implies, but it is not affordable: the cost measurement shows a 22 m bed runs
    2.7x slower per step than an 8 m one, which would spend the whole active-domain saving
    on soil the robot never touches.

    THE COMMAND IS IN THE BODY FRAME, so yaw decides where the robot ends up and a budget
    built from |vx| and |vy| alone is blind to it. The first CRM corpus showed the cost:
    a `weave` episode, whose whole purpose is to sweep yaw, left the bed after 7.33 s and
    lost 63% of its rows to `off_bed`. Nothing was wrong with the guard -- the bed had
    been sized for a robot that walks in a straight line, and weave does not.

    Integrating the schedule handles every family with one rule instead of a special case
    per family: straight runs give a long thin box, weaves give a wide one, a pure spin
    gives almost none. The 0.8 factor is the measured CRM speed envelope, the same one the
    old budget used.
    """
    import math as _m
    n = max(1, int(round((dur_s + warmup_s) / dt)))

    def one(extra_wz):
        th = x = y = 0.0
        xs = [0.0]
        ys = [0.0]
        for i in range(n):
            t = max(0.0, i * dt - warmup_s)   # warmup holds the stand pose at t=0's command
            vx, vy, wz = sched_fn(t)
            th += (wz + extra_wz) * dt
            x += speed_factor * (vx * _m.cos(th) - vy * _m.sin(th)) * dt
            y += speed_factor * (vx * _m.sin(th) + vy * _m.cos(th)) * dt
            xs.append(x)
            ys.append(y)
        return min(xs), max(xs), min(ys), max(ys)

    boxes = [one(0.0)]
    if drift_radps:
        boxes += [one(+drift_radps), one(-drift_radps)]
    return (min(b[0] for b in boxes), max(b[1] for b in boxes),
            min(b[2] for b in boxes), max(b[3] for b in boxes))

# Columns the inherited schema does not carry. Collection is the expensive step and a
# column costs almost nothing, so anything NOT derivable after the fact is recorded now.
EXTRA_FIELDS = (
    # The OU noise actually added this step. Derivable in principle by subtracting the
    # policy's own target from the applied one, but only if the scaling is reconstructed
    # exactly; logging it removes the ambiguity.
    [f"action_injection_{leg}_{seg}_rad" for leg in LEGS
     for seg in ("hip", "thigh", "calf")]
    # Push bookkeeping. NOT derivable after segmentation, because the rows that carried
    # the force are exactly the rows that were excised -- so a segment cannot say how long
    # ago it was pushed, which is the variable any recovery analysis needs.
    + ["push_active", "push_event_idx", "time_since_push_s"]
    # The push in the BODY frame. Derivable from the world force and the quaternion, and
    # precomputed because the 39-D preset reads it and a derivation that lives in two
    # places drifts.
    + ["perturb_body_x_n", "perturb_body_y_n", "perturb_body_z_n"]
    # Contact summary. Cheap, and the 4-bit code is the quantity the contact-mode work
    # keyed on.
    + ["n_feet_contact", "contact_mode"]
    # Actuator saturation. The effort limit is not in the CSV, so a reader cannot tell a
    # clipped torque from a commanded one. A saturated actuator means the target was NOT
    # achieved, which is a dynamics discontinuity worth being able to find.
    + ["n_joints_saturated", "torque_headroom_min"]
    # Solver diagnostics. Not reconstructible at all afterwards, and the first thing
    # wanted when an episode diverges.
    + ["sim_contacts", "sim_step_wall_ms"]
)


def load_params():
    p = HERE / "params"
    return (yaml.safe_load((p / "excitation.yaml").read_text()),
            yaml.safe_load((p / "presets.yaml").read_text()),
            yaml.safe_load((p / "policy.yaml").read_text()))


def build_scene(chrono, kind, urdf, spacing, step, soil, patch_x, patch_y, depth,
                travel_m=0.0, spawn_xy=(0.0, 0.0), span_xy=None):
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
        # THE BED MUST HOLD THE RUN FROM WHERE THE ROBOT ACTUALLY STARTS. The old rule
        # here demanded 2*(travel + 0.5), which is the requirement for a robot spawned at
        # the CENTRE. The collector spawns at the near edge, so that doubled the patch
        # every episode needed and priced 3.5 m of travel at an 8 m bed when 4.5 m does.
        # Particle count is linear in patch_x, so the slack was not free.
        #
        # The start-side inset is 1.0 m rather than the 0.5 m front margin because
        # commands are resampled mid-episode and families like stop_and_go and weave can
        # reverse: the robot needs room to back up without walking off behind itself.
        # THE CHECK MUST MATCH THE SIZING RULE. It used to assume a near-edge spawn and
        # demand `travel + inset + margin`, which was right when the robot started at one
        # end and walked to the other. The bed is now sized from the planned PATH and the
        # spawn is placed so the path sits centrally, so the requirement is the path's
        # span plus a margin at each end. Leaving the old form in place would have
        # rejected exactly the case that motivated the change: a weave needing an 8.1 m
        # bed for a 6.7 m span was failed against a 8.2 m demand.
        (plo_x, plo_y, _), (phi_x, phi_y, _) = crm_patch_bounds(patch_x, patch_y, depth)
        sxs, sys_ = (travel_m, travel_m) if span_xy is None else span_xy
        for axis, span, size, lo, hi in (("x", sxs, patch_x, plo_x, phi_x),
                                         ("y", sys_, patch_y, plo_y, phi_y)):
            if size < span + 2 * EDGE_MARGIN_M:
                raise SystemExit(
                    f"patch_{axis} {size:.1f} m cannot hold a {span:.1f} m path span: the "
                    f"bed runs {axis} [{lo:+.1f}, {hi:+.1f}] and the path needs "
                    f"{EDGE_MARGIN_M:.1f} m clear at each end. Need >= "
                    f"{span + 2 * EDGE_MARGIN_M:.1f} m, or a shorter episode, or a slower "
                    f"command.")
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
    from nedm.quadruped.constants import JOINT_EFFORT_NM, STAND_ACTION
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

    # LONG-RUN EPISODES ARE SLOWED, NOT LENGTHENED. Their job is to yield one
    # full-length segment for long-horizon rollout evaluation, and duration is bounded by
    # travel, so the only way to buy seconds on a fixed patch is to command less speed.
    # Without this they are indistinguishable from any other episode: the budget cuts them
    # to ~8 s, which is shorter than the 10 s rollout the evaluation wants to measure.
    if getattr(args, "long_run", False):
        _bx = args.patch_x - 1.0
        _by = args.patch_y - 1.0  # nominal bed; the real one is sized below
        _need = 0.8 * _dur_req
        _k = 1.0
        if pk_vx > 1e-6:
            _k = min(_k, _bx / (_need * pk_vx))
        if pk_vy > 1e-6:
            _k = min(_k, _by / (_need * pk_vy))
        if _k < 1.0:
            _inner, _scale = sched_fn, float(_k)

            def sched_fn(t, _f=_inner, _s=_scale):
                c = _f(t)
                return (c[0] * _s, c[1] * _s, c[2])
            pk_vx *= _scale
            pk_vy *= _scale
    # THE PATCH SETS THE EPISODE, NOT THE OTHER WAY ROUND. Duration is not a free
    # parameter: the robot runs off the bed after a fixed distance, so fix the TRAVEL
    # budget and let duration follow from the command. A 20 s episode at 1.5 m/s needs
    # 30 m of soil; the same patch holds a 20 s episode at 0.4 m/s comfortably.
    #
    # Spawning at the FAR END rather than the centre doubles the usable patch, since a
    # centred robot can only use half of it in the direction it is going.
    ep_cfg = exc["episode"]
    margin = EDGE_MARGIN_M
    # Expected achieved speed, from the measured CRM envelope: roughly 0.75-0.88 of
    # command, so 0.8 is used rather than the command itself, which would over-reserve.
    exp_vx, exp_vy = 0.8 * pk_vx, 0.8 * pk_vy

    # THE BED IS SIZED TO THE PLANNED PATH, not to a straight-line travel estimate.
    #
    # The bed grows to the episode rather than the episode shrinking to the bed, because
    # patch length is nearly free per step: 4x the particles costs 4.5% more, since every
    # SPH kernel launches over the active set rather than over all markers
    # (docs/COST.md). Clamping the episode was paying for a constraint that does not
    # exist.
    #
    # And the SHAPE of the path matters, not just its length. The command is in the body
    # frame, so yaw decides where the robot ends up, and a budget built from |vx| and |vy|
    # is blind to it. A `weave` episode in the first CRM corpus left the bed after 7.33 s
    # and lost 63% of its rows. `plan_path` dead-reckons the actual schedule, so one rule
    # covers every family instead of a special case per family.
    def _fit(dur_s):
        """Bed and spawn that hold the planned path for this duration."""
        xlo, xhi, ylo, yhi = plan_path(sched_fn, dur_s, args.warmup_s)
        px = max(args.patch_x, (xhi - xlo) + 2 * (EDGE_MARGIN_M + _SLACK_M))
        py = max(args.patch_y, (yhi - ylo) + 2 * (EDGE_MARGIN_M + _SLACK_M))
        px, py = min(px, MAX_PATCH_X), min(py, MAX_PATCH_Y)
        cells = (px / args.spacing) * (py / args.spacing) * (args.depth / args.spacing)
        if cells > MAX_PARTICLES:
            sh = (MAX_PARTICLES / cells) ** 0.5
            px, py = max(args.patch_x, px * sh), max(args.patch_y, py * sh)
        # Place the path centrally in the bed: the spawn is wherever that puts t=0.
        sx = -0.5 * (xlo + xhi)
        sy = -0.5 * (ylo + yhi)
        # THE SAME INEQUALITY build_scene WILL TEST, with room to spare. This used to be
        # `(py - span) / 2 >= EDGE_MARGIN_M`, which is algebraically identical to
        # build_scene's `py < span + 2 * EDGE_MARGIN_M` and numerically is not: once the
        # bed hits its cap the slack is gone, and the two computations round differently
        # at the boundary. Two shards of the 800-episode run died on exactly that -- an
        # 11.0 m span on a 12.0 m bed, approved here and refused there -- after 46 and 19
        # episodes, and because the manifest is written at the end, both shards' data was
        # left on disk unusable. Requiring a margin strictly larger than build_scene's
        # means this can only ever be the stricter of the two.
        tol = 1e-3
        ok = (px >= (xhi - xlo) + 2 * EDGE_MARGIN_M + tol
              and py >= (yhi - ylo) + 2 * EDGE_MARGIN_M + tol)
        return px, py, sx, sy, ok, max(xhi - xlo, 0.0), max(yhi - ylo, 0.0)

    _dur = _dur_req
    patch_x, patch_y, sx, sy, _ok, _spanx, _spany = _fit(_dur)
    if not _ok:
        # The path does not fit even at the caps. Shorten first -- a shorter episode is
        # still a real one -- and only slow the command if the floor is reached.
        _lo, _hi = ep_cfg["min_duration_s"], _dur_req
        for _ in range(24):
            _mid = 0.5 * (_lo + _hi)
            if _fit(_mid)[4]:
                _lo = _mid          # fits: try longer
            else:
                _hi = _mid          # does not fit: try shorter
        _dur = _lo
        patch_x, patch_y, sx, sy, _ok, _spanx, _spany = _fit(_dur)
    if not _ok:
        # Even the shortest episode overruns the capped bed: slow the command to fit.
        _px, _py = patch_x, patch_y
        _k = min(1.0, (_px - 2 * EDGE_MARGIN_M) / max(_spanx, 1e-6),
                 (_py - 2 * EDGE_MARGIN_M) / max(_spany, 1e-6))
        _k = max(min(_k, 1.0), 1e-3)
        _inner2, _s2 = sched_fn, float(_k)

        def sched_fn(t, _f=_inner2, _s=_s2):      # noqa: F811
            c = _f(t)
            return (c[0] * _s, c[1] * _s, c[2])
        exp_vx *= _k
        exp_vy *= _k
        pk_vx *= _k
        pk_vy *= _k
        patch_x, patch_y, sx, sy, _ok, _spanx, _spany = _fit(_dur)

    _travel = max(_spanx, _spany)
    # The bed's actual extent, for the off_bed validity check further down. Derived from
    # the bed that will be built, never recomputed independently -- that split is what
    # produced the patch-placement bug (see crm_patch_bounds).
    (plo_x, plo_y, _), (phi_x, phi_y, _) = crm_patch_bounds(
        patch_x, patch_y, args.depth)
    if args.terrain == "rigid":
        sx = sy = 0.0        # the rigid floor is sized to the travel instead
    system, robot, terrain, soil_top, dt = build_scene(
        chrono, args.terrain, urdf, args.spacing, args.step, args.soil,
        patch_x, patch_y, args.depth, travel_m=_travel,
        spawn_xy=(float(sx), float(sy)), span_xy=(_spanx, _spany))
    # The guard the placement bug got past. Checks the bed Chrono built, not the
    # arithmetic that asked for it, and fails before any simulation time is spent.
    if terrain is not None:
        assert_spawn_on_patch(terrain, (sx, sy), margin=EDGE_MARGIN_M)

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
    push_idx, last_push_t = -1, None
    _tau_last = np.zeros(12)
    _t_step = time.time()
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
            a_pol = pol.act(robot)
            raw_net = pol.last_actions.copy()      # the network's own 12 outputs
            inj = np.zeros(12, dtype=np.float32)
            if ou is not None:
                inj = (pol.sign * ou.step()).astype(np.float32)
            a = a_pol + inj
            robot.actuate(a)
        _tau_last = robot.apply_pd()

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
            _r = capture_row(
                chrono=chrono, robot=robot, terrain=terrain, soil_top_m=soil_top,
                action=robot.target, command=cmd,
                soil_z=[float("nan")] * 4, soil_ctrl=float("nan"),
                scenario_name=args.family, scenario_family=args.family,
                episode_id=f"{args.corpus}_{ep_index:04d}", split=args.split,
                sample_index=len(rows), time_s=t, perturb=push_vec,
                policy_raw=raw_net, gravity=[0.0, 0.0, -9.81])

            active = bool(push_vec[:3].any())
            if active and last_push_t is None:
                push_idx += 1
            if active:
                last_push_t = t
            q = robot.base().GetRot()
            pb = TR.world_to_body(push_vec[:3], q.e0, q.e1, q.e2, q.e3)
            tau_now = np.asarray(_tau_last, dtype=float)
            head = np.abs(JOINT_EFFORT_NM) - np.abs(tau_now)
            contacts = [1 if float(_r.get(f"foot_{l}_force_fz_n", 0.0) or 0.0) > 25.0
                        else 0 for l in LEGS]
            _r.update({
                **{f"action_injection_{l}_{sg}_rad": float(inj[3 * k + j])
                   for k, l in enumerate(LEGS)
                   for j, sg in enumerate(("hip", "thigh", "calf"))},
                "push_active": int(active),
                "push_event_idx": push_idx,
                "time_since_push_s": (round(t - last_push_t, 4)
                                      if last_push_t is not None else float("nan")),
                "perturb_body_x_n": float(pb[0]), "perturb_body_y_n": float(pb[1]),
                "perturb_body_z_n": float(pb[2]),
                "n_feet_contact": int(sum(contacts)),
                "contact_mode": int(sum(c << k for k, c in enumerate(contacts))),
                "n_joints_saturated": int((head <= 1e-6).sum()),
                "torque_headroom_min": float(head.min()) if head.size else float("nan"),
                "sim_contacts": int(system.GetNumContacts()),
                "sim_step_wall_ms": round((time.time() - _t_step) * 1e3, 3),
            })
            rows.append(_r)
            _t_step = time.time()

    jp = [f for f in csv_field_names() if f.startswith("joint_") and f.endswith("_pos_rad")]
    # The bed is passed only for CRM: the rigid floor is sized to the travel, so there is
    # no edge to leave.
    _bed = None if args.terrain == "rigid" else ((plo_x, plo_y), (phi_x, phi_y))
    # soil_top only on CRM: the rigid floor is a real collision surface, so a fallen robot
    # rests on it instead of sinking through.
    _soil_top = None if args.terrain == "rigid" else soil_top
    kept, tail, verdict = VAL.truncate(rows, jp, dt_s=rec_dt, bed=_bed,
                                       soil_top=_soil_top)

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

    # THE EXCISED ROWS ARE KEPT, SEPARATELY. They are excluded from dynamics training
    # because their cause is not a model input, but they are the only recording of how
    # this robot responds to a measured force -- and collection is the expensive step.
    # A disturbance-conditioned model would need exactly these rows, and discarding them
    # means recollecting to get them back.
    _covered = set()
    for _a, _b in segs:
        _covered.update(range(_a, _b))
    excised = [r for i, r in enumerate(kept) if i not in _covered]
    return out, excised, tail, verdict, {"sigma_rad": float(sigma), "n_rows": len(rows),
                          "kept": len(kept), "segments": len(out),
                          "excised_rows": len(excised), "failed_rows": len(tail),
                          "family": fam, "family_params": fam_p,
                          "duration_s": round(dur, 2), "pushes": n_push,
                          "long_run": bool(getattr(args, "long_run", False)),
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
    ap.add_argument("--val-fraction", type=float, default=0.2,
                    help="fraction of EPISODES held out for validation. Split by episode, "
                         "never by segment: segments from one episode share a trajectory, "
                         "so splitting them across train and val leaks.")
    ap.add_argument("--long-fraction", type=float, default=0.2,
                    help="fraction of episodes collected with NO pushes, so they yield one "
                         "full-length segment. Segmented episodes top out near 8 s, which "
                         "is too short to evaluate a 10 s rollout -- and long-horizon "
                         "fidelity is the binding constraint for any method that rolls "
                         "the model further than a gradient branch does.")
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
    fields = csv_field_names() + list(EXTRA_FIELDS)

    t0 = time.time()
    pushes_req = a.pushes
    verdicts, written, total_rows, short = [], 0, 0, 0
    n_push_rows = n_fail_rows = 0
    fam_rng = np.random.default_rng(a.seed)
    # Assigned per EPISODE, up front, so both are balanced rather than left to chance.
    n_val = int(round(a.val_fraction * a.episodes))
    val_set = set(fam_rng.permutation(a.episodes)[:n_val].tolist())
    n_long = int(round(a.long_fraction * a.episodes))
    long_set = set(fam_rng.permutation(a.episodes)[:n_long].tolist())
    fams = ([a.family] * a.episodes if a.family
            else CMD.stratified_families(a.episodes, fam_rng,
                                         exc["commands"]["families"]))
    fam_counts = {}
    skipped = []
    for k in range(a.episodes):
        a.family = fams[k]
        a.split = "val" if k in val_set else "train"
        a.pushes = 0 if k in long_set else pushes_req
        a.long_run = k in long_set
        fam_counts[fams[k]] = fam_counts.get(fams[k], 0) + 1
        # ONE IMPOSSIBLE EPISODE MUST COST ONLY ITSELF. A scene that cannot be built --
        # a path the capped bed cannot hold, a spawn off the soil -- raises SystemExit, and
        # uncaught that ended the whole shard. Two shards of the 800-episode run died that
        # way after 46 and 19 episodes, and since the manifest is written at the end, the
        # episodes already on disk were orphaned: present, and unusable, because a corpus
        # without a manifest cannot be merged or traced.
        #
        # Skipped episodes are recorded rather than dropped silently, since skipping by
        # command is a selection on command. And a run where many skip is not tolerated:
        # that is a broken setup failing on every episode, not an unlucky draw, and
        # continuing would produce a corpus of whatever happened to survive.
        try:
            segs, excised, failtail, verdict, meta = run_episode(
                chrono, k, a.seed + k, a, exc, pol_cfg, Path(a.urdf))
        except SystemExit as e:
            skipped.append({"episode": k, "family": fams[k], "reason": str(e)[:300]})
            print(f"  ep {k:3d}  SKIPPED ({fams[k]}): {str(e)[:120]}", flush=True)
            if len(skipped) >= 3 and len(skipped) > 0.2 * (k + 1):
                raise SystemExit(
                    f"{len(skipped)} of {k + 1} episodes could not be built. That is a "
                    f"broken setup, not bad luck; stopping rather than writing a corpus "
                    f"of whatever survived. Last reason: {str(e)[:200]}")
            continue
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
        # Kept out of training, kept on disk. The failing tail is where the simulator
        # broke, which is the boundary a fine-tuned policy must be held inside.
        for sub, rowset in (("pushes", excised), ("failures", failtail)):
            if not rowset:
                continue
            (out / sub).mkdir(parents=True, exist_ok=True)
            with open(out / sub / f"{a.corpus}_{k:04d}.csv", "w", newline="") as fh:
                wtr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                wtr.writeheader()
                wtr.writerows(rowset)
        n_push_rows += len(excised)
        n_fail_rows += len(failtail)
        status = "ok" if verdict.ok else f"truncated@{verdict.row}({verdict.check})"
        print(f"  ep {k:3d}  {a.split:<5s} {meta['family']:<12s} {meta['duration_s']:5.1f}s "
              f"push {meta['pushes']}  sigma {meta['sigma_rad']:.3f}  "
              f"rows {meta['n_rows']} -> kept {meta['kept']}  "
              f"seg {meta['segments']}  {status}", flush=True)

    summary = VAL.summarise(verdicts)
    man = PROV.manifest(
        "corpus", REPO,
        outputs=[{"path": str(out), "segments": written, "rows": total_rows}],
        metric_defs={"validity": "v1", "sampler": EX.SAMPLER_VERSION},
        # "fixed_command", NOT "command". provenance.manifest() records sys.argv under
        # "command" -- the invocation, which is what reproducing a run requires -- and a
        # key here of the same name silently replaced it. Every corpus manifest therefore
        # lost its invocation and stored [0.5, 0.0, 0.0] instead: the fixed-family default
        # velocity, which is not even the command used, since episodes draw from families.
        extra={"corpus": a.corpus, "terrain": a.terrain,
               "fixed_command": [a.vx, a.vy, a.wz],
               "episodes": a.episodes, "segments": written, "rows": total_rows,
               "failures": summary, "family_balance": fam_counts,
               "dropped_short_segments": short,
               # Episodes whose scene could not be built. Recorded because skipping by
               # command is a selection on command, and an empty list is itself a claim.
               "skipped_episodes": skipped,
               "split": {"val_episodes": sorted(val_set), "val_fraction": a.val_fraction},
               "long_episodes": sorted(long_set),
               "sidecar_rows": {"pushes": n_push_rows, "failures": n_fail_rows},
               "command_ranges": exc["commands"]["ranges"],
               "excitation": {"action_injection": exc["action_injection"],
                              # pushes_req, NOT a.pushes. a.pushes is the loop
                              # variable, reassigned every episode to 0 for long runs
                              # and the requested count otherwise, so at this point it
                              # holds whatever the LAST episode happened to be. Written
                              # that way, a corpus whose final episode was a long run
                              # recorded itself as having no pushes at all while three
                              # quarters of its episodes had two -- found when the
                              # merge refused two identically-configured shards.
                              "push": {"enabled": bool(pushes_req),
                                       "events_per_episode": pushes_req,
                                       "long_episodes_are_push_free": True,
                                       "direction": exc["push"]["direction"]}},
               "wall_clock_s": round(time.time() - t0, 1)},
        notes=f"{a.episodes} episodes on {a.terrain}")
    PROV.write(out / "manifest.json", man)
    print(f"\n{written} segments, {total_rows} rows, {summary['truncated']} truncated "
          f"({summary['rate']:.0%})")
    print(f"sidecars: {n_push_rows} push rows, {n_fail_rows} failure rows "
          f"(excluded from training, kept on disk)")
    print(f"wall clock {man['wall_clock_s']} s -> {out}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
