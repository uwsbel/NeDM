#!/usr/bin/env python3
"""Where does a CRM episode's wall time go, and what does the surrogate cost instead?

COST.md prices the whole step (3.81x real time at the 0.5 m active domain). The paper's
argument for a surrogate needs that total split, because "Chrono is slow" is only useful
if we know WHICH part is slow: the SPH soil, the Go2 multibody solve, or the foot-soil
coupling. A hybrid that keeps part of Chrono classical is only cheap if the part it
keeps is cheap.

THE PARTS OVERLAP IN TIME, so they are measured, never subtracted.
`ChFsiSystem::DoStepDynamics` runs the multibody advance on a separate std::thread while
the SPH advance runs on the main thread, then joins and exchanges forces/states
(ChFsiSystem.cpp). So step wall ~ max(CFD, MBD) + exchange, and "total minus SPH" is not
the multibody cost. Chrono's own timers give each part separately:

  concurrent  GetTimerStep()  the fork-join section (CFD and MBD together)
  cfd         GetTimerCFD()   ONE fluid substep (the last); x substeps per exchange step
  mbd         GetTimerMBD()   ONE multibody substep (the last); x substeps
  fsi         GetTimerFSI()   the force/state exchange after the join

Around that, the Python loop the evaluation actually runs: the policy (50 Hz), the PD law
(every physics step, in Python), and the state readback. Everything is timed on the
scene `evaluate.py` scores on, built by the same `build_scene`, sized the same way.

Inside the SPH step the kernel split (neighbour search, forces, BCE coupling,
integration, active-domain update) comes from running this script under rocprofv3
--kernel-trace --stats; this script does not try to time kernels itself.

The surrogate side times `finetune.advance`, the one function every in-model rollout
steps through, at batch 1 (one robot, the like-for-like comparison) and at the PPO batch.

Usage:
  python cost_profile.py --policy base.pt --urdf go2.urdf --model best.pt --out prof.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

CASES = {
    # (vx, vy, wz), the straight evaluation command and a turning one
    "straight": (0.5, 0.0, 0.0),
    "turn": (0.5, 0.0, 0.6),
}


def _t(obj, name):
    fn = getattr(obj, name, None)
    if fn is None:
        return None
    try:
        return float(fn())
    except Exception:  # noqa: BLE001
        return None


def chrono_case(kind, command, seconds, warmup_s, policy_path, urdf, spacing, step,
                soil, depth):
    import pychrono as chrono
    import yaml
    from nedm.quadruped.constants import STAND_ACTION
    from quadruped.lib.policy import Go2Policy
    from collect import build_scene, plan_path, EDGE_MARGIN_M, MAX_PATCH_X, MAX_PATCH_Y

    sched = lambda t, _c=tuple(command): _c  # noqa: E731
    # Sized exactly as evaluate.episode sizes a push-free episode.
    xlo, xhi, ylo, yhi = plan_path(sched, seconds, warmup_s, speed_factor=0.8)
    px = min(MAX_PATCH_X, max(8.0, (xhi - xlo) + 2 * (EDGE_MARGIN_M + 0.2)))
    py = min(MAX_PATCH_Y, max(4.0, (yhi - ylo) + 2 * (EDGE_MARGIN_M + 0.2)))
    spawn = (-0.5 * (xlo + xhi), -0.5 * (ylo + yhi))
    t0 = time.perf_counter()
    system, robot, terrain, _soil_top, dt = build_scene(
        chrono, kind, urdf, spacing, step, soil, px, py, depth,
        travel_m=max(xhi - xlo, yhi - ylo), spawn_xy=spawn,
        span_xy=(xhi - xlo, yhi - ylo))
    build_s = time.perf_counter() - t0

    fsys = terrain.GetFsiSystemSPH() if terrain is not None else None
    n_cfd = n_mbd = 1
    if fsys is not None:
        n_cfd = max(1, int(round(dt / fsys.GetStepSizeCFD())))
        n_mbd = max(1, int(round(dt / fsys.GetStepSizeMBD())))

    cfg = yaml.safe_load((HERE / "params" / "policy.yaml").read_text())
    pol = Go2Policy(policy_path, cfg=cfg)
    pol.command = np.asarray(command, dtype=np.float32)
    stepper = terrain or system

    t0 = time.perf_counter()
    for _ in range(int(warmup_s / dt)):
        robot.actuate(STAND_ACTION)
        robot.apply_pd()
        stepper.DoStepDynamics(dt)
    warmup_wall = time.perf_counter() - t0

    every = max(1, int(round(0.02 / dt)))
    n = int(seconds / dt)
    keys = ("act", "pd", "dostep", "readback", "concurrent", "cfd", "mbd", "fsi",
            "mbd_collision", "mbd_lssetup", "mbd_lssolve", "mbd_update", "mbd_advance")
    acc = {k: 0.0 for k in keys}
    n_act = 0
    t_loop = time.perf_counter()
    for i in range(n):
        a = time.perf_counter()
        if i % every == 0:
            robot.actuate(pol.act(robot))
            n_act += 1
        b = time.perf_counter()
        robot.apply_pd()
        c = time.perf_counter()
        stepper.DoStepDynamics(dt)
        d = time.perf_counter()
        # What evaluate.py reads back every step, so its cost is priced too.
        bb = robot.base()
        p, v, r = bb.GetPos(), bb.GetPosDt(), bb.GetRot()
        _ = (p.x, p.y, p.z, v.x, v.y, v.z, r.e0, bb.GetAngVelLocal().z)
        if i % every == 0:
            _ = (robot.joint_pos(), robot.joint_vel())
        e = time.perf_counter()
        if not math.isfinite(p.z):
            return {"kind": kind, "diverged_at_s": i * dt}
        acc["act"] += b - a
        acc["pd"] += c - b
        acc["dostep"] += d - c
        acc["readback"] += e - d
        if fsys is not None:
            acc["concurrent"] += _t(fsys, "GetTimerStep") or 0.0
            acc["cfd"] += (_t(fsys, "GetTimerCFD") or 0.0) * n_cfd
            acc["mbd"] += (_t(fsys, "GetTimerMBD") or 0.0) * n_mbd
            acc["fsi"] += _t(fsys, "GetTimerFSI") or 0.0
            mult = n_mbd
        else:
            # Rigid ground: the whole step is the multibody solve on this thread.
            acc["mbd"] += _t(system, "GetTimerStep") or 0.0
            mult = 1
        for k, name in (("mbd_collision", "GetTimerCollision"),
                        ("mbd_lssetup", "GetTimerLSsetup"),
                        ("mbd_lssolve", "GetTimerLSsolve"),
                        ("mbd_update", "GetTimerUpdate"),
                        ("mbd_advance", "GetTimerAdvance")):
            acc[k] += (_t(system, name) or 0.0) * mult
    loop_wall = time.perf_counter() - t_loop

    counts = {}
    if terrain is not None:
        for name in ("GetNumSPHParticles", "GetNumBoundaryBCEMarkers", "GetNumBCE"):
            v = _t(terrain, name)
            if v is not None:
                counts[name] = int(v)
    return {
        "kind": kind, "command": list(command), "patch": [round(px, 2), round(py, 2)],
        "dt_exchange_s": dt, "substeps_cfd": n_cfd, "substeps_mbd": n_mbd,
        "counts": counts, "build_s": round(build_s, 2), "warmup_wall_s": round(warmup_wall, 2),
        "sim_s": n * dt, "loop_wall_s": round(loop_wall, 3),
        "slowdown_x": round(loop_wall / (n * dt), 3),
        "ms_per_sim_s": {k: round(1000.0 * v / (n * dt), 2) for k, v in acc.items()},
        "policy_calls": n_act,
        "end_pos": [round(float(robot.base().GetPos().x), 3),
                    round(float(robot.base().GetPos().y), 3)],
        "diverged_at_s": None,
    }


def surrogate_cost(model_path, batches, steps, dev_name):
    import torch
    import finetune as F
    dev = torch.device(dev_name)
    model, ck = F.load_nnrom(torch, model_path, dev)
    ctx = int(ck["config"]["block_size"])
    S, A = len(ck["state_fields"]), len(ck["action_fields"])
    st = ck["stats"]
    mean = torch.tensor(np.asarray(st.get("state_mean", np.zeros(S))), dtype=torch.float32,
                        device=dev)
    std = torch.tensor(np.asarray(st.get("state_std", np.ones(S))), dtype=torch.float32,
                       device=dev)
    out = {"ctx": ctx, "state_dim": S, "action_dim": A,
           "params": sum(p.numel() for p in model.parameters()),
           "model_dt_s": 0.01, "device": str(torch.cuda.get_device_name(0))
           if dev.type == "cuda" else "cpu"}
    rows = []
    for B in batches:
        g = torch.Generator(device=dev).manual_seed(0)
        hs = mean + 0.1 * std * torch.randn(B, ctx, S, device=dev, generator=g)
        ha = 0.1 * torch.randn(B, ctx - 1, A, device=dev, generator=g)
        act = 0.1 * torch.randn(B, A, device=dev, generator=g)
        with torch.no_grad():
            for _ in range(10):                       # warm-up: kernel selection, allocs
                _n, hs, ha = F.advance(model, torch, hs, ha, act)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(steps):
                _n, hs, ha = F.advance(model, torch, hs, ha, act)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        per = wall / steps
        sim_per_wall = B * 0.01 / per                 # robot-seconds simulated per wall-second
        rows.append({"batch": B, "ms_per_step": round(1000 * per, 3),
                     "robot_sim_s_per_wall_s": round(sim_per_wall, 2)})
        print(json.dumps({"surrogate": rows[-1]}), flush=True)
    out["rows"] = rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--model", default=None, help="surrogate best.pt; omit to skip")
    ap.add_argument("--kinds", default="crm,rigid")
    ap.add_argument("--cases", default="straight,turn")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--spacing", type=float, default=0.02)
    ap.add_argument("--step", type=float, default=5e-4)
    ap.add_argument("--depth", type=float, default=0.20)
    ap.add_argument("--soil", default="soft")
    ap.add_argument("--batches", default="1,1024")
    ap.add_argument("--surrogate-steps", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res = {"chrono": [], "surrogate": None}
    if a.model:
        res["surrogate"] = surrogate_cost(Path(a.model),
                                          [int(x) for x in a.batches.split(",")],
                                          a.surrogate_steps, a.device)
    for kind in [k for k in a.kinds.split(",") if k]:
        for case in [c for c in a.cases.split(",") if c]:
            for rep in range(a.reps):
                try:
                    r = chrono_case(kind, CASES[case], a.seconds, a.warmup, Path(a.policy),
                                    Path(a.urdf), a.spacing, a.step, a.soil, a.depth)
                except Exception as e:  # noqa: BLE001
                    r = {"kind": kind, "error": f"{type(e).__name__}: {str(e)[:300]}"}
                r.update({"case": case, "rep": rep})
                res["chrono"].append(r)
                print(json.dumps(r), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
