from __future__ import annotations
"""Chrono simulation throughput ("FPS") probe for the four NeDM study scenes.

Measures how fast each Chrono scene can be *stepped* on this machine, so we know
the real cost of data collection / Chrono-side RL evaluation for:

    1. hmmwv_rigid  -- HMMWV on flat rigid terrain      (configs/hmmwv_overfit_v1.json)
    2. hmmwv_bumpy  -- HMMWV on a rigid heightmap patch (configs/hmmwv_bumpy_eval.json)
    3. hmmwv_crm    -- HMMWV on deformable CRM/SPH soil  (configs/hmmwv_crm_eval.json)
    4. tracked_arm  -- M113 tracked vehicle + welded LRV arm on flat rigid terrain
                       (nedm.arm_data.build_and_prepare)

Each case is built from the SAME scene code the collectors/evals use, so the
numbers reflect real physics settings (step size, tire model, solver, threads).
The timed inner loop replicates each collector's hot loop (Synchronize + Advance)
but drops CSV/tire-report/contact-poll bookkeeping, which is a few percent of the
per-step cost -- this is a *physics-stepping* probe.

Reported per case:
    * step_size_s   -- physics step (the atomic unit that is timed)
    * steps/s       -- physics steps per wall-clock second  (the raw "FPS")
    * ms/step       -- wall-clock milliseconds per physics step
    * RTF           -- real-time factor = simulated_seconds / wall_seconds
                       (the cross-case comparable number; RTF=1.0 is real time)
    * sample/s      -- recorded samples per wall-second at the collector's
                       record rate (HMMWV) or control rate (tracked_arm)

Rigid vs bumpy share step size / tire model / vehicle, so their delta isolates
the heightmap-collision cost; CRM runs at its own (much smaller) step and is the
heavy case.

Run (nedm conda env), one case per process -- Chrono does not like rebuilding
several sims in one process, so `--case all` re-execs one subprocess per case:

    ENV=/home/harry/anaconda3/envs/nedm/bin/python
    $ENV scripts/throughput/probe_sim_fps.py --case all
    $ENV scripts/throughput/probe_sim_fps.py --case hmmwv_crm --sim-seconds 3

NOTE: hmmwv_crm and tracked_arm are heavy (5e-4 s step). Keep --sim-seconds
modest and watch the machine (CRM SPH + 12 threads has frozen this box under
larger/parallel loads).
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = Path(__file__).resolve().parent
for _p in (str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CASES = ["hmmwv_rigid", "hmmwv_bumpy", "hmmwv_crm", "tracked_arm"]

# Per-case (warmup_seconds, measured sim_seconds) defaults. CRM/tracked run at a
# 5e-4 step, so a smaller measured window still covers thousands of steps.
CASE_DEFAULTS = {
    "hmmwv_rigid": (1.0, 3.0),
    "hmmwv_bumpy": (1.0, 3.0),
    "hmmwv_crm": (0.6, 2.0),
    "tracked_arm": (0.4, 2.0),
}

CASE_CONFIG = {
    "hmmwv_rigid": "configs/hmmwv_overfit_v1.json",
    "hmmwv_bumpy": "configs/hmmwv_bumpy_eval.json",
    "hmmwv_crm": "configs/hmmwv_crm_eval.json",
}


@dataclass
class ProbeResult:
    case: str
    label: str
    step_size_s: float
    phys_steps: int
    wall_s: float
    sim_s: float
    steps_per_s: float
    ms_per_step: float
    rtf: float
    sample_rate_hz: float          # recorded samples per wall-second
    sample_period_s: float         # record_step (HMMWV) or control_dt (arm)
    extra: dict


def _finish(case, label, step_size_s, phys_steps, wall_s, sim_s,
            sample_period_s, extra):
    steps_per_s = phys_steps / wall_s if wall_s > 0 else 0.0
    return ProbeResult(
        case=case,
        label=label,
        step_size_s=step_size_s,
        phys_steps=phys_steps,
        wall_s=wall_s,
        sim_s=sim_s,
        steps_per_s=steps_per_s,
        ms_per_step=1e3 * wall_s / phys_steps if phys_steps else 0.0,
        rtf=sim_s / wall_s if wall_s > 0 else 0.0,
        sample_rate_hz=(sim_s / sample_period_s) / wall_s if wall_s > 0 else 0.0,
        sample_period_s=sample_period_s,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# HMMWV cases (rigid / bumpy / crm) -- shared driver-held stepping loop
# ---------------------------------------------------------------------------
def probe_hmmwv(case: str, args: argparse.Namespace) -> ProbeResult:
    import pychrono as chrono  # noqa: F401
    import pychrono.vehicle as veh
    from nedm.hmmwv_data import (
        create_hmmwv,
        create_rigid_terrain,
        configure_chrono_data_paths,
        resolve_height_map,
    )

    config = json.loads((REPO_ROOT / CASE_CONFIG[case]).read_text())
    is_crm = config["terrain"].get("type") == "crm"
    if is_crm and args.crm_threads:
        config["simulation"]["chrono_threads"] = int(args.crm_threads)

    configure_chrono_data_paths(REPO_ROOT, config)
    step = float(config["simulation"]["step_size_s"])
    record_step = float(config["simulation"]["record_step_s"])

    hmmwv = create_hmmwv(config)
    extra: dict = {
        "terrain_type": config["terrain"].get("type"),
        "tire_model": config["vehicle"]["tire_model"],
        "config": CASE_CONFIG[case],
    }

    if is_crm:
        from nedm.hmmwv_crm import configure_crm_terrain

        extra["chrono_threads"] = int(config["simulation"].get("chrono_threads", 1))
        print(f"[{case}] building CRM terrain (threads={extra['chrono_threads']}) ...",
              flush=True)
        terrain, _wheels = configure_crm_terrain(hmmwv, config)
        extra["crm_particles"] = int(terrain.GetNumSPHParticles())
        extra["crm_boundary_bce"] = int(terrain.GetNumBoundaryBCEMarkers())
        print(f"[{case}] CRM particles={extra['crm_particles']:,} "
              f"bce={extra['crm_boundary_bce']:,}", flush=True)
    else:
        height_map_path = None
        if config["terrain"].get("type") == "rigid_heightmap":
            hm = resolve_height_map(config, "fps_probe")
            height_map_path = hm[1]
            extra["height_map"] = height_map_path.name
        terrain = create_rigid_terrain(
            hmmwv.GetSystem(), config, height_map_path=height_map_path
        )

    driver_inputs = veh.DriverInputs()
    driver_inputs.m_throttle = float(args.throttle)
    driver_inputs.m_steering = float(args.steering)
    driver_inputs.m_braking = 0.0

    system = hmmwv.GetSystem()

    def do_step() -> None:
        t = system.GetChTime()
        terrain.Synchronize(t)
        hmmwv.Synchronize(t, driver_inputs, terrain)
        # CRM: the terrain owns the coupled FSI+MBD advance -- do NOT also
        # advance the vehicle (see nedm.hmmwv_crm docstring).
        terrain.Advance(step)
        if not is_crm:
            hmmwv.Advance(step)

    warmup_steps = max(0, int(round(args.warmup_seconds / step)))
    print(f"[{case}] warmup {warmup_steps} steps ({args.warmup_seconds}s) ...", flush=True)
    for _ in range(warmup_steps):
        do_step()

    n_steps = max(1, int(round(args.sim_seconds / step)))
    print(f"[{case}] timing {n_steps} steps ({args.sim_seconds}s of sim) ...", flush=True)
    sim0 = system.GetChTime()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        do_step()
    wall = time.perf_counter() - t0
    sim = system.GetChTime() - sim0

    extra["throttle"] = float(args.throttle)
    extra["steering"] = float(args.steering)
    extra["final_speed_mps"] = float(hmmwv.GetVehicle().GetSpeed())
    label = {"hmmwv_rigid": "HMMWV / rigid flat",
             "hmmwv_bumpy": "HMMWV / rigid heightmap",
             "hmmwv_crm": "HMMWV / CRM soil"}[case]
    return _finish(case, label, step, n_steps, wall, sim, record_step, extra)


# ---------------------------------------------------------------------------
# Tracked vehicle + arm on rigid
# ---------------------------------------------------------------------------
def probe_tracked_arm(args: argparse.Namespace) -> ProbeResult:
    import random
    import pychrono.vehicle as veh
    from nedm.arm_data import (
        CONTROL_DT,
        STEP_SIZE,
        SmoothCommandSampler,
        build_and_prepare,
        clip_pose,
        _substep,
    )

    print("[tracked_arm] building M113 + arm scene and settling on tracks ...",
          flush=True)
    m113, vehicle, terrain, gripper, actuator, collision_links, _vis = build_and_prepare(
        render=False
    )

    driver_inputs = veh.DriverInputs()
    driver_inputs.m_throttle = 0.0
    driver_inputs.m_steering = 0.0
    driver_inputs.m_braking = 1.0  # base braked + chassis pinned (arm-only regime)

    n_sub = max(1, int(round(CONTROL_DT / STEP_SIZE)))
    sampler = SmoothCommandSampler(random.Random(0))
    system = m113.GetSystem()

    def control_step() -> None:
        qcmd = list(actuator.qcmd)
        action = sampler.next_action(qcmd)
        actuator.qcmd = clip_pose([qcmd[j] + action[j] for j in range(4)])
        _substep(m113, terrain, actuator, driver_inputs, n_sub)

    warmup_ctrl = max(0, int(round(args.warmup_seconds / CONTROL_DT)))
    print(f"[tracked_arm] warmup {warmup_ctrl} control steps "
          f"({warmup_ctrl * n_sub} substeps) ...", flush=True)
    for _ in range(warmup_ctrl):
        control_step()

    n_ctrl = max(1, int(round(args.sim_seconds / CONTROL_DT)))
    phys_steps = n_ctrl * n_sub
    print(f"[tracked_arm] timing {n_ctrl} control steps "
          f"({phys_steps} substeps, {args.sim_seconds}s of sim) ...", flush=True)
    sim0 = system.GetChTime()
    t0 = time.perf_counter()
    for _ in range(n_ctrl):
        control_step()
    wall = time.perf_counter() - t0
    sim = system.GetChTime() - sim0

    extra = {
        "terrain_type": "rigid_flat",
        "substeps_per_control": n_sub,
        "control_dt_s": CONTROL_DT,
        "control_steps_per_s": n_ctrl / wall if wall > 0 else 0.0,
    }
    return _finish("tracked_arm", "M113 + arm / rigid flat", STEP_SIZE,
                   phys_steps, wall, sim, CONTROL_DT, extra)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_result(res: ProbeResult) -> None:
    print()
    print(f"=== {res.case}  ({res.label}) ===")
    print(f"  step size      : {res.step_size_s*1e3:.3f} ms  "
          f"({1.0/res.step_size_s:.0f} Hz physics)")
    print(f"  measured       : {res.phys_steps} steps / {res.sim_s:.3f} s sim "
          f"in {res.wall_s:.2f} s wall")
    print(f"  throughput     : {res.steps_per_s:,.0f} steps/s   "
          f"({res.ms_per_step:.3f} ms/step)")
    print(f"  real-time factor: {res.rtf:.3f}x  "
          f"({'faster' if res.rtf >= 1 else 'slower'} than real time)")
    print(f"  sample rate    : {res.sample_rate_hz:,.0f} samples/s wall "
          f"(at {1.0/res.sample_period_s:.0f} Hz record rate)")
    for k, v in res.extra.items():
        if isinstance(v, float):
            print(f"  {k:15s}: {v:.4g}")
        else:
            print(f"  {k:15s}: {v}")


def print_comparison(results: list[ProbeResult]) -> None:
    print()
    print("=" * 92)
    print("FPS PROBE SUMMARY".center(92))
    print("=" * 92)
    header = (f"{'case':<13} {'scene':<26} {'dt(ms)':>7} {'steps/s':>10} "
              f"{'ms/step':>9} {'RTF':>8} {'sample/s':>10}")
    print(header)
    print("-" * 92)
    for r in results:
        print(f"{r.case:<13} {r.label:<26} {r.step_size_s*1e3:>7.3f} "
              f"{r.steps_per_s:>10,.0f} {r.ms_per_step:>9.3f} "
              f"{r.rtf:>7.3f}x {r.sample_rate_hz:>10,.0f}")
    print("-" * 92)
    print("RTF = simulated seconds per wall second (1.0 = real time; higher is faster).")
    print("sample/s = recorded samples per wall second at each collector's record/control rate.")


def run_case(case: str, args: argparse.Namespace) -> ProbeResult:
    if case == "tracked_arm":
        return probe_tracked_arm(args)
    return probe_hmmwv(case, args)


def resolve_windows(case: str, args: argparse.Namespace) -> argparse.Namespace:
    """Fill per-case warmup/sim-seconds defaults when not overridden."""
    warmup_default, sim_default = CASE_DEFAULTS[case]
    ns = argparse.Namespace(**vars(args))
    if ns.warmup_seconds is None:
        ns.warmup_seconds = warmup_default
    if ns.sim_seconds is None:
        ns.sim_seconds = sim_default
    return ns


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=CASES + ["all"], default="all")
    p.add_argument("--warmup-seconds", type=float, default=None,
                   help="Untimed settle window (per-case default if unset).")
    p.add_argument("--sim-seconds", type=float, default=None,
                   help="Timed window of simulated seconds (per-case default if unset).")
    p.add_argument("--throttle", type=float, default=0.4,
                   help="Constant HMMWV throttle during the probe (moving load).")
    p.add_argument("--steering", type=float, default=0.0,
                   help="Constant HMMWV steering during the probe.")
    p.add_argument("--crm-threads", type=int, default=None,
                   help="Override chrono_threads for the CRM case.")
    p.add_argument("--emit-json", action="store_true",
                   help="Print a 'PROBE_JSON {...}' line (used by --case all subprocesses).")
    return p


def run_all(args: argparse.Namespace) -> int:
    """Re-exec one subprocess per case (Chrono dislikes rebuilding sims in-process)."""
    results: list[ProbeResult] = []
    for case in CASES:
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--case", case, "--emit-json",
               "--throttle", str(args.throttle), "--steering", str(args.steering)]
        if args.warmup_seconds is not None:
            cmd += ["--warmup-seconds", str(args.warmup_seconds)]
        if args.sim_seconds is not None:
            cmd += ["--sim-seconds", str(args.sim_seconds)]
        if args.crm_threads is not None:
            cmd += ["--crm-threads", str(args.crm_threads)]
        print(f"\n>>> probing {case} (subprocess) ...", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True,
                              capture_output=True)
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            print(f"!!! {case} failed (exit {proc.returncode}); skipping", flush=True)
            continue
        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith("PROBE_JSON "):
                payload = json.loads(line[len("PROBE_JSON "):])
        if payload is None:
            print(f"!!! {case} produced no PROBE_JSON line; skipping", flush=True)
            continue
        results.append(ProbeResult(**payload))
    if results:
        print_comparison(results)
    return 0 if results else 1


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.case == "all":
        return run_all(args)

    args = resolve_windows(args.case, args)
    res = run_case(args.case, args)
    print_result(res)
    if args.emit_json:
        print("PROBE_JSON " + json.dumps(asdict(res)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
