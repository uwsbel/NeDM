"""Physics pilot: calibrate cloth friction against textbook rolling physics,
compare solvers and timesteps, and run sample two-ball shots.

Analytic references for a sphere launched with speed v0 and no spin on cloth
with sliding friction mu and rolling-resistance length rho:
  roll starts at t_r = 2 v0 / (7 mu g), with speed 5/7 v0;
  rolling deceleration is (5/7) rho g / R.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nedm.pool_ball.physics import (aim_limit_rad, launch_velocity, load_config, simulate_episode,
                                    with_overrides)


def slip(ball, R):
    """Contact-point slip velocity on the cloth: v + w x (0, 0, -R)."""
    return np.stack((ball[:, 3] - R * ball[:, 7], ball[:, 4] + R * ball[:, 6]), -1)


def solo(payload):
    config, overrides, v0, name = payload
    config = with_overrides(config, overrides)
    arrays, meta = simulate_episode(config, v0, 0.0, ball_b=False, duration_s=0.8)
    a_full = arrays["states"][:, 0]
    a = arrays["states"][:, 0]
    R, g = config["scene"]["radius_m"], config["simulation"]["gravity_mps2"]
    cloth = config["contact"]["cloth"]
    t = np.arange(len(a)) * config["simulation"]["record_step_s"]
    speed = np.hypot(a[:, 3], a[:, 4])
    s = np.linalg.norm(slip(a, R), axis=-1)
    rolling = np.flatnonzero(s < 1e-3)
    t_roll = float(t[rolling[0]]) if len(rolling) else None
    expected_t = 2 * v0 / (7 * cloth["friction"] * g)
    after = (t > expected_t + 0.05) & (t < t[-1] - 0.01) & (a[:, 0] < config["scene"]["half_length_m"] - R - 0.01)
    decel = float(-np.polyfit(t[after], speed[after], 1)[0]) if after.sum() > 10 else None
    return {"name": name, "overrides": overrides, "v0": v0, "t_roll_s": t_roll, "t_roll_expected_s": expected_t,
            "speed_at_roll": float(speed[rolling[0]]) if len(rolling) else None, "speed_expected": 5 / 7 * v0,
            "rolling_decel_mps2": decel, "rolling_decel_expected": 5 / 7 * cloth["rolling_resistance_m"] * g / R,
            "max_lateral_m": float(np.abs(a[:, 1]).max()), "max_height_dev_m": float(np.abs(a[:, 2] - config["scene"]["radius_m"]).max()), "slip_after_roll_max": float(s[rolling[0]:].max()) if len(rolling) else None,
            "wall_time_s": meta["wall_time_s"], "worst": meta["worst"], "termination": meta["termination"]}


def shot(payload):
    config, overrides, speed, aim_fraction, name = payload
    config = with_overrides(config, overrides)
    aim = aim_fraction * aim_limit_rad(config)
    vx, vy = launch_velocity(config, speed, aim)
    arrays, meta = simulate_episode(config, vx, vy)
    states = arrays["states"]
    record = config["simulation"]["record_step_s"]
    out = {"name": name, "overrides": overrides, "speed": speed, "aim_fraction": aim_fraction, "vx": vx, "vy": vy,
           **{k: meta[k] for k in ("termination", "accepted", "wall_time_s", "worst", "contact_sequence", "first_ab_time_s",
                                    "ab_contacts", "a_cushion_hits", "b_cushion_hits", "physics_steps")},
           "b_xy_at": {f"{t:.1f}": states[round(t / record), 1, :2].tolist() for t in (1.0, 1.5, 2.0, 2.5) if round(t / record) < len(states)},
           "a_xy_at": {f"{t:.1f}": states[round(t / record), 0, :2].tolist() for t in (1.0, 1.5, 2.0, 2.5) if round(t / record) < len(states)}}
    impacts = meta["impacts"]
    ab = next((e for e in impacts if e["pair"] == "AB"), None)
    ab_end = next((e for e in meta["events_full"] if e["pair"] == "AB" and e["kind"] == "end"), None)
    if ab and ab_end:
        before, after = np.array(ab["before"]), np.array(ab_end["after"])
        out["ab_contact_duration_s"] = ab_end["time_s"] - ab["time_s"]
        m = config["scene"]["mass_kg"]
        out["ab_impact"] = {"time_s": ab["time_s"], "a_speed_before": float(np.hypot(*before[0, 3:5])),
                            "b_speed_after": float(np.hypot(*after[1, 3:5])), "a_speed_after": float(np.hypot(*after[0, 3:5])),
                            "momentum_change": float(np.linalg.norm(m * (after[:, 3:5].sum(0) - before[:, 3:5].sum(0)))),
                            "a_slip_before": float(np.linalg.norm(slip(before[:1], config["scene"]["radius_m"])))}
    cushion = [e for e in impacts if e["pair"].startswith("B_")]
    if cushion:
        e = cushion[0]
        end = next((x for x in meta["events_full"] if x["pair"] == e["pair"] and x["kind"] == "end" and x["time_s"] > e["time_s"]), None)
        if end:
            axis = 0 if e["pair"][2] == "x" else 1
            before, after = np.array(e["before"]), np.array(end["after"])
            out["b_first_cushion"] = {"pair": e["pair"], "time_s": e["time_s"], "duration_s": end["time_s"] - e["time_s"],
                                      "normal_restitution": float(-after[1, 3 + axis] / before[1, 3 + axis]),
                                      "vz_after": float(after[1, 5])}
    out["_states"] = states
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pool_ball/chrono_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    parser.add_argument("--stage", choices=["solo", "shots", "all"], default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs_solo = []
    for young in (1e8, 5e8, 2e9):
        for dt in (0.00005, 0.000025, 0.0000125):
            for v0 in (1.5, 3.0):
                overrides = {"contact.young_modulus_pa": young, "simulation.step_s": dt}
                jobs_solo.append((config, overrides, v0, f"E{young:g}_dt{dt*1e6:g}us_v{v0}"))
    jobs_shot = []
    for speed in (1.5, 2.25, 3.0):
        for aim in (0.0, 0.5, -0.95):
            jobs_shot.append((config, {}, speed, aim, f"shot_v{speed}_aim{aim:+.2f}"))
    for speed, aim in ((2.25, 0.5), (3.0, -0.95), (1.5, 0.0)):
        jobs_shot.append((config, {"simulation.step_s": config["simulation"]["step_s"] / 2}, speed, aim, f"shot_v{speed}_aim{aim:+.2f}_halfdt"))
    with ProcessPoolExecutor(args.workers) as pool:
        if args.stage in ("solo", "all"):
            solos = list(pool.map(solo, jobs_solo))
            (args.output_dir / "solo.json").write_text(json.dumps(solos, indent=1))
            for r in solos:
                print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k not in ("overrides", "worst")}), flush=True)
        if args.stage in ("shots", "all"):
            shots = list(pool.map(shot, jobs_shot))
            np.savez_compressed(args.output_dir / "shots.npz", **{r["name"]: r.pop("_states") for r in shots})
            (args.output_dir / "shots.json").write_text(json.dumps(shots, indent=1))
            for r in shots:
                print(json.dumps({k: v for k, v in r.items() if k not in ("overrides",)}), flush=True)
            base = {r["name"]: r for r in shots}
            for r in shots:
                if r["name"].endswith("_halfdt"):
                    ref = base[r["name"].removesuffix("_halfdt")]
                    diffs = {t: math.dist(r["b_xy_at"][t], ref["b_xy_at"][t]) for t in r["b_xy_at"] if t in ref["b_xy_at"]}
                    print(json.dumps({"timestep_check": r["name"], "b_position_change_m": diffs}), flush=True)


if __name__ == "__main__":
    main()
