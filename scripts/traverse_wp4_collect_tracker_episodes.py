#!/usr/bin/env python
"""Tracker-driven Chrono episodes for the dynamics training set (DAgger-style).

The WP2 dynamics model was trained only on the scripted collection driver's actions.
Under the PPO tracker's actions it is too easy to accelerate: imagined time-to-goal is
10 % fast and energy 15-20 % low against Chrono (wp4 notes section 3). Fix the data,
not the calibration: drive TRAIN-split layouts with the tracker in Chrono and record
the same 20 Hz cache rows the WP2 cache holds (z1 15-D, applied action, pose, power).
The layout is static, so the episode's existing camera scene map is reused via
``source_key`` (the map trainer's ``--extra-train-cache`` looks it up in the base cache).

Routes per layout: the recorded route or an oracle-sweep candidate (default / fast /
slow) for action diversity. After the route end the vehicle parks (brake), as the
collector does, until the 400-frame episode is full.

Usage (newton, conda nedm):
  PYTHONPATH=src python scripts/traverse_wp4_collect_tracker_episodes.py \
      --policy artifacts/traverse/wp3_tracker_v1 --episodes 2000 --procs 10 \
      --out artifacts/traverse/wp2_z2_cache_dagger_v1
"""
from __future__ import annotations

import argparse, json, math, sys, time
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

SETTLE_S, CTRL_DT_S, N_FRAMES = 0.8, 0.05, 400
ROLL_PITCH_ABORT_RAD = math.radians(60.0)
ROUTE_CHOICES = {"recorded": 0.4, "oracle": 0.2, "fast": 0.2, "slow": 0.2}
SWEEP = {"oracle": {}, "fast": {"v_cruise_mps": 9.0}, "slow": {"v_cruise_mps": 4.0}}


def run_one(task: dict) -> dict:
    import pychrono.vehicle as veh
    from nedm.hmmwv_data import capture_row
    from nedm.traverse.layout import EpisodeLayout
    from nedm.traverse.scene import build_config, build_scene
    from nedm.traverse.terrain import TerrainMap
    from nedm.training.constants import DEFAULT_ROLLOUT_FIELDS, STATE_FIELD_PRESETS
    from traverse_wp3_chrono_eval import PolicyController, RouteTracker

    wall0 = time.time()
    arena_dir = (REPO_ROOT / task["arena"]).resolve()
    tmap = TerrainMap.from_dir(arena_dir)
    meta = json.loads(Path(task["meta_path"]).read_text())
    layout = EpisodeLayout.from_json(meta["layout"])
    policy = PolicyController(Path(task["policy"]))
    rt = RouteTracker(task["route"], policy.meta)
    state_fields = STATE_FIELD_PRESETS["tire_normal_force_omega"]
    row = {"key": task["key"], "out_key": task["out_key"], "route_name": task["route_name"]}

    start_z = float(tmap.height(*layout.start_xy)) + 0.75
    config = build_config(arena_dir, (*layout.start_xy, start_z), layout.start_yaw)
    scene = build_scene(config, layout, tmap, arena_dir, plan=None, render=None)
    hmmwv, system, terrain = scene.hmmwv, scene.system, scene.terrain
    vehicle = hmmwv.GetVehicle()
    engine, transmission = vehicle.GetEngine(), vehicle.GetTransmission()
    dt = float(config["simulation"]["step_size_s"])
    substeps = max(1, int(round(CTRL_DT_S / dt)))

    manual = veh.DriverInputs()
    last = np.array([0.0, 0.0, 1.0])
    z1_rows, act_rows, pose_rows, power_rows = [], [], [], []
    status, parked_at, first = "complete", None, True
    max_contact = 0.0
    frame = -int(round(SETTLE_S / CTRL_DT_S))
    while frame < N_FRAMES:
        ts = float(system.GetChTime())
        ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
        x, y = ref.GetPos().x, ref.GetPos().y
        yaw = float(ref.GetRot().GetCardanAnglesZYX().z)
        pre = capture_row(hmmwv, terrain, "dagger", task["route_name"], task["key"], "train", max(frame, 0), ts,
                          manual, include_tires=False)
        vx, yaw_rate = float(pre["vel_body_x_mps"]), float(pre["yaw_rate_radps"])
        err = rt.update(x, y, yaw, vx, first=first); first = False
        if frame < 0:
            cmd = np.array([0.0, 0.0, 1.0])
        elif parked_at is not None:
            cmd = np.array([0.0, 0.0, 1.0])  # park at the route end (collector convention)
        else:
            obs = np.concatenate([[err["e_along"] / 10.0, err["e_ct"] / 10.0, err["e_h"] / math.pi],
                                  rt.preview(x, y, yaw), [vx / 10.0, yaw_rate], last])
            cmd = policy.act(obs, last)
        manual.m_steering, manual.m_throttle, manual.m_braking = float(cmd[0]), float(cmd[1]), float(cmd[2])
        frame_contact = 0.0
        for sub in range(substeps):
            ts = float(system.GetChTime())
            terrain.Synchronize(ts)
            hmmwv.Synchronize(ts, manual, terrain)
            if sub == 0 and frame >= 0:  # collector convention: state after Synchronize, action of this interval
                state = capture_row(hmmwv, terrain, "dagger", task["route_name"], task["key"], "train", frame, ts,
                                    manual, include_tires=True)
                z1_rows.append([float(state[f]) for f in state_fields])
                act_rows.append(cmd.copy())
                pose_rows.append([float(state[f]) for f in DEFAULT_ROLLOUT_FIELDS])
                power_rows.append(float(engine.GetOutputMotorshaftTorque()) * float(transmission.GetOutputMotorshaftSpeed()) / 1000.0)
            terrain.Advance(dt)
            hmmwv.Advance(dt)
            for _, body in scene.asset_bodies:
                frame_contact = max(frame_contact, float(body.GetContactForce().Length()))
        max_contact = max(max_contact, frame_contact)
        last = cmd
        roll, pitch = float(vehicle.GetRoll()), float(vehicle.GetPitch())
        if abs(roll) > ROLL_PITCH_ABORT_RAD or abs(pitch) > ROLL_PITCH_ABORT_RAD:
            status = "rollover"; break
        if frame >= 0 and parked_at is None and abs(err["e_ct"]) > 6.0:
            status = "off_route"; break
        if frame >= 0 and parked_at is None and err["route_end"]:
            parked_at = frame
        frame += 1

    row.update(status=status, frames=len(z1_rows), parked_at=parked_at, max_contact_n=float(max_contact),
               wall_s=time.time() - wall0)
    if status == "complete" and len(z1_rows) == N_FRAMES:
        out = Path(task["out_dir"]) / f"{task['out_key']}.npz"
        np.savez(out, z1=np.asarray(z1_rows, np.float32), act=np.asarray(act_rows, np.float32),
                 pose=np.asarray(pose_rows, np.float32), power=np.asarray(power_rows, np.float32)[:, None],
                 source_key=np.array(task["key"]), route_name=np.array(task["route_name"]))
        row["written"] = True
    else:
        row["written"] = False
    return row


def build_tasks(args) -> list[dict]:
    from nedm.traverse import nrd_data as D
    from nedm.traverse.layout import EpisodeLayout
    from nedm.traverse.oracle import PlannerParams, plan_to_ring
    from nedm.traverse.terrain import TerrainMap

    keys = D.load_cache_keys(Path(args.cache))
    train_keys = D.split_keys(keys)[0]
    manifest = json.loads((Path(args.routes) / "routes_manifest.json").read_text())
    routed = set().union(*manifest["families"].values())
    train_keys = [k for k in train_keys if k in routed]
    rng = np.random.default_rng(args.seed)
    picked = [train_keys[i] for i in rng.permutation(len(train_keys))[: args.episodes]]
    tmap = TerrainMap.from_dir(Path(args.arena))
    names, probs = list(ROUTE_CHOICES), np.array(list(ROUTE_CHOICES.values()))
    tasks, counts = [], {}
    for key in picked:
        store, ep = key.split("__", 1)
        meta_path = Path(args.stores) / store / ep / "meta.json"
        if not meta_path.exists() and store == "full_v4_partial":
            meta_path = Path(args.stores) / "full_v4" / ep / "meta.json"
        meta = json.loads(meta_path.read_text())
        choice = str(rng.choice(names, p=probs))
        route = None
        if choice != "recorded":
            layout = EpisodeLayout.from_json(meta["layout"])
            plan = plan_to_ring(tmap, layout.obstacles(), layout.start_xy, layout.house_xy, replace(PlannerParams(), **SWEEP[choice]))
            if plan is not None:
                route = {"waypoints": plan.waypoints.tolist(), "speeds": plan.speeds.tolist(),
                         "headings": plan.headings.tolist(), "stations": plan.stations.tolist()}
            else:
                choice = "recorded"
        if route is None:
            with np.load(Path(args.routes) / f"{key}.npz") as r:
                route = {n: r[n].tolist() for n in ("waypoints", "speeds", "headings", "stations")}
        counts[choice] = counts.get(choice, 0) + 1
        tasks.append({"key": key, "out_key": f"{key}__dag_{choice}", "route_name": choice, "route": route,
                      "meta_path": str(meta_path), "arena": args.arena, "policy": args.policy, "out_dir": args.out})
    print(f"{len(tasks)} tasks; routes {counts}", flush=True)
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="artifacts/traverse/wp3_tracker_v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6")
    ap.add_argument("--routes", default="artifacts/traverse/wp3_routes")
    ap.add_argument("--stores", default="artifacts/traverse")
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--skip-existing", action="store_true", help="resume: skip tasks whose npz already exists")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args)
    written = sorted(p.stem for p in out.glob("*__dag_*.npz"))
    if args.skip_existing:
        have = set(written)
        tasks = [t for t in tasks if t["out_key"] not in have]
        print(f"resume: {len(have)} already written, {len(tasks)} to go", flush=True)
    (out / "tasks.json").write_text(json.dumps([{k: v for k, v in t.items() if k != "route"} for t in tasks]))
    rows, t0 = [], time.time()
    if not args.skip_existing:
        written = []
    with get_context("spawn").Pool(args.procs, maxtasksperchild=1) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, tasks), 1):
            rows.append(row)
            if row["written"]:
                written.append(row["out_key"])
            print(f"[{i}/{len(tasks)}] {row['out_key'][-40:]:40s} {row['status']:10s} frames={row['frames']:3d} "
                  f"parked={row['parked_at']} contact={row['max_contact_n']:.0f}N wall={row['wall_s']:.0f}s", flush=True)
            if i % 50 == 0 or i == len(tasks):
                (out / "cache_manifest.json").write_text(json.dumps({"episodes": sorted(written), "source_cache": args.cache}))
                with (out / "rows.jsonl").open("w") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
    print(f"done: {len(written)}/{len(tasks)} episodes written in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
