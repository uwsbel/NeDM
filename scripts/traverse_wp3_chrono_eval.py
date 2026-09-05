#!/usr/bin/env python
"""WP3/WP4 Chrono evaluation: the fragment-trained tracker driving continuously in Chrono (G6),
and the Planner-C candidate plans driven for real (validates the imagined ranking).

For each held-out episode the layout is rebuilt in Chrono from the recorded ``meta.json``
(no rendering), and every (controller, route) pair is one independent run in its own
process (repeat Chrono scene creation in one process crashes -- repo lore):

  controller  = a WP3 run dir (policy + ``policy_meta.json``) or ``follower``
                (ChPathFollowerDriver, the collection driver, as the scripted bracket)
  route       = the recorded route (ground truth exists: the real driver drove it) and,
                with ``--candidates``, the same oracle parameter sweep the NRD scorer used

The policy sees exactly the imagination env's observation, rebuilt from Chrono's true
pose/speed (pose privileged in v1 Chrono eval per plan section 1/10), with the same action
squash and steering-rate clamp. Metrics per run: completion, time to route end, energy
(engine torque x motorshaft speed, the cache's ``power``), cross-track mean/p95/max,
asset contact, rollover.

Usage (newton, conda nedm active):
  PYTHONPATH=src python scripts/traverse_wp3_chrono_eval.py --runs artifacts/traverse/wp3_tracker_v1 follower \
      --episodes 32 --procs 12 --out artifacts/traverse/wp3_chrono_eval_v1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

SETTLE_S = 0.8
CTRL_DT_S = 0.05
CONTACT_EPS_N = 1.0
ROLL_PITCH_ABORT_RAD = math.radians(60.0)
CANDIDATE_SWEEP = {
    "oracle": {},
    "shortest": {"energy_weight": 0.0},
    "energy_averse": {"energy_weight": 4.0},
    "slow": {"v_cruise_mps": 4.0},
    "fast": {"v_cruise_mps": 9.0},
    "wide_berth": {"inflation_m": 3.0},
}


# ----------------------------------------------------------------------------- route geometry (numpy mirror of tracker_env)
class RouteTracker:
    def __init__(self, route: dict, meta: dict):
        self.xy = np.asarray(route["waypoints"], np.float64)
        self.v = np.asarray(route["speeds"], np.float64)
        self.h = np.asarray(route["headings"], np.float64)
        self.s = np.asarray(route["stations"], np.float64)
        self.n = len(self.xy)
        self.ds = self.s[-1] / max(self.n - 1, 1)
        self.idx = 0
        self.offsets = np.arange(-2, int(meta["search_window"]))
        self.k = np.arange(1, int(meta["preview_points"]) + 1)
        self.spacing = float(meta["preview_spacing_m"])
        d = np.hypot(self.xy[:, 0], self.xy[:, 1])  # placeholder, replaced at first update

    def update(self, x: float, y: float, yaw: float, vx: float, first: bool = False) -> dict:
        if first:
            cand = np.arange(self.n)
        else:
            cand = np.clip(self.idx + self.offsets, 0, self.n - 1)
        d = np.hypot(self.xy[cand, 0] - x, self.xy[cand, 1] - y)
        self.idx = int(cand[int(np.argmin(d))])
        wp, h, v_ref = self.xy[self.idx], self.h[self.idx], self.v[self.idx]
        dx, dy = x - wp[0], y - wp[1]
        e_along = dx * math.cos(h) + dy * math.sin(h)
        e_ct = -dx * math.sin(h) + dy * math.cos(h)
        e_h = math.atan2(math.sin(yaw - h), math.cos(yaw - h))
        return {"e_along": e_along, "e_ct": e_ct, "e_h": e_h, "e_v": vx - v_ref, "v_ref": v_ref,
                "route_end": self.idx >= self.n - 2, "station": self.s[self.idx]}

    def preview(self, x: float, y: float, yaw: float) -> np.ndarray:
        step = np.round(self.k * self.spacing / self.ds).astype(int)
        idx = np.minimum(self.idx + step, self.n - 1)
        dx, dy = self.xy[idx, 0] - x, self.xy[idx, 1] - y
        c, s = math.cos(yaw), math.sin(yaw)
        bx, by = c * dx + s * dy, -s * dx + c * dy
        return np.stack([bx / 10.0, by / 10.0, self.v[idx] / 5.0], axis=-1).reshape(-1)


class PolicyController:
    """rsl_rl ActorCritic + empirical obs normalizer, run on CPU inside the worker."""

    def __init__(self, run_dir: Path):
        import torch
        from rsl_rl.modules import ActorCritic, EmpiricalNormalization
        self.torch = torch
        self.meta = json.loads((run_dir / "policy_meta.json").read_text())
        pol = self.meta["policy"]
        ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        payload = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        n_obs = int(self.meta["num_obs"])
        self.ac = ActorCritic(n_obs, n_obs, 3, pol["actor_hidden_dims"], pol["critic_hidden_dims"],
                              pol["activation"], pol["init_noise_std"])
        self.ac.load_state_dict(payload["model_state_dict"]); self.ac.eval()
        self.norm = EmpiricalNormalization(shape=[n_obs])
        self.norm.load_state_dict(payload["obs_norm_state_dict"]); self.norm.eval()
        self.checkpoint = ckpts[-1].name
        m = self.meta
        self.center = np.asarray(m["act_mean"] if m["action_center"] == "dataset_mean" else m["action_center"], np.float64)
        self.scale, self.low, self.high = (np.asarray(m[k], np.float64) for k in ("action_scale", "action_low", "action_high"))
        self.limit = m["steering_rate_limit"]
        self.hist = int(m["obs_history_steps"])
        self.z1_mean, self.z1_std = np.asarray(m["z1_mean"]), np.asarray(m["z1_std"])

    def act(self, obs: np.ndarray, last: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            o = self.torch.as_tensor(obs, dtype=self.torch.float32)[None]
            out = self.ac.act_inference(self.norm(o))[0].numpy().astype(np.float64)
        driver = np.clip(self.center + self.scale * np.tanh(out), self.low, self.high)
        if self.limit is not None:
            driver[0] = min(max(driver[0], last[0] - self.limit), last[0] + self.limit)
        return driver


# ----------------------------------------------------------------------------- one Chrono run
class CameraLocaliser:
    """Per-frame vehicle pose from the overhead RGB-D frame: frozen WP1 encoder stem + WP4 pose head
    (scripts/traverse_wp4_train_posehead.py). Pixel -> world uses the known arena heightmap."""

    def __init__(self, posehead_ckpt: Path, arena_dir: Path):
        import importlib.util
        import torch
        from nedm.traverse import perception as P
        from nedm.traverse.camera import CameraModel
        from nedm.traverse.storage import DEPTH_NO_HIT, DEPTH_OFFSET_M, encode_depth_mm
        from nedm.traverse.terrain import TerrainMap
        spec = importlib.util.spec_from_file_location("posehead", REPO_ROOT / "scripts" / "traverse_wp4_train_posehead.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        self.mod, self.torch = mod, torch
        payload = torch.load(posehead_ckpt, map_location="cpu", weights_only=False)
        enc_ckpt = Path(payload["encoder_ckpt"])
        enc_ckpt = enc_ckpt if enc_ckpt.is_absolute() else REPO_ROOT / enc_ckpt
        encoder = P.Encoder(z_dim=256, n_q=8)
        encoder.load_state_dict(torch.load(enc_ckpt, map_location="cpu", weights_only=False)["encoder"], strict=True)
        self.stem = encoder.backbone[:mod.STAGE].eval()
        self.head = mod.PoseHead(width=payload["config"]["width"]).eval()
        self.head.load_state_dict(payload["head"])
        self.cam, self.tmap = CameraModel(), TerrainMap.from_dir(arena_dir)
        _, _, sec = self.cam.pixel_rays(P.DEPTH_RAY_SCALE)
        self.sec = sec.astype(np.float32)
        self.h_min, self.h_max = float(self.tmap.meta["height_min_m"]), float(self.tmap.meta["height_max_m"])
        self._enc, self._nohit, self._off = encode_depth_mm, DEPTH_NO_HIT, DEPTH_OFFSET_M
        torch.set_num_threads(2)

    def __call__(self, rgb_u8: np.ndarray, depth_m: np.ndarray) -> tuple[float, float, float]:
        mm = self._enc(depth_m)  # exactly the store's quantization
        d = self._off + mm.astype(np.float32) / 1000.0
        z = self.cam.cam_height_m - d / self.sec
        z[mm == self._nohit] = self.h_min
        z_norm = (z - self.h_min) / (self.h_max - self.h_min)
        inp = np.concatenate([rgb_u8.astype(np.float32).transpose(2, 0, 1) / 255.0, z_norm[None]], 0)
        with self.torch.no_grad():
            fmap = self.stem(self.torch.from_numpy(inp)[None])
            _, u_s, v_s, yaw = self.head(fmap)
        u, v = self.mod.stage_to_img(u_s.numpy(), v_s.numpy())
        x, y = self.mod.pixel_to_world(self.cam, self.tmap, u, v)
        return float(x[0]), float(y[0]), float(math.atan2(float(yaw[0, 0]), float(yaw[0, 1])))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def run_one(task: dict) -> dict:
    import pychrono as chrono
    import pychrono.vehicle as veh
    from nedm.hmmwv_data import capture_row
    from nedm.traverse.layout import EpisodeLayout
    from nedm.traverse.scene import build_config, build_scene
    from nedm.traverse.terrain import TerrainMap
    from nedm.training.constants import STATE_FIELD_PRESETS

    wall0 = time.time()
    arena_dir = (REPO_ROOT / task["arena"]).resolve()
    tmap = TerrainMap.from_dir(arena_dir)
    meta = json.loads(Path(task["meta_path"]).read_text())
    layout = EpisodeLayout.from_json(meta["layout"])
    route = task["route"]
    row = {k: task[k] for k in ("key", "controller", "candidate")}
    row["length_m"] = float(route["stations"][-1])
    policy = None
    if task["controller"] != "follower":
        policy = PolicyController(Path(task["controller"]))
        pmeta = policy.meta
    else:
        pmeta = json.loads(Path(task["ref_meta"]).read_text())
    rt = RouteTracker(route, pmeta)
    state_fields = STATE_FIELD_PRESETS["tire_normal_force_omega"]

    start_z = float(tmap.height(*layout.start_xy)) + 0.75
    config = build_config(arena_dir, (*layout.start_xy, start_z), layout.start_yaw)
    loc_mode = task.get("localisation", "true")
    render = None
    if loc_mode != "true":
        from nedm.traverse.scene import RenderSpec
        render = RenderSpec(width=256, height=256, plan_markers=False)
    scene = build_scene(config, layout, tmap, arena_dir, plan=None, render=render)
    hmmwv, system, terrain = scene.hmmwv, scene.system, scene.terrain
    vehicle = hmmwv.GetVehicle()
    engine, transmission = vehicle.GetEngine(), vehicle.GetTransmission()
    dt = float(config["simulation"]["step_size_s"])
    substeps = max(1, int(round(CTRL_DT_S / dt)))
    obstacles = np.asarray(layout.obstacles(), np.float64)
    localiser = CameraLocaliser(Path(task["posehead"]), arena_dir) if render is not None else None
    est = None  # (x, y, yaw) the tracker uses when localisation != true
    settle_est: list[tuple[float, float, float]] = []
    loc_xy_log, loc_yaw_log = [], []
    k_xy, k_yaw = float(task.get("loc_gain_xy", 0.3)), float(task.get("loc_gain_yaw", 0.15))

    driver = None
    if policy is None:
        pts = chrono.vector_ChVector3d()
        last_s = -10.0
        for (x, y), s in zip(rt.xy, rt.s):
            if s - last_s < 2.0 and s != rt.s[-1]:
                continue
            last_s = s
            pts.append(chrono.ChVector3d(float(x), float(y), float(tmap.height(x, y)) + 0.5))
        driver = veh.ChPathFollowerDriver(vehicle, chrono.ChBezierCurve(pts), "route", float(rt.v[0]))
        driver.GetSteeringController().SetLookAheadDistance(5.0)
        driver.GetSteeringController().SetGains(0.8, 0.0, 0.0)
        driver.GetSpeedController().SetGains(0.6, 0.05, 0.0)
        driver.Initialize()

    manual = veh.DriverInputs()
    last = np.array([0.0, 0.0, 1.0])
    prev_steer = 0.0
    prev_vx = prev_vy = prev_yaw_rate = 0.0
    prev_cam = None
    hist_states: list[np.ndarray] = []
    ct_log, ev_log, eh_log, act_log = [], [], [], []
    energy_kj = 0.0
    max_contact = 0.0
    min_clear = math.inf
    status, end_time = "timeout", None
    n_frames = int(round(task["horizon_s"] / CTRL_DT_S))
    frame = -int(round(SETTLE_S / CTRL_DT_S))
    first = True
    while frame < n_frames:
        ts = float(system.GetChTime())
        ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
        x, y = ref.GetPos().x, ref.GetPos().y
        yaw = float(ref.GetRot().GetCardanAnglesZYX().z)
        state = capture_row(hmmwv, terrain, "eval", "eval", task["key"], "val", max(frame, 0), ts,
                            manual, include_tires=policy is not None and policy.hist > 0)
        vx, yaw_rate = float(state["vel_body_x_mps"]), float(state["yaw_rate_radps"])
        x_true, y_true, yaw_true = x, y, yaw
        if localiser is not None:
            scene.manager.Update()
            cam_xyz = localiser(scene.rgb_tap.take(), scene.depth_tap.take())
            if frame < 0:
                settle_est.append(cam_xyz)
                est = cam_xyz
            elif loc_mode == "camera":
                est = cam_xyz
            else:  # fused: odometry prediction from the previous frame's body velocities + camera correction
                if frame == 0 and settle_est:
                    a = np.asarray(settle_est)
                    est = (float(a[:, 0].mean()), float(a[:, 1].mean()),
                           float(math.atan2(np.sin(a[:, 2]).mean(), np.cos(a[:, 2]).mean())))
                px, py, pyaw = est
                vy_b = float(state["vel_body_y_mps"])
                px += CTRL_DT_S * (prev_vx * math.cos(pyaw) - prev_vy * math.sin(pyaw))
                py += CTRL_DT_S * (prev_vx * math.sin(pyaw) + prev_vy * math.cos(pyaw))
                pyaw += CTRL_DT_S * prev_yaw_rate
                px += k_xy * (cam_xyz[0] - px); py += k_xy * (cam_xyz[1] - py)
                # heading: the camera's yaw is the weakest channel; when moving, the direction of
                # travel from consecutive camera fixes is a second, independent heading measurement
                yaw_meas, k_use = cam_xyz[2], k_yaw
                if prev_cam is not None and vx > 1.5:
                    dx, dy = cam_xyz[0] - prev_cam[0], cam_xyz[1] - prev_cam[1]
                    if math.hypot(dx, dy) > 0.05:
                        yaw_motion = math.atan2(dy, dx) - math.atan2(prev_vy, max(prev_vx, 1e-3))
                        yaw_meas = math.atan2(0.3 * math.sin(cam_xyz[2]) + 0.7 * math.sin(yaw_motion),
                                              0.3 * math.cos(cam_xyz[2]) + 0.7 * math.cos(yaw_motion))
                        k_use = max(k_yaw, 0.3)
                pyaw = wrap_pi(pyaw + k_use * wrap_pi(yaw_meas - pyaw))
                est = (px, py, pyaw)
            prev_cam = cam_xyz
            prev_vx, prev_vy, prev_yaw_rate = vx, float(state["vel_body_y_mps"]), yaw_rate
            x, y, yaw = est
            if frame >= 0:
                loc_xy_log.append(math.hypot(x - x_true, y - y_true)); loc_yaw_log.append(abs(wrap_pi(yaw - yaw_true)))
        err = rt.update(x, y, yaw, vx, first=first); first = False

        if frame < 0:  # settle with brakes on, wheels straight (collector convention)
            cmd = np.array([0.0, 0.0, 1.0]); prev_steer = 0.0
            if driver is not None:
                driver.SetDesiredSpeed(0.0)
        elif policy is not None:
            obs = np.concatenate([[err["e_along"] / 10.0, err["e_ct"] / 10.0, err["e_h"] / math.pi],
                                  rt.preview(x, y, yaw), [vx / 10.0, yaw_rate], last])
            if policy.hist > 0:
                z1 = np.array([float(state[f]) for f in state_fields])
                hist_states.append((z1 - policy.z1_mean) / policy.z1_std)
                hist = hist_states[-policy.hist:]
                hist = [hist[0]] * (policy.hist - len(hist)) + hist
                obs = np.concatenate([obs, np.concatenate(hist)])
            cmd = policy.act(obs, last)
        else:
            driver.SetDesiredSpeed(0.0 if err["route_end"] else float(err["v_ref"]))
            cmd = None

        if frame >= 0:
            if localiser is not None:  # judge tracking with the TRUE pose, not the estimate
                err_true = RouteTracker.__new__(RouteTracker); err_true.__dict__.update(rt.__dict__)
                e_t = err_true.update(x_true, y_true, yaw_true, vx, first=False)
                ct_log.append(abs(e_t["e_ct"])); ev_log.append(abs(e_t["e_v"])); eh_log.append(abs(e_t["e_h"]))
            else:
                ct_log.append(abs(err["e_ct"])); ev_log.append(abs(err["e_v"])); eh_log.append(abs(err["e_h"]))
            c, s_ = math.cos(yaw_true), math.sin(yaw_true)
            for off in (-1.9, 0.0, 1.9):
                cx, cy = x_true + off * c, y_true + off * s_
                if len(obstacles):
                    clear = float(np.min(np.hypot(obstacles[:, 0] - cx, obstacles[:, 1] - cy) - obstacles[:, 2] - 1.3))
                    min_clear = min(min_clear, clear)

        frame_contact = 0.0
        for sub in range(substeps):
            ts = float(system.GetChTime())
            if driver is not None:
                driver.Synchronize(ts)
                inputs = driver.GetInputs()
                if frame < 0:
                    prev_steer = 0.0
                else:
                    max_d = 2.0 * dt
                    prev_steer = min(max(float(inputs.m_steering), prev_steer - max_d), prev_steer + max_d)
                inputs.m_steering = prev_steer
            else:
                manual.m_steering, manual.m_throttle, manual.m_braking = float(cmd[0]), float(cmd[1]), float(cmd[2])
                inputs = manual
            terrain.Synchronize(ts)
            hmmwv.Synchronize(ts, inputs, terrain)
            if frame >= 0:
                p_kw = float(engine.GetOutputMotorshaftTorque()) * float(transmission.GetOutputMotorshaftSpeed()) / 1000.0
                energy_kj += p_kw * dt
            if driver is not None:
                driver.Advance(dt)
            terrain.Advance(dt)
            hmmwv.Advance(dt)
            for _, body in scene.asset_bodies:
                frame_contact = max(frame_contact, float(body.GetContactForce().Length()))
        max_contact = max(max_contact, frame_contact)
        if driver is not None:
            last = np.array([prev_steer, float(inputs.m_throttle), float(inputs.m_braking)])
        else:
            last = cmd
        if frame >= 0:
            act_log.append(last.copy())

        roll, pitch = float(vehicle.GetRoll()), float(vehicle.GetPitch())
        if abs(roll) > ROLL_PITCH_ABORT_RAD or abs(pitch) > ROLL_PITCH_ABORT_RAD:
            status = "rollover"; break
        if frame >= 0 and abs(err["e_ct"]) > 6.0:
            status = "off_route"; break
        if frame >= 0 and err["route_end"]:
            status = "completed"; end_time = (frame + 1) * CTRL_DT_S; break
        frame += 1

    ct = np.asarray(ct_log) if ct_log else np.zeros(1)
    acts = np.asarray(act_log) if act_log else np.zeros((1, 3))
    row.update(status=status, completed=status == "completed",
               time_s=float(end_time if end_time is not None else task["horizon_s"]),
               energy_kj=float(energy_kj), mean_ct_m=float(ct.mean()), p95_ct_m=float(np.quantile(ct, 0.95)),
               max_ct_m=float(ct.max()), mean_speed_err_mps=float(np.mean(ev_log)) if ev_log else 0.0,
               mean_heading_err_deg=float(np.degrees(np.mean(eh_log))) if eh_log else 0.0,
               max_contact_n=float(max_contact), contact=bool(max_contact > CONTACT_EPS_N),
               min_clearance_m=float(min_clear), steer_rate_max=float(np.abs(np.diff(acts[:, 0])).max()) if len(acts) > 1 else 0.0,
               frames=len(ct_log), wall_s=time.time() - wall0, localisation=loc_mode,
               loc_xy_mean_m=float(np.mean(loc_xy_log)) if loc_xy_log else None,
               loc_xy_p95_m=float(np.quantile(loc_xy_log, 0.95)) if loc_xy_log else None,
               loc_yaw_mean_deg=float(np.degrees(np.mean(loc_yaw_log))) if loc_yaw_log else None)
    return row


# ----------------------------------------------------------------------------- batch
def build_tasks(args) -> list[dict]:
    from nedm.traverse import nrd_data as D
    from nedm.traverse.layout import EpisodeLayout
    from nedm.traverse.oracle import PlannerParams, plan_to_ring
    from nedm.traverse.terrain import TerrainMap

    keys = D.load_cache_keys(Path(args.cache))
    split = dict(zip(("train", "val", "test"), D.split_keys(keys)))[args.split]
    manifest = json.loads((Path(args.routes) / "routes_manifest.json").read_text())
    allowed = set().union(*(set(manifest["families"][f]) for f in args.families))
    keys = [k for k in split if k in allowed][: args.episodes]
    if args.route_file:
        exported_keys = set(json.loads(Path(args.route_file).read_text()))
        keys = [k for k in keys if k in exported_keys]
    tmap = TerrainMap.from_dir(Path(args.arena))
    ref_meta = next(Path(r) / "policy_meta.json" for r in args.runs if r != "follower")
    tasks = []
    for key in keys:
        store, ep = key.split("__", 1)
        meta_path = Path(args.stores) / store / ep / "meta.json"
        if not meta_path.exists() and store == "full_v4_partial":  # newton holds the complete store
            meta_path = Path(args.stores) / "full_v4" / ep / "meta.json"
        meta = json.loads(meta_path.read_text())
        with np.load(Path(args.routes) / f"{key}.npz") as r:
            routes = [("recorded", {n: r[n].tolist() for n in ("waypoints", "speeds", "headings", "stations")})]
        if args.route_file:
            exported = json.loads(Path(args.route_file).read_text())
            routes = [(c["candidate"], {n: c[n] for n in ("waypoints", "speeds", "headings", "stations")})
                      for c in exported.get(key, []) if c["candidate"] != "recorded" or args.include_recorded]
            if args.include_recorded and not any(n == "recorded" for n, _ in routes):
                with np.load(Path(args.routes) / f"{key}.npz") as r:
                    routes.append(("recorded", {n: r[n].tolist() for n in ("waypoints", "speeds", "headings", "stations")}))
        if args.candidates:
            layout = EpisodeLayout.from_json(meta["layout"])
            seen = set()
            for name, ov in CANDIDATE_SWEEP.items():
                plan = plan_to_ring(tmap, layout.obstacles(), layout.start_xy, layout.house_xy, replace(PlannerParams(), **ov))
                if plan is None:
                    continue
                sig = (len(plan.waypoints), round(plan.length_m, 2), round(float(plan.speeds.mean()), 3))
                if sig in seen:
                    continue
                seen.add(sig)
                routes.append((name, {"waypoints": plan.waypoints.tolist(), "speeds": plan.speeds.tolist(),
                                      "headings": plan.headings.tolist(), "stations": plan.stations.tolist()}))
        for ctrl in args.runs:
            for name, route in routes:
                tasks.append({"key": key, "controller": ctrl, "candidate": name, "route": route,
                              "meta_path": str(meta_path), "arena": args.arena, "horizon_s": args.horizon_s,
                              "ref_meta": str(ref_meta), "localisation": args.localisation, "posehead": args.posehead,
                              "loc_gain_xy": args.loc_gain_xy, "loc_gain_yaw": args.loc_gain_yaw})
    return tasks


def summarize(rows: list[dict]) -> dict:
    out = {}
    for ctrl in sorted({r["controller"] for r in rows}):
        for cand in sorted({r["candidate"] for r in rows}):
            sel = [r for r in rows if r["controller"] == ctrl and r["candidate"] == cand and "status" in r]
            if not sel:
                continue
            f = lambda k: float(np.mean([r[k] for r in sel]))
            done = [r for r in sel if r["completed"]]
            out[f"{Path(ctrl).name}/{cand}"] = {
                "n": len(sel), "completed": f("completed"), "contact": f("contact"),
                "rollover": float(np.mean([r["status"] == "rollover" for r in sel])),
                "off_route": float(np.mean([r["status"] == "off_route" for r in sel])),
                "time_s_completed": float(np.mean([r["time_s"] for r in done])) if done else None,
                "energy_kj_completed": float(np.mean([r["energy_kj"] for r in done])) if done else None,
                "mean_ct_m": f("mean_ct_m"), "p95_ct_m": f("p95_ct_m"), "max_ct_m": f("max_ct_m"),
                "mean_speed_err_mps": f("mean_speed_err_mps"), "steer_rate_max": f("steer_rate_max"),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="WP3 run dirs and/or 'follower'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6")
    ap.add_argument("--routes", default="artifacts/traverse/wp3_routes")
    ap.add_argument("--stores", default="artifacts/traverse")
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--families", nargs="+", default=["oracle"])
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--candidates", action="store_true")
    ap.add_argument("--route-file", default=None, help="json from the scorer's --export-routes: drive exactly those routes")
    ap.add_argument("--include-recorded", action="store_true")
    ap.add_argument("--localisation", choices=["true", "camera", "fused"], default="true",
                    help="pose the TRACKER sees: Chrono truth (v1), per-frame camera estimate, or odometry+camera filter")
    ap.add_argument("--posehead", default="artifacts/traverse/wp4_posehead_v1_amd/ckpt_best.pt")
    ap.add_argument("--loc-gain-xy", type=float, default=0.3)
    ap.add_argument("--loc-gain-yaw", type=float, default=0.15)
    ap.add_argument("--horizon-s", type=float, default=25.0)
    ap.add_argument("--procs", type=int, default=12)
    args = ap.parse_args()

    tasks = build_tasks(args)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} Chrono runs ({args.episodes} episodes x {len(args.runs)} controllers"
          f"{' x candidates' if args.candidates else ''}), {args.procs} procs", flush=True)
    rows_path = out / "rows.jsonl"; rows_path.write_text("")
    rows = []
    t0 = time.time()
    with get_context("spawn").Pool(args.procs, maxtasksperchild=1) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, tasks)):
            rows.append(row)
            with rows_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"[{i + 1}/{len(tasks)}] {row['key'][-20:]:20s} {Path(row['controller']).name:18s} {row['candidate']:13s} "
                  f"{row['status']:10s} t={row['time_s']:5.1f}s E={row['energy_kj']:6.1f}kJ ct={row['mean_ct_m']:.3f}/{row['max_ct_m']:.2f} "
                  f"contact={row['max_contact_n']:.0f}N wall={row['wall_s']:.0f}s", flush=True)
    summary = {"args": vars(args), "wall_s": time.time() - t0, "per_controller_candidate": summarize(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["per_controller_candidate"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
