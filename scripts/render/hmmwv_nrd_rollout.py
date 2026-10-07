#!/usr/bin/env python
"""Roll the paper's HMMWV tracking policy inside its NRD model and render every world.

The third way to plug a model in: a vehicle. There is no URDF and no kinematics to borrow.
The chassis pose is the environment's integrated planar pose plus the roll and pitch in the
reduced state, and each wheel is a fixed mount on the chassis, turned by the steering command
and spun by the wheel speed the model predicts.

It uses the released artifacts as they are: the Study Case I NRD model, the mixture policy
at iteration 1000, the 20 rest-start validation references and the rigid-terrain evaluation
config. The environment is not modified. One method is wrapped so a frame can be drawn
between model steps instead of only once per policy step.

    PYTHONPATH=src python scripts/render/hmmwv_nrd_rollout.py --dynamics best_val.pt --policy model_1000.pt \\
        --meshes <chrono>/data/vehicle/hmmwv --out out/hmmwv --backend madrona

The reduced state carries no height and no wheel angles. The ride height, wheel mounts, tire
radius and steering range below were measured in Chrono with the vehicle settings NeDM
collects data with (HMMWV_Full, TMEASY tires, pitman-arm steering, settled on rigid ground).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / ("artifacts/rl_runs/hmmwv_rl_15d_crm2000mix25_onehot_ofatL8_bestval51_flat20crm20_K16_64steps_"
              "ar02_state_vxvyyr_pos2_yaw2_steerlim010")
REFERENCES = REPO / "artifacts/rl_reference_sets/hmmwv_tire_normal_force_omega_val_refs_20_1100_rest_start.npz"

RIDE_HEIGHT_M = 0.5885                      # chassis reference frame above flat ground, at rest
WHEEL_MOUNT = {                             # wheel centres in the chassis frame, at rest
    "fl": (1.6564, 0.8737, -0.1305), "fr": (1.6563, -0.8737, -0.1305),
    "rl": (-1.6541, 0.8736, -0.1384), "rr": (-1.6541, -0.8736, -0.1384),
}
# Full steering turns the inner front wheel 29.3 deg and the outer 25.1 deg. Both wheels are
# drawn at the mean, which is wrong by up to 2 deg per wheel and invisible at this scale.
MAX_STEER_RAD = math.radians(27.2)
CHASSIS_COLOR, TIRE_COLOR, RIM_COLOR, TARGET_COLOR = (0.42, 0.45, 0.30), (0.08, 0.08, 0.08), (0.55, 0.55, 0.58), (0.10, 0.85, 0.25)


def arrow_mesh(length=2.4, width=1.2, thickness=0.04):
    """A flat arrowhead pointing along +X, drawn on the ground at the reference pose."""
    half, tip = width / 2, length / 2
    outline = [(tip, 0.0), (-tip, half), (-tip * 0.5, 0.0), (-tip, -half)]
    vertices = np.array([(x, y, z) for z in (0.0, thickness) for x, y in outline], dtype=np.float32)
    faces = [(4, 5, 6), (4, 6, 7), (0, 2, 1), (0, 3, 2)]           # top, then bottom, both facing outward
    for i in range(4):
        j = (i + 1) % 4
        faces += [(i, j, 4 + j), (i, 4 + j, 4 + i)]                 # the outline runs counter-clockwise
    return vertices, np.array(faces, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dynamics", type=Path, required=True, help="the HMMWV NRD model checkpoint (best_val.pt)")
    parser.add_argument("--policy", type=Path, required=True, help="the tracking policy checkpoint (model_1000.pt)")
    parser.add_argument("--meshes", type=Path, required=True, help="Chrono's data/vehicle/hmmwv folder")
    parser.add_argument("--run-dir", type=Path, default=RUN, help="RL run folder holding train_cfg.json and the eval config")
    parser.add_argument("--references", type=Path, default=REFERENCES)
    parser.add_argument("--out", required=True)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--every", type=int, default=2, help="draw one frame per this many 10 ms model steps")
    parser.add_argument("--steering-rate-limit", type=float, default=0.1, help="per policy step, as in training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-shadows", action="store_true")
    parser.add_argument("--backend", choices=["newton", "madrona"], default="madrona")
    parser.add_argument("--seed", type=int, default=0, help="fixes which reference each world follows")
    a = parser.parse_args()

    from rsl_rl.runners import OnPolicyRunner

    from nedm.hmmwv.rl.hmmwv_tracking_env import HMMWVNeuralTrackingEnv
    from nedm.render import (BatchRenderer, CollageRecorder, MeshBody, Scene, base_transform, cameras, compose,
                             depth_to_gray, quat_from_rpy, save_sheet)
    if a.backend == "newton":
        import warp as wp
        wp.config.quiet = True
    device = a.device if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- the NRD environment and its policy, exactly as the rigid-terrain evaluation builds them
    env_cfg = json.loads((a.run_dir / "eval_cfg_rigid20_val_rest_start_flatkey_pre0" / "env_cfg.json").read_text())
    # The policy was trained with a steering rate limit and the reported evaluation applies it
    # (steering_rate_limit 0.1 in the run's chrono_eval_*_steerlim010_pre0/summary.json). The saved
    # eval config leaves it out, and without it the same policy drifts 20 m off the reference.
    env_cfg.update(num_envs=a.envs, dynamics_checkpoint=str(a.dynamics), reference_path=str(a.references),
                   device=device, auto_reset=False, steering_rate_limit=a.steering_rate_limit)
    torch.manual_seed(a.seed)                   # the environment draws a reference for each world
    env = HMMWVNeuralTrackingEnv(env_cfg, device=device)
    runner = OnPolicyRunner(env, json.loads((a.run_dir / "train_cfg.json").read_text()), log_dir=None, device=device)
    checkpoint = torch.load(a.policy, map_location=device, weights_only=False)
    runner.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
    runner.obs_normalizer.load_state_dict(checkpoint["obs_norm_state_dict"])
    policy = runner.get_inference_policy(device=device)
    n, dt = env.num_envs, env.dt_s
    column = env.state_index

    # ---- the renderer: everything specific to this vehicle is in this block -------------------
    wheels = list(WHEEL_MOUNT)
    bodies = [MeshBody("chassis", path=a.meshes / "hmmwv_chassis.obj", color=CHASSIS_COLOR)]
    for wheel in wheels:
        side = "left" if wheel.endswith("l") else "right"
        bodies.append(MeshBody(f"tire_{wheel}", path=a.meshes / f"hmmwv_tire_{side}.obj", color=TIRE_COLOR))
        bodies.append(MeshBody(f"rim_{wheel}", path=a.meshes / "hmmwv_rim.obj", color=RIM_COLOR))
    vertices, faces = arrow_mesh()
    bodies.append(MeshBody("target", vertices=vertices, faces=faces, color=TARGET_COLOR))
    reference_xy = env.reference_poses[..., :2].reshape(-1, 2)
    low, high = reference_xy.min(0).values.cpu().numpy() - 30.0, reference_xy.max(0).values.cpu().numpy() + 30.0
    scene = Scene.from_meshes(bodies).add_ground(tile_size=4.0, extent=(float(low[0]), float(high[0]), float(low[1]), float(high[1])))
    t0 = time.perf_counter()
    renderer = BatchRenderer(scene, n, width=a.res, height=a.res, shadows=not a.no_shadows,
                             device="cuda:0" if device == "cuda" else "cpu", backend=a.backend)
    build_s = time.perf_counter() - t0

    mount = torch.zeros(len(wheels), 7, device=device)
    mount[:, :3] = torch.tensor([WHEEL_MOUNT[w] for w in wheels], device=device)
    front = torch.tensor([w.startswith("f") for w in wheels], device=device)
    omega = [column[f"tire_{w}_spindle_omega_radps"] for w in wheels]
    spin = torch.zeros(n, len(wheels), device=device)            # wheel angles are not in the state: integrate them
    lead_steps = int(round(1.0 / dt))

    def body_q() -> torch.Tensor:
        state = env.current_state()
        chassis = base_transform(env.current_pose(), RIDE_HEIGHT_M, state[:, column["roll_rad"]], state[:, column["pitch_rad"]])
        steer = (env.actions[:, :1] * MAX_STEER_RAD) * front                    # (n, wheels): rear wheels do not steer
        local = mount.expand(n, -1, -1).clone()
        local[..., 3:] = quat_from_rpy(torch.zeros_like(spin), spin, steer)     # steer about Z, then roll about the axle
        wheel = compose(chassis[:, None, :].expand(-1, len(wheels), -1), local)
        # The arrow is where the reference will be one second from now. At the current reference
        # pose it would sit under a vehicle that is tracking well, and show nothing.
        ahead = torch.clamp(env.ref_step_buf + lead_steps, max=env.reference_length - 1)
        target = base_transform(env.reference_poses[env.ref_ids, ahead], 0.02)
        return torch.cat([chassis[:, None], wheel.repeat_interleave(2, dim=1), target[:, None]], dim=1)    # tire and rim share a pose
    # --------------------------------------------------------------------------------------------

    # one world per reference in the collage: with a deterministic policy the rest are repeats
    first_of = {}
    for world, ref in enumerate(env.ref_ids.tolist()):
        first_of.setdefault(ref, world)
    pick = np.array([first_of[r] for r in sorted(first_of)])
    fps = int(round(1.0 / (dt * a.every)))
    rgb_video = CollageRecorder(out / "collage_rgb.mp4", n, cols=5, fps=fps, pick=pick)
    depth_video = CollageRecorder(out / "collage_depth.mp4", n, cols=5, fps=fps, pick=pick)
    follow = dict(eye_offset=(-7.5, -8.5, 5.0), look_offset=(1.0, 0.0, 0.8))
    timing = {"render": 0.0, "model": 0.0, "frames": 0, "substeps": 0}
    kept = []                                   # a few raw frames, to compare renderers or machines
    total_frames = env.max_episode_length * env.action_repeat // a.every

    def draw() -> None:
        t = time.perf_counter()
        frames = renderer.render(cameras.follow(env.current_pose()[:, :2], **follow), body_q=body_q())
        if timing["frames"] > 0:                    # the first frame includes start-up
            timing["render"] += time.perf_counter() - t
        rgb, depth = frames.rgb[:, 0], frames.depth[:, 0]
        rgb_video.add(rgb)
        depth_video.add(depth_to_gray(depth, near=2.0, far=40.0))
        if timing["frames"] % 50 == 0:
            kept.append((rgb[pick].copy(), depth[pick].copy()))
        if timing["frames"] == total_frames // 2:
            save_sheet(rgb, out / "all_worlds_midframe.png")
        timing["frames"] += 1

    model_substep = env._nn_substep

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    def substep(driver_actions) -> None:            # the one wrapped method: draw between model steps
        sync()
        t = time.perf_counter()
        model_substep(driver_actions)
        sync()
        timing["model"] += time.perf_counter() - t
        spin.add_(env.current_state()[:, omega] * dt)
        timing["substeps"] += 1
        if timing["substeps"] % a.every == 0:
            draw()

    env._nn_substep = substep
    draw()
    error_sum = torch.zeros(n, device=device)
    obs, _ = env.get_observations()
    t_start = time.perf_counter()
    with torch.no_grad():
        for step in range(env.max_episode_length):
            obs, _reward, _done, _extras = env.step(policy(obs))
            reference = env.current_reference_state_pose()[1]
            error_sum += torch.linalg.norm(reference[:, :2] - env.current_pose()[:, :2], dim=1)
            if step % 30 == 0:
                print(f"policy step {step:3d}/{env.max_episode_length}  render {1e3 * timing['render'] / max(timing['frames'] - 1, 1):6.1f} ms/frame", flush=True)
    wall = time.perf_counter() - t_start
    rgb_video.close()
    depth_video.close()
    np.savez_compressed(out / "sample_frames.npz", pick=pick, rgb=np.stack([k[0] for k in kept]),
                        depth=np.stack([k[1] for k in kept]))

    state, frames_timed = env.current_state(), timing["frames"] - 1
    mean_error = error_sum / env.max_episode_length
    info = {
        "envs": n, "res": a.res, "backend": a.backend, "seconds": round(env.max_episode_length * env.step_dt, 2),
        "model_dt_s": dt, "frames": timing["frames"], "fps": fps, "interop": renderer.interop,
        "scene_build_s": round(build_s, 2),
        "render_ms_per_frame": round(1e3 * timing["render"] / frames_timed, 2),
        "views_per_s": round(n * frames_timed / timing["render"]),
        "model_ms_per_substep": round(1e3 * timing["model"] / timing["substeps"], 2),
        "wall_s": round(wall, 1),
        "references": len(pick),
        "mean_position_error_m": round(float(mean_error.mean()), 3),
        "worst_reference_mean_error_m": round(float(mean_error[pick].max()), 3),
        "finite_at_end": int(torch.isfinite(state).all(dim=1).sum()),
        "collage_worlds": pick.tolist(),
    }
    (out / "run.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "collage_worlds"}, indent=2))


if __name__ == "__main__":
    main()
