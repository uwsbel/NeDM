#!/usr/bin/env python
"""Roll the paper's arm reaching policy inside its NRD model and render every world.

The second way to plug a model in, on a real rollout: no URDF. The reduced state holds the
four joint angles, ``ArmKinematics`` turns them into a pose for every link, and the renderer
needs only the link meshes and those poses.

It uses the released artifacts as they are: the Study Case II NRD model with the 8-D reduced
state, the reaching policy at its last iteration and the environment config saved with that
run. The start states come from the processed training cache, which is not in the repository:

    PYTHONPATH=src python scripts/core/release/download_nedm_datasets.py --dataset arm --no-raw --processed

    PYTHONPATH=src python scripts/render/arm_nrd_rollout.py --dynamics best_val.pt --policy model_1499.pt \\
        --out out/arm --backend madrona

A ball marks each world's goal, red until the grasp center comes within the success tolerance
and green from then on. It is drawn larger than the tolerance so that it can be seen.

An episode ends the moment the goal is reached, which is how the policy was trained: it never
learned to hold a pose. A world that has reached its goal therefore keeps showing its last
frame until every world's episode is over. ``--episodes`` runs more of them, each from a new
start state and a new goal, so the video cuts between episodes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from arm_scene import ARM_SCALE, REPO, arm_scene

RUN = REPO / "artifacts/rl_runs/arm_reach_adaptivekl005_lr1e4_tol005_ep150_bonus150_sigma015_8d_rom_20260727"
OPEN_COLOR, REACHED_COLOR = (0.90, 0.15, 0.10), (0.10, 0.80, 0.25)
CAMERA_EYE, CAMERA_TARGET = (4.5, -10.5, 3.4), (-0.4, 0.0, 2.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dynamics", type=Path, required=True, help="the arm NRD model checkpoint (best_val.pt)")
    parser.add_argument("--policy", type=Path, required=True, help="the reaching policy checkpoint (model_1499.pt)")
    parser.add_argument("--run-dir", type=Path, default=RUN, help="RL run folder holding env_cfg.json and train_cfg.json")
    parser.add_argument("--processed", type=Path, default=REPO / "artifacts/training_datasets/arm_dyn_v3_8d_seq16_v1",
                        help="processed cache the start states are drawn from")
    parser.add_argument("--geometry", type=Path, default=REPO / "artifacts/arm_geometry/arm_geometry_v1.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=1, help="episodes per world, each from a fresh reset")
    parser.add_argument("--marker-radius", type=float, default=0.2, help="radius of the goal ball, m")
    parser.add_argument("--arm-scale", type=float, default=ARM_SCALE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-shadows", action="store_true")
    parser.add_argument("--backend", choices=["newton", "madrona"], default="madrona")
    parser.add_argument("--seed", type=int, default=0, help="fixes the start states, the goals and the collage sample")
    a = parser.parse_args()

    import trimesh
    from rsl_rl.runners import OnPolicyRunner

    from nedm.render import (BatchRenderer, CollageRecorder, MeshBody, cameras, depth_to_gray, save_sheet,
                             transform_from_matrix)
    from nedm.tracked_arm.rl.arm_reaching_env import ArmReachingEnv
    if a.backend == "newton":
        import warp as wp
        wp.config.quiet = True
    device = a.device if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- the NRD environment and its policy, as the run saved them -----------------------------
    env_cfg = json.loads((a.run_dir / "env_cfg.json").read_text())
    env_cfg.update(num_envs=a.envs, dynamics_checkpoint=str(a.dynamics), processed_dataset_dir=str(a.processed),
                   geometry_path=str(a.geometry), device=device, auto_reset=False)
    torch.manual_seed(a.seed)                   # the environment draws a start state and a goal for each world
    env = ArmReachingEnv(env_cfg, device=device)
    runner = OnPolicyRunner(env, json.loads((a.run_dir / "train_cfg.json").read_text()), log_dir=None, device=device)
    checkpoint = torch.load(a.policy, map_location=device, weights_only=False)
    runner.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
    runner.obs_normalizer.load_state_dict(checkpoint["obs_norm_state_dict"])
    policy = runner.get_inference_policy(device=device)
    n, dt, kin = env.num_envs, env.dt_s, env.kin
    tolerance = float(env.cfg["reward"]["success_tolerance_m"])

    # ---- the renderer: everything specific to the arm is in this block and in arm_scene.py ------
    ball = trimesh.creation.icosphere(subdivisions=2, radius=a.marker_radius)
    vertices, faces = np.asarray(ball.vertices, dtype=np.float32), np.asarray(ball.faces, dtype=np.int32)
    markers = [MeshBody("goal_open", vertices=vertices, faces=faces, color=OPEN_COLOR),
               MeshBody("goal_reached", vertices=vertices, faces=faces, color=REACHED_COLOR)]
    t0 = time.perf_counter()
    renderer = BatchRenderer(arm_scene(kin, a.arm_scale, extra_bodies=markers), n, width=a.res, height=a.res,
                             shadows=not a.no_shadows, device="cuda:0" if device == "cuda" else "cpu",
                             backend=a.backend)
    build_s = time.perf_counter() - t0
    hidden = torch.tensor([0.0, 0.0, -50.0], device=device)         # under the ground, where no camera ray goes

    def ee_error() -> torch.Tensor:
        return torch.linalg.norm(env.current_ee_base() - env.goal_base, dim=1)

    shown_q = env.current_q().clone()           # the joint angles drawn: frozen once a world's episode has ended
    reached = torch.zeros(n, 1, dtype=torch.bool, device=device)

    def body_q() -> torch.Tensor:
        links = transform_from_matrix(kin.link_transforms(shown_q))                         # (n, links, 7)
        goal = env.goal_base @ kin.base_to_world[:3, :3].T + kin.base_to_world[:3, 3]       # arm-base frame -> world
        marker = torch.zeros(n, 2, 7, device=device)
        marker[..., 6] = 1.0
        marker[:, 0, :3] = torch.where(reached, hidden, goal)       # two balls, one color each: show one, hide the other
        marker[:, 1, :3] = torch.where(reached, goal, hidden)
        return torch.cat([links, marker], dim=1)
    # --------------------------------------------------------------------------------------------

    fps = int(round(1.0 / (dt * env.action_repeat)))
    rgb_video = CollageRecorder(out / "collage_rgb.mp4", n, sample=min(n, 50), cols=min(n, 10), fps=fps, seed=a.seed)
    depth_video = CollageRecorder(out / "collage_depth.mp4", n, sample=min(n, 50), cols=min(n, 10), fps=fps, seed=a.seed)
    pick = rgb_video.pick
    camera = cameras.look_at(torch.tensor([CAMERA_EYE], device=device).repeat(n, 1), torch.tensor([CAMERA_TARGET], device=device))
    timing = {"render": 0.0, "model": 0.0, "frames": 0, "substeps": 0}
    kept = []                                   # a few raw frames, to compare renderers or machines
    steps = env.max_episode_length
    total_frames = a.episodes * (steps + 1)

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    def draw() -> None:
        t = time.perf_counter()
        frames = renderer.render(camera, body_q=body_q())
        if timing["frames"] > 0:                    # the first frame includes start-up
            timing["render"] += time.perf_counter() - t
        rgb, depth = frames.rgb[:, 0], frames.depth[:, 0]
        rgb_video.add(rgb)
        depth_video.add(depth_to_gray(depth, near=2.0, far=25.0))
        if timing["frames"] % 25 == 0:
            kept.append((rgb[pick].copy(), depth[pick].copy()))
        if timing["frames"] == total_frames // 2:
            save_sheet(rgb, out / "all_worlds_midframe.png")
        timing["frames"] += 1

    model_substep = env._nn_substep

    def substep(raw_dq):                            # wrapped only to time the model
        sync()
        t = time.perf_counter()
        result = model_substep(raw_dq)
        sync()
        timing["model"] += time.perf_counter() - t
        timing["substeps"] += 1
        return result

    env._nn_substep = substep
    start_error, final_error, reach_step = [], [], []
    t_start = time.perf_counter()
    with torch.no_grad():
        for episode in range(a.episodes):
            obs, _ = env.reset()
            shown_q.copy_(env.current_q())
            reached.zero_()
            error_at_end = ee_error()
            start_error.append(error_at_end.clone())
            reached_at = torch.full((n,), -1, dtype=torch.long, device=device)
            draw()
            for step in range(steps):
                obs, _reward, _done, _extras = env.step(policy(obs))
                live = ~reached[:, 0]
                shown_q[live] = env.current_q()[live]
                error_at_end[live] = ee_error()[live]
                newly = live & (error_at_end < tolerance)
                reached_at[newly] = step + 1
                reached[newly] = True
                draw()
            final_error.append(error_at_end.clone())
            reach_step.append(reached_at)
            print(f"episode {episode + 1}/{a.episodes}: {int(reached.sum())} of {n} reached", flush=True)
    wall = time.perf_counter() - t_start
    start_error, final_error, reach_step = torch.cat(start_error), torch.cat(final_error), torch.cat(reach_step)
    got_there = reach_step > 0
    rgb_video.close()
    depth_video.close()
    np.savez_compressed(out / "sample_frames.npz", pick=pick, rgb=np.stack([k[0] for k in kept]),
                        depth=np.stack([k[1] for k in kept]))

    frames_timed = timing["frames"] - 1
    info = {
        "envs": n, "res": a.res, "backend": a.backend, "episodes": a.episodes,
        "seconds_per_episode": round(steps * env.action_repeat * dt, 2),
        "model_dt_s": dt, "success_tolerance_m": tolerance, "marker_radius_m": a.marker_radius,
        "frames": timing["frames"], "fps": fps, "interop": renderer.interop,
        "scene_build_s": round(build_s, 2),
        "render_ms_per_frame": round(1e3 * timing["render"] / frames_timed, 2),
        "views_per_s": round(n * frames_timed / timing["render"]),
        "model_ms_per_step": round(1e3 * timing["model"] / timing["substeps"], 2),
        "wall_s": round(wall, 1),
        "reached": int(got_there.sum()), "attempts": int(got_there.numel()),
        "median_start_error_m": round(float(start_error.median()), 3),
        "median_final_error_m": round(float(final_error.median()), 4),
        "worst_final_error_m": round(float(final_error.max()), 3),
        "median_time_to_reach_s": round(float(reach_step[got_there].float().median()) * env.action_repeat * dt, 2),
        "finite_at_end": int(torch.isfinite(env.current_state()).all(dim=1).sum()),
        "collage_worlds": pick.tolist(),
    }
    (out / "run.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "collage_worlds"}, indent=2))


if __name__ == "__main__":
    main()
