#!/usr/bin/env python
"""Roll a Go2 policy inside a frozen NRD model and render every world at every control step.

The worked example for ``nedm.render``: an articulated robot from a URDF, 1024 worlds,
torch tensors handed to the renderer without leaving the GPU. Run on an RTX 3090 and on an
AMD Instinct MI210 (docs/batch_rendering.md has the numbers).

The Go2 study is not on main yet. Point ``--quadruped`` at a checkout of the branch
``kyle/quadruped-pipeline``; this script imports its ``quadruped/finetune.py`` for the model
loader, the observation builder and the one function that steps the model.

    # once, needs the dataset: draw the branch starts
    PYTHONPATH=src python scripts/render/go2_nrd_rollout.py starts --quadruped <checkout> \\
        --model best.pt --corpus <go2_crm_v2> --out starts_1024.pt
    # roll and render
    PYTHONPATH=src python scripts/render/go2_nrd_rollout.py run --quadruped <checkout> \\
        --model best.pt --policy policy_ft.pt --starts starts_1024.pt --urdf go2_description.urdf --out out/go2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

LEGS = ("FL", "FR", "RL", "RR")
SEGMENTS = ("hip", "thigh", "calf")
FOOT_RADIUS_M = 0.022
LINK_COLORS = {"base": (0.82, 0.82, 0.85), "hip": (0.20, 0.20, 0.22), "thigh": (0.62, 0.64, 0.68),
               "calf": (0.25, 0.25, 0.28), "foot": (0.08, 0.08, 0.08)}


def link_color(link: str):
    return LINK_COLORS.get(link.split("_")[-1], LINK_COLORS["base"])


def quadruped_imports(checkout: Path):
    sys.path[:0] = [str(checkout / "quadruped"), str(checkout), str(checkout / "src")]
    import finetune  # noqa: PLC0415
    import train  # noqa: PLC0415
    return finetune, train


def cmd_starts(a) -> None:
    import torch
    import yaml
    finetune, train = quadruped_imports(a.quadruped)
    checkpoint = torch.load(a.model, map_location="cpu", weights_only=False)
    context = checkpoint["config"]["block_size"]
    params = a.quadruped / "quadruped" / "params"
    control_dt = 1.0 / float(yaml.safe_load((params / "excitation.yaml").read_text())["episode"]["control_hz"])
    policy_cfg = yaml.safe_load((params / "policy.yaml").read_text())
    corpus = train.Corpus(Path(a.corpus), checkpoint["config"]["preset"], context, extra_fields=finetune.EXTRA_FIELDS)
    pool = finetune.build_pool(corpus, context, control_dt)
    batch = finetune.start_batch(torch, corpus, pool, a.envs, random.Random(a.seed), context,
                                 list(policy_cfg["joints"]["policy_to_chrono"]), "cpu")
    batch.pop("picks")
    torch.save({**batch, "state_fields": checkpoint["state_fields"], "seed": a.seed, "corpus": str(a.corpus)}, a.out)
    print(f"wrote {a.out}: {a.envs} starts drawn from {len(pool):,} control rows")


def cmd_run(a) -> None:
    import torch
    import yaml
    if a.backend == "newton":
        import warp as wp
        wp.config.quiet = True

    from nedm.render import (BatchRenderer, CollageRecorder, PlanarPose, Scene, base_transform, cameras,
                                  depth_to_gray, save_sheet, tilt_from_gravity)
    finetune, _train = quadruped_imports(a.quadruped)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model, checkpoint = finetune.load_nnrom(torch, a.model, device)
    fields = checkpoint["state_fields"]
    column = {name: i for i, name in enumerate(fields)}
    params = a.quadruped / "quadruped" / "params"
    policy_cfg = yaml.safe_load((params / "policy.yaml").read_text())
    excitation = yaml.safe_load((params / "excitation.yaml").read_text())
    control_dt = 1.0 / float(excitation["episode"]["control_hz"])
    dt_s = float(checkpoint["config"].get("dt_s") or 1.0 / float(excitation["episode"]["record_hz"]))
    hold = int(round(control_dt / dt_s))
    observe = finetune.ObsBuilder(torch, fields, policy_cfg).to(device)
    policy = torch.jit.load(str(a.policy), map_location=device).eval()

    starts = torch.load(a.starts, map_location=device, weights_only=False)
    if starts["state_fields"] != fields:
        raise SystemExit("the starts and the model disagree about the state layout")
    n = min(a.envs, starts["states"].shape[0])
    hist_s, hist_a = starts["states"][:n], starts["acts"][:n]
    last_raw, command = starts["last_raw"][:n], starts["cmd"][:n]
    steps = int(round(a.seconds / control_dt))

    # ---- the renderer: everything specific to this robot is in these few lines -------------
    t0 = time.perf_counter()
    scene = Scene.from_urdf(a.urdf, floating=True, colors=link_color).add_ground()
    renderer = BatchRenderer(scene, n, width=a.res, height=a.res, shadows=not a.no_shadows,
                             device="cuda:0" if device.type == "cuda" else "cpu", backend=a.backend)
    build_s = time.perf_counter() - t0
    # URDF joint angle = sign * Chrono joint angle (quadruped/params/policy.yaml), matched by name.
    sign = float(policy_cfg["sign"]["value"])
    joint_map = renderer.joint_map(fields, {f"{leg}_{seg}_joint": (f"joint_{leg.lower()}_{seg}_pos_rad", sign)
                                            for leg in LEGS for seg in SEGMENTS})
    feet = [i for i, name in enumerate(renderer.body_names) if name.endswith("_foot")]
    gravity = [column[f"grav_body_{axis}"] for axis in "xyz"]
    velocity = [column["vel_body_x_mps"], column["vel_body_y_mps"], column["yaw_rate_radps"]]
    pose = PlanarPose(n, dt_s, device)
    ground_z = 0.0

    def joint_q(state):
        roll, pitch = tilt_from_gravity(state[:, gravity])
        return joint_map.joint_q(state, base_transform(pose.xy_yaw, state[:, column["pos_z_m"]] - ground_z, roll, pitch))
    # ----------------------------------------------------------------------------------------

    fps = int(round(1.0 / control_dt))
    rgb_video = CollageRecorder(out / "collage_rgb.mp4", n, a.sample, fps=fps, seed=a.seed)
    depth_video = CollageRecorder(out / "collage_depth.mp4", n, a.sample, fps=fps, seed=a.seed)
    kept = []

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        # pos_z_m is in the soil bed's frame, not height above the ground: find the ground.
        renderer.set_joint_q(joint_q(hist_s[:, -1]))
        ground_z = float(renderer.body_q()[:, feet, 2].min(dim=1).values.quantile(0.5)) - FOOT_RADIUS_M
        print(f"ground at z = {ground_z:.3f} m in the data's frame (median lowest foot)")

        trajectory = np.zeros((steps + 1, n, renderer.coord_count), dtype=np.float32)
        model_s = render_s = first_s = 0.0
        for c in range(steps + 1):
            if c > 0:
                sync(); t = time.perf_counter()
                obs = observe.observe(hist_s[:, -1], command, last_raw)
                last_raw = torch.clamp(policy(obs), *observe.act_clip)
                action = observe.action_from_raw(last_raw)
                for _ in range(hold):
                    nxt, hist_s, hist_a = finetune.advance(model, torch, hist_s, hist_a, action)
                    pose.step(*(nxt[:, i] for i in velocity))
                sync(); model_s += time.perf_counter() - t
            q = joint_q(hist_s[:, -1])
            sync(); t = time.perf_counter()
            frames = renderer.render(cameras.follow(pose.xy_yaw[:, :2].float()), joint_q=q)
            elapsed = time.perf_counter() - t
            if c == 0:
                first_s = elapsed       # includes kernel compilation and the BVH build
            else:
                render_s += elapsed
            trajectory[c] = q.cpu().numpy()
            rgb, depth = frames.rgb[:, 0], frames.depth[:, 0]
            rgb_video.add(rgb)
            depth_video.add(depth_to_gray(depth))
            if c % 25 == 0:
                kept.append((rgb[rgb_video.pick].copy(), depth[rgb_video.pick].copy()))
            if c == steps // 2:
                save_sheet(rgb, out / "all_worlds_midframe.png")
            if c % 50 == 0:
                print(f"step {c:4d}/{steps}  model {1e3 * model_s / max(c, 1):6.1f} ms/step  "
                      f"render {1e3 * render_s / max(c, 1):6.1f} ms/frame", flush=True)
    rgb_video.close()
    depth_video.close()
    np.savez_compressed(out / "sample_frames.npz", pick=rgb_video.pick, rgb=np.stack([k[0] for k in kept]),
                        depth=np.stack([k[1] for k in kept]))
    np.savez_compressed(out / "traj.npz", joint_q=trajectory, cam_dt=control_dt, z_offset=ground_z)

    final = hist_s[:, -1]
    finite = torch.isfinite(final).all(dim=1)
    info = {
        "envs": n, "res": a.res, "seconds": a.seconds, "control_steps": steps, "torch": torch.__version__,
        "torch_device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "backend": a.backend, "render_device": str(renderer.device), "interop": renderer.interop,
        "scene_build_s": round(build_s, 2), "first_frame_s": round(first_s, 2),
        "model_ms_per_control_step": round(1e3 * model_s / steps, 2),
        "render_ms_per_frame": round(1e3 * render_s / steps, 2),
        "views_per_s": round(n * steps / render_s),
        "finite_at_end": int(finite.sum()),
        "upright_at_end": int((finite & (final[:, column["grav_body_z"]] < -0.5)).sum()),
        "sampled_envs": rgb_video.pick.tolist(),
    }
    (out / "run.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "sampled_envs"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("starts", cmd_starts), ("run", cmd_run)):
        p = sub.add_parser(name)
        p.add_argument("--quadruped", type=Path, required=True, help="checkout of kyle/quadruped-pipeline")
        p.add_argument("--model", required=True)
        p.add_argument("--envs", type=int, default=1024)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", required=True)
        p.set_defaults(fn=fn)
        if name == "starts":
            p.add_argument("--corpus", required=True)
        else:
            p.add_argument("--policy", required=True)
            p.add_argument("--starts", required=True)
            p.add_argument("--urdf", type=Path, required=True)
            p.add_argument("--seconds", type=float, default=5.0)
            p.add_argument("--res", type=int, default=128)
            p.add_argument("--sample", type=int, default=50, help="worlds in the collage")
            p.add_argument("--device", default="cuda")
            p.add_argument("--no-shadows", action="store_true")
            p.add_argument("--backend", choices=["newton", "madrona"], default="newton")
    a = parser.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
