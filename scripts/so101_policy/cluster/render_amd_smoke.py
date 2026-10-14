"""Combined smoke on ONE GPU, ONE process: the converged NRD world (GraspNrdSim inside PickPlaceVecEnv) driven by the
state PPO teacher, the batch renderer (Newton / Warp backend) drawing every decision's state from the top and wrist
cameras, and a camera student trained online on those images (build8_amd_render.md, step 3).

The student follows the user's reformulation: inputs = top + wrist RGB + the arm's q (6) and qdot (6) only; output =
6 joint targets (5 arm + jaw), label = the joint targets the teacher's decoder sent to the world in the LAST 20 ms step
of the decision (the decoder is teacher-side only). This is a smoke (it shows that torch and Warp share the HIP runtime
and that the three parts run together, with timings and memory), not a training run.

    python3.12 -P render_amd_smoke.py --nrd nrd_c1_s62.pt --teacher nC1_s62_b_model_00171.pt \
        --tasks tasks_dagger_8192.npz --rows 0:256 --res 128 --assets so101_assets.npz --out smoke_dir

Tasks: DAgger rows (seeds 510,000+); never gate / gate2 / test seeds (``teacher.check_seeds``).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def gpu_used_gb():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2 ** 30


def make_env(B, teacher, nrd, graph):
    """``vision.teacher.make_env`` (the evaluation env of the teacher's decoder, auto reset off, 8 s horizon) with the
    CUDA-graph switch exposed."""
    from nedm.so101_policy.vec_env import EnvConfig, PickPlaceVecEnv
    cfg = EnvConfig(world="nrd", num_envs=B, device="cuda", ckpts=[str(nrd)], members="split", auto_reset=False,
                    eval_rules=True, task=dict(horizon=400, obs_rates=teacher.obs_rates), starts="",
                    nrd_cuda_graph=bool(graph), decoder=dict(teacher.decoder_cfg), **teacher.files)
    env = PickPlaceVecEnv(cfg)
    assert env.num_obs == teacher.obs_dim and env.num_actions == teacher.act_dim
    return env


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nrd", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--rows", default="0:256")
    p.add_argument("--assets", default="")
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--no-shadows", action="store_true")
    p.add_argument("--goal", default="tray", help="goal marker style (tray / plate / frame)")
    p.add_argument("--order", default="tiled", choices=["pixel", "view", "tiled"],
                   help="Newton ray order (tiled 8x8: 3x faster than pixel at 288 px on MI350X)")
    p.add_argument("--tile", default="8x8")
    p.add_argument("--decisions", type=int, default=81, help="decisions per episode batch (8 s horizon = 80)")
    p.add_argument("--train-batch", type=int, default=128, help="samples per training step (fixed)")
    p.add_argument("--train-per-decision", type=int, default=1, help="training steps after each decision")
    p.add_argument("--no-graph", action="store_true", help="NRD step without the CUDA (HIP) graph")
    p.add_argument("--amp", default="bf16", choices=["bf16", "off"])
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    from nedm.render.so101 import SO101Assets, SO101Renderer
    from nedm.so101_policy.vision import dataset as V
    from nedm.so101_policy.vision import demos as D
    from nedm.so101_policy.vision import teacher as TT
    from nedm.so101_policy.vision.perstep import PerStepConfig, PerStepStudent
    import warp as wp

    lo, hi = (int(x) for x in a.rows.split(":"))
    t = D.load_tasks(a.tasks, np.arange(lo, hi))
    TT.check_seeds(t["seeds"])
    B = len(t["seeds"])
    dev = torch.device("cuda")
    info = dict(host=socket.gethostname(), device=torch.cuda.get_device_name(0), torch=torch.__version__,
                warp=wp.__version__, hip=getattr(torch.version, "hip", None), worlds=B, res=a.res,
                seeds=[int(t["seeds"][0]), int(t["seeds"][-1])], job=os.environ.get("SLURM_JOB_ID", ""),
                ld_preload=os.environ.get("LD_PRELOAD", ""), graph=not a.no_graph, goal=a.goal, order=a.order,
                shadows=not a.no_shadows)
    mem0 = gpu_used_gb()

    # 1. teacher + NRD world (GraspNrdSim, converged NRD c1 s62)
    t0 = time.time()
    teacher = TT.Teacher(a.teacher, device="cuda")
    env = make_env(B, teacher, a.nrd, not a.no_graph)
    us = []
    world_step = env.world.step

    def step_hook(u):
        us.append(u)
        return world_step(u)

    env.world.step = step_hook
    g = lambda k: torch.as_tensor(t[k], dtype=torch.float64, device=dev)         # noqa: E731
    goal3 = torch.stack([g("goal_xy")[:, 0], g("goal_xy")[:, 1], g("goal_yaw")], -1)
    obs = env.reset_tasks(torch.arange(B, device=dev), D.start_state(t, dev), g("goal_xy"), g("goal_yaw"),
                          torch.as_tensor(t["seeds"], dtype=torch.long, device=dev))
    info["setup_env_s"] = round(time.time() - t0, 2)

    # 2. renderer (Newton / Warp), the student's cameras
    t0 = time.time()
    ass = SO101Assets.load(a.assets) if a.assets else None
    rend = SO101Renderer(B, a.res, a.res, backend="newton", shadows=not a.no_shadows, assets=ass, cameras=V.CAMERAS,
                         top_eye=V.TOP_EYE, top_target=V.TOP_TARGET, goal=a.goal)
    if a.order != "pixel":      # Newton's RenderConfig, read when the render kernel is built (first render)
        from newton._src.sensors.warp_raytrace.types import RenderOrder
        rc = rend.renderer.backend._camera.render_config
        rc.render_order = RenderOrder.TILED if a.order == "tiled" else RenderOrder.VIEW_PRIORITY
        rc.tile_width, rc.tile_height = (int(v) for v in a.tile.split("x"))
    info["setup_render_s"] = round(time.time() - t0, 2)

    # 3. camera student: images + q + qdot -> 6 joint targets
    torch.manual_seed(0)
    cfg = PerStepConfig(res=a.res, cameras=V.CAMERAS, act_dim=6, robot_dim=12, amp=a.amp == "bf16")
    student = PerStepStudent(cfg, dict(robot_mean=torch.zeros(12), robot_std=torch.ones(12)), pretrained=False).to(dev)
    opt = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=1e-4)
    n_params = sum(p_.numel() for p_ in student.parameters())

    keep, sheet = None, []
    tm = dict(render=0.0, world=0.0, train=0.0)
    losses, step_ms, views = [], [], 0
    buf, n_buf = None, 0                             # every live (state, label) of the run, images uint8 on the GPU
    torch.cuda.synchronize()
    mem_peak = gpu_used_gb()
    for k in range(a.decisions):
        live = env.task.tr.end == 0
        if not bool(live.any()):
            break
        ws = env.ws
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        rgb = rend.render(ws.q, ws.cube_pos, ws.cube_quat, goal3)                       # [B, 2, H, W, 3] uint8
        torch.cuda.synchronize()
        tm["render"] += time.perf_counter() - t1
        views += rgb.shape[0] * rgb.shape[1]
        if k in (0, 10, 20, 40):
            sheet.append(rgb[:4].cpu().numpy())
        robot = torch.cat([ws.q, ws.qd], -1).float()
        t1 = time.perf_counter()
        us.clear()
        act = teacher.act(obs)
        obs, _, done, ex = env.step(act)
        torch.cuda.synchronize()
        tm["world"] += time.perf_counter() - t1
        target = us[-1].float()                                                          # [B, 6] joint targets (rad)
        ep = ex["episode"]
        keep = {n: v.clone() for n, v in ep.items()} if keep is None else {
            n: torch.where(done, v, keep[n]) for n, v in ep.items()}
        idx = live.nonzero().flatten()
        if buf is None:
            cap = a.decisions * B
            buf = dict(rgb=torch.empty((cap, *rgb.shape[1:]), dtype=torch.uint8, device=dev),
                       robot=torch.empty((cap, 12), device=dev), target=torch.empty((cap, 6), device=dev))
        m = len(idx)
        buf["rgb"][n_buf: n_buf + m] = rgb[idx]
        buf["robot"][n_buf: n_buf + m] = robot[idx]
        buf["target"][n_buf: n_buf + m] = target[idx]
        n_buf += m
        # training steps on a fixed-size batch drawn from everything recorded so far (constant shapes)
        if n_buf < a.train_batch:
            continue
        for _ in range(a.train_per_decision):
            t1 = time.perf_counter()
            pick = torch.randint(n_buf, (a.train_batch,), device=dev)
            x = V.to_input(buf["rgb"][pick], shift_px=4)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp == "bf16"):
                pred = student(x, buf["robot"][pick])
            loss = F.mse_loss(pred.float(), buf["target"][pick])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t1
            tm["train"] += dt
            step_ms.append(dt * 1e3)
            losses.append(float(loss.detach()))
        mem_peak = max(mem_peak, gpu_used_gb())

    n_train = len(step_ms)
    steady = step_ms[10:] if len(step_ms) > 20 else step_ms
    succ = keep["success"].float().mean().item() if keep is not None else float("nan")
    ended = int((env.task.tr.end != 0).sum())
    info.update(
        decisions=k + 1, train_steps=n_train, train_batch=a.train_batch, samples=n_buf,
        world_steps=int(env.world_steps), views=views,
        render_s=round(tm["render"], 2), world_s=round(tm["world"], 2), train_s=round(tm["train"], 2),
        render_views_per_s=round(views / max(tm["render"], 1e-9)),
        world_steps_per_s=round(env.world_steps / max(tm["world"], 1e-9)),
        train_step_ms_first=round(step_ms[0], 1) if step_ms else None,
        train_step_ms_median_after10=round(float(np.median(steady)), 1) if steady else None,
        train_samples_per_s=round(a.train_batch / (float(np.median(steady)) / 1e3)) if steady else None,
        student_params=n_params, loss_first=losses[0] if losses else None,
        loss_last5=float(np.mean(losses[-5:])) if losses else None,
        teacher_success_in_nrd=round(succ, 4), ended=ended,
        gpu_used_gb_before=round(mem0, 2), gpu_used_gb_peak=round(mem_peak, 2),
        torch_max_allocated_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
        interop=rend.renderer.backend.interop, wall_s=round(time.time() - t_start, 1))
    print(json.dumps(info, indent=1), flush=True)
    (out / "smoke.json").write_text(json.dumps(info, indent=1))
    np.savez_compressed(out / "frames.npz", rgb=np.stack(sheet), decisions=np.array([0, 10, 20, 40][: len(sheet)]),
                        cameras=np.array(V.CAMERAS))
    try:
        from PIL import Image
        s = np.stack(sheet)                                         # [T, 4, C, H, W, 3]
        T_, W_, C_, H, Wd, _ = s.shape
        grid = s.transpose(1, 3, 0, 2, 4, 5).reshape(W_ * H, T_ * C_ * Wd, 3)
        Image.fromarray(grid).save(out / "frames.png")
    except Exception as e:  # noqa: BLE001
        print("sheet failed:", e)
    torch.save(dict(cfg=cfg.__dict__, state_dict=student.state_dict()), out / "student_smoke.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
