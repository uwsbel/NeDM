"""Chrono transfer of the distilled z2-history student, with Irrlicht visualization.

Observation path on the true plant, per policy step (10 Hz):

    Chrono::Sensor camera frame -> frozen NRD encoder -> normalize (z2_mean/z2_std)
        -> 4-step history (0.1 s apart) -> [history, goal/L] -> student -> elbow torque

No z1 reaches the student. Goals are the held-out pairs of the NRD evaluation;
the current goal is a GREEN sphere (radius = tolerance), reached goals leave a
GRAY ball, timeouts a RED one. The markers are added to the system only AFTER
the sensor's OptiX scene has been built, so the camera the student sees never
contains them (verified at start-up by counting green pixels in a sensor frame
with a marker placed in view).

    PYTHONPATH=src DISPLAY=:1 python scripts/evaluation/visualize_dpend_student_chrono.py \
        --student-run-dir artifacts/rl_runs/<student run> --num-goals 10
"""

from __future__ import annotations

import pychrono as chrono  # noqa: F401  # load Chrono before torch/libstdc++ users

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = REPO_ROOT / "scripts" / "evaluation"
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import nedm.double_pendulum_data as dp  # noqa: E402
from eval_dpend_nrd_student import load_student  # noqa: E402
from nedm.nrd.context_bank import load_context_bank  # noqa: E402
from nedm.rl.dpend_nrd_reach_env import DPendNRDReachEnv, make_eval_pairs  # noqa: E402
from visualize_dpend_nrd_rl_chrono import PARKED, GoalMarkers, make_visual_system, render, teleport_to_context  # noqa: E402


class ChronoLatentHistory:
    """Rolling window of normalized latents for the Chrono side (H, z2_dim)."""

    def __init__(self, env: DPendNRDReachEnv, history_len: int) -> None:
        self.env = env
        self.history_len = history_len
        self.context_indices = [env.block_size - 1 - env.action_repeat * (history_len - 1 - i) for i in range(history_len)]
        self.buf = torch.zeros(history_len, env.z2_dim, dtype=torch.float32, device=env.device)

    def normalize(self, raw: torch.Tensor) -> torch.Tensor:
        return (raw - self.env.z2_mean) / self.env.z2_std

    def seed_from_context(self, context_z2_raw: np.ndarray) -> None:
        """From the recorded context's encoder latents (same frames the NRD student was seeded with)."""
        raw = torch.as_tensor(context_z2_raw[self.context_indices], dtype=torch.float32, device=self.env.device)
        self.buf = self.normalize(raw)

    @torch.no_grad()
    def push_frame(self, frame: np.ndarray) -> torch.Tensor:
        raw = self.env.model.encode_frame_sequence(torch.from_numpy(frame)[None, None].to(self.env.device))[0, 0]
        self.buf = torch.roll(self.buf, shifts=-1, dims=0)
        self.buf[-1] = self.normalize(raw)
        return raw

    def observation(self, goal_xz: np.ndarray) -> torch.Tensor:
        goal = torch.as_tensor(goal_xz, dtype=torch.float32, device=self.env.device) / self.env.total_length
        return torch.cat([self.buf.flatten(), goal])[None]


def green_pixel_count(frame: np.ndarray) -> int:
    f = frame.astype(np.int16)
    return int(((f[..., 1] > 140) & (f[..., 0] < 120) & (f[..., 2] < 120)).sum())


@torch.no_grad()
def run_goal(scene, tap, env, student, history, goal_xz, max_steps, vis, realtime_factor, first_step_uses_seeded_history, frame_log=None):
    tolerance = env.success_tolerance
    timer = chrono.ChRealtimeStepTimer() if realtime_factor > 0 else None
    current = dp.read_state(scene)
    dists = [math.hypot(current["tip_x_m"] - goal_xz[0], current["tip_z_m"] - goal_xz[1])]
    actions: list[float] = []
    success = spin = window_closed = False
    steps_taken = 0
    for step in range(max_steps):
        # Manual trigger: exactly one sensor render of the CURRENT state per Update().
        scene.manager.Update()
        frame = tap.take()
        pushed = not (step == 0 and first_step_uses_seeded_history)
        if pushed:
            history.push_frame(frame)
        action = float(student(history.observation(goal_xz)).flatten()[0].clamp(-1.0, 1.0).item())
        actions.append(action)
        if frame_log is not None:
            # The very array the encoder consumed (uint8 HxWx3, top-down RGB), plus what the student did with it.
            frame_log.append({"frame": frame.copy(), "step": step, "sim_time_s": scene.system.GetChTime(), "pushed_to_history": pushed,
                              "action": action, "distance_m": dists[-1], "goal_xz": [float(goal_xz[0]), float(goal_xz[1])]})
        scene.elbow_torque.SetSetpoint(action * dp.TAU_MAX_NM, scene.system.GetChTime())
        steps_taken = step + 1
        terminated = False
        for _ in range(env.action_repeat):
            for _ in range(dp.SUBSTEPS_PER_CONTROL):
                scene.system.DoStepDynamics(dp.DT_SIM_S)
            current = dp.read_state(scene)
            dist = math.hypot(current["tip_x_m"] - goal_xz[0], current["tip_z_m"] - goal_xz[1])
            dists.append(dist)
            if not render(vis):
                window_closed = terminated = True
                break
            if timer is not None:
                timer.Spin(dp.CONTROL_DT_S / realtime_factor)
            if dist <= tolerance:
                success = terminated = True
                break
            if abs(current["omega1_radps"]) > env.omega_limit or abs(current["omega2_radps"]) > env.omega_limit:
                spin = terminated = True
                break
        if terminated:
            break
    return {
        "success": success, "spin": spin, "time_out": not success and not spin and not window_closed, "window_closed": window_closed,
        "policy_steps": steps_taken, "elapsed_s": (len(dists) - 1) * dp.CONTROL_DT_S,
        "initial_distance_m": dists[0], "final_distance_m": dists[-1], "min_distance_m": min(dists),
        "action_abs_mean": float(np.mean(np.abs(actions))) if actions else 0.0,
    }


def save_student_input_frames(out_dir: Path, frame_logs: list[list[dict]], results: list[dict], tolerance_m: float) -> None:
    """PNG per consumed frame (raw, exactly the encoder input), an index, a 10 fps GIF and a contact sheet.

    The GIF/contact sheet carry a thin annotation strip BELOW the image (the 128x128
    sensor image itself is left untouched) with goal/step/action/distance and the
    goal projected as a small ring, so you can relate what the student saw to what it did."""
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    annotated: list[Image.Image] = []
    scale = 3
    focal_px = (dp.IMAGE_WIDTH / 2.0) / math.tan(dp.CAMERA_HFOV_RAD / 2.0)
    ring_px = max(2.0, focal_px * tolerance_m / dp.CAMERA_DISTANCE_M * scale)
    for goal_index, (log, rec) in enumerate(zip(frame_logs, results)):
        for entry in log:
            name = f"goal{goal_index + 1:02d}_step{entry['step']:03d}.png"
            Image.fromarray(entry["frame"]).save(out_dir / name)  # raw encoder input
            index.append({"file": name, "goal": goal_index + 1, "pair": rec["pair"], **{k: v for k, v in entry.items() if k != "frame"}})
            big = Image.fromarray(entry["frame"]).resize((dp.IMAGE_WIDTH * scale, dp.IMAGE_HEIGHT * scale), Image.NEAREST)
            canvas = Image.new("RGB", (big.width, big.height + 28), (20, 20, 24))
            canvas.paste(big, (0, 0))
            draw = ImageDraw.Draw(canvas)
            gu, gv = dp.project_to_pixel(*entry["goal_xz"])
            cx, cy = gu * scale, gv * scale
            draw.ellipse([cx - ring_px, cy - ring_px, cx + ring_px, cy + ring_px], outline=(80, 230, 120), width=1)
            draw.text((4, big.height + 4), f"goal {goal_index + 1} step {entry['step']:2d} t={entry['sim_time_s']:.2f}s  a={entry['action']:+.2f}  "
                                            f"d={entry['distance_m'] * 1000:5.1f} mm  {'hist' if entry['pushed_to_history'] else 'seeded'}", fill=(235, 235, 235))
            annotated.append(canvas)
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    if annotated:
        annotated[0].save(out_dir / "student_input_10hz.gif", save_all=True, append_images=annotated[1:], duration=100, loop=0)
        cols = 10
        rows = math.ceil(len(annotated) / cols)
        w, h = annotated[0].size
        sheet = Image.new("RGB", (cols * w, rows * h), (20, 20, 24))
        for i, im in enumerate(annotated):
            sheet.paste(im, ((i % cols) * w, (i // cols) * h))
        sheet.save(out_dir / "contact_sheet.png")
    print(f"saved {len(index)} student input frames -> {out_dir} (PNG per frame, index.json, student_input_10hz.gif, contact_sheet.png)", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student-run-dir", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=str, default="student_best.pt")
    parser.add_argument("--num-goals", type=int, default=10)
    parser.add_argument("--first-pair", type=int, default=0)
    parser.add_argument("--pairs-seed", type=int, default=20260826)
    parser.add_argument("--reset-between", action="store_true", help="teleport to each pair's context and re-seed the history (paired protocol)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--window", type=int, nargs=2, default=[1280, 960])
    parser.add_argument("--save-frames", action="store_true",
                        help="save every sensor frame the student consumed (PNG + index.json + 10 fps GIF + contact sheet) under the output dir")
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--no-hold", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = args.device
    torch.set_float32_matmul_precision("high")
    run_dir = args.student_run_dir.resolve()
    ckpt = run_dir / args.student_checkpoint
    student, payload = load_student(ckpt, device)
    cfg = payload["config"]
    bank_path = Path(cfg["eval"]["context_bank"])
    env_cfg = dict(cfg["env_cfg"])
    env_cfg.update({"num_envs": 1, "device": device, "auto_reset": False, "observe_z2": False, "context_bank": str(bank_path)})
    env = DPendNRDReachEnv(env_cfg, device=device)
    if not torch.allclose(env.z2_mean.cpu(), payload["z2_mean"].cpu(), atol=1e-6):
        raise ValueError("student z2 normalization does not match the NRD checkpoint")
    H = int(cfg["history_len"])
    max_steps = int(args.max_steps) if args.max_steps is not None else int(env.max_episode_length)
    bank = load_context_bank(bank_path)
    context_ids, goals = make_eval_pairs(bank, args.first_pair + args.num_goals, args.pairs_seed, env.cfg["goal"], env.link_lengths, env.success_tolerance)
    context_ids = context_ids[args.first_pair:]
    goals_np = goals.numpy()[args.first_pair:]
    nrd_ref = {}
    ref_path = run_dir / f"student_vs_teacher_eval_{ckpt.stem}" / "per_pair.json"
    if ref_path.is_file():
        nrd_ref = {r["pair"]: r for r in json.load(open(ref_path))}
    print(f"student={run_dir.name}/{ckpt.name} (iter {payload['iteration']}) H={H} tolerance={env.success_tolerance * 1000:.0f} mm "
          f"goals={args.num_goals} mode={'reset-between' if args.reset_between else 'continuous'} realtime x{args.realtime_factor}", flush=True)

    # Scene with the sensor camera. First sensor update BEFORE any marker exists: the
    # OptiX scene is built from the bodies present now, so markers stay invisible to it.
    # The manual-trigger scheduler launches at most one render per Update() and only
    # after simulation time has advanced, so the two check frames are one control
    # period apart; the context state is re-teleported (exactly) afterwards.
    scene = dp.build_scene(with_camera=True)
    tap = dp.FrameTap(scene.camera)
    context_state = bank["states"][int(context_ids[0]), -1, :]
    teleport_to_context(scene, context_state)
    scene.manager.Update()
    clean_frame = tap.take()
    markers = GoalMarkers(scene, args.num_goals, env.success_tolerance)
    dp._advance_to_next_boundary(scene)
    state0 = dp.read_state(scene)
    markers.show_goal(state0["tip_x_m"], state0["tip_z_m"])  # put a marker right on the tip, in full view
    scene.manager.Update()
    test_frame = tap.take()
    greens = green_pixel_count(test_frame)
    print(f"sensor marker check: green pixels with a marker in view = {greens} (clean frame {green_pixel_count(clean_frame)}); "
          f"{'OK - sensor does not see the markers' if greens == 0 else 'WARNING - markers ARE visible to the sensor'}", flush=True)
    markers.goal.SetPos(PARKED)
    dp._advance_to_next_boundary(scene)
    teleport_to_context(scene, context_state)
    vis = None if args.no_vis else make_visual_system(scene, *args.window, f"{run_dir.name}: student (camera -> encoder -> z2 history) x {args.num_goals} goals")

    history = ChronoLatentHistory(env, H)
    history.seed_from_context(bank["z2"][int(context_ids[0])])
    results = []
    frame_logs: list[list[dict]] = []
    for index in range(args.num_goals):
        seeded = index == 0
        if args.reset_between and index > 0:
            teleport_to_context(scene, bank["states"][int(context_ids[index]), -1, :])
            history.seed_from_context(bank["z2"][int(context_ids[index])])
            seeded = True
        goal = goals_np[index]
        markers.show_goal(float(goal[0]), float(goal[1]))
        started = time.time()
        frame_log: list[dict] | None = [] if args.save_frames else None
        rec = run_goal(scene, tap, env, student, history, goal, max_steps, vis, float(args.realtime_factor), seeded, frame_log)
        if frame_log is not None:
            frame_logs.append(frame_log)
        rec.update({"pair": int(args.first_pair + index), "context_id": int(context_ids[index]), "goal_x_m": float(goal[0]), "goal_z_m": float(goal[1]), "wall_s": time.time() - started})
        outcome = "SUCCESS" if rec["success"] else ("SPIN" if rec["spin"] else ("CLOSED" if rec["window_closed"] else "TIMEOUT"))
        ref = nrd_ref.get(rec["pair"])
        ref_txt = f" | NRD-student on this pair: {'success' if ref and ref['student_success'] > 0.5 else 'timeout' if ref else 'n/a'}" if ref else ""
        print(f"goal {index + 1:2d}/{args.num_goals} pair {rec['pair']:3d} at ({goal[0]:+.3f}, {goal[1]:+.3f}) m: {outcome:7s} t={rec['elapsed_s']:.2f}s "
              f"closest={rec['min_distance_m'] * 1000:5.1f} mm final={rec['final_distance_m'] * 1000:5.1f} mm |a|={rec['action_abs_mean']:.2f}{ref_txt}", flush=True)
        if rec["success"]:
            markers.mark_reached(float(goal[0]), float(goal[1]))
        elif rec["time_out"] or rec["spin"]:
            markers.mark_timeout(float(goal[0]), float(goal[1]))
        results.append(rec)
        if rec["window_closed"]:
            break
        scene.elbow_torque.SetSetpoint(0.0, scene.system.GetChTime())
        dp._advance_to_next_boundary(scene)
        render(vis)
    markers.goal.SetPos(PARKED)
    n_ok = sum(int(r["success"]) for r in results)
    print(f"\n{n_ok}/{len(results)} goals reached in Chrono with the camera-driven student; median closest approach "
          f"{np.median([r['min_distance_m'] for r in results]) * 1000:.1f} mm; simulated {sum(r['elapsed_s'] for r in results):.1f} s", flush=True)
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / f"chrono_student_consecutive_goals_{ckpt.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "consecutive_goals.json").write_text(json.dumps(
        {"student_checkpoint": str(ckpt), "mode": "reset-between" if args.reset_between else "continuous", "tolerance_m": env.success_tolerance,
         "sensor_marker_green_pixels": greens, "results": results}, indent=2))
    print(f"wrote {output_dir / 'consecutive_goals.json'}")
    if args.save_frames:
        save_student_input_frames(output_dir / "student_input_frames", frame_logs, results, env.success_tolerance)
    if vis is not None and not args.no_hold and not (results and results[-1]["window_closed"]):
        print("chain finished -- window stays open (close it or Ctrl-C to exit)", flush=True)
        while render(vis):
            time.sleep(0.02)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
