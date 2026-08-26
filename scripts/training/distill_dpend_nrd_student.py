"""Online DAgger distillation: privileged z1 teacher -> z2-history student, inside the frozen NRD.

Plan: docs/vision/double_pen/NRD_double_pendulum_teacher_student_distillation_plan.md
(sections 5-9). Structure follows ~/Genesis/examples/manipulation/behavior_cloning.py
(FIFO experience buffer, teacher from the RL runner's inference policy, supervised
action imitation, TensorBoard + periodic checkpoints); the executed action is the
plan's beta-mixture instead of Genesis' closeness gate.

    PYTHONPATH=src python scripts/training/distill_dpend_nrd_student.py \
        --teacher-run-dir artifacts/rl_runs/dpend_nrd_reach_z1_armreward_lowerhalf_seed1_20260826 --seed 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.nrd.context_bank import load_context_bank  # noqa: E402
from nedm.rl.dpend_distill import (  # noqa: E402
    ReplayBuffer,
    StudentHistory,
    StudentPolicy,
    action_agreement,
    load_teacher,
    rollout_pairs,
)
from nedm.rl.dpend_nrd_reach_env import DEFAULT_EVAL_CONTEXT_BANK, DPendNRDReachEnv, make_eval_pairs  # noqa: E402

DEFAULT_TEACHER_RUN = Path("artifacts/rl_runs/dpend_nrd_reach_z1_armreward_lowerhalf_seed1_20260826")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher-run-dir", type=Path, default=DEFAULT_TEACHER_RUN)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None, help="default: <run>/model_1499.pt")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64])
    # DAgger (plan section 8)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--teacher-decay-iterations", type=int, default=50)
    parser.add_argument("--replay-capacity", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs-per-iteration", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    # held-out evaluation during training
    parser.add_argument("--eval-context-bank", type=Path, default=DEFAULT_EVAL_CONTEXT_BANK)
    parser.add_argument("--eval-pairs", type=int, default=100)
    parser.add_argument("--pairs-seed", type=int, default=20260826)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/rl_runs"))
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args(argv)


def beta_schedule(iteration: int, decay_iterations: int) -> float:
    return max(0.0, 1.0 - iteration / float(decay_iterations))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = args.device
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    teacher_run = args.teacher_run_dir.resolve()
    teacher_ckpt = args.teacher_checkpoint.resolve() if args.teacher_checkpoint else teacher_run / "model_1499.pt"
    teacher_env_cfg = json.loads((teacher_run / "env_cfg.json").read_text())
    teacher_train_cfg = json.loads((teacher_run / "train_cfg.json").read_text())
    if teacher_env_cfg.get("observe_z2", True):
        raise ValueError("the teacher must be a z1 (privileged) policy")

    run_name = args.run_name or f"dpend_nrd_student_z2hist{args.history_len}_from_{teacher_run.name.split('_seed')[0].replace('dpend_nrd_reach_', '')}_seed{args.seed}_{datetime.now().strftime('%Y%m%d')}"
    run_dir = (args.output_root / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    # Training env (teacher observation path; task identical to the teacher's run).
    env_cfg = dict(teacher_env_cfg)
    env_cfg.update({"num_envs": int(args.num_envs), "device": device, "seed": int(args.seed), "auto_reset": True, "observe_z2": False})
    env = DPendNRDReachEnv(env_cfg, device=device)
    teacher, teacher_iter = load_teacher(teacher_train_cfg, teacher_ckpt, env, device)

    # Held-out eval env: fixed pairs from the validation bank, student-controlled, beta = 0.
    eval_cfg = dict(env_cfg)
    eval_cfg.update({"num_envs": int(args.eval_pairs), "auto_reset": False, "context_bank": str(args.eval_context_bank)})
    eval_env = DPendNRDReachEnv(eval_cfg, device=device)
    bank = load_context_bank(args.eval_context_bank)
    eval_ctx, eval_goals = make_eval_pairs(bank, int(args.eval_pairs), int(args.pairs_seed), env.cfg["goal"], env.link_lengths, env.success_tolerance)

    student = StudentPolicy(args.history_len, env.z2_dim, tuple(args.hidden_dims)).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=float(args.learning_rate))
    history = StudentHistory(env, args.history_len)
    replay = ReplayBuffer(int(args.replay_capacity), student.input_dim, torch.device(device))
    mix_generator = torch.Generator(device=device)
    mix_generator.manual_seed(int(args.seed) + 1000)
    batch_generator = torch.Generator(device=device)
    batch_generator.manual_seed(int(args.seed) + 2000)

    config = {
        "teacher_run_dir": str(teacher_run),
        "teacher_checkpoint": str(teacher_ckpt),
        "teacher_iteration": teacher_iter,
        "seed": int(args.seed),
        "history_len": int(args.history_len),
        "hidden_dims": [int(v) for v in args.hidden_dims],
        "student_input_dim": student.input_dim,
        "context_indices": history.context_indices,
        "dagger": {
            "num_envs": int(args.num_envs),
            "rollout_steps_per_iteration": int(args.rollout_steps),
            "iterations": int(args.iterations),
            "teacher_decay_iterations": int(args.teacher_decay_iterations),
            "replay_capacity_samples": int(args.replay_capacity),
        },
        "student_training": {
            "optimizer": "Adam",
            "learning_rate": float(args.learning_rate),
            "batch_size": int(args.batch_size),
            "epochs_per_iteration": int(args.epochs_per_iteration),
            "max_grad_norm": float(args.max_grad_norm),
            "loss": "SmoothL1(student_action, teacher_action)",
        },
        "eval": {"context_bank": str(args.eval_context_bank), "pairs": int(args.eval_pairs), "pairs_seed": int(args.pairs_seed), "every": int(args.eval_every)},
        "env_cfg": env_cfg,
    }
    (run_dir / "distill_cfg.json").write_text(json.dumps(config, indent=2))
    writer = SummaryWriter(str(run_dir))
    print(f"run={run_dir.name}\nteacher={teacher_ckpt} (iter {teacher_iter}) student_input={student.input_dim} "
          f"history_len={args.history_len} context_indices={history.context_indices}", flush=True)

    def save(name: str, iteration: int, beta: float, metrics: dict[str, Any]) -> None:
        torch.save(
            {
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iteration": iteration,
                "beta": beta,
                "config": config,
                "metrics": metrics,
                "z2_mean": env.z2_mean.detach().cpu(),
                "z2_std": env.z2_std.detach().cpu(),
            },
            run_dir / f"{name}.pt",
        )

    @torch.no_grad()
    def evaluate(iteration: int) -> dict[str, Any]:
        student.eval()
        rec = rollout_pairs(eval_env, teacher, student, "student", eval_ctx, eval_goals, int(eval_env.max_episode_length), args.history_len)
        agreement = action_agreement(rec)
        student.train()
        out = {
            "success_rate": float(rec["success"].mean()),
            "timeout_rate": float(rec["time_out"].mean()),
            "spin_rate": float(rec["spin"].mean()),
            "ood_rate": float(rec["ood"].mean()),
            "min_distance_median_m": float(np.median(rec["min_distance_m"])),
            "within_1cm": float((rec["min_distance_m"] <= 0.01).mean()),
            "within_3cm": float((rec["min_distance_m"] <= 0.03).mean()),
            "time_to_success_median_s": float(np.median(rec["success_time_s"][rec["success"] > 0.5])) if (rec["success"] > 0.5).any() else float("nan"),
            "action_mae_vs_teacher": agreement["mae"],
            "action_sign_disagreement": agreement["sign_disagreement_rate"],
        }
        for key, value in out.items():
            writer.add_scalar(f"eval_student/{key}", value, iteration)
        return out

    obs_teacher, _ = env.get_observations()
    history.reset_from_env()
    best_success = -1.0
    history_log: list[dict[str, Any]] = []
    episode_stats: dict[str, list[float]] = {}
    started_all = time.time()

    for iteration in range(int(args.iterations)):
        beta = beta_schedule(iteration, int(args.teacher_decay_iterations))
        t0 = time.time()
        teacher_control = 0
        total_control = 0
        episode_stats.clear()
        with torch.no_grad():
            student.eval()
            for _ in range(int(args.rollout_steps)):
                teacher_action = teacher(obs_teacher)                       # label for the visited state
                student_obs = history.observation()
                student_action = student(student_obs)
                replay.add(student_obs, teacher_action)
                use_teacher = torch.rand(env.num_envs, 1, device=device, generator=mix_generator) < beta
                executed = torch.where(use_teacher, teacher_action, student_action)
                teacher_control += int(use_teacher.sum())
                total_control += env.num_envs
                obs_teacher, _, dones, extras = env.step(executed)
                history.after_step(dones)
                if "episode" in extras:
                    for key in ("/episode/success_rate", "/episode/spin_rate", "/episode/ood_rate", "/episode/timeout_rate", "/episode/min_distance_m"):
                        episode_stats.setdefault(key, []).append(float(extras["episode"][key]))
            student.train()
        collect_s = time.time() - t0

        t0 = time.time()
        losses, maes, batches = 0.0, 0.0, 0
        for obs_batch, action_batch in replay.batches(int(args.batch_size), int(args.epochs_per_iteration), batch_generator):
            pred = student(obs_batch)
            loss = F.smooth_l1_loss(pred, action_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), float(args.max_grad_norm))
            optimizer.step()
            losses += float(loss)
            maes += float((pred - action_batch).abs().mean())
            batches += 1
        train_s = time.time() - t0
        loss_mean = losses / max(batches, 1)
        mae_mean = maes / max(batches, 1)
        teacher_frac = teacher_control / max(total_control, 1)

        writer.add_scalar("loss/action_smooth_l1", loss_mean, iteration)
        writer.add_scalar("loss/action_mae", mae_mean, iteration)
        writer.add_scalar("dagger/beta", beta, iteration)
        writer.add_scalar("dagger/teacher_control_fraction", teacher_frac, iteration)
        writer.add_scalar("dagger/student_control_fraction", 1.0 - teacher_frac, iteration)
        writer.add_scalar("buffer/size", replay.size, iteration)
        writer.add_scalar("speed/collect_s", collect_s, iteration)
        writer.add_scalar("speed/train_s", train_s, iteration)
        for key, values in episode_stats.items():
            writer.add_scalar("rollout" + key.replace("/episode", ""), float(np.mean(values)), iteration)

        line = (f"it {iteration:4d} beta={beta:.2f} teacher_ctrl={teacher_frac:.2f} loss={loss_mean:.4f} mae={mae_mean:.3f} "
                f"buffer={replay.size} rollout_succ={np.mean(episode_stats.get('/episode/success_rate', [float('nan')])):.3f} "
                f"collect={collect_s:.1f}s train={train_s:.1f}s")
        record: dict[str, Any] = {"iteration": iteration, "beta": beta, "teacher_control_fraction": teacher_frac, "loss": loss_mean, "mae": mae_mean, "buffer": replay.size}
        if (iteration + 1) % int(args.eval_every) == 0 or iteration == int(args.iterations) - 1:
            metrics = evaluate(iteration)
            record["eval"] = metrics
            line += (f" | held-out student: succ={metrics['success_rate']:.2f} spin={metrics['spin_rate']:.2f} "
                     f"mind={metrics['min_distance_median_m'] * 1000:.1f}mm mae={metrics['action_mae_vs_teacher']:.3f}")
            if beta == 0.0 and metrics["success_rate"] > best_success:
                best_success = metrics["success_rate"]
                save("student_best", iteration, beta, metrics)
                line += " *best*"
        print(line, flush=True)
        history_log.append(record)
        if (iteration + 1) % int(args.save_every) == 0:
            save(f"student_{iteration + 1:04d}", iteration, beta, record.get("eval", {}))

    final_metrics = history_log[-1].get("eval", {})
    save("student_last", int(args.iterations) - 1, 0.0, final_metrics)
    (run_dir / "training_log.json").write_text(json.dumps(history_log, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(
        {"run": run_dir.name, "teacher": str(teacher_ckpt), "final_eval": final_metrics, "best_student_success": best_success,
         "wall_s": time.time() - started_all, "iterations": int(args.iterations)}, indent=2))
    writer.close()
    print(f"done in {(time.time() - started_all) / 60:.1f} min; best held-out student success {best_success:.2f}; wrote {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
