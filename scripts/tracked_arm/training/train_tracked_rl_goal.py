"""Train a PPO goal-reaching policy against the frozen tracked-vehicle ROM."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from rsl_rl.runners import OnPolicyRunner


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.rl.defaults import (
    DEFAULT_TRACKED_DYNAMICS_CHECKPOINT,
    DEFAULT_TRACKED_PROCESSED_DATASET_DIR,
)
from nedm.rl.dynamics import resolve_dynamics_checkpoint_path
from nedm.rl.tracked_goal_env import TrackedGoalReachingEnv, default_env_cfg, merge_env_cfg


class NoOpSummaryWriter:
    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        return None

    def save_file(self, *args: Any, **kwargs: Any) -> None:
        return None


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def configure_torch_runtime(device: str, matmul_precision: str) -> None:
    torch.set_float32_matmul_precision(matmul_precision)
    if device.startswith("cuda") and torch.cuda.is_available():
        allow_tf32 = matmul_precision != "highest"
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO goal-reaching policy on the frozen tracked-vehicle ROM.")
    parser.add_argument("--exp-name", type=str, default="tracked-goal-reach")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--max-iterations", type=int, default=1500)
    parser.add_argument("--num-steps-per-env", type=int, default=32)
    parser.add_argument("--num-learning-epochs", type=int, default=5)
    parser.add_argument("--num-mini-batches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--schedule", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--desired-kl", type=float, default=0.01)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dynamics-checkpoint", type=Path, default=DEFAULT_TRACKED_DYNAMICS_CHECKPOINT)
    parser.add_argument("--processed-dataset-dir", type=Path, default=DEFAULT_TRACKED_PROCESSED_DATASET_DIR)
    parser.add_argument("--dynamics-context-steps", type=int, default=None)
    parser.add_argument("--action-repeat", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--goal-radius-min", type=float, default=None)
    parser.add_argument("--goal-radius-max", type=float, default=None)
    parser.add_argument("--goal-angle-abs", type=float, default=None, help="Symmetric forward cone half-angle (rad).")
    parser.add_argument("--progress-weight", type=float, default=None)
    parser.add_argument("--success-bonus", type=float, default=None)
    parser.add_argument("--success-tolerance-m", type=float, default=None)
    parser.add_argument("--time-penalty", type=float, default=None, help="Per-step reward cost (encourages fastest goal reaching).")
    parser.add_argument("--workspace-radius-m", type=float, default=None, help="Out-of-bounds termination radius; must exceed max goal radius.")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--init-noise-std", type=float, default=1.0)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/rl_runs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--logger", type=str, default="tensorboard",
                        help="RSL-RL logger: tensorboard, wandb, neptune, or none for smoke tests.")
    return parser.parse_args(argv)


def get_train_cfg(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": float(args.desired_kl),
            "entropy_coef": float(args.entropy_coef),
            "gamma": float(args.gamma),
            "lam": 0.95,
            "learning_rate": float(args.learning_rate),
            "max_grad_norm": 1.0,
            "num_learning_epochs": int(args.num_learning_epochs),
            "num_mini_batches": int(args.num_mini_batches),
            "schedule": args.schedule,
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
            "init_noise_std": float(args.init_noise_std),
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": args.exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": int(args.max_iterations),
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": int(args.num_steps_per_env),
        "save_interval": int(args.save_interval),
        "empirical_normalization": True,
        "logger": args.logger,
        "seed": int(args.seed),
        "torch": {"matmul_precision": args.matmul_precision},
    }


def get_env_cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg = default_env_cfg()
    cfg.update(
        {
            "num_envs": int(args.num_envs),
            "device": resolve_device(args.device),
            "dynamics_checkpoint": str(args.dynamics_checkpoint),
            "processed_dataset_dir": str(args.processed_dataset_dir),
            "dynamics_context_steps": int(args.dynamics_context_steps)
            if args.dynamics_context_steps is not None
            else None,
            "action_repeat": int(args.action_repeat),
            "max_episode_steps": int(args.max_episode_steps),
            "auto_reset": True,
        }
    )
    if args.goal_radius_min is not None or args.goal_radius_max is not None:
        lo, hi = cfg["goal"]["radius_m"]
        cfg["goal"]["radius_m"] = [
            float(args.goal_radius_min) if args.goal_radius_min is not None else lo,
            float(args.goal_radius_max) if args.goal_radius_max is not None else hi,
        ]
    if args.goal_angle_abs is not None:
        cfg["goal"]["angle_rad"] = [-float(args.goal_angle_abs), float(args.goal_angle_abs)]
    if args.progress_weight is not None:
        cfg["reward"]["progress_weight"] = float(args.progress_weight)
    if args.success_bonus is not None:
        cfg["reward"]["success_bonus"] = float(args.success_bonus)
    if args.success_tolerance_m is not None:
        cfg["reward"]["success_tolerance_m"] = float(args.success_tolerance_m)
    if args.time_penalty is not None:
        cfg["reward"]["time_penalty"] = float(args.time_penalty)
    if args.workspace_radius_m is not None:
        cfg["termination"]["workspace_radius_m"] = float(args.workspace_radius_m)
    return merge_env_cfg(cfg)


def make_run_dir(args: argparse.Namespace) -> Path:
    run_name = args.run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_stem = args.dynamics_checkpoint.parents[2].name
        run_name = f"{args.exp_name}_{checkpoint_stem}_{timestamp}"
    run_dir = (args.output_root / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.device = resolve_device(args.device)
    args.dynamics_checkpoint = resolve_dynamics_checkpoint_path(args.dynamics_checkpoint)
    configure_torch_runtime(args.device, args.matmul_precision)
    run_dir = make_run_dir(args)
    env_cfg = get_env_cfg(args)
    train_cfg = get_train_cfg(args)

    (run_dir / "env_cfg.json").write_text(json.dumps(env_cfg, indent=2))
    (run_dir / "train_cfg.json").write_text(json.dumps(train_cfg, indent=2))

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    env = TrackedGoalReachingEnv(env_cfg, device=args.device)
    runner = OnPolicyRunner(env, train_cfg, log_dir=str(run_dir), device=args.device)
    if str(args.logger).lower() in {"none", "off", "disabled"}:
        runner.writer = NoOpSummaryWriter()
        runner.logger_type = "none"
        print("logger disabled; scalar logging is disabled, checkpoint saves remain enabled")

    print(f"Starting tracked-vehicle goal-reaching RL in {run_dir}")
    print(
        f"device={args.device} num_envs={env.num_envs} action_repeat={env.action_repeat} "
        f"step_dt={env.step_dt:.3f}s obs={env.num_obs} act={env.num_actions}"
    )
    print(f"dynamics_checkpoint={Path(env_cfg['dynamics_checkpoint']).resolve()}")
    print(f"seed_prefixes={env.num_seed_prefixes} context={env.context_steps} "
          f"goal_radius={env_cfg['goal']['radius_m']} goal_angle={env_cfg['goal']['angle_rad']}")
    runner.learn(num_learning_iterations=train_cfg["runner"]["max_iterations"], init_at_random_ep_len=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
