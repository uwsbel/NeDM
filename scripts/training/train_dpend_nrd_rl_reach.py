"""Train a PPO goal-reaching policy for the double pendulum inside the frozen NRD.

Two policies are trained against the SAME frozen NRD transition model and differ
only in whether the policy observes the camera latent z2 (plan section 1):

    --policy-obs z1      Policy A: [normalize_state(z1), g, e]
    --policy-obs z1z2    Policy B: [normalize_state(z1), normalize_z2(z2), g, e]

Everything else (PPO, architecture apart from input width, seeds, reset bank,
goal distribution, reward, termination) is shared.
"""

from __future__ import annotations

import argparse
import json
import math
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

from nedm.rl.dpend_nrd_reach_env import (  # noqa: E402
    DEFAULT_NRD_CHECKPOINT,
    DEFAULT_TRAIN_CONTEXT_BANK,
    DPendNRDReachEnv,
    default_env_cfg,
    merge_env_cfg,
)


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
    parser = argparse.ArgumentParser(description="PPO goal reaching for the double pendulum inside the frozen NRD.")
    parser.add_argument("--policy-obs", choices=["z1", "z1z2"], required=True)
    parser.add_argument("--exp-name", type=str, default="dpend-nrd-reach")
    parser.add_argument("--reward-preset", choices=["plan", "arm"], default="plan",
                        help="plan: the task document's reward; arm: the arm reach study's exponential recipe "
                             "scale-mapped to L=0.6 m (exp(-d/0.06), action-rate 0.02, bonus 150, tol 2 cm, no failure charge)")
    parser.add_argument("--ppo-preset", choices=["plan", "arm"], default="plan",
                        help="arm: desired_kl 0.005, lr 1e-4, entropy 0.001, 3 epochs x 16 minibatches, noise 0.3, 64 steps/env")
    parser.add_argument("--ee-error-scale-m", type=float, default=None)
    parser.add_argument("--action-rate-weight", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=800)
    parser.add_argument("--num-steps-per-env", type=int, default=24)
    parser.add_argument("--num-learning-epochs", type=int, default=5)
    parser.add_argument("--num-mini-batches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--schedule", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--desired-kl", type=float, default=0.01)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--init-noise-std", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nrd-checkpoint", type=Path, default=DEFAULT_NRD_CHECKPOINT)
    parser.add_argument("--context-bank", type=Path, default=DEFAULT_TRAIN_CONTEXT_BANK)
    parser.add_argument("--action-repeat", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--goal-theta-range-deg", type=float, nargs=2, default=None,
                        help="polar goal angle range in degrees (plan default 0 360)")
    parser.add_argument("--goal-r-frac", type=float, nargs=2, default=None, help="radius range as fractions of L")
    parser.add_argument("--distance-weight", type=float, default=None)
    parser.add_argument("--progress-weight", type=float, default=None)
    parser.add_argument("--success-bonus", type=float, default=None)
    parser.add_argument("--angular-velocity-change-weight", type=float, default=None)
    parser.add_argument("--success-tolerance-m", type=float, default=None)
    parser.add_argument("--failure-penalty-mode", choices=["remaining_distance", "none"], default=None,
                        help="charge failure terminations the remaining-horizon distance penalty (default) or nothing")
    parser.add_argument("--z2-guard-margin", type=float, default=None,
                        help="OOD guard = margin x per-dim |z2_norm| max over the reset bank; <=0 disables")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/rl_runs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--logger", type=str, default="tensorboard")
    args = parser.parse_args(argv)
    if args.ppo_preset == "arm":
        defaults = parser.parse_args(["--policy-obs", args.policy_obs])
        for name, value in (("desired_kl", 0.005), ("learning_rate", 1.0e-4), ("entropy_coef", 0.001),
                            ("num_learning_epochs", 3), ("num_mini_batches", 16), ("init_noise_std", 0.3),
                            ("num_steps_per_env", 64)):
            if getattr(args, name) == getattr(defaults, name):
                setattr(args, name, value)
    return args


def get_train_cfg(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": float(args.desired_kl),
            "entropy_coef": float(args.entropy_coef),
            "gamma": 0.99,
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
            "seed": int(args.seed),
            "nrd_checkpoint": str(args.nrd_checkpoint),
            "context_bank": str(args.context_bank),
            "observe_z2": args.policy_obs == "z1z2",
            "action_repeat": int(args.action_repeat),
            "max_episode_steps": int(args.max_episode_steps),
            "auto_reset": True,
        }
    )
    if args.reward_preset == "arm":
        cfg["reward"].update(
            {
                "type": "exponential",
                "ee_error_scale_m": 0.06,
                "action_rate_weight": 0.02,
                "success_bonus": 150.0,
                "success_tolerance_m": 0.02,
                "angular_velocity_change_weight": 0.0,
                "failure_penalty_mode": "none",
            }
        )
    if args.goal_theta_range_deg is not None:
        cfg["goal"]["theta_range_rad"] = [math.radians(v) for v in args.goal_theta_range_deg]
    if args.goal_r_frac is not None:
        cfg["goal"]["r_min_frac"], cfg["goal"]["r_max_frac"] = (float(v) for v in args.goal_r_frac)
    for arg_name, key in (
        ("ee_error_scale_m", "ee_error_scale_m"),
        ("action_rate_weight", "action_rate_weight"),
        ("distance_weight", "distance_weight"),
        ("progress_weight", "progress_weight"),
        ("success_bonus", "success_bonus"),
        ("angular_velocity_change_weight", "angular_velocity_change_weight"),
        ("success_tolerance_m", "success_tolerance_m"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            cfg["reward"][key] = float(value)
    if args.failure_penalty_mode is not None:
        cfg["reward"]["failure_penalty_mode"] = args.failure_penalty_mode
    if args.z2_guard_margin is not None:
        cfg["termination"]["z2_guard_margin"] = float(args.z2_guard_margin) if args.z2_guard_margin > 0 else None
    return merge_env_cfg(cfg)


def make_run_dir(args: argparse.Namespace) -> Path:
    run_name = args.run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_stem = args.nrd_checkpoint.resolve().parents[1].name
        run_name = f"{args.exp_name}_{args.policy_obs}_{checkpoint_stem}_seed{args.seed}_{timestamp}"
    run_dir = (args.output_root / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.device = resolve_device(args.device)
    configure_torch_runtime(args.device, args.matmul_precision)
    run_dir = make_run_dir(args)
    env_cfg = get_env_cfg(args)
    train_cfg = get_train_cfg(args)
    (run_dir / "env_cfg.json").write_text(json.dumps(env_cfg, indent=2))
    (run_dir / "train_cfg.json").write_text(json.dumps(train_cfg, indent=2))

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    env = DPendNRDReachEnv(env_cfg, device=args.device)
    runner = OnPolicyRunner(env, train_cfg, log_dir=str(run_dir), device=args.device)
    if str(args.logger).lower() in {"none", "off", "disabled"}:
        runner.writer = NoOpSummaryWriter()
        runner.logger_type = "none"
        print("logger disabled; scalar logging is disabled, checkpoint saves remain enabled")

    print(f"Starting double-pendulum NRD reaching RL in {run_dir}")
    print(
        f"policy_obs={args.policy_obs} num_obs={env.num_obs} device={args.device} num_envs={env.num_envs} "
        f"action_repeat={env.action_repeat} (policy dt {env.policy_dt_s:.3f} s) max_steps={env.max_episode_length}"
    )
    print(f"nrd_checkpoint={Path(env_cfg['nrd_checkpoint']).resolve()} (epoch {env.payload.get('epoch')})")
    print(f"context_bank={Path(env_cfg['context_bank']).resolve()} ({env.num_contexts} windows)")
    runner.learn(num_learning_iterations=train_cfg["runner"]["max_iterations"], init_at_random_ep_len=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
