"""WP3: train the route tracker inside the frozen WP2 NRD with PPO (plan §10, G6).

Mirrors ``scripts/training/train_hmmwv_rl_tracking.py`` (the state-only HMMWV
tracking study that transferred to Chrono) -- same rsl_rl runner, same PPO
block, same actor/critic sizes -- on the traverse imagination env
(``nedm.traverse.tracker_env``). ``--smoke`` runs the scripted pure-pursuit
controller and a random policy for a few steps and prints tracking statistics
and throughput; use it locally before launching on the cluster.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nedm.traverse.tracker_env import TraverseTrackingEnv, merge_env_cfg, pure_pursuit_actions


class NoOpSummaryWriter:
    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        return None

    def save_file(self, *args: Any, **kwargs: Any) -> None:
        return None


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dynamics-checkpoint", default="artifacts/traverse/wp2_mapv2_index_amd/ckpt_best.pt")
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6")
    ap.add_argument("--routes", default="artifacts/traverse/wp3_routes")
    ap.add_argument("--split", default="train")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--max-bank-episodes", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--fragment-steps", type=int, nargs=2, default=[20, 60])
    ap.add_argument("--obs-history-steps", type=int, default=0)
    ap.add_argument("--steering-rate-limit", type=float, default=0.1)
    ap.add_argument("--cross-track-sigma", type=float, default=1.0)
    ap.add_argument("--heading-sigma", type=float, default=0.35)
    ap.add_argument("--speed-sigma", type=float, default=1.0)
    ap.add_argument("--cross-track-weight", type=float, default=2.0)
    ap.add_argument("--heading-weight", type=float, default=0.8)
    ap.add_argument("--speed-weight", type=float, default=0.5)
    ap.add_argument("--action-rate-weight", type=float, default=0.2)
    ap.add_argument("--throttle-brake-weight", type=float, default=0.05)
    ap.add_argument("--max-cross-track", type=float, default=6.0)
    # PPO (HMMWV tracking study values; plan §10's arm preset is lr 1e-4/kl 0.005/ent 1e-3/noise 0.3)
    ap.add_argument("--max-iterations", type=int, default=1000)
    ap.add_argument("--num-steps-per-env", type=int, default=64)
    ap.add_argument("--num-learning-epochs", type=int, default=5)
    ap.add_argument("--num-mini-batches", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--desired-kl", type=float, default=0.01)
    ap.add_argument("--entropy-coef", type=float, default=0.003)
    ap.add_argument("--init-noise-std", type=float, default=0.7)
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256, 128])
    ap.add_argument("--save-interval", type=int, default=100)
    ap.add_argument("--logger", default="tensorboard")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-steps", type=int, default=60)
    return ap.parse_args(argv)


def env_cfg_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return merge_env_cfg({
        "num_envs": args.num_envs, "device": args.device,
        "dynamics_checkpoint": args.dynamics_checkpoint, "arena": args.arena,
        "cache": args.cache, "routes": args.routes, "split": args.split,
        "families": args.families, "max_bank_episodes": args.max_bank_episodes,
        "fragment_steps_min": args.fragment_steps[0], "fragment_steps_max": args.fragment_steps[1],
        "obs_history_steps": args.obs_history_steps,
        "steering_rate_limit": args.steering_rate_limit,
        "reward": {
            "cross_track_sigma_m": args.cross_track_sigma, "heading_sigma_rad": args.heading_sigma,
            "speed_sigma_mps": args.speed_sigma, "cross_track_weight": args.cross_track_weight,
            "heading_weight": args.heading_weight, "speed_weight": args.speed_weight,
            "action_rate_weight": args.action_rate_weight,
            "throttle_brake_weight": args.throttle_brake_weight,
        },
        "termination": {"max_cross_track_m": args.max_cross_track},
        "seed": args.seed,
    })


def train_cfg_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "algorithm": {
            "class_name": "PPO", "clip_param": 0.2, "desired_kl": float(args.desired_kl),
            "entropy_coef": float(args.entropy_coef), "gamma": 0.99, "lam": 0.95,
            "learning_rate": float(args.learning_rate), "max_grad_norm": 1.0,
            "num_learning_epochs": int(args.num_learning_epochs),
            "num_mini_batches": int(args.num_mini_batches), "schedule": "adaptive",
            "use_clipped_value_loss": True, "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu", "actor_hidden_dims": list(args.hidden_dims),
            "critic_hidden_dims": list(args.hidden_dims),
            "init_noise_std": float(args.init_noise_std), "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1, "experiment_name": Path(args.out).name, "load_run": -1,
            "log_interval": 1, "max_iterations": int(args.max_iterations), "record_interval": -1,
            "resume": False, "resume_path": None, "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": int(args.num_steps_per_env),
        "save_interval": int(args.save_interval),
        "empirical_normalization": True,
        "logger": args.logger,
        "seed": int(args.seed),
    }


@torch.no_grad()
def smoke(env: TraverseTrackingEnv, steps: int) -> dict[str, float]:
    out = {}
    for name in ("pure_pursuit", "random"):
        env.reset()
        ct, sp, rew, fails, ends, n_done = [], [], [], 0, 0, 0
        torch.cuda.synchronize() if env.device.type == "cuda" else None
        t0 = time.time()
        for _ in range(steps):
            act = (pure_pursuit_actions(env) if name == "pure_pursuit"
                   else torch.randn(env.num_envs, 3, device=env.device))
            _, r, dones, extras = env.step(act)
            ct.append(extras["log"]["/tracking/cross_track_abs_m"].item())
            sp.append(extras["log"]["/tracking/speed_err_abs_mps"].item())
            rew.append(r.mean().item())
            if "episode" in extras:
                k = int(dones.sum())
                n_done += k
                fails += extras["episode"]["/episode/fail_rate"].item() * k
                ends += extras["episode"]["/episode/route_end_rate"].item() * k
        torch.cuda.synchronize() if env.device.type == "cuda" else None
        dt = time.time() - t0
        out[name] = {"cross_track_abs_m": sum(ct) / len(ct), "speed_err_abs_mps": sum(sp) / len(sp),
                     "reward": sum(rew) / len(rew), "done": n_done,
                     "fail_frac": fails / max(n_done, 1), "route_end_frac": ends / max(n_done, 1),
                     "env_steps_per_s": steps * env.num_envs / dt}
        print(name, json.dumps({k: round(v, 4) for k, v in out[name].items()}), flush=True)
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    env_cfg = env_cfg_from_args(args)
    train_cfg = train_cfg_from_args(args)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    env = TraverseTrackingEnv(env_cfg, device=args.device)
    print(f"env: {env.num_envs} envs, {env.bank.n_episodes} bank episodes ({args.split}), "
          f"obs {env.num_obs}-D, context {env.context}, bank load {env.bank.load_s:.1f}s, "
          f"total init {time.time() - t0:.1f}s", flush=True)
    (out / "env_cfg.json").write_text(json.dumps(env_cfg, indent=2))
    (out / "train_cfg.json").write_text(json.dumps(train_cfg, indent=2))
    if args.smoke:
        res = smoke(env, args.smoke_steps)
        (out / "smoke.json").write_text(json.dumps(res, indent=2))
        return 0
    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(env, train_cfg, log_dir=str(out), device=args.device)
    if str(args.logger).lower() in {"none", "off"}:
        runner.writer = NoOpSummaryWriter(); runner.logger_type = "none"
    runner.learn(num_learning_iterations=train_cfg["runner"]["max_iterations"], init_at_random_ep_len=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
