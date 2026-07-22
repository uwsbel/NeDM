"""Re-run a goal-reaching RL training from a previous run's saved configs, verbatim.

Loads ``env_cfg.json`` + ``train_cfg.json`` from an existing run directory and
launches an identical training into a fresh run dir — used to reproduce a run
with only the frozen dynamics ROM swapped. Because the ROM is referenced by an
on-disk path (``.../<rom_run>/checkpoints/best_val.pt``) that was overwritten with
the new checkpoint, loading the saved env_cfg verbatim already picks up the new
model; ``--dynamics-checkpoint`` can override the path explicitly if desired.

Loading the saved JSON (rather than rebuilding from CLI + ``default_env_cfg()``)
guarantees byte-identical config regardless of any later default drift.

Run in the nedm env, e.g.:

    PYTHONPATH=src python scripts/relaunch_rl_from_saved_cfg.py \
        --kind tracked \
        --src-run artifacts/rl_runs/tracked_goal_v2_far \
        --output-root artifacts/rl_runs \
        --run-name tracked_goal_v2_far_rollsel_rom_20260721 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_env(kind: str, env_cfg: dict, device: str):
    if kind == "arm":
        from nedm.rl.arm_reaching_env import ArmReachingEnv

        return ArmReachingEnv(env_cfg, device=device)
    if kind == "tracked":
        from nedm.rl.tracked_goal_env import TrackedGoalReachingEnv

        return TrackedGoalReachingEnv(env_cfg, device=device)
    raise ValueError(f"unknown --kind {kind!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["arm", "tracked"], required=True)
    ap.add_argument("--src-run", type=Path, required=True, help="Old run dir with env_cfg.json/train_cfg.json.")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--dynamics-checkpoint",
        type=str,
        default=None,
        help="Override the ROM path; default keeps the saved env_cfg path (already the new model).",
    )
    ap.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override runner.max_iterations; default keeps the saved train_cfg value.",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env_cfg = json.loads((args.src_run / "env_cfg.json").read_text())
    train_cfg = json.loads((args.src_run / "train_cfg.json").read_text())

    env_cfg["device"] = args.device
    if args.dynamics_checkpoint is not None:
        env_cfg["dynamics_checkpoint"] = args.dynamics_checkpoint
    if args.max_iterations is not None:
        train_cfg["runner"]["max_iterations"] = int(args.max_iterations)

    # Same torch runtime setup the train scripts apply (matmul precision + TF32).
    matmul = str(train_cfg.get("torch", {}).get("matmul_precision", "high"))
    torch.set_float32_matmul_precision(matmul)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        allow_tf32 = matmul != "highest"
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

    run_dir = (args.output_root / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)  # refuse to clobber an existing run
    (run_dir / "env_cfg.json").write_text(json.dumps(env_cfg, indent=2))
    (run_dir / "train_cfg.json").write_text(json.dumps(train_cfg, indent=2))

    seed = int(train_cfg.get("seed", 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = build_env(args.kind, env_cfg, args.device)
    runner = OnPolicyRunner(env, train_cfg, log_dir=str(run_dir), device=args.device)

    dyn = Path(env_cfg["dynamics_checkpoint"]).resolve()
    print(f"Re-running {args.kind} RL in {run_dir}")
    print(f"dynamics_checkpoint={dyn}")
    print(f"exists={dyn.is_file()} device={args.device} num_envs={env.num_envs} "
          f"max_iterations={train_cfg['runner']['max_iterations']}")
    runner.learn(
        num_learning_iterations=train_cfg["runner"]["max_iterations"],
        init_at_random_ep_len=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
