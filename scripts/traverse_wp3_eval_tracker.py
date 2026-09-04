"""Held-out evaluation of a WP3 tracker inside the NRD (imagination-side G6 numbers).

Rolls fixed-length fragments on the held-out split from real recorded context
windows and reports cross-track / heading / speed tracking statistics for the
policy and for the scripted pure-pursuit baseline over the same fragments.
Chrono evaluation (the real G6) is a separate step; this is what the tracker
looks like inside the model it was trained in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nedm.traverse.tracker_env import TraverseTrackingEnv, merge_env_cfg, pure_pursuit_actions


def load_policy(run_dir: Path, env, device):
    from rsl_rl.runners import OnPolicyRunner
    train_cfg = json.loads((run_dir / "train_cfg.json").read_text())
    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=device)
    runner.load(str(ckpts[-1]), load_optimizer=False)
    return runner.get_inference_policy(device=device), ckpts[-1].name


@torch.no_grad()
def evaluate(env: TraverseTrackingEnv, policy, steps: int, seed: int) -> dict:
    env.gen.manual_seed(seed)
    n, dev = env.num_envs, env.device
    env.reset_idx(torch.arange(n, device=dev),
                  fragment_steps=torch.full((n,), steps, device=dev, dtype=torch.long))
    env._compute_observations()
    active = torch.ones(n, dtype=torch.bool, device=dev)
    ct_all, sp_all, hd_all = [], [], []
    ct_max = torch.zeros(n, device=dev)
    failed = torch.zeros(n, dtype=torch.bool, device=dev)
    for _ in range(steps):
        _, _, dones, extras = env.step(policy(env.obs_buf))
        err = env._route_errors()
        ct = err["e_ct"].abs()
        ct_all.append(ct[active]); sp_all.append(err["e_v"].abs()[active]); hd_all.append(err["e_h"].abs()[active])
        ct_max = torch.where(active, torch.maximum(ct_max, ct), ct_max)
        failed |= active & dones.bool() & ~env.time_out_buf
        active &= ~dones.bool()
    ct = torch.cat(ct_all)
    return {"mean_ct_m": float(ct.mean()), "p95_ct_m": float(ct.quantile(0.95)),
            "mean_episode_max_ct_m": float(ct_max.mean()), "p95_episode_max_ct_m": float(ct_max.quantile(0.95)),
            "mean_speed_err_mps": float(torch.cat(sp_all).mean()),
            "mean_heading_err_deg": float(torch.rad2deg(torch.cat(hd_all).mean())),
            "fail_rate": float(failed.float().mean()), "fragments": n, "steps": steps}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dynamics-checkpoint", default="artifacts/traverse/wp2_mapv2_index_amd/ckpt_best.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="artifacts/traverse/wp3_tracker_eval.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    results = {}
    env = None
    for run in args.runs:
        run_dir = Path(run)
        env_cfg = json.loads((run_dir / "env_cfg.json").read_text())
        env_cfg.update({"num_envs": args.num_envs, "split": args.split, "auto_reset": False,
                        "fragment_steps_min": args.steps, "fragment_steps_max": args.steps,
                        "device": args.device, "dynamics_checkpoint": args.dynamics_checkpoint,
                        "max_bank_episodes": 0})
        env = TraverseTrackingEnv(merge_env_cfg(env_cfg), device=args.device)
        policy, ckpt = load_policy(run_dir, env, args.device)
        results[run_dir.name] = {"checkpoint": ckpt, **evaluate(env, policy, args.steps, args.seed)}
        print(run_dir.name, json.dumps(results[run_dir.name]), flush=True)
    if env is not None and env.obs_history_steps == 0:
        results["pure_pursuit"] = evaluate(env, lambda obs: pure_pursuit_actions(env), args.steps, args.seed)
        print("pure_pursuit", json.dumps(results["pure_pursuit"]), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
