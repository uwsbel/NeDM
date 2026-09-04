"""Evaluate a Go2 policy trained in the frozen NRD model against Chrono.

THIS IS THE CLAIM THE WHOLE PIPELINE EXISTS TO TEST: that a policy trained
inside a learned dynamics model transfers to the simulator the model was fit to.
The number that matters is the tracking error in Chrono, not in the model.

IT ALWAYS REPORTS A REPLAY BASELINE, AND THAT IS NOT OPTIONAL.
------------------------------------------------------------
A policy's Chrono error is uninterpretable on its own. Some of it is the policy,
and some is that this harness is not a bit-exact reconstruction of the collector
that produced the references: same plant settings, but a fresh world, a fresh
contact history, a spawn at a mid-episode pose the robot never actually reached
by walking, and a low-level policy whose 5-step observation history starts empty.
Legged contact is chaotic enough that none of that is free.

So every run first drives Chrono with the REFERENCE'S OWN ACTIONS -- the exact
command series the collection recorded -- and reports the error that produces.
That is the floor. A policy error near the floor means the policy tracks as well
as the recorded commands did; a policy error far above it is the policy. Without
the floor, a harness artefact and a policy failure are the same number, and the
temptation is to attribute it to whichever one you were hoping for.

The floor is reported per reference, not just pooled, because it varies with what
the reference was doing: a pivot segment and a straight-line segment do not
degrade the same way under a restarted contact history.

--policy-checkpoint IS OPTIONAL. With no policy this runs the baseline alone,
which is the right thing to do while training is still going and is how the
harness was validated before there was a policy to evaluate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.rl.go2_chrono_tracking_env import (  # noqa: E402
    Go2ChronoCRMTrackingEnv,
    Go2ChronoTrackingEnv,
    go2_default_chrono_env_cfg,
)

DEFAULT_IMPORTED_POLICY = Path.home() / "sbel-artifacts/checkpoints/go2_cts_150k.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="RL run directory. Its env_cfg.json supplies the reward and "
                             "termination the policy was trained under.")
    parser.add_argument("--policy-checkpoint", type=Path, default=None,
                        help="model_*.pt from the run. Defaults to the highest-numbered one "
                             "in --run-dir. Omit entirely to run the replay baseline alone.")
    parser.add_argument("--chrono-config", type=Path, default=Path("configs/go2_chrono_eval_flat.json"))
    parser.add_argument("--reference-path", type=Path, default=None,
                        help="Defaults to the reference set the run trained against.")
    parser.add_argument("--imported-policy-ckpt", type=Path, default=DEFAULT_IMPORTED_POLICY)
    parser.add_argument("--dynamics-checkpoint", type=Path, default=None,
                        help="Only for its metadata: state fields, normalization, dt.")
    parser.add_argument("--num-references", type=int, default=5)
    parser.add_argument("--reference-index", type=int, default=None,
                        help="Evaluate one reference by index instead of the first N.")
    parser.add_argument("--reference-indices", type=str, default=None,
                        help="Comma-separated reference indices. Taking the first N is a "
                             "MOTION-BLIND draw -- on the training set it produced eight "
                             "references with a median path of 0.143 m over 6 s, against "
                             "0.500 m available from the same pool. Naming the indices is "
                             "how the eval set stops being an accident of ordering.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--horizon-s", type=float, default=None,
                        help="Episode horizon in SECONDS. Prefer this over --max-steps: the "
                             "model's rollout horizons are 5 s and 10 s and SELECTION is at "
                             "10 s, so a floor measured at some other horizon cannot be laid "
                             "beside a level-2 number. Converted with the env's own step_dt.")
    parser.add_argument("--pre-roll-time-s", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def latest_policy_checkpoint(run_dir: Path) -> Path:
    """Highest model_<n>.pt by the NUMBER, not by name or mtime.

    Lexicographic order puts model_999 after model_1000, and mtime is wrong the
    moment a directory is copied. Both have bitten this repo.
    """
    candidates = []
    for path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"no model_<n>.pt in {run_dir}")
    return max(candidates)[1]


def build_env(args: argparse.Namespace, run_cfg: dict[str, Any] | None) -> Any:
    cfg = go2_default_chrono_env_cfg()
    if run_cfg is not None:
        # TAKE THE REWARD AND TERMINATION FROM THE RUN, NOT FROM THE DEFAULTS.
        # A policy is only fairly scored against the objective it optimised. If
        # the module defaults have moved since the run, scoring against them
        # measures the edit, not the policy.
        for key in ("reward", "termination", "action_low", "action_high",
                    "action_scale", "action_repeat", "obs_history_steps",
                    "reference_preview_steps", "max_episode_steps"):
            if key in run_cfg:
                cfg[key] = run_cfg[key]
    cfg.update({
        "num_envs": 1,
        "device": args.device,
        "chrono_config": str(args.chrono_config),
        "imported_policy_ckpt": str(args.imported_policy_ckpt),
        "auto_reset": False,
    })
    if args.dynamics_checkpoint is not None:
        cfg["dynamics_checkpoint"] = str(args.dynamics_checkpoint)
    elif run_cfg is not None and run_cfg.get("dynamics_checkpoint"):
        cfg["dynamics_checkpoint"] = run_cfg["dynamics_checkpoint"]
    if args.reference_path is not None:
        cfg["reference_path"] = str(args.reference_path)
    elif run_cfg is not None and run_cfg.get("reference_path"):
        cfg["reference_path"] = run_cfg["reference_path"]
    if args.pre_roll_time_s is not None:
        cfg["pre_roll_time_s"] = float(args.pre_roll_time_s)
    if args.horizon_s is not None:
        # step_dt is dt_s * action_repeat and both come from the checkpoint /
        # config, so convert here rather than asking the caller to do arithmetic
        # that would silently drift if action_repeat changed.
        dt_s = float(cfg.get("dt_s") or 0.01)
        cfg["max_episode_steps"] = int(round(args.horizon_s / (dt_s * int(cfg["action_repeat"]))))
    if args.max_steps is not None:
        cfg["max_episode_steps"] = int(args.max_steps)

    terrain_type = json.loads(Path(args.chrono_config).read_text())["terrain"]["type"]
    env_cls = Go2ChronoCRMTrackingEnv if terrain_type == "crm" else Go2ChronoTrackingEnv
    cfg["initial_reference_ids"] = [0]
    return env_cls(cfg), env_cls, cfg


def load_policy(args: argparse.Namespace, env: Any, run_dir: Path) -> Any:
    from rsl_rl.runners import OnPolicyRunner

    train_cfg = json.loads((run_dir / "train_cfg.json").read_text())
    checkpoint = args.policy_checkpoint or latest_policy_checkpoint(run_dir)
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=args.device)
    runner.load(str(checkpoint))
    print(f"policy: {checkpoint}")
    return runner.get_inference_policy(device=args.device)


def roll_one(env: Any, reference_id: int, policy: Any | None) -> dict[str, Any]:
    """One reference. policy=None replays the reference's own actions."""
    env.reset_idx(torch.tensor([0], device=env.device),
                  reference_ids=torch.tensor([reference_id], device=env.device))
    env._compute_observations()
    obs = env.obs_buf

    errors: list[float] = []
    rewards: list[float] = []
    taken: list[list[float]] = []
    steps = 0
    for _ in range(env.max_episode_length):
        if policy is None:
            # THE REFERENCE'S OWN ACTIONS, at the reference cursor the env is
            # about to consume. _set_driver_action_np is bypassed deliberately:
            # the base's step() applies action scaling and clamping to a POLICY
            # output, and the reference action is already in command units.
            ref_step = int(env.ref_step_buf[0].item())
            raw = env.reference_set.actions[reference_id, ref_step]
            action = torch.as_tensor(raw, dtype=torch.float32, device=env.device).unsqueeze(0)
            obs, reward, dones, extras = env.step_raw_actions(action)
        else:
            with torch.no_grad():
                action = policy(obs)
            obs, reward, dones, extras = env.step(action)
        errors.append(float(extras["log"]["/tracking/position_error_m"]))
        rewards.append(float(reward[0]))
        # The COMMAND that reached the plant, after scaling and clamping. A policy
        # emitting a constant produces entirely ordinary-looking position errors,
        # so "did the policy actually act" cannot be read off the errors.
        taken.append([float(v) for v in env.sims[0].command])
        steps += 1
        if bool(dones[0]):
            break

    err = np.asarray(errors, dtype=np.float64)
    act = np.asarray(taken, dtype=np.float64)
    return {
        "action_mean": act.mean(axis=0).tolist(),
        "action_std": act.std(axis=0).tolist(),
        "action_span": (act.max(axis=0) - act.min(axis=0)).tolist(),
        "reference_id": reference_id,
        "episode_id": env.reference_set.episode_ids[reference_id],
        "scenario_family": env.reference_set.scenario_families[reference_id],
        "steps": steps,
        "completed": steps == env.max_episode_length,
        "mean_position_error_m": float(err.mean()),
        "final_position_error_m": float(err[-1]),
        "max_position_error_m": float(err.max()),
        "mean_reward": float(np.mean(rewards)),
    }


def summarize(label: str, rows: list[dict[str, Any]],
              floor_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Per reference first, pooled last, and the pooled line always states its mix.

    THE POOLED NUMBER IS ABOUT THE MIX, NOT ABOUT THE SUBJECT. On these
    references arc is 32x constant, so a pooled floor compared against a policy
    evaluated on a different family composition is the errdist mistake again --
    dividing by a mean taken over families whose difficulty differs by an order
    of magnitude. When a floor is supplied, each reference is compared to ITS OWN
    floor and the pooled ratio is reported only alongside the spread of the
    per-reference ratios.
    """
    mean = float(np.mean([r["mean_position_error_m"] for r in rows]))
    final = float(np.mean([r["final_position_error_m"] for r in rows]))
    completed = sum(int(r["completed"]) for r in rows)
    floor_by_id = {r["reference_id"]: r for r in (floor_rows or [])}
    ratio_col = " " * 9 if not floor_by_id else f"{'x floor':>9}"
    print(f"\n{label}")
    print(f"  {'reference':<34} {'family':<28} {'steps':>6} {'mean err':>9} {'final':>8}{ratio_col}")
    ratios = []
    for r in rows:
        cell = ""
        base = floor_by_id.get(r["reference_id"])
        if base is not None and base["mean_position_error_m"] > 0:
            ratio = r["mean_position_error_m"] / base["mean_position_error_m"]
            r["over_own_floor"] = ratio
            ratios.append(ratio)
            cell = f"{ratio:>9.2f}"
        print(f"  {r['episode_id']:<34} {r['scenario_family']:<28} "
              f"{r['steps']:>6} {r['mean_position_error_m']:>9.4f} "
              f"{r['final_position_error_m']:>8.4f}{cell}")
    # Strip the "go2_<terrain>_" prefix and "_command" suffix, and NOTHING ELSE.
    # A .split("_")[-1] here collapsed stop_and_go to "go" and both vel_step and
    # yaw_step to "step", printing 7 labels for 8 families -- a mix label that
    # understates the mix, on the very line whose job is to state it.
    families = sorted({
        "_".join(r["scenario_family"].removesuffix("_command").split("_")[2:])
        for r in rows
    })
    print(f"  {'POOLED':<34} {'mix: ' + ','.join(families):<28} {'':>6} {mean:>9.4f} {final:>8.4f}"
          f"   ({completed}/{len(rows)} to the horizon)")
    out = {"mean_position_error_m": mean, "final_position_error_m": final,
           "completed": completed, "num_references": len(rows),
           "family_mix": families, "per_reference": rows}
    if ratios:
        print(f"  PER-REFERENCE x floor: min {min(ratios):.2f}  median "
              f"{float(np.median(ratios)):.2f}  max {max(ratios):.2f}")
        out["over_own_floor"] = {"min": min(ratios), "median": float(np.median(ratios)),
                                 "max": max(ratios)}
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_cfg = None
    if args.run_dir is not None:
        run_cfg = json.loads((args.run_dir / "env_cfg.json").read_text())

    env, env_cls, cfg = build_env(args, run_cfg)
    print(f"env: {env_cls.__name__}  references: {env.num_references}  "
          f"horizon: {env.max_episode_length} steps ({env.max_episode_length * env.step_dt:.2f} s)")

    if args.reference_indices:
        reference_ids = [int(v) for v in args.reference_indices.split(",") if v.strip() != ""]
        bad = [i for i in reference_ids if not 0 <= i < env.num_references]
        if bad:
            raise SystemExit(f"reference indices out of range for this set: {bad}")
    elif args.reference_index is not None:
        reference_ids = [int(args.reference_index)]
    else:
        reference_ids = list(range(min(args.num_references, env.num_references)))

    result: dict[str, Any] = {
        "chrono_config": str(args.chrono_config),
        "reference_path": cfg["reference_path"],
        "reference_ids": reference_ids,
    }

    # THE BASELINE RUNS FIRST AND ALWAYS. See the module docstring.
    baseline_rows = [roll_one(env, rid, policy=None) for rid in reference_ids]
    result["replay_baseline"] = summarize(
        "REPLAY BASELINE -- Chrono driven by the reference's own recorded commands", baseline_rows)

    if args.run_dir is not None and (args.policy_checkpoint or list(args.run_dir.glob("model_*.pt"))):
        policy = load_policy(args, env, args.run_dir)
        policy_rows = [roll_one(env, rid, policy=policy) for rid in reference_ids]
        result["policy"] = summarize("POLICY -- trained in the frozen NRD model",
                                     policy_rows, floor_rows=baseline_rows)
        floor = result["replay_baseline"]["mean_position_error_m"]
        got = result["policy"]["mean_position_error_m"]
        print(f"\nPOLICY vs REPLAY FLOOR (pooled, mix "
              f"{','.join(result['policy']['family_mix'])}): {got:.4f} m against {floor:.4f} m")
        print("  Read the PER-REFERENCE ratios above first. The pooled figure is a statement")
        print("  about this family mix, not about the policy: arc and constant differ by ~30x,")
        print("  so pooling across an unmatched mix is the errdist mistake in another costume.")
        print("  The floor is what this harness costs: a fresh world, a spawn the robot never")
        print("  walked to, and an empty low-level observation history.")
        result["policy_over_floor_pooled"] = got / floor
    else:
        print("\nNo policy given -- baseline only. Pass --run-dir once PPO has written a checkpoint.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
