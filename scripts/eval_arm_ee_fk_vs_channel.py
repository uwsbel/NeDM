"""Study: FK(predicted q) vs the direct ``ee_base`` (P_ee) channel for open-loop EE drift.

The arm dynamics model predicts state = [q, qd, qcmd, ee_base]. There are therefore two
ways to read the end-effector out of an open-loop rollout:

  (A) **Direct P_ee channel** -- read the predicted ``ee_base`` channels straight off the
      rolled state (what ``eval_arm_rollout.py`` reports; the ~2% / ~0.5% @0.5 s headline).
  (B) **FK(pred q)** -- take the predicted joint angles ``q`` from the rolled state and run
      the batched product-of-exponentials forward kinematics (``ArmKinematics.ee_base``).

Both are scored against the recorded ``ee_base`` (Chrono ground truth: the finger-midpoint
grasp point in the arm-base frame, per ``arm_data.gripper_center`` -> ``base_frame``). The FK
uses the *same* grasp point + base frame (``extract_arm_geometry.py``), so FK adds no frame
bias -- the FK-fidelity floor FK(gt q) vs recorded ee_base is ~45 um.

Question answered: does channeling EE through FK(pred q) drift less than trusting the model's
own ee_base channel? This shares the exact rollout recipe as ``eval_arm_rollout.py`` (seed the
first ``context`` recorded steps, then autoregress ``next = state + predict_next_delta`` while
feeding recorded actions), and adds the FK read-out + the FK floor + q-drift diagnostics.

Run in the nedm env:

    PYTHONPATH=src python scripts/eval_arm_ee_fk_vs_channel.py \
        --checkpoint artifacts/training_runs/arm_transformer_full_v1 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.rl.arm_kinematics import ArmKinematics
from nedm.rl.dynamics import load_frozen_dynamics
from nedm.training.dataset import load_rollout_split


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FK(pred q) vs direct ee_base channel EE drift.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Run directory or .pt checkpoint of a trained arm dynamics model.")
    parser.add_argument("--processed-dataset-dir", type=Path, default=None,
                        help="Override the processed dataset dir (defaults to the checkpoint's).")
    parser.add_argument("--geometry", type=Path,
                        default=REPO_ROOT / "artifacts/arm_geometry/arm_geometry_v1.json",
                        help="Arm geometry JSON for the FK.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to roll out.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda.")
    parser.add_argument("--horizons-s", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0],
                        help="Rollout horizons in seconds to report EE drift at.")
    parser.add_argument("--max-episodes", type=int, default=400,
                        help="Number of held-out episodes to roll out (seeded random subset).")
    parser.add_argument("--seed", type=int, default=0, help="Episode-subset sampling seed.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON metrics path.")
    return parser.parse_args(argv)


def _idx(state_fields: list[str], names: list[str]) -> list[int]:
    return [state_fields.index(n) for n in names]


@torch.no_grad()
def _rollout_states(model, states, actions, context, max_steps):
    """Open-loop rollout: return predicted next-states (steps, D), aligned to states[context:]."""
    total = states.shape[0]
    steps = min(max_steps, total - context)
    if steps <= 0:
        return None
    hist_states = states[:context].clone()
    hist_actions = actions[:context].clone()
    preds = []
    for step in range(steps):
        delta = model.predict_next_delta(
            hist_states[-context:].unsqueeze(0),
            hist_actions[-context:].unsqueeze(0),
            terrain=None,
        ).squeeze(0)
        next_state = hist_states[-1] + delta
        preds.append(next_state)
        if context + step < actions.shape[0]:
            hist_actions = torch.cat([hist_actions, actions[context + step].unsqueeze(0)], dim=0)
        hist_states = torch.cat([hist_states, next_state.unsqueeze(0)], dim=0)
    return torch.stack(preds, dim=0)


def _summ(errs: np.ndarray, disps: np.ndarray) -> dict:
    mean_disp = float(disps.mean())
    rmse = float(np.sqrt((errs ** 2).mean()))
    return {
        "episodes": int(errs.size),
        "ee_rmse_m": rmse,
        "ee_mean_m": float(errs.mean()),
        "ee_p90_m": float(np.percentile(errs, 90)),
        "gt_disp_mean_m": mean_disp,
        "errdist": float(rmse / mean_disp) if mean_disp > 1e-6 else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dynamics = load_frozen_dynamics(
        checkpoint_path=args.checkpoint,
        device=args.device,
        processed_dataset_dir=args.processed_dataset_dir,
    )
    model = dynamics.model
    context = dynamics.context_steps
    dt_s = dynamics.dt_s
    state_fields = list(dynamics.metadata["state_fields"])
    q_idx = torch.tensor(_idx(state_fields, [f"q_{i}" for i in range(4)]),
                         dtype=torch.long, device=args.device)
    ee_idx = torch.tensor(_idx(state_fields, ["ee_base_x", "ee_base_y", "ee_base_z"]),
                          dtype=torch.long, device=args.device)

    ak = ArmKinematics.from_json(args.geometry, device=args.device, dtype=torch.float32)

    processed_dir = Path(
        args.processed_dataset_dir if args.processed_dataset_dir is not None
        else dynamics.config["processed_dataset_dir"]
    ).resolve()
    episodes = load_rollout_split(processed_dir, args.split)["episodes"]

    rng = np.random.default_rng(args.seed)
    if args.max_episodes < len(episodes):
        chosen = rng.choice(len(episodes), size=args.max_episodes, replace=False)
        episodes = [episodes[int(i)] for i in chosen]

    horizons_s = sorted(args.horizons_s)
    horizons_steps = [max(1, int(round(h / dt_s))) for h in horizons_s]
    max_steps = max(horizons_steps)

    # Per-horizon accumulators for each read-out method.
    acc = {m: {h: [] for h in horizons_steps} for m in ("channel", "fk", "fk_floor")}
    disp = {h: [] for h in horizons_steps}
    qdrift = {h: [] for h in horizons_steps}  # mean |q_pred - q_gt| over 4 joints (rad)
    rolled = 0

    for episode in episodes:
        states = torch.from_numpy(episode["states"]).to(args.device)
        actions = torch.from_numpy(episode["actions"]).to(args.device)
        preds = _rollout_states(model, states, actions, context, max_steps)
        if preds is None:
            continue
        rolled += 1
        steps = preds.shape[0]
        gt = states[context:context + steps]

        ee_channel = preds[:, ee_idx]                 # (S,3) direct P_ee
        ee_fk = ak.ee_base(preds[:, q_idx])           # (S,3) FK(pred q)
        ee_gt = gt[:, ee_idx]                          # (S,3) Chrono ground truth
        ee_floor = ak.ee_base(gt[:, q_idx])            # (S,3) FK(gt q): fidelity floor

        err_channel = torch.linalg.norm(ee_channel - ee_gt, dim=-1)
        err_fk = torch.linalg.norm(ee_fk - ee_gt, dim=-1)
        err_floor = torch.linalg.norm(ee_floor - ee_gt, dim=-1)
        q_abs = (preds[:, q_idx] - gt[:, q_idx]).abs().mean(dim=-1)  # (S,)

        start_ee = states[context - 1][ee_idx]
        for h in horizons_steps:
            if steps >= h:
                acc["channel"][h].append(float(err_channel[h - 1]))
                acc["fk"][h].append(float(err_fk[h - 1]))
                acc["fk_floor"][h].append(float(err_floor[h - 1]))
                disp[h].append(float(torch.linalg.norm(ee_gt[h - 1] - start_ee)))
                qdrift[h].append(float(q_abs[h - 1]))

    metrics = {
        "checkpoint": str(dynamics.checkpoint_path),
        "processed_dataset_dir": str(processed_dir),
        "geometry": str(args.geometry),
        "split": args.split,
        "context_steps": context,
        "dt_s": dt_s,
        "episodes_rolled": rolled,
        "horizons": {},
    }
    for h_s, h in zip(horizons_s, horizons_steps, strict=True):
        disps = np.asarray(disp[h], dtype=np.float64)
        if disps.size == 0:
            metrics["horizons"][f"{h_s:.2f}s"] = {"episodes": 0}
            continue
        entry = {
            "steps": h,
            "q_drift_mean_rad": float(np.mean(qdrift[h])),
            "channel": _summ(np.asarray(acc["channel"][h]), disps),
            "fk": _summ(np.asarray(acc["fk"][h]), disps),
            "fk_floor": _summ(np.asarray(acc["fk_floor"][h]), disps),
        }
        ch, fk = entry["channel"]["ee_rmse_m"], entry["fk"]["ee_rmse_m"]
        entry["winner"] = "fk" if fk < ch else "channel"
        entry["fk_rmse_ratio_vs_channel"] = float(fk / ch) if ch > 0 else None
        metrics["horizons"][f"{h_s:.2f}s"] = entry

    print(json.dumps(metrics, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
