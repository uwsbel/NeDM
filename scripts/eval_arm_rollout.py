"""Open-loop multi-step rollout EE-accuracy eval for the arm dynamics model.

The arm dynamics model (``configs/arm_transformer_v1.json``) predicts per-step state
deltas for state = [q, qdot, qcmd, ee_base]. The RL-readiness question is: if a policy
rolls this model forward for k steps, how far does the *predicted* end-effector position
drift from the Chrono ground truth? This script answers that directly — it seeds each
held-out episode with the first ``context`` recorded steps, then autoregressively rolls
``next = state + predict_next_delta`` while feeding the recorded actions, and compares the
predicted ``ee_base`` channels against ground truth at several time horizons.

Two EE-readout modes, auto-detected from the model's state fields:

* **channel** (15-D model): ``ee_base`` is a state channel, so the predicted EE is read
  straight off the rolled state. The GT is the recorded ``ee_base`` state channel.
* **fk** (12-D ``[q, qd, qcmd]`` model): there is no ``ee_base`` channel, so the predicted
  EE is ``FK(predicted q)`` via ``ArmKinematics`` on the known arm geometry. The GT is the
  *same* Chrono-recorded ``ee_base`` — it lives in the processed ``rollout`` array
  (``rollout_fields = ee_base_{x,y,z}``) rather than the state — so the two modes are scored
  against identical ground truth and are directly comparable.

Either way there is no world-pose integration. The qcmd channels are rolled purely from the
model too (no deterministic overwrite), so this is a conservative bound — the RL env's exact
qcmd update can only reduce error.

Run in the nedm env:

    PYTHONPATH=src python scripts/eval_arm_rollout.py \
        --checkpoint artifacts/training_runs/arm_transformer_v1 --device cuda

    # 12-D model (FK mode auto-selected; geometry defaults to arm_geometry_v1.json):
    PYTHONPATH=src python scripts/eval_arm_rollout.py \
        --checkpoint artifacts/training_runs/arm_transformer_noee_v1 --device cuda
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

from nedm.rl.dynamics import load_frozen_dynamics
from nedm.training.dataset import load_rollout_split


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arm dynamics open-loop EE-accuracy eval.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Run directory or .pt checkpoint of a trained arm dynamics model.",
    )
    parser.add_argument(
        "--processed-dataset-dir",
        type=Path,
        default=None,
        help="Override the processed dataset dir (defaults to the checkpoint's).",
    )
    parser.add_argument("--split", type=str, default="val", help="Dataset split to roll out.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda.")
    parser.add_argument(
        "--horizons-s",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0],
        help="Rollout horizons in seconds to report EE drift at.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=400,
        help="Number of held-out episodes to roll out (seeded random subset).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Episode-subset sampling seed.")
    parser.add_argument(
        "--geometry",
        type=Path,
        default=REPO_ROOT / "artifacts/arm_geometry/arm_geometry_v1.json",
        help="Arm geometry JSON for FK mode (models without an ee_base state channel).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON metrics path.")
    return parser.parse_args(argv)


def _ee_indices(state_fields: list[str]) -> list[int]:
    return [state_fields.index(f"ee_base_{axis}") for axis in ("x", "y", "z")]


@torch.no_grad()
def _rollout_episode(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    context: int,
    max_steps: int,
    *,
    ee_idx: torch.Tensor | None = None,
    q_idx: torch.Tensor | None = None,
    arm_kin=None,
    rollout: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return (pred_ee, gt_ee) of shape (steps, 3), aligned step-for-step, or None.

    ``pred_ee[k]`` is the end-effector of the rolled state after k+1 prediction steps;
    ``gt_ee[k]`` is the recorded end-effector at the same time index. In **channel** mode
    (``ee_idx`` given) both come from the ``ee_base`` state channel; in **fk** mode
    (``arm_kin``/``q_idx`` given) the prediction is ``FK(predicted q)`` and the GT is the
    recorded ``ee_base`` from the ``rollout`` array (aligned with ``states``).
    """
    total = states.shape[0]
    steps_avail = total - context
    if steps_avail <= 0:
        return None
    steps = min(max_steps, steps_avail)

    hist_states = states[:context].clone()
    hist_actions = actions[:context].clone()
    pred_states: list[torch.Tensor] = []
    for step in range(steps):
        delta = model.predict_next_delta(
            hist_states[-context:].unsqueeze(0),
            hist_actions[-context:].unsqueeze(0),
            terrain=None,
        ).squeeze(0)
        next_state = hist_states[-1] + delta
        pred_states.append(next_state)
        if context + step < actions.shape[0]:
            hist_actions = torch.cat([hist_actions, actions[context + step].unsqueeze(0)], dim=0)
        hist_states = torch.cat([hist_states, next_state.unsqueeze(0)], dim=0)

    pred = torch.stack(pred_states, dim=0)  # (steps, state_dim)
    if arm_kin is not None:  # fk mode
        pred_ee = arm_kin.ee_base(pred[:, q_idx])
        gt_ee = rollout[context : context + steps]
    else:  # channel mode
        pred_ee = pred[:, ee_idx]
        gt_ee = states[context : context + steps][:, ee_idx]
    return pred_ee, gt_ee


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

    # Auto-select the EE readout: channel mode when ee_base is a state field, else FK mode.
    has_ee_channel = all(f"ee_base_{axis}" in state_fields for axis in ("x", "y", "z"))
    ee_idx = None
    q_idx = None
    arm_kin = None
    if has_ee_channel:
        ee_mode = "channel"
        ee_idx = torch.tensor(_ee_indices(state_fields), dtype=torch.long, device=args.device)
    else:
        ee_mode = "fk"
        from nedm.rl.arm_kinematics import ArmKinematics

        if not args.geometry.exists():
            raise FileNotFoundError(
                f"FK mode needs an arm geometry JSON but {args.geometry} does not exist "
                "(pass --geometry)."
            )
        arm_kin = ArmKinematics.from_json(
            str(args.geometry), device=args.device, dtype=torch.float32
        )
        q_idx = torch.tensor(
            [state_fields.index(f"q_{i}") for i in range(arm_kin.num_joints)],
            dtype=torch.long,
            device=args.device,
        )

    processed_dir = Path(
        args.processed_dataset_dir
        if args.processed_dataset_dir is not None
        else dynamics.config["processed_dataset_dir"]
    ).resolve()
    split_data = load_rollout_split(processed_dir, args.split)
    episodes = split_data["episodes"]

    rng = np.random.default_rng(args.seed)
    if args.max_episodes < len(episodes):
        chosen = rng.choice(len(episodes), size=args.max_episodes, replace=False)
        episodes = [episodes[int(i)] for i in chosen]

    horizons_s = sorted(args.horizons_s)
    horizons_steps = [max(1, int(round(h / dt_s))) for h in horizons_s]
    max_steps = max(horizons_steps)

    # Per-horizon accumulators of cumulative EE drift, plus ground-truth displacement.
    horizon_err: dict[int, list[float]] = {h: [] for h in horizons_steps}
    horizon_gt_disp: dict[int, list[float]] = {h: [] for h in horizons_steps}
    one_step_sq = 0.0
    one_step_count = 0
    rolled = 0

    for episode in episodes:
        states = torch.from_numpy(episode["states"]).to(args.device)
        actions = torch.from_numpy(episode["actions"]).to(args.device)
        rollout = (
            torch.from_numpy(episode["rollout"]).to(args.device) if ee_mode == "fk" else None
        )
        result = _rollout_episode(
            model,
            states,
            actions,
            context,
            max_steps,
            ee_idx=ee_idx,
            q_idx=q_idx,
            arm_kin=arm_kin,
            rollout=rollout,
        )
        if result is None:
            continue
        pred_ee, gt_ee = result
        rolled += 1
        err = torch.linalg.norm(pred_ee - gt_ee, dim=-1)  # (steps,)
        one_step_sq += float(err[0].pow(2).item())
        one_step_count += 1
        # Recorded EE at the last seed step, in the same space as gt_ee (displacement ref).
        start_ee = rollout[context - 1] if ee_mode == "fk" else states[context - 1][ee_idx]
        for h in horizons_steps:
            if err.shape[0] >= h:
                horizon_err[h].append(float(err[h - 1].item()))
                horizon_gt_disp[h].append(float(torch.linalg.norm(gt_ee[h - 1] - start_ee).item()))

    metrics: dict[str, object] = {
        "checkpoint": str(dynamics.checkpoint_path),
        "processed_dataset_dir": str(processed_dir),
        "split": args.split,
        "ee_mode": ee_mode,
        "geometry": str(args.geometry) if ee_mode == "fk" else None,
        "context_steps": context,
        "dt_s": dt_s,
        "episodes_rolled": rolled,
        "one_step_ee_rmse_m": float(np.sqrt(one_step_sq / max(one_step_count, 1))),
        "horizons": {},
    }
    for h_s, h in zip(horizons_s, horizons_steps, strict=True):
        errs = np.asarray(horizon_err[h], dtype=np.float64)
        disps = np.asarray(horizon_gt_disp[h], dtype=np.float64)
        if errs.size == 0:
            metrics["horizons"][f"{h_s:.2f}s"] = {"episodes": 0}
            continue
        mean_disp = float(disps.mean())
        metrics["horizons"][f"{h_s:.2f}s"] = {
            "steps": h,
            "episodes": int(errs.size),
            "ee_rmse_m": float(np.sqrt((errs ** 2).mean())),
            "ee_mean_m": float(errs.mean()),
            "ee_p90_m": float(np.percentile(errs, 90)),
            "gt_disp_mean_m": mean_disp,
            "errdist": float(np.sqrt((errs ** 2).mean()) / mean_disp) if mean_disp > 1e-6 else None,
        }

    print(json.dumps(metrics, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
