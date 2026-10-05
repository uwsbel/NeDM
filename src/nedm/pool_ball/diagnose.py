"""Find where free rollouts first go wrong: step, which switches fired, and the
predicted vs true states around it (validation episodes only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nedm.pool_ball.evaluate import load_data
from nedm.pool_ball.model import BALL, load_pool
from nedm.pool_ball.train import window_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--threshold-m", type=float, default=0.02)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, packet = load_pool(args.checkpoint, "cuda")
    data = load_data(args.data, "cuda")
    stride = data["stride"]
    val = torch.nonzero(data["splits"] == 1).flatten()[: args.episodes]
    truth = data["states"][val][:, ::stride]
    labels = window_labels(data["contacts"][val], stride)[:, ::stride]
    with torch.no_grad():
        pred, gates = model.rollout(truth[:, 0], truth.shape[1] - 1, return_gates=True)
        # Teacher-forced one-step check on the same grid: true state in, gates as predicted.
        s = truth[:, :-1].reshape(-1, 14)
        one, info = model.details(s)
        one = one.reshape(truth.shape[0], -1, 14)
        tf_logits = info["logits"].reshape(truth.shape[0], truth.shape[1] - 1, -1)
    err = (pred[..., BALL:BALL + 2] - truth[..., BALL:BALL + 2]).norm(dim=-1)
    err_a = (pred[..., :2] - truth[..., :2]).norm(dim=-1)
    one_err = (one - truth[:, 1:]).abs()
    out = {"stage": packet["stage"], "update": packet["update"], "episodes": len(val), "rows": []}
    first_bad = []
    for i in range(len(val)):
        bad = torch.nonzero((err[i] > args.threshold_m) | (err_a[i] > args.threshold_m)).flatten()
        if not len(bad):
            continue
        k = int(bad[0])
        first_bad.append(k)
        lo = max(0, k - 3)
        rows = []
        for j in range(lo, min(k + 2, truth.shape[1] - 1)):
            rows.append({"step": j, "gates_pred": gates[i, j].tolist(), "labels_true": labels[i, j].int().tolist(),
                         "tf_logits": tf_logits[i, j].tolist(),
                         "pred_A": pred[i, j, :4].tolist(), "true_A": truth[i, j, :4].tolist(),
                         "pred_B": pred[i, j, BALL:BALL + 4].tolist(), "true_B": truth[i, j, BALL:BALL + 4].tolist()})
        if len(out["rows"]) < 12:
            out["rows"].append({"episode": int(val[i]), "first_bad_step": k, "trace": rows})
    gate_counts_pred = gates.sum(1).mean(0).tolist() if gates is not None else None
    out.update(bad_episodes=len(first_bad), first_bad_step_hist=np.bincount(np.array(first_bad) // 10, minlength=26).tolist() if first_bad else [],
               mean_gate_firings_per_episode=gate_counts_pred, mean_true_windows_per_episode=labels.float().sum(1).mean(0).tolist(),
               teacher_forced_p95={name: float(torch.quantile(one_err[..., cols].amax(-1).flatten().double(), 0.95))
                                   for name, cols in (("A_pos", [0, 1]), ("A_vel", [2, 3]), ("B_pos", [7, 8]), ("B_vel", [9, 10]))},
               teacher_forced_max={name: float(one_err[..., cols].amax().double())
                                   for name, cols in (("A_pos", [0, 1]), ("A_vel", [2, 3]), ("B_pos", [7, 8]), ("B_vel", [9, 10]))})
    print(json.dumps(out, indent=1)[:20000])


if __name__ == "__main__":
    main()
