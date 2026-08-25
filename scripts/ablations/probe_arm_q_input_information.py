#!/usr/bin/env python
"""How much does the arm's next-step joint acceleration depend on q beyond qd/qcmd?

Reviewer question: "why feed [q, qd] to the arm ROM instead of qd only and recover
q by integrating qd?"  This probe answers the *one-step* half of that question on
the processed 8-D arm cache: matched-capacity MLPs regress the normalized
per-step delta_qd (i.e. the joint acceleration) from different input feature sets
and we compare held-out MSE / R^2.  The closed-loop half is the transformer
ablation (configs/ablations/arm_transformer_8d_{qdonly,integq}_v1.json).

Feature sets (all normalized with the cache statistics):
  full            [q, qd, qcmd]              -- the deployed input token
  no_q            [qd, qcmd]                 -- reviewer's proposal, no history
  no_q_hist16     [qd, qcmd] x 16 steps      -- what a qd-only ctx-16 transformer sees
  full_hist16     [q, qd, qcmd] x 16 steps   -- deployed token with history
  pd_err_only     [qd, qcmd - q]             -- knows the PD error but not absolute q
  full_pd_err     [q, qd, qcmd - q]          -- reparameterized deployed token
  drop_base_yaw   [q1..q3, qd, qcmd - q]     -- is base yaw a cyclic coordinate?

Physics: M(q) qdd + C(q, qd) qd + g(q) = clamp(Kp (qcmd - q) - Kd qd), plus joint
limits and hull/self collision -- every term depends on q, so a qd-only network
is non-identifiable at one step; with history it can partially infer (qcmd - q)
from the PD response (a fragile in-context system-ID shortcut).

Usage:
    PYTHONPATH=src python scripts/ablations/probe_arm_q_input_information.py \
        [--dataset-dir artifacts/training_datasets/arm_dyn_v3_8d_seq16_v1] [--seeds 3] [--epochs 60]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FEATURE_SETS = ["full", "no_q", "no_q_hist16", "full_hist16", "pd_err_only", "full_pd_err", "drop_base_yaw"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/training_datasets/arm_dyn_v3_8d_seq16_v1"))
    parser.add_argument("--history", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON results path.")
    return parser.parse_args()


def load_split(root: Path, split: str):
    states = np.load(root / f"{split}_states.npy").astype(np.float32)
    actions = np.load(root / f"{split}_actions.npy").astype(np.float32)
    targets = np.load(root / f"{split}_targets.npy").astype(np.float32)
    starts = np.load(root / f"{split}_episode_starts.npy")
    lengths = np.load(root / f"{split}_episode_lengths.npy")
    return states, actions, targets, starts, lengths


def windows(states, actions, targets, starts, lengths, history: int):
    """Rows with >= history-1 in-episode predecessors -> ([N,H,8], [N,H,4], [N,8])."""
    idx = np.concatenate([np.arange(s + history - 1, s + l) for s, l in zip(starts, lengths) if l >= history])
    window_index = idx[:, None] + np.arange(-history + 1, 1)[None, :]
    return states[window_index], actions[window_index], targets[idx]


def build_features(kind: str, s_win, a_win, norm) -> np.ndarray:
    q, qd, qc = s_win[:, :, :4], s_win[:, :, 4:], a_win
    qn = (q - norm["state_mean"][:4]) / norm["state_std"][:4]
    qdn = (qd - norm["state_mean"][4:]) / norm["state_std"][4:]
    qcn = (qc - norm["action_mean"]) / norm["action_std"]
    en = (qc - q) / norm["action_std"]  # PD error, in action units
    cur = lambda x: x[:, -1, :]  # noqa: E731
    hist = lambda x: x.reshape(len(x), -1)  # noqa: E731
    table = {
        "full": [cur(qn), cur(qdn), cur(qcn)],
        "no_q": [cur(qdn), cur(qcn)],
        "no_q_hist16": [hist(qdn), hist(qcn)],
        "full_hist16": [hist(qn), hist(qdn), hist(qcn)],
        "pd_err_only": [cur(qdn), cur(en)],
        "full_pd_err": [cur(qn), cur(qdn), cur(en)],
        "drop_base_yaw": [cur(qn)[:, 1:], cur(qdn), cur(en)],
    }
    return np.concatenate(table[kind], axis=1)


def fit_mlp(x, y, x_val, y_val, *, seed, epochs, width, depth, device, batch=4096, lr=1e-3):
    torch.manual_seed(seed)
    x, y, x_val, y_val = (torch.tensor(a, device=device) for a in (x, y, x_val, y_val))
    layers, dim = [], x.shape[1]
    for _ in range(depth):
        layers += [nn.Linear(dim, width), nn.SiLU()]
        dim = width
    net = nn.Sequential(*layers, nn.Linear(dim, y.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(x) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    for _ in range(epochs):
        perm = torch.randperm(len(x), device=device)
        for i in range(len(x) // batch):
            b = perm[i * batch : (i + 1) * batch]
            loss = ((net(x[b]) - y[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        pred = torch.cat([net(x_val[i : i + 65536]) for i in range(0, len(x_val), 65536)])
        resid = pred - y_val
        mse = resid.pow(2).mean().item()
        r2 = 1 - resid.pow(2).sum(0) / (y_val - y_val.mean(0)).pow(2).sum(0)
    return mse, r2.cpu().numpy()


def main() -> int:
    args = parse_args()
    root = args.dataset_dir
    meta = json.load(open(root / "metadata.json"))
    norm = {k: np.asarray(v, dtype=np.float32) for k, v in meta["normalization"].items()}
    t0 = time.time()
    s_tr, a_tr, y_tr = windows(*load_split(root, "train"), args.history)
    s_va, a_va, y_va = windows(*load_split(root, "val"), args.history)
    print(f"train windows {len(s_tr)}, val windows {len(s_va)}  ({time.time() - t0:.1f}s)")
    y_tr = ((y_tr - norm["target_mean"]) / norm["target_std"])[:, 4:]  # normalized delta_qd
    y_va = ((y_va - norm["target_mean"]) / norm["target_std"])[:, 4:]

    results = {}
    print(f"{'features':16s} {'val MSE(dqd,norm) mean±sd':>28s}   R2 per joint (mean over seeds)")
    for kind in FEATURE_SETS:
        x_tr = build_features(kind, s_tr, a_tr, norm)
        x_va = build_features(kind, s_va, a_va, norm)
        mses, r2s = [], []
        for seed in range(args.seeds):
            mse, r2 = fit_mlp(x_tr, y_tr, x_va, y_va, seed=seed, epochs=args.epochs, width=args.width, depth=args.depth, device=args.device)
            mses.append(mse)
            r2s.append(r2)
        results[kind] = {"input_dim": int(x_tr.shape[1]), "mse_mean": float(np.mean(mses)), "mse_std": float(np.std(mses)), "mse_seeds": [float(m) for m in mses], "r2_mean": [float(v) for v in np.mean(r2s, 0)]}
        print(f"{kind:16s} {np.mean(mses):16.4f} ± {np.std(mses):.4f}   {np.round(np.mean(r2s, 0), 3)}   [{x_tr.shape[1]}-D]  ({time.time() - t0:.0f}s)")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"dataset_dir": str(root), "history": args.history, "epochs": args.epochs, "seeds": args.seeds, "width": args.width, "depth": args.depth, "results": results}, open(args.output, "w"), indent=2)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
