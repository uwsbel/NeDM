#!/usr/bin/env python
"""Calibrate a drive-power model on the recorded episodes (plan open item: energy).

The learned power head matches recorded drives when replaying recorded actions but
under-reports by ~1.6x under the tracker's actions in imagination, while the
imagined kinematics (time-to-goal) are right. So fit power from KINEMATIC features
of z1 only -- speed, acceleration, grade (pitch), cornering -- which are what the
dynamics model predicts well, and optionally the actions for comparison.

Writes coefficients + held-out readout to artifacts/traverse/wp4_power_calib/.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nedm.traverse import nrd_data as D
from nedm.traverse.nrd_model import DT_S

from nedm.traverse.power_calib import KINDS, backward_accel, power_features


def load(cache: Path, keys: list[str]):
    z1, act, pw = [], [], []
    for k in keys:
        with np.load(cache / f"{k}.npz") as d:
            z1.append(d["z1"]); act.append(d["act"]); pw.append(d["power"][:, 0])
    return np.stack(z1), np.stack(act), np.stack(pw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6")
    ap.add_argument("--out", default="artifacts/traverse/wp4_power_calib")
    ap.add_argument("--train-episodes", type=int, default=2000)
    ap.add_argument("--ridge", type=float, default=1e-3)
    args = ap.parse_args()
    cache = Path(args.cache)
    keys = D.load_cache_keys(cache)
    tr, va, _ = D.split_keys(keys)
    rng = np.random.default_rng(0)
    tr = [tr[i] for i in rng.permutation(len(tr))[: args.train_episodes]]
    t0 = time.time()
    z1_tr, act_tr, p_tr = load(cache, tr)
    z1_va, act_va, p_va = load(cache, va)
    print(f"loaded {len(tr)} train / {len(va)} val episodes in {time.time() - t0:.0f}s; "
          f"power kW: mean {p_tr.mean():.1f}, p1 {np.percentile(p_tr, 1):.1f}, p99 {np.percentile(p_tr, 99):.1f}, "
          f"neg frac {(p_tr < 0).mean():.3f}", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    readout = {}
    for kind in KINDS:
        feat = lambda z, a: power_features(z, a, backward_accel(z[:, 0], DT_S), kind, np)
        X = np.concatenate([feat(z, a) for z, a in zip(z1_tr, act_tr)])
        y = p_tr.reshape(-1)
        mu, sd = X.mean(0), X.std(0) + 1e-9; sd[0] = 1.0; mu[0] = 0.0
        Xn = (X - mu) / sd
        w = np.linalg.solve(Xn.T @ Xn + args.ridge * len(y) * np.eye(Xn.shape[1]), Xn.T @ y)
        def predict(z, a):
            return ((feat(z, a) - mu) / sd) @ w
        pred_va = np.stack([predict(z, a) for z, a in zip(z1_va, act_va)])
        r2 = 1 - ((pred_va - p_va) ** 2).sum() / ((p_va - p_va.mean()) ** 2).sum()
        e_rec, e_pred = p_va.sum(1) * DT_S, pred_va.sum(1) * DT_S
        e_pred_clip = np.maximum(pred_va, 0).sum(1) * DT_S
        res = {"features": Xn.shape[1], "step_r2": float(r2), "step_rmse_kw": float(np.sqrt(((pred_va - p_va) ** 2).mean())),
               "episode_energy": {"mean_rec_kj": float(e_rec.mean()), "mean_pred_kj": float(e_pred.mean()),
                                  "mae_kj": float(np.abs(e_pred - e_rec).mean()), "corr": float(np.corrcoef(e_rec, e_pred)[0, 1]),
                                  "mae_kj_clip0": float(np.abs(e_pred_clip - e_rec).mean())},
               "w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist()}
        readout[kind] = res
        print(f"{kind:9s} step R2 {res['step_r2']:.3f}  episode energy corr {res['episode_energy']['corr']:.3f}  MAE {res['episode_energy']['mae_kj']:.1f} kJ", flush=True)
    (out / "power_calib.json").write_text(json.dumps(readout, indent=1))


if __name__ == "__main__":
    main()
