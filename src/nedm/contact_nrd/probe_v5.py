"""Hygiene checks for version-5 models on validation shots (no test data).

1. Own-history gap: free rollouts with the model's own history against rollouts whose history differences are
   the TRUE ones, re-anchored at the predicted current state (the oracle history). Ratio of the median error of
   the target body at the target time (own / oracle). Close to 1 means the history does not amplify own errors.
2. Mid-start: rollouts that start 3 model steps before a true contact with no history (missing, masked) against
   the same rollouts started with the true history; error of every moving body 0.5 s later.
3. Older-token sensitivity: change of the next state when the older history gets Gaussian noise of the core's
   own one-step error size (sigma from training), relative to sigma.

python -m nedm.contact_nrd.probe_v5 --data D --run RUN --output OUT.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from nedm.contact_nrd.evaluate import load_data, window_labels
from nedm.contact_nrd.model_v5 import UnifiedNRD, load_v5


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=600)
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID")
    torch.set_num_threads(1)
    model, packet = load_v5(args.run / "best.pt", "cuda")
    data = load_data(args.data, "cuda", model_step_s=model.dt)
    stride, target, k = data["stride"], data["target_step"], model.k
    system = data["system"]
    slot = model.moving.index(system["target_body"])
    val = torch.nonzero(data["splits"] == 1).flatten()[: args.episodes]
    truth = data["states"][val][:, ::stride]                                   # [N, T, D, 9]
    out = {"run": str(args.run), "dt_s": model.dt, "history": k, "episodes": len(val)}

    # 1. own vs oracle history
    def rollout(oracle):
        hist, ok = model.start(truth[:, 0])
        errs = []
        for t in range(target):
            if oracle and k > 1:
                past = torch.stack([truth[:, max(0, t - j)] for j in range(k - 1, -1, -1)], 1)   # true [N, K, D, 9]
                hist = hist[:, -1:] + (past - past[:, -1:])
            state = model(hist, ok)
            hist, ok = UnifiedNRD.advance(hist, ok, state)
        return (hist[:, -1, slot, :3] - truth[:, target, slot, :3]).norm(dim=-1).cpu().numpy()
    own, oracle = rollout(False), rollout(True)
    out["own_history_target_median_mm"] = float(np.median(own) * 1e3)
    out["oracle_history_target_median_mm"] = float(np.median(oracle) * 1e3)
    out["own_over_oracle_median"] = float(np.median(own) / max(np.median(oracle), 1e-12))
    out["own_history_target_p95_mm"] = float(np.quantile(own, 0.95) * 1e3)
    out["oracle_history_target_p95_mm"] = float(np.quantile(oracle, 0.95) * 1e3)

    # 2. mid-start 3 steps before a true contact
    labels = window_labels(data["contacts"][val], stride)[:, ::stride]       # [N, T-1, P] on the model grid
    any_c = labels.any(-1) & (torch.arange(labels.shape[1], device=labels.device)[None, :] >= 3 + k)   # first contact after a full history
    first = any_c.float().argmax(1)
    has = any_c.any(1) & (first + 3 + round(0.5 / model.dt) < truth.shape[1])
    idx = torch.nonzero(has).flatten()
    if len(idx) == 0:
        raise RuntimeError("no shot has a contact after a full history")
    t0 = first[idx] - 3
    horizon = round(0.5 / model.dt)
    rows = torch.arange(len(idx), device=idx.device)
    s0 = truth[idx, t0]
    cold_h, cold_ok = model.start(s0)
    warm_h = torch.stack([truth[idx, t0 - j] for j in range(k - 1, -1, -1)], 1)
    warm_ok = torch.ones(len(idx), k, dtype=torch.bool, device=idx.device)
    for _ in range(horizon):
        cold_h, cold_ok = UnifiedNRD.advance(cold_h, cold_ok, model(cold_h, cold_ok))
        warm_h, warm_ok = UnifiedNRD.advance(warm_h, warm_ok, model(warm_h, warm_ok))
    end = truth[idx, t0 + horizon]
    cold = (cold_h[:, -1, :, :3] - end[..., :3]).norm(dim=-1).amax(-1).cpu().numpy()
    warm = (warm_h[:, -1, :, :3] - end[..., :3]).norm(dim=-1).amax(-1).cpu().numpy()
    out["midstart_shots"] = int(len(idx))
    out["midstart_cold_median_mm"], out["midstart_warm_median_mm"] = float(np.median(cold) * 1e3), float(np.median(warm) * 1e3)
    out["midstart_cold_p95_mm"], out["midstart_warm_p95_mm"] = float(np.quantile(cold, 0.95) * 1e3), float(np.quantile(warm, 0.95) * 1e3)

    # 3. older-token sensitivity (one step, true histories at contact steps)
    sigma = torch.tensor(packet["sigma"], dtype=torch.float64, device="cuda")
    pos = torch.nonzero(labels.any(-1))
    pos = pos[(pos[:, 1] >= k)][:20000]
    h = torch.stack([truth[pos[:, 0], pos[:, 1] - j] for j in range(k - 1, -1, -1)], 1)
    okf = torch.ones(len(pos), k, dtype=torch.bool, device=h.device)
    base = model(h, okf)
    hn = h.clone()
    hn[:, :-1] += torch.randn_like(hn[:, :-1]) * sigma
    moved = (model(hn, okf) - base)[..., 3:6].norm(dim=-1).amax(-1)
    out["older_noise_velocity_response_over_sigma_median"] = float((moved / sigma[3:6].norm().clamp_min(1e-30)).median()) if k > 1 else 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
