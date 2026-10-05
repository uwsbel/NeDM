"""Where does the error come from? Teacher-forced one-step errors per event
type (physical units) and a breakdown of free-rollout errors at t by whether
the predicted events match (validation only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nedm.pool_ball.evaluate import free_metrics, load_data
from nedm.pool_ball.model import BALL, load_pool
from nedm.pool_ball.train import window_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, packet = load_pool(args.checkpoint, "cuda")
    data = load_data(args.data, "cuda")
    stride = data["stride"]
    val = torch.nonzero(data["splits"] == 1).flatten()
    states = data["states"][val]
    labels = window_labels(data["contacts"][val], stride)
    n = labels.shape[1]
    out = {"stage": packet["stage"], "update": packet["update"]}
    with torch.no_grad():
        for name, idx in (("ab", 0), ("a_cushion", 1), ("b_cushion", 2)):
            pairs = torch.nonzero(labels[..., idx] & (labels.sum(-1) == 1))
            pairs = pairs[torch.randperm(len(pairs), device=pairs.device)[:50000]]
            s, y = states[pairs[:, 0], pairs[:, 1]], states[pairs[:, 0], pairs[:, 1] + stride]
            gates = torch.zeros(len(s), model.logits(s[:1]).shape[-1], dtype=s.dtype, device=s.device)
            gates[:, 0 if model.contact == "scalar_shared" else idx] = 1
            pred, info = model.details(s, gates=gates)
            free_gates = model.gates(info["logits"])
            err = pred - y
            entry = {"count": len(s), "gate_on_fraction": float((free_gates[:, 0 if model.contact == "scalar_shared" else idx] > 0.5).double().mean())}
            for who, sl in (("A", slice(0, BALL)), ("B", slice(BALL, 2 * BALL))):
                e = err[:, sl]
                v_true = y[:, sl][:, 2:4]
                v_pred = pred[:, sl][:, 2:4]
                ang = torch.atan2(v_pred[:, 1], v_pred[:, 0]) - torch.atan2(v_true[:, 1], v_true[:, 0])
                ang = (ang + np.pi) % (2 * np.pi) - np.pi
                moving = v_true.norm(dim=-1) > 0.05
                entry[who] = {q: {"pos_mm": float(torch.quantile(e[:, :2].norm(dim=-1), qq) * 1e3),
                                  "vel_mps": float(torch.quantile(e[:, 2:4].norm(dim=-1), qq)),
                                  "dir_mrad": float(torch.quantile(ang[moving].abs(), qq) * 1e3) if moving.any() else None,
                                  "spin_radps": float(torch.quantile(e[:, 4:7].norm(dim=-1), qq))}
                              for q, qq in (("median", 0.5), ("p95", 0.95))}
            out[name] = entry
    report = free_metrics(model, data, val)
    per = [e for e in report["per_episode"] if e["finite"]]
    ok = [e for e in per if e["event_ok"]]
    bad = [e for e in per if not e["event_ok"]]
    for label, group in (("events_ok", ok), ("events_wrong", bad)):
        b = np.array([e["b_at_target_m"] for e in group]) * 1e3 if group else np.array([np.nan])
        out[f"rollout_{label}"] = {"count": len(group), "b_at_t_median_mm": float(np.median(b)), "b_at_t_p90_mm": float(np.quantile(b, 0.9)),
                                   "b_rmse_median_mm": float(np.median([e["b_rmse_m"] for e in group]) * 1e3) if group else None}
    kinds = {}
    for e in bad:
        t, p = e["events_true"], e["events_pred"]
        key = ("departure" if abs(t[0] - p[0]) > 1 else "") + ("|B_cushions" if t[1] != p[1] else "") + ("|A_cushions" if t[2] != p[2] else "")
        kinds[key] = kinds.get(key, 0) + 1
    out["wrong_event_kinds"] = kinds
    eligible = data["eligible"].cpu().numpy()
    el = np.array([e["b_at_target_m"] for e in per if eligible[e["index"]]]) * 1e3
    out["eligible_b_at_t_mm"] = {"count": len(el), "median": float(np.median(el)), "p75": float(np.quantile(el, 0.75)),
                                 "p90": float(np.quantile(el, 0.9)), "within_10mm": float((el <= 10).mean())}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
