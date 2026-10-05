"""Data loading and free-rollout evaluation for the unified contact NRD (any system)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def window_labels(contacts, stride):
    """[N, T-1, P] interval labels -> [N, T-stride, P]: pair in contact during the step starting at each record."""
    c = contacts.to(torch.float32)
    cum = torch.cat((torch.zeros_like(c[:, :1]), c.cumsum(1)), 1)
    return (cum[:, stride:] - cum[:, :-stride]) > 0.5


def load_data(root, device, max_episodes=None, model_step_s=None):
    root = Path(root)
    system = json.loads((root / "system.json").read_text())
    index = json.loads((root / "index.json").read_text())
    with np.load(root / "unified_data.npz") as packet:
        arrays = {k: packet[k] for k in ("states", "contacts", "lengths", "splits", "launches")}
    episodes = index["episodes"]
    if max_episodes:
        parts = []
        for split, count in ((0, max_episodes), (1, max(1, max_episodes // 5)), (2, max(1, max_episodes // 5))):
            members = np.flatnonzero(arrays["splits"] == split)
            if len(members):
                parts.append(members[np.linspace(0, len(members) - 1, min(count, len(members))).round().astype(int)])
        keep = np.unique(np.concatenate(parts))
        arrays = {k: v[keep] for k, v in arrays.items()}
        episodes = [episodes[k] for k in keep]
    record = system["record_step_s"]
    if model_step_s:   # a different model step on the same records (e.g. 20 ms)
        system = {**system, "model_step_s": float(model_step_s)}
    stride = round(system["model_step_s"] / record)
    eligible = eligibility(system, arrays["states"], episodes)
    return dict(system=system, index=index, episodes=episodes, stride=stride, dt=stride * record, record_dt=record,
                target_step=round(system["target_time_s"] / (stride * record)),
                states=torch.as_tensor(arrays["states"], device=device, dtype=torch.float64),
                contacts=torch.as_tensor(arrays["contacts"], device=device).bool(),
                lengths=torch.as_tensor(arrays["lengths"], device=device),
                splits=torch.as_tensor(arrays["splits"], device=device),
                launches=torch.as_tensor(arrays["launches"], device=device, dtype=torch.float64),
                eligible=torch.as_tensor(eligible, device=device))


def eligibility(system, states, episodes):
    """Pool: the predeclared target-worthy filter from the pool study. Other systems: every episode."""
    if system["name"] != "pool":
        return np.ones(len(episodes), dtype=bool)
    from nedm.pool_ball.evaluate import target_eligibility
    s14 = np.concatenate([np.concatenate((states[:, :, b, 0:2], states[:, :, b, 3:5], states[:, :, b, 6:9]), -1) for b in (0, 1)], -1)
    return target_eligibility(system["physics_config"], system["target_time_s"], s14, episodes)


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)), "p95": float(np.quantile(values, 0.95)),
            "max": float(values.max())}


def rising(x):
    """Rising edges along time of a [T, P] boolean array -> counts per pair."""
    x = np.asarray(x, dtype=bool)
    return (x[1:] & ~x[:-1]).sum(0) + x[0]


@torch.no_grad()
def free_metrics(model, data, indices, save_traces=None, batch=512):
    """Rollouts from each episode's first state on the model grid (phase 0)."""
    if abs(model.dt - data["dt"]) > 1e-12:
        raise ValueError("model and data steps differ")
    stride, target = data["stride"], data["target_step"]
    system = data["system"]
    body = system["target_body"]
    slot = model.moving.index(body)
    horizon = system.get("rmse_horizon", "target" if system["name"] == "pool" else "episode")
    labels = window_labels(data["contacts"][indices], stride)[:, ::stride].cpu().numpy()
    entries, traces = [], []
    for start in range(0, len(indices), batch):
        chosen = indices[start:start + batch]
        truth = data["states"][chosen][:, ::stride]
        steps = truth.shape[1] - 1
        pred, gates = model.rollout(truth[:, 0], steps, return_gates=True)
        pred, gates, truth_np = pred.cpu().numpy(), gates.cpu().numpy() > 0.5, truth.cpu().numpy()
        for row, idx in enumerate(chosen.tolist()):
            n = (int(data["lengths"][idx]) - 1) // stride + 1
            p, t = pred[row, :n], truth_np[row, :n]
            item = {"index": idx, "finite": bool(np.isfinite(p).all())}
            if item["finite"]:
                err = np.linalg.norm(p[:, slot, :3] - t[:, slot, :3], axis=-1)
                upto = target + 1 if horizon == "target" else n
                k = min(target, n - 1)
                g = gates[row, : n - 1]
                truth_events = rising(labels[start + row, : n - 1])
                pred_events = rising(g)
                if system["name"] == "ball" and (truth_events == pred_events).all() and pred_events.min() > 0:
                    # Ground before wall, as the ball study's contact-order check.
                    pred_events = pred_events if np.argmax(g[:, 0]) < np.argmax(g[:, 1]) else pred_events + 1
                item.update(rmse_m=float(np.sqrt(np.mean(err[:upto] ** 2))), at_target_m=float(err[k]),
                            end_m=float(err[-1]), max_m=float(err[:upto].max()),
                            all_bodies_rmse_m=float(np.sqrt(np.mean(np.linalg.norm(p[:upto, :, :3] - t[:upto, :, :3], axis=-1) ** 2))),
                            event_ok=bool((truth_events == pred_events).all()),
                            events_true=truth_events.tolist(), events_pred=pred_events.tolist())
            entries.append(item)
        if save_traces is not None:
            traces.append(pred)
    finite = [e for e in entries if e["finite"]]
    big = dict(mean=1e3, median=1e3, p95=1e3, max=1e3)
    result = {"episodes": len(entries), "finite_fraction": len(finite) / len(entries),
              "event_ok_fraction": sum(e.get("event_ok", False) for e in entries) / len(entries), "per_episode": entries,
              "target_step": target, "dt_s": data["dt"]}
    for key in ("rmse_m", "at_target_m", "end_m", "max_m", "all_bodies_rmse_m"):
        result[key] = quantiles([e[key] for e in finite]) if finite else big
    eligible = data["eligible"].cpu().numpy()
    chosen = [e for e in finite if eligible[e["index"]]]
    result["eligible_episodes"] = len(chosen)
    result["eligible_at_target_m"] = quantiles([e["at_target_m"] for e in chosen]) if chosen else big
    result["over_10mm_at_target"] = sum(e["at_target_m"] > 0.01 for e in finite)
    if system["name"] == "ball":
        # The bouncing-ball study's own score: trajectory p95 + 0.5 endpoint p95 + order and finiteness penalties.
        result["selection_score"] = (result["rmse_m"]["p95"] + 0.5 * result["end_m"]["p95"]
                                     + 10 * (1 - result["event_ok_fraction"]) + 100 * (1 - result["finite_fraction"]))
    else:
        result["selection_score"] = (result["eligible_at_target_m"]["p95"] + 0.5 * result["rmse_m"]["p95"]
                                     + 0.05 * (1 - result["event_ok_fraction"]) + (1 - result["finite_fraction"]))
    if save_traces is not None:
        np.savez_compressed(save_traces, predicted=np.concatenate(traces), indices=indices.cpu().numpy(), dt_s=model.dt)
    return result
