"""Data loading and matched free-rollout evaluation for the pool NRD.

Rollouts start from each episode's launch state and run on the 10 ms grid
(phase 0); predictions are compared with Chrono at the same instants. Scene
geometry is used only to score events (cushion hits), never in the model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from nedm.pool_ball.model import BALL, initial_state


def load_data(root, device, max_episodes=None, model_step_s=None):
    root = Path(root)
    index = json.loads((root / "campaign_index.json").read_text())
    with np.load(root / "model_data.npz") as packet:
        arrays = {k: packet[k] for k in ("states", "contacts", "lengths", "splits", "launches")}
    keep = None
    if max_episodes:
        # Evenly spaced subset of each split (episodes are stored cell by
        # cell, so a prefix would cover only the slowest launches).
        parts = []
        for split, count in ((0, max_episodes), (1, max(1, max_episodes // 5)), (2, max(1, max_episodes // 5))):
            members = np.flatnonzero(arrays["splits"] == split)
            if len(members):
                parts.append(members[np.linspace(0, len(members) - 1, min(count, len(members))).round().astype(int)])
        keep = np.unique(np.concatenate(parts))
        arrays = {k: v[keep] for k, v in arrays.items()}
    record = index["record_step_s"]
    stride = round((model_step_s or index["campaign"]["model_step_s"]) / record)   # optional other model step (e.g. 20 ms)
    episodes = index["episodes"] if not max_episodes else [index["episodes"][k] for k in keep]
    eligible = target_eligibility(index["config"], index["campaign"]["target_time_s"], arrays["states"], episodes)
    return dict(index=index, stride=stride, dt=stride * record, record_dt=record, episodes=episodes,
                eligible=torch.as_tensor(eligible, device=device),
                target_step=round(index["campaign"]["target_time_s"] / (stride * record)),
                states=torch.as_tensor(arrays["states"], device=device, dtype=torch.float64),
                contacts=torch.as_tensor(arrays["contacts"], device=device).bool(),
                lengths=torch.as_tensor(arrays["lengths"], device=device),
                splits=torch.as_tensor(arrays["splits"], device=device),
                launches=torch.as_tensor(arrays["launches"], device=device, dtype=torch.float64),
                scene=index["config"]["scene"])


def cut_angle_deg(config, aim_rad):
    a, b = (np.asarray(config["scene"][k], float) for k in ("ball_a_xy_m", "ball_b_xy_m"))
    ratio = np.linalg.norm(b - a) * np.sin(np.asarray(aim_rad, float)) / (2 * config["scene"]["radius_m"])
    return np.degrees(np.arcsin(np.clip(ratio, -1, 1)))


def target_eligibility(config, t_target, states, episodes, max_cut_deg=55.0):
    """Predeclared filter for target-worthy episodes (well-posed B positions at t).

    |cut| <= 55 deg; exactly one A-B contact in the episode; B moved >= 0.3 m;
    B >= 80 mm (surface) from every cushion at t; at most one B cushion contact
    before t, and the last one >= 0.15 s before t and >= 150 mm from a corner;
    A >= 120 mm from B at t.
    """
    scene = config["scene"]
    R, hx, hy = scene["radius_m"], scene["half_length_m"], scene["half_width_m"]
    record = config["simulation"]["record_step_s"]
    k = round(t_target / record)
    out = np.zeros(len(episodes), dtype=bool)
    for i, e in enumerate(episodes):
        if abs(cut_angle_deg(config, e["aim_rad"])) > max_cut_deg or e["ab_contacts"] != 1:
            continue
        b, a = states[i, k, 7:9].astype(float), states[i, k, 0:2].astype(float)
        if np.linalg.norm(b - np.asarray(scene["ball_b_xy_m"])) < 0.3 or np.linalg.norm(a - b) < 0.12:
            continue
        if min(hx - abs(b[0]), hy - abs(b[1])) - R < 0.08:
            continue
        hits = [ev for ev in e["events"] if ev["kind"] == "start" and ev["pair"].startswith("B_") and ev["time_s"] < t_target]
        if len(hits) > 1:
            continue
        if hits:
            ev = hits[-1]
            if t_target - ev["time_s"] < 0.15:
                continue
            j = round(ev["time_s"] / record)
            px, py = states[i, j, 7], states[i, j, 8]
            # Distance along the cushion from the contact to the nearest corner.
            along = (hx - abs(px)) if ev["pair"][2] == "y" else (hy - abs(py))
            if along < 0.15:
                continue
        out[i] = True
    return out


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)), "p95": float(np.quantile(values, 0.95)),
            "max": float(values.max())}


def cushion_hits(track, scene, upto):
    """Count cushion rebounds of one ball: sign flips of the velocity component
    normal to a cushion while the ball is within 5 cm of it. track [T, 7]."""
    hx, hy = scene["half_length_m"] - scene["radius_m"], scene["half_width_m"] - scene["radius_m"]
    t = track[: upto + 1]
    count = 0
    for axis, half in ((0, hx), (1, hy)):
        v, p = t[:, 2 + axis], t[:, axis]
        flips = (np.sign(v[:-1]) != np.sign(v[1:])) & (np.abs(v[:-1]) > 0.02) & (np.abs(p[1:]) > half - 0.05)
        count += int(flips.sum())
    return count


def departure_step(track, threshold=0.02):
    moving = np.flatnonzero(np.hypot(track[:, 2], track[:, 3]) > threshold)
    return int(moving[0]) if len(moving) else -1


@torch.no_grad()
def free_metrics(model, data, indices, save_traces=None):
    if abs(model.dt - data["dt"]) > 1e-12:
        raise ValueError(f"model step {model.dt} differs from data step {data['dt']}")
    stride, target = data["stride"], data["target_step"]
    scene = data["scene"]
    entries, traces = [], []
    for chosen in indices.split(512):
        truth = data["states"][chosen][:, ::stride]
        steps = truth.shape[1] - 1
        prediction = model.rollout(truth[:, 0], steps)
        pred = prediction.cpu().numpy()
        actual = truth.cpu().numpy()
        for row, idx in enumerate(chosen.tolist()):
            p, t = pred[row], actual[row]
            finite = bool(np.isfinite(p).all())
            item = {"index": idx, "finite": finite}
            if finite:
                b_err = np.linalg.norm(p[:, BALL:BALL + 2] - t[:, BALL:BALL + 2], axis=-1)
                a_err = np.linalg.norm(p[:, :2] - t[:, :2], axis=-1)
                events_true = (departure_step(t[:, BALL:]), cushion_hits(t[:, BALL:], scene, target), cushion_hits(t[:, :BALL], scene, target))
                events_pred = (departure_step(p[:, BALL:]), cushion_hits(p[:, BALL:], scene, target), cushion_hits(p[:, :BALL], scene, target))
                item.update(b_rmse_m=float(np.sqrt(np.mean(b_err[: target + 1] ** 2))), b_at_target_m=float(b_err[target]),
                            b_max_m=float(b_err[: target + 1].max()), b_end_m=float(b_err[-1]),
                            a_rmse_m=float(np.sqrt(np.mean(a_err[: target + 1] ** 2))), a_at_target_m=float(a_err[target]),
                            events_true=events_true, events_pred=events_pred,
                            event_ok=bool(abs(events_true[0] - events_pred[0]) <= 1 and events_true[1:] == events_pred[1:]))
            entries.append(item)
        if save_traces is not None:
            traces.append(pred)
    finite = [e for e in entries if e["finite"]]
    result = {"episodes": len(entries), "finite_fraction": len(finite) / len(entries),
              "event_ok_fraction": sum(e.get("event_ok", False) for e in entries) / len(entries), "per_episode": entries,
              "target_step": target, "dt_s": data["dt"]}
    big = dict(mean=1e3, median=1e3, p95=1e3, max=1e3)
    for key in ("b_rmse_m", "b_at_target_m", "b_max_m", "b_end_m", "a_rmse_m", "a_at_target_m"):
        result[key] = quantiles([e[key] for e in finite]) if finite else big
    result["b_at_target_over_10mm"] = sum(e["b_at_target_m"] > 0.01 for e in finite)
    eligible = data["eligible"].cpu().numpy()
    chosen = [e for e in finite if eligible[e["index"]]]
    result["eligible_episodes"] = len(chosen)
    result["eligible_b_at_target_m"] = quantiles([e["b_at_target_m"] for e in chosen]) if chosen else big
    result["eligible_event_ok_fraction"] = (sum(e["event_ok"] for e in chosen) / len(chosen)) if chosen else 0.0
    # Model selection (validation only): B's error at t on target-worthy
    # episodes first, whole-trajectory B error second, event failures penalised.
    result["selection_score"] = (result["eligible_b_at_target_m"]["p95"] + 0.5 * result["b_rmse_m"]["p95"]
                                 + 0.05 * (1 - result["event_ok_fraction"]) + (1 - result["finite_fraction"]))
    if save_traces is not None:
        np.savez_compressed(save_traces, predicted=np.concatenate(traces), indices=indices.cpu().numpy(), dt_s=model.dt)
    return result


def launch_gradient_check(model, data, launches):
    """Autograd vs central differences of B's position at the target step w.r.t. (vx, vy)."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dtype, device = model.state_mean.dtype, model.state_mean.device
    action = torch.tensor(launches, dtype=dtype, device=device, requires_grad=True)
    steps = data["target_step"]

    def endpoint(a):
        return model.rollout(initial_state(data["scene"], a), steps)[:, -1, BALL:BALL + 2]

    y = endpoint(action)
    jacobian = torch.stack([torch.autograd.grad(y[:, k].sum(), action, retain_graph=True)[0] for k in (0, 1)], 1)
    rows = []
    with torch.no_grad():
        for epsilon in (1e-6, 1e-5, 1e-4):
            columns = []
            for k in (0, 1):
                offset = torch.zeros_like(action)
                offset[:, k] = epsilon
                columns.append((endpoint(action + offset) - endpoint(action - offset)) / (2 * epsilon))
            difference = torch.stack(columns, -1)
            error = (difference - jacobian).abs()
            rows.append(dict(epsilon=epsilon, max_abs_error=float(error.max()),
                             max_relative_error=float((error / (jacobian.abs() + 1e-6)).max())))
    return dict(launches=launches, jacobian=jacobian.detach().cpu().tolist(), finite_differences=rows,
                gate=model.config.get("gate", "hard"))
