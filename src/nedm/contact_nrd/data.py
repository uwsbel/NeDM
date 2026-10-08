"""Data loading, contact-window labels and the free-rollout metrics of the contact NRD.

A data directory holds system.json (bodies, candidate contact pairs, record and model steps, target body and
time) and unified_data.npz with ``contacts`` [N, T-1, P] (pair p in contact during record interval k),
``lengths`` [N] (valid records), ``splits`` [N] (0 train, 1 validation, 2 test) and the moving bodies' states:
either one ``states`` array [N, T, D, 9] of [position, velocity, angular velocity] per body (ball, pool), or one
array per moving body, named as the body, with the channels of its ``channel_types``, plus ``action`` [N, T, A]
for the actuated body (arm).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def load_data(root, device, model_step_s=None):
    """Load a data directory; states [N, T, D, S] and actions [N, T, D, A] are padded to the largest body (float64).
    model_step_s replaces the model step of system.json (the records stay the same)."""
    root = Path(root)
    system = json.loads((root / "system.json").read_text())
    if model_step_s:
        system = {**system, "model_step_s": float(model_step_s)}
    with np.load(root / "unified_data.npz") as packet:
        legacy = "states" in packet.files and "action" not in packet.files
        if legacy:   # 9-number bodies without actions: describe them like the other bodies
            sphere_types = ["position"] * 3 + ["velocity"] * 3 + ["angular_velocity"] * 3
            bodies = []
            for b in system["bodies"]:
                if b["moving"]:
                    b = {**b, "state_dim": 9, "action_dim": 0, "channel_types": sphere_types,
                         "channels": ["x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz"]}
                bodies.append(b)
            system = {**system, "bodies": bodies}
        moving = [b for b in system["bodies"] if b["moving"]]
        if legacy:
            st = packet["states"]
            arrays = {b["name"]: st[:, :, d] for d, b in enumerate(moving)}
        else:
            arrays = {b["name"]: packet[b["name"]] for b in moving}
        for key in ("contacts", "lengths", "splits"):
            arrays[key] = packet[key]
        arrays["action"] = packet["action"] if "action" in packet.files else None
    n = len(arrays["lengths"])
    smax = max(int(b["state_dim"]) for b in moving)
    amax = max(1, max(int(b.get("action_dim", 0)) for b in moving))
    t = arrays[moving[0]["name"]].shape[1]
    states = np.zeros((n, t, len(moving), smax), np.float64)
    actions = np.zeros((n, t, len(moving), amax), np.float64)
    for d, b in enumerate(moving):
        states[:, :, d, : int(b["state_dim"])] = arrays[b["name"]]
        if int(b.get("action_dim", 0)):
            actions[:, :, d, : int(b["action_dim"])] = arrays["action"]
    record = system["record_step_s"]
    stride = round(system["model_step_s"] / record)
    return dict(system=system, stride=stride, dt=stride * record, record_dt=record,
                target_step=round(system["target_time_s"] / (stride * record)),
                states=torch.as_tensor(states, device=device),
                actions=torch.as_tensor(actions, device=device),
                contacts=torch.as_tensor(arrays["contacts"], device=device).bool(),
                lengths=torch.as_tensor(arrays["lengths"], device=device),
                splits=torch.as_tensor(arrays["splits"], device=device))


def step_average_rates(data, kinds):
    """Replace the rate channels of bodies of the given kinds by their step-average rates over one model step:
    rate(t) = (pose(t) - pose(t - step)) / step (joint angles, positions), and for quaternions the rotation vector of
    q(t) * conj(q(t - step)) divided by the step (world frame). The first model step keeps the recorded rates (a
    rollout starts from a recorded state). With these rates the pose follows exactly from pose + step * new rate."""
    st, s, dt = data["states"], data["stride"], data["dt"]
    types = {"position": 0, "angle": 0, "quaternion": 1, "velocity": 2, "angular_rate": 2, "angular_velocity": 3}
    mov = [b for b in data["system"]["bodies"] if b["moving"]]
    out = st.clone()
    for d, b in enumerate(mov):
        if b.get("kind", "body") not in kinds:
            continue
        ct = b.get("channel_types", [])
        cols = [[c for c, t in enumerate(ct) if types.get(t) == k] for k in range(4)]
        pc, qc, vc, wc = (torch.tensor(c, dtype=torch.long, device=st.device) for c in cols)
        if len(pc) and len(pc) == len(vc):
            out[:, s:, d, vc] = (st[:, s:, d, pc] - st[:, :-s, d, pc]) / dt
        if len(qc) == 4 and len(wc) == 3:
            q1, q0 = st[:, s:, d, qc], st[:, :-s, d, qc]
            w0, v0 = q0[..., :1], -q0[..., 1:]                                   # conj(q0)
            w1, v1 = q1[..., :1], q1[..., 1:]
            w = w1 * w0 - (v1 * v0).sum(-1, keepdim=True)
            v = w1 * v0 + w0 * v1 + torch.cross(v1, v0, dim=-1)
            sign = torch.where(w < 0, -1.0, 1.0)
            w, v = w * sign, v * sign
            n = v.norm(dim=-1, keepdim=True)
            ang = 2 * torch.atan2(n, w)
            rotvec = torch.where(n > 1e-12, v / n.clamp_min(1e-12) * ang, 2 * v)
            out[:, s:, d, wc] = rotvec / dt
    data["states"] = out
    return data


def window_labels(contacts, stride):
    """[N, T-1, P] interval labels -> [N, T-stride, P]: pair in contact during the step starting at each record."""
    c = contacts.to(torch.float32)
    cum = torch.cat((torch.zeros_like(c[:, :1]), c.cumsum(1)), 1)
    return (cum[:, stride:] - cum[:, :-stride]) > 0.5


def dilate(mask, reach):
    """Dilate a [N, T, P] boolean mask along time by +-reach."""
    x = mask.permute(0, 2, 1).to(torch.float32)
    return F.max_pool1d(x, 2 * reach + 1, stride=1, padding=reach).permute(0, 2, 1) > 0.5


def atomic_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)), "p95": float(np.quantile(values, 0.95)),
            "max": float(values.max())}


def rising(x):
    """Rising edges along time of a [T, P] boolean array -> counts per pair."""
    x = np.asarray(x, dtype=bool)
    return (x[1:] & ~x[:-1]).sum(0) + x[0]


def quat_angle(q1, q2):
    dot = (q1 * q2).sum(-1).abs().clamp(max=1.0)
    return 2 * torch.arccos(dot)


def channel_index(system, body_name, ctype):
    body = next(b for b in system["bodies"] if b["name"] == body_name)
    return [c for c, t in enumerate(body["channel_types"]) if t == ctype]


@torch.no_grad()
def free_metrics(model, data, indices, batch=256, length_scale_m=0.05):
    """Rollouts from each episode's first state with its recorded commands (model grid).
    Target body (the ball, pool ball B, the T): position error and orientation error at the target time, path RMS
    error up to the target time; the actuated body's TCP path RMS error (if it has position channels); contact
    events per pair (rising edges of the predicted switches against the recorded contacts).
    Score = p95 target position error + 0.5 p95 path error + L * p95 target angle error + event penalty (L = 5 cm)."""
    if abs(model.dt - data["dt"]) > 1e-12:
        raise ValueError("model and data steps differ")
    system = data["system"]
    stride, target = data["stride"], data["target_step"]
    moving = [b for b in system["bodies"] if b["moving"]]
    tb = system["target_body"]
    tslot = model.moving.index(tb)
    tname = system["bodies"][tb]["name"]
    pos_c = channel_index(system, tname, "position")
    quat_c = channel_index(system, tname, "quaternion")
    arm = next((n for n, b in enumerate(moving) if int(b.get("action_dim", 0))), None)
    ee_c = channel_index(system, moving[arm]["name"], "position") if arm is not None else []
    labels = window_labels(data["contacts"][indices], stride)
    spread = torch.zeros_like(labels)   # step-average rates: the contact switch also covers the next window
    spread[:, stride:] = labels[:, :-stride]
    labels = (labels | spread)[:, ::stride].cpu().numpy()
    entries = []
    for start in range(0, len(indices), batch):
        chosen = indices[start:start + batch]
        truth = data["states"][chosen][:, ::stride]
        acts = data["actions"][chosen][:, ::stride]
        steps = truth.shape[1] - 1
        pred, gates = model.rollout(truth[:, 0], acts[:, :steps], return_gates=True)
        perr = (pred[:, :, tslot, pos_c] - truth[:, :, tslot, pos_c]).norm(dim=-1)                  # [B, T]
        aerr = quat_angle(pred[:, :, tslot, quat_c], truth[:, :, tslot, quat_c]) if quat_c else torch.zeros_like(perr)
        eerr = (pred[:, :, arm, ee_c] - truth[:, :, arm, ee_c]).norm(dim=-1) if ee_c else torch.zeros_like(perr)
        finite = torch.isfinite(pred).all(-1).all(-1).all(-1)
        g = gates.cpu().numpy() > 0.5
        for row, idx in enumerate(chosen.tolist()):
            n = (int(data["lengths"][idx]) - 1) // stride + 1
            k = min(target, n - 1)
            item = {"index": idx, "finite": bool(finite[row])}
            if item["finite"]:
                te, tl = rising(labels[start + row, : n - 1]), rising(g[row, : n - 1])
                item.update(target_pos_m=float(perr[row, k]), target_angle_rad=float(aerr[row, k]),
                            path_rmse_m=float(perr[row, : k + 1].square().mean().sqrt()),
                            ee_rmse_m=float(eerr[row, : k + 1].square().mean().sqrt()),
                            final_pos_m=float(perr[row, n - 1]), final_angle_rad=float(aerr[row, n - 1]),
                            event_ok=bool((te == tl).all()), events_true=te.tolist(), events_pred=tl.tolist(),
                            true_displacement_m=float((truth[row, k, tslot, pos_c] - truth[row, 0, tslot, pos_c]).norm()),
                            true_rotation_rad=float(quat_angle(truth[row, k, tslot, quat_c], truth[row, 0, tslot, quat_c])) if quat_c else 0.0)
            entries.append(item)
    finite = [e for e in entries if e["finite"]]
    big = dict(mean=1e3, median=1e3, p95=1e3, max=1e3)
    out = {"episodes": len(entries), "finite_fraction": len(finite) / len(entries),
           "event_ok_fraction": sum(e.get("event_ok", False) for e in entries) / len(entries), "per_episode": entries,
           "target_step": target, "dt_s": data["dt"]}
    for key in ("target_pos_m", "target_angle_rad", "path_rmse_m", "ee_rmse_m", "final_pos_m", "final_angle_rad"):
        out[key] = quantiles([e[key] for e in finite]) if finite else big
    moved = [e for e in finite if e["true_displacement_m"] > 0.005 or e["true_rotation_rad"] > 0.05]
    out["moved_episodes"] = len(moved)
    for key in ("target_pos_m", "target_angle_rad"):
        out["moved_" + key] = quantiles([e[key] for e in moved]) if moved else big
    out["selection_score"] = (out["target_pos_m"]["p95"] + 0.5 * out["path_rmse_m"]["p95"]
                              + length_scale_m * out["target_angle_rad"]["p95"]
                              + 0.05 * (1 - out["event_ok_fraction"]) + (1 - out["finite_fraction"]))
    return out
