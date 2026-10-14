"""Data loading, contact-window labels, the free-rollout metrics and the task metrics of the contact NRD.

A data directory holds system.json (bodies, candidate contact pairs, record and model steps, target body and
time) and unified_data.npz with ``contacts`` [N, T-1, P] (pair p in contact during record interval k),
``lengths`` [N] (valid records), ``splits`` [N] (0 train, 1 validation, 2 test) and the moving bodies' states:
either one ``states`` array [N, T, D, 9] of [position, velocity, angular velocity] per body (ball, pool), or one
array per moving body, named as the body, with the channels of its ``channel_types``, plus ``action`` [N, T, A]
for the actuated body (arm). Optional (contact_nrd/collection_contract.md): ``family`` [N] with system.json
``families`` (checkpoint selection uses the families with select: true; reports per family) and an ``evaluation``
block in system.json (task_metrics replaces free_metrics: state errors, contact events, outcome flags such as a
fall or a drop; configs/contact_nrd/evaluation/README.md; its timing_exclude names a per-record integer array, e.g.
``phase``, that load_data then reads). Data without them (the released cases) give exactly the free_metrics
results.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def load_data(root, device, model_step_s=None):
    """Load a data directory; states [N, T, D, S] and actions [N, T, D, A] are padded to the largest body (float64).
    model_step_s replaces the model step of system.json (the records stay the same). A ``family`` array and
    system.json ``families`` (each with select: true or false) come together or not at all. record_arrays: the
    per-record integer arrays [N, T] that the evaluation block names (numpy)."""
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
        families = system.get("families")
        if ("family" in packet.files) != bool(families):
            raise ValueError("unified_data.npz family and system.json families must come together")
        family = packet["family"].astype(np.int64) if families else None
        if family is not None and (family.min() < 0 or family.max() >= len(families)):
            raise ValueError("family values must index system.json families")
        if families and not all(isinstance(f.get("select"), bool) for f in families):
            raise ValueError("every system.json family needs select: true or false")
        record_arrays = read_record_arrays(system, packet)
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
                splits=torch.as_tensor(arrays["splits"], device=device),
                family=None if family is None else torch.as_tensor(family, device=device), record_arrays=record_arrays)


def read_record_arrays(system, packet):
    """The per-record integer arrays [N, T] that system.json's evaluation block names (its timing_exclude array; for
    SO101 ``phase``) from an open unified_data.npz -> {name: array}. Raises ValueError if one is missing or has the
    wrong shape or type, or if the system.json list that names its values (timing_exclude names) lacks a value."""
    exclude = (system.get("evaluation") or {}).get("timing_exclude")
    if not exclude:
        return {}
    name = exclude.get("array")
    x = packet[name] if name in packet.files else None
    if x is None or x.shape != (len(packet["lengths"]), packet["contacts"].shape[1] + 1) or x.dtype.kind not in "iu":
        raise ValueError(f"evaluation timing_exclude: unified_data.npz needs an integer array {name!r} [N, T] "
                         "(configs/contact_nrd/evaluation/README.md)")
    names = exclude.get("names")
    listed = system.get(names) if names is not None else None
    if names is not None and not (isinstance(listed, list) and set(exclude.get("values", [])) <= set(listed)):
        raise ValueError(f"evaluation timing_exclude: system.json needs a list {names!r} with {exclude.get('values')}")
    return {name: x}


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
    data["rate_kinds"] = sorted(kinds)
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
    Score = p95 target position error + 0.5 p95 path error + L * p95 target angle error + event penalty (L = 5 cm).
    A rollout counts as not finite if its states or one of its errors (an overflow of a far-diverged state) are not
    finite."""
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
                if not all(math.isfinite(v) for v in item.values() if isinstance(v, float)):
                    item = {"index": idx, "finite": False}   # a finite rollout so far out that an error overflows
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


# ---- families and task metrics (data with families and an "evaluation" block) ---------------------------------
TASK_ERRORS = {"pos_m": "position", "angle_rad": "quaternion", "joint_rms_rad": "angle", "vel_mps": "velocity"}
WINDOW_MODES = ("windows", "long_windows", "event_windows")
MODES = ("free",) + WINDOW_MODES
FAIL = 1e3   # the error of a rollout that is non-finite by the step of the statistic; the cap of every error
NONE = dict(mean=None, median=None, p95=None, max=None)   # a statistic without any rollout (null in the reports)
ZERO = dict(mean=0.0, median=0.0, p95=0.0, max=0.0)


def family_groups(data, indices):
    """{family name: the given episodes of that family} ({} if the data have no families)."""
    if data.get("family") is None:
        return {}
    fam = data["family"][indices]
    groups = {f["name"]: indices[fam == k] for k, f in enumerate(data["system"]["families"])}
    return {k: v for k, v in groups.items() if len(v)}


def selected_episodes(data, indices):
    """The given episodes of the families with select: true (checkpoint selection); all of them without families."""
    if data.get("family") is None:
        return indices
    keep = torch.tensor([f["select"] for f in data["system"]["families"]], device=indices.device)
    return indices[keep[data["family"][indices]]]


@torch.no_grad()
def evaluate(model, data, indices, by_family=False, score_only=False):
    """Validation and test metrics: task_metrics if system.json has an "evaluation" block, else free_metrics (the
    released cases, unchanged). by_family adds "families" {name: metrics of its episodes} and "selected" (the
    families with select: true; checkpoint selection compares the selection_score of these). score_only (the checks
    during training) skips the rollout modes that the score does not use (the reports)."""
    spec = data["system"].get("evaluation")
    if spec is not None:
        return task_metrics(model, data, indices, spec, by_family, score_only)
    report = free_metrics(model, data, indices)
    if by_family and data.get("family") is not None:
        def brief(r):
            return {k: v for k, v in r.items() if k != "per_episode"}
        report["families"] = {k: brief(free_metrics(model, data, v)) for k, v in family_groups(data, indices).items()}
        chosen = selected_episodes(data, indices)
        if len(chosen):
            report["selected"] = brief(free_metrics(model, data, chosen))
    return report


def model_labels(contacts, stride):
    """[N, T-1, P] interval labels -> [N, W, P] on the model grid in the model's convention (step or step before)."""
    labels = window_labels(contacts, stride)
    spread = torch.zeros_like(labels)
    spread[:, stride:] = labels[:, :-stride]
    return (labels | spread)[:, ::stride]


def physical_labels(contacts, stride):
    """[N, T-1, P] interval labels -> [N, W, P] on the model grid: the pair touched during the step (no dilation)."""
    return window_labels(contacts, stride)[:, ::stride]


def physical_switches(gates):
    """Model switches [R, H, P] -> the contact they stand for [R, H-1, P]: a switch covers its step and the step
    after a contact, so the pair touched during step k if the switches of steps k and k+1 are on."""
    return gates[:, :-1] & gates[:, 1:]


def match_events(true, pred, tol, steps=False):
    """Order-keeping matching of two sorted step lists within tol steps: the most matches, then the smallest sum of
    |pred - true| -> (pred - true of the matches, missed, false); steps: also the true steps of the matches."""
    true, pred = [int(x) for x in true], [int(x) for x in pred]
    n, m = len(true), len(pred)
    if not n or not m:
        return ([], n, m, []) if steps else ([], n, m)
    best = [[(0, 0)] * (m + 1) for _ in range(n + 1)]          # (matches, -cost) of true[i:] and pred[j:]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            b = max(best[i + 1][j], best[i][j + 1])
            d = abs(pred[j] - true[i])
            if d <= tol:
                c, k = best[i + 1][j + 1]
                b = max(b, (c + 1, k - d))
            best[i][j] = b
    i = j = 0
    dts, at = [], []
    while i < n and j < m:
        d = pred[j] - true[i]
        c, k = best[i + 1][j + 1]
        if abs(d) <= tol and best[i][j] == (c + 1, k - abs(d)):
            dts.append(d)
            at.append(true[i])
            i, j = i + 1, j + 1
        elif best[i][j] == best[i + 1][j]:
            i += 1
        else:
            j += 1
    return (dts, n - len(dts), m - len(dts), at) if steps else (dts, n - len(dts), m - len(dts))


def contact_tally(gates, labels, steps, tol, row0=0):
    """Contact events of R rollouts: gates, labels [R, H, P] bool, steps [R] valid steps. Onsets and offsets are
    switch changes between valid steps (not at the first step). -> counts [R, P, 2 (onset, offset), 4 (true,
    predicted, missed, false)], acc [R, P] correct steps, den [R] valid steps, timing [M, 5] (row0 + row, pair,
    kind, predicted - true step, true step) of the matched events."""
    r, h, p = gates.shape
    ok = np.arange(h)[None, :] < np.asarray(steps)[:, None]
    acc = ((gates == labels) & ok[..., None]).sum(1)
    move = ok[:, 1:, None]
    counts = np.zeros((r, p, 2, 4), np.int64)
    timing = []
    for k, (tl, tp) in enumerate((((labels[:, 1:] & ~labels[:, :-1]) & move, (gates[:, 1:] & ~gates[:, :-1]) & move),
                                  ((~labels[:, 1:] & labels[:, :-1]) & move, (~gates[:, 1:] & gates[:, :-1]) & move))):
        counts[:, :, k, 0], counts[:, :, k, 1] = tl.sum(1), tp.sum(1)
        for row, q in zip(*np.nonzero(tl.any(1) | tp.any(1))):
            dts, missed, false, at = match_events(np.flatnonzero(tl[row, :, q]), np.flatnonzero(tp[row, :, q]), tol, True)
            counts[row, q, k, 2:] = missed, false
            timing += [(row0 + row, q, k, d, a + 1) for d, a in zip(dts, at)]
    return dict(counts=counts, acc=acc, den=ok.sum(1), timing=np.asarray(timing, np.int64).reshape(-1, 5))


def _merge(tallies, pairs):
    if not tallies:
        return dict(counts=np.zeros((0, pairs, 2, 4), np.int64), acc=np.zeros((0, pairs), np.int64),
                    den=np.zeros(0, np.int64), timing=np.zeros((0, 5), np.int64))
    return {k: np.concatenate([t[k] for t in tallies]) for k in tallies[0]}


def exclude_timing(raw, mode, values, pairs, stride):
    """Move the matched events of the rollouts of one mode whose recorded step lies at a record with an excluded value
    (values [E, T] bool over the evaluated episodes) of one of the pairs from "timing" to "left_out" (both views)."""
    rows = np.arange(values.shape[0]) if mode == "free" else raw["row"]
    start = np.zeros(len(rows), np.int64) if mode == "free" else raw["start"]
    for key in ("contacts", "contacts_physical"):
        tim = raw[key]["timing"]
        rec = np.minimum((start[tim[:, 0]] + tim[:, 4]) * stride, values.shape[1] - 1)
        out = values[rows[tim[:, 0]], rec] & np.isin(tim[:, 1], pairs)
        raw[key]["timing"], raw[key]["left_out"] = tim[~out], tim[out]


def _q(values, empty=NONE):
    values = np.asarray(values, dtype=float)
    return quantiles(values) if len(values) else dict(empty)


def rotation_angle(q1, q2):
    """Angle of the rotation between two quaternions [..., 4], 2 atan2(|v|, |w|) of conj(q1) q2: exact 0 for equal
    quaternions and independent of their norms (quat_angle, 2 arccos |q1.q2|, reads up to 5e-4 rad for equal float32
    records; free_metrics keeps it)."""
    w1, v1 = q1[..., :1], q1[..., 1:]
    w2, v2 = q2[..., :1], q2[..., 1:]
    w = (w1 * w2 + (v1 * v2).sum(-1, keepdim=True)).squeeze(-1)
    v = w1 * v2 - w2 * v1 - torch.cross(v1, v2, dim=-1)
    return 2 * torch.atan2(v.norm(dim=-1), w.abs())


def body_errors(pred, truth, cols):
    """pred, truth [..., S] of one body -> {error name: [...]} for the channel groups in cols."""
    out = {}
    for name, c in cols.items():
        if name == "angle_rad":
            out[name] = rotation_angle(pred[..., c], truth[..., c])
        elif name == "joint_rms_rad":
            out[name] = (pred[..., c] - truth[..., c]).square().mean(-1).sqrt()
        else:
            out[name] = (pred[..., c] - truth[..., c]).norm(dim=-1)
    return out


def _steps(seconds, dt):
    k = round(seconds / dt)
    if abs(k * dt - seconds) > 1e-9:
        raise ValueError(f"{seconds} s is not a multiple of the model step {dt} s")
    return k


def compile_flags(spec, system, pair_names):
    """The "flags" block -> {name: (conditions that must all hold, conditions of which one must hold)}. A condition
    is (kind, body slot or pair index, channel index or quaternion columns, value): a state channel "above" or
    "below" a value, the body's "tilt" (angle between its z axis and the world z axis, from its quaternion) above a
    value, or a contact "pair" switch equal to on (true by default)."""
    moving = [b for b in system["bodies"] if b["moving"]]
    names = [b["name"] for b in moving]
    out = {}
    for name, flag in (spec or {}).items():
        groups = []
        for key in ("all", "any"):
            conds = []
            for c in flag.get(key, []):
                try:
                    if "pair" in c:
                        conds.append(("pair", pair_names.index(c["pair"]), None, bool(c.get("on", True))))
                        continue
                    d = names.index(c["body"])
                    if "tilt_above_rad" in c:
                        qc = [k for k, t in enumerate(moving[d]["channel_types"]) if t == "quaternion"]
                        if len(qc) != 4:
                            raise ValueError
                        conds.append(("tilt", d, qc, float(c["tilt_above_rad"])))
                    else:
                        kind = "above" if "above" in c else "below"
                        conds.append((kind, d, moving[d]["channels"].index(c["channel"]), float(c[kind])))
                except (KeyError, ValueError):
                    raise ValueError(f"evaluation flag {name!r}: bad condition {c}") from None
            groups.append(conds)
        if not any(groups):
            raise ValueError(f"evaluation flag {name!r} has no condition")
        out[name] = tuple(groups)
    return out


def _condition(c, x, switches):
    kind, d, col, value = c
    if kind == "pair":
        return switches[..., d] == value
    if kind == "tilt":
        q = x[..., d, col]
        return 1 - 2 * (q[..., 1].square() + q[..., 2].square()) / q.square().sum(-1) < math.cos(value)
    return x[..., d, col] > value if kind == "above" else x[..., d, col] < value


def flag_trace(flag, x, switches):
    """Steps [R, H] at which a compiled flag holds: x [R, H, D, S] the states after each step, switches [R, H, P]
    bool the contact switches of those steps."""
    every, anyof = flag
    hit = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    for c in every:
        hit &= _condition(c, x, switches)
    if anyof:
        hit &= torch.stack([_condition(c, x, switches) for c in anyof]).any(0)
    return hit


def _flags_raw(flags, pred, truth, gates, labels, steps):
    """Outcome flags of R rollouts (states [R, H+1, D, S]; gates, labels [R, H, P] bool arrays; steps [R] valid
    steps): the recorded flag from the recorded states and labels, the predicted flag from the rollout and its
    switches -> {name: {rec, pred: [R] bool (the flag holds at a valid step), rec_first, pred_first: [R] first
    such step (-1 none)}}."""
    dev = pred.device
    on = torch.arange(pred.shape[1] - 1, device=dev)[None] < torch.as_tensor(steps, device=dev)[:, None]
    out = {}
    for name, flag in flags.items():
        item = {}
        for tag, x, sw in (("rec", truth, labels), ("pred", pred, gates)):
            hit = flag_trace(flag, x[:, 1:], torch.as_tensor(sw, device=dev)) & on
            item[tag] = hit.any(1).cpu().numpy()
            item[tag + "_first"] = torch.where(hit.any(1), hit.int().argmax(1) + 1, -1).cpu().numpy()
        out[name] = item
    return out


def _merge_flags(parts, names):
    if not parts:
        return {k: dict(rec=np.zeros(0, bool), pred=np.zeros(0, bool), rec_first=np.zeros(0, np.int64),
                        pred_first=np.zeros(0, np.int64)) for k in names}
    return {k: {kk: np.concatenate([p[k][kk] for p in parts]) for kk in parts[0][k]} for k in names}


def _free_raw(model, data, indices, cols, labels, phys, flags, batch, tol):
    """First-state rollouts of the episodes (recorded commands): errors at the target and the last valid step and
    whether the rollout is finite up to each of them (finite_at; finite: up to the last valid step), contact events
    (model convention and physical), outcome flags."""
    stride, target = data["stride"], data["target_step"]
    system = data["system"]
    tslot = model.moving.index(system["target_body"])
    tpos = channel_index(system, system["bodies"][system["target_body"]]["name"], "position")
    n_all = ((data["lengths"][indices] - 1) // stride + 1).cpu().numpy()
    e = len(indices)
    out = dict(finite=np.zeros(e, bool), path=np.full(e, np.nan) if tpos else None,
               err={at: [{m: np.full(e, np.nan) for m in c} for c in cols] for at in ("target", "final")})
    out["finite_at"] = {"target": np.zeros(e, bool), "final": out["finite"]}
    tallies, ptallies, fparts = [], [], []
    for start in range(0, e, batch):
        chosen = indices[start:start + batch]
        sl = slice(start, start + len(chosen))
        truth = data["states"][chosen][:, ::stride]
        acts = data["actions"][chosen][:, ::stride]
        steps = truth.shape[1] - 1
        pred, gates = model.rollout(truth[:, 0], acts[:, :steps], return_gates=True)
        n = torch.as_tensor(n_all[sl], device=pred.device)
        k = (n - 1).clamp_max(target)
        rr = torch.arange(len(chosen), device=pred.device)
        t = torch.arange(steps + 1, device=pred.device)
        upto = (~torch.isfinite(pred).all(-1).all(-1)).cumsum(1) == 0          # [B, T] finite up to each step
        out["finite"][sl] = upto[rr, n - 1].cpu().numpy()
        out["finite_at"]["target"][sl] = upto[rr, k].cpu().numpy()
        for at, kk in (("target", k), ("final", n - 1)):
            for d, c in enumerate(cols):
                for m, v in body_errors(pred[rr, kk, d], truth[rr, kk, d], c).items():
                    out["err"][at][d][m][sl] = v.cpu().numpy()
        if tpos:
            perr = (pred[:, :, tslot, tpos] - truth[:, :, tslot, tpos]).norm(dim=-1)
            on = t[None] <= k[:, None]
            out["path"][sl] = (torch.where(on, perr.square(), 0.0).sum(1) / on.sum(1)).sqrt().cpu().numpy()
        g, valid = gates.cpu().numpy() > 0.5, n_all[sl] - 1
        tallies.append(contact_tally(g, labels[sl, :steps], valid, tol, start))
        ptallies.append(contact_tally(physical_switches(g), phys[sl, :steps - 1], valid - 1, tol, start))
        fparts.append(_flags_raw(flags, pred, truth, g, labels[sl, :steps], valid))
    out["contacts"] = _merge(tallies, labels.shape[-1])
    out["contacts_physical"] = _merge(ptallies, labels.shape[-1])
    out["flags"] = _merge_flags(fparts, flags)
    return out


def window_starts(n_all, spec, dt):
    """Start steps of the windows of each episode (n_all valid steps): first_start_s, + start_every_s, ... while the
    shortest horizon fits; with end_window also the start of the window of the longest horizon that ends at the
    last valid step (the episode's end, e.g. just before a fall)."""
    hsteps = [_steps(float(h), dt) for h in spec["horizons_s"]]
    first, every = _steps(float(spec.get("first_start_s", 0.0)), dt), _steps(float(spec["start_every_s"]), dt)
    out = []
    for n in n_all:
        s = np.arange(first, n - min(hsteps), every)
        if spec.get("end_window") and n > 1:
            s = np.union1d(s, [max(0, n - 1 - max(hsteps))])
        out.append(s.astype(np.int64))
    return out


def event_starts(phys, n_all, spec, dt):
    """Start steps of the event windows: lead_s before each recorded contact onset and/or offset (spec "events") of
    any pair (physical labels [E, W, P]; step 0 for an event less than lead_s after the episode start), at least
    min_gap_s apart, while the shortest horizon fits."""
    lead, gap = _steps(float(spec.get("lead_s", 0.0)), dt), _steps(float(spec.get("min_gap_s", 0.0)), dt)
    hmin = min(_steps(float(h), dt) for h in spec["horizons_s"])
    kinds = spec.get("events", ["onset", "offset"])
    out = []
    for r, n in enumerate(n_all):
        x = phys[r, : max(0, n - 1)]
        hit = np.zeros(len(x), bool)
        if "onset" in kinds:
            hit[1:] |= (x[1:] & ~x[:-1]).any(-1)
        if "offset" in kinds:
            hit[1:] |= (~x[1:] & x[:-1]).any(-1)
        keep, last = [], None
        for s in np.maximum(np.flatnonzero(hit) - lead, 0):
            if s < n - hmin and (last is None or s - last >= max(1, gap)):
                keep.append(s)
                last = s
        out.append(np.asarray(keep, np.int64))
    return out


def _window_raw(model, data, indices, cols, labels, phys, flags, starts, horizons, batch, tol):
    """Rollouts from the recorded K-state history at the given start steps of each episode (recorded commands) ->
    errors at each horizon (NaN where the horizon passes the episode end) and whether the rollout is finite up to it
    (finite_at; finite: up to the last step that fits in the episode), contact events (model convention and
    physical), the recorded contacts during the first step (code: bit p = pair p) and outcome flags."""
    dt, stride = data["dt"], data["stride"]
    dev = data["states"].device
    keys = [str(float(h)) for h in horizons]
    hsteps = [_steps(float(h), dt) for h in horizons]
    hmax = max(hsteps)
    n_all = ((data["lengths"][indices] - 1) // stride + 1).cpu().numpy()
    rows = np.concatenate([np.full(len(s), r) for r, s in enumerate(starts)] + [np.zeros(0)]).astype(np.int64)
    starts = np.concatenate(list(starts) + [np.zeros(0)]).astype(np.int64)
    fit = np.minimum(hmax, n_all[rows] - 1 - starts)
    w = len(rows)
    out = dict(row=rows, start=starts, fit=fit, hsteps=dict(zip(keys, hsteps)), finite=np.zeros(w, bool),
               code=(phys[rows, starts].astype(np.int64) << np.arange(phys.shape[-1])).sum(-1),
               err={h: [{m: np.full(w, np.nan) for m in c} for c in cols] for h in keys},
               finite_at={h: np.zeros(w, bool) for h in keys})
    t_rec = data["states"].shape[1]
    lag = torch.arange(1 - model.k, 1, device=dev)
    ahead = torch.arange(hmax + 1, device=dev)
    tallies, ptallies, fparts = [], [], []
    for b0 in range(0, w, batch):
        sl = slice(b0, min(w, b0 + batch))
        ep = indices[torch.as_tensor(rows[sl], device=dev)]
        s = torch.as_tensor(starts[sl], device=dev)
        rec = (s[:, None] + lag) * stride
        ok = rec >= 0
        hist = data["states"][ep[:, None], rec.clamp_min(0)] * ok[..., None, None]
        grid = ((s[:, None] + ahead) * stride).clamp_max(t_rec - 1)
        truth = data["states"][ep[:, None], grid]
        pred, gates = model.rollout_from(hist, ok, data["actions"][ep[:, None], grid[:, :-1]], return_gates=True)
        f = torch.as_tensor(fit[sl], device=dev)
        upto = (~torch.isfinite(pred).all(-1).all(-1)).cumsum(1) == 0          # [B, H+1] finite up to each step
        out["finite"][sl] = upto[torch.arange(len(f), device=dev), f].cpu().numpy()
        for h, hs in zip(keys, hsteps):
            out["finite_at"][h][sl] = upto[:, hs].cpu().numpy()
            for d, c in enumerate(cols):
                for m, v in body_errors(pred[:, hs, d], truth[:, hs, d], c).items():
                    out["err"][h][d][m][sl] = torch.where(f >= hs, v, torch.nan).cpu().numpy()
        at = rows[sl, None], np.minimum(starts[sl, None] + np.arange(hmax), labels.shape[1] - 1)
        g = gates.cpu().numpy() > 0.5
        tallies.append(contact_tally(g, labels[at], fit[sl], tol, b0))
        ptallies.append(contact_tally(physical_switches(g), phys[at][:, :-1], fit[sl] - 1, tol, b0))
        fparts.append(_flags_raw(flags, pred, truth, g, labels[at], fit[sl]))
    out["contacts"] = _merge(tallies, labels.shape[-1])
    out["contacts_physical"] = _merge(ptallies, labels.shape[-1])
    out["flags"] = _merge_flags(fparts, flags)
    return out


def contact_summary(tally, rows, pair_names, dt):
    """Event statistics of the rollouts `rows` (bool) per pair and over all pairs ("all"). Timing in seconds
    (absolute, matched events only; 0 if none matched); timing_left_out: matched events that the timing leaves out
    (exclude_timing); event_error_rate = (missed + false) / true events. Without any rollout the rates, timings and
    accuracies are None."""
    counts, tim = tally["counts"][rows], tally["timing"]
    tim = tim[rows[tim[:, 0]]] if len(tim) else tim
    out_ = tally.get("left_out", np.zeros((0, 5), np.int64))
    out_ = out_[rows[out_[:, 0]]] if len(out_) else out_
    den = int(tally["den"][rows].sum())
    some = bool(rows.any())
    out = {}
    for name, sel in [(n, [p]) for p, n in enumerate(pair_names)] + [("all", list(range(len(pair_names))))]:
        entry = {}
        in_pair = np.isin(tim[:, 1], sel)
        for k, kind in enumerate(("onset", "offset")):
            true, pred, missed, false = (int(x) for x in counts[:, sel, k].sum((0, 1)))
            dts = tim[in_pair & (tim[:, 2] == k), 3]
            entry[kind] = dict(true=true, predicted=pred, matched=true - missed, missed=missed, false=false,
                               timing_s=_q(np.abs(dts) * dt, ZERO if some else NONE),
                               bias_s=(float(dts.mean() * dt) if len(dts) else 0.0) if some else None,
                               timing_left_out=int((np.isin(out_[:, 1], sel) & (out_[:, 2] == k)).sum()))
        c = counts[:, sel].sum((0, 1, 2))
        entry["event_error_rate"] = float((c[2] + c[3]) / max(1, c[0])) if some else None
        entry["timing_s"] = _q(np.abs(tim[in_pair, 3]) * dt, ZERO if some else NONE)
        entry["gate_accuracy"] = float(tally["acc"][rows][:, sel].sum() / max(1, den * len(sel))) if some else None
        out[name] = entry
    return out


def flag_summary(flags, rows, finite, dt):
    """Outcome flags of the rollouts `rows` (bool; finite: finite up to the last valid step): rollouts, non-finite
    rollouts, recorded flags (all rollouts); predicted, matched, missed and false flags (finite rollouts);
    disagreement = (missed + false + nonfinite) / rollouts (a non-finite rollout counts as wrong); first_s = timing
    error of the first flagged step of the rollouts flagged in both (0 if none). Without any rollout disagreement
    and first_s are None."""
    good, n = rows & finite, int(rows.sum())
    out = {}
    for name, f in flags.items():
        rec, pred = f["rec"][good], f["pred"][good]
        both = rec & pred
        missed, false, nonfinite = int((rec & ~pred).sum()), int((~rec & pred).sum()), int((rows & ~finite).sum())
        out[name] = dict(rollouts=n, nonfinite=nonfinite, recorded=int(f["rec"][rows].sum()), predicted=int(pred.sum()),
                         matched=int(both.sum()), missed=missed, false=false,
                         disagreement=float((missed + false + nonfinite) / n) if n else None,
                         first_s=_q(np.abs(f["pred_first"][good][both] - f["rec_first"][good][both]) * dt, ZERO if n else NONE))
    return out


def task_score(summary, spec, strict=True, group="all"):
    """Selection score: sum over spec["score"] of weight x statistic / scale; a statistic is a "/" path in the
    summary (e.g. "windows/1.0/robot/pos_m/p95"). A statistic without any rollout (None, e.g. a horizon longer than
    every episode of the group) raises with strict (the groups that checkpoint selection uses); otherwise the score
    and that term are None (not available)."""
    terms, total, empty = [], 0.0, []
    for term in spec["score"]:
        v = summary
        for key in term["stat"].split("/"):
            if not isinstance(v, dict) or key not in v:
                raise ValueError(f"evaluation score: no statistic {term['stat']!r}")
            v = v[key]
        weight, scale = float(term.get("weight", 1.0)), float(term["scale"])
        if v is None:
            empty.append(term["stat"])
            terms.append(dict(stat=term["stat"], value=None, scale=scale, weight=weight, term=None))
            continue
        terms.append(dict(stat=term["stat"], value=float(v), scale=scale, weight=weight, term=weight * float(v) / scale))
        total += terms[-1]["term"]
    if empty and strict:
        raise ValueError(f"evaluation score of {group}: no rollouts for {', '.join(empty)} (checkpoint selection needs "
                         "every scored statistic; are the episodes shorter than the horizon?)")
    return (None if empty else total), terms


def _capped(v, good):
    """Errors as the statistics count them: FAIL for a rollout that is not finite up to the step (good False) or whose
    error is not finite (an overflow of a far-diverged state), at most FAIL otherwise (a finite rollout that diverges
    does not count worse than a non-finite one)."""
    return np.where(good & np.isfinite(v), np.minimum(v, FAIL), FAIL)


def _errors_at(err, use, good, names):
    """{body: {error: statistics}} of the rollouts `use` (bool); errors capped at FAIL (_capped)."""
    return {names[d]: {m: _q(_capped(v, good)[use]) for m, v in e.items()} for d, e in enumerate(err)}


def _horizon_errors(wr, chosen, names):
    """{horizon: {"windows": count, "nonfinite": count, body: {error: statistics}}} of the windows `chosen` (bool)
    that fit the horizon; nonfinite of them are not finite up to the horizon and count as errors of FAIL (_capped)."""
    out = {}
    for h, per_body in wr["err"].items():
        use, good = chosen & (wr["fit"] >= wr["hsteps"][h]), wr["finite_at"][h]
        out[h] = {"windows": int(use.sum()), "nonfinite": int((use & ~good).sum()), **_errors_at(per_body, use, good, names)}
    return out


def _summary(raw, sel, names, pair_names, dt, spec, counted, strict=True, group="all"):
    """Statistics and score of the episodes `sel` (bool over the evaluated episodes) -> (summary, non-finite flags).
    Only the rollout modes in `counted` (those the score uses) enter the episode's non-finite flag; the others are
    reports. strict: see task_score."""
    out = {"episodes": int(sel.sum())}
    bad = np.zeros(len(sel), bool)
    if "free" in raw:
        fr = raw["free"]
        if "free" in counted:
            bad |= ~fr["finite"]
        free = {"nonfinite_fraction": float((~fr["finite"][sel]).mean()) if sel.any() else None}
        for at in ("target", "final"):
            good = fr["finite_at"][at]
            free[at] = {"nonfinite": int((sel & ~good).sum()), **_errors_at(fr["err"][at], sel, good, names)}
        if fr["path"] is not None:
            free["path_rmse_m"] = _q(_capped(fr["path"], fr["finite_at"]["target"])[sel])
        free["contacts"] = contact_summary(fr["contacts"], sel, pair_names, dt)
        free["contacts_physical"] = contact_summary(fr["contacts_physical"], sel, pair_names, dt)
        if fr["flags"]:
            free["flags"] = flag_summary(fr["flags"], sel, fr["finite"], dt)
        out["free"] = free
    for mode in WINDOW_MODES:
        if mode not in raw:
            continue
        wr = raw[mode]
        wsel = sel[wr["row"]]
        if mode in counted:
            bad[wr["row"][~wr["finite"]]] = True
        win = {"windows": int(wsel.sum()),
               "nonfinite_fraction": float((~wr["finite"][wsel]).mean()) if wsel.any() else None,
               **_horizon_errors(wr, wsel, names)}
        win["contacts"] = contact_summary(wr["contacts"], wsel, pair_names, dt)
        win["contacts_physical"] = contact_summary(wr["contacts_physical"], wsel, pair_names, dt)
        if wr["flags"]:
            win["flags"] = flag_summary(wr["flags"], wsel, wr["finite"], dt)
        win["by_start_contact"] = {
            "+".join(p for k, p in enumerate(pair_names) if code >> k & 1) or "none":
                _horizon_errors(wr, wsel & (wr["code"] == code), names) for code in np.unique(wr["code"][wsel])}
        out[mode] = win
    out["nonfinite_fraction"] = float(bad[sel].mean()) if sel.any() else None
    out["finite_fraction"] = 1.0 - out["nonfinite_fraction"] if sel.any() else None
    out["selection_score"], out["score_terms"] = task_score(out, spec, strict, group)
    return out, bad


def _per_episode(raw, indices, bad, names, families, famrow):
    """Per-episode records: scored_finite (every rollout of the episode in the modes that the score uses is finite),
    nonfinite_modes (all modes, reports too, with a non-finite rollout of the episode); errors (raw, None for a
    rollout that is not finite up to the step or an error that is not finite); windows: the largest error over the
    episode's windows (None if one of them is not finite up to the horizon) and the number of window contact-event
    errors and flag disagreements (a non-finite window counts as one of each flag)."""
    def num(x):
        return float(x) if np.isfinite(x) else None
    idx = indices.cpu().numpy()
    broken = {m: ~r["finite"] if m == "free" else np.isin(np.arange(len(idx)), r["row"][~r["finite"]]) for m, r in raw.items()}
    win = {}
    if "windows" in raw:
        wr = raw["windows"]
        for h, per_body in wr["err"].items():
            fits, good = wr["fit"] >= wr["hsteps"][h], wr["finite_at"][h]
            failed = np.zeros(len(idx), bool)
            failed[wr["row"][fits & ~good]] = True
            for d, e in enumerate(per_body):
                for m, v in e.items():
                    worst = np.full(len(idx), np.nan)
                    np.fmax.at(worst, wr["row"][fits & good], v[fits & good])
                    win[(h, d, m)] = np.where(failed, np.nan, worst)
        wmiss = np.zeros(len(idx), np.int64)
        np.add.at(wmiss, wr["row"], wr["contacts"]["counts"][..., 2:].sum((1, 2, 3)))
        wflag = np.zeros(len(idx), np.int64)
        for f in wr["flags"].values():
            np.add.at(wflag, wr["row"], (f["rec"] != f["pred"]) | ~wr["finite"])
    out = []
    for r, i in enumerate(idx):
        item = {"index": int(i), "scored_finite": not bool(bad[r]), "nonfinite_modes": [m for m in raw if broken[m][r]]}
        if famrow is not None:
            item["family"] = families[famrow[r]]["name"]
        if "free" in raw:
            fr = raw["free"]
            item["free"] = {at: {names[d]: {m: num(v[r]) if fr["finite_at"][at][r] else None for m, v in e.items()}
                                 for d, e in enumerate(fr["err"][at])} for at in ("target", "final")}
            if fr["path"] is not None:
                item["free"]["path_rmse_m"] = num(fr["path"][r]) if fr["finite_at"]["target"][r] else None
            item["free"]["event_errors"] = int(fr["contacts"]["counts"][r, :, :, 2:].sum())
            if fr["flags"]:
                item["free"]["flags"] = {k: dict(recorded=bool(f["rec"][r]),
                                                 predicted=bool(f["pred"][r]) if fr["finite"][r] else None)
                                         for k, f in fr["flags"].items()}
        if "windows" in raw:
            item["windows_worst"] = {}
            for (h, d, m), v in win.items():
                item["windows_worst"].setdefault(h, {}).setdefault(names[d], {})[m] = num(v[r])
            item["windows_event_errors"] = int(wmiss[r])
            if raw["windows"]["flags"]:
                item["windows_flag_errors"] = int(wflag[r])
        out.append(item)
    return out


@torch.no_grad()
def task_metrics(model, data, indices, spec, by_family=False, score_only=False):
    """Task metrics of an "evaluation" block (configs/contact_nrd/evaluation/README.md); data with step-average rates.

    free: one rollout per episode from its first state with its recorded commands (as free_metrics); errors at the
    target time ("target") and at the last valid record ("final"); the target body's path RMS up to the target time.
    windows: rollouts from the recorded K-state history at first_start_s, + start_every_s, ... (and, with end_window,
    one that ends at the episode's last valid step) with the recorded commands; errors at each horizon. long_windows:
    the same with their own starts and (long) horizons. event_windows: the same from lead_s before each recorded
    contact onset and/or offset. A mode whose statistics the score does not use is a report: skipped with
    score_only, and its non-finite rollouts do not enter nonfinite_fraction. Errors of each moving body: pos_m (norm
    of the position channels), angle_rad (quaternion angle), joint_rms_rad (RMS of the angle channels), vel_mps (norm
    of the velocity channels); by_start_contact splits the window errors by the recorded contacts during the first
    step. Contacts per pair: onsets and offsets of the switches against the recorded labels in the model's
    convention (step or step before; "contacts") and of the contact the switches stand for against the undilated
    labels ("contacts_physical"), matched within contact_tolerance_steps; missed, false, timing, per-step switch
    accuracy; timing_exclude leaves the matched events at records with given values of a per-record array (e.g. the
    controller phase) out of the timing (exclude_timing). flags: named outcome conditions (fall, lift, drop; see
    compile_flags) evaluated on the recorded and on the predicted rollout; their agreement.
    Error statistics (mean, median, p95, max) over every rollout that reaches the step (horizon, target, last valid
    record); a rollout that is not finite up to that step counts as an error of FAIL (1000), and "nonfinite" next to
    each statistic counts them; every other error counts as at most FAIL (a non-finite error, too); a statistic
    without any rollout is None. Flags: a non-finite rollout counts as a disagreement. nonfinite_fraction = episodes
    with a non-finite rollout in a mode that the score uses (per episode: scored_finite; nonfinite_modes lists all
    modes, reports too).
    selection_score = task_score: None for a family with select: false that has no rollout for a scored statistic;
    an error for all episodes, the selected families and each family with select: true. by_family adds "families"
    and "selected" (see evaluate)."""
    if abs(model.dt - data["dt"]) > 1e-12:
        raise ValueError("model and data steps differ")
    if "rate_kinds" not in data:
        raise ValueError("task metrics compare step-average rates: call step_average_rates first")
    system = data["system"]
    moving = [b for b in system["bodies"] if b["moving"]]
    names = [b["name"] for b in moving]
    cols = [{m: c for m, c in ((m, [k for k, t in enumerate(b["channel_types"]) if t == typ]) for m, typ in TASK_ERRORS.items())
             if c} for b in moving]
    meta = system.get("pair_metadata") or [{} for _ in system["pairs"]]
    pair_names = [m.get("name", f"pair{p}") for p, m in enumerate(meta)]
    pair_names = [n if pair_names.count(n) == 1 else f"{n}_{p}" for p, n in enumerate(pair_names)]
    tol, batch = int(spec.get("contact_tolerance_steps", 1)), int(spec.get("batch", 1024))
    flags = compile_flags(spec.get("flags"), system, pair_names)
    stride, dt = data["stride"], data["dt"]
    labels = model_labels(data["contacts"][indices], stride).cpu().numpy()
    phys = physical_labels(data["contacts"][indices], stride).cpu().numpy()
    n_all = ((data["lengths"][indices] - 1) // stride + 1).cpu().numpy()
    used = {t["stat"].split("/")[0] for t in spec["score"]}
    counted = [m for m in MODES if m in spec and m in used]   # the scored modes; the others are reports
    if not counted:
        raise ValueError(f"the evaluation score must use the statistics of one of {', '.join(m for m in MODES if m in spec)}")
    exclude = spec.get("timing_exclude")
    if exclude:
        try:
            values = exclude["values"]
            if "names" in exclude:
                values = [system[exclude["names"]].index(v) for v in values]
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
                raise ValueError
            out_pairs = [pair_names.index(q) for q in exclude.get("pairs", pair_names)]
            out_steps = np.isin(data["record_arrays"][exclude["array"]][indices.cpu().numpy()], values)
        except (KeyError, ValueError):
            raise ValueError(f"evaluation timing_exclude {exclude}: unknown array, value or pair (values are integers, "
                             "or names of the system.json list given by names)") from None
    raw = {}
    for mode in MODES:
        if mode not in spec or (score_only and mode not in counted):
            continue
        if mode == "free":
            raw["free"] = _free_raw(model, data, indices, cols, labels, phys, flags, batch, tol)
        else:
            w = spec[mode]
            starts = event_starts(phys, n_all, w, dt) if mode == "event_windows" else window_starts(n_all, w, dt)
            raw[mode] = _window_raw(model, data, indices, cols, labels, phys, flags, starts, w["horizons_s"], batch, tol)
        if exclude:
            exclude_timing(raw[mode], mode, out_steps, out_pairs, stride)
    everyone = np.ones(len(indices), bool)
    report, bad = _summary(raw, everyone, names, pair_names, dt, spec, counted)
    families = system.get("families")
    famrow = data["family"][indices].cpu().numpy() if data.get("family") is not None else None
    if by_family and famrow is not None:
        report["families"] = {f["name"]: _summary(raw, famrow == k, names, pair_names, dt, spec, counted, f["select"],
                                                  f"family {f['name']}")[0]
                              for k, f in enumerate(families) if (famrow == k).any()}
        keep = np.array([f["select"] for f in families])[famrow]
        if keep.any():
            report["selected"] = _summary(raw, keep, names, pair_names, dt, spec, counted, True, "the selected families")[0]
    report.update(target_step=data["target_step"], dt_s=dt, evaluation=spec,
                  per_episode=_per_episode(raw, indices, bad, names, families, famrow))
    return report
