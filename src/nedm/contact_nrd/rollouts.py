"""Simulator-versus-NRD rollouts of chosen episodes: the input of the videos and contact timelines
(``nedm.contact_nrd.collect.render``). No simulator.

    PYTHONPATH=src python -m nedm.contact_nrd.rollouts --data DIR --model CKPT --out OUT.npz \\
        (--episodes ROW ... | --pick N [--families NAME ...]) [--mode first|window] [--start-s S] [--horizon-s H]

The rollouts are those of the evaluation (data.task_metrics, configs/contact_nrd/evaluation/README.md), on the model
grid (20 ms) with the recorded commands:

| Mode | Start | Evaluation mode |
|---|---|---|
| ``first`` | The episode's first recorded state; the whole episode, or the first H seconds | ``free`` |
| ``window`` | The recorded K-state history that ends at S seconds (entries before the episode start are masked); H seconds | ``windows`` |

The states are those that the evaluation compares: the rate channels of the integrated body kinds are 20 ms step
averages (data.step_average_rates). ROW is a row of unified_data.npz. Only the chosen rows are read (the file is
memory-mapped), so the 19 GB G1 file is fine. --pick takes N validation episodes (split 1) of each family (never a
core-only family), evenly spread over the family's rows, among the episodes long enough for the whole rollout.

Output (np.savez_compressed). E episodes, H steps, D moving bodies, S padded channels, P pairs. A rollout that reaches
the episode end early has ``steps`` < H; after it the recorded states repeat the last valid state.

| Key | Shape | Meaning |
|---|---|---|
| ``system`` | str | System name of system.json |
| ``meta`` | str | JSON: data, checkpoint and its SHA-256, mode, start, horizon, steps, bodies and channels, pair names |
| ``episode``, ``episode_id`` | [E] int64 | Row in unified_data.npz; the data's episode id |
| ``family``, ``family_name``, ``split`` | [E] | Family index and name; split (0 train, 1 validation, 2 test) |
| ``time_s`` | [H+1] | Time of each model-grid state from the episode start |
| ``steps`` | [E] int64 | Valid steps of each rollout |
| ``recorded``, ``predicted`` | [E, H+1, D, S] float64 | Recorded and predicted states (index 0: the start state) |
| ``labels`` | [E, H, P] bool | Recorded contact of step k in the evaluation's switch view: the pair touched in step k or in step k-1 (data.model_labels) |
| ``labels_physical`` | [E, H, P] bool | The pair touched in step k (data.physical_labels) |
| ``switches`` | [E, H, P] bool | The model's contact switch of step k (collision logit >= 0) |
| ``event_counts`` | [E, P, 2, 4] int64 | Onsets, offsets x (recorded, predicted, missed, false), as the evaluation counts them (data.contact_tally) |
| ``gate_accuracy`` | [E, P] | Fraction of valid steps where the switch equals the label |
| ``error/<body>/<name>`` | [E, H+1] | Errors of data.body_errors (pos_m, angle_rad, joint_rms_rad, vel_mps) |
| ``start_raw`` | [E, D, S] float64 | The recorded start state with the file's instantaneous rates (a simulator replay starts from it) |
| ``actions`` | [E, H, D, A] float64 | The recorded command of each model step |
"""
from __future__ import annotations

import argparse
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import torch

from nedm.contact_nrd.data import (CORE_ONLY, TASK_ERRORS, _steps, body_errors, contact_tally, file_sha256,
                                   model_labels, physical_labels, step_average_rates, add_derived, add_relative)
from nedm.contact_nrd.model import load_model


def npz_members(path):
    """{name: read-only np.memmap} of the arrays of an uncompressed .npz (np.savez): rows are read on access, not
    the whole file."""
    out = {}
    with zipfile.ZipFile(path) as z, open(path, "rb") as fh:
        for info in z.infolist():
            if not info.filename.endswith(".npy"):
                continue
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"{path}: {info.filename} is compressed; this reader needs np.savez (no compression)")
            fh.seek(info.header_offset)
            head = fh.read(30)
            if head[:4] != b"PK\x03\x04":
                raise ValueError(f"{path}: bad local header of {info.filename}")
            name_len, extra_len = struct.unpack("<HH", head[26:30])
            fh.seek(info.header_offset + 30 + name_len + extra_len)
            version = np.lib.format.read_magic(fh)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
            else:
                raise ValueError(f"{path}: {info.filename} has .npy format {version}")
            if dtype.hasobject:
                raise ValueError(f"{path}: {info.filename} holds Python objects")
            out[info.filename[:-4]] = np.memmap(path, dtype=dtype, mode="r", offset=fh.tell(), shape=shape,
                                                order="F" if fortran else "C")
    return out


def load_rows(root, rows, device, model_step_s):
    """The given rows of a data directory as data.load_data gives them (states [E, T, D, S] and actions [E, T, D, A]
    padded to the largest body, float64; contacts, lengths, splits, family; system, stride, dt), without the other
    episodes. Data with one ``states`` array for all balls (bouncing ball, pool) are not supported."""
    root = Path(root)
    system = json.loads((root / "system.json").read_text())
    system = {**system, "model_step_s": float(model_step_s)}
    arrays = npz_members(root / "unified_data.npz")
    if "states" in arrays and "action" not in arrays:
        raise ValueError("data with one states array for all bodies (bouncing ball, pool) are not supported")
    rows = np.asarray(rows, np.int64)
    n = len(arrays["lengths"])
    if rows.min() < 0 or rows.max() >= n:
        raise ValueError(f"rows must lie in 0 .. {n - 1}")

    def take(key):
        return np.array(arrays[key][rows])

    moving = [b for b in system["bodies"] if b["moving"]]
    record = system["record_step_s"]
    stride = round(system["model_step_s"] / record)
    smax = max(int(b["state_dim"]) for b in moving)
    amax = max(1, max(int(b.get("action_dim", 0)) for b in moving))
    t = arrays[moving[0]["name"]].shape[1]
    states = np.zeros((len(rows), t, len(moving), smax), np.float64)
    actions = np.zeros((len(rows), t, len(moving), amax), np.float64)
    for d, b in enumerate(moving):
        states[:, :, d, : int(b["state_dim"])] = take(b["name"])
        if int(b.get("action_dim", 0)):
            actions[:, :, d, : int(b["action_dim"])] = take("action")
    families = system.get("families")
    return dict(system=system, stride=stride, dt=stride * record, record_dt=record,
                target_step=round(system["target_time_s"] / (stride * record)),
                states=torch.as_tensor(states, device=device), actions=torch.as_tensor(actions, device=device),
                contacts=torch.as_tensor(take("contacts"), device=device).bool(),
                lengths=torch.as_tensor(take("lengths"), device=device),
                splits=torch.as_tensor(take("splits"), device=device),
                family=torch.as_tensor(take("family").astype(np.int64), device=device) if families else None,
                episode_id=take("episode_id") if "episode_id" in arrays else rows.copy(), rows=rows)


def pick_rows(root, per_family, families, first_step, steps, stride):
    """per_family validation rows (split 1) of each family (all families but core-only ones, or the named ones),
    evenly spread over the family's rows, among the episodes with at least first_step + steps model steps."""
    system = json.loads((Path(root) / "system.json").read_text())
    arrays = npz_members(Path(root) / "unified_data.npz")
    known = [f["name"] for f in system.get("families", [])]
    if not known:
        raise ValueError("--pick needs data with families")
    unknown = sorted(set(families or ()) - set(known))
    if unknown:
        raise ValueError(f"unknown families {unknown}; the data have {known}")
    splits, family = np.asarray(arrays["splits"]), np.asarray(arrays["family"])
    avail = (np.asarray(arrays["lengths"]).astype(np.int64) - 1) // stride
    out = []
    for k, f in enumerate(system["families"]):
        if f.get("use") == CORE_ONLY or (families and f["name"] not in families):
            continue
        cand = np.flatnonzero((splits == 1) & (family == k) & (avail >= first_step + steps))
        if not len(cand):
            print(f"family {f['name']}: no validation episode with {first_step + steps} model steps", flush=True)
            continue
        out += cand[np.unique(((np.arange(per_family) + 0.5) * len(cand) / per_family).astype(np.int64))].tolist()
    return out


@torch.no_grad()
def rollouts(model, data, mode, start_step, horizon_steps, tol):
    """Rollouts of all loaded episodes as the evaluation runs them (_free_raw for "first", _window_raw for "window")
    -> dict of numpy arrays (the keys of the module docstring, without the bookkeeping)."""
    stride, dev = data["stride"], data["states"].device
    raw = data["states"]
    step_average_rates(data, set(model.config["integrate"]["kinds"]))
    add_derived(data, model.config.get("derived"))
    add_relative(data, model.config.get("relative"))
    st, act = data["states"], data["actions"]
    e, t_rec = st.shape[:2]
    n_valid = ((data["lengths"] - 1) // stride + 1).cpu().numpy()          # grid states of each episode
    if mode == "first":
        start_step = 0
        h = (t_rec - 1) // stride if horizon_steps is None else min(horizon_steps, (t_rec - 1) // stride)
        grid = torch.arange(h + 1, device=dev) * stride
        truth = st[:, grid]
        pred, gates = model.rollout(truth[:, 0], act[:, grid[:-1]], return_gates=True)
        fit = np.minimum(h, n_valid - 1)
    else:
        h = horizon_steps
        lag = torch.arange(1 - model.k, 1, device=dev)
        rec = (start_step + lag) * stride
        ok = (rec >= 0)[None].expand(e, -1)
        hist = st[:, rec.clamp_min(0)] * ok[..., None, None]
        grid = ((start_step + torch.arange(h + 1, device=dev)) * stride).clamp_max(t_rec - 1)
        truth = st[:, grid]
        pred, gates = model.rollout_from(hist, ok, act[:, grid[:-1]], return_gates=True)
        fit = np.minimum(h, n_valid - 1 - start_step)
        if (fit < 1).any():
            short = data["rows"][fit < 1].tolist()
            raise ValueError(f"rows {short} end before {start_step * data['dt']:.2f} s + one step")
    lab = model_labels(data["contacts"], stride)
    phys = physical_labels(data["contacts"], stride)
    at = (start_step + torch.arange(h, device=dev)).clamp_max(lab.shape[1] - 1)
    labels, physical = lab[:, at].cpu().numpy(), phys[:, at].cpu().numpy()
    g = gates.cpu().numpy() > 0.5
    tally = contact_tally(g, labels, fit, tol)
    moving = [b for b in data["system"]["bodies"] if b["moving"]]
    out = dict(recorded=truth.cpu().numpy(), predicted=pred.cpu().numpy(), labels=labels, labels_physical=physical,
               switches=g, steps=fit.astype(np.int64), event_counts=tally["counts"],
               gate_accuracy=tally["acc"] / np.maximum(1, tally["den"])[:, None],
               time_s=(start_step + np.arange(h + 1)) * data["dt"],
               start_raw=raw[:, grid[0]].cpu().numpy(), actions=act[:, grid[:-1]].cpu().numpy())
    for d, b in enumerate(moving):
        cols = {m: c for m, c in ((m, [k for k, t in enumerate(b["channel_types"]) if t == typ])
                                  for m, typ in TASK_ERRORS.items()) if c}
        for m, v in body_errors(pred[:, :, d], truth[:, :, d], cols).items():
            out[f"error/{b['name']}/{m}"] = v.cpu().numpy()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="data directory (system.json, unified_data.npz)")
    parser.add_argument("--model", type=Path, required=True, help="checkpoint, or a run directory with best.pt")
    parser.add_argument("--out", type=Path, required=True, help="output .npz")
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--episodes", type=int, nargs="+", metavar="ROW", help="rows of unified_data.npz")
    who.add_argument("--pick", type=int, metavar="N", help="N validation episodes per family")
    parser.add_argument("--families", nargs="+", help="--pick: only these families")
    parser.add_argument("--mode", choices=("first", "window"), default="first")
    parser.add_argument("--start-s", type=float, default=0.0, help="window: start time (a multiple of the model step)")
    parser.add_argument("--horizon-s", type=float, help="rollout length; first: default the whole episode; "
                                                        "window: required")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    path = args.model / "best.pt" if args.model.is_dir() else args.model
    model, _ = load_model(path, args.device)
    dt = model.dt
    if args.mode == "window" and args.horizon_s is None:
        parser.error("--mode window needs --horizon-s")
    start = _steps(args.start_s, dt) if args.mode == "window" else 0
    horizon = None if args.horizon_s is None else _steps(args.horizon_s, dt)
    if horizon is not None and horizon < 1:
        parser.error("--horizon-s must be at least one model step")
    system = json.loads((args.data / "system.json").read_text())
    stride = round(dt / system["record_step_s"])
    rows = args.episodes or pick_rows(args.data, args.pick, args.families, start, horizon or 1, stride)
    if not rows:
        raise SystemExit("no episode chosen")
    data = load_rows(args.data, rows, args.device, dt)
    moving = [b for b in data["system"]["bodies"] if b["moving"]]
    extra = {b["name"]: len(model.kin_cols) for b in moving if model.kin is not None and b["name"] == model.config["derived"]["body"]}
    if model.rel is not None:
        extra[model.rel["body"]] = len(model.rel_cols)
    if (model.n_bodies, model.smax) != (len(moving), max(int(b["state_dim"]) + extra.get(b["name"], 0) for b in moving)):
        raise SystemExit("the checkpoint was trained on other bodies than the data have")
    spec = data["system"].get("evaluation") or {}
    tol = int(spec.get("contact_tolerance_steps", 1))
    out = rollouts(model, data, args.mode, start, horizon, tol)
    families = data["system"].get("families") or []
    fam = data["family"].cpu().numpy() if data["family"] is not None else np.full(len(rows), -1)
    system = data["system"]
    meta_pairs = system.get("pair_metadata") or [{} for _ in system["pairs"]]
    pairs = [m.get("name", f"pair{p}") for p, m in enumerate(meta_pairs)]
    pairs = [n if pairs.count(n) == 1 else f"{n}_{p}" for p, n in enumerate(pairs)]   # as data.task_metrics
    names = [families[k]["name"] if k >= 0 else "" for k in fam]
    meta = dict(system=system["name"], data=str(args.data.resolve()), model=str(path.resolve()),
                model_sha256=file_sha256(path), mode=args.mode, start_s=start * dt,
                horizon_s=(len(out["time_s"]) - 1) * dt, dt_s=dt, record_step_s=data["record_dt"], history=model.k,
                contact_tolerance_steps=tol, pairs=pairs, target_body=system["bodies"][system["target_body"]]["name"],
                bodies=[{k: b[k] for k in ("name", "kind", "state_dim", "channels", "channel_types") if k in b}
                        for b in moving],
                rates="rate channels of the integrated body kinds are 20 ms step averages (data.step_average_rates); "
                      "start_raw keeps the file's instantaneous rates")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, system=np.array(meta["system"]), meta=np.array(json.dumps(meta)),
                        episode=np.asarray(rows, np.int64), episode_id=np.asarray(data["episode_id"], np.int64),
                        family=fam, family_name=np.array(names), split=data["splits"].cpu().numpy(), **out)
    tb = meta["target_body"]
    for r, row in enumerate(rows):
        k = int(out["steps"][r])
        errs = ", ".join(f"{m} {out[f'error/{tb}/{m}'][r, k]:.4g}" for m in TASK_ERRORS if f"error/{tb}/{m}" in out)
        c = out["event_counts"][r]
        print(f"row {row} ({names[r] or 'no family'}): {k} steps from "
              f"{start * dt:.2f} s, {tb} at the end: {errs}; switch accuracy "
              + " ".join(f"{p} {a:.3f}" for p, a in zip(pairs, out["gate_accuracy"][r]))
              + f"; events recorded {int(c[:, :, 0].sum())}, predicted {int(c[:, :, 1].sum())}, missed "
              f"{int(c[:, :, 2].sum())}, false {int(c[:, :, 3].sum())}", flush=True)
    print(args.out)


if __name__ == "__main__":
    main()
