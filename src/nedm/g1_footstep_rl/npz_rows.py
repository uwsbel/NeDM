"""Read chosen rows of a contact NRD data directory without loading the whole file (the G1 training data are 19 GB).

npz_members memory-maps the arrays of an uncompressed unified_data.npz (np.savez); load_rows returns the given rows
in the layout of nedm.contact_nrd.data.load_data. Used by the start bank (bank.py) and the start filter of the NRD
environment (planks_env.StartBank).
"""
from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import torch


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
