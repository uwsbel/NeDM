"""Write a contact NRD data directory: system.json, unified_data.npz and manifest.json.

The contract (array names, shapes, units, padding, contact labels) is in contact_nrd/collection_contract.md. A
collector builds one dict per episode (see ``pack_episodes``), packs the episodes of a worker into a shard
(``write_shard``), and the main process merges the shards and writes the directory (``merge_shards``,
``write_dataset``). numpy only.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

STATE_TYPES = ("position", "quaternion", "angle", "velocity", "angular_velocity", "angular_rate")
REPO_ROOT = Path(__file__).resolve().parents[4]


# ---- rotations (quaternions wxyz) ---------------------------------------------------------------
def quat_to_matrix(q):
    """[..., 4] unit quaternions (w, x, y, z) -> [..., 3, 3] rotation matrices (body -> world)."""
    q = np.asarray(q, np.float64)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
                     np.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
                     np.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)), -2)


def body_to_world(q, v_body):
    """Rotate body-frame vectors [..., 3] into the world frame with quaternions [..., 4] (MuJoCo free-joint qvel[3:6]
    is the angular velocity in the body frame; the contract stores it in the world frame)."""
    return np.einsum("...ij,...j->...i", quat_to_matrix(q), np.asarray(v_body, np.float64))


def quat_continuous(q):
    """[T, 4] quaternions of one episode -> unit length, w >= 0 at the first record, sign continuous along T."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    out = q.copy()
    if out[0, 0] < 0:
        out[0] = -out[0]
    for t in range(1, len(out)):
        if np.dot(out[t - 1], out[t]) < 0:
            out[t] = -out[t]
    return out


# ---- episodes -> padded arrays -------------------------------------------------------------------
def pack_episodes(episodes, system):
    """Pad a list of episodes to the system's records_per_episode and stack them.

    Each episode is a dict:
      states        {moving body name: [L, state_dim]} (L valid records, L >= 2)
      action        [L, A] applied command of the interval that starts at each record (only if a body acts)
      contacts      [L-1, P] 0/1: pair p touched during record interval k (between records k and k+1)
      per_record    optional {name: [L, ...]} extra arrays at the records (e.g. teacher command)
      per_interval  optional {name: [L-1, ...]} extra arrays of the intervals (e.g. max normal force)
      scalars       optional {name: number} one value per episode (family, episode_id, seed, outcome, ...)
    Padding: states, action and per-record arrays repeat their last valid record; contacts and per-interval arrays
    are zero after the last valid interval. Returns {array name: stacked array} with lengths [N] (int32).
    """
    t_rec = int(system["records_per_episode"])
    moving = [b for b in system["bodies"] if b["moving"]]
    n_pairs = len(system["pairs"])
    out = {b["name"]: [] for b in moving}
    out.update(action=[], contacts=[], lengths=[])
    extra_rec, extra_int, extra_sc = {}, {}, {}
    acting = any(int(b.get("action_dim", 0)) for b in moving)
    for ep in episodes:
        length = len(ep["states"][moving[0]["name"]])
        if not 2 <= length <= t_rec:
            raise ValueError(f"episode length {length} outside [2, {t_rec}]")
        for b in moving:
            x = np.asarray(ep["states"][b["name"]], np.float32)
            if x.shape != (length, int(b["state_dim"])):
                raise ValueError(f"{b['name']}: shape {x.shape}, expected {(length, int(b['state_dim']))}")
            out[b["name"]].append(_pad_repeat(x, t_rec))
        if acting:
            a = np.asarray(ep["action"], np.float32)
            if len(a) != length:
                raise ValueError(f"action has {len(a)} records, states {length}")
            out["action"].append(_pad_repeat(a, t_rec))
        c = np.asarray(ep["contacts"], np.uint8)
        if c.shape != (length - 1, n_pairs):
            raise ValueError(f"contacts shape {c.shape}, expected {(length - 1, n_pairs)}")
        out["contacts"].append(_pad_zero(c, t_rec - 1))
        out["lengths"].append(length)
        for k, v in (ep.get("per_record") or {}).items():
            extra_rec.setdefault(k, []).append(_pad_repeat(np.asarray(v), t_rec))
        for k, v in (ep.get("per_interval") or {}).items():
            extra_int.setdefault(k, []).append(_pad_zero(np.asarray(v), t_rec - 1))
        for k, v in (ep.get("scalars") or {}).items():
            extra_sc.setdefault(k, []).append(v)
    arrays = {k: np.stack(v) for k, v in out.items() if v}
    arrays["lengths"] = np.asarray(out["lengths"], np.int32)
    for group in (extra_rec, extra_int):
        for k, v in group.items():
            if len(v) != len(episodes):
                raise ValueError(f"extra array {k} missing in some episodes")
            arrays[k] = np.stack(v)
    for k, v in extra_sc.items():
        if len(v) != len(episodes):
            raise ValueError(f"scalar {k} missing in some episodes")
        arrays[k] = np.asarray(v)
    return arrays


def _pad_repeat(x, n):
    if len(x) == n:
        return x
    return np.concatenate((x, np.repeat(x[-1:], n - len(x), 0)), 0)


def _pad_zero(x, n):
    if len(x) == n:
        return x
    return np.concatenate((x, np.zeros((n - len(x),) + x.shape[1:], x.dtype)), 0)


# ---- splits ---------------------------------------------------------------------------------------
def assign_splits(families, groups, fractions=(0.9, 0.1), seed=0, split_ids=(0, 1)):
    """Deterministic family-stratified split. families [N] family index, groups [N] group id (episodes of one
    group, e.g. the branches of one parent trajectory, always share a split). Within each family the groups are
    shuffled with the seed and cut by ``fractions``; returns splits [N] int8 with the values of ``split_ids``."""
    families, groups = np.asarray(families), np.asarray(groups)
    if abs(sum(fractions) - 1) > 1e-9 or len(fractions) != len(split_ids):
        raise ValueError("fractions must sum to 1 and match split_ids")
    splits = np.full(len(families), -1, np.int8)
    for f in np.unique(families):
        sel = families == f
        uniq = np.unique(groups[sel])
        rng = np.random.default_rng([int(seed), int(f)])
        uniq = uniq[rng.permutation(len(uniq))]
        cuts = np.round(np.cumsum(fractions)[:-1] * len(uniq)).astype(int)
        for part, sid in zip(np.split(uniq, cuts), split_ids):
            splits[sel & np.isin(groups, part)] = sid
    return splits


# ---- shards and the data directory --------------------------------------------------------------
def write_shard(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def merge_shards(paths):
    """Concatenate shard files (same array names) along the episode axis, in the given order."""
    parts = []
    for p in paths:
        with np.load(p) as f:
            parts.append({k: f[k] for k in f.files})
    keys = list(parts[0])
    for p, part in zip(paths, parts):
        if list(part) != keys:
            raise ValueError(f"shard {p} has arrays {list(part)}, expected {keys}")
    return {k: np.concatenate([part[k] for part in parts], 0) for k in keys}


def check_arrays(system, arrays):
    """Shape and dtype checks of the arrays that contact NRD training reads (raises ValueError)."""
    moving = [b for b in system["bodies"] if b["moving"]]
    n = len(arrays["lengths"])
    t_rec = int(system["records_per_episode"])
    for b in moving:
        s = int(b["state_dim"])
        if len(b["channels"]) != s or len(b["channel_types"]) != s:
            raise ValueError(f"{b['name']}: channels/channel_types length differs from state_dim {s}")
        bad = set(b["channel_types"]) - set(STATE_TYPES)
        if bad:
            raise ValueError(f"{b['name']}: unknown channel types {bad}")
        if "channel_mask" in b and len(b["channel_mask"]) != s:
            raise ValueError(f"{b['name']}: channel_mask length differs from state_dim")
        if arrays[b["name"]].shape != (n, t_rec, s):
            raise ValueError(f"{b['name']}: shape {arrays[b['name']].shape}, expected {(n, t_rec, s)}")
    acting = [b for b in moving if int(b.get("action_dim", 0))]
    if len(acting) > 1:
        raise ValueError("the loader supports one actuated body")
    if acting:
        a = int(acting[0]["action_dim"])
        if arrays["action"].shape != (n, t_rec, a):
            raise ValueError(f"action shape {arrays['action'].shape}, expected {(n, t_rec, a)}")
    if arrays["contacts"].shape != (n, t_rec - 1, len(system["pairs"])):
        raise ValueError(f"contacts shape {arrays['contacts'].shape}")
    if arrays["splits"].shape != (n,) or not np.isin(arrays["splits"], (0, 1, 2)).all():
        raise ValueError("splits must be [N] with values 0, 1, 2")
    names = [b["name"] for b in system["bodies"]]
    for i, j in system["pairs"]:
        if not (0 <= i < len(names) and 0 <= j < len(names)) or not system["bodies"][i]["moving"]:
            raise ValueError(f"pair {(i, j)}: the first body must be a moving body index")
    if "pair_metadata" in system and len(system["pair_metadata"]) != len(system["pairs"]):
        raise ValueError("pair_metadata must have one entry per pair")
    check_evaluation_arrays(system, arrays)


def check_evaluation_arrays(system, arrays):
    """The per-record integer array that the evaluation block's timing_exclude names (SO101: phase) must exist."""
    exclude = (system.get("evaluation") or {}).get("timing_exclude")
    if not exclude:
        return
    n, t_rec = len(arrays["lengths"]), int(system["records_per_episode"])
    x = arrays.get(exclude["array"])
    if x is None or x.shape != (n, t_rec) or x.dtype.kind not in "iu":
        raise ValueError(f"evaluation timing_exclude needs an integer array {exclude['array']!r} [N, T]")
    names = exclude.get("names")
    if names is not None and not set(exclude["values"]) <= set(system.get(names) or []):
        raise ValueError(f"evaluation timing_exclude: system.json {names!r} must list {exclude['values']}")


def write_dataset(out_dir, system, arrays, manifest=None):
    """Write unified_data.npz, system.json and manifest.json (file hashes, environment, counts) into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = REPO_ROOT / "configs/contact_nrd/evaluation" / f"{system['name']}.json"
    if "evaluation" not in system and spec.exists():   # the task metrics of training and test (see that folder)
        system = {**system, "evaluation": json.loads(spec.read_text())}
    arrays = dict(arrays)
    arrays["lengths"] = np.asarray(arrays["lengths"], np.int32)
    arrays["splits"] = np.asarray(arrays["splits"], np.int8)
    arrays["contacts"] = np.asarray(arrays["contacts"], np.uint8)
    check_arrays(system, arrays)
    tmp = out / "unified_data.tmp.npz"
    np.savez(tmp, **arrays)
    tmp.replace(out / "unified_data.npz")
    (out / "system.json").write_text(json.dumps(system, indent=1, allow_nan=False) + "\n")
    info = dict(manifest or {})
    info.setdefault("created_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    info.setdefault("environment", environment_info())
    info.setdefault("code", git_info())
    info["episodes"] = int(len(arrays["lengths"]))
    info["valid_records"] = int(arrays["lengths"].sum())
    info["split_counts"] = {str(s): int((arrays["splits"] == s).sum()) for s in (0, 1, 2)}
    if "family" in arrays and "families" in system:
        names = [f["name"] for f in system["families"]]
        info["family_counts"] = {names[f]: {str(s): int(((arrays["family"] == f) & (arrays["splits"] == s)).sum())
                                            for s in (0, 1, 2)} for f in range(len(names))}
    info["arrays"] = {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()}
    info["files"] = {name: dict(bytes=(out / name).stat().st_size, sha256=sha256_file(out / name))
                     for name in ("unified_data.npz", "system.json")}
    (out / "manifest.json").write_text(json.dumps(info, indent=1, allow_nan=False) + "\n")
    return info


# ---- provenance -----------------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_info():
    pkgs = {}
    for name in ("mujoco", "numpy", "scipy", "onnxruntime"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return dict(python=sys.version.split()[0], platform=platform.platform(), host=platform.node(),
                cpu_count=os.cpu_count(), packages=pkgs)


def git_info(root=REPO_ROOT):
    def run(*cmd):
        try:
            return subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
    commit = run("git", "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None}
    status = run("git", "status", "--porcelain") or ""
    diff = run("git", "diff", "HEAD") or ""
    untracked = sorted((run("git", "ls-files", "--others", "--exclude-standard", "--", "src", "configs") or "").split())
    h = hashlib.sha256(diff.encode())
    for name in untracked:   # new files are not in the diff: hash their names and contents too
        h.update(name.encode())
        h.update((Path(root) / name).read_bytes())
    return dict(commit=commit.strip(), dirty=bool(status.strip()), untracked_files=len(untracked),
                diff_sha256=h.hexdigest() if (diff or untracked) else None)


def refresh_evaluation(data_dir):
    """Replace the "evaluation" block of a data directory's system.json by the current
    configs/contact_nrd/evaluation/<name>.json and update the system.json hash in manifest.json. The arrays do not
    change. Returns True if the block changed."""
    root = Path(data_dir)
    system = json.loads((root / "system.json").read_text())
    spec = json.loads((REPO_ROOT / "configs/contact_nrd/evaluation" / f"{system['name']}.json").read_text())
    if system.get("evaluation") == spec:
        return False
    system["evaluation"] = spec
    if spec.get("timing_exclude"):
        with np.load(root / "unified_data.npz") as f:   # reads only the arrays asked for
            check_evaluation_arrays(system, {"lengths": f["lengths"], spec["timing_exclude"]["array"]:
                                             f[spec["timing_exclude"]["array"]] if spec["timing_exclude"]["array"] in f.files else None})
    (root / "system.json").write_text(json.dumps(system, indent=1, allow_nan=False) + "\n")
    info = json.loads((root / "manifest.json").read_text())
    info["files"]["system.json"] = dict(bytes=(root / "system.json").stat().st_size, sha256=sha256_file(root / "system.json"))
    info.setdefault("evaluation_refreshed_utc", []).append(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (root / "manifest.json").write_text(json.dumps(info, indent=1, allow_nan=False) + "\n")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Refresh the evaluation block of data directories.")
    parser.add_argument("refresh_evaluation", nargs="+", type=Path, metavar="DIR")
    for d in parser.parse_args().refresh_evaluation:
        print(d, "changed" if refresh_evaluation(d) else "unchanged")
