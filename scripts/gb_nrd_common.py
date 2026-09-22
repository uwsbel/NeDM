#!/usr/bin/env python
"""Shared code for the mixed rigid/deformable-soil dynamics model (PLAN B4) and its consumers (PLAN B5 tracker env).

Model (``GBNRDModel``): the WP2 map model's causal transformer (``nedm.traverse.nrd_model.WP2MapModel`` /
``nedm.training.model_transformer.ContinuousTransformer``) with the learned scene-map token replaced by an MLP on the
raw ego-aligned elevation crop from the Chrono-frame v2 grid (``scripts/gb_crop.py``): ``k*k`` relative heights (/2 m)
plus the validity flag -> 64-D token, computed at every pose (training, rollout loss, validation, imagination); the
crop is never stored.  Per token the input is ``[z1 (17, z-scored), token (64), action (3, z-scored), domain one-hot
(2, only when cond='tag')]``; heads predict the delta of the normalised z1 and the normalised power (kW).  The pose is
integrated outside the network with ``nrd_model.integrate_pose`` (yaw first, then the world velocity uses the new yaw,
planar, 50 ms).

Cache (PLAN 'Cache contract for B', schema 3, written by ``gb_build_cache.py``): per-episode npz with ``z1 (T,17) f32``,
``act (T,3)``, ``pose (T,3)``, ``power (T,1)``, ``stalled (T,) bool``, ``hold_ok (T,) bool``, ``desired_speed (T,)``,
``route_{waypoints,speeds,headings,stations}``, ``domain`` (0 rigid, 1 crm), ``group``, ``status``; a
``cache_manifest.json`` with ``episodes``, ``domain_of``, ``group_of``, ``split_of``, ``status_of``, ``schema: 3``.
``load_cache`` pads variable-length episodes with their last row and returns a ``valid`` mask; every consumer honours
it.  The group split comes from the manifest only (``split_of``); ``heldout_groups`` lists the val/test groups so the
tracker bank and the trainer can assert that none is present in their training set.

Checkpoint (``save_nrd`` / ``load_nrd``): the WP2 payload keys (``model``, ``config``, ``normalization``, ``step``,
``metrics``, ``z1_dim``, ``delta_scale``) plus ``model_kind='gb_nrd'``, ``crop_k``, ``crop_half_m``, ``cond``,
``grid_path``, ``grid_sha256``, ``domain_vocab``, ``token_dim`` and, in ``ckpt_last.pt`` only, ``train_state``
(optimizer, schedule, step, RNG) for ``--resume``.

Synthetic cache for self-tests: ``python scripts/gb_nrd_common.py --make-synthetic-cache DIR`` writes a schema-3
mini-cache (both domains, straight routes, first-order speed response, stalls and brake taps) on the real v2 grid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT / "src"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gb_crop  # noqa: E402
from nedm.traverse import nrd_data as D  # noqa: E402
from nedm.traverse.nrd_model import DT_S, PITCH, ROLL, VX, VY, YAW_RATE, integrate_pose  # noqa: E402,F401
from nedm.training.model_transformer import ContinuousTransformer, TransformerConfig  # noqa: E402

MODEL_KIND = "gb_nrd"
DOMAIN_VOCAB = ("rigid", "crm")  # cache ``domain`` 0 / 1
CACHE_SCHEMA = 3
DEFAULT_GRID = "artifacts/traverse/crm_f104_v1/grids/arena_f104_50h_v1/grid.npz"
DEFAULT_TWIN_SPLIT = "artifacts/traverse/crm_night2_v1/datasets/twin_crm.npz"  # PLAN: the twin group split is the only split
Z1_DIM, ACT_DIM = 17, 3
ROUTE_FIELDS = ("waypoints", "speeds", "headings", "stations")
SETTLE_ACTION = (0.0, 0.0, 1.0)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(p: str | Path) -> Path:
    """Relative paths are taken from the repo root (the scripts' convention), absolute ones as given."""
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


# =============================================================================================== cache
def read_manifest(cache_dir) -> dict:
    man = json.loads((resolve_path(cache_dir) / "cache_manifest.json").read_text())
    if int(man.get("schema", 0)) != CACHE_SCHEMA:
        raise ValueError(f"{cache_dir}: cache schema {man.get('schema')} != {CACHE_SCHEMA}")
    for key in ("episodes", "domain_of", "group_of", "split_of", "status_of"):
        if key not in man:
            raise ValueError(f"{cache_dir}: manifest lacks {key!r}")
    return man


def heldout_groups(man: dict) -> set[str]:
    """Groups whose split is val or test (never in a training set / fragment bank)."""
    return {man["group_of"][k] for k in man["episodes"] if man["split_of"][k] in ("val", "test")}


def check_group_split_consistency(man: dict) -> None:
    seen: dict[str, str] = {}
    for k in man["episodes"]:
        g, s = man["group_of"][k], man["split_of"][k]
        if seen.setdefault(g, s) != s:
            raise ValueError(f"group {g} appears in splits {seen[g]} and {s}: the split must be by group")


def twin_split_of_groups(path) -> dict[str, str]:
    """group -> split read from a twin dataset (``crm_night2_v1/datasets/twin_*.npz``: arrays ``group`` and ``split``)."""
    with np.load(resolve_path(path), allow_pickle=True) as z:
        groups, splits = z["group"].tolist(), z["split"].tolist()
    out: dict[str, str] = {}
    for g, s in zip(groups, splits):
        if out.setdefault(str(g), str(s)) != str(s):
            raise ValueError(f"{path}: group {g} carries two splits")
    return out


def check_split_against_twin(man: dict, twin_path: str | None = None) -> dict:
    """Cross-check the manifest's group split against the twin group split, independently of the cache builder.
    ``twin_path``: None = the file named in the manifest's build report (fallback ``DEFAULT_TWIN_SPLIT``), 'none' = skip
    (printed).  A manifest flagged ``synthetic`` is skipped when no path is given.  Raises when a cache group is unknown
    to the twin split or carries a different split."""
    if isinstance(twin_path, str) and twin_path.lower() == "none":
        print("split cross-check against the twin split SKIPPED (twin_split none)", flush=True)
        return {"checked": False, "reason": "skipped by request"}
    if twin_path is None and man.get("synthetic"):
        print("split cross-check against the twin split skipped: synthetic cache", flush=True)
        return {"checked": False, "reason": "synthetic cache"}
    report = man.get("report") if isinstance(man.get("report"), dict) else {}
    # an explicit path is used as given; otherwise the manifest's record, then the repo default (another machine)
    candidates = [twin_path] if twin_path else [p for p in (report.get("twin_split"), DEFAULT_TWIN_SPLIT) if p]
    path = next((p for p in candidates if resolve_path(p).exists()), None)
    if path is None:
        raise FileNotFoundError(f"twin split not found (tried {candidates}); sync it or pass twin_split=PATH / 'none'")
    twin = twin_split_of_groups(path)
    cache: dict[str, str] = {}
    for k in man["episodes"]:
        cache.setdefault(man["group_of"][k], man["split_of"][k])
    unknown = sorted(g for g in cache if g not in twin)
    differ = sorted((g, cache[g], twin[g]) for g in cache if g in twin and cache[g] != twin[g])
    if unknown or differ:
        raise ValueError(f"cache split disagrees with the twin split {path}: {len(unknown)} groups unknown to it {unknown[:5]}, "
                         f"{len(differ)} groups with a different split (group, cache, twin) {differ[:5]}")
    n = {s: sum(1 for g in cache if cache[g] == s) for s in ("train", "val", "test")}
    print(f"split cross-check: {len(cache)} cache groups agree with the twin split {path} {n}", flush=True)
    return {"checked": True, "twin_split": str(path), "groups": len(cache), "groups_by_split": n}


def select_keys(man: dict, splits=None, domains=None, max_episodes: int = 0, seed: int = 0) -> list[str]:
    """Episode keys of the requested splits / domains; ``max_episodes`` > 0 subsamples deterministically."""
    keys = [k for k in man["episodes"]
            if (splits is None or man["split_of"][k] in splits)
            and (domains is None or int(man["domain_of"][k]) in domains)]
    if max_episodes and len(keys) > max_episodes:
        order = np.random.default_rng(seed).permutation(len(keys))[:max_episodes]
        keys = [keys[i] for i in sorted(order)]
    return keys


@dataclass
class CacheData:
    """Stacked, last-row-padded arrays of one set of episodes (physical units)."""

    keys: list[str]
    z1: np.ndarray            # (N, T, 17) f32
    act: np.ndarray           # (N, T, 3) f32
    pose: np.ndarray          # (N, T, 3) f32
    power: np.ndarray         # (N, T, 1) f32 kW
    stalled: np.ndarray       # (N, T) bool
    hold_ok: np.ndarray       # (N, T) bool
    desired_speed: np.ndarray  # (N, T) f32
    valid: np.ndarray         # (N, T) bool recorded frames
    n_valid: np.ndarray       # (N,) int
    domain: np.ndarray        # (N,) int 0 rigid / 1 crm
    group: list[str]
    split: list[str]
    status: list[str]
    routes: list[dict]        # per episode: waypoints (L,2), speeds, headings, stations

    @property
    def n_episodes(self) -> int:
        return int(self.z1.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.z1.shape[1])


def _pad_rows(a: np.ndarray, n: int) -> np.ndarray:
    if a.shape[0] >= n:
        return a[:n]
    return np.concatenate([a, np.repeat(a[-1:], n - a.shape[0], axis=0)], axis=0)


def _route_from_npz(z) -> dict:
    out = {}
    for f in ROUTE_FIELDS:
        name = f"route_{f}" if f"route_{f}" in z.files else f
        if name not in z.files:
            raise KeyError(f"route field {f} missing (looked for route_{f} and {f})")
        out[f] = np.asarray(z[name], np.float32)
    return out


def load_cache(cache_dir, keys: list[str], max_frames: int = 0, with_routes: bool = True) -> CacheData:
    """Load ``keys`` of a schema-3 cache; episodes longer than ``max_frames`` (> 0) are cut."""
    cache_dir = resolve_path(cache_dir)
    man = read_manifest(cache_dir)
    cols = {n: [] for n in ("z1", "act", "pose", "power", "stalled", "hold_ok", "desired_speed")}
    routes, domain, lengths = [], [], []
    for k in keys:
        with np.load(cache_dir / f"{k}.npz", allow_pickle=False) as z:
            T = int(z["z1"].shape[0])
            if max_frames and T > max_frames:
                T = max_frames
            cols["z1"].append(np.asarray(z["z1"][:T], np.float32))
            cols["act"].append(np.asarray(z["act"][:T], np.float32))
            cols["pose"].append(np.asarray(z["pose"][:T], np.float32))
            pw = np.asarray(z["power"][:T], np.float32)
            cols["power"].append(pw.reshape(T, 1) if pw.ndim == 1 else pw)
            cols["stalled"].append(np.asarray(z["stalled"][:T], bool))
            cols["hold_ok"].append(np.asarray(z["hold_ok"][:T], bool))
            cols["desired_speed"].append(np.asarray(z["desired_speed"][:T], np.float32))
            routes.append(_route_from_npz(z) if with_routes else {})
            dom = int(z["domain"]) if "domain" in z.files else int(man["domain_of"][k])
            if dom != int(man["domain_of"][k]):
                raise ValueError(f"{k}: file domain {dom} != manifest {man['domain_of'][k]}")
            domain.append(dom)
            lengths.append(T)
        if cols["z1"][-1].shape[-1] != Z1_DIM:
            raise ValueError(f"{k}: z1 is {cols['z1'][-1].shape[-1]}-D, expected {Z1_DIM}")
    lengths = np.asarray(lengths, np.int64)
    T = int(lengths.max()) if len(lengths) else 0
    stack = lambda rows: np.stack([_pad_rows(r, T) for r in rows]) if rows else np.zeros((0, 0))
    valid = np.arange(T)[None, :] < lengths[:, None]
    return CacheData(keys=list(keys), z1=stack(cols["z1"]), act=stack(cols["act"]), pose=stack(cols["pose"]),
                     power=stack(cols["power"]), stalled=stack(cols["stalled"]) & valid,
                     hold_ok=stack(cols["hold_ok"]) & valid, desired_speed=stack(cols["desired_speed"]),
                     valid=valid, n_valid=lengths, domain=np.asarray(domain, np.int64),
                     group=[man["group_of"][k] for k in keys], split=[man["split_of"][k] for k in keys],
                     status=[man["status_of"][k] for k in keys], routes=routes)


def fit_normalizer(data: CacheData, eps: float = 1e-6) -> D.Normalizer:
    """Per-channel z-score statistics over the recorded frames (both domains pooled, as the training set is)."""
    m = data.valid.reshape(-1)
    stats = lambda a: (a.reshape(-1, a.shape[-1])[m].mean(0), np.maximum(a.reshape(-1, a.shape[-1])[m].std(0), eps))
    z1_mean, z1_std = stats(data.z1)
    act_mean, act_std = stats(data.act)
    p_mean, p_std = stats(data.power)
    return D.Normalizer(z1_mean.astype(np.float32), z1_std.astype(np.float32), np.zeros(0, np.float32),
                        np.ones(0, np.float32), act_mean.astype(np.float32), act_std.astype(np.float32),
                        p_mean.astype(np.float32), p_std.astype(np.float32))


# =============================================================================================== model
class CropTokenizer(nn.Module):
    """Raw elevation crop at the pose -> 64-D token: MLP(k*k heights + valid flag -> hidden -> token_dim)."""

    def __init__(self, grid: dict, k: int, half_m: float, token_dim: int = 64, hidden: int = 128, device="cpu"):
        super().__init__()
        self.crop = gb_crop.EgoCrop(grid, device)
        self.k, self.half_m, self.token_dim = int(k), float(half_m), int(token_dim)
        self.mlp = nn.Sequential(nn.Linear(self.k * self.k + 1, hidden), nn.GELU(), nn.Linear(hidden, token_dim))

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """pose (..., 3) -> token (..., token_dim)."""
        h, ok = self.crop(pose, self.k, self.half_m)
        x = torch.cat([h.flatten(-2), ok.to(h.dtype).unsqueeze(-1)], dim=-1)
        return self.mlp(x.to(self.mlp[0].weight.dtype))  # the crop is always float32; follow a .half()/.bfloat16() MLP


class GBNRDModel(nn.Module):
    """Backbone over [z1, crop token, action, (domain one-hot)]; heads: delta z1 (normalised), power (normalised)."""

    def __init__(self, cfg: dict, grid: dict, device="cpu"):
        super().__init__()
        self.cfg = dict(cfg)
        self.cond = str(cfg["cond"])
        if self.cond not in ("tag", "notag"):
            raise ValueError(f"cond must be tag or notag, got {self.cond!r}")
        self.z1_dim, self.act_dim = int(cfg.get("z1_dim", Z1_DIM)), int(cfg.get("act_dim", ACT_DIM))
        self.token_dim = int(cfg.get("token_dim", 64))
        self.n_domains = len(DOMAIN_VOCAB) if self.cond == "tag" else 0
        self.tokenizer = CropTokenizer(grid, int(cfg["crop_k"]), float(cfg["crop_half_m"]), self.token_dim,
                                       int(cfg.get("token_hidden", 128)), device)
        self.backbone = ContinuousTransformer(TransformerConfig(
            input_dim=self.z1_dim + self.token_dim + self.act_dim + self.n_domains, block_size=int(cfg["block_size"]),
            n_layer=int(cfg["n_layer"]), n_head=int(cfg["n_head"]), n_embd=int(cfg["n_embd"]),
            dropout=float(cfg["dropout"]), bias=bool(cfg["bias"])))
        hidden, n_embd = int(cfg["head_hidden_dim"]), int(cfg["n_embd"])
        mlp = lambda out: nn.Sequential(nn.Linear(n_embd, hidden), nn.GELU(), nn.Linear(hidden, out))
        self.state_head, self.power_head = mlp(self.z1_dim), mlp(1)
        self.to(device)

    @property
    def context(self) -> int:
        return int(self.cfg["block_size"])

    def token(self, pose: torch.Tensor) -> torch.Tensor:
        return self.tokenizer(pose)

    def forward(self, z1: torch.Tensor, token: torch.Tensor, act: torch.Tensor, domain: torch.Tensor | None = None):
        """z1 (B, L, 17) normalised, token (B, L, 64), act (B, L, 3) normalised, domain (B,) int -> (delta (B, L, 17), power (B, L, 1))."""
        parts = [z1, token, act]
        if self.n_domains:
            if domain is None:
                raise ValueError("a cond='tag' model needs the domain of every window")
            oh = F.one_hot(domain.long(), self.n_domains).to(z1.dtype)
            parts.append(oh[:, None, :].expand(-1, z1.shape[1], -1))
        feat = self.backbone(torch.cat(parts, dim=-1))
        return self.state_head(feat), self.power_head(feat)


def model_config(*, cond: str, crop_k: int, crop_half_m: float, block_size: int = 16, n_layer: int = 6, n_head: int = 8,
                 n_embd: int = 256, dropout: float = 0.0, token_dim: int = 64, token_hidden: int = 128) -> dict:
    return {"block_size": int(block_size), "n_layer": int(n_layer), "n_head": int(n_head), "n_embd": int(n_embd),
            "dropout": float(dropout), "bias": False, "head_hidden_dim": int(n_embd), "z1_dim": Z1_DIM,
            "act_dim": ACT_DIM, "token_dim": int(token_dim), "token_hidden": int(token_hidden), "cond": str(cond),
            "crop_k": int(crop_k), "crop_half_m": float(crop_half_m)}


def save_nrd(path, model: GBNRDModel, norm: D.Normalizer, step: int, metrics: dict, grid_path: str, grid_sha256: str,
             delta_scale=None, extra: dict | None = None, train_state: dict | None = None) -> None:
    cfg = dict(model.cfg)
    payload = {"model_kind": MODEL_KIND, "model": model.state_dict(), "config": cfg, "normalization": norm.to_dict(),
               "step": int(step), "metrics": metrics, "z1_dim": model.z1_dim, "crop_k": cfg["crop_k"],
               "crop_half_m": cfg["crop_half_m"], "cond": cfg["cond"], "token_dim": cfg["token_dim"],
               "grid_path": str(grid_path), "grid_sha256": grid_sha256, "domain_vocab": list(DOMAIN_VOCAB),
               "delta_scale": None if delta_scale is None else [float(v) for v in delta_scale]}
    if extra:
        payload.update(extra)
    if train_state is not None:
        payload["train_state"] = train_state
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_nrd(path, device, grid_path: str | None = None, frozen: bool = True):
    """Checkpoint -> (model on device, D.Normalizer, payload). ``grid_path`` overrides the stored one (other machine);
    the grid's sha256 is checked against the checkpoint's."""
    payload = torch.load(resolve_path(path), map_location="cpu", weights_only=False)
    if payload.get("model_kind") != MODEL_KIND:
        raise ValueError(f"{path}: model_kind {payload.get('model_kind')!r} != {MODEL_KIND!r}")
    gp = grid_path or payload["grid_path"]
    gpath = resolve_path(gp)
    if not gpath.exists():
        raise FileNotFoundError(f"grid {gp} not found; pass grid_path= to load_nrd (checkpoint grid sha256 {payload['grid_sha256'][:16]})")
    sha = file_sha256(gpath)
    if sha != payload["grid_sha256"]:
        raise ValueError(f"grid {gp} sha256 {sha[:16]} != checkpoint's {payload['grid_sha256'][:16]}")
    model = GBNRDModel(payload["config"], gb_crop.load_grid(str(gpath)), device)
    missing, unexpected = model.load_state_dict(payload["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    if frozen:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    norm = D.Normalizer.from_dict(payload["normalization"])
    return model, norm, payload


# =============================================================================================== synthetic cache
def integrate_pose_np(pose: np.ndarray, vx: float, vy: float, yaw_rate: float) -> np.ndarray:
    """numpy twin of nrd_model.integrate_pose for one step."""
    yaw = pose[2] + DT_S * yaw_rate
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([pose[0] + DT_S * (c * vx - s * vy), pose[1] + DT_S * (s * vx + c * vy), yaw])


def stalled_flags(vx: np.ndarray, throttle: np.ndarray, min_run: int = 20) -> np.ndarray:
    """PLAN cache contract: |vx| < 0.3 and throttle > 0.3, in runs of >= min_run frames."""
    raw = (np.abs(vx) < 0.3) & (throttle > 0.3)
    out = np.zeros_like(raw)
    i = 0
    while i < len(raw):
        if raw[i]:
            j = i
            while j < len(raw) and raw[j]:
                j += 1
            if j - i >= min_run:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def hold_ok_flags(act: np.ndarray) -> np.ndarray:
    """PLAN cache contract: |action[k+1] - action[k]| <= 0.1 per channel and no throttle/brake flip; last frame False."""
    T = act.shape[0]
    ok = np.zeros(T, bool)
    if T < 2:
        return ok
    d = np.abs(act[1:] - act[:-1]).max(1) <= 0.1 + 1e-9
    flip = ((act[:-1, 1] > 0) & (act[1:, 2] > 0)) | ((act[:-1, 2] > 0) & (act[1:, 1] > 0))
    ok[:-1] = d & ~flip
    return ok


def make_synthetic_cache(out_dir, grid_path: str = DEFAULT_GRID, n_episodes: int = 40, n_frames: int = 300,
                         seed: int = 20260921, n_groups: int = 10, val_groups: int = 2, test_groups: int = 1) -> dict:
    """Schema-3 mini-cache with plausible dynamics for self-tests (not data).  Episodes alternate domains; groups of
    ``n_episodes / n_groups`` episodes; the last ``val_groups + test_groups`` groups are held out.  Each episode drives a
    straight route (0.5 m waypoints, 3 m/s) with a scripted follower (steer on cross-track + heading, throttle on the
    speed error, rate-limited); half the episodes stall for 100 frames (vx -> 0 under throttle, status 'stall'); every
    episode has one 1 s brake tap in the middle.  Pose follows nrd_model.integrate_pose."""
    out_dir = resolve_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    grid = gb_crop.load_grid(str(resolve_path(grid_path)))
    per_group = max(n_episodes // n_groups, 1)
    man = {"schema": CACHE_SCHEMA, "episodes": [], "domain_of": {}, "group_of": {}, "split_of": {}, "status_of": {},
           "synthetic": True, "grid_path": str(grid_path), "seed": seed}
    n_stalled_ep = 0
    for i in range(n_episodes):
        g = min(i // per_group, n_groups - 1)
        split = "test" if g >= n_groups - test_groups else ("val" if g >= n_groups - test_groups - val_groups else "train")
        dom = i % 2
        key = f"syn_{DOMAIN_VOCAB[dom]}_{i:04d}"
        group = f"syn_group_{g:03d}"
        # route: straight line, 0.5 m spacing, inside +-30 m
        x0, y0 = rng.uniform(-28, -20), rng.uniform(-25, 25)
        yaw0 = rng.uniform(-0.25, 0.25)
        L = 50.0
        n_wp = int(L / 0.5) + 1
        s = np.arange(n_wp) * 0.5
        wps = np.stack([x0 + s * math.cos(yaw0), y0 + s * math.sin(yaw0)], 1).astype(np.float32)
        v_ref = 3.0
        route = {"waypoints": wps, "speeds": np.full(n_wp, v_ref, np.float32),
                 "headings": np.full(n_wp, yaw0, np.float32), "stations": s.astype(np.float32)}
        stall = (i % 4) in (1, 2)  # half the episodes, both domains
        stall_start = int(rng.integers(120, 160)) if stall else -1
        brake_at = int(rng.integers(60, 90))
        T = n_frames
        z1 = np.zeros((T, Z1_DIM), np.float32)
        act = np.zeros((T, ACT_DIM), np.float32)
        pose = np.zeros((T, 3), np.float32)
        power = np.zeros((T, 1), np.float32)
        p = np.array([x0, y0 + rng.normal(0, 0.2), yaw0 + rng.normal(0, 0.05)])
        vx, vy, yr = 0.0, 0.0, 0.0
        last = np.array(SETTLE_ACTION)
        k_t, drag = (7.0, 0.12) if dom == 0 else (5.0, 0.35)  # deformable soil: weaker response, more drag
        for t in range(T):
            # state at frame t (before the action of interval t)
            roll = 0.02 * math.sin(0.1 * t) + rng.normal(0, 0.003)
            pitch = 0.01 * math.cos(0.07 * t) + rng.normal(0, 0.003)
            fz = 5000.0 + 300.0 * rng.normal(size=4) + (400.0 if dom else 0.0)
            omega = np.full(4, vx / 0.47) + rng.normal(0, 0.2, 4)
            eng = 80.0 + 40.0 * vx + rng.normal(0, 2.0)
            torque = 300.0 * last[1] + rng.normal(0, 5.0)
            z1[t] = np.concatenate([[vx, vy, roll, pitch, rng.normal(0, 0.01), rng.normal(0, 0.01), yr], fz, omega, [eng, torque]])
            pose[t] = p
            power[t, 0] = max(eng * torque / 1000.0, 0.0)
            # scripted follower for interval t
            d = np.hypot(wps[:, 0] - p[0], wps[:, 1] - p[1])
            j = int(np.argmin(d))
            h = yaw0
            e_ct = -(p[0] - wps[j, 0]) * math.sin(h) + (p[1] - wps[j, 1]) * math.cos(h)
            e_h = math.atan2(math.sin(p[2] - h), math.cos(p[2] - h))
            steer = float(np.clip(-0.4 * e_ct - 0.8 * e_h, -1, 1))
            steer = float(np.clip(steer, last[0] - 0.1, last[0] + 0.1))
            if brake_at <= t < brake_at + 20:
                thr, brk = 0.0, 0.5
            else:
                thr, brk = float(np.clip(0.3 + 0.15 * (v_ref - vx), 0.0, 1.0)), 0.0
                if stall and stall_start <= t < stall_start + 100:
                    thr = 0.6
            a = np.array([steer, thr, brk])
            act[t] = a
            last = a
            # dynamics over interval t
            stuck = stall and stall_start <= t < stall_start + 100
            dv = (k_t * thr - 12.0 * brk * (vx > 0) - drag * vx) * DT_S + rng.normal(0, 0.02)
            if stuck:  # wheels spin, the vehicle sinks to a halt within ~10 frames and stays there
                vx = max(0.7 * vx, 0.0) + abs(rng.normal(0, 0.01))
            else:
                vx = max(vx + dv, 0.0)
            vy = 0.9 * vy + rng.normal(0, 0.01)
            yr = 0.8 * yr + 0.2 * (1.2 * steer * vx / (1.0 + 0.1 * vx)) + rng.normal(0, 0.005)
            p = integrate_pose_np(p, vx, vy, yr)
        stalled = stalled_flags(z1[:, 0], act[:, 1])
        hold_ok = hold_ok_flags(act)
        status = "stall" if stall else "completed"
        n_stalled_ep += int(stalled.any())
        np.savez(out_dir / f"{key}.npz", z1=z1, act=act, pose=pose, power=power, stalled=stalled, hold_ok=hold_ok,
                 desired_speed=np.full(T, v_ref, np.float32), route_waypoints=route["waypoints"],
                 route_speeds=route["speeds"], route_headings=route["headings"], route_stations=route["stations"],
                 domain=np.int64(dom), group=np.array(group), status=np.array(status))
        man["episodes"].append(key)
        man["domain_of"][key] = dom
        man["group_of"][key] = group
        man["split_of"][key] = split
        man["status_of"][key] = status
    man["n_stalled_episodes"] = n_stalled_ep
    (out_dir / "cache_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    return man


def _selftest_model(grid_path: str, device: str) -> dict:
    """Shape / determinism checks of the model and a save/load round trip in a temp dir."""
    import tempfile
    grid = gb_crop.load_grid(str(resolve_path(grid_path)))
    rep = {}
    for cond in ("tag", "notag"):
        cfg = model_config(cond=cond, crop_k=8, crop_half_m=6.0, n_layer=2, n_head=4, n_embd=64)
        torch.manual_seed(0)
        m = GBNRDModel(cfg, grid, device).eval()
        B, L = 5, 16
        pose = torch.tensor(np.c_[np.random.default_rng(0).uniform(-30, 30, (B * L, 2)), np.random.default_rng(1).uniform(-3, 3, (B * L, 1))], dtype=torch.float32, device=device).view(B, L, 3)
        tok = m.token(pose)
        z1 = torch.randn(B, L, Z1_DIM, device=device); act = torch.randn(B, L, ACT_DIM, device=device)
        dom = torch.tensor([0, 1, 0, 1, 1], device=device)
        with torch.no_grad():
            d, pw = m(z1, tok, act, dom if cond == "tag" else None)
        rep[f"{cond}_shapes"] = [list(tok.shape), list(d.shape), list(pw.shape)]
        if cond == "tag":
            with torch.no_grad():
                d2, _ = m(z1, tok, act, 1 - dom)
            rep["tag_changes_output"] = bool((d - d2).abs().max() > 1e-6)
        norm = D.Normalizer(np.zeros(Z1_DIM, np.float32), np.ones(Z1_DIM, np.float32), np.zeros(0, np.float32), np.ones(0, np.float32),
                            np.zeros(3, np.float32), np.ones(3, np.float32), np.zeros(1, np.float32), np.ones(1, np.float32))
        with tempfile.TemporaryDirectory() as td:
            pth = Path(td) / "ckpt.pt"
            save_nrd(pth, m, norm, 0, {}, grid_path, file_sha256(resolve_path(grid_path)))
            m2, _, payload = load_nrd(pth, device)
            with torch.no_grad():
                d3, _ = m2(z1, m2.token(pose), act, dom if cond == "tag" else None)
            rep[f"{cond}_roundtrip_max_abs"] = float((d - d3).abs().max())
            rep[f"{cond}_payload_keys"] = sorted(payload.keys())
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--make-synthetic-cache", default="", help="write a schema-3 mini-cache to this directory")
    ap.add_argument("--grid", default=DEFAULT_GRID)
    ap.add_argument("--n-episodes", type=int, default=40)
    ap.add_argument("--n-frames", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--selftest", action="store_true", help="model shape / round-trip checks")
    a = ap.parse_args()
    if a.make_synthetic_cache:
        man = make_synthetic_cache(a.make_synthetic_cache, a.grid, a.n_episodes, a.n_frames, a.seed)
        cnt = {s: sum(1 for k in man["episodes"] if man["split_of"][k] == s) for s in ("train", "val", "test")}
        print(json.dumps({"episodes": len(man["episodes"]), "splits": cnt, "stalled_episodes": man["n_stalled_episodes"],
                          "heldout_groups": sorted(heldout_groups(man))}, indent=1))
    if a.selftest:
        print(json.dumps(_selftest_model(a.grid, "cuda" if torch.cuda.is_available() else "cpu"), indent=1))
