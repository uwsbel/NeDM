"""Episode cache, group split, normalisation and window batching for the dynamics model (and the tracker bank).

Cache (schema 3): one npz per episode with ``z1 (T,17)`` (``state.STATE_FIELDS``), ``act (T,3)``, ``pose (T,3)``
world x, y, yaw, ``power (T,1)`` kW, ``stalled (T,)`` (inside a run of >= 20 frames with |vx| < 0.3 m/s under
throttle > 0.3), ``hold_ok (T,)`` (the action was held from frame k to k+1: no channel moved by more than 0.1 and no
throttle/brake flip), ``domain`` (0 rigid, 1 soil) and ``route_*``; plus ``cache_manifest.json`` with ``episodes``,
``domain_of``, ``group_of``, ``split_of``, ``status_of``. The split is by start/goal group and comes from the manifest
only; it is cross-checked against the group split of the soil twin dataset (``twin_crm.npz``).

Relative paths are taken from the repository root, so the defaults are the paths the release downloader restores.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from nedm.traversing.training.state import Z1_DIM

ROOT = Path(__file__).resolve().parents[4]
DOMAIN_VOCAB = ("rigid", "crm")  # cache ``domain`` 0 / 1
CACHE_SCHEMA = 3
ROUTE_FIELDS = ("waypoints", "speeds", "headings", "stations")
DEFAULT_CACHE = "artifacts/traverse/generalist_20260921/B_tracker/cache_v3"
DEFAULT_GRID = "artifacts/traverse/crm_f104_v1/grids/arena_f104_50h_v1/grid.npz"
DEFAULT_TWIN_SPLIT = "artifacts/traverse/crm_night2_v1/datasets/twin_crm.npz"


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------ manifest and split
def read_manifest(cache_dir) -> dict:
    man = json.loads((resolve_path(cache_dir) / "cache_manifest.json").read_text())
    if int(man.get("schema", 0)) != CACHE_SCHEMA:
        raise ValueError(f"{cache_dir}: cache schema {man.get('schema')} != {CACHE_SCHEMA}")
    for key in ("episodes", "domain_of", "group_of", "split_of", "status_of"):
        if key not in man:
            raise ValueError(f"{cache_dir}: manifest lacks {key!r}")
    return man


def heldout_groups(man: dict) -> set[str]:
    """Groups whose split is val or test (never in a training set or a tracker fragment bank)."""
    return {man["group_of"][k] for k in man["episodes"] if man["split_of"][k] in ("val", "test")}


def check_group_split_consistency(man: dict) -> None:
    seen: dict[str, str] = {}
    for k in man["episodes"]:
        g, s = man["group_of"][k], man["split_of"][k]
        if seen.setdefault(g, s) != s:
            raise ValueError(f"group {g} appears in splits {seen[g]} and {s}: the split must be by group")


def check_split_against_twin(man: dict, twin_path: str | None = None) -> dict:
    """Check that every cache group has the same split in the twin dataset (``group`` / ``split`` arrays).
    ``twin_path`` None = ``DEFAULT_TWIN_SPLIT``; 'none' skips the check."""
    if isinstance(twin_path, str) and twin_path.lower() == "none":
        print("split cross-check against the twin split SKIPPED (twin_split none)", flush=True)
        return {"checked": False, "reason": "skipped by request"}
    path = twin_path or DEFAULT_TWIN_SPLIT
    if not resolve_path(path).exists():
        raise FileNotFoundError(f"twin split {path} not found; download it or pass twin_split=PATH / 'none'")
    twin: dict[str, str] = {}
    with np.load(resolve_path(path), allow_pickle=True) as z:
        for g, s in zip(z["group"].tolist(), z["split"].tolist()):
            if twin.setdefault(str(g), str(s)) != str(s):
                raise ValueError(f"{path}: group {g} carries two splits")
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


def select_keys(man: dict, splits=None, max_episodes: int = 0, seed: int = 0) -> list[str]:
    """Episode keys of the requested splits in manifest order; ``max_episodes`` > 0 subsamples deterministically."""
    keys = [k for k in man["episodes"] if splits is None or man["split_of"][k] in splits]
    if max_episodes and len(keys) > max_episodes:
        order = np.random.default_rng(seed).permutation(len(keys))[:max_episodes]
        keys = [keys[i] for i in sorted(order)]
    return keys


# ------------------------------------------------------------------------------------------ loading
@dataclass
class CacheData:
    """Stacked arrays of a set of episodes in physical units, padded with each episode's last row."""

    keys: list[str]
    z1: np.ndarray        # (N, T, 17) f32
    act: np.ndarray       # (N, T, 3) f32
    pose: np.ndarray      # (N, T, 3) f32
    power: np.ndarray     # (N, T, 1) f32 kW
    stalled: np.ndarray   # (N, T) bool
    hold_ok: np.ndarray   # (N, T) bool
    valid: np.ndarray     # (N, T) bool recorded frames
    n_valid: np.ndarray   # (N,) int
    domain: np.ndarray    # (N,) int 0 rigid / 1 soil
    group: list[str]
    split: list[str]

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


def load_cache(cache_dir, keys: list[str], max_frames: int = 0) -> CacheData:
    """Load ``keys``; episodes longer than ``max_frames`` (> 0) are cut."""
    cache_dir = resolve_path(cache_dir)
    man = read_manifest(cache_dir)
    cols = {n: [] for n in ("z1", "act", "pose", "power", "stalled", "hold_ok")}
    domain, lengths = [], []
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
                     hold_ok=stack(cols["hold_ok"]) & valid, valid=valid, n_valid=lengths,
                     domain=np.asarray(domain, np.int64), group=[man["group_of"][k] for k in keys],
                     split=[man["split_of"][k] for k in keys])


# ------------------------------------------------------------------------------------------ normalisation
@dataclass
class Normalizer:
    """Per-channel z-score statistics; the field order is the checkpoint's ``normalization`` dict (z2 is empty)."""

    z1_mean: np.ndarray
    z1_std: np.ndarray
    z2_mean: np.ndarray
    z2_std: np.ndarray
    act_mean: np.ndarray
    act_std: np.ndarray
    power_mean: np.ndarray
    power_std: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {k: np.asarray(v).astype(float).tolist() for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(payload: dict[str, list[float]]) -> "Normalizer":
        return Normalizer(**{k: np.asarray(v, dtype=np.float32) for k, v in payload.items()})


def fit_normalizer(data: CacheData, eps: float = 1e-6) -> Normalizer:
    """Statistics over the recorded frames of the training set (both domains pooled)."""
    m = data.valid.reshape(-1)
    stats = lambda a: (a.reshape(-1, a.shape[-1])[m].mean(0), np.maximum(a.reshape(-1, a.shape[-1])[m].std(0), eps))
    z1_mean, z1_std = stats(data.z1)
    act_mean, act_std = stats(data.act)
    p_mean, p_std = stats(data.power)
    return Normalizer(z1_mean.astype(np.float32), z1_std.astype(np.float32), np.zeros(0, np.float32),
                      np.ones(0, np.float32), act_mean.astype(np.float32), act_std.astype(np.float32),
                      p_mean.astype(np.float32), p_std.astype(np.float32))


# ------------------------------------------------------------------------------------------ batching
class Batcher:
    """Normalised windows over one split (arrays on the host); windows lie inside the recorded frames."""

    def __init__(self, data: CacheData, norm: Normalizer, context: int):
        self.context = context
        self.z1 = ((data.z1 - norm.z1_mean) / norm.z1_std).astype(np.float32)
        self.act = ((data.act - norm.act_mean) / norm.act_std).astype(np.float32)
        self.power = ((data.power - norm.power_mean) / norm.power_std).astype(np.float32)
        self.pose = data.pose.astype(np.float32)
        self.hold = data.hold_ok
        self.stalled = data.stalled
        self.valid = data.valid
        self.n_valid = data.n_valid.astype(np.int64)
        self.domain = data.domain.astype(np.int64)
        self.n_episodes, self.n_frames = data.z1.shape[0], data.z1.shape[1]
        self.dom_eps = {d: np.nonzero(self.domain == d)[0] for d in range(len(DOMAIN_VOCAB))}

    def sample_anchors(self, rng, batch: int, L: int, domain_frac: float) -> tuple[np.ndarray, np.ndarray]:
        """``round(domain_frac * batch)`` soil windows then the rest rigid; episodes drawn in proportion to their
        number of windows, the start frame uniform within the episode."""
        n_win = np.maximum(self.n_valid - L + 1, 0)
        avail = {d: eps[n_win[eps] > 0] for d, eps in self.dom_eps.items()}
        n_dom = sum(1 for d in avail if len(avail[d]))
        if n_dom == 0:
            raise ValueError("no episode offers a window of length %d" % L)
        counts = {}
        if n_dom == 1:
            d = next(d for d in avail if len(avail[d]))
            counts[d] = batch
        else:
            counts[1] = int(round(domain_frac * batch))
            counts[0] = batch - counts[1]
        ep_all, t0_all = [], []
        for d, n in counts.items():
            if n <= 0:
                continue
            eps = avail[d]
            p = n_win[eps] / n_win[eps].sum()
            ep = eps[rng.choice(len(eps), n, p=p)]
            t0 = np.minimum((rng.random(n) * n_win[ep]).astype(np.int64), n_win[ep] - 1)
            ep_all.append(ep); t0_all.append(t0)
        return np.concatenate(ep_all), np.concatenate(t0_all)

    def window_batch(self, ep: np.ndarray, t0: np.ndarray, L: int, device) -> dict:
        idx = t0[:, None] + np.arange(L)[None, :]
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device, non_blocking=True)
        return {"z1": to(self.z1[ep[:, None], idx]), "act": to(self.act[ep[:, None], idx]),
                "power": to(self.power[ep[:, None], idx]), "pose": to(self.pose[ep[:, None], idx]),
                "hold": to(self.hold[ep[:, None], idx].astype(np.float32)),
                "domain": to(self.domain[ep]), "ep": ep, "t0": t0}

    def sample(self, rng, batch: int, device, extra_steps: int = 1, domain_frac: float = 0.5) -> dict:
        L = self.context + extra_steps
        ep, t0 = self.sample_anchors(rng, batch, L, domain_frac)
        return self.window_batch(ep, t0, L, device)


def domain_delta_weights(train: Batcher) -> np.ndarray:
    """Per-channel loss weights w_i = mean_j(s_j) / s_i, mean-normalised to 1, with s_i^2 the mean over domains of
    the per-domain variance of the normalised one-step delta: a channel whose delta scale is set by one domain does
    not dominate the loss in the other."""
    var = []
    for d, eps in train.dom_eps.items():
        if len(eps) == 0:
            continue
        dz = np.diff(train.z1[eps], axis=1)
        dm = train.valid[eps][:, 1:] & train.valid[eps][:, :-1]
        if dm.sum() < 2:
            continue
        var.append(dz[dm].reshape(-1, train.z1.shape[-1]).var(0))
    s = np.sqrt(np.mean(np.stack(var), 0)) + 1e-6
    w = s.mean() / s
    return (w * (w.size / w.sum())).astype(np.float32)
