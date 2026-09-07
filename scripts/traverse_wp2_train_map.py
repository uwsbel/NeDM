"""WP2 spatial-token runs: ego-crop of the static scene map instead of pooled z2.

Same backbone, data, split and budget as the G3b triad; only the sensor token
changes. Two modes, and the comparison between them is the experiment:

  index    the scene map is encoded once and CROPPED at the dead-reckoned pose
           each step. Nothing is predicted forward, so there is no latent drift
           -- the failure mode G4 measured simply cannot occur.
  predict  a head predicts the next crop and it is fed back autoregressively,
           preserving the "NRD rolls out its sensor latent" claim.

index is the strong baseline (the layout is static, so indexing is exact). If
predict cannot beat it, the prediction branch is not earning its place here --
the same question the z2 persistence test asked of the pooled token, which
predict won at 1 s.

Multi-arena data (plan §28, 2026-09-06): ``--split-by arena`` takes schema-v2 caches
(``traverse_wp7_build_cache.py``: variable-length episodes incl. stalls / rollovers, one
arena per episode) and splits by TERRAIN INSTANCE (``--val-arenas`` / ``--test-arenas``),
never by episode; every crop takes its episode's own height field from a bank of arenas;
window sampling, losses and rollout metrics honour the recorded-frame mask.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nedm.traverse import nrd_data as D
from nedm.traverse.nrd_model import WP2MapModel  # noqa: F401 (checkpoint class)
from nedm.traverse.terrain import TerrainMap
from nedm.training.model_transformer import ContinuousTransformer, TransformerConfig
from traverse_wp2_train import DT_S, POSE_CHANNELS, TERRAIN_CHANNELS, integrate_pose


class MapBatcher:
    """Windows over one split. ``heightmaps`` (A, 1, H, W) on ``device`` is the crop's height-field bank;
    ``split.arena_idx`` picks each episode's row. Only windows inside the recorded frames are sampled."""

    EVENT_KINDS = ("approach", "stuck", "launch", "recovery", "matched")

    def __init__(self, split: D.CacheSplit, norm: D.Normalizer, context: int, maps: np.ndarray,
                 heightmaps: torch.Tensor, events: dict | None = None):
        self.context = context
        self.keys = list(split.keys)
        self.events = {k: events[k] for k in self.keys if events and k in events}  # key -> events.json record
        self._event_tables: dict[int, dict] = {}
        self.z1 = ((split.z1 - norm.z1_mean) / norm.z1_std).astype(np.float32)
        self.act = ((split.act - norm.act_mean) / norm.act_std).astype(np.float32)
        self.power = ((split.power - norm.power_mean) / norm.power_std).astype(np.float32)
        self.power_raw = split.power.astype(np.float32)
        self.pose = split.pose.astype(np.float32)
        self.maps = maps  # (N, C, 64, 64) float16
        self.n_episodes, self.n_frames = split.z1.shape[0], split.z1.shape[1]
        self.n_valid = split.n_valid
        self.valid = split.valid_mask()
        self.arena_idx = split.arena_idx if split.arena_idx is not None else np.zeros(self.n_episodes, np.int64)
        self.arena_ids = list(split.arena_ids) or ["arena"]
        self.heightmaps = heightmaps  # (A, 1, H, W) on device
        self.status = list(split.status)

    def event_table(self, extra_steps: int) -> dict:
        """Per event kind, the (episode, t0_lo, t0_hi) ranges of window anchors whose rollout span (frames
        context .. context+K-1 after t0) contains the event -- stalled-episode events from
        ``traverse_wp7_stall_diagnosis.py events``: 'approach' = the stop inside the span, 'stuck' = context and
        span after the stop (stationary, throttle on), 'launch' = the first frames of a launch failure,
        'recovery' = the resume frame inside the span, 'matched' = a feasible sibling passing the stalled runs'
        stop station inside the span."""
        K = extra_steps
        if K in self._event_tables:
            return self._event_tables[K]
        L = self.context + K
        tab = {kind: [] for kind in self.EVENT_KINDS}
        key_idx = {k: i for i, k in enumerate(self.keys)}
        for k, ev in self.events.items():
            i = key_idx[k]; n = int(self.n_valid[i])
            if n < L:
                continue
            span = lambda f: (max(0, f - self.context - K + 1), min(f - self.context, n - L))  # anchors with frame f in the span
            if ev.get("stop") is not None:
                lo, hi = span(int(ev["stop"]))
                if hi >= lo:
                    tab["approach"].append((i, lo, hi))
                lo, hi = int(ev["stop"]), n - L
                if hi >= lo:
                    tab["stuck"].append((i, lo, hi))
            if ev.get("launch"):
                tab["launch"].append((i, 0, max(0, min(40, n - L))))
            for f in ev.get("resume", []):
                lo, hi = span(int(f))
                if hi >= lo:
                    tab["recovery"].append((i, lo, hi))
            for f in ev.get("matched", []):
                lo, hi = span(int(f))
                if hi >= lo:
                    tab["matched"].append((i, lo, hi))
        out = {}
        for kind, ranges in tab.items():
            if ranges:
                a = np.asarray(ranges, np.int64)
                out[kind] = (a, (a[:, 2] - a[:, 1] + 1).astype(np.float64))
        self._event_tables[K] = out
        return out

    def sample_event_anchors(self, rng, n: int, extra_steps: int, kinds=None) -> tuple[np.ndarray, np.ndarray]:
        """``n`` (episode, t0) anchors, balanced over the event kinds present (uniform over windows within a kind)."""
        tab = self.event_table(extra_steps)
        kinds = [k for k in (kinds or self.EVENT_KINDS) if k in tab]
        if not kinds or n <= 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        which = rng.integers(0, len(kinds), n)
        ep, t0 = np.zeros(n, np.int64), np.zeros(n, np.int64)
        for ki, kind in enumerate(kinds):
            m = which == ki
            if not m.any():
                continue
            a, w = tab[kind]
            r = rng.choice(len(a), int(m.sum()), p=w / w.sum())
            ep[m] = a[r, 0]
            t0[m] = a[r, 1] + (rng.random(int(m.sum())) * w[r]).astype(np.int64)
        return ep, t0

    def sample(self, rng, batch: int, device: str, extra_steps: int = 1, event_frac: float = 0.0, event_kinds=None):
        """``extra_steps`` frames after the context window (1 = one-step training; K for a
        K-step autoregressive rollout loss). Windows lie inside the recorded frames; an episode is
        drawn in proportion to the windows it offers (uniform over windows, as with uniform episodes).
        ``event_frac`` of the batch is drawn from the stall-event windows instead (event_table)."""
        L = self.context + extra_steps
        n_ev = int(round(event_frac * batch)) if self.events else 0
        n_win = np.maximum(self.n_valid - L + 1, 0)
        p = n_win / n_win.sum()
        ep = rng.choice(self.n_episodes, batch - n_ev, p=p)
        t0 = np.minimum((rng.random(batch - n_ev) * n_win[ep]).astype(np.int64), n_win[ep] - 1)
        if n_ev:
            ep_e, t0_e = self.sample_event_anchors(rng, n_ev, extra_steps, event_kinds)
            ep, t0 = np.concatenate([ep, ep_e]), np.concatenate([t0, t0_e])
        idx = t0[:, None] + np.arange(L)[None, :]
        out = {}
        for name, src in (("z1", self.z1), ("act", self.act), ("power", self.power),
                          ("pose", self.pose)):
            out[name] = torch.from_numpy(src[ep[:, None], idx]).to(device, non_blocking=True)
        out["map"] = torch.from_numpy(self.maps[ep]).to(device, non_blocking=True).float()
        out["hm"] = self.heightmaps[torch.from_numpy(self.arena_idx[ep]).to(device)]
        return out


def rollout_loss(model, batch, context: int, steps: int, z1_mean, z1_std, w=None, progress_weight: float = 0.0, context_noise: float = 0.0):
    """K-step autoregressive loss under the RECORDED actions: predicted state fed back, map
    re-cropped at the dead-reckoned pose (exactly the imagination env's step). Targets the
    closed-loop speed bias that one-step teacher forcing does not see."""
    z1, act, pose_gt, maps, hm = batch["z1"], batch["act"], batch["pose"], batch["map"], batch["hm"]
    token_hist = model.cropper(maps, pose_gt[:, :context], hm)
    z1_hist = z1[:, :context]
    if context_noise > 0:
        z1_hist = z1_hist + context_noise * torch.randn_like(z1_hist)
    pose = pose_gt[:, context - 1]
    l_z1 = l_p = l_prog = 0.0
    cum_pred = cum_rec = 0.0
    for step in range(steps):
        window = slice(step, step + context)
        delta, power, _ = model(z1_hist[:, -context:], token_hist[:, -context:], act[:, window])
        z1_next = z1_hist[:, -1] + delta[:, -1]
        tgt = z1[:, context + step]
        l_z1 = l_z1 + (F.huber_loss(z1_next * w, tgt * w, delta=1.0) if w is not None else F.huber_loss(z1_next, tgt, delta=1.0))
        l_p = l_p + F.huber_loss(power[:, -1], batch["power"][:, context + step], delta=1.0)
        if progress_weight > 0:  # distance travelled along the body x axis: the quantity a stall zeroes and a drift inflates
            cum_pred = cum_pred + (z1_next[:, 0] * z1_std[0] + z1_mean[0]) * DT_S
            cum_rec = cum_rec + (tgt[:, 0] * z1_std[0] + z1_mean[0]) * DT_S
            l_prog = l_prog + F.huber_loss(cum_pred, cum_rec, delta=1.0)  # metres
        pose = integrate_pose(pose, z1_next * z1_std + z1_mean)
        nxt = model.cropper(maps, pose.unsqueeze(1), hm)[:, 0]
        z1_hist = torch.cat([z1_hist, z1_next.unsqueeze(1)], dim=1)
        token_hist = torch.cat([token_hist, nxt.unsqueeze(1)], dim=1)
    parts = {"ro_z1": float(l_z1.detach()) / steps, "ro_power": float(l_p.detach()) / steps}
    total = (l_z1 + l_p) / steps
    if progress_weight > 0:
        total = total + progress_weight * l_prog / steps
        parts["ro_progress_m"] = float(l_prog.detach()) / steps
    return total, parts


def step_loss(model, batch, mode: str, w=None, context_noise: float = 0.0):
    """``w`` (z1_dim,) rescales the state channels in the loss. With --delta-scale it is the inverse
    per-step delta std, so slowly varying channels (vx: one-step change ~0.03 of the state std) get the
    same weight as the noisy tire channels instead of being ignored -- the 10 % speed bias lives there."""
    token = model.cropper(batch["map"], batch["pose"], batch["hm"])
    z_in = batch["z1"][:, :-1]
    if context_noise > 0:
        z_in = z_in + context_noise * torch.randn_like(z_in)
    delta, power, token_next = model(z_in, token[:, :-1], batch["act"][:, :-1])
    tgt = batch["z1"][:, 1:] - z_in  # the delta that brings the (noisy) input to the clean next state
    loss = F.huber_loss(delta * w, tgt * w, delta=1.0) if w is not None else F.huber_loss(delta, tgt, delta=1.0)
    parts = {"z1": float(loss.detach())}
    lp = F.huber_loss(power, batch["power"][:, 1:], delta=1.0)
    loss = loss + lp
    parts["power"] = float(lp.detach())
    if mode == "predict" and token_next is not None:
        target = ((token[:, 1:] - model.tok_mean) / model.tok_std).detach()
        lt = F.mse_loss(token_next, target)
        loss = loss + lt
        parts["token"] = float(lt.detach())
    return loss, parts


@torch.no_grad()
def fit_token_stats(model, data: MapBatcher, batch: int, n_batches: int, device: str,
                    seed: int) -> None:
    rng = np.random.default_rng(seed)
    crops = []
    for _ in range(n_batches):
        b = data.sample(rng, batch, device)
        crops.append(model.cropper(b["map"], b["pose"], b["hm"]).reshape(-1, model.token_dim))
    t = torch.cat(crops)
    model.tok_mean.copy_(t.mean(0))
    model.tok_std.copy_(t.std(0).clamp_min(1e-3))
    print(f"token stats over {t.shape[0]} crops: mean|.|={float(model.tok_mean.abs().mean()):.4f} "
          f"std={float(model.tok_std.mean()):.4f}", flush=True)


@torch.no_grad()
def rollout_eval(model, data: MapBatcher, norm, mode: str, context: int, horizons: list[int],
                 n_episodes: int, device: str, seed: int = 7,
                 crop_pose: str = "deadreckon", per_arena: bool = False) -> dict:
    """crop_pose: where the index crop is taken during rollout. "deadreckon" is the
    honest setting (pose integrated from predicted z1); "gt" reads the map at the
    true pose and isolates how much long-horizon error is *reading the wrong place*
    rather than the token lacking information (the pose-drift test).

    Episodes shorter than context + h are excluded from the horizon-h means (recorded-frame mask)."""
    model.eval()
    eligible = np.nonzero(data.n_valid >= context + min(horizons))[0]
    eps = np.random.default_rng(seed).choice(eligible, size=min(n_episodes, len(eligible)), replace=False)
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    z1_gt, act = to_t(data.z1[eps]), to_t(data.act[eps])
    pose_gt, maps = to_t(data.pose[eps]), to_t(data.maps[eps]).float()
    hm = data.heightmaps[torch.from_numpy(data.arena_idx[eps]).to(device)]
    n_valid = torch.from_numpy(data.n_valid[eps]).to(device)
    arena_of = data.arena_idx[eps]
    power_gt = to_t(data.power_raw[eps])[..., 0]
    z1_mean, z1_std = to_t(norm.z1_mean.astype(np.float32)), to_t(norm.z1_std.astype(np.float32))
    p_mean, p_std = float(norm.power_mean[0]), float(norm.power_std[0])

    z1_hist = z1_gt[:, :context].clone()
    pose_hist = pose_gt[:, :context].clone()
    token_hist = model.cropper(maps, pose_hist, hm)
    pose = pose_gt[:, context - 1].clone()
    cv_pose = pose_gt[:, context - 1].clone()
    cv_state = z1_gt[:, context - 1] * z1_std + z1_mean
    e_pred = torch.zeros(len(eps), device=device)
    e_gt = torch.zeros(len(eps), device=device)
    results: dict[str, float] = {}
    mmean = lambda v, m: float(v[m].mean()) if int(m.sum()) else float("nan")

    for step in range(max(horizons)):
        window = slice(step, step + context)
        delta, power, token_next = model(z1_hist[:, -context:], token_hist[:, -context:],
                                         act[:, window])
        z1_next = z1_hist[:, -1] + delta[:, -1]
        z1_hist = torch.cat([z1_hist, z1_next.unsqueeze(1)], dim=1)
        pose = integrate_pose(pose, z1_next * z1_std + z1_mean)
        cv_pose = integrate_pose(cv_pose, cv_state)
        # index: re-crop at the pose we just dead-reckoned (or the true pose for
        # the pose-drift test). predict: the head's normalized output, de-normalized.
        if mode == "index":
            crop_at = pose_gt[:, context + step] if crop_pose == "gt" else pose
            nxt = model.cropper(maps, crop_at.unsqueeze(1), hm)[:, 0]
        else:
            nxt = token_next[:, -1] * model.tok_std + model.tok_mean
        token_hist = torch.cat([token_hist, nxt.unsqueeze(1)], dim=1)

        live = n_valid > context + step  # the target frame was recorded
        kw = power[:, -1, 0] * p_std + p_mean
        e_pred = e_pred + torch.where(live, kw * DT_S, torch.zeros_like(kw))
        e_gt = e_gt + torch.where(live, power_gt[:, context + step] * DT_S, torch.zeros_like(kw))
        h = step + 1
        if h in horizons:
            frame = context - 1 + h
            m = n_valid > frame
            gt = pose_gt[:, frame]
            err = (z1_hist[:, frame] - z1_gt[:, frame]).abs()
            yaw_err = torch.rad2deg(torch.abs((pose[:, 2] - gt[:, 2] + math.pi) % (2 * math.pi) - math.pi))
            results[f"z1_mae_norm@{h}"] = mmean(err.mean(1), m)
            results[f"terrain_mae_norm@{h}"] = mmean(err[:, TERRAIN_CHANNELS].mean(1), m)
            results[f"pose_chan_mae_norm@{h}"] = mmean(err[:, POSE_CHANNELS].mean(1), m)
            results[f"pose_err_m@{h}"] = mmean((pose[:, :2] - gt[:, :2]).norm(dim=1), m)
            results[f"cv_pose_err_m@{h}"] = mmean((cv_pose[:, :2] - gt[:, :2]).norm(dim=1), m)
            results[f"yaw_err_deg@{h}"] = mmean(yaw_err, m)
            results[f"energy_err_kj@{h}"] = mmean((e_pred - e_gt).abs(), m)
            results[f"n@{h}"] = int(m.sum())
            if mode == "predict":
                true_token = model.cropper(maps, pose_gt[:, frame].unsqueeze(1), hm)[:, 0]
                results[f"token_cos@{h}"] = mmean(F.cosine_similarity(token_hist[:, frame], true_token, dim=-1), m)
            if per_arena and len(data.arena_ids) > 1:
                for ai, aid in enumerate(data.arena_ids):
                    ma = m & torch.from_numpy(arena_of == ai).to(device)
                    if int(ma.sum()):
                        results[f"z1_mae_norm@{h}/{aid}"] = mmean(err.mean(1), ma)
    model.train()
    return results


@torch.no_grad()
def stall_eval(model, data: MapBatcher, norm, context: int, device: str, max_per_kind: int = 128, seed: int = 11) -> dict:
    """Stall reproduction on the split's event windows under the RECORDED controls (the §12.2 tests, in-training):
    stuck (context 0.2 s after the stop, 3 s): predicted |vx| at the end (Chrono ~0); approach (context ends 2 s
    before the stop, 4 s): predicted vx 2 s after the stop; launch (frames 0-15, 3 s): predicted vx at 3 s (Chrono
    ~0); recovery (context ends 1 s before the resume, 3 s) and matched (a feasible sibling at the stalled runs' stop
    station, 4 s): |predicted - recorded| vx at the end -- the guards against 'always stop'. stall_score = mean."""
    model.eval()
    specs = {"stuck": (lambda ev: [int(ev["stop"]) + 4] if ev.get("stop") is not None else [], 60, "abs"),
             "approach": (lambda ev: [int(ev["stop"]) - context - 40] if ev.get("stop") is not None else [], 80, "abs"),
             "launch": (lambda ev: [0] if ev.get("launch") else [], 60, "abs"),
             "recovery": (lambda ev: [int(f) - context - 20 for f in ev.get("resume", [])], 60, "err"),
             "matched": (lambda ev: [int(f) - context - 40 for f in ev.get("matched", [])], 80, "err")}
    rng = np.random.default_rng(seed)
    key_idx = {k: i for i, k in enumerate(data.keys)}
    z1_mean, z1_std = float(norm.z1_mean[0]), float(norm.z1_std[0])
    out, scores = {}, []
    for kind, (anchors_of, K, mode) in specs.items():
        cand = []
        for k, ev in data.events.items():
            i = key_idx[k]
            for t0 in anchors_of(ev):
                if 0 <= t0 and t0 + context + K <= data.n_valid[i]:
                    cand.append((i, t0))
        if not cand:
            continue
        cand = np.asarray(cand)[rng.permutation(len(cand))[:max_per_kind]]
        ep, t0 = cand[:, 0], cand[:, 1]
        idx = t0[:, None] + np.arange(context + K)[None, :]
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        z1, act, pose_gt = to_t(data.z1[ep[:, None], idx]), to_t(data.act[ep[:, None], idx]), to_t(data.pose[ep[:, None], idx])
        maps = to_t(data.maps[ep]).float(); hm = data.heightmaps[torch.from_numpy(data.arena_idx[ep]).to(device)]
        z1_hist = z1[:, :context]; token_hist = model.cropper(maps, pose_gt[:, :context], hm); pose = pose_gt[:, context - 1]
        z1_mean_t = torch.tensor(norm.z1_mean.astype(np.float32), device=device); z1_std_t = torch.tensor(norm.z1_std.astype(np.float32), device=device)
        for step in range(K):
            delta, _, _ = model(z1_hist[:, -context:], token_hist[:, -context:], act[:, step:step + context])
            z1_next = z1_hist[:, -1] + delta[:, -1]
            pose = integrate_pose(pose, z1_next * z1_std_t + z1_mean_t)
            nxt = model.cropper(maps, pose.unsqueeze(1), hm)[:, 0]
            z1_hist = torch.cat([z1_hist, z1_next.unsqueeze(1)], dim=1); token_hist = torch.cat([token_hist, nxt.unsqueeze(1)], dim=1)
        pred = z1_hist[:, -1, 0] * z1_std + z1_mean; rec = z1[:, -1, 0] * z1_std + z1_mean
        val = float(pred.abs().mean()) if mode == "abs" else float((pred - rec).abs().mean())
        out[f"stall_{kind}_vx"] = val; out[f"stall_{kind}_rec"] = float(rec.mean()); out[f"stall_{kind}_n"] = int(len(ep))
        if mode == "abs":
            out[f"stall_{kind}_hold"] = float((pred.abs() < 0.5).float().mean())
        scores.append(val)
    if scores:
        out["stall_score"] = float(np.mean(scores))
    model.train()
    return out


def load_maps(cache: Path, keys: list[str], key: str = "map") -> np.ndarray:
    return np.stack([np.load(cache / f"{k}.npz")[key] for k in keys])


def load_cache_with_maps(cache: Path, keys: list[str], map_key: str, base_cache: Path | None,
                         n_frames: int | None, arena_id: str | None) -> tuple[D.CacheSplit, np.ndarray]:
    """Episodes + scene maps of one cache. Maps come from the files themselves (``map_key``) or, for
    tracker-driven caches that reuse a recorded layout, from ``base_cache`` via each file's ``source_key``."""
    split = D.load_split(cache, keys, with_z2=False, n_frames=n_frames, arena_id=arena_id)
    with np.load(cache / f"{keys[0]}.npz") as probe:
        own = map_key in probe.files
    if own:
        maps = load_maps(cache, keys, map_key)
    else:
        src = [str(np.load(cache / f"{k}.npz")["source_key"]) for k in keys]
        maps = load_maps(base_cache, src, map_key)
    return split, maps


def heightmap_bank(arena_dirs: dict[str, Path], device: str) -> torch.Tensor:
    """(A, 1, H, W) crop height fields, rows in ``sorted(arena_dirs)`` order (= CacheSplit.arena_ids)."""
    grids = [torch.tensor(TerrainMap.from_dir(Path(arena_dirs[a])).height_grid, dtype=torch.float32) for a in sorted(arena_dirs)]
    return torch.stack(grids)[:, None].to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="", help="schema-v1 base cache (episode split, one arena)")
    ap.add_argument("--caches", nargs="*", default=[],
                    help="schema-v2 multi-arena caches (traverse_wp7_build_cache.py); used with --split-by arena")
    ap.add_argument("--split-by", choices=["episode", "arena"], default="episode",
                    help="arena: hold out whole terrain instances (--val-arenas / --test-arenas), never episodes")
    ap.add_argument("--val-arenas", nargs="*", default=[])
    ap.add_argument("--test-arenas", nargs="*", default=[], help="sealed: loaded by nothing here")
    ap.add_argument("--n-frames", type=int, default=0, help="pad variable-length episodes to this many frames (0: longest)")
    ap.add_argument("--arena", default="assets/traverse/arena_v1", help="arena of schema-v1 caches / the model's default crop map")
    ap.add_argument("--map-mode", choices=["index", "predict"], required=True)
    ap.add_argument("--map-key", default="map")
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--token-dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-episodes", type=int, default=256)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 40, 100])
    ap.add_argument("--selection", default="z1_mae_norm")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--max-train-episodes", type=int, default=0)
    ap.add_argument("--rollout-steps", type=int, default=0,
                    help="K > 0 adds a K-step autoregressive rollout loss (state fed back, map re-cropped)")
    ap.add_argument("--rollout-weight", type=float, default=1.0)
    ap.add_argument("--progress-weight", type=float, default=0.0,
                    help="adds a Huber loss on the cumulative distance (m) along the rollout: penalises the speed drift a stall exposes")
    ap.add_argument("--vx-weight", type=float, default=1.0, help="extra weight on the vx channel in the state losses")
    ap.add_argument("--context-noise", type=float, default=0.0,
                    help="std of Gaussian noise added to the normalised state context in training (targets clean): robustness to the model's own errors in closed loop")
    ap.add_argument("--events", nargs="*", default=["auto"],
                    help="events.json files (traverse_wp7_stall_diagnosis.py events); 'auto' = <cache>/events.json of every --caches dir")
    ap.add_argument("--event-frac", type=float, default=0.0,
                    help="fraction of each batch drawn from stall-event windows (approach / stuck / launch / recovery / matched), balanced over kinds")
    ap.add_argument("--event-kinds", nargs="*", default=None)
    ap.add_argument("--extra-train-cache", nargs="*", default=[],
                    help="extra cache dirs appended to the TRAIN split only (e.g. tracker-driven Chrono "
                         "episodes on arena_v1); scene maps from the files or from --cache via source_key")
    ap.add_argument("--z1-extra-cache", default="",
                    help="sidecar dir (traverse_wp5_build_z1_sidecar.py) whose z1_extra channels are appended to "
                         "15-D z1 caches (e.g. engine speed + motorshaft torque -> 17-D state)")
    ap.add_argument("--delta-scale", action="store_true",
                    help="weight each z1 channel in the state losses by 1/std of its normalized one-step delta")
    ap.add_argument("--init-from", default="",
                    help="ckpt_best.pt of a finished run to start from (two-stage predict / fine-tune / eval-only)")
    ap.add_argument("--freeze-cropper", action="store_true",
                    help="freeze the map projection so the prediction target is stationary")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="also freeze backbone + z1/power heads: only the token head trains")
    ap.add_argument("--token-stat-batches", type=int, default=64)
    ap.add_argument("--eval-only", action="store_true",
                    help="no training: roll out --init-from under both crop-pose settings "
                         "and write posedrift_readout.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = args.n_frames or None
    arena_dirs: dict[str, Path] = {}

    t0 = time.time()
    base_cache = Path(args.cache) if args.cache else None
    if args.split_by == "arena":
        if not args.caches:
            raise SystemExit("--split-by arena needs --caches")
        parts = []
        for c in args.caches:
            c = Path(c)
            manifest = json.loads((c / "cache_manifest.json").read_text())
            arena_dirs.update({k: Path(v) for k, v in manifest.get("arenas", {}).items()})
            keys = D.load_cache_keys(c)
            sp, mp = load_cache_with_maps(c, keys, args.map_key, base_cache, n_frames, None)
            parts.append((sp, mp))
        split_all, maps_all = parts[0]
        for sp, mp in parts[1:]:
            split_all, maps_all = D.concat_splits(split_all, sp), np.concatenate([maps_all, mp])
        unknown = (set(args.val_arenas) | set(args.test_arenas)) - set(split_all.arena_ids)
        if unknown:
            raise SystemExit(f"unknown arenas {sorted(unknown)}; caches hold {split_all.arena_ids}")
        name_of = np.asarray(split_all.arena_ids)[split_all.arena_idx]
        is_val = np.isin(name_of, args.val_arenas); is_test = np.isin(name_of, args.test_arenas)
        sel = lambda m: D.CacheSplit(keys=[k for k, f in zip(split_all.keys, m) if f], z1=split_all.z1[m], z2=split_all.z2[m], act=split_all.act[m],
                                     pose=split_all.pose[m], power=split_all.power[m], terrain=split_all.terrain[m],
                                     valid=None if split_all.valid is None else split_all.valid[m], arena_idx=split_all.arena_idx[m],
                                     arena_ids=split_all.arena_ids, status=[s for s, f in zip(split_all.status, m) if f])
        train_split, val_split = sel(~is_val & ~is_test), sel(is_val)
        train_maps, val_maps = maps_all[~is_val & ~is_test], maps_all[is_val]
        train_keys, val_keys, test_keys = train_split.keys, val_split.keys, [k for k, f in zip(split_all.keys, is_test) if f]
        cnt = lambda sp: {a: int((sp.arena_idx == i).sum()) for i, a in enumerate(sp.arena_ids) if int((sp.arena_idx == i).sum())}
        print(f"split by arena: train {len(train_keys)} {cnt(train_split)} | val {len(val_keys)} {cnt(val_split)} | test {len(test_keys)} (untouched)", flush=True)
        st = lambda sp: {s: int(sum(x == s for x in sp.status)) for s in sorted(set(sp.status))}
        print(f"  episode outcomes: train {st(train_split)} val {st(val_split)}", flush=True)
    else:
        cache = base_cache
        keys = D.load_cache_keys(cache)
        train_keys, val_keys, test_keys = D.split_keys(keys)
        if args.max_train_episodes:
            train_keys = train_keys[: args.max_train_episodes]
            val_keys = val_keys[: max(args.eval_episodes, 8)]
        print(f"split: {len(train_keys)}/{len(val_keys)}/{len(test_keys)} (test untouched)", flush=True)
        train_split = D.load_split(cache, train_keys, with_z2=False, n_frames=n_frames)
        val_split = D.load_split(cache, val_keys, with_z2=False, n_frames=n_frames)
        train_maps = load_maps(cache, train_keys, args.map_key)
        val_maps = load_maps(cache, val_keys, args.map_key)
        if args.z1_extra_cache:
            train_split = D.with_z1_extra(train_split, Path(args.z1_extra_cache))
            val_split = D.with_z1_extra(val_split, Path(args.z1_extra_cache))
            print(f"z1 extra channels from {args.z1_extra_cache}: z1 is now {train_split.z1.shape[-1]}-D", flush=True)
        arena_dirs[train_split.arena_ids[0]] = Path(args.arena)
    for extra in args.extra_train_cache:
        extra = Path(extra)
        ekeys = D.load_cache_keys(extra)
        esplit, emaps = load_cache_with_maps(extra, ekeys, args.map_key, base_cache, train_split.n_frames, None)
        if esplit.z1.shape[-1] == 15 and train_split.z1.shape[-1] == 17 and args.z1_extra_cache:
            esplit = D.with_z1_extra(esplit, Path(args.z1_extra_cache))
        if esplit.z1.shape[-1] != train_split.z1.shape[-1]:
            raise ValueError(f"{extra}: z1 is {esplit.z1.shape[-1]}-D but the training data is "
                             f"{train_split.z1.shape[-1]}-D; collect it with the matching --preset")
        for a in esplit.arena_ids:
            arena_dirs.setdefault(a, Path(args.arena))
        train_split = D.concat_splits(train_split, esplit)
        train_maps = np.concatenate([train_maps, emaps])
        print(f"extra train cache {extra}: +{len(ekeys)} episodes (val/test untouched)", flush=True)
    for a in set(train_split.arena_ids) | set(val_split.arena_ids):
        arena_dirs.setdefault(a, Path(args.arena))
    # the val split must index the same bank rows as the train split
    all_ids = sorted(arena_dirs)
    remap = lambda sp: np.asarray([all_ids.index(sp.arena_ids[i]) for i in sp.arena_idx], np.int64) if sp.arena_idx is not None else np.zeros(sp.n_episodes, np.int64)
    train_split.arena_idx, train_split.arena_ids = remap(train_split), all_ids
    val_split.arena_idx, val_split.arena_ids = remap(val_split), all_ids
    hm_bank = heightmap_bank(arena_dirs, device)
    print(f"loaded cache + maps in {time.time() - t0:.1f}s  maps {train_maps.shape}  frames {train_split.n_frames}  "
          f"height-field bank {tuple(hm_bank.shape)} for {all_ids}", flush=True)
    if train_split.valid is not None:
        nv = train_split.n_valid
        print(f"  recorded frames per episode: mean {nv.mean():.0f} min {nv.min()} max {nv.max()} "
              f"({int((nv < args.context + 1).sum())} too short for one window)", flush=True)

    payload = torch.load(args.init_from, map_location="cpu") if args.init_from else None
    norm = (D.Normalizer.from_dict(payload["normalization"]) if payload
            else D.Normalizer.fit(train_split))
    z1_dim, act_dim = train_split.z1.shape[-1], train_split.act.shape[-1]
    events: dict = {}
    ev_paths = [Path(c) / "events.json" for c in args.caches] if args.events == ["auto"] else [Path(p) for p in args.events]
    for pth in ev_paths:
        if pth.exists():
            events.update(json.loads(pth.read_text()))
    train_data = MapBatcher(train_split, norm, args.context, train_maps, hm_bank, events)
    val_data = MapBatcher(val_split, norm, args.context, val_maps, hm_bank, events)
    if events:
        K = max(args.rollout_steps, 1)
        cnt = {k: (len(v[0]), int(v[1].sum())) for k, v in train_data.event_table(K).items()}
        print(f"stall events: {len(train_data.events)} train / {len(val_data.events)} val episodes with records; "
              f"train event ranges (n, windows) at K={K}: {cnt}; event_frac {args.event_frac}", flush=True)
    elif args.event_frac > 0:
        raise SystemExit("--event-frac needs events.json (traverse_wp7_stall_diagnosis.py events)")
    loss_w = None
    if args.delta_scale:
        dz = np.diff(train_data.z1, axis=1)
        dm = train_data.valid[:, 1:] & train_data.valid[:, :-1]
        d_std = dz[dm].reshape(-1, z1_dim).std(0) + 1e-6
        loss_w = torch.tensor((d_std.mean() / d_std).astype(np.float32), device=device)  # mean weight 1
        print("delta-scale weights:", np.round(loss_w.cpu().numpy(), 2).tolist(), flush=True)
    if args.vx_weight != 1.0:
        loss_w = torch.ones(z1_dim, device=device) if loss_w is None else loss_w
        loss_w = loss_w.clone(); loss_w[0] = loss_w[0] * args.vx_weight
    del train_split, val_split

    cfg = {"block_size": args.context, "n_layer": args.n_layer, "n_head": args.n_head,
           "n_embd": args.n_embd, "dropout": args.dropout, "bias": False,
           "head_hidden_dim": args.n_embd}
    model = WP2MapModel(z1_dim, act_dim, cfg, Path(args.arena), args.token_dim,
                        predict_token=(args.map_mode == "predict")).to(device)
    if payload is not None:
        state = {k: v for k, v in payload["model"].items() if k != "cropper.heightmap"}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"init from {args.init_from} (step {payload.get('step')}) "
              f"missing={missing} unexpected={unexpected}", flush=True)
    if args.map_mode == "predict":
        fit_token_stats(model, train_data, args.batch, args.token_stat_batches, device,
                        args.seed + 2)
    if args.freeze_cropper:
        for p in model.cropper.parameters():
            p.requires_grad_(False)
    if args.freeze_backbone:
        for module in (model.backbone, model.state_head, model.power_head):
            for p in module.parameters():
                p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"map-mode={args.map_mode} token_dim={args.token_dim} params={n_params/1e6:.2f}M "
          f"trainable={n_train/1e6:.2f}M", flush=True)

    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        p = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(p, 1.0)))

    (out_dir / "config.json").write_text(json.dumps(
        {**vars(args), "model": cfg, "n_params": n_params,
         "split_counts": [len(train_keys), len(val_keys), len(test_keys)],
         "arena_ids": all_ids, "arena_dirs": {k: str(v) for k, v in arena_dirs.items()},
         "normalization": norm.to_dict()}, indent=2))
    if args.eval_only:
        assert payload is not None, "--eval-only needs --init-from"
        readout = {"init_from": args.init_from, "episodes": args.eval_episodes,
                   "map_mode": args.map_mode}
        for cp in ("deadreckon", "gt"):
            readout[cp] = rollout_eval(model, val_data, norm, args.map_mode, args.context,
                                       args.horizons, args.eval_episodes, device, crop_pose=cp, per_arena=True)
            print(cp, json.dumps(readout[cp]), flush=True)
        (out_dir / "posedrift_readout.json").write_text(json.dumps(readout, indent=2))
        return

    log = out_dir / "train_log.jsonl"; log.write_text("")
    rng = np.random.default_rng(args.seed)
    z1_mean_t = torch.tensor(norm.z1_mean.astype(np.float32), device=device)
    z1_std_t = torch.tensor(norm.z1_std.astype(np.float32), device=device)
    best = {"metric": float("inf"), "step": -1}
    start = time.time()

    def save(path: Path, step: int, m: dict) -> None:
        torch.save({"model": model.state_dict(), "config": cfg, "map_mode": args.map_mode,
                    "normalization": norm.to_dict(), "step": step, "metrics": m,
                    "z1_dim": z1_dim, "z1_extra_cache": args.z1_extra_cache, "arena_ids": all_ids,
                    "delta_scale": loss_w.cpu().tolist() if loss_w is not None else None}, path)

    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        if args.rollout_steps > 0:
            batch = train_data.sample(rng, args.batch, device, extra_steps=args.rollout_steps, event_frac=args.event_frac, event_kinds=args.event_kinds)
            one = {k: v[:, : args.context + 1] if v.dim() >= 2 and k not in ("map", "hm") else v for k, v in batch.items()}
            loss, parts = step_loss(model, one, args.map_mode, loss_w, args.context_noise)
            ro, ro_parts = rollout_loss(model, batch, args.context, args.rollout_steps, z1_mean_t, z1_std_t, loss_w, args.progress_weight, args.context_noise)
            loss = loss + args.rollout_weight * ro
            parts.update(ro_parts)
        else:
            loss, parts = step_loss(model, train_data.sample(rng, args.batch, device, event_frac=args.event_frac, event_kinds=args.event_kinds), args.map_mode, loss_w, args.context_noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if (step + 1) % 100 == 0:
            with log.open("a") as fh:
                fh.write(json.dumps({"phase": "train", "step": step + 1,
                                     "loss": float(loss.detach()), **parts,
                                     "sps": (step + 1) * args.batch / (time.time() - start)}) + "\n")
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            vr = np.random.default_rng(args.seed + 1)
            with torch.no_grad():
                vls = [float(step_loss(model, val_data.sample(vr, args.batch, device),
                                       args.map_mode, loss_w)[0]) for _ in range(args.val_batches)]
            m = rollout_eval(model, val_data, norm, args.map_mode, args.context,
                             args.horizons, args.eval_episodes, device, per_arena=True)
            if val_data.events:
                m.update(stall_eval(model, val_data, norm, args.context, device))
            sel = m[args.selection] if args.selection in m else m[f"{args.selection}@{max(args.horizons)}"]
            rec = {"phase": "val", "step": step + 1, "val_loss": float(np.mean(vls)),
                   "selection": sel, **m}
            with log.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            save(out_dir / "ckpt_last.pt", step + 1, m)
            if sel < best["metric"]:
                best = {"metric": sel, "step": step + 1, **m}
                save(out_dir / "ckpt_best.pt", step + 1, m)
    (out_dir / "g3_readout.json").write_text(json.dumps(
        {"variant": f"map-{args.map_mode}", "best": best, "wall_s": time.time() - start,
         "split_counts": [len(train_keys), len(val_keys), len(test_keys)]}, indent=2))
    print(f"done: best {args.selection} {best['metric']:.4f} @ step {best['step']}", flush=True)


if __name__ == "__main__":
    main()
