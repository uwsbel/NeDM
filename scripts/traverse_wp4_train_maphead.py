#!/usr/bin/env python
"""Planner-B map head: decode a planner-facing map from the frozen camera scene map.

Input  : the per-episode static scene feature map (64 x 64 x 64, encoder stage-2,
         image coordinates) that the dynamics model already indexes -- camera only.
Output : bird's-eye occupancy logits + normalized elevation on a world-aligned
         grid (default 128 x 128 over the 80 m arena = 0.625 m/cell, row 0 = north).
Labels : analytic footprint discs from the layout manifest (plan section 5) and
         the arena heightmap (identical for every episode -> memorized; kept for
         the ablation ladder's "full predicted map" rung).

The warp from image to world uses the flat-ground camera model only (no
heightmap); the convolutions absorb the residual perspective shift (< 1.2 m).

Usage (smoke):  PYTHONPATH=src python scripts/traverse_wp4_train_maphead.py --out artifacts/traverse/wp4_maphead_smoke --steps 100 --limit-train 400
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nedm.traverse import nrd_data as D
from nedm.traverse.camera import CameraModel
from nedm.traverse.masks import bev_occupancy
from nedm.traverse.terrain import TerrainMap

SMALL = ("rock", "tree")


class MapHead(nn.Module):
    """Fixed flat-ground warp (image -> world BEV) followed by a small conv decoder."""

    def __init__(self, arena_dir: Path, grid: int = 128, in_ch: int = 64, width: int = 64):
        super().__init__()
        tmap = TerrainMap.from_dir(arena_dir)
        cam = CameraModel()
        self.grid, self.size_m = grid, float(tmap.size_m)
        self.h_min, self.h_max = float(tmap.meta["height_min_m"]), float(tmap.meta["height_max_m"])
        half = self.size_m / 2
        xs = (torch.arange(grid) + 0.5) / grid * self.size_m - half
        ys = half - (torch.arange(grid) + 0.5) / grid * self.size_m  # row 0 = north
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        scale = cam.f_px / cam.cam_height_m  # flat ground, z = 0
        u = cam.cx + scale * gx
        v = cam.cy - scale * gy
        warp = torch.stack([(u + 0.5) * 2 / cam.width - 1, (v + 0.5) * 2 / cam.width - 1], -1)
        self.register_buffer("warp", warp[None])  # (1, G, G, 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(width, width // 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(width // 2, 2, 1))

    def forward(self, maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """maps (B, 64, 64, 64) -> occupancy logits (B, G, G), elevation in [0,1] (B, G, G)."""
        feats = F.grid_sample(maps.float(), self.warp.expand(maps.shape[0], -1, -1, -1), mode="bilinear",
                              padding_mode="border", align_corners=False)
        out = self.net(feats)
        return out[:, 0], out[:, 1]

    def elevation_m(self, elev01: torch.Tensor) -> torch.Tensor:
        return self.h_min + elev01.clamp(0, 1) * (self.h_max - self.h_min)


def load_maps_and_labels(cache: Path, stores: Path, keys: list[str], grid: int, map_key: str, size_m: float):
    maps = np.empty((len(keys), 64, 64, 64), np.float16)
    occ = np.zeros((len(keys), grid, grid), bool)
    small = np.zeros((len(keys), grid, grid), bool)
    assets = []
    for i, k in enumerate(keys):
        with np.load(cache / f"{k}.npz") as d:
            maps[i] = d[map_key]
        store, ep = k.split("__", 1)
        layout = json.loads((stores / store / ep / "meta.json").read_text())["layout"]
        occ[i] = bev_occupancy(layout, size_m, grid)
        small[i] = bev_occupancy({"assets": [a for a in layout["assets"] if a["kind"] in SMALL]}, size_m, grid)
        assets.append([(a["kind"], a["x_m"], a["y_m"], a["footprint_radius_m"]) for a in layout["assets"]])
    return maps, occ, small, assets


def elevation_label(tmap: TerrainMap, grid: int) -> np.ndarray:
    h = tmap.height_grid[::-1]  # row 0 -> north (+y)
    f = h.shape[0] // grid
    h = h[: f * grid, : f * grid].reshape(grid, f, grid, f).mean((1, 3))
    return ((h - tmap.meta["height_min_m"]) / (tmap.meta["height_max_m"] - tmap.meta["height_min_m"])).astype(np.float32)


@torch.no_grad()
def evaluate(model: MapHead, maps: torch.Tensor, occ: torch.Tensor, small: torch.Tensor, elev: torch.Tensor,
             assets: list, grid: int, size_m: float, batch: int = 128) -> dict:
    inter = union = tp_small = n_small = fp = 0.0
    se_elev = 0.0
    det = {"rock": [0, 0], "tree": [0, 0], "house": [0, 0]}
    probs_all = []
    for i in range(0, maps.shape[0], batch):
        logit, el = model(maps[i:i + batch])
        pred = torch.sigmoid(logit) > 0.5
        t = occ[i:i + batch]
        inter += (pred & t).sum().item(); union += (pred | t).sum().item()
        tp_small += (pred & small[i:i + batch]).sum().item(); n_small += small[i:i + batch].sum().item()
        fp += (pred & ~t).sum().item()
        se_elev += ((model.elevation_m(el) - model.elevation_m(elev)) ** 2).sum().item()
        probs_all.append(pred.cpu().numpy())
    pred_np = np.concatenate(probs_all)
    half = size_m / 2
    cell = size_m / grid
    for i, alist in enumerate(assets):
        for kind, x, y, r in alist:
            xs = (np.arange(grid) + 0.5) * cell - half
            ys = half - (np.arange(grid) + 0.5) * cell
            gx, gy = np.meshgrid(xs, ys)
            m = (gx - x) ** 2 + (gy - y) ** 2 <= r * r
            if m.sum() == 0:
                continue
            det[kind][1] += 1
            det[kind][0] += int(pred_np[i][m].mean() >= 0.5)
    n_px = maps.shape[0] * grid * grid
    return {"iou": inter / max(union, 1), "small_recall_px": tp_small / max(n_small, 1),
            "fp_area_m2_per_layout": fp * cell * cell / maps.shape[0],
            "elev_rmse_m": math.sqrt(se_elev / n_px),
            "detect_rate": {k: (v[0] / v[1] if v[1] else None) for k, v in det.items()},
            "n_assets": {k: v[1] for k, v in det.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6")
    ap.add_argument("--stores", default="artifacts/traverse")
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--map-key", default="map_v2")
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--pos-weight", type=float, default=5.0)
    ap.add_argument("--small-weight", type=float, default=3.0)
    ap.add_argument("--elev-weight", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    dev = torch.device(args.device)
    cache, stores, arena = Path(args.cache), Path(args.stores), Path(args.arena)
    tmap = TerrainMap.from_dir(arena)
    keys = D.load_cache_keys(cache)
    tr, va, _ = D.split_keys(keys)
    if args.limit_train:
        tr = tr[: args.limit_train]
    if args.limit_val:
        va = va[: args.limit_val]
    t0 = time.time()
    m_tr, o_tr, s_tr, _ = load_maps_and_labels(cache, stores, tr, args.grid, args.map_key, tmap.size_m)
    m_va, o_va, s_va, a_va = load_maps_and_labels(cache, stores, va, args.grid, args.map_key, tmap.size_m)
    print(f"loaded {len(tr)} train / {len(va)} val maps in {time.time() - t0:.0f}s; "
          f"occupied frac {o_tr.mean():.4f}, small-object frac {s_tr.mean():.4f}", flush=True)
    elev = torch.tensor(elevation_label(tmap, args.grid), device=dev)
    m_tr, o_tr, s_tr = torch.tensor(m_tr, device=dev), torch.tensor(o_tr, device=dev), torch.tensor(s_tr, device=dev)
    m_va, o_va, s_va = torch.tensor(m_va, device=dev), torch.tensor(o_va, device=dev), torch.tensor(s_va, device=dev)

    model = MapHead(arena, args.grid, width=args.width).to(dev)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    pos_w = torch.tensor(args.pos_weight, device=dev)
    best, log = -1.0, (out / "train_log.jsonl").open("w")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        idx = torch.randint(0, m_tr.shape[0], (args.batch,), device=dev)
        logit, el = model(m_tr[idx])
        target = o_tr[idx].float()
        w = 1.0 + (args.small_weight - 1.0) * s_tr[idx].float()
        bce = (F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_w, reduction="none") * w).mean()
        mse = F.mse_loss(el, elev.expand_as(el))
        loss = bce + args.elev_weight * mse
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0 or step == 1:
            rec = {"step": step, "loss": loss.item(), "bce": bce.item(), "elev_mse": mse.item(),
                   "lr": sched.get_last_lr()[0], "sps": step / (time.time() - t0)}
            log.write(json.dumps(rec) + "\n"); log.flush()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            ev = evaluate(model, m_va, o_va, s_va, elev, a_va, args.grid, tmap.size_m)
            ev["step"] = step
            print(json.dumps(ev), flush=True)
            log.write(json.dumps({"eval": ev}) + "\n"); log.flush()
            if ev["iou"] > best:
                best = ev["iou"]
                torch.save({"model": model.state_dict(), "config": vars(args), "eval": ev, "step": step}, out / "ckpt_best.pt")
                (out / "readout.json").write_text(json.dumps(ev, indent=1))
    print(f"done; best val IoU {best:.4f}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
