#!/usr/bin/env python
"""Live planning inputs for the held-out comparison: one pre-departure camera frame per layout.

The scene maps in the dynamics cache (``map_v2``) are vehicle-free medians over 16 frames spread across
the recorded episode, i.e. they use views the vehicle only gets after it has driven the route. A live
planner has exactly one view at t = 0: the fixed camera's frame with the vehicle parked at the start.
This script encodes that single frame (frame ``--frame`` of the recording, default 0) with the same
WP1 encoder stem, writes a self-contained mini cache (all base arrays, ``map_v2`` replaced by the
single-frame map, the full median kept as ``map_v2_full``) and reports how the camera map decoded from
the single frame compares with the median map: occupancy IoU against the true layout, disc counts,
goal error, and occupied cells near the parked vehicle (the vehicle itself is in the picture).
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nedm.traverse import nrd_data as D, perception as P
from nedm.traverse.layout import EpisodeLayout
from nedm.traverse.planner_b import MapDecoder, goal_from_map, occupancy_discs
from nedm.traverse.storage import EpisodeReader
from nedm.traverse.terrain import TerrainMap
from traverse_wp2_encode_map import MAP_STAGE, EpisodeMedian
from traverse_wp4_train_maphead import bev_occupancy


def vehicle_mask_xy(helper: EpisodeMedian, x: float, y: float, yaw: float) -> np.ndarray:
    """Camera-image mask of the vehicle body at a pose estimate (same box + 3 px dilation as the cache builder)."""
    from nedm.traverse.masks import VEHICLE_HGT_M, VEHICLE_LEN_M, VEHICLE_WID_M, _box_corners, _rasterize_solids
    from traverse_wp2_encode_map import MASK_DILATE
    cam, _, _, _ = helper._camera()
    gz = float(helper.tmap.height(x, y))
    mask = _rasterize_solids([_box_corners(x, y, yaw, VEHICLE_LEN_M, VEHICLE_WID_M, gz, gz + VEHICLE_HGT_M)], cam)
    out = mask.copy()
    for dy in range(-MASK_DILATE, MASK_DILATE + 1):
        for dx in range(-MASK_DILATE, MASK_DILATE + 1):
            out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return out


def fill_masked(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill masked pixels of img (H, W[, C]) from their known 3x3 neighbours, growing inwards (no scipy)."""
    out = img.astype(np.float32).copy()
    known = ~mask
    if out.ndim == 2:
        out = out[..., None]
    for _ in range(512):
        if known.all():
            break
        acc = np.zeros_like(out); cnt = np.zeros(known.shape, np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                k = np.roll(np.roll(known, dy, 0), dx, 1); v = np.roll(np.roll(out, dy, 0), dx, 1)
                acc += v * k[..., None]; cnt += k
        grow = ~known & (cnt > 0)
        out[grow] = acc[grow] / cnt[grow][:, None]
        known |= grow
    return out[..., 0] if img.ndim == 2 else out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/traverse/wp5_live_cache_test")
    ap.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6")
    ap.add_argument("--routes", default="artifacts/traverse/wp3_routes")
    ap.add_argument("--stores", default="artifacts/traverse")
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--encoder", default="artifacts/traverse/wp1_v6/ckpt_warmup.pt")
    ap.add_argument("--maphead", default="artifacts/traverse/wp4_maphead_v2/ckpt_best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--families", nargs="+", default=["oracle"])
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--occ-threshold", type=float, default=0.85)
    ap.add_argument("--start-poses", default="artifacts/traverse/wp4_start_poses/test_frame0_start_poses.json",
                    help="camera start-pose estimates: the vehicle masks ITSELF out of its frame at this pose")
    ap.add_argument("--no-mask-vehicle", action="store_true", help="leave the parked vehicle in the frame (breaks the dynamics model's ego crop)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache, out = Path(args.cache), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = D.load_cache_keys(cache)
    split = dict(zip(("train", "val", "test"), D.split_keys(keys)))[args.split]
    manifest = json.loads((Path(args.routes) / "routes_manifest.json").read_text())
    allowed = set().union(*(set(manifest["families"][f]) for f in args.families))
    keys = [k for k in split if k in allowed][: args.episodes]
    tmap = TerrainMap.from_dir(Path(args.arena))
    enc = P.Encoder(z_dim=256, n_q=8).to(dev)
    enc.load_state_dict(torch.load(args.encoder, map_location=dev, weights_only=False)["encoder"], strict=True); enc.eval()
    stem = enc.backbone[:MAP_STAGE]
    helper = EpisodeMedian(keys, Path(args.stores), Path(args.arena))
    decoder = MapDecoder(Path(args.maphead), Path(args.arena), dev)
    grid = decoder.grid
    start_est = json.loads(Path(args.start_poses).read_text()) if Path(args.start_poses).exists() else {}
    stats = {}
    for key in keys:
        store, ep = key.split("__", 1)
        reader = EpisodeReader(Path(args.stores) / store / ep)
        win = reader.read_window(args.frame, 1); reader.close()
        fi = {n: i for i, n in enumerate(win["state_fields"])}
        rgb = win["rgb"][0].astype(np.float32) / 255.0
        elev = helper._elevation(win["depth_mm"][0])
        n_masked = 0
        if not args.no_mask_vehicle:  # the vehicle removes itself from its own view at the estimated pose
            ex, ey, eyaw = start_est[key]["est"] if key in start_est else (float(win["states"][0][fi["pos_x_m"]]), float(win["states"][0][fi["pos_y_m"]]), float(win["states"][0][fi["yaw_rad"]]))
            mask = vehicle_mask_xy(helper, ex, ey, eyaw)
            n_masked = int(mask.sum())
            rgb, elev = fill_masked(rgb, mask), fill_masked(elev, mask)
        inp = torch.from_numpy(np.concatenate([rgb.transpose(2, 0, 1), elev[None]]).astype(np.float32))[None].to(dev)
        with torch.no_grad():
            live = stem(inp).float().cpu().numpy().astype(np.float16)[0]
        with np.load(cache / f"{key}.npz") as d:
            arrays = {k: d[k] for k in d.files}
        full = arrays["map_v2"]
        arrays["map_v2_full"] = full
        arrays["map_v2"] = live
        np.savez(out / f"{key}.npz", **arrays)
        # how does the single-frame camera map compare?
        layout = EpisodeLayout.from_json(json.loads((Path(args.stores) / store / ep / "meta.json").read_text())["layout"])
        truth = bev_occupancy(json.loads((Path(args.stores) / store / ep / "meta.json").read_text())["layout"], decoder.size_m, grid)
        row = {}
        for name, m in (("live", live), ("full", full)):
            occ, _ = decoder(m)
            o = occ >= args.occ_threshold
            discs = occupancy_discs(occ, decoder.size_m, args.occ_threshold, mode="cells")
            g = goal_from_map(occ, decoder.size_m, args.occ_threshold)
            cell = decoder.size_m / grid; half = decoder.size_m / 2
            iy, ix = np.nonzero(o)
            cx, cy = (ix + 0.5) * cell - half, half - (iy + 0.5) * cell
            near = int((np.hypot(cx - layout.start_xy[0], cy - layout.start_xy[1]) < 4.0).sum())
            row[name] = {"iou_true": float((o & truth).sum() / max((o | truth).sum(), 1)), "cells": int(o.sum()),
                         "true_cells": int(truth.sum()), "discs": len(discs),
                         "goal_err_m": float(np.hypot(g[0] - layout.house_xy[0], g[1] - layout.house_xy[1])),
                         "cells_within_4m_of_start": near,
                         "missed_true_cells": int((truth & ~o).sum()), "false_cells": int((o & ~truth).sum())}
        row["masked_px"] = n_masked
        stats[key] = row
        print(f"{key}: masked {n_masked} px | live IoU {row['live']['iou_true']:.3f} ({row['live']['cells']} cells, {row['live']['discs']} discs, goal err {row['live']['goal_err_m']:.2f} m, "
              f"{row['live']['cells_within_4m_of_start']} cells near start) | full IoU {row['full']['iou_true']:.3f} ({row['full']['cells']} cells, goal err {row['full']['goal_err_m']:.2f} m, "
              f"{row['full']['cells_within_4m_of_start']} near start)", flush=True)
    (out / "cache_manifest.json").write_text(json.dumps({"episodes": keys, "source_cache": str(cache), "frame": args.frame, "encoder": args.encoder,
                                                         "vehicle_masked": not args.no_mask_vehicle, "start_poses": args.start_poses}))
    (out / "live_inputs.json").write_text(json.dumps(stats, indent=1))
    for name in ("live", "full"):
        print(f"{name:5s}: IoU {np.mean([s[name]['iou_true'] for s in stats.values()]):.3f}  cells {np.mean([s[name]['cells'] for s in stats.values()]):.0f} "
              f"(true {np.mean([s[name]['true_cells'] for s in stats.values()]):.0f})  discs {np.mean([s[name]['discs'] for s in stats.values()]):.0f}  "
              f"goal err mean {np.mean([s[name]['goal_err_m'] for s in stats.values()]):.2f} max {np.max([s[name]['goal_err_m'] for s in stats.values()]):.2f} m  "
              f"near-start cells mean {np.mean([s[name]['cells_within_4m_of_start'] for s in stats.values()]):.1f} max {np.max([s[name]['cells_within_4m_of_start'] for s in stats.values()])}  "
              f"missed {np.mean([s[name]['missed_true_cells'] for s in stats.values()]):.1f}  false {np.mean([s[name]['false_cells'] for s in stats.values()]):.1f}")


if __name__ == "__main__":
    main()
