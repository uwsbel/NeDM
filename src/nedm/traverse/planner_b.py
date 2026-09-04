"""Planner-B: the oracle's search on a CAMERA-DERIVED map (plan section 7, WP4).

The map head decodes bird's-eye occupancy (+ elevation) from the frozen scene
feature map. Every occupied cell becomes a small disc obstacle, so the existing
direction-aware A* / smoothing / footprint validation / speed profile run
unchanged on the predicted map. Terrain comes either from the true heightmap
(ablation rung "predicted occupancy + memorized terrain") or from the predicted
elevation (rung "full predicted map"). Nothing here reads the layout manifest.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from nedm.traverse.oracle import Obstacle, OracleGrid, PlanCandidate, PlannerParams, a_star, plan_to_ring
from nedm.traverse.terrain import TerrainMap


class MapDecoder:
    """Frozen map head -> occupancy probability and elevation (m), world grid, row 0 = north."""

    def __init__(self, ckpt_path: Path, arena_dir: Path, device: str = "cpu"):
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "maphead", Path(__file__).resolve().parents[3] / "scripts" / "traverse_wp4_train_maphead.py")
        mod = importlib.util.module_from_spec(spec); sys.modules["maphead"] = spec.loader.exec_module(mod) or mod
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = payload["config"]
        self.model = mod.MapHead(Path(arena_dir), cfg["grid"], width=cfg["width"]).to(device).eval()
        self.model.load_state_dict(payload["model"])
        self.device = device
        self.grid = cfg["grid"]
        self.size_m = self.model.size_m
        self.tmap_meta = TerrainMap.from_dir(Path(arena_dir)).meta

    @torch.no_grad()
    def __call__(self, scene_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = torch.as_tensor(scene_map, device=self.device)[None]
        logit, el = self.model(m)
        return torch.sigmoid(logit)[0].cpu().numpy(), self.model.elevation_m(el)[0].cpu().numpy()

    def terrain(self, elevation_m: np.ndarray, smooth_sigma_cells: float = 1.0) -> TerrainMap:
        """Predicted elevation (row 0 = north) as a TerrainMap (row 0 = south). A small
        Gaussian blur tames the cell-to-cell noise the planner's slope caps would otherwise see."""
        h = elevation_m[::-1].astype(np.float64)
        if smooth_sigma_cells > 0:
            r = int(3 * smooth_sigma_cells)
            k = np.exp(-0.5 * (np.arange(-r, r + 1) / smooth_sigma_cells) ** 2); k /= k.sum()
            pad = np.pad(h, r, mode="edge")
            h = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, pad)
            h = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, h)
        return TerrainMap(h, self.size_m, self.tmap_meta)


def _label_components(occ: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """8-connected components of a boolean grid (numpy only): list of (rows, cols)."""
    seen = np.zeros_like(occ, dtype=bool)
    comps = []
    for sy, sx in zip(*np.nonzero(occ)):
        if seen[sy, sx]:
            continue
        stack, rows, cols = [(sy, sx)], [], []
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            rows.append(y); cols.append(x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < occ.shape[0] and 0 <= nx < occ.shape[1] and occ[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        comps.append((np.asarray(rows), np.asarray(cols)))
    return comps


def occupancy_discs(occ_prob: np.ndarray, size_m: float, threshold: float = 0.5,
                    min_cells: int = 2, mode: str = "components") -> list[Obstacle]:
    """Predicted occupancy -> disc obstacles for the oracle's planner.

    ``mode="cells"``: one 0.44 m disc per occupied cell (jagged blob boundaries make
    A* paths that the smoother cannot bring under the curvature cap -- 9 % no-path).
    ``mode="components"``: one enclosing disc per 8-connected blob (centroid, radius =
    farthest member cell + half a cell diagonal), i.e. the same obstacle shape the
    oracle sees. Blobs below ``min_cells`` are dropped as speckle."""
    g = occ_prob.shape[0]
    cell = size_m / g
    half = size_m / 2
    occ = occ_prob >= threshold
    r_cell = cell * np.sqrt(0.5)
    out: list[Obstacle] = []
    for iy, ix in _label_components(occ):
        if len(iy) < min_cells:
            continue
        xs = (ix + 0.5) * cell - half
        ys = half - (iy + 0.5) * cell
        if mode == "cells":
            out += [(float(x), float(y), float(r_cell)) for x, y in zip(xs, ys)]
        else:
            cx, cy = float(xs.mean()), float(ys.mean())
            out.append((cx, cy, float(np.hypot(xs - cx, ys - cy).max() + r_cell)))
    return out


def plan_on_predicted_map(decoder: MapDecoder, scene_map: np.ndarray, start_xy, ring_center,
                          params: PlannerParams | None = None, true_terrain: TerrainMap | None = None,
                          threshold: float = 0.5, elev_smooth: float = 1.0) -> tuple[PlanCandidate | None, dict]:
    """Camera-only planning. ``true_terrain`` given -> ablation rung with memorized terrain.
    On failure ``info["reason"]`` says whether A* found no path or smoothing/validation rejected it."""
    params = params or PlannerParams()
    # Cell-disc blobs have jagged inflated boundaries; the oracle's 12 curvature-repair
    # passes leave 9 % of A* paths above the curvature cap, 40 passes rescue 7 of 9.
    params = replace(params, curvature_repair_iterations=max(params.curvature_repair_iterations, 40))
    occ, elev = decoder(scene_map)
    tmap = true_terrain if true_terrain is not None else decoder.terrain(elev, elev_smooth)
    discs = occupancy_discs(occ, decoder.size_m, threshold, mode="cells")
    plan = plan_to_ring(tmap, discs, start_xy, ring_center, params)
    info = {"occupied_cells": len(discs), "occ": occ, "elev": elev, "reason": None}
    if plan is None:
        grid = OracleGrid(tmap, discs, params)
        ring = np.hypot(grid.node_x - ring_center[0], grid.node_y - ring_center[1])
        goal = (np.abs(ring - params.approach_ring_m) <= params.approach_ring_tol_m) & grid.valid
        if not grid.valid[grid.to_index(*start_xy)]:
            info["reason"] = "start_blocked"
        elif a_star(grid, start_xy, goal) is None:
            info["reason"] = "no_astar_path"
        else:
            info["reason"] = "validation_rejected"
    return plan, info
