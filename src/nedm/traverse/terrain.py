"""Authored fixed-arena heightmap for the HMMWV traversal study (plan §3.1).

Generates a single composed heightfield (craters + hills + smoothed roughness),
exports it as the 8-bit grayscale BMP that Chrono's RigidTerrain heightmap patch
consumes, and provides ``TerrainMap`` — the privileged-oracle view of the SAME
quantized heights, so oracle and simulator agree to quantization error.

The BMP is the single source of truth: ``TerrainMap`` always loads heights back
from the BMP, never from the float field used during generation.

Chrono maps the image over the patch with an orientation convention we do not
hard-code; ``orientation`` in the metadata is calibrated empirically against
``RigidTerrain.GetHeight`` by the WP0a slice (see scene.calibrate_orientation)
and written back here.

CLI: PYTHONPATH=src python -m nedm.traverse.terrain --out assets/traverse/arena_v1
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ARENA_SIZE_M = 80.0
ARENA_PIXELS = 512
SLOPE_CAP = math.tan(math.radians(20.0))  # global drivability target (plan §3.1)

BMP_NAME = "arena_000.bmp"
META_NAME = "arena_meta.json"


def _gaussian_blur(field_2d: np.ndarray, sigma_px: float) -> np.ndarray:
    """Separable gaussian blur, numpy-only (no scipy in the nedm env)."""
    radius = max(1, int(math.ceil(3.0 * sigma_px)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_px) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(field_2d, radius, mode="reflect")
    blurred = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), 1, padded)
    blurred = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="valid"), 0, blurred)
    return blurred


@dataclass
class TerrainFeature:
    kind: str  # "hill" | "crater"
    x_m: float
    y_m: float
    sigma_m: float
    amplitude_m: float  # positive for hills, positive depth for craters

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "sigma_m": self.sigma_m,
            "amplitude_m": self.amplitude_m,
        }


@dataclass
class ArenaSpec:
    size_m: float = ARENA_SIZE_M
    pixels: int = ARENA_PIXELS
    seed: int = 7
    n_hills: int = 6
    hill_sigma_range_m: tuple[float, float] = (4.0, 7.5)
    hill_height_range_m: tuple[float, float] = (1.5, 3.0)
    n_craters: int = 6
    crater_sigma_range_m: tuple[float, float] = (2.0, 4.0)
    crater_depth_range_m: tuple[float, float] = (1.0, 2.0)
    crater_rim_fraction: float = 0.25
    roughness_amplitude_m: float = 0.15
    roughness_corr_m: float = 2.0
    feature_margin_m: float = 8.0  # keep feature centers off the patch edge
    slope_cap: float = SLOPE_CAP


def _sample_separated_centers(
    rng: np.random.Generator, sigmas: list[float], lo: float, hi: float
) -> list[tuple[float, float]]:
    """Feature centers with pairwise separation >= 1.2*(sigma_i + sigma_j), so
    single-feature slope caps do not stack. Falls back to best-effort."""
    centers: list[tuple[float, float]] = []
    for i, sigma in enumerate(sigmas):
        best: tuple[float, float] | None = None
        best_clearance = -math.inf
        for _ in range(200):
            cand = (float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi)))
            clearance = min(
                (
                    math.hypot(cand[0] - cx, cand[1] - cy) - 1.2 * (sigma + sigmas[j])
                    for j, (cx, cy) in enumerate(centers)
                ),
                default=math.inf,
            )
            if clearance >= 0.0:
                best = cand
                break
            if clearance > best_clearance:
                best_clearance = clearance
                best = cand
        centers.append(best if best is not None else (0.0, 0.0))
        _ = i
    return centers


def _limit_slopes(height: np.ndarray, res: float, cap: float, iterations: int = 30) -> np.ndarray:
    """Diffuse material only where |grad h| exceeds the cap: blend toward a
    meter-scale blur inside (dilated) steep regions, leave the rest untouched."""
    sigma_px = 1.0 / res
    for _ in range(iterations):
        gy, gx = np.gradient(height, res)
        slope = np.sqrt(gx**2 + gy**2)
        if float(np.quantile(slope, 0.999)) <= cap and float(slope.max()) <= 1.2 * cap:
            break
        steep = (slope > 0.98 * cap).astype(np.float64)
        steep = _gaussian_blur(steep, 2.0)  # cheap dilation
        blend = np.clip(steep / 0.05, 0.0, 1.0) * 0.5
        height = (1.0 - blend) * height + blend * _gaussian_blur(height, sigma_px)
    return height


def generate_height_field(spec: ArenaSpec) -> tuple[np.ndarray, list[TerrainFeature]]:
    """Float heightfield h[iy, ix] with ix along +x and iy along +y (row 0 = -y)."""
    rng = np.random.default_rng(spec.seed)
    half = spec.size_m / 2.0
    res = spec.size_m / spec.pixels
    coords = -half + (np.arange(spec.pixels) + 0.5) * res
    grid_x, grid_y = np.meshgrid(coords, coords)  # h[iy, ix]

    height = np.zeros_like(grid_x)
    features: list[TerrainFeature] = []
    lo = -half + spec.feature_margin_m
    hi = half - spec.feature_margin_m

    hill_sigmas = [float(rng.uniform(*spec.hill_sigma_range_m)) for _ in range(spec.n_hills)]
    crater_sigmas = [float(rng.uniform(*spec.crater_sigma_range_m)) for _ in range(spec.n_craters)]
    centers = _sample_separated_centers(rng, hill_sigmas + crater_sigmas, lo, hi)

    for sigma, (cx, cy) in zip(hill_sigmas, centers[: spec.n_hills]):
        # Gaussian max slope is A * exp(-0.5) / sigma; cap A so a lone hill
        # stays under the drivability slope cap.
        amp_cap = spec.slope_cap * sigma / math.exp(-0.5)
        amp = float(min(rng.uniform(*spec.hill_height_range_m), 0.95 * amp_cap))
        r2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        height += amp * np.exp(-0.5 * r2 / sigma**2)
        features.append(TerrainFeature("hill", cx, cy, sigma, amp))

    for sigma, (cx, cy) in zip(crater_sigmas, centers[spec.n_hills :]):
        amp_cap = spec.slope_cap * sigma / math.exp(-0.5)
        depth = float(min(rng.uniform(*spec.crater_depth_range_m), 0.95 * amp_cap))
        r = np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
        bowl = -depth * np.exp(-0.5 * (r / sigma) ** 2)
        rim = spec.crater_rim_fraction * depth * np.exp(-0.5 * ((r - 1.6 * sigma) / (0.5 * sigma)) ** 2)
        height += bowl + rim
        features.append(TerrainFeature("crater", cx, cy, sigma, depth))

    if spec.roughness_amplitude_m > 0:
        noise = rng.standard_normal(height.shape)
        noise = _gaussian_blur(noise, spec.roughness_corr_m / res)
        noise *= spec.roughness_amplitude_m / max(noise.std(), 1e-9)
        height += noise

    # Drivability repair: local slope-limited diffusion (audit lives in metadata).
    height = _limit_slopes(height, res, spec.slope_cap)
    return height, features


def _max_slope(height: np.ndarray, res: float) -> float:
    gy, gx = np.gradient(height, res)
    return float(np.sqrt(gx**2 + gy**2).max())


def _slope_stats(height: np.ndarray, res: float) -> dict[str, float]:
    gy, gx = np.gradient(height, res)
    slope = np.sqrt(gx**2 + gy**2)
    return {
        "max": float(slope.max()),
        "p999": float(np.quantile(slope, 0.999)),
        "p99": float(np.quantile(slope, 0.99)),
        "median": float(np.median(slope)),
        "flat5deg_fraction": float((slope < math.tan(math.radians(5.0))).mean()),
    }


def write_arena(spec: ArenaSpec, out_dir: Path) -> Path:
    """Generate, quantize to 8-bit BMP, and write metadata. Returns out_dir."""
    height, features = generate_height_field(spec)
    res = spec.size_m / spec.pixels

    h_min = float(math.floor(height.min() * 20.0) / 20.0)
    h_max = float(math.ceil(height.max() * 20.0) / 20.0)
    gray = np.round((height - h_min) / (h_max - h_min) * 255.0).clip(0, 255).astype(np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Image row 0 renders at the top; our row 0 is -y. Store the array as-is and
    # let the calibrated orientation transform absorb every convention at once.
    Image.fromarray(gray, mode="L").save(out_dir / BMP_NAME)

    quantized = h_min + gray.astype(np.float64) / 255.0 * (h_max - h_min)
    meta = {
        "size_m": spec.size_m,
        "pixels": spec.pixels,
        "resolution_m_per_px": res,
        "height_min_m": h_min,
        "height_max_m": h_max,
        "quantization_step_m": (h_max - h_min) / 255.0,
        "seed": spec.seed,
        "bmp": BMP_NAME,
        # Transform from the raw BMP array to h[iy, ix] world axes; calibrated
        # against RigidTerrain.GetHeight by the WP0a slice, then frozen.
        "orientation": {"rot90": 0, "flipud": False, "calibrated": False},
        "features": [f.to_json() for f in features],
        "slope_stats": _slope_stats(quantized, res),
        "slope_cap": spec.slope_cap,
    }
    with (out_dir / META_NAME).open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return out_dir


def _apply_orientation(raw: np.ndarray, orientation: dict[str, Any]) -> np.ndarray:
    arr = np.rot90(raw, int(orientation.get("rot90", 0)))
    if orientation.get("flipud", False):
        arr = np.flipud(arr)
    return arr


class TerrainMap:
    """Privileged-oracle heightfield: bilinear height/gradient queries in world
    coordinates over the SAME quantized data Chrono loads from the BMP."""

    def __init__(self, height: np.ndarray, size_m: float, meta: dict[str, Any]):
        if height.shape[0] != height.shape[1]:
            raise ValueError("TerrainMap expects a square grid")
        self.height_grid = height  # h[iy, ix], row 0 at -y
        self.size_m = float(size_m)
        self.meta = meta
        self.pixels = height.shape[0]
        self.res = self.size_m / self.pixels
        self.half = self.size_m / 2.0
        gy, gx = np.gradient(height, self.res)
        self.grad_x = gx
        self.grad_y = gy

    @classmethod
    def from_dir(cls, arena_dir: Path) -> "TerrainMap":
        with (arena_dir / META_NAME).open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        raw = np.asarray(Image.open(arena_dir / meta["bmp"]).convert("L"), dtype=np.float64)
        gray = _apply_orientation(raw, meta.get("orientation", {}))
        height = meta["height_min_m"] + gray / 255.0 * (meta["height_max_m"] - meta["height_min_m"])
        return cls(height, meta["size_m"], meta)

    def _fractional_index(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fx = (np.asarray(x) + self.half) / self.res - 0.5
        fy = (np.asarray(y) + self.half) / self.res - 0.5
        return np.clip(fx, 0.0, self.pixels - 1.001), np.clip(fy, 0.0, self.pixels - 1.001)

    def _bilinear(self, grid: np.ndarray, x: Any, y: Any) -> np.ndarray:
        fx, fy = self._fractional_index(x, y)
        ix0 = np.floor(fx).astype(int)
        iy0 = np.floor(fy).astype(int)
        ix1 = np.minimum(ix0 + 1, self.pixels - 1)
        iy1 = np.minimum(iy0 + 1, self.pixels - 1)
        ax = fx - ix0
        ay = fy - iy0
        top = grid[iy0, ix0] * (1 - ax) + grid[iy0, ix1] * ax
        bot = grid[iy1, ix0] * (1 - ax) + grid[iy1, ix1] * ax
        return top * (1 - ay) + bot * ay

    def height(self, x: Any, y: Any) -> np.ndarray:
        return self._bilinear(self.height_grid, x, y)

    def gradient(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        return self._bilinear(self.grad_x, x, y), self._bilinear(self.grad_y, x, y)

    def slope(self, x: Any, y: Any) -> np.ndarray:
        gx, gy = self.gradient(x, y)
        return np.sqrt(gx**2 + gy**2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the authored traversal arena heightmap.")
    parser.add_argument("--out", default="assets/traverse/arena_v1", help="Output directory.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size-m", type=float, default=ARENA_SIZE_M)
    parser.add_argument("--pixels", type=int, default=ARENA_PIXELS)
    args = parser.parse_args(argv)

    spec = ArenaSpec(size_m=args.size_m, pixels=args.pixels, seed=args.seed)
    out_dir = write_arena(spec, Path(args.out))
    tmap = TerrainMap.from_dir(out_dir)
    stats = tmap.meta["slope_stats"]
    print(f"arena written to {out_dir}")
    print(
        f"height [{tmap.meta['height_min_m']:.2f}, {tmap.meta['height_max_m']:.2f}] m, "
        f"quantization {tmap.meta['quantization_step_m']*100:.1f} cm"
    )
    print(
        "slope: max {max:.3f} p99.9 {p999:.3f} p99 {p99:.3f} median {median:.3f} "
        "flat(<5deg) fraction {flat5deg_fraction:.2f}".format(**stats)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
