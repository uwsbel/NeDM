"""Controlled bowl arena for the crater-learnability diagnostic (plan §B1).

``terrain.generate_height_field`` cannot build this feature. It caps every crater's
amplitude at ``0.95 * slope_cap * sigma / exp(-0.5)``, which pins a crater's peak radial
slope at ``0.95 * slope_cap`` no matter what depth is asked for, and ``_limit_slopes``
then diffuses whatever is still above the cap. A gaussian is also steep over only about
one sigma, so it never presents a sustained wall. This module therefore builds its own
height field and does NOT run the slope repair — the whole point is a wall steeper than
the drivability cap.

Parametrisation (single-valued by construction, so it stays a legal heightmap)::

    r      = hypot(x - cx, y - cy)
    u(r)   = clip((r - R_bottom) / W, 0, 1)              position along the wall band
    s(r)   = tan(theta(u)) * win(r)                      radial slope magnitude
    win(r) = S((r - (R_b - rho)) / 2rho) * (1 - S((r - (R_b + W - rho)) / 2rho))
    S(t)   = clip(t,0,1)^2 (3 - 2 clip(t,0,1))           C1 smoothstep
    z(r)   = -D + integral_0^r s dr', with W solved by bisection so z -> 0 outside
    z(r,phi) = (1 - w(phi)) z_wall(r) + w(phi) z_entry(r)   asymmetric entry ramp
    w(phi) = exp(-0.5 (angdiff(phi, phi_entry) / phi_halfwidth)^2)

``theta(u)`` is constant (a cone) or linear in ``u`` (a flare). ``rho`` (``round_m``) is
the ONLY smoothing and exists solely so 8-bit quantisation cannot alias the wall: it is
required to be at least ``MIN_ROUND_PX`` pixels. Depth and wall angle are exact by
construction, not the residue of a repair pass.

The three B1 arms (deep / shallow / flat) are written from ONE bowl field on ONE shared
height range, so after quantisation the surrounding plain and its roughness are
bit-identical outside the rim (``max(R_top, R_top_entry) + round_m``) and every difference
between arms is inside the depression itself.

FRAME. A spec is written in the SIMULATED frame — the frame ``TerrainMap`` exposes and
Chrono drives in. The BMP is re-oriented on load by ``terrain._apply_orientation``
(rot90 0 / flipud True for every arena in this study), so the array actually stored is
the inverse-oriented field and the feature centre stored in ``arena_meta.json`` is the
inverse-oriented point; ``TerrainMap.features`` then hands back the true bowl centre.
Both inversions are asserted against ``terrain`` at write time.

CLI: ``scripts/traverse_wp9_bowl_arena.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from nedm.traverse.terrain import (
    ARENA_PIXELS,
    ARENA_SIZE_M,
    BMP_NAME,
    META_NAME,
    TerrainMap,
    _apply_orientation,
    _gaussian_blur,
    _slope_stats,
    orient_xy,
)

# Rounding radius floor, in pixels. Below ~1.5 px the wall's toe and rim corners turn over
# inside a single 8-bit height step, so the quantised surface carries a one-pixel stair
# where the float field has a rounded corner. This is the ONLY smoothing applied.
MIN_ROUND_PX = 1.5

# HMMWV_Full, measured in sim (vehicle.GetWheelbase(); collision-hull bottom minus the tire
# contact plane; atan(clearance / front overhang); RigidTerrain.GetCoefficientFriction()).
WHEELBASE_M = 3.378
CLEARANCE_M = 0.4035
APPROACH_ANGLE_DEG = 35.1
TIRE_FRICTION = 0.9


@dataclass(frozen=True)
class BowlSpec:
    """One bowl, in the simulated frame. Angles are from horizontal, depths positive."""

    wall_deg: float = 47.0
    depth_m: float = 3.2
    bottom_radius_m: float = 4.5
    entry_deg: float = 15.0
    entry_azimuth_deg: float = 180.0  # the approach corridor comes from -x
    entry_halfwidth_deg: float = 28.0
    wall_top_deg: float | None = None  # linear flare wall_deg -> wall_top_deg over the band
    entry_top_deg: float | None = None
    centre_x_m: float = 0.0
    centre_y_m: float = 0.0
    round_m: float = 0.25
    roughness_amplitude_m: float = 0.03
    roughness_corr_m: float = 3.0
    size_m: float = ARENA_SIZE_M
    pixels: int = ARENA_PIXELS
    seed: int = 901
    relief_ring_margin_m: float = 1.5  # clearance for the feature_relief_m sampling ring

    @property
    def res_m(self) -> float:
        return self.size_m / self.pixels

    def __post_init__(self) -> None:
        if self.round_m < MIN_ROUND_PX * self.res_m:
            raise ValueError(
                f"round_m {self.round_m} m is below the quantisation-safety floor "
                f"{MIN_ROUND_PX * self.res_m:.3f} m ({MIN_ROUND_PX} px at {self.res_m:.5f} m/px)"
            )
        if self.bottom_radius_m <= 0.0 or self.depth_m <= 0.0:
            raise ValueError("bottom_radius_m and depth_m must be positive")


@dataclass(frozen=True)
class ArmSpec:
    """One arm of the deep/shallow/flat triple, derived from the same bowl field.

    ``scale`` multiplies the bowl vertically: the rim stays where it is and the wall angle
    falls with the depth (3.2 m at 47 deg -> 0.9 m at about 17 deg), which is the control
    the B1 probe drove. ``clip`` instead raises the floor (``max(z, -depth)``), keeping the
    upper wall bit-identical to the deep arm; that leaves a 0.9 m deep pit still walled at
    the full ``wall_deg``, which is steeper than the 35.1 deg approach angle, so it is NOT
    a driveable control and is offered only for ablations.
    """

    tag: str
    depth_m: float | None = None  # None = the spec's own depth (the deep arm)
    mode: str = "scale"  # "scale" | "clip"


DEFAULT_ARMS: tuple[ArmSpec, ...] = (
    ArmSpec("deep", None, "scale"),
    ArmSpec("shallow", 0.9, "scale"),
    ArmSpec("flat", 0.0, "scale"),
)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _radial_slope(
    r: np.ndarray, r_bottom: float, band_w: float, round_m: float, theta0: float, theta1: float
) -> np.ndarray:
    """|dz/dr| of the wall band, windowed by a C1 smoothstep at the toe and at the rim."""
    u = np.clip((r - r_bottom) / band_w, 0.0, 1.0)
    window = _smoothstep((r - (r_bottom - round_m)) / (2.0 * round_m)) * (
        1.0 - _smoothstep((r - (r_bottom + band_w - round_m)) / (2.0 * round_m))
    )
    return np.tan(np.radians(theta0 + (theta1 - theta0) * u)) * window


def _radial_profile(
    r_grid: np.ndarray, r_bottom: float, depth_m: float, round_m: float, theta0: float, theta1: float
) -> tuple[np.ndarray, float]:
    """``z(r)`` reaching EXACTLY ``-depth_m`` at the centre and 0 outside; returns (z, band_w).

    The band width is the free variable: bisect it until the wall's integrated rise equals
    the requested depth, so the requested angle is honoured everywhere on the band.
    """
    dr = float(r_grid[1] - r_grid[0])

    def rise(width: float) -> float:
        s = _radial_slope(r_grid, r_bottom, width, round_m, theta0, theta1)
        return float(np.sum(0.5 * (s[1:] + s[:-1])) * dr)

    lo, hi = 1e-3, float(r_grid[-1])
    if rise(hi) < depth_m:
        raise ValueError(f"wall angles {theta0}/{theta1} deg cannot reach {depth_m} m within the grid")
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rise(mid) < depth_m else (lo, mid)
    band_w = 0.5 * (lo + hi)
    s = _radial_slope(r_grid, r_bottom, band_w, round_m, theta0, theta1)
    z = -depth_m + np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * dr)])
    return z - z[-1], band_w


def bowl_field(spec: BowlSpec) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """(bowl, roughness, info) on the simulated-frame grid ``h[iy, ix]``, row 0 at -y.

    ``bowl`` is 0 outside the rim and ``-depth_m`` at the centre; ``roughness`` is the
    zero-mean field added to every arm unchanged.
    """
    res = spec.res_m
    half = spec.size_m / 2.0
    coords = -half + (np.arange(spec.pixels) + 0.5) * res
    grid_x, grid_y = np.meshgrid(coords, coords)
    dx, dy = grid_x - spec.centre_x_m, grid_y - spec.centre_y_m
    radius, azimuth = np.hypot(dx, dy), np.arctan2(dy, dx)

    r_grid = np.arange(0.0, 2.0 * spec.size_m, 0.25 * res)
    z_wall, w_wall = _radial_profile(
        r_grid, spec.bottom_radius_m, spec.depth_m, spec.round_m,
        spec.wall_deg, spec.wall_top_deg if spec.wall_top_deg is not None else spec.wall_deg,
    )
    z_entry, w_entry = _radial_profile(
        r_grid, spec.bottom_radius_m, spec.depth_m, spec.round_m,
        spec.entry_deg, spec.entry_top_deg if spec.entry_top_deg is not None else spec.entry_deg,
    )

    d_az = np.abs(
        np.arctan2(
            np.sin(azimuth - math.radians(spec.entry_azimuth_deg)),
            np.cos(azimuth - math.radians(spec.entry_azimuth_deg)),
        )
    )
    weight = np.exp(-0.5 * (d_az / math.radians(spec.entry_halfwidth_deg)) ** 2)
    bowl = (1.0 - weight) * np.interp(radius, r_grid, z_wall, right=0.0) + weight * np.interp(
        radius, r_grid, z_entry, right=0.0
    )

    rng = np.random.default_rng(spec.seed)
    roughness = _gaussian_blur(rng.standard_normal(bowl.shape), spec.roughness_corr_m / res)
    roughness *= spec.roughness_amplitude_m / max(float(roughness.std()), 1e-9)

    r_top = spec.bottom_radius_m + w_wall
    r_top_entry = spec.bottom_radius_m + w_entry
    info = {
        "wall_band_m": w_wall,
        "entry_band_m": w_entry,
        "R_bottom_m": spec.bottom_radius_m,
        "R_top_m": r_top,
        "R_top_entry_m": r_top_entry,
        "round_px": spec.round_m / res,
        # feature_relief_m samples a ring at 1.75 sigma; put that ring on the plain OUTSIDE
        # the long entry ramp so the reported relief is the real depth (see module docstring).
        "relief_sigma_m": (max(r_top, r_top_entry) + spec.relief_ring_margin_m) / 1.75,
    }
    return bowl, roughness, info


def arm_depth_m(spec: BowlSpec, arm: ArmSpec) -> float:
    return spec.depth_m if arm.depth_m is None else float(arm.depth_m)


def arm_field(spec: BowlSpec, bowl: np.ndarray, roughness: np.ndarray, arm: ArmSpec) -> np.ndarray:
    """One arm's float height field. Outside the rim ``bowl`` is exactly 0 under both modes,
    so every arm shares the plain and its roughness bit-for-bit after quantisation."""
    depth = arm_depth_m(spec, arm)
    if arm.mode == "scale":
        shaped = (depth / spec.depth_m) * bowl
    elif arm.mode == "clip":
        shaped = np.maximum(bowl, -depth)
    else:
        raise ValueError(f"unknown arm mode {arm.mode!r} (expected 'scale' or 'clip')")
    return shaped + roughness


def _unapply_orientation(arr: np.ndarray, orientation: dict[str, Any]) -> np.ndarray:
    """Inverse of ``terrain._apply_orientation``: simulated-frame array -> stored BMP array."""
    if orientation.get("flipud", False):
        arr = np.flipud(arr)
    return np.rot90(arr, -int(orientation.get("rot90", 0)))


def _unorient_xy(
    x: float, y: float, size_m: float, pixels: int, orientation: dict[str, Any]
) -> tuple[float, float]:
    """Inverse of ``terrain.orient_xy``: simulated-frame point -> generation-frame point."""
    res = size_m / pixels
    half = size_m / 2.0
    ix = (x + half) / res - 0.5
    iy = (y + half) / res - 0.5
    if orientation.get("flipud", False):
        iy = pixels - 1 - iy
    for _ in range(int(orientation.get("rot90", 0)) % 4):
        ix, iy = pixels - 1 - iy, ix  # inverse of (ix, iy) -> (iy, n-1-ix)
    return -half + (ix + 0.5) * res, -half + (iy + 0.5) * res


def quantise(height: np.ndarray, h_min: float, h_max: float) -> np.ndarray:
    """terrain.write_arena's 8-bit mapping, with the range supplied so arms share it."""
    return np.round((height - h_min) / (h_max - h_min) * 255.0).clip(0, 255).astype(np.uint8)


def shared_height_range(fields: list[np.ndarray]) -> tuple[float, float]:
    lo = min(float(f.min()) for f in fields)
    hi = max(float(f.max()) for f in fields)
    return math.floor(lo * 20.0) / 20.0, math.ceil(hi * 20.0) / 20.0


def write_bowl_arms(
    spec: BowlSpec,
    out_root: Path,
    orientation: dict[str, Any],
    arms: tuple[ArmSpec, ...] = DEFAULT_ARMS,
    prefix: str = "arena_bowl",
) -> dict[str, Path]:
    """Write every arm as an 8-bit BMP + ``arena_meta.json``, on one shared height range.

    Same contract as ``terrain.write_arena``, so ``TerrainMap.from_dir`` and
    ``scene.build_config`` load these arenas by exactly the same path as every other arena.
    """
    bowl, roughness, info = bowl_field(spec)
    fields = {arm.tag: arm_field(spec, bowl, roughness, arm) for arm in arms}
    h_min, h_max = shared_height_range(list(fields.values()))
    step = (h_max - h_min) / 255.0

    written: dict[str, Path] = {}
    for arm in arms:
        gray = quantise(fields[arm.tag], h_min, h_max)
        raw = _unapply_orientation(gray, orientation)
        assert np.array_equal(_apply_orientation(raw, orientation), gray), "orientation inverse"

        cx, cy = _unorient_xy(spec.centre_x_m, spec.centre_y_m, spec.size_m, spec.pixels, orientation)
        back = orient_xy(cx, cy, spec.size_m, spec.pixels, orientation)
        assert math.isclose(float(back[0]), spec.centre_x_m, abs_tol=1e-9) and math.isclose(
            float(back[1]), spec.centre_y_m, abs_tol=1e-9
        ), "feature coordinate inverse"

        out_dir = out_root / f"{prefix}_{arm.tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(raw, mode="L").save(out_dir / BMP_NAME)

        depth = arm_depth_m(spec, arm)
        quantized = h_min + gray.astype(np.float64) / 255.0 * (h_max - h_min)
        meta = {
            "size_m": spec.size_m,
            "pixels": spec.pixels,
            "resolution_m_per_px": spec.res_m,
            "height_min_m": h_min,
            "height_max_m": h_max,
            "quantization_step_m": step,
            "seed": spec.seed,
            "bmp": BMP_NAME,
            "orientation": orientation,
            # Generation-frame coordinates, as terrain.write_arena writes them: reading them
            # back through TerrainMap.features returns the simulated-frame centre above.
            "features": [
                {
                    "kind": "bowl",
                    "x_m": float(cx),
                    "y_m": float(cy),
                    "sigma_m": info["relief_sigma_m"],
                    "amplitude_m": depth,
                    "R_bottom_m": info["R_bottom_m"],
                    "R_top_m": info["R_top_m"],
                    "R_top_entry_m": info["R_top_entry_m"],
                }
            ],
            "slope_stats": _slope_stats(quantized, spec.res_m),
            "slope_cap": None,  # deliberately unrepaired: the wall must exceed the cap
            "bowl": {
                **asdict(spec),
                **{k: float(v) for k, v in info.items()},
                "arm": arm.tag,
                "arm_mode": arm.mode,
                "arm_depth_m": depth,
                "arm_wall_deg": (
                    math.degrees(math.atan(depth / spec.depth_m * math.tan(math.radians(spec.wall_deg))))
                    if arm.mode == "scale"
                    else spec.wall_deg
                ),
                "centre_sim_x_m": spec.centre_x_m,
                "centre_sim_y_m": spec.centre_y_m,
                "shared_height_range": [h_min, h_max],
            },
        }
        with (out_dir / META_NAME).open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        written[arm.tag] = out_dir
    return written


# --------------------------------------------------------------------------------------
# Measurement: everything below reads the WRITTEN BMP back through TerrainMap, never the
# float field, so the numbers are what Chrono's rigid terrain actually gets.
# --------------------------------------------------------------------------------------


def radial_profile_measured(
    tmap: TerrainMap, sector_halfwidth_deg: float = 45.0, r_max_pad_m: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    """Sector-averaged ``z(r)`` on the exit side (opposite the entry ramp) of the loaded map."""
    bowl = tmap.meta["bowl"]
    feat = tmap.features[0]
    cx, cy = float(feat["x_m"]), float(feat["y_m"])
    coords = -tmap.half + (np.arange(tmap.pixels) + 0.5) * tmap.res
    grid_x, grid_y = np.meshgrid(coords, coords)
    radius = np.hypot(grid_x - cx, grid_y - cy)
    azimuth = np.degrees(np.arctan2(grid_y - cy, grid_x - cx))
    exit_az = float(bowl["entry_azimuth_deg"]) + 180.0
    d_az = np.abs((azimuth - exit_az + 180.0) % 360.0 - 180.0)
    sector = d_az < sector_halfwidth_deg

    edges = np.arange(0.0, float(bowl["R_top_m"]) + r_max_pad_m, tmap.res)
    centres, means = [], []
    for i in range(len(edges) - 1):
        mask = sector & (radius >= edges[i]) & (radius < edges[i + 1])
        if int(mask.sum()) >= 3:
            centres.append(0.5 * (edges[i] + edges[i + 1]))
            means.append(float(tmap.height_grid[mask].mean()))
    return np.asarray(centres), np.asarray(means)


def measure_arena(arena_dir: Path) -> dict[str, float]:
    """Achieved geometry of a written arm, measured through ``TerrainMap`` on the BMP."""
    tmap = TerrainMap.from_dir(arena_dir)
    meta = tmap.meta
    bowl = meta["bowl"]
    feat = tmap.features[0]
    r, z = radial_profile_measured(tmap)

    plain = z[r > float(bowl["R_top_m"]) + 1.0]
    plain_level = float(plain.mean()) if plain.size else 0.0
    depth = plain_level - float(z.min())

    grade = np.gradient(z, r)
    angles = np.degrees(np.arctan(grade))
    # Wall angle over the constant-angle band only: between 20% and 80% of the descent, so
    # neither rounded corner is inside the fit.
    band = (z < plain_level - 0.2 * depth) & (z > plain_level - 0.8 * depth) & (r > 0.5)
    if int(band.sum()) >= 2:
        fit = float(np.polyfit(r[band], z[band], 1)[0])
        wall_band_deg = math.degrees(math.atan(abs(fit)))
    else:
        wall_band_deg = float("nan")
    # Worst grade a wheelbase-long chord sees (what the vehicle must actually climb).
    chord = 0.0
    for i in range(len(r)):
        j = int(np.searchsorted(r, r[i] + WHEELBASE_M))
        if j < len(r):
            chord = max(chord, (z[j] - z[i]) / (r[j] - r[i]))

    coords = -tmap.half + (np.arange(tmap.pixels) + 0.5) * tmap.res
    grid_x, grid_y = np.meshgrid(coords, coords)
    radius = np.hypot(grid_x - float(feat["x_m"]), grid_y - float(feat["y_m"]))
    off_bowl = radius > max(float(bowl["R_top_m"]), float(bowl["R_top_entry_m"])) + 2.0
    gy, gx = np.gradient(tmap.height_grid, tmap.res)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

    step = float(meta["quantization_step_m"])
    return {
        "arm": bowl["arm"],
        "requested_depth_m": float(bowl["arm_depth_m"]),
        "requested_wall_deg": float(bowl["arm_wall_deg"]),
        "measured_depth_m": float(depth),
        "measured_wall_deg": wall_band_deg,
        "measured_wall_max_deg": float(angles.max()),
        "chord_grade_deg": math.degrees(math.atan(chord)),
        "quant_step_m": step,
        "quant_steps_over_depth": float(depth / step) if step else float("nan"),
        "round_px": float(bowl["round_px"]),
        "plain_max_slope_deg": float(slope_deg[off_bowl].max()),
        "plain_p999_slope_deg": float(np.quantile(slope_deg[off_bowl], 0.999)),
        "arena_max_slope_deg": float(slope_deg.max()),
        "feature_relief_m": float(tmap.feature_relief_m(feat)),
        "R_top_m": float(bowl["R_top_m"]),
        "R_top_entry_m": float(bowl["R_top_entry_m"]),
        "exceeds_traction_limit": bool(math.tan(math.radians(wall_band_deg)) > TIRE_FRICTION),
        "exceeds_approach_angle": bool(wall_band_deg > APPROACH_ANGLE_DEG),
    }


def grey_difference_outside_rim(
    arena_a: Path, arena_b: Path, margin_m: float = 0.0
) -> dict[str, float]:
    """Max/count of 8-bit height differences outside the outermost rim between two arms.

    The rim radius is ``max(R_top, R_top_entry) + round_m``: the C1 rounding window extends
    one rounding radius past the nominal rim, so that annulus is still part of the feature.
    """
    a = TerrainMap.from_dir(arena_a)
    b = TerrainMap.from_dir(arena_b)
    meta = a.meta
    bowl = meta["bowl"]
    feat = a.features[0]
    coords = -a.half + (np.arange(a.pixels) + 0.5) * a.res
    grid_x, grid_y = np.meshgrid(coords, coords)
    radius = np.hypot(grid_x - float(feat["x_m"]), grid_y - float(feat["y_m"]))
    rim = max(float(bowl["R_top_m"]), float(bowl["R_top_entry_m"])) + float(bowl["round_m"])
    outside = radius > rim + margin_m
    ga = np.asarray(Image.open(arena_a / meta["bmp"]).convert("L"), dtype=np.int32)
    gb = np.asarray(Image.open(arena_b / b.meta["bmp"]).convert("L"), dtype=np.int32)
    diff = np.abs(ga - gb)[_unapply_orientation(outside, meta.get("orientation", {}))]
    return {
        "rim_radius_m": float(rim + margin_m),
        "pixels_compared": int(outside.sum()),
        "max_grey_diff": int(diff.max()) if diff.size else 0,
        "n_differing": int((diff > 0).sum()),
    }
