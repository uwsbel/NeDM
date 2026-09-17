#!/usr/bin/env python3
"""Compare physical heightfield centerlines; these are never model inputs.

Reads the same quantized BMP and orientation metadata used by the traversal
TerrainMap/Chrono scene. The line interpolation is explicitly TerrainMap's
bilinear lookup, not an assertion of exact triangle-mesh collision heights.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nedm.traverse.terrain import TerrainMap


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample_profile(arena: Path, center_xy, heading_deg: float, station,
                   grade_span_m: float) -> dict:
    terrain = TerrainMap.from_dir(arena)
    center = np.asarray(center_xy, dtype=float)
    heading = np.deg2rad(heading_deg)
    direction = np.array([np.cos(heading), np.sin(heading)])
    xy = center + np.asarray(station)[:, None] * direction
    margin = grade_span_m / 2
    if np.max(np.abs(xy)) + margin >= terrain.half - terrain.res / 2:
        raise ValueError("Profile or slope sample extends outside the heightfield")
    height = terrain.height(xy[:, 0], xy[:, 1])
    ahead = xy + margin * direction
    behind = xy - margin * direction
    grade = (terrain.height(ahead[:, 0], ahead[:, 1]) -
             terrain.height(behind[:, 0], behind[:, 1])) / grade_span_m
    peak = int(np.argmax(height))
    baseline = float((height[0] + height[-1]) / 2)
    rise = height - baseline
    ascent = rise[:peak + 1]
    relief = float(rise.max())
    def first_crossing(fraction):
        indices = np.flatnonzero(ascent >= fraction * relief)
        if not len(indices):
            return None
        j = int(indices[0])
        if j == 0 or ascent[j] == ascent[j - 1]:
            return float(station[j])
        alpha = (fraction * relief - ascent[j - 1]) / (ascent[j] - ascent[j - 1])
        return float(station[j - 1] + alpha * (station[j] - station[j - 1]))
    crossing10, crossing90 = first_crossing(.1), first_crossing(.9)
    native_ahead, native_behind = xy + terrain.res / 2 * direction, xy - terrain.res / 2 * direction
    native_grade = (terrain.height(native_ahead[:, 0], native_ahead[:, 1]) -
                    terrain.height(native_behind[:, 0], native_behind[:, 1])) / terrain.res
    active = np.flatnonzero(rise > max(.02, .01 * float(rise.max())))
    meta_path = arena / "arena_meta.json"
    bmp_path = arena / terrain.meta["bmp"]
    return {"station": np.asarray(station), "xy": xy, "height": height,
            "grade": grade, "slope_deg": np.rad2deg(np.arctan(grade)),
            "provenance": {
                "arena": str(arena.resolve()),
                "arena_meta_sha256": sha256(meta_path),
                "bmp_path": str(bmp_path.resolve()), "bmp_sha256": sha256(bmp_path),
                "center_xy_m": center.tolist(), "travel_heading_deg": heading_deg,
                "grid_resolution_m": terrain.res,
                "height_quantization_step_m": float((terrain.meta["height_max_m"] - terrain.meta["height_min_m"]) / 255),
                "orientation": terrain.meta.get("orientation", {}),
                "orientation_calibration_scope": terrain.meta.get("orientation_calibration_scope"),
                "orientation_provenance": terrain.meta.get("orientation_provenance"),
                "authored_feature": terrain.meta.get("smooth_hill", terrain.meta.get("authored_training_probe", terrain.meta.get("features", []))),
                "peak_elevation_m": float(height[peak]),
                "peak_station_m": float(station[peak]),
                "endpoint_mean_elevation_m": baseline,
                "relief_above_endpoint_mean_m": float(height[peak] - baseline),
                "positive_grade_max_deg": float(np.rad2deg(np.arctan(grade.max()))),
                "negative_grade_min_deg": float(np.rad2deg(np.arctan(grade.min()))),
                "one_grid_span_positive_grade_max_deg": float(np.rad2deg(np.arctan(native_grade.max()))),
                "one_grid_span_m": terrain.res,
                "ascent_10pct_station_m": crossing10, "ascent_90pct_station_m": crossing90,
                "ascent_10_to_90pct_width_m": crossing90 - crossing10 if crossing10 is not None and crossing90 is not None else None,
                "ascent_width_definition": "First rising-side 10% to 90% of relief above endpoint mean; linear interpolation between samples.",
                "approx_active_width_m": float(station[active[-1]] - station[active[0]]) if len(active) else 0.,
                "active_width_definition": "Elevation above endpoint mean exceeds max(0.02m, 1% of relief).",
            }}


def render(profiles, labels, out: Path, grade_span_m: float):
    colors = ("#9d5764", "#197f85")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1]})
    fig.subplots_adjust(left=.10, right=.965, bottom=.18, top=.85, hspace=.16)
    fig.text(.10, .95, "Terrain shape along the direction of travel", fontsize=20, weight="bold", color="#203047")
    fig.text(.10, .906, "Original berm and revised smooth hill | actual quantized heightfield", fontsize=12, color="#5a6473")
    for p, label, color in zip(profiles, labels, colors):
        info = p["provenance"]
        width = info["ascent_10_to_90pct_width_m"]
        details = f"{info['relief_above_endpoint_mean_m']:.2f} m relief; 10-90% rise over {width:.2f} m" if width is not None else "No complete rising section"
        axes[0].plot(p["station"], p["height"], color=color, lw=2.3, label=label + "\n" + details)
        axes[1].plot(p["station"], p["slope_deg"], color=color, lw=2.1,
                     label=f"{label}: peak uphill {info['positive_grade_max_deg']:.1f}°")
    for ax in axes:
        ax.axvline(0, color="#7d8794", lw=.8, ls=":")
        ax.grid(axis="both", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Elevation (m)")
    axes[0].legend(loc="upper right", framealpha=.94, fontsize=9)
    axes[0].margins(y=.16)
    axes[1].axhline(0, color="#7d8794", lw=.8)
    axes[1].legend(loc="upper right", framealpha=.94, fontsize=9)
    axes[1].set_ylabel("Signed slope (degrees)")
    axes[1].set_xlabel("Distance from each feature center along travel (m)  →")
    peak_grade = max(float(np.abs(p["slope_deg"]).max()) for p in profiles)
    axes[1].set_ylim(-max(5., peak_grade * 1.15), max(5., peak_grade * 1.15))
    axes[1].set_xlim(profiles[0]["station"][[0, -1]])
    coordinates = "; ".join(f"{label}: center ({p['provenance']['center_xy_m'][0]:g}, {p['provenance']['center_xy_m'][1]:g}) m, heading {p['provenance']['travel_heading_deg']:g}°"
                            for p, label in zip(profiles, labels))
    fig.text(.10, .108, coordinates, fontsize=8.5, color="#5a6473")
    fig.text(.10, .061, f"Slope = atan(rise / run) over a centered {grade_span_m:g} m span. Heights use the stored BMP and calibrated orientation;\n"
             "bilinear sampling may differ slightly from Chrono's terrain triangles. No additional curve smoothing is applied.", fontsize=9, color="#5a6473")
    fig.text(.10, .025, "Geometry is shown only to verify the physical scene. It is not a model input; slope alone does not establish a vehicle stall.", fontsize=9, color="#5a6473")
    fig.savefig(out / "terrain_profile_comparison.png", dpi=180, facecolor="white")
    fig.savefig(out / "terrain_profile_comparison.pdf", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-arena", type=Path, required=True)
    ap.add_argument("--new-arena", type=Path, required=True)
    ap.add_argument("--old-center-xy", type=float, nargs=2, required=True)
    ap.add_argument("--new-center-xy", type=float, nargs=2, required=True)
    ap.add_argument("--old-heading-deg", type=float, default=0.)
    ap.add_argument("--new-heading-deg", type=float, default=0.)
    ap.add_argument("--old-label", default="Original steep berm")
    ap.add_argument("--new-label", default="Smooth steep hill")
    ap.add_argument("--half-length-m", type=float, default=12.)
    ap.add_argument("--sample-step-m", type=float, default=.05)
    ap.add_argument("--grade-span-m", type=float, default=.5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if min(args.half_length_m, args.sample_step_m, args.grade_span_m) <= 0:
        raise ValueError("All distance parameters must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a new output directory to preserve previous diagnostic artifacts")
    args.out.mkdir(parents=True, exist_ok=True)
    station = np.linspace(-args.half_length_m, args.half_length_m,
                          int(np.ceil(2 * args.half_length_m / args.sample_step_m)) + 1)
    profiles = [sample_profile(arena, center, heading, station, args.grade_span_m)
                for arena, center, heading in (
                    (args.old_arena, args.old_center_xy, args.old_heading_deg),
                    (args.new_arena, args.new_center_xy, args.new_heading_deg))]
    labels = [args.old_label, args.new_label]
    with (args.out / "terrain_profiles.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["profile", "station_m", "world_x_m", "world_y_m", "elevation_m", "grade_rise_per_run", "slope_deg"])
        for p, label in zip(profiles, labels):
            for s, xy, h, g, angle in zip(p["station"], p["xy"], p["height"], p["grade"], p["slope_deg"]):
                writer.writerow([label, float(s), *map(float, xy), float(h), float(g), float(angle)])
    render(profiles, labels, args.out, args.grade_span_m)
    outputs = ("terrain_profiles.csv", "terrain_profile_comparison.png", "terrain_profile_comparison.pdf")
    manifest = {"schema": 1, "purpose": "Physical-scene verification; never a scorer input or stall label",
                "sampling": {"height": "TerrainMap bilinear lookup of quantized and oriented BMP",
                             "sample_step_m": float(station[1] - station[0]),
                             "grade": "atan centered elevation difference divided by grade_span_m",
                             "grade_span_m": args.grade_span_m, "curve_smoothing": "none"},
                "profiles": [{"label": label, **p["provenance"]} for p, label in zip(profiles, labels)],
                "script_sha256": sha256(Path(__file__)),
                "outputs_sha256": {name: sha256(args.out / name) for name in outputs}}
    (args.out / "terrain_profile_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
