#!/usr/bin/env python
"""Terrain challenges for the pilot (plan §24 step 1): parameterised crossings of arena_v1's craters and hills.

For every terrain feature in ``arena_meta.json`` and every approach heading, one layout: the vehicle starts
22 m before the feature centre, the goal house stands 20 m beyond it, no other assets. Hills also get
"shoulder" layouts whose line is offset by one sigma (a side-slope crossing). Per layout the sweep drives:

* ``direct_v{v}`` — straight through the centre at a constant cruise speed v (launch ramp / terminal taper as
  the oracle's profile), v = 2 … 8 m/s: is the crossing feasible, at what cost, where does it fail?
* ``slope_aware`` — the same line with the oracle's rule-based slope-capped profile (7 m/s cruise): the
  classical planner's answer;
* ``detour_L`` / ``detour_R`` — around the feature at 5 m/s: the alternative the planners could prefer.

Writes ``<out>/<challenge>/meta.json`` (runner + planner layout), ``<out>/tasks.json`` for
``traverse_wp3_chrono_eval.py --tasks-file`` (the slope-aware run also dumps the t = 0 camera frame and rest
state for the live planning inputs) and ``<out>/challenges.json``.
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nedm.traverse.layout import Asset, EpisodeLayout, LayoutParams
from nedm.traverse.oracle import OracleGrid, PlannerParams, speed_profile
from nedm.traverse.planner_s import _catmull_rom, _resample
from nedm.traverse.terrain import TerrainMap

SPEEDS = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
HEADINGS_DEG = (0, 45, 90, 135, 180, 225, 270, 315)
STANDOFF_M, BEYOND_M = 18.0, 17.0  # minimum start distance before the centre / house distance beyond it (searched outwards for flat ground)


def const_profile(pts: np.ndarray, v: float, params: PlannerParams) -> np.ndarray:
    """Constant cruise speed with the oracle's curvature cap, launch ramp and terminal taper."""
    tang = np.gradient(pts, axis=0)
    a, b, c = pts[:-2], pts[1:-1], pts[2:]
    ab, bc, ac = b - a, c - b, c - a
    cross = np.abs(ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]); den = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ac, axis=1)
    kappa = np.concatenate([[0.0], np.where(den > 1e-9, 2 * cross / np.maximum(den, 1e-9), 0.0), [0.0]])
    vv = np.minimum(v, np.sqrt(params.a_lat_max / np.maximum(kappa, 1e-6)))
    seg = np.hypot(*np.diff(pts, axis=0).T); station = np.concatenate([[0.0], np.cumsum(seg)])
    taper = np.clip((station[-1] - station) / params.terminal_taper_m, 0.0, 1.0)
    vv = vv * taper + params.v_terminal_mps * (1.0 - taper)
    vv[0] = min(vv[0], params.v_launch_mps)
    for i in range(1, len(vv)):
        vv[i] = min(vv[i], math.sqrt(vv[i - 1] ** 2 + 2.0 * params.a_accel * seg[i - 1]))
    for i in range(len(vv) - 2, -1, -1):
        vv[i] = min(vv[i], math.sqrt(vv[i + 1] ** 2 + 2.0 * params.a_decel * seg[i]))
    return vv


def flat(tmap: TerrainMap, p: np.ndarray, bound: float, slope_max: float = math.tan(math.radians(10.0))) -> bool:
    """Same test as the layout generator's placement: centre + 4 offsets under 8 degrees, inside the keep-within bound."""
    if np.abs(p).max() > bound:
        return False
    xs = np.array([p[0], p[0] + 1.5, p[0] - 1.5, p[0], p[0]]); ys = np.array([p[1], p[1], p[1], p[1] + 1.5, p[1] - 1.5])
    return float(tmap.slope(xs, ys).max()) <= slope_max


def route(pts: np.ndarray, speeds: np.ndarray) -> dict:
    tang = np.gradient(pts, axis=0)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    return {"waypoints": pts.tolist(), "speeds": speeds.tolist(), "headings": np.arctan2(tang[:, 1], tang[:, 0]).tolist(),
            "stations": np.concatenate([[0.0], np.cumsum(seg)]).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="artifacts/traverse/wp6_challenges")
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--bound", type=float, default=36.0, help="start / house must lie within +-bound (arena keep-within)")
    ap.add_argument("--detour-speed", type=float, default=5.0)
    args = ap.parse_args()
    out, arena = Path(args.out), Path(args.arena)
    tmap = TerrainMap.from_dir(arena)
    params = PlannerParams()
    grid = OracleGrid(tmap, [], params)
    lp = LayoutParams(); hl, hw, hh = lp.house_size_m
    house_r = 0.5 * math.hypot(hl, hw) + 0.3
    features = tmap.meta["features"]
    tasks, challenges, skipped = [], [], []
    for fi, f in enumerate(features):
        c = np.array([f["x_m"], f["y_m"]]); sigma = float(f["sigma_m"])
        variants = [(h, 0.0) for h in HEADINGS_DEG]
        if f["kind"] == "hill":
            variants += [(h, sigma) for h in (0, 90, 180, 270)]  # shoulder crossings: side slope
        for hdeg, offset in variants:
            th = math.radians(hdeg); u = np.array([math.cos(th), math.sin(th)]); n = np.array([-u[1], u[0]])
            centre = c + offset * n
            # start and house on ground as flat as the regular layouts demand (the vehicle spawns 0.75 m up and settles)
            start = next((centre - d * u for d in np.arange(STANDOFF_M, 33.0, 1.0) if flat(tmap, centre - d * u, args.bound)), None)
            house = next((centre + d * u for d in np.arange(BEYOND_M, 29.0, 1.0) if flat(tmap, centre + d * u, args.bound - 3.0)), None)
            if start is None or house is None:
                skipped.append((f["kind"], fi, hdeg, offset)); continue
            end = house - params.approach_ring_m * u
            cid = f"ch_{f['kind']}{fi}_h{hdeg:03d}" + ("_shoulder" if offset else "")
            layout = EpisodeLayout(episode_id=cid, seed=0, assets=[Asset(kind="house", x_m=float(house[0]), y_m=float(house[1]), yaw_rad=th,
                                                                          footprint_radius_m=house_r, dims={"length_m": hl, "width_m": hw, "wall_height_m": hh})],
                                   house_xy=(float(house[0]), float(house[1])), house_yaw=th, start_xy=(float(start[0]), float(start[1])), start_yaw=th)
            d = out / cid; d.mkdir(parents=True, exist_ok=True)
            (d / "meta.json").write_text(json.dumps({"layout": layout.to_json(), "challenge": {"feature": f, "heading_deg": hdeg, "offset_m": offset}}, indent=1))
            line = _resample(np.stack([start, end]), params.sample_step_m)
            hts = tmap.height(line[:, 0], line[:, 1])
            gx, gy = tmap.gradient(line[:, 0], line[:, 1]); along = gx * u[0] + gy * u[1]
            challenges.append({"id": cid, "kind": f["kind"], "feature": fi, "heading_deg": hdeg, "offset_m": offset, "sigma_m": sigma, "amplitude_m": f["amplitude_m"],
                               "standoff_m": float(np.linalg.norm(centre - start)), "beyond_m": float(np.linalg.norm(house - centre)),
                               "line_max_up_deg": float(np.degrees(np.arctan(max(along.max(), 0)))), "line_max_down_deg": float(np.degrees(np.arctan(max(-along.min(), 0)))),
                               "line_relief_m": float(hts.max() - hts.min()), "meta_path": str(d / "meta.json")})
            base = {"key": cid, "meta_path": str(d / "meta.json")}
            for v in SPEEDS:
                tasks.append({**base, "candidate": f"direct_v{v:.0f}", "route": route(line, const_profile(line, v, params))})
            tasks.append({**base, "candidate": "slope_aware", "route": route(line, speed_profile(line, grid, params)), "dump_frame0": str(d / "frame0.npz")})
            R = max(2.5 * sigma, 8.0)
            for side, sgn in (("L", 1.0), ("R", -1.0)):
                ctrl = np.stack([start, centre - 14 * u, centre - 7 * u + sgn * R * n, centre + sgn * R * n, centre + 7 * u + sgn * R * n, centre + 11 * u + sgn * 0.3 * R * n, end])
                pts = _resample(_catmull_rom(ctrl[None], 240)[0], params.sample_step_m)
                tasks.append({**base, "candidate": f"detour_{side}", "route": route(pts, const_profile(pts, args.detour_speed, params))})
    (out / "tasks.json").write_text(json.dumps(tasks))
    (out / "challenges.json").write_text(json.dumps(challenges, indent=1))
    print(f"skipped {len(skipped)} variants with no flat start / house")
    print(f"{len(challenges)} challenge layouts ({sum(c['kind'] == 'crater' for c in challenges)} crater, {sum(c['kind'] == 'hill' for c in challenges)} hill), {len(tasks)} runs -> {out}")
    for c in challenges:
        print(f"  {c['id']:26s} standoff {c['standoff_m']:4.1f} beyond {c['beyond_m']:4.1f} | up {c['line_max_up_deg']:4.1f} deg  down {c['line_max_down_deg']:4.1f} deg  relief {c['line_relief_m']:.2f} m")


if __name__ == "__main__":
    main()
