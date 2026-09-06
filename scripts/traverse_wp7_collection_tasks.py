#!/usr/bin/env python
"""Collection layouts and runs for the learning comparison (plan §28 step 1), one arena at a time.

Per arena three kinds of layout, all tracker-driven in Chrono (``traverse_wp3_chrono_eval.py --tasks-file``
with ``record``: every run's 20 Hz rows are kept whatever the outcome):

* **crossings** of every terrain feature (hills also on the shoulder = side slope) at several headings:
  ``direct_v{2..9}`` constant-speed lines, ``slope_aware`` (the rule-based profile) and two detours -- the
  feasibility map of the pilot, now on many terrain instances (momentum climbs, speed-penalised crossings,
  detour-only craters);
* **sequences**: a climb through feature A followed by a turn onto / across feature B (centre or shoulder), with
  a two-speed profile (v1 through A, v2 from the braking zone before B): the momentum-then-braking decision the
  single crossings never exercise. The curvature cap is relaxed so fast entries into the turn are driven too;
* **free-form** sampled routes between random flat start / goal pairs at random cruise speeds (action diversity).

Writes ``<out>/<layout>/meta.json`` (runner + planner layout, kind, parameters), one frame-0 camera dump per layout
(the first run of the layout), ``<out>/tasks.json`` and ``<out>/layouts.json``.
"""
from __future__ import annotations

import argparse, json, math, sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nedm.traverse.layout import Asset, EpisodeLayout, LayoutParams
from nedm.traverse.oracle import OracleGrid, PlannerParams, speed_profile
from nedm.traverse.planner_s import _catmull_rom, _resample, sample_routes
from nedm.traverse.terrain import TerrainMap
from traverse_wp6_challenges import const_profile, flat, route

SPEEDS = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
SEQ_SPEEDS = ((3.0, 3.0), (8.0, 8.0), (8.0, 3.0), (5.0, 7.0), (7.0, 4.0), (9.0, 5.0), (4.0, 8.0), (6.0, 6.0))


def two_speed_profile(pts: np.ndarray, v1: float, v2: float, s_switch: float, params: PlannerParams) -> np.ndarray:
    """v1 until station ``s_switch``, v2 after (deceleration begins before it at a_decel); curvature cap, launch ramp,
    terminal taper as ``const_profile``."""
    seg = np.hypot(*np.diff(pts, axis=0).T); station = np.concatenate([[0.0], np.cumsum(seg)])
    a, b, c = pts[:-2], pts[1:-1], pts[2:]
    ab, bc, ac = b - a, c - b, c - a
    cross = np.abs(ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]); den = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ac, axis=1)
    kappa = np.concatenate([[0.0], np.where(den > 1e-9, 2 * cross / np.maximum(den, 1e-9), 0.0), [0.0]])
    vv = np.where(station < s_switch, v1, v2).astype(float)
    vv = np.minimum(vv, np.sqrt(params.a_lat_max / np.maximum(kappa, 1e-6)))
    taper = np.clip((station[-1] - station) / params.terminal_taper_m, 0.0, 1.0)
    vv = vv * taper + params.v_terminal_mps * (1.0 - taper)
    vv[0] = min(vv[0], params.v_launch_mps)
    for i in range(1, len(vv)):
        vv[i] = min(vv[i], math.sqrt(vv[i - 1] ** 2 + 2.0 * params.a_accel * seg[i - 1]))
    for i in range(len(vv) - 2, -1, -1):
        vv[i] = min(vv[i], math.sqrt(vv[i + 1] ** 2 + 2.0 * params.a_decel * seg[i]))
    return vv


def make_layout(cid: str, start, house, th_start: float, th_house: float, lp: LayoutParams) -> EpisodeLayout:
    hl, hw, hh = lp.house_size_m
    house_r = 0.5 * math.hypot(hl, hw) + 0.3
    return EpisodeLayout(episode_id=cid, seed=0, assets=[Asset(kind="house", x_m=float(house[0]), y_m=float(house[1]), yaw_rad=th_house,
                                                               footprint_radius_m=house_r, dims={"length_m": hl, "width_m": hw, "wall_height_m": hh})],
                         house_xy=(float(house[0]), float(house[1])), house_yaw=th_house, start_xy=(float(start[0]), float(start[1])), start_yaw=th_start)


def search(tmap, origin, u, d0, d1, bound, smax):
    return next((origin + d * u for d in np.arange(d0, d1, 1.0) if flat(tmap, origin + d * u, bound, smax)), None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arena", required=True, help="e.g. assets/traverse/arena_f101")
    ap.add_argument("--out", required=True, help="e.g. artifacts/traverse/wp7_collect_f101")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headings", type=int, nargs="+", default=[0, 90, 180, 270])
    ap.add_argument("--speeds", type=float, nargs="+", default=list(SPEEDS))
    ap.add_argument("--n-sequences", type=int, default=12)
    ap.add_argument("--seq-speeds", type=int, default=6, help="(v1, v2) combinations per sequence layout")
    ap.add_argument("--n-freeform", type=int, default=8)
    ap.add_argument("--freeform-routes", type=int, default=6)
    ap.add_argument("--bound", type=float, default=36.0)
    ap.add_argument("--place-slope-deg", type=float, default=14.0)
    ap.add_argument("--detour-speed", type=float, default=5.0)
    ap.add_argument("--seq-a-lat-max", type=float, default=5.0, help="relaxed curvature cap for the sequence profiles (fast turn entries are driven)")
    ap.add_argument("--wide-detours", action="store_true",
                    help="add detour_wide_L/R (radius 4 sigma, >= 12 m, 4 m/s) to every crossing: the 'no candidate works' task needs wider alternatives")
    args = ap.parse_args()
    out, arena = Path(args.out), Path(args.arena)
    arena_id = arena.name
    rng = np.random.default_rng(args.seed + 7919 * int(arena.name.split("_f")[-1]) if "_f" in arena.name else args.seed)
    tmap = TerrainMap.from_dir(arena)
    params = PlannerParams()
    grid = OracleGrid(tmap, [], params)
    lp = LayoutParams()
    smax = math.tan(math.radians(args.place_slope_deg))
    features = tmap.meta["features"]
    tasks, layouts, skipped = [], [], []

    def add_layout(cid: str, layout: EpisodeLayout, kind: str, info: dict, runs: list[tuple[str, dict]]):
        d = out / cid; d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps({"layout": layout.to_json(), "kind": kind, "arena": str(arena), **info}, indent=1))
        layouts.append({"id": cid, "kind": kind, "arena": arena_id, "meta_path": str(d / "meta.json"), "n_runs": len(runs), **info})
        for i, (cand, rt) in enumerate(runs):
            t = {"key": cid, "meta_path": str(d / "meta.json"), "candidate": cand, "route": rt, "arena": str(arena),
                 "record": str(out / "records" / f"{cid}__{cand}.npz")}
            if i == 0:
                t["dump_frame0"] = str(out / "frame0" / f"{cid}.npz")
            tasks.append(t)

    # ---- crossings
    for fi, f in enumerate(features):
        c = np.array([f["x_m"], f["y_m"]]); sigma = float(f["sigma_m"])
        variants = [(h, 0.0) for h in args.headings]
        if f["kind"] == "hill":
            variants += [(h, sigma) for h in args.headings[::2]]
        for hdeg, offset in variants:
            th = math.radians(hdeg); u = np.array([math.cos(th), math.sin(th)]); n = np.array([-u[1], u[0]])
            centre = c + offset * n
            start = search(tmap, centre, -u, 18.0, 33.0, args.bound, smax)
            house = search(tmap, centre, u, 17.0, 29.0, args.bound - 3.0, smax)
            if start is None or house is None:
                skipped.append((f["kind"], fi, hdeg, offset)); continue
            end = house - params.approach_ring_m * u
            cid = f"{arena_id}__x_{f['kind']}{fi}_h{hdeg:03d}" + ("_sh" if offset else "")
            layout = make_layout(cid, start, house, th, th, lp)
            line = _resample(np.stack([start, end]), params.sample_step_m)
            gx, gy = tmap.gradient(line[:, 0], line[:, 1]); along = gx * u[0] + gy * u[1]
            hts = tmap.height(line[:, 0], line[:, 1])
            runs = [(f"direct_v{v:.0f}", route(line, const_profile(line, v, params))) for v in args.speeds]
            runs.append(("slope_aware", route(line, speed_profile(line, grid, params))))
            radii = [("detour", max(2.5 * sigma, 8.0), args.detour_speed)]
            if args.wide_detours:
                radii.append(("detour_wide", max(4.0 * sigma, 12.0), 4.0))
            for tag, R, v_det in radii:
                for side, sgn in (("L", 1.0), ("R", -1.0)):
                    ctrl = np.stack([start, centre - 14 * u, centre - 7 * u + sgn * R * n, centre + sgn * R * n, centre + 7 * u + sgn * R * n, centre + 11 * u + sgn * 0.3 * R * n, end])
                    pts = _resample(_catmull_rom(ctrl[None], 240)[0], params.sample_step_m)
                    if np.abs(pts).max() > params.arena_keep_within_m:  # a wide loop may leave the arena
                        continue
                    runs.append((f"{tag}_{side}", route(pts, const_profile(pts, v_det, params))))
            add_layout(cid, layout, "crossing", {"feature": f, "feature_index": fi, "heading_deg": hdeg, "offset_m": offset,
                                                "line_max_up_deg": float(np.degrees(np.arctan(max(along.max(), 0)))),
                                                "line_max_down_deg": float(np.degrees(np.arctan(max(-along.min(), 0)))),
                                                "line_relief_m": float(hts.max() - hts.min())}, runs)

    # ---- sequences: through A, then turn onto / across B
    seq_params = replace(params, a_lat_max=args.seq_a_lat_max)
    pairs = []
    for i, fa in enumerate(features):
        for j, fb in enumerate(features):
            if i == j:
                continue
            d = math.hypot(fa["x_m"] - fb["x_m"], fa["y_m"] - fb["y_m"])
            if 14.0 <= d <= 28.0:
                pairs.append((i, j, d))
    rng.shuffle(pairs)
    n_seq = 0
    for i, j, d in pairs:
        if n_seq >= args.n_sequences:
            break
        fa, fb = features[i], features[j]
        A, B = np.array([fa["x_m"], fa["y_m"]]), np.array([fb["x_m"], fb["y_m"]])
        u1 = (B - A) / d; n1 = np.array([-u1[1], u1[0]])
        offset = float(rng.choice([0.0, fb["sigma_m"], -fb["sigma_m"]]))
        turn = float(rng.choice([-60.0, -35.0, 35.0, 60.0]))
        th2 = math.atan2(u1[1], u1[0]) + math.radians(turn); u2 = np.array([math.cos(th2), math.sin(th2)])
        P = B + offset * n1
        start = search(tmap, A, -u1, 16.0, 31.0, args.bound, smax)
        house = search(tmap, P, u2, 17.0, 29.0, args.bound - 3.0, smax)
        if start is None or house is None:
            skipped.append(("seq", i, j)); continue
        end = house - params.approach_ring_m * u2
        ctrl = np.stack([start, A - 6 * u1, A, A + 6 * u1, P, P + 6 * u2, end])
        pts = _resample(_catmull_rom(ctrl[None], 300)[0], params.sample_step_m)
        seg = np.hypot(*np.diff(pts, axis=0).T); station = np.concatenate([[0.0], np.cumsum(seg)])
        s_P = float(station[np.argmin(np.hypot(pts[:, 0] - P[0], pts[:, 1] - P[1]))])
        s_switch = max(s_P - 8.0, 0.0)  # braking zone: 8 m before B
        cid = f"{arena_id}__s_{fa['kind']}{i}_{fb['kind']}{j}_t{int(turn):+d}" + ("_sh" if offset else "")
        layout = make_layout(cid, start, house, math.atan2(u1[1], u1[0]), th2, lp)
        combos = [SEQ_SPEEDS[k] for k in rng.choice(len(SEQ_SPEEDS), size=min(args.seq_speeds, len(SEQ_SPEEDS)), replace=False)]
        runs = [(f"seq_v{v1:.0f}_{v2:.0f}", route(pts, two_speed_profile(pts, v1, v2, s_switch, seq_params))) for v1, v2 in combos]
        runs.append(("slope_aware", route(pts, speed_profile(pts, grid, params))))
        gx, gy = tmap.gradient(pts[:, 0], pts[:, 1]); tang = np.gradient(pts, axis=0); tn = tang / np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
        along = gx * tn[:, 0] + gy * tn[:, 1]; cross = -gx * tn[:, 1] + gy * tn[:, 0]
        add_layout(cid, layout, "sequence", {"feature_a": fa, "feature_b": fb, "pair_distance_m": d, "offset_m": offset, "turn_deg": turn,
                                            "s_switch_m": s_switch, "length_m": float(station[-1]),
                                            "max_up_deg": float(np.degrees(np.arctan(max(along.max(), 0)))), "max_cross_deg": float(np.degrees(np.arctan(np.abs(cross).max())))}, runs)
        n_seq += 1

    # ---- free-form sampled routes
    n_free = 0
    for attempt in range(200):
        if n_free >= args.n_freeform:
            break
        s = np.array([rng.uniform(-args.bound + 2, args.bound - 2), rng.uniform(-args.bound + 2, args.bound - 2)])
        h = np.array([rng.uniform(-args.bound + 5, args.bound - 5), rng.uniform(-args.bound + 5, args.bound - 5)])
        dist = float(np.linalg.norm(h - s))
        if not (30.0 <= dist <= 55.0) or not flat(tmap, s, args.bound, smax) or not flat(tmap, h, args.bound - 3.0, smax):
            continue
        th = math.atan2(*(h - s)[::-1])
        cid = f"{arena_id}__f_{n_free:02d}"
        layout = make_layout(cid, s, h, th, th, lp)
        cands, _ = sample_routes(s, th, h, 60, rng, tmap, layout.obstacles(), params, lateral_frac=(0.04, 0.18), v_range=(2.0, 9.0))
        if len(cands) < args.freeform_routes:
            continue
        order = np.argsort([c.meta["v_cruise"] for c in cands])
        pick = [cands[k] for k in order[np.linspace(0, len(cands) - 1, args.freeform_routes).astype(int)]]  # spread over cruise speed
        runs = [(f"free_{k}_v{c.meta['v_cruise']:.0f}", {"waypoints": c.waypoints.tolist(), "speeds": c.speeds.tolist(), "headings": c.headings.tolist(), "stations": c.stations.tolist()})
                for k, c in enumerate(pick)]
        add_layout(cid, layout, "freeform", {"distance_m": dist}, runs)
        n_free += 1

    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.json").write_text(json.dumps(tasks))
    (out / "layouts.json").write_text(json.dumps(layouts, indent=1))
    kinds = {k: sum(l["kind"] == k for l in layouts) for k in ("crossing", "sequence", "freeform")}
    print(f"{arena_id}: {len(layouts)} layouts {kinds}, {len(tasks)} runs, skipped {len(skipped)} -> {out}")
    for l in layouts:
        if l["kind"] == "sequence":
            print(f"  {l['id']:44s} d={l['pair_distance_m']:4.1f} turn {l['turn_deg']:+.0f} off {l['offset_m']:+.1f} | up {l['max_up_deg']:4.1f} cross {l['max_cross_deg']:4.1f} deg | {l['length_m']:.0f} m")


if __name__ == "__main__":
    main()
