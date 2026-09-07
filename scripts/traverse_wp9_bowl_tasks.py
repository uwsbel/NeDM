#!/usr/bin/env python
"""B1 confirmation and B2 collection task files for the controlled bowl (energy/crater plan, section B).

Emits ``tasks.json`` files for ``scripts/traverse_wp3_chrono_eval.py --tasks-file`` on the three bowl
arenas written by ``scripts/traverse_wp9_bowl_arena.py`` (``assets/traverse/arena_bowl_{deep,shallow,flat}``).

Two products, one geometry:

**B1 (``--out-b1``, default ``artifacts/traverse/wp9_bowl_b1``)** — the recorded, reproducible version of
the scratch sweep that chose the 47 deg / 3.2 m bowl.  96 runs:

===========================================  ====  ==========================================================
group                                        runs  purpose
===========================================  ====  ==========================================================
``core``   3 arms x 4 headings x 3 levels     72    the trap claim: deep does not escape, shallow/flat cross
           x 2 control families                     (levels = commanded 2 / 4 / 6 m/s)
``excl``   deep x 4 headings x level 8 x 2      8    the EXCLUDED arm: 8 m/s pitches the vehicle over.  Run and
                                                     recorded so the exclusion is documented by data
``excl``   shallow+flat x heading 0 x level 8   4    the same level on the two controls (it is driveable there)
``detour`` deep, exterior route, L and R        2    the alternative that must succeed on the deep arena
``pert``   2 boundary cases x 3 perturbed ICs   6    small initial-condition perturbations (plan B1)
``cal``    flat, open-loop throttle staircase   1    measures approach speed vs throttle: defines the schedule
                                                     family's four levels (feed back with --schedule-throttles)
``cam``    3 arms, heading 0, 0.25 s            3    frame-0 RGB-D dump: the bowl as the camera sees it
===========================================  ====  ==========================================================

**B2 (``--out-b2``, default ``artifacts/traverse/wp9_bowl_b2``)** — the tiny controlled corpus: the full
3 surfaces x 4 headings x 4 speed levels x 2 control families = 96 episodes, plus 4 verified detours and
12 frame-0 camera dumps.  84 of the 96 matrix episodes, 2 of the 4 detours and 3 of the 12 dumps are
LITERALLY the B1 runs -- their ``record`` path points into ``<out-b1>/records`` -- so with
``--skip-existing`` only the 23 genuinely new runs execute.  ``episodes.json`` says which is which.

Both control families:

* ``tracker``  -- the existing WP3 tracker (``--tracker``) following a straight route through the bowl
  centre at a constant commanded speed (the oracle launch ramp / curvature cap / terminal taper);
* ``schedule`` -- an independently specified open-loop schedule: zero steering, no braking, a constant
  approach throttle held until ``--schedule-full-at-s``, then SUSTAINED FULL THROTTLE to the horizon.
  Nothing in this arm reads the vehicle state, so it is a genuine control intervention (plan B2).

Every run: 60 s horizon, stall early-abort OFF, off-route cutoff OFF, rollover/pitchover abort at a
generous 85 deg (trapped runs legitimately sit at 77-80 deg pitch nose-down against the 47 deg wall;
the 8 m/s pitch-over exceeds 89 deg), and 20 Hz recording of state / actions / pose / vertical position /
power / termination reason / valid length.

Usage::

  PYTHONPATH=src python scripts/traverse_wp9_bowl_tasks.py            # writes both tasks files
  PYTHONPATH=src python scripts/traverse_wp9_bowl_tasks.py --schedule-throttles 0.18 0.30 0.48 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from nedm.traverse.layout import Asset, EpisodeLayout, LayoutParams
from nedm.traverse.oracle import PlannerParams
from nedm.traverse.planner_s import _resample
from nedm.traverse.terrain import TerrainMap
from traverse_wp6_challenges import const_profile, route

ARMS = ("deep", "shallow", "flat")
CLAIM_SPEEDS = (2.0, 4.0, 6.0)      # the narrowed claim
EXCLUDED_SPEED = 8.0                # pitches the vehicle over on the deep arena; run, recorded, excluded
HEADINGS_DEG = (0.0, 12.0, 348.0, 24.0)  # direction of travel; 0 = straight down the entry-ramp axis
WHEELBASE_M = 3.378


# ----------------------------------------------------------------------------- geometry helpers
def unit(deg: float) -> tuple[np.ndarray, np.ndarray]:
    th = math.radians(deg)
    u = np.array([math.cos(th), math.sin(th)])
    return u, np.array([-u[1], u[0]])


def direct_line(u: np.ndarray, r_start: float, r_end: float, step: float) -> np.ndarray:
    s = np.arange(-r_start, r_end + 1e-9, step)
    return s[:, None] * u[None, :]


def detour_line(u: np.ndarray, n: np.ndarray, r_start: float, r_end: float, offset_m: float,
                side: float, step: float) -> np.ndarray:
    """Raised-cosine lateral excursion around the bowl: tangent to the approach heading at both ends
    (so the start pose and the goal approach are the direct route's), zero curvature there, and a single
    smooth hump of amplitude ``offset_m``.  Curvature is bounded by ``offset_m * 0.5 * (2 pi / L)^2``."""
    s = np.arange(-r_start, r_end + 1e-9, 0.25)
    t = (s + r_start) / (r_end + r_start)
    lat = side * offset_m * 0.5 * (1.0 - np.cos(2.0 * np.pi * t))
    return _resample(s[:, None] * u[None, :] + lat[:, None] * n[None, :], step)


def chord_grade_deg(tmap: TerrainMap, pts: np.ndarray, chord_m: float = WHEELBASE_M) -> tuple[float, float]:
    """(steepest descent, steepest climb) in degrees over a wheelbase-long chord of the path."""
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    z = np.asarray(tmap.height(pts[:, 0], pts[:, 1]))
    k = max(1, int(round(chord_m / float(np.median(seg)))))
    g = (z[k:] - z[:-k]) / (s[k:] - s[:-k])
    return float(np.degrees(math.atan(g.min()))), float(np.degrees(math.atan(g.max())))


def curvature(pts: np.ndarray) -> np.ndarray:
    a, b, c = pts[:-2], pts[1:-1], pts[2:]
    ab, bc, ac = b - a, c - b, c - a
    cross = np.abs(ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0])
    den = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ac, axis=1)
    return np.where(den > 1e-9, 2 * cross / np.maximum(den, 1e-9), 0.0)


def corridor(tmap: TerrainMap, pts: np.ndarray, half_width_m: float = 1.5) -> dict:
    """Terrain seen by a vehicle-wide corridor around the path (the route alone can miss a rim)."""
    tang = np.gradient(pts, axis=0)
    tn = tang / np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    nn = np.stack([-tn[:, 1], tn[:, 0]], axis=1)
    z, r = [], []
    for off in np.linspace(-half_width_m, half_width_m, 5):
        q = pts + off * nn
        z.append(np.asarray(tmap.height(q[:, 0], q[:, 1])))
        r.append(np.hypot(q[:, 0], q[:, 1]))
    z, r = np.concatenate(z), np.concatenate(r)
    return {"z_min_m": float(z.min()), "radius_min_m": float(r.min()),
            "max_abs_coord_m": float(np.abs(pts).max())}


def make_layout(cid: str, start: np.ndarray, house: np.ndarray, yaw: float, lp: LayoutParams,
                with_house: bool = False) -> EpisodeLayout:
    """The goal point is kept as ``house_xy`` (routes and camera framing need it) but NO house body is
    placed by default.

    Plan section B1 requires houses and obstacles removed from the crossing so that failure is dominated
    by the crater. It also matters mechanically: the open-loop schedule arms ignore the route and keep
    driving, so on the shallow and flat arms -- the very controls that must demonstrate a clean successful
    crossing -- every scheduled run drove into the house and recorded ~280 kN of asset contact. The deep
    arm was never affected (the vehicle is trapped long before the goal), so the trap claim did not depend
    on this, but the controls did.
    """
    hl, hw, hh = lp.house_size_m
    house_r = 0.5 * math.hypot(hl, hw) + 0.3
    assets = [Asset(kind="house", x_m=float(house[0]), y_m=float(house[1]), yaw_rad=yaw,
                    footprint_radius_m=house_r,
                    dims={"length_m": hl, "width_m": hw, "wall_height_m": hh})] if with_house else []
    return EpisodeLayout(episode_id=cid, seed=0, assets=assets,
                         house_xy=(float(house[0]), float(house[1])), house_yaw=yaw,
                         start_xy=(float(start[0]), float(start[1])), start_yaw=yaw)


# ----------------------------------------------------------------------------- control families
def schedule_spec(throttle: float, full_at_s: float) -> dict:
    """Constant approach throttle, then sustained full throttle; no braking, no steering, no feedback."""
    thr = [[0.0, float(throttle)]] if throttle < 1.0 else [[0.0, 1.0]]
    if throttle < 1.0:
        thr.append([float(full_at_s), 1.0])
    return {"steering": [[0.0, 0.0]], "throttle": thr, "braking": []}


def cal_schedule(throttles: list[float], dwell_s: float) -> dict:
    return {"steering": [[0.0, 0.0]], "braking": [],
            "throttle": [[round(i * dwell_s, 3), float(p)] for i, p in enumerate(throttles)]}


# ----------------------------------------------------------------------------- task construction
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arena-root", default="assets/traverse")
    ap.add_argument("--arena-prefix", default="arena_bowl")
    ap.add_argument("--out-b1", default="artifacts/traverse/wp9_bowl_b1")
    ap.add_argument("--out-b2", default="artifacts/traverse/wp9_bowl_b2")
    ap.add_argument("--emit", choices=["b1", "b2", "both"], default="both")
    ap.add_argument("--tracker", default="artifacts/traverse/wp3_tracker_v1")
    ap.add_argument("--headings", type=float, nargs="+", default=list(HEADINGS_DEG),
                    help="direction of travel in degrees; 0 drives straight down the entry-ramp axis")
    ap.add_argument("--speeds", type=float, nargs="+", default=list(CLAIM_SPEEDS))
    ap.add_argument("--excluded-speed", type=float, default=EXCLUDED_SPEED)
    ap.add_argument("--start-radius-m", type=float, default=34.0)
    ap.add_argument("--goal-radius-m", type=float, default=32.0)
    ap.add_argument("--a-accel", type=float, default=2.5,
                    help="route profile acceleration; 2.5 (vs the repo default 1.5, still far under the "
                         "measured 6.04 m/s^2) so even the 8 m/s level reaches its commanded speed before the rim")
    ap.add_argument("--horizon-s", type=float, default=60.0)
    ap.add_argument("--park-s", type=float, default=1.5)
    ap.add_argument("--roll-pitch-abort-deg", type=float, default=85.0)
    ap.add_argument("--detour-offset-m", type=float, default=16.0)
    ap.add_argument("--detour-speed", type=float, default=4.0)
    ap.add_argument("--schedule-throttles", type=float, nargs="+", default=[0.20, 0.30, 0.50, 1.00],
                    help="the schedule family's four approach-throttle levels (level 4 = sustained full "
                         "throttle from t=0); set from the calibration run's measured approach speeds")
    ap.add_argument("--schedule-full-at-s", type=float, default=20.0,
                    help="every schedule level goes to full throttle at this time and holds it to the horizon")
    ap.add_argument("--schedule-throttles-source", default="provisional",
                    help="provenance string recorded in design.json for --schedule-throttles")
    ap.add_argument("--cal-throttles", type=float, nargs="+", default=[0.15, 0.22, 0.32, 0.45, 0.65])
    ap.add_argument("--cal-dwell-s", type=float, default=3.0)
    ap.add_argument("--cal-horizon-s", type=float, default=15.0)
    ap.add_argument("--cal-start-radius-m", type=float, default=36.0)
    ap.add_argument("--cal-goal-radius-m", type=float, default=36.0)
    ap.add_argument("--perturb-lateral-m", type=float, default=0.6)
    ap.add_argument("--perturb-yaw-deg", type=float, default=2.0)
    ap.add_argument("--camera-horizon-s", type=float, default=0.25)
    args = ap.parse_args()

    out1, out2 = Path(args.out_b1), Path(args.out_b2)
    arena_root = Path(args.arena_root)
    arenas = {a: arena_root / f"{args.arena_prefix}_{a}" for a in ARMS}
    tmaps = {a: TerrainMap.from_dir(p) for a, p in arenas.items()}
    metas = {a: json.loads((p / "arena_meta.json").read_text()) for a, p in arenas.items()}
    bowl = metas["deep"]["bowl"]
    cx, cy = bowl["centre_sim_x_m"], bowl["centre_sim_y_m"]
    if (cx, cy) != (0.0, 0.0):
        raise SystemExit(f"this generator assumes the bowl is centred at the origin, got ({cx}, {cy})")
    R_bottom, R_top, R_top_entry = (float(bowl[k]) for k in ("R_bottom_m", "R_top_m", "R_top_entry_m"))
    params = replace(PlannerParams(), a_accel=float(args.a_accel))
    lp = LayoutParams()
    tracker = str(Path(args.tracker))
    ref_meta = str(Path(args.tracker) / "policy_meta.json")
    levels = [*args.speeds, args.excluded_speed]
    if len(args.schedule_throttles) != len(levels):
        raise SystemExit(f"--schedule-throttles needs {len(levels)} values (one per speed level)")

    # ---- shared run settings: every abort a deliberately trapped vehicle could trip is off
    base = {"arena": None, "horizon_s": float(args.horizon_s), "park_s": float(args.park_s),
            "localisation": "true", "ref_meta": ref_meta,
            "stall_abort_s": None,            # plan B2: keep the post-trapping attempts
            "off_route_m": None,              # a trapped vehicle is off its route by definition
            "roll_pitch_abort_deg": float(args.roll_pitch_abort_deg)}

    layouts: dict[str, dict] = {}
    geometry: list[dict] = []

    def add_layout(lid: str, arm: str, hdeg: float, start: np.ndarray, house: np.ndarray, yaw_deg: float,
                   kind: str, info: dict) -> str:
        d = out1 / lid
        d.mkdir(parents=True, exist_ok=True)
        layout = make_layout(lid, start, house, math.radians(yaw_deg), lp)
        (d / "meta.json").write_text(json.dumps(
            {"layout": layout.to_json(), "kind": kind, "arena": str(arenas[arm]), "arm": arm,
             "heading_deg": hdeg, "bowl": {"R_bottom_m": R_bottom, "R_top_m": R_top,
                                           "R_top_entry_m": R_top_entry, "centre_xy_m": [cx, cy]},
             **info}, indent=1))
        layouts[lid] = {"id": lid, "arm": arm, "heading_deg": hdeg, "kind": kind,
                        "arena": str(arenas[arm]), "meta_path": str(d / "meta.json"),
                        "start_xy": [float(start[0]), float(start[1])], "start_yaw_deg": yaw_deg,
                        "house_xy": [float(house[0]), float(house[1])], **info}
        return str(d / "meta.json")

    # ---- layouts: one per (arm, heading); the start/goal are identical across arms by construction,
    #      so the approach is matched until the rim (the three arenas are bit-identical outside 16.693 m)
    routes: dict[tuple[float, str], dict] = {}
    for hdeg in args.headings:
        u, n = unit(hdeg)
        start = -args.start_radius_m * u
        house = args.goal_radius_m * u
        r_end = args.goal_radius_m - params.approach_ring_m
        line = direct_line(u, args.start_radius_m, r_end, params.sample_step_m)
        det = {side_tag: detour_line(u, n, args.start_radius_m, r_end, args.detour_offset_m, sgn,
                                     params.sample_step_m)
               for side_tag, sgn in (("L", 1.0), ("R", -1.0))}
        for v in levels:
            routes[(hdeg, "direct", v)] = route(line, const_profile(line, v, params))
        for side_tag, pts in det.items():
            routes[(hdeg, f"detour_{side_tag}")] = route(pts, const_profile(pts, args.detour_speed, params))
        for arm in ARMS:
            t = tmaps[arm]
            down, up = chord_grade_deg(t, line)
            geometry.append({"arm": arm, "heading_deg": hdeg, "route": "direct",
                             "entry_chord_grade_deg": down, "exit_chord_grade_deg": up,
                             "start_slope_deg": float(np.degrees(np.arctan(t.slope(*start)))),
                             "goal_slope_deg": float(np.degrees(np.arctan(t.slope(*house)))),
                             **corridor(t, line)})
            for side_tag, pts in det.items():
                d_down, d_up = chord_grade_deg(t, pts)
                geometry.append({"arm": arm, "heading_deg": hdeg, "route": f"detour_{side_tag}",
                                 "entry_chord_grade_deg": d_down, "exit_chord_grade_deg": d_up,
                                 "kappa_max_1_per_m": float(curvature(pts).max()),
                                 "length_m": float(np.hypot(*np.diff(pts, axis=0).T).sum()),
                                 **corridor(t, pts)})
            add_layout(f"bowl_{arm}_h{int(hdeg):03d}", arm, hdeg, start, house, hdeg, "crossing",
                       {"start_radius_m": args.start_radius_m, "goal_radius_m": args.goal_radius_m,
                        "route_end_radius_m": r_end,
                        "approach_run_to_rim_m": args.start_radius_m - R_top_entry})

    # ---- perturbed initial conditions for the boundary cases (deep arm, primary heading)
    h0 = args.headings[0]
    u0, n0 = unit(h0)
    perturbs = [("p1", +args.perturb_lateral_m, 0.0), ("p2", -args.perturb_lateral_m, 0.0),
                ("p3", 0.0, args.perturb_yaw_deg)]
    for tag, lat, dyaw in perturbs:
        s = -args.start_radius_m * u0 + lat * n0
        add_layout(f"bowl_deep_h{int(h0):03d}_{tag}", "deep", h0, s, args.goal_radius_m * u0, h0 + dyaw,
                   "crossing_perturbed",
                   {"perturb_lateral_m": lat, "perturb_yaw_deg": dyaw, "of_layout": f"bowl_deep_h{int(h0):03d}",
                    "start_radius_m": args.start_radius_m, "goal_radius_m": args.goal_radius_m,
                    "route_end_radius_m": args.goal_radius_m - params.approach_ring_m})

    # ---- the throttle-calibration layout (flat arena, straight line, no bowl anywhere)
    cal_line = direct_line(u0, args.cal_start_radius_m, args.cal_goal_radius_m - params.approach_ring_m,
                           params.sample_step_m)
    cal_route = route(cal_line, const_profile(cal_line, 9.0, params))
    add_layout("bowl_cal_throttle", "flat", h0, -args.cal_start_radius_m * u0,
               args.cal_goal_radius_m * u0, h0, "throttle_calibration",
               {"throttles": args.cal_throttles, "dwell_s": args.cal_dwell_s,
                "run_length_m": float(cal_route["stations"][-1])})

    # ---- task builders -------------------------------------------------------
    def rec(out: Path, key: str, cand: str) -> str:
        return str(out / "records" / f"{key}__{cand}.npz")

    def tracker_task(out: Path, arm: str, lid: str, hdeg: float, v: float, cand: str, route_key) -> dict:
        return {**base, "key": lid, "meta_path": layouts[lid]["meta_path"], "arena": str(arenas[arm]),
                "candidate": cand, "controller": tracker, "route": routes[route_key],
                "record": rec(out, lid, cand),
                "arm": arm, "heading_deg": hdeg, "family": "tracker", "commanded_speed_mps": v}

    def schedule_task(out: Path, arm: str, lid: str, hdeg: float, li: int, cand: str, route_key) -> dict:
        p = float(args.schedule_throttles[li])
        return {**base, "key": lid, "meta_path": layouts[lid]["meta_path"], "arena": str(arenas[arm]),
                "candidate": cand, "controller": "schedule", "route": routes[route_key],
                "schedule": schedule_spec(p, args.schedule_full_at_s), "record": rec(out, lid, cand),
                "arm": arm, "heading_deg": hdeg, "family": "schedule", "approach_throttle": p}

    def level_tags(i: int, v: float) -> tuple[str, str]:
        return f"trk_v{v:.0f}", f"sch_L{i + 1}_t{int(round(100 * args.schedule_throttles[i])):03d}"

    def matrix_entry(out: Path, arm: str, hdeg: float, i: int) -> list[dict]:
        lid = f"bowl_{arm}_h{int(hdeg):03d}"
        v = levels[i]
        tt, st = level_tags(i, v)
        return [tracker_task(out, arm, lid, hdeg, v, tt, (hdeg, "direct", v)),
                schedule_task(out, arm, lid, hdeg, i, st, (hdeg, "direct", v))]

    def detour_task(out: Path, arm: str, hdeg: float, side: str) -> dict:
        lid = f"bowl_{arm}_h{int(hdeg):03d}"
        return tracker_task(out, arm, lid, hdeg, args.detour_speed, f"detour_{side}", (hdeg, f"detour_{side}"))

    def camera_task(out: Path, arm: str, hdeg: float) -> dict:
        lid = f"bowl_{arm}_h{int(hdeg):03d}"
        t = {**base, "key": lid, "meta_path": layouts[lid]["meta_path"], "arena": str(arenas[arm]),
             "candidate": "cam", "controller": "schedule", "route": routes[(hdeg, "direct", levels[0])],
             "schedule": {"steering": [], "throttle": [], "braking": [[0.0, 1.0]]},
             "horizon_s": float(args.camera_horizon_s), "park_s": 0.0,
             "dump_frame0": str(out / "frame0" / f"{lid}.npz"), "record": rec(out, lid, "cam"),
             "arm": arm, "heading_deg": hdeg, "family": "camera"}
        return t

    # ---- B1 ------------------------------------------------------------------
    b1: list[dict] = []
    for arm in ARMS:
        for hdeg in args.headings:
            for i in range(len(args.speeds)):
                b1 += matrix_entry(out1, arm, hdeg, i)
    iE = len(levels) - 1
    for hdeg in args.headings:                       # excluded level, deep arm, every heading
        b1 += matrix_entry(out1, "deep", hdeg, iE)
    for arm in ("shallow", "flat"):                  # excluded level on the controls, primary heading
        b1 += matrix_entry(out1, arm, h0, iE)
    b1 += [detour_task(out1, "deep", h0, "L"), detour_task(out1, "deep", h0, "R")]
    for tag, _, _ in perturbs:                       # boundary cases with perturbed initial conditions
        lid = f"bowl_deep_h{int(h0):03d}_{tag}"
        i6 = args.speeds.index(6.0) if 6.0 in args.speeds else len(args.speeds) - 1
        v6 = args.speeds[i6]
        b1.append(tracker_task(out1, "deep", lid, h0, v6, f"trk_v{v6:.0f}_{tag}", (h0, "direct", v6)))
        b1.append(schedule_task(out1, "deep", lid, h0, iE, f"sch_L{iE + 1}_{tag}", (h0, "direct", levels[iE])))
    b1.append({**base, "key": "bowl_cal_throttle", "meta_path": layouts["bowl_cal_throttle"]["meta_path"],
               "arena": str(arenas["flat"]), "candidate": "cal_staircase", "controller": "schedule",
               "route": cal_route, "schedule": cal_schedule(args.cal_throttles, args.cal_dwell_s),
               "horizon_s": float(args.cal_horizon_s), "park_s": 0.0,
               "record": rec(out1, "bowl_cal_throttle", "cal_staircase"),
               "arm": "flat", "heading_deg": h0, "family": "calibration"})
    b1 += [camera_task(out1, arm, h0) for arm in ARMS]

    # ---- B2 ------------------------------------------------------------------
    b1_records = {t["record"] for t in b1}
    b2: list[dict] = []
    for arm in ARMS:
        for hdeg in args.headings:
            for i in range(len(levels)):
                b2 += matrix_entry(out2, arm, hdeg, i)
    b2 += [detour_task(out2, "deep", h0, "L"), detour_task(out2, "deep", h0, "R"),
           detour_task(out2, "shallow", h0, "L"),
           detour_task(out2, "deep", args.headings[1], "L")]
    for arm in ARMS:
        for hdeg in args.headings:
            b2.append(camera_task(out2, arm, hdeg))
    # reuse: any B2 run whose B1 twin exists is redirected at the B1 record (and its frame-0 dump)
    n_reused = 0
    for t in b2:
        b1_rec = t["record"].replace(str(out2), str(out1), 1)
        if b1_rec in b1_records:
            t["record"] = b1_rec
            if "dump_frame0" in t:
                t["dump_frame0"] = t["dump_frame0"].replace(str(out2), str(out1), 1)
            t["reuse"] = "b1"
            n_reused += 1
        else:
            t["reuse"] = "new"

    # ---- write ---------------------------------------------------------------
    design = {
        "plan": "docs/vision/hmmwv_traverse/energy_and_crater_experiment_plan.md section B",
        "arenas": {a: str(p) for a, p in arenas.items()},
        "bowl": {"centre_xy_m": [cx, cy], "R_bottom_m": R_bottom, "R_top_m": R_top,
                 "R_top_entry_m": R_top_entry, "wall_deg": bowl["wall_deg"], "depth_m": bowl["depth_m"],
                 "entry_deg": bowl["entry_deg"], "entry_azimuth_deg": bowl["entry_azimuth_deg"],
                 "entry_halfwidth_deg": bowl["entry_halfwidth_deg"],
                 "arm_depth_m": {a: metas[a]["bowl"]["arm_depth_m"] for a in ARMS}},
        "headings_deg": list(args.headings),
        "speed_levels_mps": levels, "claim_levels_mps": list(args.speeds),
        "excluded_level_mps": args.excluded_speed,
        "excluded_because": "8 m/s pitches the vehicle over on the deep arena (prototype: 89.1 deg pitch, "
                            "chassis contact 99 kN); run and recorded, excluded from the trap claim",
        "control_families": {
            "tracker": {"controller": tracker, "route": "straight line through the bowl centre",
                        "profile": "const_profile", "a_accel": args.a_accel,
                        "a_lat_max": params.a_lat_max, "v_launch_mps": params.v_launch_mps,
                        "terminal_taper_m": params.terminal_taper_m},
            "schedule": {"controller": "schedule", "approach_throttles": list(args.schedule_throttles),
                         "approach_throttles_source": args.schedule_throttles_source,
                         "full_throttle_from_s": args.schedule_full_at_s,
                         "steering": 0.0, "braking": 0.0,
                         "note": "open loop; the level is an approach THROTTLE, the achieved entry speed is "
                                 "measured per run, not commanded. The nominal route it carries is the "
                                 "tracker route of the SAME LEVEL INDEX -- nothing follows it, it only "
                                 "drives cross-track/station logging and the route-end completion test"}},
        "run_settings": {"horizon_s": args.horizon_s, "park_s": args.park_s,
                         "stall_abort_s": None, "off_route_m": None,
                         "roll_pitch_abort_deg": args.roll_pitch_abort_deg, "localisation": "true",
                         "record_preset": "tire_normal_force_omega_pt (default)",
                         "post_entry_observation_s": "horizon minus the measured entry time; the entry is "
                                                     "reached in 9-15 s from a 17.56 m approach run"},
        "start_radius_m": args.start_radius_m, "goal_radius_m": args.goal_radius_m,
        "approach_run_to_rim_m": args.start_radius_m - R_top_entry,
        "detour": {"offset_m": args.detour_offset_m, "speed_mps": args.detour_speed,
                   "shape": "raised cosine, tangent to the approach heading at both ends"},
        "perturbations": [{"tag": t, "lateral_m": l, "yaw_deg": y} for t, l, y in perturbs],
        "calibration": {"throttles": args.cal_throttles, "dwell_s": args.cal_dwell_s,
                        "horizon_s": args.cal_horizon_s, "arm": "flat"},
        "geometry_measured_on_bmp": geometry,
        "counts": {"b1": len(b1), "b2": len(b2), "b2_reused_from_b1": n_reused,
                   "b2_new_runs": len(b2) - n_reused},
    }
    if args.emit in ("b1", "both"):
        out1.mkdir(parents=True, exist_ok=True)
        (out1 / "tasks.json").write_text(json.dumps(b1, indent=1))
        (out1 / "layouts.json").write_text(json.dumps(list(layouts.values()), indent=1))
        (out1 / "design.json").write_text(json.dumps(design, indent=1))
    if args.emit in ("b2", "both"):
        out2.mkdir(parents=True, exist_ok=True)
        (out2 / "tasks.json").write_text(json.dumps(b2, indent=1))
        (out2 / "design.json").write_text(json.dumps(design, indent=1))
        (out2 / "episodes.json").write_text(json.dumps(
            [{"key": t["key"], "candidate": t["candidate"], "arm": t["arm"],
              "heading_deg": t["heading_deg"], "family": t["family"],
              "level_mps": t.get("commanded_speed_mps"), "approach_throttle": t.get("approach_throttle"),
              "controller": t["controller"], "record": t["record"], "meta_path": t["meta_path"],
              "arena": t["arena"], "reuse": t["reuse"]} for t in b2], indent=1))

    # ---- report --------------------------------------------------------------
    def tally(ts):
        out = {}
        for t in ts:
            out[t["family"]] = out.get(t["family"], 0) + 1
        return out
    print(f"B1 {len(b1)} runs {tally(b1)} -> {out1}")
    print(f"B2 {len(b2)} runs {tally(b2)} | reused from B1 {n_reused}, new {len(b2) - n_reused} -> {out2}")
    print(f"bowl: centre ({cx}, {cy}), R_bottom {R_bottom} m, steep rim {R_top:.3f} m, "
          f"entry-ramp rim {R_top_entry:.3f} m; approach run to the rim {args.start_radius_m - R_top_entry:.2f} m")
    print("geometry measured on the written BMPs (wheelbase chord):")
    seen = set()
    for g in geometry:
        k = (g["arm"], g["route"])
        if g["route"] == "direct" or k not in seen:
            seen.add(k)
            print(f"  {g['arm']:8s} h{int(g['heading_deg']):03d} {g['route']:9s} entry {g['entry_chord_grade_deg']:7.2f} "
                  f"exit {g['exit_chord_grade_deg']:6.2f} deg | corridor z_min {g['z_min_m']:7.3f} m "
                  f"r_min {g['radius_min_m']:6.2f} m max|xy| {g['max_abs_coord_m']:5.2f} m"
                  + (f" kappa_max {g['kappa_max_1_per_m']:.4f}" if "kappa_max_1_per_m" in g else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
