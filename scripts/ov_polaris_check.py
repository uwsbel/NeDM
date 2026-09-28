#!/usr/bin/env python3
"""Checks for module M1 (Polaris vehicle + dispatcher) of the offroad_vehicles_20260927 study.

Subcommands
  make-data      write the private vehicle-data folder assets/traverse/vehicles/ov_polaris/ from the stock Polaris
                 files of a Chrono data folder: re-framed top-level vehicle + chassis JSON (reference at mid-wheelbase,
                 every chassis-frame location + (1.35763, 0, -0.42) m), the variant top-level files (stock driveline,
                 power-corrected driveline + gearbox, shafts 4WD), the belly points (lower envelope of the visual
                 chassis mesh on a 0.10 m grid, in the re-framed chassis frame) and MANIFEST.json (sha256 of every file)
  rigid          build every Polaris arm and settle it 0.8 s braked on a flat rigid patch (SMC, friction 0.9, 2 ms step,
                 TMEASY tyres): mass, spindle positions, launch height, vertical speed at 0.8 s
  flat-soil      one vehicle on a flat 80 x 80 m soil patch with the production soil (crm_main.json, the vehicle's own
                 soil wheels through the same swaps as the dispatcher): --mode straight = 0.8 s braked settle, 5 s at
                 2, 4, 6 m/s, then 4 s full brake; --mode turn = settle, 3 m/s through a 10 m radius half circle
                 (curvature 0.10 /m); --mode settle = settle only (+ 1.2 s at rest).  Frozen follower (make_driver).
  episode-check  crm_qa.check + launch height + finite 17-column state + vehicle block on dispatcher run folders
  compare        array-by-array comparison of two run folders (same-GPU identity check)
Run soil subcommands locally only inside ``flock /tmp/luffy_crm.lock``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import ov_vehicle as ovv  # noqa: E402

DEFAULT_CFG = ROOT / "artifacts/traverse/crm_f104_v1/configs/crm_main.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jdump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


# ============================================================================================ make-data
def _shift(v):
    s = ovv.FRAME_SHIFT_M
    return [round(float(v[0]) + s["x"], 9), round(float(v[1]) + s["y"], 9), round(float(v[2]) + s["z"], 9)]


def cmd_make_data(a):
    stock = Path(a.chrono_data) / "vehicle" / "Polaris"
    out = ovv.ASSET_DIR
    sub = out / ovv.PRIVATE_SUBDIR
    if out.exists() and any(out.iterdir()) and not a.force:
        raise SystemExit(f"{out} exists and is not empty (use --force to rewrite it before it is frozen)")
    sub.mkdir(parents=True, exist_ok=True)
    top = json.loads((stock / "Polaris.json").read_text())
    chassis = json.loads((stock / "Polaris_Chassis.json").read_text())
    # --- only the keys we know how to re-frame may be present
    assert set(top) == {"Name", "Type", "Template", "Chassis", "Axles", "Steering Subsystems", "Driveline"}, set(top)
    axle_keys = {"Suspension Input File", "Suspension Location", "Steering Index", "Left Wheel Input File",
                 "Right Wheel Input File", "Left Brake Input File", "Right Brake Input File", "Antirollbar Input File",
                 "Antirollbar Location"}
    for ax in top["Axles"]:
        assert set(ax) <= axle_keys, set(ax) - axle_keys
    for st in top["Steering Subsystems"]:
        assert set(st) == {"Input File", "Location", "Orientation"} and st["Orientation"] == [1, 0, 0, 0], st
    assert set(chassis) == {"Name", "Type", "Template", "Components", "Driver Position", "Visualization"}, set(chassis)
    # --- chassis: centroidal frames and the driver position shifted, the visual mesh dropped (it would sit 0.42 m low)
    c = copy.deepcopy(chassis)
    c["Name"] = ("Polaris chassis, reference at mid-wheelbase 0.023 m above the front axle line (ov_polaris; stock "
                 "Polaris_Chassis.json with every location + (1.35763, 0, -0.42) m; visual mesh dropped)")
    for comp in c["Components"]:
        assert comp["Centroidal Frame"]["Orientation"] == [1, 0, 0, 0]
        comp["Centroidal Frame"]["Location"] = _shift(comp["Centroidal Frame"]["Location"])
    c["Driver Position"]["Location"] = _shift(c["Driver Position"]["Location"])
    c.pop("Visualization")
    jdump(sub / "Polaris_ovc_Chassis.json", c)

    def top_variant(name, driveline):
        v = copy.deepcopy(top)
        v["Name"] = name
        v["Chassis"]["Input File"] = f"{ovv.PRIVATE_SUBDIR}/Polaris_ovc_Chassis.json"
        for ax in v["Axles"]:
            ax["Suspension Location"] = _shift(ax["Suspension Location"])
            if "Antirollbar Location" in ax:
                ax["Antirollbar Location"] = _shift(ax["Antirollbar Location"])
        for st in v["Steering Subsystems"]:
            st["Location"] = _shift(st["Location"])
        v["Driveline"]["Input File"] = driveline
        return v

    jdump(sub / "Polaris_ovc_stock.json", top_variant(
        "Polaris, chassis reference at mid-wheelbase (ov_polaris), stock SimpleDriveline as shipped",
        "Polaris/Polaris_DrivelineSimple.json"))
    jdump(sub / "Polaris_ovc_pc.json", top_variant(
        "Polaris, chassis reference at mid-wheelbase (ov_polaris), power-corrected SimpleDriveline (conical ratios 1.0)",
        f"{ovv.PRIVATE_SUBDIR}/Polaris_ovc_DrivelineSimple_pc.json"))
    jdump(sub / "Polaris_ovc_4wd.json", top_variant(
        "Polaris, chassis reference at mid-wheelbase (ov_polaris), Chrono shafts 4WD driveline Polaris_4WD.json",
        "Polaris/Polaris_4WD.json"))
    d = json.loads((stock / "Polaris_DrivelineSimple.json").read_text())
    assert d["Front Conical Gear Ratio"] == 0.25 and d["Rear Conical Gear Ratio"] == 0.25, d
    d["Name"] = "Polaris simple driveline, power-corrected (ov_polaris): conical ratios 1.0, the 0.25 reduction moved into the gearbox"
    d["Front Conical Gear Ratio"], d["Rear Conical Gear Ratio"] = 1.0, 1.0
    jdump(sub / "Polaris_ovc_DrivelineSimple_pc.json", d)
    t = json.loads((stock / "Polaris_AutomaticTransmissionSimpleMap.json").read_text())
    t["Name"] = "Polaris simple-map transmission, power-corrected (ov_polaris): every gear ratio x 0.25"
    gb = t["Gear Box"]
    gb["Reverse Gear Ratio"] = round(gb["Reverse Gear Ratio"] * 0.25, 9)
    gb["Forward Gear Ratios"] = [round(r * 0.25, 9) for r in gb["Forward Gear Ratios"]]
    jdump(sub / "Polaris_ovc_AutomaticTransmissionSimpleMap_pc.json", t)

    # --- belly points: lowest visual-mesh vertex in every occupied 0.10 m cell of the chassis x-y plane
    obj = stock / "meshes" / "Polaris_chassis.obj"
    verts = []
    with obj.open() as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
    v = np.asarray(verts, np.float64)
    g = float(a.grid_m)
    cell = np.floor(v[:, :2] / g).astype(np.int64)
    key = cell[:, 0] * 100000 + cell[:, 1]
    order = np.lexsort((v[:, 2], key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    low = v[order][first]
    s = ovv.FRAME_SHIFT_M
    pts = np.stack([low[:, 0] + s["x"], low[:, 1] + s["y"], low[:, 2] + s["z"]], 1)
    belly = {"schema": "ov_polaris_belly_points_v1", "source_obj": "Polaris/meshes/Polaris_chassis.obj",
             "source_obj_sha256": sha(obj), "source_kind": "VISUAL mesh (the JSON Polaris has no chassis collision shape)",
             "frame": ("re-framed Polaris chassis reference frame (x forward, y left, z up; stock frame + "
                       f"({s['x']}, {s['y']}, {s['z']}) m)"),
             "grid_m": g, "n_vertices": int(len(v)), "n_points": int(len(pts)),
             "lowest_point_z_m": float(pts[:, 2].min()),
             "bbox_stock_frame_min": v.min(0).round(4).tolist(), "bbox_stock_frame_max": v.max(0).round(4).tolist(),
             "method": ("for every occupied cell of a grid_m grid in the chassis x-y plane, the lowest mesh vertex in "
                        "that cell (a triangle's lowest point is one of its vertices, so the overall minimum is exact); "
                        "points are real vertices shifted into the re-framed chassis frame"),
             "points": np.round(pts, 6).tolist()}
    jdump(out / ovv.BELLY_FILE_NAME, belly)
    (out / "README.md").write_text(
        "# Private Polaris vehicle data (offroad_vehicles_20260927, module M1)\n\n"
        "Written by `scripts/ov_polaris_check.py make-data` from Chrono's stock `data/vehicle/Polaris/` files.\n"
        "At run time `scripts/ov_vehicle.py` builds a temporary vehicle-data folder with two links, `Polaris_ov/` (this\n"
        "folder's `Polaris_ov/`) and `Polaris/` (the Chrono build's stock folder), points Chrono's vehicle data path at it\n"
        "while the vehicle is built, and restores the path afterwards. `MANIFEST.json` holds the sha256 of every file;\n"
        "the vehicle switch refuses to build if any file differs.\n\n"
        "- `Polaris_ov/Polaris_ovc_Chassis.json`: the stock chassis with the reference point moved from the front axle\n"
        "  (0.397 m below the axle line, i.e. below the ground at rest) to mid-wheelbase, 0.023 m above the front axle\n"
        "  line: every location + (1.35763, 0, -0.42) m. Same mass and inertia. The visual mesh line is dropped (it would\n"
        "  be drawn 0.42 m too low; rendering only).\n"
        "- `Polaris_ov/Polaris_ovc_stock.json`: the stock vehicle file with the same shift of every location; stock\n"
        "  driveline (`Polaris/Polaris_DrivelineSimple.json`, which has Chrono's reduction defect). Primary arm `polaris`\n"
        "  (and `polaris_w08`, which differs only in the soil wheels).\n"
        "- `Polaris_ov/Polaris_ovc_pc.json` + `Polaris_ovc_DrivelineSimple_pc.json` +\n"
        "  `Polaris_ovc_AutomaticTransmissionSimpleMap_pc.json`: power-corrected arm `polaris_pc` (conical ratios 1.0,\n"
        "  every gear ratio x 0.25).\n"
        "- `Polaris_ov/Polaris_ovc_4wd.json`: arm `polaris_4wd`, Chrono's shafts driveline `Polaris/Polaris_4WD.json`.\n"
        f"- `{ovv.BELLY_FILE_NAME}`: points on the underside of the visual chassis mesh (lowest vertex per 0.10 m cell),\n"
        "  in the re-framed chassis frame, for the belly-in-soil diagnostic.\n")
    files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}
    stock_used = {f"Polaris/{n}": sha(stock / n) for n in ("Polaris.json", "Polaris_Chassis.json", "Polaris_DrivelineSimple.json",
                                                             "Polaris_AutomaticTransmissionSimpleMap.json")}
    stock_used["Polaris/meshes/Polaris_chassis.obj"] = sha(obj)
    jdump(out / "MANIFEST.json", {"schema": "ov_polaris_manifest_v1", "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                                  "generator": "scripts/ov_polaris_check.py make-data", "generator_sha256": sha(__file__),
                                  "stock_source_folder": str(stock), "stock_source_sha256": stock_used,
                                  "frame_shift_m": ovv.FRAME_SHIFT_M, "files_sha256": files})
    print(json.dumps({"out": str(out), "files": files, "belly_points": len(pts), "belly_lowest_z_m": belly["lowest_point_z_m"]}, indent=1))
    if a.compare_s1:
        s1 = Path(a.compare_s1)
        for mine, theirs in (("Polaris_ovc_stock.json", "Polaris_ovc_stock.json"), ("Polaris_ovc_Chassis.json", "Polaris_ovc_Chassis.json"),
                             ("Polaris_ovc_4wd.json", "Polaris_ovc_shafts4WD.json")):
            m, t_ = json.loads((sub / mine).read_text()), json.loads((s1 / theirs).read_text())
            print(mine, "vs S1", theirs, "numerically equal:", _num_equal(m, t_, skip=("Name",)))


def _num_equal(x, y, skip=()):
    if isinstance(x, dict) and isinstance(y, dict):
        kx, ky = set(x) - set(skip), set(y) - set(skip)
        return kx == ky and all(_num_equal(x[k], y[k], skip) for k in kx)
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(_num_equal(p, q, skip) for p, q in zip(x, y))
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return abs(float(x) - float(y)) < 1e-9
    return x == y


# ============================================================================================ rigid
def cmd_rigid(a):
    import pychrono as ch
    import pychrono.vehicle as veh
    ch.SetChronoDataPath(a.chrono_data + "/")
    (getattr(veh, "SetVehicleDataPath", None) or veh.SetDataPath)(a.chrono_data + "/vehicle/")
    results = []
    for arm in a.arms:
        cfg = {"vehicle": ovv.polaris_vehicle_block(arm, {"x_m": 0.0, "y_m": 0.0, "z_m": ovv.POLARIS_SPAWN_DZ_M, "yaw_rad": 0.0}, "TMEASY"),
               "simulation": {"tire_step_size_s": 1e-3}}
        state = {}
        t0 = time.time()
        m = ovv.create_polaris(cfg, state)
        system = m.GetSystem()
        mat = ch.ChContactMaterialSMC()
        mat.SetFriction(0.9)
        mat.SetRestitution(0.01)
        mat.SetYoungModulus(2e7)
        terrain = veh.RigidTerrain(system)
        terrain.AddPatch(mat, ch.CSYSNORM, 200.0, 200.0)
        terrain.Initialize()
        v = m.GetVehicle()
        inp = veh.DriverInputs()
        inp.m_braking, inp.m_throttle, inp.m_steering = 1.0, 0.0, 0.0
        dt, t, rows = 2e-3, 0.0, []
        for k in range(int(round(0.8 / dt))):
            terrain.Synchronize(t)
            m.Synchronize(t, inp, terrain)
            terrain.Advance(dt)
            m.Advance(dt)
            t = (k + 1) * dt
            if (k + 1) % 25 == 0:
                ref = v.GetChassisBody().GetFrameRefToAbs()
                rows.append({"t": round(t, 3), "ref_height_m": float(ref.GetPos().z), "vz_mps": float(ref.GetPosDt().z)})
        end = rows[-1]
        r = {"arm": arm, "build": state["build"], "json_files": sorted(state["files_sha256"]),
             "launch_height_m_at_0.8s": round(end["ref_height_m"], 4), "vz_mps_at_0.8s": round(end["vz_mps"], 4),
             "max_abs_vz_0.6_0.8s": round(max(abs(r_["vz_mps"]) for r_ in rows if r_["t"] >= 0.6 - 1e-9), 4),
             "wall_s": round(time.time() - t0, 2)}
        r["pass"] = {"mass_1378.33": abs(state["build"]["mass_kg"] - 1378.33) < 0.01,
                     "launch_height_0.30_0.45": 0.30 <= end["ref_height_m"] <= 0.45,
                     "vz_below_0.05": abs(end["vz_mps"]) < 0.05}
        results.append(r)
        print(json.dumps(r), flush=True)
    if a.out:
        jdump(a.out, results)
    return results


# ============================================================================================ flat soil driveability
STRAIGHT_STEPS = ((0.0, 5.0, 2.0), (5.0, 10.0, 4.0), (10.0, 15.0, 6.0))
BRAKE_START_S, BRAKE_END_S = 15.0, 19.0
TURN_SPEED, TURN_RADIUS = 3.0, 10.0


def flat_arena(folder, size, px=64):
    from PIL import Image
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((px, px), 128, np.uint8), mode="L").save(folder / "flat.bmp")
    meta = {"bmp": "flat.bmp", "size_m": float(size), "pixels": px, "height_min_m": -0.5, "height_max_m": 0.5,
            "orientation": {"rot90": 0, "flipud": True}, "features": [], "note": "ov_polaris_check flat patch"}
    (folder / "arena_meta.json").write_text(json.dumps(meta, indent=1))
    return folder


def make_route(mode, half):
    if mode in ("straight", "settle"):
        x0 = -half + 4.0
        xs = np.arange(x0, half - 2.0, 0.5)
        wp = np.stack([xs, np.zeros_like(xs)], 1)
        seg = None
    else:
        lead = np.stack([np.arange(-25.0, -15.0, 0.5), np.full(20, -10.0)], 1)
        ang = np.linspace(-math.pi / 2, math.pi / 2, 64)
        arc = np.stack([-15.0 + TURN_RADIUS * np.cos(ang), TURN_RADIUS * np.sin(ang)], 1)
        back = np.stack([np.arange(-15.5, -30.0, -0.5), np.full(29, 10.0)], 1)
        wp = np.concatenate([lead, arc, back])
        seg = (len(lead), len(lead) + len(arc))
    st = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
    hd = np.arctan2(np.gradient(wp[:, 1]), np.gradient(wp[:, 0]))
    route = {"waypoints": wp.tolist(), "stations": st.tolist(), "speeds": [2.0 if mode != "turn" else TURN_SPEED] * len(wp),
             "headings": hd.tolist()}
    return route, seg


def cross_track(poly, p):
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    d = np.linalg.norm(a + t[:, None] * ab - p, axis=1)
    return float(d.min())


def build_vehicle(vehicle, scene, hd, crm_collect, agv, arena, start_xyz, yaw, dt, chrono_data, source):
    """Vehicle + wheel geometry through the same swaps the dispatcher installs."""
    from nedm.hmmwv_data import configure_chrono_data_paths
    state, record, belly_factory = {}, None, None
    if vehicle == "hmmwv":
        config = scene.build_config(arena, start_xyz, yaw, step_size_s=dt, tire_step_size_s=dt)
        create, build = hd.create_hmmwv, crm_collect.build_crm
        record = ovv.HmmwvRecord("soil", wrapper=str(HERE / "ov_crm_collect.py"))
        wheel = "production tyre mesh"
    elif vehicle == "gator":
        config = agv.make_build_config(scene.build_config)(arena, start_xyz, yaw, step_size_s=dt, tire_step_size_s=dt)
        create = agv.create_gator
        wheel = agv.GATOR_SOIL_WHEEL
        build = agv.make_build_crm(crm_collect.build_crm, wheel)
        opts = argparse.Namespace(vehicle="gator")
        record = agv.VehicleRecord("soil", opts, chrono_data=chrono_data, soil_wheel_geom=wheel, wrapper=str(HERE / "ov_crm_collect.py"))
        belly_factory = lambda tmap: agv.BellyClearance(tmap, chrono_data=chrono_data)  # noqa: E731
    elif vehicle in ovv.POLARIS_NAMES:
        config = ovv.make_build_config(scene.build_config, vehicle)(arena, start_xyz, yaw, step_size_s=dt, tire_step_size_s=dt)
        record = ovv.PolarisRecord(vehicle, "soil", chrono_data=chrono_data, wrapper=str(HERE / "ov_crm_collect.py"))
        state = record.state
        create = lambda c: ovv.create_polaris(c, state)  # noqa: E731
        wheel = record.soil_wheel
        build = agv.make_build_crm(crm_collect.build_crm, wheel)
        belly_factory = lambda tmap: ovv.make_belly(tmap, chrono_data=chrono_data)  # noqa: E731
    else:
        raise SystemExit(f"flat-soil does not handle {vehicle}")
    config["vehicle"]["tire_model"] = "RIGID_MESH"
    config["vehicle"]["chassis_collision"] = "NONE"
    config["chrono_data_root"] = str(Path(chrono_data).resolve())
    config["vehicle_data_root"] = str(Path(chrono_data).resolve() / "vehicle")
    configure_chrono_data_paths(source, config)
    return config, create, build, record, belly_factory, wheel


def analyze(mode, R, seg, xy, depth_limit, nonfinite, has_belly):
    """Pass rules of PLAN 2.1 / S3 2.1 on the recorded frames (pure numpy; unit-testable)."""
    first = R[0]
    launch_pass = (abs(first["speed"]) <= 1.0 and max(abs(first["roll"]), abs(first["pitch"])) <= math.radians(25)
                   and 0.0 <= first["h"] <= 1.2)
    out = {"anchor": {"speed_mps": float(abs(first["speed"])), "roll_deg": math.degrees(first["roll"]),
                      "pitch_deg": math.degrees(first["pitch"]), "ref_height_m": first["h"], "vz_mps": first["vz"]},
           "launch_check_pass": bool(launch_pass), "anchor_vz_below_0.05": abs(first["vz"]) < 0.05,
           "max_sinkage_m": float(max(max(r["sink"]) for r in R)),
           "max_abs_pitch_deg": float(max(abs(math.degrees(r["pitch"])) for r in R)),
           "max_abs_roll_deg": float(max(abs(math.degrees(r["roll"])) for r in R)),
           "min_belly_clearance_m": float(min(r["belly"] for r in R)) if has_belly else None,
           "gears_used": sorted({r["gear"] for r in R})}
    base = {"launch": bool(launch_pass), "vz": abs(first["vz"]) < 0.05, "finite": nonfinite == 0,
            "no_breakthrough": out["max_sinkage_m"] < depth_limit}
    if mode == "settle":
        out["creep_max_speed_mps"] = float(max(abs(r["speed"]) for r in R))
        out["pass"] = base
    elif mode == "straight":
        steps = []
        for s, e, v in STRAIGHT_STEPS:
            w = [r for r in R if s + 4.0 - 1e-9 <= r["t"] < e - 1e-9]
            if not w:
                steps.append({"target_mps": v, "window_s": [s + 4.0, e], "frames": 0, "within_10pct": False})
                continue
            sp = np.array([r["speed"] for r in w])
            steps.append({"target_mps": v, "window_s": [s + 4.0, e], "frames": len(w), "mean_mps": float(sp.mean()),
                          "min_mps": float(sp.min()), "max_mps": float(sp.max()),
                          "within_10pct": bool(np.all(np.abs(sp - v) <= 0.1 * v)),
                          "mean_abs_slip": float(np.mean([np.abs(r["slip"]).mean() for r in w])),
                          "mean_sinkage_m": float(np.mean([np.mean(r["sink"]) for r in w]))})
        br = [r for r in R if r["t"] >= BRAKE_START_S - 1e-9]
        stop = next((r for r in br if abs(r["speed"]) < 0.1), None)
        out["steps"] = steps
        out["lateral_drift_max_m"] = float(max(abs(r["y"]) for r in R if r["t"] < BRAKE_START_S))
        out["brake"] = {"frames": len(br), "speed_at_brake_mps": br[0]["speed"] if br else None,
                        "stopped": stop is not None, "stop_time_s": (stop["t"] - BRAKE_START_S) if stop else None,
                        "stop_distance_m": (stop["x"] - br[0]["x"]) if stop else None,
                        "min_speed_after_brake_mps": float(min(r["speed"] for r in br)) if br else None,
                        "max_abs_pitch_deg_braking": float(max(abs(math.degrees(r["pitch"])) for r in br)) if br else None}
        out["pass"] = {**base, "speeds_within_10pct": all(s_["within_10pct"] for s_ in steps),
                       "lateral_drift_below_0.5": out["lateral_drift_max_m"] < 0.5,
                       "brake_stops": stop is not None,
                       "no_back_creep_0.4": br != [] and out["brake"]["min_speed_after_brake_mps"] > -0.4,
                       "no_pitch_over_25": br != [] and out["brake"]["max_abs_pitch_deg_braking"] < 25}
    else:
        poly = xy[seg[0]:seg[1]]
        on_arc = [r for r in R if r["x"] > -15.0 + 1e-6 and r["t"] > 0]
        ct = np.array([cross_track(poly, np.array([r["x"], r["y"]])) for r in on_arc]) if on_arc else np.array([np.nan])
        sp = np.array([r["speed"] for r in on_arc]) if on_arc else np.array([np.nan])
        completed = bool(R[-1]["x"] < -24.0 and R[-1]["y"] > 5.0)
        out["turn"] = {"radius_m": TURN_RADIUS, "speed_target_mps": TURN_SPEED, "frames_on_arc": len(on_arc),
                       "cross_track_rms_m": float(np.sqrt(np.mean(ct ** 2))), "cross_track_max_m": float(ct.max()),
                       "speed_mean_on_arc_mps": float(sp.mean()), "completed": completed,
                       "max_abs_steering": float(max(abs(r["steering"]) for r in R))}
        out["pass"] = {**base, "completed": completed,
                       "cross_track_rms_below_0.5": bool(out["turn"]["cross_track_rms_m"] < 0.5)}
    return out


def cmd_flat_soil(a):
    import pychrono as chrono
    import pychrono.fsi as fsi
    import pychrono.vehicle as veh
    import ag_vehicle as agv
    import crm_collect
    import nedm.hmmwv_data as hd
    import nedm.traverse.scene as scene
    import traverse_fdm_rgbd_diverse_chrono as frozen
    from nedm.hmmwv_data import WHEEL_SPECS, capture_row
    from nedm.training.constants import STATE_FIELD_PRESETS
    from nedm.traverse.terrain import TerrainMap

    cfg = crm_collect.merged_config(a.cfg)
    dt = float(cfg["step_s"])
    DT, SETTLE_S = frozen.DT, frozen.SETTLE_S
    tmp = Path(tempfile.mkdtemp(prefix="ov_flat_"))
    import atexit
    import shutil
    atexit.register(shutil.rmtree, str(tmp), True)
    arena = flat_arena(tmp / "arena_flat", a.size)
    tmap = TerrainMap.from_dir(arena)
    route, seg = make_route(a.mode, a.size / 2)
    xy = np.asarray(route["waypoints"], float)
    x0, y0 = xy[0]
    ground = float(tmap.height(x0, y0))
    config, create, build, record, belly_factory, wheel = build_vehicle(
        a.vehicle, scene, hd, crm_collect, agv, arena, (x0, y0, ground + .75), float(route["headings"][0]), dt,
        a.chrono_data, ROOT)
    t0 = time.time()
    model = create(config)
    vehicle, system = model.GetVehicle(), model.GetSystem()
    terrain = build(chrono, veh, fsi, system, vehicle, arena, tmap.meta, cfg)
    build_s = time.time() - t0
    try:
        nbce = [int(terrain.GetNumBCE(vehicle.GetWheel(ax, sd).GetSpindle())) for _, ax, sd in WHEEL_SPECS]
    except Exception as exc:  # noqa: BLE001
        nbce = f"n/a ({type(exc).__name__})"
    belly = belly_factory(tmap) if belly_factory else None
    if belly is not None:
        record.belly = belly
    driver = frozen.make_driver(chrono, veh, vehicle, route, tmap)
    tire_radii = {name: float(vehicle.GetTire(axle, side).GetRadius()) for name, axle, side in WHEEL_SPECS}
    fields = STATE_FIELD_PRESETS["tire_normal_force_omega_pt"]
    engine, transmission = vehicle.GetEngine(), vehicle.GetTransmission()
    if a.mode == "straight":
        horizon = BRAKE_END_S
    elif a.mode == "turn":
        horizon = 26.0
    else:
        horizon = 1.2
    if a.max_s is not None:
        horizon = min(horizon, float(a.max_s))    # smoke test of the script only
    substeps = int(round(DT / dt))
    frame, previous_steer, wp, sim_t = -int(round(SETTLE_S / DT)), 0.0, 0, 0.0
    rows, nonfinite, inputs = [], 0, None
    wall0 = time.time()
    while frame < int(round(horizon / DT)):
        ref = model.GetChassis().GetBody().GetFrameRefToAbs()
        pos = np.array([ref.GetPos().x, ref.GetPos().y])
        wp = frozen.nearest_index(xy, pos, wp)
        tt = frame * DT
        if frame < 0 or a.mode == "settle":
            desired = 0.0
        elif a.mode == "straight":
            desired = next((v for s, e, v in STRAIGHT_STEPS if s <= tt < e), 0.0)
        else:
            desired = TURN_SPEED
        braking_phase = a.mode == "straight" and tt >= BRAKE_START_S
        driver.SetDesiredSpeed(desired)
        for sub in range(substeps):
            ts = sim_t
            driver.Synchronize(ts)
            inputs = driver.GetInputs()
            previous_steer = 0. if frame < 0 else float(np.clip(inputs.m_steering, previous_steer - 2. * dt, previous_steer + 2. * dt))
            inputs.m_steering = previous_steer
            if braking_phase:
                inputs.m_throttle, inputs.m_braking = 0.0, 1.0
            terrain.Synchronize(ts)
            model.Synchronize(ts, inputs, terrain)
            if sub == 0 and frame >= 0:
                row = capture_row(model, terrain, "flat", "ov_check", "flat", "development", frame, ts, inputs, include_tires=False)
                row.update(crm_collect.crm_tire_fields(chrono, veh, vehicle, terrain, WHEEL_SPECS, tire_radii))
                row["engine_motor_speed_radps"] = float(engine.GetMotorSpeed())
                row["engine_motorshaft_torque_nm"] = float(engine.GetOutputMotorshaftTorque())
                state = np.array([float(row[f]) for f in fields], np.float32)
                if not np.isfinite(state).all():
                    nonfinite += 1
                gz = float(tmap.height(row["pos_x_m"], row["pos_y_m"]))
                sink = [tire_radii[n] - (row[f"{n}_spindle_z_m"] - gz) for n, _, _ in WHEEL_SPECS]
                r = {"t": round(tt, 3), "x": row["pos_x_m"], "y": row["pos_y_m"], "h": row["pos_z_m"] - gz,
                     "speed": row["speed_mps"], "vz": row["vel_world_z_mps"], "roll": row["roll_rad"], "pitch": row["pitch_rad"],
                     "yaw": row["yaw_rad"], "desired": desired, "throttle": float(inputs.m_throttle),
                     "braking": float(inputs.m_braking), "steering": float(inputs.m_steering), "sink": sink,
                     "slip": [row[f"{n}_slip_ratio"] for n, _, _ in WHEEL_SPECS],
                     "fz": [row[f"{n}_force_wheel_fz_n"] for n, _, _ in WHEEL_SPECS],
                     "gear": int(transmission.GetCurrentGear()), "engine_radps": row["engine_motor_speed_radps"]}
                if belly is not None:
                    belly.on_frame(frame, model.GetChassisBody())
                    r["belly"] = belly.clear[-1]
                rows.append(r)
            driver.Advance(dt)
            terrain.Advance(dt)
            sim_t += dt
        frame += 1
        if a.mode == "turn" and frame * DT > 4.0 and rows and rows[-1]["x"] < -24.0 and rows[-1]["y"] > 5.0:
            break
    wall = time.time() - wall0
    R = rows
    out = {"vehicle": a.vehicle, "mode": a.mode, "config": str(a.cfg), "config_sha256": sha(a.cfg), "step_s": dt,
           "patch_m": a.size, "n_sph": int(terrain.GetNumSPHParticles()), "bce_per_wheel": nbce,
           "soil_wheel": wheel, "build_s": round(build_s, 2), "wall_s": round(wall, 2),
           "sim_s": round(len(R) * DT + SETTLE_S, 2), "rtf": round((len(R) * DT + SETTLE_S) / max(wall, 1e-9), 3),
           "horizon_s": horizon, "truncated_smoke": a.max_s is not None, "nonfinite_state_frames": nonfinite,
           "breakthrough_depth_m": float(cfg["depth_m"]) + float(cfg.get("breakthrough_margin_m", 0.06))}
    if record is not None:
        blk = record.block("outcome.json")
        json.dumps(blk)
        out["vehicle_block_keys"] = sorted(blk)
        if a.vehicle in ovv.POLARIS_NAMES:
            out["build"] = record.state.get("build")
    out.update(analyze(a.mode, R, seg, xy, out["breakthrough_depth_m"], nonfinite, belly is not None))
    out["all_pass"] = bool(all(out["pass"].values()))
    if a.rows:
        out["rows"] = R
    with open(a.out, "a") as f:
        f.write(json.dumps(out) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "vehicle_block_keys")}), flush=True)


# ============================================================================================ episode checks
REQUIRED_BLOCK = {"polaris": ("name", "json_sha256", "private_data", "frame_shift_m", "spawn_dz_m", "soil_wheel_geometry",
                              "driveline", "driveline_defect_note", "build", "wrapper_sha256", "ov_vehicle_sha256"),
                  "gator": ("name", "soil_wheel_geometry", "wrapper_sha256", "gator_data_sha256"),
                  "hmmwv": ("name", "wrapper_sha256", "ov_vehicle_sha256")}


def check_run(d, expect=None):
    import crm_qa
    d = Path(d)
    qa = crm_qa.check(str(d))
    o = json.loads((d / "outcome.json").read_text())
    launch = json.loads((d / "initial_state_validation.json").read_text())
    z = np.load(d / "trajectory.npz")
    st = z["state"]
    vb = o.get("vehicle")
    name = (vb or {}).get("name")
    kind = "polaris" if (name or "").startswith("polaris") else name
    missing = [k for k in REQUIRED_BLOCK.get(kind, ("name",)) if vb is None or vb.get(k) is None]
    other = {f: "vehicle" in json.loads((d / f).read_text()) for f in ("collection_request.json", "f104_episode.json") if (d / f).exists()}
    h = float(launch["chassis_reference_height_above_bmp_m"])
    r = {"run": d.name, "status": o["status"], "elapsed_s": o["elapsed_s"], "goal_time_s": o.get("goal_time_s"),
         "wall_s": round(o["wall_s"], 1), "rtf": round(o["crm"]["rtf_sim_over_wall"], 3), "host": qa.get("host"),
         "qa_ok": qa["ok"], "qa_flag": qa["flag"], "launch_passed": bool(launch["passed"]), "launch_height_m": round(h, 4),
         "state_shape": list(st.shape), "state_finite": bool(np.isfinite(st).all()),
         "vehicle_block": name, "vehicle_block_missing_keys": missing, "vehicle_block_in_other_files": other,
         "vehicle_extra_npz": (d / "vehicle_extra.npz").exists(),
         "max_wheel_sinkage_m": round(o["crm"]["max_wheel_sinkage_below_bmp_m"], 4)}
    if vb and vb.get("belly"):
        r["belly_min_clearance_m"] = vb["belly"].get("min_clearance_m")
    if vb and vb.get("build"):
        r["mass_kg"] = vb["build"].get("mass_kg")
    ok = qa["ok"] and launch["passed"] and st.shape[1:] == (17,) and bool(np.isfinite(st).all()) and not missing \
        and all(other.values()) and (expect is None or name == expect)
    if kind == "polaris":
        ok = ok and 0.30 <= h <= 0.45 and r["vehicle_extra_npz"]
    r["pass"] = bool(ok)
    return r


def cmd_episode_check(a):
    res = [check_run(d, a.expect) for d in a.runs]
    for r in res:
        print(json.dumps(r))
    if a.out:
        jdump(a.out, res)


def compare_runs(d1, d2):
    d1, d2 = Path(d1), Path(d2)
    out = {"a": str(d1), "b": str(d2), "files": {}}
    same = True
    for f in ("trajectory.npz", "crm_extra.npz", "command_reference.npz", "anchor_state.npz", "vehicle_extra.npz"):
        p, q = d1 / f, d2 / f
        if not p.exists() and not q.exists():
            continue
        if p.exists() != q.exists():
            out["files"][f] = "present in one run only"
            same = False
            continue
        x, y = np.load(p), np.load(q)
        diff = []
        for k in sorted(set(x.files) | set(y.files)):
            if k not in x.files or k not in y.files:
                diff.append(k + " (missing)")
            elif not (x[k].shape == y[k].shape and np.array_equal(x[k], y[k], equal_nan=x[k].dtype.kind == "f")):
                diff.append(k)
        out["files"][f] = "identical" if not diff else {"differs": diff}
        same = same and not diff
    o1, o2 = (json.loads((d / "outcome.json").read_text()) for d in (d1, d2))
    out["status"] = [o1["status"], o2["status"]]
    out["elapsed_s"] = [o1["elapsed_s"], o2["elapsed_s"]]
    out["arrays_identical"] = bool(same)
    out["same_outcome"] = o1["status"] == o2["status"]
    return out


def cmd_compare(a):
    r = compare_runs(a.a, a.b)
    print(json.dumps(r))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("make-data")
    m.add_argument("--chrono-data", required=True)
    m.add_argument("--grid-m", type=float, default=0.10)
    m.add_argument("--force", action="store_true")
    m.add_argument("--compare-s1", default=None, help="S1 scratch Polaris_ov folder to compare the re-framed JSON with")
    r = sp.add_parser("rigid")
    r.add_argument("--chrono-data", required=True)
    r.add_argument("--arms", nargs="+", default=list(ovv.POLARIS_NAMES))
    r.add_argument("--out", default=None)
    f = sp.add_parser("flat-soil")
    f.add_argument("--vehicle", required=True, choices=("hmmwv", "gator") + ovv.POLARIS_NAMES)
    f.add_argument("--mode", required=True, choices=("settle", "straight", "turn"))
    f.add_argument("--cfg", default=str(DEFAULT_CFG))
    f.add_argument("--chrono-data", required=True)
    f.add_argument("--size", type=float, default=80.0)
    f.add_argument("--rows", action="store_true", help="also store every frame")
    f.add_argument("--max-s", type=float, default=None, help="truncate the drive (script smoke test only)")
    f.add_argument("--out", required=True, help="append one JSON line")
    e = sp.add_parser("episode-check")
    e.add_argument("runs", nargs="+")
    e.add_argument("--expect", default=None, help="expected vehicle block name")
    e.add_argument("--out", default=None)
    c = sp.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out", default=None)
    a = p.parse_args()
    {"make-data": cmd_make_data, "rigid": cmd_rigid, "flat-soil": cmd_flat_soil, "episode-check": cmd_episode_check,
     "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
