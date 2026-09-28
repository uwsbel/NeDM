#!/usr/bin/env python3
"""Checks for the M113 module (offroad_vehicles_20260927, M2): scripts/ov_m113.py + configs/crm_m113.json.

  build    construct both arms (no soil), print mass / shoes / sprocket radius / bindings           (gate a, any node)
  rigid    rigid-ground checks, CPU only (seconds):
             identity   ov_m113 construction (simple brake) vs veh.M113 wrapper: max |pose difference| over a
                        0.8 s braked settle + 2 s full throttle (must be 0)
             json_x1    the JSON transmission route with the stock ratios vs the C++ transmission (must be 0)
             hold       braked on a tilted rigid plane (10 / 15 deg): simple vs shafts brake (sprocket rotation,
                        chassis speed, ground-shoe speed)
             drive      full throttle on flat rigid ground, m113 vs m113_g4 (speed at 2 / 5 / 8 s)
  tilt     braked hold on TILTED SOIL (gate b): a flat 14 x 6 x 0.24 m soil box with the production soil / SPH /
           solver statements of crm_collect.build_crm (the arena Construct call is swapped for the box, gravity
           tilted by the slope), 0.8 s braked settle, then braked for --hold-s, then optionally full throttle for
           --throttle-s.  Vehicles: m113 / m113_g4 (with --m113-pad / --m113-brake) or gator (ag_vehicle cylinders,
           production 1 ms config) as the reference.  Holds = |mean speed over the last 1 s of the hold| <= 0.10 m/s.
  qa       one finished run folder: crm_qa.check, (n, 17) finite state, launch check, vehicle block, rtf  (gate c/d)

Local soil runs only under flock /tmp/luffy_crm.lock.  Outputs are JSON lines on stdout (and --out).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import ov_m113 as om  # noqa: E402


def _chrono(data_root=None):
    import pychrono as chrono
    import pychrono.vehicle as veh
    if data_root:
        chrono.SetChronoDataPath(str(Path(data_root)) + "/")
        (getattr(veh, "SetVehicleDataPath", None) or veh.SetDataPath)(str(Path(data_root) / "vehicle") + "/")
    return chrono, veh


def _config(x=0.0, y=0.0, z=0.75, yaw=0.0, model="M113"):
    return {"vehicle": {"model": model, "contact_method": "SMC", "chassis_fixed": False,
                        "init": {"x_m": x, "y_m": y, "z_m": z, "yaw_rad": yaw}},
            "simulation": {"tire_step_size_s": om.STEP_S}}


def _emit(rec, out=None):
    line = json.dumps(rec, default=float)
    print(line, flush=True)
    if out:
        with open(out, "a") as f:
            f.write(line + "\n")


# ============================================================================================ build
def cmd_build(a):
    chrono, veh = _chrono(a.chrono_data)
    rows = []
    for arm in om.VEHICLES:
        m = om.create_m113(_config(), arm, a.m113_brake, a.m113_pad)
        info = m.info()
        info.update(arm=arm, brake=a.m113_brake, pad=a.m113_pad, bodies=len(list(m.system.GetBodies())),
                    pad_geometry_boxes=len(om.pad_geometry(chrono, a.m113_pad).coll_boxes),
                    belly_points=int(len(om.belly_points())))
        rows.append(info)
    import pychrono.fsi as fsi   # the soil module: only chrono-build-fsi has it
    need = {"veh.CRMTerrain": hasattr(veh, "CRMTerrain"), "fsi.SPHParameters": hasattr(fsi, "SPHParameters"),
            "veh.M113_Vehicle_SinglePin": hasattr(veh, "M113_Vehicle_SinglePin"),
            "veh.ReadTransmissionJSON": hasattr(veh, "ReadTransmissionJSON"),
            "chrono.ChShaftsClutch": hasattr(chrono, "ChShaftsClutch")}
    _emit({"check": "build", "arms": rows, "bindings": need, "pychrono": str(Path(chrono.__file__).parent),
           "pass": all(need.values()) and all(r["mass_kg"] > 11000 and r["shoes"] == {"left": 63, "right": 64} for r in rows)}, a.out)


# ============================================================================================ rigid
def _rigid(chrono, veh, system, slope_deg=0.0, mu=0.9):
    th = math.radians(slope_deg)
    system.SetGravitationalAcceleration(chrono.ChVector3d(-9.81 * math.sin(th), 0, -9.81 * math.cos(th)))
    mat = chrono.ChContactMaterialSMC()
    mat.SetYoungModulus(2e7); mat.SetFriction(mu); mat.SetRestitution(0.01)   # scene.build_config rigid terrain
    terrain = veh.RigidTerrain(system)
    terrain.AddPatch(mat, chrono.CSYSNORM, 200.0, 200.0)
    terrain.Initialize()
    return terrain


def _solver(chrono, system, threads=4):
    system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)            # as crm_collect.build_crm
    system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
    system.SetNumThreads(threads, 1, 1)
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)


def _wrapper_m113(chrono, veh):
    m = veh.M113()
    m.SetContactMethod(chrono.ChContactMethod_SMC); m.SetTrackShoeType(veh.TrackShoeType_SINGLE_PIN)
    m.SetDrivelineType(veh.DrivelineTypeTV_BDS); m.SetEngineType(veh.EngineModelType_SHAFTS)
    m.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SHAFTS); m.SetBrakeType(veh.BrakeType_SIMPLE)
    m.SetChassisCollisionType(veh.CollisionType_NONE); m.SetChassisFixed(False)
    m.SetInitPosition(chrono.ChCoordsysd(chrono.ChVector3d(0, 0, 0.75), chrono.QUNIT)); m.Initialize()
    for f in ("SetChassisVisualizationType", "SetSprocketVisualizationType", "SetIdlerVisualizationType",
              "SetSuspensionVisualizationType", "SetIdlerWheelVisualizationType", "SetRoadWheelVisualizationType",
              "SetTrackShoeVisualizationType"):
        getattr(m, f)(chrono.VisualizationType_NONE)
    m.GetSystem().SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    return m


def _drive_rigid(chrono, veh, model, sync, advance, system, T, control, slope=0.0, log_dt=0.25):
    terrain = _rigid(chrono, veh, system, slope)
    inputs = veh.DriverInputs()
    t, poses, logs, nl = 0.0, [], [], 0.0
    chassis = model.GetChassisBody()
    while t < T - 1e-9:
        inputs.m_steering, inputs.m_throttle, inputs.m_braking = control(t)
        terrain.Synchronize(t); sync(t, inputs); terrain.Advance(om.STEP_S); advance(om.STEP_S); t += om.STEP_S
        ref = chassis.GetFrameRefToAbs(); p = ref.GetPos()
        poses.append((p.x, p.y, p.z))
        if t >= nl - 1e-9:
            nl += log_dt
            logs.append({"t": round(t, 3), "x": p.x, "vx": ref.GetPosDt().x})
    return np.asarray(poses), logs


def cmd_rigid(a):
    chrono, veh = _chrono(a.chrono_data)
    settle_then_throttle = lambda t: (0.0, 0.0, 1.0) if t < 0.8 else (0.0, 1.0, 0.0)   # noqa: E731
    out = {"check": "rigid"}
    # identity: ov_m113 (simple brake) vs the veh.M113 wrapper
    w = _wrapper_m113(chrono, veh); _solver(chrono, w.GetSystem())
    pw, _ = _drive_rigid(chrono, veh, w, lambda t, i: w.Synchronize(t, i), w.Advance, w.GetSystem(), 2.8, settle_then_throttle)
    m = om.create_m113(_config(), "m113", "simple", "flat"); _solver(chrono, m.system)
    pm, lm = _drive_rigid(chrono, veh, m, m.Synchronize, m.Advance, m.system, 2.8, settle_then_throttle)
    out["identity_max_abs_pose_diff_m"] = float(np.abs(pw - pm).max())
    out["identity_end_x_m"] = [float(pw[-1, 0]), float(pm[-1, 0])]
    # the JSON route with stock ratios vs the C++ transmission
    saved = om.G4_TRANSMISSION
    om.G4_TRANSMISSION = om.ASSET_DIR / "M113_AutomaticTransmissionShafts_x1_check.json"
    om.ARMS["m113_x1check"] = {"gear_scale": 1.000001, "transmission": "x1 check"}
    try:
        mj = om.create_m113(_config(), "m113_x1check", "simple", "flat"); _solver(chrono, mj.system)
        pj, _ = _drive_rigid(chrono, veh, mj, mj.Synchronize, mj.Advance, mj.system, 2.8, settle_then_throttle)
        out["json_x1_max_abs_pose_diff_m"] = float(np.abs(pj - pm).max())
    finally:
        om.G4_TRANSMISSION = saved
        om.ARMS.pop("m113_x1check", None)
    # braked hold on tilted rigid ground
    hold = []
    for slope in (10.0, 15.0):
        for brake in ("simple", "shafts"):
            mm = om.create_m113(_config(), "m113", brake, "flat"); _solver(chrono, mm.system)
            spr = mm.sprockets[veh.LEFT].GetGearBody(); ang = [0.0]
            chassis = mm.GetChassisBody()

            def adv(dt, mm=mm, spr=spr, ang=ang, chassis=chassis):
                mm.Advance(dt)
                ref = chassis.GetFrameRefToAbs()
                ang[0] += (spr.GetAngVelParent() - ref.GetAngVelParent()).Dot(ref.GetRot().GetAxisY()) * dt
            p, logs = _drive_rigid(chrono, veh, mm, mm.Synchronize, adv, mm.system, 2.8, lambda t: (0.0, 0.0, 1.0), slope)
            last = [r["vx"] for r in logs if r["t"] > 1.8]
            hold.append({"slope_deg": slope, "brake": brake, "vx_at_0.8s": [r["vx"] for r in logs if abs(r["t"] - 0.75) < 1e-6 or abs(r["t"] - 1.0) < 1e-6],
                         "mean_vx_last_1s": float(np.mean(last)), "x_travel_m": float(p[-1, 0] - p[0, 0]),
                         "sprocket_rotation_rel_chassis_rad": float(ang[0])})
    out["hold"] = hold
    # full throttle on flat rigid ground, stock vs gearbox / 4
    drive = []
    for arm in om.VEHICLES:
        mm = om.create_m113(_config(), arm, "shafts", "flat"); _solver(chrono, mm.system)
        _, logs = _drive_rigid(chrono, veh, mm, mm.Synchronize, mm.Advance, mm.system, 8.8, settle_then_throttle, 0.0, 0.5)
        drive.append({"arm": arm, "speed_at_s": {str(k): float(np.interp(0.8 + k, [r["t"] for r in logs], [r["vx"] for r in logs])) for k in (1, 2, 5, 8)}})
    out["drive_flat"] = drive
    out["pass_identity"] = out["identity_max_abs_pose_diff_m"] == 0.0
    _emit(out, a.out)


# ============================================================================================ tilted soil
class _BoxTerrain:
    """Wraps the real CRMTerrain inside the unmodified build_crm: Construct -> a flat box, gravity -> tilted."""

    def __init__(self, real, owner):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def RegisterVehicle(self, v):   # noqa: N802
        self._owner.registered = True
        return self._real.RegisterVehicle(getattr(v, "_ov_real", v))

    def AddRigidBody(self, body, geometry, check):   # noqa: N802
        self._owner.bodies.append(body)
        return self._real.AddRigidBody(body, geometry, check)

    def SetGravitationalAcceleration(self, g):   # noqa: N802
        th = math.radians(self._owner.slope)
        c = self._owner.chrono
        return self._real.SetGravitationalAcceleration(c.ChVector3d(-9.81 * math.sin(th), 0, -9.81 * math.cos(th)))

    def Construct(self, *args):   # noqa: N802
        o, c = self._owner, self._owner.chrono
        sides = args[-1]
        return self._real.Construct(c.ChVector3d(o.L, o.W, o.depth), c.ChVector3d(o.L / 2, 0, 0), sides)


class _BoxVeh:
    def __init__(self, veh, chrono, slope, L, W, depth):
        self._veh, self.chrono, self.slope, self.L, self.W, self.depth = veh, chrono, slope, L, W, depth
        self.bodies, self.registered, self.terrain = [], False, None

    def __getattr__(self, name):
        return getattr(self._veh, name)

    def CRMTerrain(self, *a):   # noqa: N802
        self.terrain = _BoxTerrain(self._veh.CRMTerrain(*a), self)
        return self.terrain


def cmd_tilt(a):
    chrono, veh = _chrono(a.chrono_data)
    import pychrono.fsi as fsi
    import crm_collect
    cfg = crm_collect.merged_config(a.crm_config)
    L, W, depth = a.length, a.width, float(cfg["depth_m"])
    x0 = a.x0
    if a.vehicle in om.VEHICLES:
        model = om.create_m113(_config(x0, 0.0, depth + om.FROZEN_SPAWN_DZ_M), a.vehicle, a.m113_brake, a.m113_pad)
        vehicle, system, chassis = model.standin, model.system, model.GetChassisBody()
        rec = om.M113Record(a.vehicle, a.m113_brake, a.m113_pad)
        build = om.make_build_crm(crm_collect.build_crm, {"pad": a.m113_pad}, rec)
        sync = model.Synchronize
        if abs(float(cfg["step_s"]) - om.STEP_S) > 1e-12:
            raise ValueError("M113 tilt runs need configs/crm_m113.json")
    elif a.vehicle == "gator":
        import ag_vehicle as agv
        g = agv.create_gator({"vehicle": {**agv.gator_vehicle_block({"x_m": x0, "y_m": 0.0, "z_m": depth + agv.GATOR_SPAWN_DZ_M, "yaw_rad": 0.0},
                                                                     "RIGID_MESH", "NONE")},
                              "simulation": {"tire_step_size_s": float(cfg["step_s"])}})
        vehicle, system, chassis, model, rec = g.GetVehicle(), g.GetSystem(), g.GetChassisBody(), g, None
        wheel = agv.soil_wheel(argparse.Namespace(gator_soil_radius_front=None, gator_soil_radius_rear=None,
                                                  gator_soil_width_front=None, gator_soil_width_rear=None, gator_soil_mesh=False))
        build = agv.make_build_crm(crm_collect.build_crm, wheel)
        sync = lambda t, i: g.Synchronize(t, i, terr)   # noqa: E731
    else:
        raise ValueError(a.vehicle)
    proxy = _BoxVeh(veh, chrono, a.slope, L, W, depth)
    meta = {"size_m": L, "bmp": "unused.bmp", "height_min_m": 0.0, "height_max_m": 1.0}
    t_build = time.time()
    terr = build(chrono, proxy, fsi, system, vehicle, Path("."), meta, cfg)
    terr = getattr(terr, "_real", terr)
    build_s = time.time() - t_build
    dt = float(cfg["step_s"])
    inputs = veh.DriverInputs()
    T = 0.8 + a.hold_s + a.throttle_s
    t, logs, nl, w0 = 0.0, [], 0.0, time.time()
    sprocket_rot = 0.0
    shoes = model.shoe_bodies() if a.vehicle in om.VEHICLES else []
    spr = model.sprockets[veh.LEFT].GetGearBody() if a.vehicle in om.VEHICLES else None
    status = "ok"
    while t < T - 1e-9:
        if t < 0.8 + a.hold_s:
            inputs.m_steering, inputs.m_throttle, inputs.m_braking = 0.0, 0.0, 1.0
        else:
            inputs.m_steering, inputs.m_throttle, inputs.m_braking = 0.0, 1.0, 0.0
        terr.Synchronize(t)
        sync(t, inputs)
        terr.Advance(dt)
        t += dt
        ref = chassis.GetFrameRefToAbs()
        if spr is not None:
            sprocket_rot += (spr.GetAngVelParent() - ref.GetAngVelParent()).Dot(ref.GetRot().GetAxisY()) * dt
        if t >= nl - 1e-9:
            nl += 0.1
            p = ref.GetPos()
            r = {"t": round(t, 3), "x": p.x - x0, "z": p.z, "vx": ref.GetPosDt().x, "wall": time.time() - w0}
            if shoes:
                zs = np.array([b.GetPos().z for b in shoes])
                low = [b for b, z in zip(shoes, zs) if z < zs.min() + 0.05]
                r["v_ground_shoes"] = float(np.mean([b.GetPosDt().x for b in low]))
                r["pad_bottom_min_minus_surface_m"] = float(zs.min() + om.pad_bottom_z(a.m113_pad) - depth)
                r["sprocket_rot_rad"] = sprocket_rot
                r["fz_soil_kn"] = float(sum(terr.GetFsiBodyForce(b).z for b in shoes)) / 1000.0
            logs.append(r)
            if not all(math.isfinite(v) for v in (p.x, p.y, p.z)) or abs(p.z - depth) > 3:
                status = "unstable"; break
    wall = time.time() - w0
    hold_end = 0.8 + a.hold_s
    last = [r for r in logs if hold_end - 1.0 <= r["t"] <= hold_end + 1e-9]
    at_settle = min(logs, key=lambda r: abs(r["t"] - 0.8))
    mean_v = float(np.mean([r["vx"] for r in last])) if last else float("nan")
    res = {"check": "tilt", "vehicle": a.vehicle, "slope_deg": a.slope, "pad": a.m113_pad if a.vehicle in om.VEHICLES else None,
           "brake": a.m113_brake if a.vehicle in om.VEHICLES else "stock", "config": a.crm_config, "step_s": dt,
           "status": status, "n_sph": int(terr.GetNumSPHParticles()), "coupling": rec.coupling if rec else len(proxy.bodies),
           "build_s": build_s, "sim_s": t, "wall_s": wall, "wall_per_sim_s": wall / max(t, 1e-9),
           "vx_at_settle_end": at_settle["vx"], "mean_vx_last_1s_of_hold": mean_v,
           "travel_during_hold_m": float(next(r["x"] for r in logs if r["t"] >= hold_end - 1e-9) - at_settle["x"]) if logs else None,
           "holds": bool(abs(mean_v) <= 0.10), "launch_speed_ok": bool(abs(at_settle["vx"]) <= 1.0)}
    if a.throttle_s > 0:
        tail = [r for r in logs if r["t"] > hold_end]
        res["throttle_end_vx"] = tail[-1]["vx"] if tail else None
        res["throttle_travel_m"] = (tail[-1]["x"] - tail[0]["x"]) if tail else None
    for k in ("v_ground_shoes", "sprocket_rot_rad", "fz_soil_kn", "pad_bottom_min_minus_surface_m"):
        if logs and k in logs[-1]:
            res[k + "_at_hold_end"] = next(r[k] for r in logs if r["t"] >= hold_end - 1e-9)
    res["logs"] = logs[::2]
    _emit(res, a.out)


# ============================================================================================ qa of a run folder
def cmd_qa(a):
    import crm_qa
    rows = []
    for d in a.runs:
        d = str(Path(d).resolve())
        q = crm_qa.check(d)
        o = json.load(open(Path(d) / "outcome.json"))
        z = np.load(Path(d) / "trajectory.npz")
        st = z["state"]
        launch = json.load(open(Path(d) / "initial_state_validation.json"))
        v = o.get("vehicle") or {}
        cols = [str(s) for s in z["state_fields"]]
        rows.append({"run": d, "crm_qa_ok": q["ok"], "crm_qa_flag": q["flag"], "status": o["status"],
                     "elapsed_s": o["elapsed_s"], "wall_s": o["wall_s"], "rtf_sim_over_wall": o["crm"]["rtf_sim_over_wall"],
                     "wall_per_sim_s": 1.0 / max(o["crm"]["rtf_sim_over_wall"], 1e-9), "physics_dt_s": o["crm"]["physics_dt_s"],
                     "state_shape": list(st.shape), "state_finite": bool(np.isfinite(st).all()), "state_fields": cols,
                     "launch_passed": launch.get("passed"), "launch": {k: launch.get(k) for k in ("body_horizontal_speed_mps", "chassis_reference_height_above_bmp_m", "start_xy_error_m")},
                     "vehicle_block": v.get("name"), "vehicle_soil_coupling": v.get("soil_coupling"),
                     "belly_min_clearance_m": (v.get("belly") or {}).get("min_clearance_m"),
                     "max_wheel_sinkage_below_bmp_m": o["crm"].get("max_wheel_sinkage_below_bmp_m"),
                     "quarter_fz_mean_kn": [float(x) for x in (st[:, 7:11].mean(0) / 1000.0)],
                     "sprocket_omega_abs_mean": [float(x) for x in np.abs(st[:, 11:15]).mean(0)],
                     "goal_reached": o["goal_reached"], "n_sph": o["crm"]["n_sph"]})
    _emit({"check": "qa", "runs": rows, "pass": all(r["crm_qa_ok"] and r["state_finite"] and r["state_shape"][1:] == [17]
                                                  and r["vehicle_block"] for r in rows)}, a.out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("build", "rigid", "tilt", "qa"))
    p.add_argument("runs", nargs="*")
    p.add_argument("--chrono-data", default=None, help="Chrono data root (contains vehicle/)")
    p.add_argument("--crm-config", default=str(HERE.parent / "configs" / "crm_m113.json"))
    p.add_argument("--vehicle", default="m113", choices=om.VEHICLES + ("gator",))
    p.add_argument("--m113-pad", default=om.DEFAULT_PAD, choices=om.PADS)
    p.add_argument("--m113-brake", default=om.DEFAULT_BRAKE, choices=om.BRAKES)
    p.add_argument("--slope", type=float, default=10.0)
    p.add_argument("--hold-s", type=float, default=2.0)
    p.add_argument("--throttle-s", type=float, default=0.0)
    p.add_argument("--length", type=float, default=14.0)
    p.add_argument("--width", type=float, default=6.0)
    p.add_argument("--x0", type=float, default=7.0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    {"build": cmd_build, "rigid": cmd_rigid, "tilt": cmd_tilt, "qa": cmd_qa}[a.cmd](a)


if __name__ == "__main__":
    main()
