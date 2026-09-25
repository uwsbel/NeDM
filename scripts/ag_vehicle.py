"""Vehicle switch for the f104-family collectors (arena_gator_20260925, module E2).

One switch selects the vehicle: ``--vehicle gator|hmmwv`` on the wrapper collectors' command line, else the environment
variable ``NEDM_VEHICLE``, else ``hmmwv``.  With ``hmmwv`` the wrappers (``ag_gen_collect_ext.py``,
``ag_crm_collect.py``) install nothing and call the unmodified collectors' ``main`` -- the HMMWV code path is the
original one.  With ``gator`` they patch, at run time and without editing any file, module attributes that the
frozen loop and the soil collectors look up when they run:

  nedm.traverse.scene.build_config   -> the returned config describes the Gator and the spawn height is
                                        ground + 0.35 m instead of + 0.75 m (the three call sites pass
                                        ground + 0.75; the wrapper subtracts 0.40)
  nedm.traverse.scene.create_hmmwv   -> (rigid scene builder) create_vehicle: builds veh.Gator for model "Gator",
  nedm.hmmwv_data.create_hmmwv          otherwise the original create_hmmwv (soil collectors import it from here)
  crm_collect.build_crm              -> (soil) the original build_crm with the wheel geometry swapped: one cylinder per
                                        axle with a calibrated radius/width instead of the HMMWV tyre mesh
  <module>.dump                      -> adds a top-level "vehicle" block to outcome.json, collection_request.json,
                                        f104_episode.json, simulation_provenance.json and collection_meta.json and
                                        writes vehicle_extra.npz next to outcome.json
  observer / frame hook              -> belly-clearance diagnostic (lowest chassis-hull point minus the undisturbed
                                        surface, every recorded frame) and the chassis vertical speed

The Gator itself is Chrono's stock ``veh.Gator``: SMC contact, stock driveline (SIMPLE: rear-wheel drive with the
limited-slip split), stock brakes (SIMPLE, rear axle only, no locking), stock engine and gearbox (the wrapper has no
choice), TMEASY tyres on rigid ground with chassis collision hulls as the HMMWV, rigid tyres + soil cylinders and no
chassis collision on soil, the same tyre step and the same visualisation-off lines as ``create_hmmwv``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VEHICLES = ("hmmwv", "gator")
FROZEN_SPAWN_DZ_M = 0.75        # the literal at traverse_fdm_rgbd_diverse_chrono.py:162, crm_collect.py:177, crm_collect_ext.py:114
GATOR_SPAWN_DZ_M = 0.35         # S2 2.3: +0.35 m is settled at the 0.8 s anchor (vertical speed -0.01 m/s), +0.75 m still bounces
# Soil wheel geometry (one cylinder per axle, spin axis = spindle y, stock tyre widths).  Calibrated on flat soil
# (scripts/ag_soil_calibrate.py: 0.08 m spacing, 1 ms step, crm_main.json soil, production build_crm) = nominal radius
# - 0.09 m: the Gator's sinkage then matches the HMMWV production tyre mesh both at the settled anchor (0.000 vs +0.006 m,
# axle means) and while driving at full throttle (-0.012 vs -0.014 m); see NOTES_E2.md section 3.
GATOR_SOIL_WHEEL = {"front": {"radius_m": 0.19575, "width_m": 0.254},
                    "rear": {"radius_m": 0.2275, "width_m": 0.3048}}
GATOR_NOMINAL = {"front": {"radius_m": 0.28575, "width_m": 0.254}, "rear": {"radius_m": 0.3175, "width_m": 0.3048}}
BELLY_FILE = HERE / "ag_gator_belly.json"
ANNOTATED = ("outcome.json", "collection_request.json", "f104_episode.json", "simulation_provenance.json",
             "collection_meta.json")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


# ============================================================================================ command line switch
def switch_parser():
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--vehicle", choices=VEHICLES, default=None)
    p.add_argument("--gator-soil-radius-front", type=float, default=None)
    p.add_argument("--gator-soil-radius-rear", type=float, default=None)
    p.add_argument("--gator-soil-width-front", type=float, default=None)
    p.add_argument("--gator-soil-width-rear", type=float, default=None)
    p.add_argument("--gator-soil-mesh", action="store_true",
                   help="diagnostic only: couple the stock per-axle Gator tyre meshes (gator_tire{F,R}_coarse.obj) instead of the cylinders")
    p.add_argument("--runtime-fingerprint", default=None)
    return p


def parse_switch(argv):
    """-> (vehicle, options, remaining argv).  The remaining argv (order kept) goes to the unmodified collector."""
    opts, rest = switch_parser().parse_known_args(list(argv))
    vehicle = opts.vehicle or os.environ.get("NEDM_VEHICLE", "hmmwv").strip().lower() or "hmmwv"
    if vehicle not in VEHICLES:
        raise ValueError(f"NEDM_VEHICLE / --vehicle must be one of {VEHICLES}, got {vehicle!r}")
    gator_only = [k for k in ("gator_soil_radius_front", "gator_soil_radius_rear", "gator_soil_width_front",
                              "gator_soil_width_rear") if getattr(opts, k) is not None] + (["gator_soil_mesh"] if opts.gator_soil_mesh else [])
    if vehicle != "gator" and gator_only:
        raise ValueError(f"{gator_only} given without the Gator selected")
    opts.vehicle = vehicle
    return vehicle, opts, rest


def soil_wheel(opts):
    w = json.loads(json.dumps(GATOR_SOIL_WHEEL))
    for axle in ("front", "rear"):
        for q in ("radius", "width"):
            v = getattr(opts, f"gator_soil_{q}_{axle}", None)
            if v is not None:
                w[axle][f"{q}_m"] = float(v)
    for axle in ("front", "rear"):
        if not (0.05 < w[axle]["radius_m"] < 0.5 and 0.05 < w[axle]["width_m"] < 0.5):
            raise ValueError(f"implausible soil wheel {axle}: {w[axle]}")
    if getattr(opts, "gator_soil_mesh", False):
        w = {"front": {"mesh": "gator/gator_tireF_coarse.obj"}, "rear": {"mesh": "gator/gator_tireR_coarse.obj"},
             "diagnostic": "stock Gator tyre meshes (not watertight; ragged marker sets at 0.08 m, S2 2.5)"}
    return w


# ============================================================================================ vehicle factory
def gator_vehicle_block(init, tire_model="TMEASY", chassis_collision="HULLS"):
    return {"model": "Gator", "contact_method": "SMC", "chassis_fixed": False, "init": dict(init),
            "tire_model": tire_model, "chassis_collision": chassis_collision,
            "driveline": "SIMPLE", "brake": "SIMPLE", "brake_locking": False,
            "engine_model": "Gator_EngineSimple (stock, fixed by veh.Gator)",
            "transmission_model": "Gator_AutomaticTransmissionSimple (stock, one gear, fixed by veh.Gator)",
            "steering_type": "Gator_RackPinion (stock)", "spawn_dz_m": GATOR_SPAWN_DZ_M}


def create_gator(config):
    """Mirror of nedm.hmmwv_data.create_hmmwv for veh.Gator (same order of the shared setters)."""
    import pychrono as chrono
    import pychrono.vehicle as veh
    from nedm.hmmwv_data import CONTACT_METHODS, TIRE_MODELS

    vc, init = config["vehicle"], config["vehicle"]["init"]
    if vc["model"] != "Gator":
        raise ValueError("create_gator needs vehicle.model == 'Gator'")
    gator = veh.Gator()
    gator.SetContactMethod(CONTACT_METHODS[vc["contact_method"]])
    gator.SetChassisFixed(bool(vc["chassis_fixed"]))
    gator.SetInitPosition(chrono.ChCoordsysd(chrono.ChVector3d(init["x_m"], init["y_m"], init["z_m"]),
                                             chrono.QuatFromAngleZ(float(init.get("yaw_rad", 0.0)))))
    if "fwd_vel_mps" in init:
        gator.SetInitFwdVel(float(init["fwd_vel_mps"]))
    gator.SetDrivelineType({"SIMPLE": veh.DrivelineTypeWV_SIMPLE, "RWD": veh.DrivelineTypeWV_RWD}[vc.get("driveline", "SIMPLE")])
    gator.SetBrakeType({"SIMPLE": veh.BrakeType_SIMPLE, "SHAFTS": veh.BrakeType_SHAFTS}[vc.get("brake", "SIMPLE")])
    gator.EnableBrakeLocking(bool(vc.get("brake_locking", False)))
    tire = vc["tire_model"]
    if tire not in ("TMEASY", "RIGID", "RIGID_MESH"):
        raise ValueError(f"veh.Gator has no {tire} tyres (any other type silently leaves the vehicle without tyres)")
    gator.SetTireType(TIRE_MODELS[tire])
    gator.SetTireStepSize(config["simulation"]["tire_step_size_s"])
    gator.SetChassisCollisionType({"NONE": veh.CollisionType_NONE, "PRIMITIVES": veh.CollisionType_PRIMITIVES,
                                   "HULLS": veh.CollisionType_HULLS, "MESH": veh.CollisionType_MESH}[str(vc.get("chassis_collision", "NONE"))])
    gator.Initialize()
    gator.SetChassisVisualizationType(chrono.VisualizationType_NONE)
    gator.SetSuspensionVisualizationType(chrono.VisualizationType_NONE)
    gator.SetSteeringVisualizationType(chrono.VisualizationType_NONE)
    gator.SetWheelVisualizationType(chrono.VisualizationType_NONE)
    gator.SetTireVisualizationType(chrono.VisualizationType_NONE)
    gator.GetSystem().SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)   # as create_hmmwv
    return gator


def make_create_vehicle(original):
    def create_vehicle(config):
        if config["vehicle"].get("model") == "Gator":
            return create_gator(config)
        return original(config)
    create_vehicle.ag_original = original
    return create_vehicle


def make_build_config(original):
    """Same signature as scene.build_config; the vehicle block becomes the Gator's and the spawn z drops by 0.40 m."""
    def build_config(arena_dir, start_xyz, start_yaw, *a, **k):
        x, y, z = start_xyz
        cfg = original(arena_dir, (x, y, z - FROZEN_SPAWN_DZ_M + GATOR_SPAWN_DZ_M), start_yaw, *a, **k)
        cfg["vehicle"] = gator_vehicle_block(cfg["vehicle"]["init"], "TMEASY", cfg["vehicle"].get("chassis_collision", "HULLS"))
        return cfg
    build_config.ag_original = original
    return build_config


# ============================================================================================ soil wheel geometry
class _TerrainProxy:
    """Forwards everything to the real CRMTerrain except AddRigidBody, whose geometry is replaced per axle."""

    def __init__(self, real, owner):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def AddRigidBody(self, body, geometry, check_embedded):   # noqa: N802 (Chrono name)
        axle = self._owner.spindle_axle[int(body.GetIdentifier())]
        self._owner.swapped.append(axle)
        return self._real.AddRigidBody(body, self._owner.geometries[axle], check_embedded)


class _VehProxy:
    """Stands in for pychrono.vehicle inside the original build_crm: CRMTerrain(...) returns a _TerrainProxy."""

    def __init__(self, veh, spindle_axle, geometries):
        self._veh, self.spindle_axle, self.geometries, self.swapped, self.terrain = veh, spindle_axle, geometries, [], None

    def __getattr__(self, name):
        return getattr(self._veh, name)

    def CRMTerrain(self, *a):   # noqa: N802
        self.terrain = _TerrainProxy(self._veh.CRMTerrain(*a), self)
        return self.terrain


_KEEP_ALIVE = []


def make_build_crm(original, wheel):
    """The original crm_collect.build_crm (solver, soil, SPH, construct and initialise: unchanged statements) with the
    per-wheel ChBodyGeometry replaced by one cylinder per axle (front: axle 0, rear: axle 1)."""
    def build_crm(chrono, veh, fsi, system, vehicle, arena, meta, cfg):
        spindle_axle, geometries = {}, {}
        for ia, axle in enumerate(vehicle.GetAxles()):
            for w in axle.GetWheels():
                spindle_axle[int(w.GetSpindle().GetIdentifier())] = ia
        for ia, key in ((0, "front"), (1, "rear")):
            g = chrono.ChBodyGeometry()
            if "mesh" in wheel[key]:
                g.coll_meshes.append(chrono.TrimeshShape(chrono.VNULL, chrono.QUNIT,
                                                         veh.GetVehicleDataFile(wheel[key]["mesh"]), chrono.VNULL))
            else:
                g.coll_cylinders.append(chrono.CylinderShape(chrono.VNULL, chrono.ChVector3d(0, 1, 0),
                                                             float(wheel[key]["radius_m"]), float(wheel[key]["width_m"])))
            geometries[ia] = g
        proxy = _VehProxy(veh, spindle_axle, geometries)
        _KEEP_ALIVE.append(proxy)
        terrain = original(chrono, proxy, fsi, system, vehicle, arena, meta, cfg)
        if sorted(proxy.swapped) != [0, 0, 1, 1]:
            raise RuntimeError(f"soil wheel swap expected axles [0, 0, 1, 1], got {proxy.swapped}")
        return terrain._real if isinstance(terrain, _TerrainProxy) else terrain
    build_crm.ag_original = original
    return build_crm


# ============================================================================================ diagnostics
def _rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class BellyClearance:
    """Per recorded frame: min over the chassis-hull underside samples of (point z - undisturbed surface z), plus the
    chassis reference vertical speed.  The surface is the arena BMP sampled in Chrono's own convention (samples on
    the patch edges, TerrainMap evaluated at 511/512 of the world coordinate), i.e. the rigid patch / the soil
    surface before any deformation."""

    def __init__(self, tmap, chrono_data=None):
        spec = json.loads(BELLY_FILE.read_text())
        self.points = np.asarray(spec["points"], np.float64)
        self.tmap, self.spec_sha = tmap, sha(BELLY_FILE)
        self.obj_sha_expected, self.obj_sha_runtime = spec["source_obj_sha256"], None
        if chrono_data:
            obj = Path(chrono_data) / "vehicle" / spec["source_obj"]
            self.obj_sha_runtime = sha(obj) if obj.is_file() else None
            if self.obj_sha_runtime != self.obj_sha_expected:
                raise RuntimeError(f"{obj}: sha256 {self.obj_sha_runtime} differs from the belly sample file's {self.obj_sha_expected}")
        self.scale = (tmap.pixels - 1) / tmap.pixels
        self.frames, self.clear, self.argmin, self.vz, self.below = [], [], [], [], []

    def measure(self, chassis_body):
        ref = chassis_body.GetFrameRefToAbs()
        p, q = ref.GetPos(), ref.GetRot()
        world = self.points @ _rotmat((q.e0, q.e1, q.e2, q.e3)).T + np.array([p.x, p.y, p.z])
        ground = self.tmap.height(world[:, 0] * self.scale, world[:, 1] * self.scale)
        c = world[:, 2] - ground
        i = int(np.argmin(c))
        return float(c[i]), i, float(ref.GetPosDt().z), int((c < 0.).sum())

    def on_frame(self, frame, chassis_body):
        c, i, vz, below = self.measure(chassis_body)
        self.frames.append(int(frame)); self.clear.append(c); self.argmin.append(i); self.vz.append(vz); self.below.append(below)

    def summary(self):
        if not self.clear:
            return {"frames": 0}
        c, vz = np.asarray(self.clear), np.asarray(self.vz)
        neg = np.flatnonzero(c < 0.)
        return {"frames": int(len(c)), "anchor_clearance_m": float(c[0]), "min_clearance_m": float(c.min()),
                "min_clearance_frame": int(self.frames[int(c.argmin())]),
                "min_clearance_point_chassis_xyz_m": self.points[self.argmin[int(c.argmin())]].round(4).tolist(),
                "frames_below_surface": int(len(neg)), "fraction_frames_below_surface": float(len(neg) / len(c)),
                "first_below_surface_s": float(self.frames[int(neg[0])] * 0.05) if len(neg) else None,
                "anchor_chassis_vz_mps": float(vz[0]), "max_abs_chassis_vz_first_1s_mps": float(np.abs(vz[:20]).max()),
                "definition": "clearance = lowest sampled point of the chassis collision hull underside (471 grid points "
                              "at 0.10 m + 78 hull vertices, scripts/ag_gator_belly.json) minus the undisturbed arena "
                              "surface under it (BMP, Chrono node-on-edge convention); negative = hull below the "
                              "original surface (on soil the chassis is not coupled, so it passes through)",
                "belly_points_sha256": self.spec_sha, "chassis_col_obj_sha256": self.obj_sha_expected,
                "chassis_col_obj_sha256_runtime": self.obj_sha_runtime}

    def save(self, out):
        np.savez_compressed(Path(out) / "vehicle_extra.npz", frame=np.asarray(self.frames, np.int64),
                            belly_clearance_min_m=np.asarray(self.clear, np.float32),
                            belly_argmin_point=np.asarray(self.argmin, np.int32),
                            belly_points_below_surface=np.asarray(self.below, np.int32),
                            chassis_vz_mps=np.asarray(self.vz, np.float32),
                            belly_points_chassis_m=self.points.astype(np.float32))


# ============================================================================================ provenance
class VehicleRecord:
    """The "vehicle" block written into the collectors' JSON files (Gator mode only)."""

    def __init__(self, world, opts, chrono_data=None, soil_wheel_geom=None, wrapper=None):
        self.world, self.opts, self.soil_wheel, self.belly = world, opts, soil_wheel_geom, None
        data = Path(chrono_data) / "vehicle" / "gator" if chrono_data else None
        self.data_sha = ({str(p.relative_to(data.parent)): sha(p) for p in sorted(data.rglob("*")) if p.is_file()}
                         if data is not None and data.is_dir() else None)
        self.wrapper = wrapper
        self.fingerprint = None

    def block(self, name=None):
        b = {"name": "gator", "model": "pychrono.vehicle.Gator (Chrono stock model)", "world": self.world,
             "switch": "ag_vehicle.py (--vehicle gator / NEDM_VEHICLE=gator)",
             "ag_vehicle_sha256": sha(__file__), "wrapper": self.wrapper,
             "wrapper_sha256": sha(self.wrapper) if self.wrapper else None,
             "spawn_dz_m": GATOR_SPAWN_DZ_M, "frozen_spawn_dz_m": FROZEN_SPAWN_DZ_M,
             "contact_method": "SMC", "driveline": "SIMPLE (stock: rear axle driven, limited-slip split)",
             "brake": "SIMPLE (stock: rear axle only), no locking",
             "engine": "Gator_EngineSimple + Gator_AutomaticTransmissionSimple (stock, one gear)",
             "tyres": "TMEASY" if self.world == "rigid" else "RIGID_MESH wheel bodies coupled to soil through one cylinder per axle",
             "chassis_collision": "HULLS (gator_chassis_col.obj)" if self.world == "rigid" else "NONE (chassis not coupled to the soil)",
             "follower": "frozen make_driver (gains 0.8 / 0.6, 0.05; 5 m look-ahead) unchanged",
             "gator_data_sha256": self.data_sha, "runtime_fingerprint": self.fingerprint}
        if self.soil_wheel is not None:
            mesh = "mesh" in self.soil_wheel.get("front", {})
            b["soil_wheel_geometry"] = {**self.soil_wheel, "nominal_tyre": GATOR_NOMINAL,
                                        "shape": ("chrono.TrimeshShape per axle (DIAGNOSTIC stock tyre meshes)" if mesh else
                                                  "chrono.CylinderShape per axle, axis = spindle y, centred on the spindle"),
                                        "calibrated_default": GATOR_SOIL_WHEEL}
            if mesh:
                b["tyres"] = "RIGID_MESH wheel bodies coupled to soil through the stock Gator tyre meshes (diagnostic)"
        if name == "outcome.json" and self.belly is not None:
            b["belly"] = self.belly.summary()
        return b


def make_dump(original, record):
    """record.belly is looked up at call time (the frozen loop's namespace is copied before the observer exists)."""
    def dump(path, value):
        name = Path(path).name
        if name == "outcome.json" and record.belly is not None:
            record.belly.save(Path(path).parent)
        if name in ANNOTATED and isinstance(value, dict):
            value = {**value, "vehicle": record.block(name)}
        return original(path, value)
    dump.ag_original = original
    return dump


def check_gator_fingerprint(path):
    """The Gator runtime fingerprint must bind the vehicle library and the Gator data files (it also lists the HMMWV
    data, so gen_collect_ext's own HMMWV check keeps passing)."""
    runtime = json.loads(Path(path).read_text())["runtime_sha256"]
    if not (runtime and any("_vehicle.so" in k for k in runtime) and any("/vehicle/gator/" in k for k in runtime)):
        raise ValueError(f"Runtime fingerprint {path} lacks the native vehicle library or the Gator data files")
    return {"path": str(Path(path).resolve()), "sha256": sha(path), "entries": len(runtime),
            "gator_entries": sum("/vehicle/gator/" in k for k in runtime)}
