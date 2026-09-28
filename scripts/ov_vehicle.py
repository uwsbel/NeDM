"""Vehicle switch and factory for the Polaris / M113 soil study (offroad_vehicles_20260927, module M1).

A copy of the scripts/ag_vehicle.py pattern (that file stays frozen and knows only hmmwv / gator).  The soil dispatcher
scripts/ov_crm_collect.py reads ``--vehicle <name>`` and installs, at run time and without editing any file, the same
module-attribute swaps ag_vehicle uses, before the unmodified collector (crm_collect.py or crm_collect_ext.py) runs:

  hmmwv        nothing but the JSON writer: a "vehicle" block is added to outcome.json, collection_request.json and
               f104_episode.json.  Vehicle, soil wheels, spawn and every array are the frozen collector's own.
  gator        delegated to ag_vehicle (the same functions ag_crm_collect.py installs), so Gator rows are the same rows.
  polaris      Chrono's JSON Polaris (MRZR data, 1,378 kg, 4WD) with the chassis reference moved to mid-wheelbase
               (copy of the two top-level JSON files in assets/traverse/vehicles/ov_polaris/), stock SimpleDriveline as
               shipped (known power defect, see DRIVELINE_NOTE), spawn ground + 0.40 m, soil wheels = one calibrated
               cylinder per wheel r = 0.25 m, stock width 0.2121 m.
  polaris_pc   the re-framed Polaris with the power-corrected driveline (conical ratios 1.0, gearbox ratios x 0.25).
  polaris_4wd  the re-framed Polaris with Chrono's shafts driveline Polaris/Polaris_4WD.json (open differentials).
  polaris_w08  the primary Polaris with soil cylinders r = 0.33 m (+0.08 m; wheel-size sensitivity arm).
  m113, m113_g4  handled by scripts/ov_m113.py (module M2), imported lazily by the dispatcher.

``NEDM_VEHICLE`` is never read: if it is set in the environment the switch refuses to run (this study selects the
vehicle only with ``--vehicle``; absent flag = hmmwv).

The Polaris swaps (all at run time):
  nedm.traverse.scene.build_config   -> Polaris vehicle block; spawn = collector's ground + 0.75 - 0.75 + 0.40 m
  nedm.hmmwv_data.create_hmmwv,      -> create_vehicle: builds the Polaris for model "Polaris" (PolarisModel adapter),
  nedm.traverse.scene.create_hmmwv      otherwise the original create_hmmwv
  crm_collect.build_crm              -> ag_vehicle.make_build_crm with the Polaris cylinders (unchanged soil statements)
  <module>.dump                      -> "vehicle" block + vehicle_extra.npz (belly clearance from the visual chassis mesh)
The vehicle data path is pointed at a private runtime folder (two symlinks: Polaris_ov -> the asset folder, Polaris ->
the build's own stock Polaris folder) only while the vehicle is constructed, then restored.  Every file the loader will
open is checked to exist (a missing JSON aborts the Chrono loader with no Python exception) and hashed into the record.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VEHICLES = ("hmmwv", "gator", "polaris", "polaris_pc", "polaris_4wd", "polaris_w08", "m113", "m113_g4")
POLARIS_NAMES = ("polaris", "polaris_pc", "polaris_4wd", "polaris_w08")
M113_NAMES = ("m113", "m113_g4")
FROZEN_SPAWN_DZ_M = 0.75          # crm_collect.py:177, crm_collect_ext.py:114
POLARIS_SPAWN_DZ_M = 0.40         # S1 5.1: re-framed model settled at the 0.8 s anchor (vz 0.004 m/s); +0.75 m bounces
FRAME_SHIFT_M = {"x": 1.35763, "y": 0.0, "z": -0.42}   # added to every stock chassis-frame location (S1 section 4)
POLARIS_NOMINAL_TYRE = {"radius_m": 0.330229, "width_m": 0.2121}   # Polaris_RigidTire.json
POLARIS_SOIL_RADIUS_M = 0.25      # S1 5.2: stock 0.330 - 0.08; settled/driving sinkage matches the HMMWV mesh
POLARIS_W08_SOIL_RADIUS_M = 0.33  # +0.08 m sensitivity arm
ASSET_DIR = HERE.parent / "assets" / "traverse" / "vehicles" / "ov_polaris"
PRIVATE_SUBDIR = "Polaris_ov"
BELLY_FILE_NAME = "ov_polaris_belly.json"
ANNOTATED = ("outcome.json", "collection_request.json", "f104_episode.json", "simulation_provenance.json",
             "collection_meta.json")

DRIVELINE_NOTE = ("Chrono's ChSimpleDriveline as shipped (Polaris/Polaris_DrivelineSimple.json: 50/50 front/rear, "
                  "limited-slip bias 2.0, conical ratio 0.25) has a reduction defect (ChSimpleDriveline.cpp:106,115, "
                  "commit dfff7a809): the driveshaft speed is wheel speed x 0.25 but wheel torque is driveshaft torque / "
                  "0.25, so the engine sees 1/16 of its kinematic speed, the gearbox stays in first and wheel power is "
                  "about 16x engine power. The engine-speed, engine-torque and power/energy columns are not physical. "
                  "At 0-6 m/s it behaves like a one-gear ~50 kW 4WD vehicle (S1 3.3, S5 3.2).")
POLARIS_ARMS = {
    "polaris": {
        "vehicle_json": "Polaris_ov/Polaris_ovc_stock.json",
        "engine_json": "Polaris/Polaris_EngineSimpleMap.json",
        "transmission_json": "Polaris/Polaris_AutomaticTransmissionSimpleMap.json",
        "soil_radius_m": POLARIS_SOIL_RADIUS_M,
        "driveline": "stock SimpleDriveline as shipped (Polaris/Polaris_DrivelineSimple.json)",
        "driveline_defect": DRIVELINE_NOTE,
    },
    "polaris_pc": {
        "vehicle_json": "Polaris_ov/Polaris_ovc_pc.json",
        "engine_json": "Polaris/Polaris_EngineSimpleMap.json",
        "transmission_json": "Polaris_ov/Polaris_ovc_AutomaticTransmissionSimpleMap_pc.json",
        "soil_radius_m": POLARIS_SOIL_RADIUS_M,
        "driveline": ("power-corrected SimpleDriveline (Polaris_ov/Polaris_ovc_DrivelineSimple_pc.json: conical ratios "
                      "1.0, split and limited-slip bias as stock) + gearbox ratios x 0.25 "
                      "(Polaris_ov/Polaris_ovc_AutomaticTransmissionSimpleMap_pc.json); S5 5.1 templates"),
        "driveline_defect": ("power conserved (the stock 0.25 reduction moved into the gearbox); no torque converter: "
                             "wheelspin raises engine speed and can shift the gearbox up (S1 3.3.2)"),
    },
    "polaris_4wd": {
        "vehicle_json": "Polaris_ov/Polaris_ovc_4wd.json",
        "engine_json": "Polaris/Polaris_EngineSimpleMap.json",
        "transmission_json": "Polaris/Polaris_AutomaticTransmissionSimpleMap.json",
        "soil_radius_m": POLARIS_SOIL_RADIUS_M,
        "driveline": "Chrono ShaftsDriveline4WD Polaris/Polaris_4WD.json (open differentials, power-correct), stock gearbox",
        "driveline_defect": ("power-correct; no torque converter: under wheelspin the gearbox shifts up and does not come "
                             "back down (rolled back on rigid 20-25 deg, S1 5.1)"),
    },
    "polaris_w08": {
        "vehicle_json": "Polaris_ov/Polaris_ovc_stock.json",
        "engine_json": "Polaris/Polaris_EngineSimpleMap.json",
        "transmission_json": "Polaris/Polaris_AutomaticTransmissionSimpleMap.json",
        "soil_radius_m": POLARIS_W08_SOIL_RADIUS_M,
        "driveline": "stock SimpleDriveline as shipped (Polaris/Polaris_DrivelineSimple.json)",
        "driveline_defect": DRIVELINE_NOTE,
    },
}
POLARIS_TYRES = {"TMEASY": "Polaris/Polaris_TMeasyTire.json", "RIGID": "Polaris/Polaris_RigidTire.json",
                 "RIGID_MESH": "Polaris/Polaris_RigidTire.json"}   # soil: the collectors ask for RIGID_MESH; the
# Polaris rigid tyre (cylinder contact) is used, as in S1's smokes; its contact is unused on CRM (FSI cylinders carry it)


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
    # Gator-only overrides (same names and meaning as ag_vehicle; calibration / sensitivity runs only)
    p.add_argument("--gator-soil-radius-front", type=float, default=None)
    p.add_argument("--gator-soil-radius-rear", type=float, default=None)
    p.add_argument("--gator-soil-width-front", type=float, default=None)
    p.add_argument("--gator-soil-width-rear", type=float, default=None)
    p.add_argument("--gator-soil-mesh", action="store_true")
    p.add_argument("--runtime-fingerprint", default=None)
    return p


def parse_switch(argv, environ=None):
    """-> (vehicle, options, remaining argv).  The remaining argv (order kept) goes to the unmodified collector (or,
    for m113*, to ov_m113.install, which may strip its own flags)."""
    environ = os.environ if environ is None else environ
    if environ.get("NEDM_VEHICLE", "").strip():
        raise RuntimeError("NEDM_VEHICLE is set in the environment; this study never uses it (select the vehicle with "
                           "--vehicle and submit jobs with 'env -u NEDM_VEHICLE sbatch ...')")
    opts, rest = switch_parser().parse_known_args(list(argv))
    vehicle = opts.vehicle or "hmmwv"
    gator_only = [k for k in ("gator_soil_radius_front", "gator_soil_radius_rear", "gator_soil_width_front",
                              "gator_soil_width_rear") if getattr(opts, k) is not None] + (["gator_soil_mesh"] if opts.gator_soil_mesh else [])
    if vehicle != "gator" and gator_only:
        raise ValueError(f"{gator_only} given without the Gator selected")
    opts.vehicle = vehicle
    return vehicle, opts, rest


# ============================================================================================ private vehicle data
def load_manifest(asset_dir=ASSET_DIR):
    """Verify the private vehicle-data folder against its MANIFEST (every listed file present, sha256 equal)."""
    asset_dir = Path(asset_dir)
    mpath = asset_dir / "MANIFEST.json"
    if not mpath.is_file():
        raise FileNotFoundError(f"private Polaris data folder incomplete: {mpath} missing (stage assets/traverse/vehicles/ov_polaris)")
    man = json.loads(mpath.read_text())
    bad = {}
    for rel, want in man["files_sha256"].items():
        p = asset_dir / rel
        got = sha(p) if p.is_file() else None
        if got != want:
            bad[rel] = got
    if bad:
        raise RuntimeError(f"private Polaris data folder {asset_dir} differs from its MANIFEST: {bad}")
    return man, sha(mpath)


def _json_refs(path):
    """Every '... Input File' reference of a Chrono vehicle JSON.  Regex, not json.loads: some stock Chrono files are
    not strict JSON (RapidJSON accepts comments / trailing commas)."""
    import re
    text = Path(path).read_text()
    return [m.group(2) for m in re.finditer(r'"([^"]*Input File)"\s*:\s*"([^"]+)"', text)]


class PrivateDataPath:
    """A runtime vehicle-data folder holding Polaris_ov (-> the asset folder) and Polaris (-> the build's stock folder).
    Created once per process, removed at exit (symlinks only)."""

    _cache = {}

    @classmethod
    def get(cls, build_vehicle_data):
        key = str(Path(build_vehicle_data).resolve())
        if key not in cls._cache:
            cls._cache[key] = cls(key)
        return cls._cache[key]

    def __init__(self, build_vehicle_data):
        stock = Path(build_vehicle_data) / "Polaris"
        if not (stock / "Polaris.json").is_file():
            raise FileNotFoundError(f"stock Polaris data missing under {build_vehicle_data}")
        self.root = Path(tempfile.mkdtemp(prefix="ov_polaris_vehdata_"))
        os.symlink(str(ASSET_DIR / PRIVATE_SUBDIR), str(self.root / PRIVATE_SUBDIR))
        os.symlink(str(stock), str(self.root / "Polaris"))
        self.stock = stock
        atexit.register(shutil.rmtree, str(self.root), True)

    def resolve(self, rel):
        return self.root / rel

    def files_for(self, arm):
        """Every JSON the loader opens for this arm (checked present) -> {relative path: sha256}."""
        a = POLARIS_ARMS[arm]
        top = self.resolve(a["vehicle_json"])
        rels = [a["vehicle_json"]]
        todo = [a["vehicle_json"]]
        while todo:
            r = todo.pop()
            p = self.resolve(r)
            if not p.is_file():
                raise FileNotFoundError(f"Polaris JSON {r} not found at {p} (would abort the Chrono loader)")
            for ref in _json_refs(p):
                if ref not in rels:
                    rels.append(ref)
                    todo.append(ref)
        for extra in (a["engine_json"], a["transmission_json"], POLARIS_TYRES["RIGID"], POLARIS_TYRES["TMEASY"]):
            if extra not in rels:
                rels.append(extra)
        out = {}
        for r in rels:
            p = self.resolve(r)
            if not p.is_file():
                raise FileNotFoundError(f"Polaris JSON {r} not found at {p} (would abort the Chrono loader)")
            out[r] = sha(p)
        assert top.is_file()
        return out


# ============================================================================================ Polaris model
class PolarisModel:
    """Duck-typed stand-in for the veh.HMMWV_Full wrapper: the collectors call GetVehicle, GetSystem, GetChassis,
    GetChassisBody, Synchronize(t, inputs, terrain), Advance(dt) and the five visualisation setters."""

    def __init__(self, vehicle_json, engine_json, transmission_json, tire_json, init_pos, init_yaw, tire_step,
                 fwd_vel=0.0):
        import pychrono as ch
        import pychrono.vehicle as veh
        self.vehicle = veh.WheeledVehicle(str(vehicle_json), ch.ChContactMethod_SMC)
        self.vehicle.Initialize(ch.ChCoordsysd(ch.ChVector3d(*init_pos), ch.QuatFromAngleZ(float(init_yaw))), float(fwd_vel))
        self.vehicle.GetChassis().SetFixed(False)
        engine = veh.ReadEngineJSON(str(engine_json))
        trans = veh.ReadTransmissionJSON(str(transmission_json))
        self.vehicle.InitializePowertrain(veh.ChPowertrainAssembly(engine, trans))
        self.tires = []
        for axle in self.vehicle.GetAxles():
            for wheel in axle.GetWheels():
                t = veh.ReadTireJSON(str(tire_json))
                t.SetStepsize(float(tire_step))
                self.vehicle.InitializeTire(t, wheel, ch.VisualizationType_NONE)
                self.tires.append(t)
        self.SetChassisVisualizationType(ch.VisualizationType_NONE)
        self.SetSuspensionVisualizationType(ch.VisualizationType_NONE)
        self.SetSteeringVisualizationType(ch.VisualizationType_NONE)
        self.SetWheelVisualizationType(ch.VisualizationType_NONE)
        self.SetTireVisualizationType(ch.VisualizationType_NONE)
        self.vehicle.GetSystem().SetCollisionSystemType(ch.ChCollisionSystem.Type_BULLET)   # as create_hmmwv

    def GetVehicle(self):   # noqa: N802 (Chrono names)
        return self.vehicle

    def GetSystem(self):   # noqa: N802
        return self.vehicle.GetSystem()

    def GetChassis(self):   # noqa: N802
        return self.vehicle.GetChassis()

    def GetChassisBody(self):   # noqa: N802
        return self.vehicle.GetChassisBody()

    def Synchronize(self, t, inputs, terrain):   # noqa: N802
        self.vehicle.Synchronize(t, inputs, terrain)

    def Advance(self, dt):   # noqa: N802
        self.vehicle.Advance(dt)

    def SetChassisVisualizationType(self, v):   # noqa: N802
        self.vehicle.SetChassisVisualizationType(v)

    def SetSuspensionVisualizationType(self, v):   # noqa: N802
        self.vehicle.SetSuspensionVisualizationType(v)

    def SetSteeringVisualizationType(self, v):   # noqa: N802
        self.vehicle.SetSteeringVisualizationType(v)

    def SetWheelVisualizationType(self, v):   # noqa: N802
        self.vehicle.SetWheelVisualizationType(v)

    def SetTireVisualizationType(self, v):   # noqa: N802
        self.vehicle.SetTireVisualizationType(v)


def build_info(model):
    """Measured on the constructed vehicle: mass and spindle positions relative to the chassis reference."""
    import pychrono.vehicle as veh
    v = model.GetVehicle()
    ref = v.GetChassisBody().GetFrameRefToAbs()
    sp = {}
    for name, axle, side in (("fl", 0, veh.LEFT), ("fr", 0, veh.RIGHT), ("rl", 1, veh.LEFT), ("rr", 1, veh.RIGHT)):
        loc = ref.TransformPointParentToLocal(v.GetSpindlePos(axle, side))
        sp[name] = [round(float(loc.x), 5), round(float(loc.y), 5), round(float(loc.z), 5)]
    return {"mass_kg": round(float(v.GetMass()), 3), "spindles_rel_reference_m": sp,
            "tire_radius_m": round(float(v.GetTire(0, veh.LEFT).GetRadius()), 6)}


def create_polaris(config, state=None):
    """Mirror of nedm.hmmwv_data.create_hmmwv for the Polaris (config['vehicle'] = polaris_vehicle_block(...))."""
    import pychrono.vehicle as veh
    vc, init = config["vehicle"], config["vehicle"]["init"]
    if vc.get("model") != "Polaris":
        raise ValueError("create_polaris needs vehicle.model == 'Polaris'")
    arm = vc["arm"]
    a = POLARIS_ARMS[arm]
    tire = vc["tire_model"]
    if tire not in POLARIS_TYRES:
        raise ValueError(f"no Polaris tyre for {tire!r}")
    load_manifest()
    build_data = veh.GetVehicleDataFile("")           # the build's vehicle data folder (configure_chrono_data_paths)
    private = PrivateDataPath.get(build_data)
    files = private.files_for(arm)                    # every file present, hashed (raises before the loader aborts)
    setter = getattr(veh, "SetVehicleDataPath", None) or veh.SetDataPath
    setter(str(private.root) + "/")
    try:
        model = PolarisModel(private.resolve(a["vehicle_json"]), private.resolve(a["engine_json"]),
                             private.resolve(a["transmission_json"]), private.resolve(POLARIS_TYRES[tire]),
                             (init["x_m"], init["y_m"], init["z_m"]), float(init.get("yaw_rad", 0.0)),
                             config["simulation"]["tire_step_size_s"], float(init.get("fwd_vel_mps", 0.0)))
    finally:
        setter(build_data)
    if veh.GetVehicleDataFile("") != build_data:
        raise RuntimeError("vehicle data path not restored after the Polaris construction")
    if state is not None:
        state["files_sha256"] = files
        state["build"] = build_info(model)
        state["tire_json"] = POLARIS_TYRES[tire]
    return model


def polaris_vehicle_block(arm, init, tire_model="TMEASY", chassis_collision="NONE"):
    a = POLARIS_ARMS[arm]
    return {"model": "Polaris", "arm": arm, "contact_method": "SMC", "chassis_fixed": False, "init": dict(init),
            "tire_model": tire_model, "chassis_collision": chassis_collision,
            "vehicle_json": a["vehicle_json"], "engine_json": a["engine_json"],
            "transmission_json": a["transmission_json"], "spawn_dz_m": POLARIS_SPAWN_DZ_M}


def make_create_vehicle(original, state=None):
    def create_vehicle(config):
        if config["vehicle"].get("model") == "Polaris":
            return create_polaris(config, state)
        return original(config)
    create_vehicle.ov_original = original
    return create_vehicle


def make_build_config(original, arm):
    """Same signature as scene.build_config; the vehicle block becomes the Polaris arm's and the spawn z is
    ground + 0.40 m instead of + 0.75 m."""
    def build_config(arena_dir, start_xyz, start_yaw, *a, **k):
        x, y, z = start_xyz
        cfg = original(arena_dir, (x, y, z - FROZEN_SPAWN_DZ_M + POLARIS_SPAWN_DZ_M), start_yaw, *a, **k)
        cfg["vehicle"] = polaris_vehicle_block(arm, cfg["vehicle"]["init"], "TMEASY", cfg["vehicle"].get("chassis_collision", "NONE"))
        return cfg
    build_config.ov_original = original
    return build_config


def polaris_soil_wheel(arm):
    r = float(POLARIS_ARMS[arm]["soil_radius_m"])
    w = float(POLARIS_NOMINAL_TYRE["width_m"])
    return {"front": {"radius_m": r, "width_m": w}, "rear": {"radius_m": r, "width_m": w}}


# ============================================================================================ belly diagnostic
def _belly_base():
    import ag_vehicle as agv   # numpy only at import
    return agv.BellyClearance


def make_belly(tmap, chrono_data=None):
    """ag_vehicle.BellyClearance reading the Polaris belly points (lower envelope of the visual chassis mesh)."""
    base = _belly_base()

    class PolarisBelly(base):
        def __init__(self, tmap, chrono_data=None):   # noqa: D401 (same fields as ag_vehicle.BellyClearance)
            path = ASSET_DIR / BELLY_FILE_NAME
            spec = json.loads(path.read_text())
            self.points = np.asarray(spec["points"], np.float64)
            self.tmap, self.spec_sha = tmap, sha(path)
            self.obj_sha_expected, self.obj_sha_runtime = spec["source_obj_sha256"], None
            self.spec_meta = {k: spec[k] for k in spec if k != "points"}
            if chrono_data:
                obj = Path(chrono_data) / "vehicle" / spec["source_obj"]
                self.obj_sha_runtime = sha(obj) if obj.is_file() else None
                if self.obj_sha_runtime != self.obj_sha_expected:
                    raise RuntimeError(f"{obj}: sha256 {self.obj_sha_runtime} differs from the belly sample file's {self.obj_sha_expected}")
            self.scale = (tmap.pixels - 1) / tmap.pixels
            self.frames, self.clear, self.argmin, self.vz, self.below = [], [], [], [], []

        def summary(self):
            s = super().summary()
            if s.get("frames"):
                s["definition"] = ("clearance = lowest sampled point of the Polaris VISUAL chassis mesh underside (lower "
                                   f"envelope on a {self.spec_meta.get('grid_m')} m grid, {len(self.points)} points, "
                                   f"assets/traverse/vehicles/ov_polaris/{BELLY_FILE_NAME}; the JSON Polaris has no "
                                   "chassis collision shape) minus the undisturbed arena surface under it (BMP, Chrono "
                                   "node-on-edge convention); negative = body below the original surface (on soil the "
                                   "chassis is not coupled, so it passes through)")
                s["belly_points_source"] = "visual mesh " + self.spec_meta.get("source_obj", "")
                s.pop("chassis_col_obj_sha256", None)
                s.pop("chassis_col_obj_sha256_runtime", None)
                s["chassis_visual_obj_sha256"] = self.obj_sha_expected
                s["chassis_visual_obj_sha256_runtime"] = self.obj_sha_runtime
            return s

    return PolarisBelly(tmap, chrono_data)


# ============================================================================================ provenance
class PolarisRecord:
    """The "vehicle" block written into the collectors' JSON files for the Polaris arms."""

    def __init__(self, arm, world, chrono_data=None, wrapper=None):
        self.arm, self.world, self.wrapper, self.chrono_data = arm, world, wrapper, chrono_data
        self.soil_wheel = polaris_soil_wheel(arm)
        self.belly = None
        self.state = {}
        self.manifest, self.manifest_sha = load_manifest()

    def block(self, name=None):
        a = POLARIS_ARMS[self.arm]
        b = {"name": self.arm, "model": "pychrono.vehicle.WheeledVehicle from Chrono's Polaris JSON (MRZR data), re-framed",
             "world": self.world, "switch": "ov_vehicle.py (--vehicle %s)" % self.arm,
             "ov_vehicle_sha256": sha(__file__), "wrapper": self.wrapper,
             "wrapper_sha256": sha(self.wrapper) if self.wrapper else None,
             "vehicle_json": a["vehicle_json"], "engine_json": a["engine_json"],
             "transmission_json": a["transmission_json"], "tire_json": self.state.get("tire_json"),
             "json_sha256": self.state.get("files_sha256"),
             "private_data": {"folder": "assets/traverse/vehicles/ov_polaris", "manifest_sha256": self.manifest_sha,
                              "files_sha256": self.manifest["files_sha256"]},
             "frame_shift_m": dict(FRAME_SHIFT_M),
             "frame_note": ("chassis reference moved from the stock front-axle point (0.397 m below the axle line) to "
                            "mid-wheelbase, 0.023 m above the front axle line: every chassis-frame location of the two "
                            "top-level JSON files + frame_shift_m; physics-identical to the stock model by test (S1 5.1)"),
             "spawn_dz_m": POLARIS_SPAWN_DZ_M, "frozen_spawn_dz_m": FROZEN_SPAWN_DZ_M,
             "contact_method": "SMC", "driveline": a["driveline"], "driveline_defect_note": a["driveline_defect"],
             "brake": "BrakeShafts (stock, 2,000 N m on all four wheels)",
             "engine": "Polaris_EngineSimpleMap (stock) + " + a["transmission_json"],
             "tyres": ("RIGID wheel bodies (Polaris_RigidTire.json) coupled to soil through one cylinder per wheel"
                       if self.world == "soil" else "TMEASY"),
             "chassis_collision": "NONE (the JSON Polaris has no chassis collision shape; chassis not coupled to the soil)",
             "follower": "frozen make_driver (gains 0.8 / 0.6, 0.05; 5 m look-ahead) unchanged",
             "build": self.state.get("build")}
        if self.world == "soil":
            b["soil_wheel_geometry"] = {**self.soil_wheel, "nominal_tyre": POLARIS_NOMINAL_TYRE,
                                        "shape": "chrono.CylinderShape per wheel (one per axle definition), axis = spindle y, centred on the spindle",
                                        "calibrated_default": polaris_soil_wheel("polaris"),
                                        "calibration": "S1 5.2: flat soil 0.08 m / 1 ms, settled +0.003/+0.004 m and driving -0.008/-0.016 m sinkage vs HMMWV mesh +0.006/-0.014 m"}
        if name == "outcome.json" and self.belly is not None:
            b["belly"] = self.belly.summary()
        return b


class HmmwvRecord:
    """Vehicle block for HMMWV rows through the new dispatcher (nothing else is patched)."""

    def __init__(self, world, wrapper=None, crm_config=None):
        self.world, self.wrapper, self.belly, self.crm_config = world, wrapper, None, crm_config

    def block(self, name=None):
        return {"name": "hmmwv", "model": "pychrono.vehicle.HMMWV_Full via nedm.hmmwv_data.create_hmmwv (unchanged)",
                "world": self.world, "switch": "ov_vehicle.py (--vehicle hmmwv or no --vehicle)",
                "ov_vehicle_sha256": sha(__file__), "wrapper": self.wrapper,
                "wrapper_sha256": sha(self.wrapper) if self.wrapper else None,
                "patched": "JSON writer only (this block); vehicle, soil wheels, spawn and arrays are the frozen collector's",
                "spawn_dz_m": FROZEN_SPAWN_DZ_M,
                "soil_wheel_geometry": "production tyre mesh (crm config tire_mesh) on every wheel, unchanged build_crm",
                "follower": "frozen make_driver (gains 0.8 / 0.6, 0.05; 5 m look-ahead) unchanged"}


def make_dump(original, record):
    """Same as ag_vehicle.make_dump (record.belly looked up at call time)."""
    def dump(path, value):
        name = Path(path).name
        if name == "outcome.json" and record.belly is not None:
            record.belly.save(Path(path).parent)
        if name in ANNOTATED and isinstance(value, dict):
            value = {**value, "vehicle": record.block(name)}
        return original(path, value)
    dump.ov_original = original
    return dump
