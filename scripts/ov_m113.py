"""Chrono's M113 tracked vehicle in the frozen soil collector (offroad_vehicles_20260927, module M2).

Imported lazily by the soil dispatcher scripts/ov_crm_collect.py (module M1) for ``--vehicle m113`` / ``--vehicle
m113_g4``.  Interface (see ov_crm_collect.py):

    install(ctx) -> dict(scene_hook=..., frame_hook=..., argv=...) for ctx.vehicle in VEHICLES, else None
        ctx: types.SimpleNamespace(vehicle, base_module, crm_collect, argv, wrapper, source_root, chrono_data, ...)
        argv (returned) = ctx.argv without the M113-only flags below.

M113-only flags (removed from the argv before the unmodified collector sees it; diagnostics only, defaults = the
declared model):
    --m113-pad {flat,thick,grouser}   soil-coupled geometry of each track shoe (default flat)
    --m113-brake {shafts,simple}      track brake (default shafts, see "brake" below)

Nothing is patched for any other vehicle name.  For m113 / m113_g4 the same kind of run-time module-attribute swaps
as ag_vehicle.py are installed (no file is edited):

  nedm.hmmwv_data.create_hmmwv,      -> builds the M113 for model "M113" (else the original function) and returns an
  nedm.traverse.scene.create_hmmwv      M113Model object that stands in for the HMMWV wrapper the collector calls hmmwv
  nedm.traverse.scene.build_config   -> vehicle block "M113"; spawn unchanged (ground + 0.75 m; settles to ~0.62 m)
  crm_collect.build_crm              -> the original statements; RegisterVehicle gets the real tracked vehicle and each
                                        of the 127 track shoes is coupled to the soil as one pad box (instead of the
                                        tyre mesh on 4 spindles); refuses any soil step other than 0.5 ms
  crm_collect.crm_tire_fields        -> the four "wheels" become four track quarters (mapping below)
  traverse_fdm_rgbd_diverse_chrono.make_driver
                                     -> the frozen follower is built on the real tracked vehicle (same gains)
  <module>.dump                      -> "vehicle" block in outcome.json, collection_request.json, f104_episode.json
                                        (+ vehicle_extra.npz with the hull clearance next to outcome.json)
Only the base collector crm_collect is supported (crm_collect_ext binds crm_tire_fields and the follower differently).

The vehicle
  Chrono's M113 C++ model (pychrono.vehicle.M113_Vehicle_SinglePin, the class veh.M113 builds; same construction
  sequence and defaults as veh.M113.Initialize, checked bit for bit on rigid ground by scripts/ov_m113_check.py rigid):
  SMC contact, single-pin shoes (63 left + 64 right), brake-differential steering driveline (BDS: steering brakes
  the inner track), shafts engine (M113_EngineShafts) + automatic shafts transmission, no chassis collision, all
  visualisation off, Bullet.  11,343 kg.  Chassis reference point = the front sprocket axis (stock), 2.0 m ahead of
  the centre of mass: goal / exit / blockage tests and the pose columns refer to that point.
  - m113      stock transmission M113_AutomaticTransmissionShafts (ratios 0.240 / 0.427 / 0.685 / 0.962).
  - m113_g4   gearbox ratios / 4 (declared arm: the stock drive push is <= 0.35 of the weight): the same transmission
              written as JSON (assets/traverse/vehicles/ov_m113/M113_AutomaticTransmissionShafts_g4.json, every other
              value = the C++ class's) and loaded with ReadTransmissionJSON.
  - brake     "shafts" (default): Chrono's ChTrackBrakeShafts construction (a brake shaft on the chassis + a
              ChShaftsClutch to each sprocket axle, 10,000 N m, shaft inertia 0.4 = M113_TrackBrakeShafts.json),
              added from Python, with the BDS input combination of ChTrackDrivelineBDS::CombineDriverInputs
              (left = braking + max(steering, 0), right = braking + max(-steering, 0)); the stock simple brakes then get
              zero input.  Reason (ov_m113_check.py rigid): the stock M113_BrakeSimple is a ChLinkLockBrake whose
              Coulomb torque flips sign every step (it never sticks), so the braked M113 rolls back at ~0.8 m/s on a
              10 deg RIGID slope; and veh.M113.SetBrakeType(SHAFTS) is silently ignored for single-pin shoes
              (M113_TrackAssemblySinglePin always builds M113_BrakeSimple).  "simple" = stock, diagnostic only.
  - soil      step 0.5 ms (configs/crm_m113.json; 1 ms is unstable with SMC shoes).  Each shoe body carries one pad
              box: flat = 0.154 x 0.38 x 0.03 m (full pitch x full width, bottom face = the shoe's pad bottom, shoe
              z = -0.03 m); thick = 0.154 x 0.38 x 0.16 m, same bottom; grouser = flat + a 0.04 x 0.38 x 0.08 m
              transverse ridge under the pad centre.  Chrono's own filter (boxes > 2 (layers - 1) spacing) keeps no
              M113 box at 0.08 m spacing, i.e. the demo recipe would give the tracks no soil contact.

Quarter mapping of the four wheel fields (tire_fl, tire_fr, tire_rl, tire_rr = left front, right front, left rear,
right rear), recomputed every call because the shoes circulate:
  quarter           = side (left / right track) x (shoe chassis-frame x > X_SPLIT_M = the middle road wheel -> front)
  force fx, fy, fz  = sum of the FSI force on the quarter's shoes, in the chassis heading frame (heading = chassis
                      y axis x up, as for a wheel's spin axis): fz = the quarter's soil vertical load
  spindle omega     = that side's sprocket angular speed about the chassis y axis (front and rear quarter share it)
  wheel_vx          = forward speed of the quarter's middle road-wheel hub (road wheel 1 front, 3 rear)
  slip ratio        = (omega x sprocket assembly radius - wheel_vx) / max(|wheel_vx|, 0.1)
  spindle position  = the lowest shoe pad-bottom point of the quarter (pad bottom-face centre, world frame); tyre
                      radius = 0.  The collector's breakthrough rule then reads "the lowest pad of a quarter lies more
                      than soil depth + margin (0.30 m) below the undisturbed surface for 5 frames": same threshold
                      semantics as "a tyre's lowest point below the surface by more than 0.30 m".
The 17-column state (tire_normal_force_omega_pt) keeps its layout: 7 chassis columns, 4 quarter soil loads, 4 sprocket
speeds (L, R, L, R), engine speed and torque.  Belly points = the M113 hull underside from M113_Chassis.cpp (bottom
box x -4.17..0.041 m, y +-0.85 m, z -0.143 m on a 0.10 m grid, plus the glacis bottom edge), chassis frame.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VEHICLES = ("m113", "m113_g4")
PADS = ("flat", "thick", "grouser")
BRAKES = ("shafts", "simple")
DEFAULT_PAD = "flat"
DEFAULT_BRAKE = "shafts"
STEP_S = 0.0005
FROZEN_SPAWN_DZ_M = 0.75          # crm_collect.py:177 (kept for the M113: settles from 0.75 to ~0.62 m, S2 1.2)
SHOES_EXPECTED = 127              # 63 left + 64 right (M113_TrackAssemblySinglePin.cpp:64,73)
X_SPLIT_M = -1.989                # chassis-frame x of the middle road wheel (road wheels at -0.655 ... -3.322 m)
TRACK_Y_M = 1.0795                # track centre lines at +-1.0795 m (M113_Vehicle.cpp:49)
MIDDLE_WHEEL = {"front": 1, "rear": 3}
PAD_BOTTOM_Z_M = -0.03            # shoe-frame z of the pad bottom face (M113_TrackShoeSinglePin.cpp:84)
BRAKE_MAX_NM = 10000.0            # M113_BrakeSimple.h:39 / M113_TrackBrakeShafts.json
BRAKE_SHAFT_INERTIA = 0.4         # M113_TrackBrakeShafts.json
ASSET_DIR = HERE.parent / "assets" / "traverse" / "vehicles" / "ov_m113"
G4_TRANSMISSION = ASSET_DIR / "M113_AutomaticTransmissionShafts_g4.json"
ARMS = {"m113": {"gear_scale": 1.0, "transmission": "M113_AutomaticTransmissionShafts (C++ class, stock ratios 0.240/0.427/0.685/0.962, reverse -0.151)"},
        "m113_g4": {"gear_scale": 0.25, "transmission": "assets/traverse/vehicles/ov_m113/M113_AutomaticTransmissionShafts_g4.json (= the C++ class with every gear ratio / 4)"}}
STOCK_GEARS = (0.240, 0.427, 0.685, 0.962)   # M113_AutomaticTransmissionShafts.cpp:39-44 (reverse -0.151)
ANNOTATED = ("outcome.json", "collection_request.json", "f104_episode.json", "simulation_provenance.json",
             "collection_meta.json")
M113_FLAGS = ("--m113-pad", "--m113-brake")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


# ============================================================================================ geometry
def pad_boxes(pad):
    """-> list of (centre xyz, dims xyz) in the shoe body frame; all bottoms at PAD_BOTTOM_Z_M except the grouser tip."""
    flat = ((0.0, 0.0, -0.015), (0.154, 0.38, 0.03))
    if pad == "flat":
        return [flat]
    if pad == "thick":   # three marker layers, same bottom face (S2 "thick")
        return [((0.0, 0.0, 0.05), (0.154, 0.38, 0.16))]
    if pad == "grouser":  # flat pad + one transverse ridge (0.08 m = one particle spacing) under the pad centre
        return [flat, ((0.0, 0.0, PAD_BOTTOM_Z_M - 0.04), (0.04, 0.38, 0.08))]
    raise ValueError(f"unknown pad {pad!r}; one of {PADS}")


def pad_bottom_z(pad):
    return min(c[2] - 0.5 * d[2] for c, d in pad_boxes(pad))


def pad_geometry(chrono, pad):
    g = chrono.ChBodyGeometry()
    for c, d in pad_boxes(pad):
        g.coll_boxes.append(chrono.BoxShape(chrono.ChVector3d(*c), chrono.QUNIT, chrono.ChVector3d(*d)))
    return g


def belly_points(grid=0.10):
    """M113 hull underside in the chassis frame (M113_Chassis.cpp:60-76: width 1.70 m, bottom from x = -4.17 to 0.041 m
    at z = -0.143 m, glacis from (0.041, -0.143) to (0.214, 0.343))."""
    xs = np.arange(-4.17, 0.041 + 1e-9, grid)
    ys = np.linspace(-0.85, 0.85, int(round(1.70 / grid)) + 1)
    pts = [(x, y, -0.143) for x in np.r_[xs, 0.041] for y in ys]
    for s in np.linspace(0.0, 1.0, 11)[1:]:
        x, z = 0.041 + s * (0.214 - 0.041), -0.143 + s * (0.343 + 0.143)
        pts += [(x, y, z) for y in ys]
    return np.asarray(pts, np.float64)


# ============================================================================================ vehicle
class _Tire:
    def GetRadius(self):   # noqa: N802 (Chrono names)
        return 0.0


class _Wheel:
    def __init__(self, body):
        self._body = body

    def GetSpindle(self):   # noqa: N802
        return self._body


class _Axle:
    def __init__(self, bodies):
        self._wheels = [_Wheel(b) for b in bodies]

    def GetWheels(self):   # noqa: N802
        return self._wheels


class TrackedStandIn:
    """Stands in for the wheeled vehicle inside the frozen collector: forwards everything to the real ChTrackedVehicle
    and adds the few wheel queries the collector makes (one fake axle listing the 127 shoe bodies for build_crm's
    coupling loop, tyre radius 0, spindle position = lowest pad bottom of a track quarter)."""

    def __init__(self, real, model):
        object.__setattr__(self, "_ov_real", real)
        object.__setattr__(self, "_ov_model", model)

    def __getattr__(self, name):
        return getattr(self._ov_real, name)

    def GetAxles(self):   # noqa: N802
        return [_Axle(self._ov_model.shoe_bodies())]

    def GetTire(self, axle, side):   # noqa: N802
        return _Tire()

    def GetSpindlePos(self, axle, side):   # noqa: N802
        return self._ov_model.quarter_lowest(axle, side)


class M113Model:
    """The object the collector calls ``hmmwv``: GetVehicle / GetSystem / GetChassis / GetChassisBody / Synchronize."""

    def __init__(self, chrono, veh, config, arm, brake, pad):
        self.chrono, self.veh, self.arm, self.brake, self.pad = chrono, veh, arm, brake, pad
        vc, init = config["vehicle"], config["vehicle"]["init"]
        # === the veh.M113.Initialize sequence (M113.cpp:117-226) for SINGLE_PIN + BDS, SMC, no chassis collision
        v = veh.M113_Vehicle_SinglePin(False, veh.DrivelineTypeTV_BDS, veh.BrakeType_SIMPLE, False, False, False,
                                       chrono.ChContactMethod_SMC, veh.CollisionType_NONE)
        v.GetSystem().SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)   # m_collsysType default BULLET
        v.CreateTrack(True)
        for side in (veh.LEFT, veh.RIGHT):
            v.GetTrackAssembly(side).SetWheelCollisionType(True, True, True)
        pos = chrono.ChCoordsysd(chrono.ChVector3d(init["x_m"], init["y_m"], init["z_m"]),
                                 chrono.QuatFromAngleZ(float(init.get("yaw_rad", 0.0))))
        v.Initialize(pos, float(init.get("fwd_vel_mps", 0.0)))
        v.GetDriveline().SetGyrationMode(False)
        self.engine = veh.M113_EngineShafts("Engine")
        if ARMS[arm]["gear_scale"] == 1.0:
            self.transmission = veh.M113_AutomaticTransmissionShafts("Transmission")
            self.transmission_file = None
        else:
            if not G4_TRANSMISSION.is_file():
                raise FileNotFoundError(G4_TRANSMISSION)
            self.transmission = veh.ReadTransmissionJSON(str(G4_TRANSMISSION))
            self.transmission_file = str(G4_TRANSMISSION)
        v.InitializePowertrain(veh.ChPowertrainAssembly(self.engine, self.transmission))
        v.InitializeInertiaProperties()
        for f in ("SetChassisVisualizationType", "SetSprocketVisualizationType", "SetIdlerVisualizationType",
                  "SetSuspensionVisualizationType", "SetIdlerWheelVisualizationType", "SetRoadWheelVisualizationType",
                  "SetTrackShoeVisualizationType"):
            getattr(v, f)(chrono.VisualizationType_NONE)
        v.GetSystem().SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)   # as create_hmmwv
        self.vehicle = v
        self.system = v.GetSystem()
        self.standin = TrackedStandIn(v, self)
        self.sides = (veh.LEFT, veh.RIGHT)
        self._shoes = {s: [v.GetTrackShoe(s, i).GetShoeBody() for i in range(v.GetNumTrackShoes(s))] for s in self.sides}
        n = sum(len(b) for b in self._shoes.values())
        if n != SHOES_EXPECTED:
            raise RuntimeError(f"M113 has {n} track shoes, expected {SHOES_EXPECTED}")
        self.sprockets = {s: v.GetTrackAssembly(s).GetSprocket() for s in self.sides}
        self.sprocket_radius = float(self.sprockets[veh.LEFT].GetAssemblyRadius())
        self.middle_wheels = {(q, s): v.GetTrackAssembly(s).GetTrackSuspension(MIDDLE_WHEEL[q]).GetRoadWheel().GetBody()
                              for s in self.sides for q in ("front", "rear")}
        self.bottom_local = chrono.ChVector3d(0.0, 0.0, pad_bottom_z(pad))
        self.inputs = veh.DriverInputs()          # what the vehicle receives when the shafts brake is in use
        self.clutches = {}
        if brake == "shafts":   # ChTrackBrakeShafts::Construct, from Python, on both sprockets
            chassis = v.GetChassisBody()
            for s in self.sides:
                shaft = chrono.ChShaft()
                shaft.SetInertia(BRAKE_SHAFT_INERTIA)
                self.system.Add(shaft)
                conn = chrono.ChShaftBodyRotation()
                conn.Initialize(shaft, chassis, chrono.ChVector3d(0, 1, 0))
                self.system.Add(conn)
                clutch = chrono.ChShaftsClutch()
                clutch.SetTorqueLimit(BRAKE_MAX_NM)
                clutch.Initialize(shaft, self.sprockets[s].GetAxle())
                clutch.SetModulation(0)
                self.system.Add(clutch)
                self.clutches[s] = (shaft, conn, clutch)
        elif brake != "simple":
            raise ValueError(f"unknown brake {brake!r}; one of {BRAKES}")
        self.side_of = {int(b.GetIdentifier()): s for s in self.sides for b in self._shoes[s]}

    # ---------------------------------------------------------------- collector-facing calls
    def GetVehicle(self):   # noqa: N802
        return self.standin

    def GetSystem(self):   # noqa: N802
        return self.system

    def GetChassis(self):   # noqa: N802
        return self.vehicle.GetChassis()

    def GetChassisBody(self):   # noqa: N802
        return self.vehicle.GetChassisBody()

    def Synchronize(self, t, inputs, terrain=None):   # noqa: N802 (terrain unused: the tracked vehicle has no tyres)
        if self.brake == "simple":
            return self.vehicle.Synchronize(t, inputs)
        b, s = float(inputs.m_braking), float(inputs.m_steering)
        per_side = {self.veh.LEFT: b + max(s, 0.0), self.veh.RIGHT: b + max(-s, 0.0)}   # ChTrackDrivelineBDS
        for side, (_, _, clutch) in self.clutches.items():
            clutch.SetTorqueLimit(BRAKE_MAX_NM * max(per_side[side], 0.0))
            clutch.SetModulation(1.0 if per_side[side] > 0.0 else 0.0)
        self.inputs.m_throttle = float(inputs.m_throttle)
        self.inputs.m_steering = 0.0     # BDS uses steering only to brake (done above); nothing else reads it
        self.inputs.m_braking = 0.0      # stock simple brakes get no input
        if hasattr(inputs, "m_clutch"):
            self.inputs.m_clutch = float(inputs.m_clutch)
        return self.vehicle.Synchronize(t, self.inputs)

    def Advance(self, dt):   # noqa: N802 (not called on soil: the CRM terrain advances the registered vehicle)
        return self.vehicle.Advance(dt)

    # ---------------------------------------------------------------- track queries
    def shoe_bodies(self):
        return [b for s in self.sides for b in self._shoes[s]]

    def _quarter_of(self, axle, side):
        return ("front" if int(axle) == 0 else "rear"), side

    def _split(self):
        """-> (chassis reference frame, {(front|rear, side): [shoe bodies]}), cached per simulated time."""
        t = float(self.system.GetChTime())
        cache = getattr(self, "_split_cache", None)
        if cache is not None and cache[0] == t:
            return cache[1], cache[2]
        ref = self.vehicle.GetChassisBody().GetFrameRefToAbs()
        out = {}
        for s in self.sides:
            for b in self._shoes[s]:
                q = "front" if ref.TransformPointParentToLocal(b.GetPos()).x > X_SPLIT_M else "rear"
                out.setdefault((q, s), []).append(b)
        self._split_cache = (t, ref, out)
        return ref, out

    def _bottom(self, body):
        return body.GetFrameRefToAbs().TransformPointLocalToParent(self.bottom_local)

    def quarter_lowest(self, axle, side):
        _, groups = self._split()
        pts = [self._bottom(b) for b in groups.get(self._quarter_of(axle, side), [])]
        if not pts:
            raise RuntimeError(f"track quarter {self._quarter_of(axle, side)} has no shoe")
        return min(pts, key=lambda p: p.z)

    def quarter_fields(self, terrain, wheel_specs):
        chrono = self.chrono
        up = chrono.ChVector3d(0, 0, 1)
        ref, groups = self._split()
        y_axis = ref.GetRot().GetAxisY()
        heading = y_axis.Cross(up).GetNormalized()
        lateral = up.Cross(heading)
        fields = {}
        for name, axle, side in wheel_specs:
            key = self._quarter_of(axle, side)
            bodies = groups.get(key, [])
            force = chrono.ChVector3d(0, 0, 0)
            for b in bodies:
                force = force + terrain.GetFsiBodyForce(b)
            omega = float(self.sprockets[side].GetGearBody().GetAngVelParent().Dot(y_axis))
            wheel_vx = float(self.middle_wheels[key].GetPosDt().Dot(heading))
            low = min((self._bottom(b) for b in bodies), key=lambda p: p.z)
            fields[f"{name}_force_wheel_fx_n"] = float(force.Dot(heading))
            fields[f"{name}_force_wheel_fy_n"] = float(force.Dot(lateral))
            fields[f"{name}_force_wheel_fz_n"] = float(force.Dot(up))
            fields[f"{name}_spindle_omega_radps"] = omega
            fields[f"{name}_wheel_vx_mps"] = wheel_vx
            fields[f"{name}_slip_ratio"] = (omega * self.sprocket_radius - wheel_vx) / max(abs(wheel_vx), 0.1)
            fields[f"{name}_spindle_z_m"] = float(low.z)
        return fields

    def info(self):
        v = self.vehicle
        return {"mass_kg": float(v.GetMass()), "shoes": {"left": len(self._shoes[self.veh.LEFT]),
                                                         "right": len(self._shoes[self.veh.RIGHT])},
                "sprocket_assembly_radius_m": self.sprocket_radius,
                "gear_ratios_forward": [round(r * ARMS[self.arm]["gear_scale"], 6) for r in STOCK_GEARS],
                "transmission_file": self.transmission_file,
                "transmission_file_sha256": sha(self.transmission_file) if self.transmission_file else None}


def create_m113(config, arm, brake=DEFAULT_BRAKE, pad=DEFAULT_PAD):
    import pychrono as chrono
    import pychrono.vehicle as veh
    if config["vehicle"].get("model") != "M113":
        raise ValueError("create_m113 needs vehicle.model == 'M113'")
    if config["vehicle"].get("contact_method", "SMC") != "SMC":
        raise ValueError("the M113 runs with SMC contact only (NSC jams the track, S5 3.3)")
    return M113Model(chrono, veh, config, arm, brake, pad)


def m113_vehicle_block(init, arm, brake, pad):
    return {"model": "M113", "arm": arm, "contact_method": "SMC", "chassis_fixed": False, "init": dict(init),
            "tire_model": "none (tracked)", "chassis_collision": "NONE", "track_shoe": "SINGLE_PIN",
            "driveline": "BDS", "brake": brake, "soil_pad": pad, "engine_model": "M113_EngineShafts",
            "transmission_model": ARMS[arm]["transmission"], "spawn_dz_m": FROZEN_SPAWN_DZ_M}


# ============================================================================================ run-time swaps
def make_create_vehicle(original, opts):
    def create_vehicle(config):
        if config["vehicle"].get("model") == "M113":
            model = create_m113(config, opts["arm"], opts["brake"], opts["pad"])
            opts["model"] = model
            return model
        return original(config)
    create_vehicle.ov_original = original
    return create_vehicle


def make_build_config(original, opts):
    def build_config(arena_dir, start_xyz, start_yaw, *a, **k):
        cfg = original(arena_dir, start_xyz, start_yaw, *a, **k)
        cfg["vehicle"] = m113_vehicle_block(cfg["vehicle"]["init"], opts["arm"], opts["brake"], opts["pad"])
        return cfg
    build_config.ov_original = original
    return build_config


class _TerrainProxy:
    def __init__(self, real, owner):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def RegisterVehicle(self, vehicle):   # noqa: N802
        return self._real.RegisterVehicle(getattr(vehicle, "_ov_real", vehicle))

    def AddRigidBody(self, body, geometry, check_embedded):   # noqa: N802
        self._owner.swapped.append(int(body.GetIdentifier()))
        self._owner.bodies.append(body)
        return self._real.AddRigidBody(body, self._owner.geometry, check_embedded)


class _VehProxy:
    def __init__(self, veh, geometry):
        self._veh, self.geometry, self.swapped, self.bodies, self.terrain = veh, geometry, [], [], None

    def __getattr__(self, name):
        return getattr(self._veh, name)

    def CRMTerrain(self, *a):   # noqa: N802
        self.terrain = _TerrainProxy(self._veh.CRMTerrain(*a), self)
        return self.terrain


_KEEP_ALIVE = []


def make_build_crm(original, opts, record):
    def build_crm(chrono, veh, fsi, system, vehicle, arena, meta, cfg):
        model = getattr(vehicle, "_ov_model", None)
        if model is None:
            return original(chrono, veh, fsi, system, vehicle, arena, meta, cfg)
        if abs(float(cfg["step_s"]) - STEP_S) > 1e-12:
            raise ValueError(f"the M113 needs the 0.5 ms soil step (configs/crm_m113.json); got step_s={cfg['step_s']}")
        proxy = _VehProxy(veh, pad_geometry(chrono, opts["pad"]))
        _KEEP_ALIVE.append(proxy)
        terrain = original(chrono, proxy, fsi, system, vehicle, arena, meta, cfg)
        expected = sorted(int(b.GetIdentifier()) for b in model.shoe_bodies())
        if sorted(proxy.swapped) != expected:
            raise RuntimeError(f"soil coupling expected the {len(expected)} shoe bodies once each, got {len(proxy.swapped)}")
        real = terrain._real if isinstance(terrain, _TerrainProxy) else terrain
        nbce = sorted({int(real.GetNumBCE(b)) for b in proxy.bodies})
        record.coupling = {"fsi_bodies": len(proxy.swapped), "bce_per_shoe": nbce,
                           "bce_total": int(sum(int(real.GetNumBCE(b)) for b in proxy.bodies))}
        if nbce == [0]:
            raise RuntimeError("the track pads received no soil marker points")
        return real
    build_crm.ov_original = original
    return build_crm


def make_tire_fields(original):
    def crm_tire_fields(chrono, veh, vehicle, terrain, wheel_specs, tire_radii):
        model = getattr(vehicle, "_ov_model", None)
        if model is None:
            return original(chrono, veh, vehicle, terrain, wheel_specs, tire_radii)
        return model.quarter_fields(terrain, wheel_specs)
    crm_tire_fields.ov_original = original
    return crm_tire_fields


def make_make_driver(original):
    def make_driver(chrono, veh, vehicle, route, tmap):
        return original(chrono, veh, getattr(vehicle, "_ov_real", vehicle), route, tmap)
    make_driver.ov_original = original
    return make_driver


def _belly_class():
    import ag_vehicle as agv   # numpy only at import

    class M113Belly(agv.BellyClearance):
        def __init__(self, tmap):   # noqa: D401 (same fields as ag_vehicle.BellyClearance)
            self.points = belly_points()
            self.tmap = tmap
            self.spec_sha = hashlib.sha256(np.ascontiguousarray(self.points).tobytes()).hexdigest()
            self.obj_sha_expected = self.obj_sha_runtime = None
            self.scale = (tmap.pixels - 1) / tmap.pixels
            self.frames, self.clear, self.argmin, self.vz, self.below = [], [], [], [], []

        def summary(self):
            s = super().summary()
            if s.get("frames"):
                s["definition"] = (f"clearance = lowest of {len(self.points)} points on the M113 hull underside "
                                   "(M113_Chassis.cpp belly boxes: bottom x -4.17..0.041 m, y +-0.85 m, z -0.143 m on "
                                   "a 0.10 m grid + the glacis bottom edge; ov_m113.belly_points) minus the undisturbed "
                                   "arena surface under it (BMP, Chrono node-on-edge convention); negative = hull below "
                                   "the original surface (the chassis is not coupled to the soil)")
                s["belly_points_sha256"] = self.spec_sha
                for k in ("chassis_col_obj_sha256", "chassis_col_obj_sha256_runtime"):
                    s.pop(k, None)
            return s

    return M113Belly


class M113Record:
    """The "vehicle" block written into the collector's JSON files for m113 / m113_g4."""

    def __init__(self, arm, brake, pad, wrapper=None):
        self.arm, self.brake, self.pad, self.wrapper = arm, brake, pad, wrapper
        self.belly, self.coupling, self.model, self.lib_sha = None, None, None, None
        self.g4_sha = sha(G4_TRANSMISSION) if (arm == "m113_g4" and G4_TRANSMISSION.is_file()) else None
        try:
            import pychrono.vehicle as veh
            lib = sorted(Path(veh.__file__).parent.glob("_vehicle*.so"))
            self.lib_sha = {p.name: sha(p) for p in lib} or None
            self.lib_dir = str(Path(veh.__file__).parent)
        except Exception:   # noqa: BLE001
            self.lib_dir = None

    def block(self, name=None):
        b = {"name": self.arm, "model": "pychrono.vehicle.M113_Vehicle_SinglePin (Chrono's M113 C++ model, as veh.M113 builds it)",
             "world": "soil", "switch": f"ov_m113.py via the dispatcher (--vehicle {self.arm})",
             "ov_m113_sha256": sha(__file__), "wrapper": self.wrapper,
             "wrapper_sha256": sha(self.wrapper) if self.wrapper and Path(self.wrapper).is_file() else None,
             "contact_method": "SMC", "track_shoe": "SINGLE_PIN (63 left + 64 right)",
             "driveline": "BDS (brake-differential steering: steering brakes the inner track), conical ratio 0.5",
             "brake": ({"shafts": "ChTrackBrakeShafts construction from Python: brake shaft + ChShaftsClutch per sprocket, "
                                  "10,000 N m, BDS input combination (stock M113_BrakeSimple gets zero input; it never "
                                  "sticks and SetBrakeType(SHAFTS) is ignored for single-pin shoes)",
                        "simple": "stock M113_BrakeSimple (ChLinkLockBrake, 10,000 N m; diagnostic)"}[self.brake]),
             "engine": "M113_EngineShafts (stock)", "transmission": ARMS[self.arm]["transmission"],
             "gear_scale": ARMS[self.arm]["gear_scale"], "g4_transmission_json_sha256": self.g4_sha,
             "soil_step_s": STEP_S, "soil_config_note": "configs/crm_m113.json = crm_main.json with step_s 0.0005",
             "soil_pad": {"name": self.pad, "boxes_center_dims_m": pad_boxes(self.pad),
                          "pad_bottom_z_shoe_frame_m": pad_bottom_z(self.pad)},
             "soil_coupling": self.coupling,
             "chassis_reference": "front sprocket axis (stock M113 frame); centre of mass 2.0 m behind it",
             "spawn_dz_m": FROZEN_SPAWN_DZ_M,
             "quarter_mapping": {"wheels": "tire_fl/fr/rl/rr = left front / right front / left rear / right rear track quarter",
                                 "split": f"shoe chassis-frame x > {X_SPLIT_M} m (middle road wheel) = front",
                                 "force": "sum of FSI force on the quarter's shoes, chassis heading frame",
                                 "spindle_omega": "that side's sprocket angular speed about the chassis y axis",
                                 "wheel_vx": "forward speed of road wheel 1 (front) / 3 (rear) hub",
                                 "slip_ratio": "(omega x sprocket assembly radius - wheel_vx) / max(|wheel_vx|, 0.1)",
                                 "spindle_z": "lowest pad-bottom point of the quarter; tyre radius 0"},
             "soil_breakthrough_rule": "lowest pad of a track quarter more than depth + margin (0.30 m) below the undisturbed surface for 5 frames",
             "follower": "frozen make_driver (gains 0.8 / 0.6, 0.05; 5 m look-ahead) on the real tracked vehicle",
             "pychrono_vehicle_lib_sha256": self.lib_sha, "pychrono_vehicle_lib_dir": self.lib_dir,
             "build": self.model.info() if self.model is not None else None}
        if name == "outcome.json" and self.belly is not None:
            b["belly"] = self.belly.summary()
        return b


def make_dump(original, record):
    def dump(path, value):
        name = Path(path).name
        if name == "outcome.json" and record.belly is not None:
            record.belly.save(Path(path).parent)
        if name in ANNOTATED and isinstance(value, dict):
            value = {**value, "vehicle": record.block(name)}
        return original(path, value)
    dump.ov_original = original
    return dump


def _strip_flags(argv):
    """-> (pad, brake, argv without the M113-only flags)."""
    argv, found = list(argv), {}
    for flag in M113_FLAGS:
        for i, a in enumerate(list(argv)):
            if a == flag:
                found[flag] = argv[i + 1]
                del argv[i:i + 2]
                break
            if a.startswith(flag + "="):
                found[flag] = a.split("=", 1)[1]
                del argv[i]
                break
    pad, brake = found.get("--m113-pad", DEFAULT_PAD), found.get("--m113-brake", DEFAULT_BRAKE)
    if pad not in PADS or brake not in BRAKES:
        raise ValueError(f"--m113-pad must be one of {PADS}, --m113-brake one of {BRAKES}")
    return pad, brake, argv


def install(ctx):
    """See the module docstring.  ctx.vehicle not in VEHICLES -> None (nothing patched)."""
    vehicle = getattr(ctx, "vehicle", None) if not isinstance(ctx, dict) else ctx.get("vehicle")
    if vehicle not in VEHICLES:
        return None
    get = (lambda k, d=None: ctx.get(k, d)) if isinstance(ctx, dict) else (lambda k, d=None: getattr(ctx, k, d))
    crm_collect = get("crm_collect")
    if crm_collect is None:
        import crm_collect   # noqa: F811
    base_module = get("base_module") or crm_collect
    if base_module is not crm_collect:
        raise NotImplementedError("the M113 runs only through the base collector crm_collect (not crm_collect_ext)")
    pad, brake, argv = _strip_flags(get("argv") or [])
    import sys
    source = get("source_root")
    if source:
        sys.path.insert(0, str(Path(source) / "src"))
        sys.path.insert(0, str(Path(source) / "scripts"))
    import nedm.hmmwv_data as hd
    import nedm.traverse.scene as scene
    import traverse_fdm_rgbd_diverse_chrono as frozen
    opts = {"arm": vehicle, "brake": brake, "pad": pad}
    record = M113Record(vehicle, brake, pad, wrapper=get("wrapper"))
    hd.create_hmmwv = make_create_vehicle(hd.create_hmmwv, opts)
    scene.create_hmmwv = make_create_vehicle(scene.create_hmmwv, opts)
    scene.build_config = make_build_config(scene.build_config, opts)
    crm_collect.build_crm = make_build_crm(crm_collect.build_crm, opts, record)
    crm_collect.crm_tire_fields = make_tire_fields(crm_collect.crm_tire_fields)
    frozen.make_driver = make_make_driver(frozen.make_driver)
    crm_collect.dump = make_dump(crm_collect.dump, record)
    Belly = _belly_class()
    state = {}

    def scene_hook(**k):
        state["model"] = k["hmmwv"]
        record.model = opts.get("model")
        record.belly = Belly(k["tmap"])

    def frame_hook(frame, **k):
        record.belly.on_frame(frame, state["model"].GetChassisBody())

    return {"scene_hook": scene_hook, "frame_hook": frame_hook, "argv": argv, "record": record}
