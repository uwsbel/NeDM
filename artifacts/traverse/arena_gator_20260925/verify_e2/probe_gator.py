#!/usr/bin/env python3
"""VERIFY_E2 probe: run the REAL wrapper collector (ag_gen_collect_ext.py or ag_crm_collect.py) with its normal
arguments, but first wrap ag_vehicle.create_gator (and, for soil, ag_vehicle.make_build_crm) so that what the
collector actually builds is written to <out>/../<name>.probe.json:

  vehicle class, isinstance(veh.Gator), total mass, chassis mass, config init z vs the arena surface at the start,
  tyre model, chassis collision type, driveline template + driven axle indexes, brakes per axle and side,
  tyre radii, spindle positions after Initialize; for soil also the geometry actually handed to AddRigidBody per
  axle (cylinder radius / width / axis) and the number of BCE markers CRM generated per wheel.

The collector itself is not changed; the wrappers only log and return the original objects.
usage: probe_gator.py rigid|soil <collector args ...>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import ag_vehicle as agv  # noqa: E402

world, argv = sys.argv[1], sys.argv[2:]
out = Path(argv[argv.index("--out") + 1]).resolve()
probe_path = out.parent / (out.name + ".probe.json")
log = {"world": world, "argv": argv}


def dump():
    probe_path.write_text(json.dumps(log, indent=1, default=str) + "\n")


orig_create = agv.create_gator


def logged_create(config):
    import pychrono as chrono
    import pychrono.vehicle as veh
    from nedm.traverse.terrain import TerrainMap
    g = orig_create(config)
    try:
        inspect_vehicle(g, config, chrono, veh, TerrainMap)
    except Exception as exc:  # noqa: BLE001
        log["inspect_error"] = f"{type(exc).__name__}: {exc}"
        dump()
    return g


def inspect_vehicle(g, config, chrono, veh, TerrainMap):
    v = g.GetVehicle()
    init = config["vehicle"]["init"]
    tm = TerrainMap.from_dir(ARENA)
    log["create_gator_called"] = True
    log["wrapper_class"] = type(g).__name__
    log["isinstance_veh_Gator"] = isinstance(g, veh.Gator)
    log["isinstance_veh_HMMWV_Full"] = isinstance(g, veh.HMMWV_Full)
    log["vehicle_mass_kg"] = float(v.GetMass())
    log["chassis_mass_kg"] = float(g.GetChassis().GetMass())
    log["config_vehicle_block"] = config["vehicle"]
    log["init_z_m"] = float(init["z_m"])
    log["surface_at_start_m"] = float(tm.height(init["x_m"], init["y_m"]))
    log["init_z_minus_surface_m"] = float(init["z_m"]) - log["surface_at_start_m"]
    dl = v.GetDriveline()
    log["driveline_template"] = dl.GetTemplateName()
    log["driveline_name"] = dl.GetName()
    log["driven_axle_indexes"] = list(dl.GetDrivenAxleIndexes())
    brakes = {}
    for ax in range(v.GetNumberAxles()):
        for side, nm in ((veh.LEFT, "L"), (veh.RIGHT, "R")):
            b = v.GetBrake(ax, side)
            brakes[f"axle{ax}{nm}"] = None if b is None else b.GetTemplateName() + ":" + b.GetName()
    log["brakes"] = brakes
    tyres = {}
    for ax in range(v.GetNumberAxles()):
        for side, nm in ((veh.LEFT, "L"), (veh.RIGHT, "R")):
            t = v.GetTire(ax, side)
            tyres[f"axle{ax}{nm}"] = {"template": t.GetTemplateName(), "radius_m": float(t.GetRadius()),
                                      "width_m": float(t.GetWidth()),
                                      "spindle_z_m": float(v.GetSpindlePos(ax, side).z),
                                      "spindle_x_m": float(v.GetSpindlePos(ax, side).x)}
    log["tyres"] = tyres
    log["engine"] = v.GetEngine().GetTemplateName() + ":" + v.GetEngine().GetName()
    log["transmission"] = v.GetTransmission().GetTemplateName() + ":" + v.GetTransmission().GetName()
    ch = g.GetChassisBody()
    log["chassis_ref_z_after_init_m"] = float(ch.GetFrameRefToAbs().GetPos().z)
    for key, fn in (("chassis_collision_shapes", lambda: ch.GetCollisionModel().GetNumShapes()),
                    ("chassis_collision_enabled", lambda: bool(ch.IsCollisionEnabled()))):
        try:
            log[key] = fn()
        except Exception as exc:  # noqa: BLE001
            log[key] = f"n/a ({type(exc).__name__}: {exc})"
    dump()
    return g


agv.create_gator = logged_create
orig_make_build_crm = agv.make_build_crm


def logged_make_build_crm(original, wheel):
    inner = orig_make_build_crm(original, wheel)

    def build_crm(chrono, veh, fsi, system, vehicle, arena, meta, cfg):
        terrain = inner(chrono, veh, fsi, system, vehicle, arena, meta, cfg)
        try:
            inspect_soil(terrain, vehicle, veh, cfg)
        except Exception as exc:  # noqa: BLE001
            log["inspect_soil_error"] = f"{type(exc).__name__}: {exc}"
            dump()
        return terrain
    return build_crm


def inspect_soil(terrain, vehicle, veh, cfg):
        proxy = agv._KEEP_ALIVE[-1]
        geo = {}
        for ia, g in proxy.geometries.items():
            cyl = [{"radius_m": float(c.radius), "length_m": float(c.length),
                    "axis_world_of_rot_z": [float(c.rot.GetAxisZ().x), float(c.rot.GetAxisZ().y), float(c.rot.GetAxisZ().z)], "pos": [float(c.pos.x), float(c.pos.y), float(c.pos.z)]}
                   for c in g.coll_cylinders]
            geo[f"axle{ia}"] = {"cylinders": cyl, "meshes": len(g.coll_meshes)}
        log["soil_geometry_handed_to_AddRigidBody"] = geo
        log["swapped_axles"] = list(proxy.swapped)
        bce = {}
        for ax in range(vehicle.GetNumberAxles()):
            for side, nm in ((veh.LEFT, "L"), (veh.RIGHT, "R")):
                bce[f"axle{ax}{nm}"] = int(terrain.GetNumBCE(vehicle.GetWheel(ax, side).GetSpindle()))
        log["bce_markers_per_wheel"] = bce
        log["cfg_tire_mesh_key_unused"] = cfg.get("tire_mesh")
        log["spacing_m"] = cfg["spacing_m"]
        dump()


agv.make_build_crm = logged_make_build_crm
case = json.loads(Path(argv[argv.index("--case") + 1]).read_text())
ARENA = (Path(argv[argv.index("--source-root") + 1]).resolve() / case["arena"]).resolve()

if world == "rigid":
    import ag_gen_collect_ext
    ag_gen_collect_ext.main(argv)
else:
    import ag_crm_collect
    ag_crm_collect.main(argv)
log["collector_returned"] = True
dump()
