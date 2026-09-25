#!/usr/bin/env python3
"""Soil wheel calibration for the Gator (arena_gator_20260925, E2).  Local, one GPU, run under flock /tmp/luffy_crm.lock.

A flat 16 x 16 m arena (constant-grey BMP) is built through the PRODUCTION soil code: crm_collect.merged_config(<cfg>)
+ crm_collect.build_crm (for the Gator wrapped by ag_vehicle.make_build_crm with the cylinders under test), the vehicle
through scene.build_config + create_hmmwv / ag_vehicle.create_vehicle exactly as the soil collectors call them
(RIGID_MESH tyres, no chassis collision, tyre step = CRM step, spawn ground + 0.75 m HMMWV / + 0.35 m Gator).  Then
0.8 s at full brake (the settle), then --throttle-s of full throttle, steering 0.

Sinkage per wheel = nominal tyre radius - (spindle z - flat ground z), the soil collectors' definition (negative = the
tyre bottom rides above the undisturbed surface).  Output: one JSON line per run (settled sinkage at 0.8 s, sinkage and
speed every 0.25 s, soil force / weight, marker counts, wall time).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))


def flat_arena(folder, size=16.0, px=64):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((px, px), 128, np.uint8), mode="L").save(folder / "flat.bmp")
    meta = {"bmp": "flat.bmp", "size_m": size, "pixels": px, "height_min_m": -0.5, "height_max_m": 0.5,
            "orientation": {"rot90": 0, "flipud": True}, "features": [], "note": "ag_soil_calibrate flat patch"}
    (folder / "arena_meta.json").write_text(json.dumps(meta, indent=1))
    return folder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vehicle", choices=("hmmwv", "gator"), required=True)
    p.add_argument("--radius-front", type=float)
    p.add_argument("--radius-rear", type=float)
    p.add_argument("--width-front", type=float)
    p.add_argument("--width-rear", type=float)
    p.add_argument("--cfg", default=str(ROOT / "artifacts/traverse/crm_f104_v1/configs/crm_main.json"))
    p.add_argument("--chrono-data", default="/home/harry/chrono/data")
    p.add_argument("--settle-s", type=float, default=0.8)
    p.add_argument("--throttle-s", type=float, default=2.0)
    p.add_argument("--x0", type=float, default=-4.0)
    p.add_argument("--tag", default="")
    p.add_argument("--out", required=True, help="append one JSON line here")
    a = p.parse_args()

    import pychrono as chrono
    import pychrono.fsi as fsi
    import pychrono.vehicle as veh
    import ag_vehicle as agv
    import crm_collect
    import nedm.hmmwv_data as hd
    import nedm.traverse.scene as scene
    from nedm.hmmwv_data import WHEEL_SPECS, configure_chrono_data_paths
    from nedm.traverse.terrain import TerrainMap

    cfg = crm_collect.merged_config(a.cfg)
    dt = float(cfg["step_s"])
    arena = flat_arena(Path(tempfile.mkdtemp(prefix="ag_calib_")) / "arena_flat16")
    tmap = TerrainMap.from_dir(arena)
    ground = float(tmap.height(0.0, 0.0))
    build_config, create, build_crm = scene.build_config, hd.create_hmmwv, crm_collect.build_crm
    wheel = None
    if a.vehicle == "gator":
        class O:  # noqa: E701
            pass
        o = O()
        for k in ("radius_front", "radius_rear", "width_front", "width_rear"):
            setattr(o, "gator_soil_" + k, getattr(a, k))
        wheel = agv.soil_wheel(o)
        build_config = agv.make_build_config(scene.build_config)
        create = agv.make_create_vehicle(hd.create_hmmwv)
        build_crm = agv.make_build_crm(crm_collect.build_crm, wheel)
    config = build_config(arena, (a.x0, 0.0, ground + agv.FROZEN_SPAWN_DZ_M), 0.0, step_size_s=dt, tire_step_size_s=dt)
    config["vehicle"]["tire_model"] = "RIGID_MESH"
    config["vehicle"]["chassis_collision"] = "NONE"
    config["chrono_data_root"] = a.chrono_data
    config["vehicle_data_root"] = a.chrono_data + "/vehicle"
    configure_chrono_data_paths(ROOT, config)
    wrapper = create(config)
    vehicle, system = wrapper.GetVehicle(), wrapper.GetSystem()
    t0 = time.time()
    terrain = build_crm(chrono, veh, fsi, system, vehicle, arena, tmap.meta, cfg)
    build_s = time.time() - t0
    radii = [float(vehicle.GetTire(ax, sd).GetRadius()) for _, ax, sd in WHEEL_SPECS]
    spindles = [vehicle.GetWheel(ax, sd).GetSpindle() for _, ax, sd in WHEEL_SPECS]
    try:
        nbce = [int(terrain.GetNumBCE(s)) for s in spindles]
    except Exception as exc:  # noqa: BLE001
        nbce = f"n/a ({type(exc).__name__})"
    weight = float(vehicle.GetMass()) * 9.81
    inp = veh.DriverInputs()
    t, rows, settled = 0.0, [], None
    n_settle, n_total = int(round(a.settle_s / dt)), int(round((a.settle_s + a.throttle_s) / dt))
    t1 = time.time()
    for k in range(n_total):
        inp.m_steering = 0.0
        inp.m_braking, inp.m_throttle = (1.0, 0.0) if k < n_settle else (0.0, 1.0)
        terrain.Synchronize(t)
        wrapper.Synchronize(t, inp, terrain)
        terrain.Advance(dt)
        t = (k + 1) * dt
        if (k + 1) % int(round(0.05 / dt)) == 0 or k + 1 == n_total:
            sink = [radii[i] - (float(vehicle.GetSpindlePos(ax, sd).z) - ground) for i, (_, ax, sd) in enumerate(WHEEL_SPECS)]
            fz = [float(terrain.GetFsiBodyForce(s).z) for s in spindles]
            ref = wrapper.GetChassisBody().GetFrameRefToAbs()
            row = {"t": round(t, 3), "x": float(ref.GetPos().x), "speed": float(vehicle.GetSpeed()),
                   "chassis_vz": float(ref.GetPosDt().z), "chassis_ref_height": float(ref.GetPos().z - ground),
                   "sink": [round(s, 4) for s in sink], "fz_over_weight": round(sum(fz) / weight, 3),
                   "fz": [round(f) for f in fz]}
            rows.append(row)
            if k + 1 == n_settle:
                settled = row
    wall = time.time() - t1
    sim = a.settle_s + a.throttle_s
    late = [r for r in rows if a.settle_s - 0.2 - 1e-9 <= r["t"] <= a.settle_s + 1e-9]   # last 0.2 s of the settle
    late_sink = np.mean([r["sink"] for r in late], 0)
    thr = [r for r in rows if r["t"] > a.settle_s + 1e-9]
    out = {"tag": a.tag, "vehicle": a.vehicle, "wheel": wheel, "cfg": cfg.get("name", a.cfg), "spacing": cfg["spacing_m"],
           "step_s": dt, "n_sph": int(terrain.GetNumSPHParticles()), "bce_per_wheel": nbce, "nominal_radius": radii,
           "weight_n": weight, "build_s": build_s, "wall_s": wall, "rtf": sim / wall, "settled": settled,
           "settled_mean_sink_front": float(np.mean(settled["sink"][:2])), "settled_mean_sink_rear": float(np.mean(settled["sink"][2:])),
           "late_settle_mean_sink_front": float(late_sink[:2].mean()), "late_settle_mean_sink_rear": float(late_sink[2:].mean()),
           "throttle_mean_sink_front": float(np.mean([r["sink"][:2] for r in thr])),
           "throttle_mean_sink_rear": float(np.mean([r["sink"][2:] for r in thr])),
           "speed_1s": float(min(thr, key=lambda r: abs(r["t"] - a.settle_s - 1.0))["speed"]),
           "speed_end": float(thr[-1]["speed"]), "x_end": float(thr[-1]["x"]),
           "end": rows[-1], "rows": rows}
    with open(a.out, "a") as f:
        f.write(json.dumps(out) + "\n")
    print(json.dumps({k: (round(out[k], 4) if isinstance(out[k], float) else out[k]) for k in (
        "tag", "vehicle", "wheel", "bce_per_wheel", "settled_mean_sink_front", "settled_mean_sink_rear",
        "late_settle_mean_sink_front", "late_settle_mean_sink_rear", "throttle_mean_sink_front", "throttle_mean_sink_rear",
        "speed_1s", "speed_end", "rtf")}), json.dumps(settled))


if __name__ == "__main__":
    main()
