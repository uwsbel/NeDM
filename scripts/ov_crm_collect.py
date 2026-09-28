#!/usr/bin/env python3
"""Soil (CRM) dispatcher for the Polaris / M113 study (offroad_vehicles_20260927, module M1): scripts/crm_collect.py or
scripts/crm_collect_ext.py + ``--vehicle``.  A copy of scripts/ag_crm_collect.py's logic that calls ov_vehicle.  The file
name contains ``crm_collect`` so crm_worker.py forwards ``--crm-config``.

  --vehicle hmmwv|gator|polaris|polaris_pc|polaris_4wd|polaris_w08|m113|m113_g4
                                   absent = hmmwv; NEDM_VEHICLE is never read (set = refusal)
  --base crm_collect|crm_collect_ext
                                   which unmodified collector runs the episode (default crm_collect)
  --gator-soil-radius-front/-rear, --gator-soil-width-front/-rear, --gator-soil-mesh
                                   Gator-only overrides, exactly as ag_crm_collect.py
Every other argument goes to the base collector unchanged (for m113*: to ov_m113.install first, which may strip its
own flags).

hmmwv:   only the JSON writer is wrapped (a "vehicle" block); the base collector's main(argv) runs as when its file is
         executed directly, with the same vehicle, soil and arrays.
gator:   the swaps of ag_crm_collect.install_gator, built from ag_vehicle's own functions (same rows); the vehicle
         block is ag_vehicle's, naming this dispatcher as the wrapper.
polaris*: ov_vehicle's swaps (see scripts/ov_vehicle.py): create_hmmwv -> Polaris for model "Polaris", build_config ->
         Polaris block + spawn ground + 0.40 m, build_crm -> one cylinder per wheel, dump -> vehicle block +
         vehicle_extra.npz; belly clearance recorded every frame through the collector's scene/frame hooks.
m113*:   ``import ov_m113`` (module M2) and call ``ov_m113.install(ctx)`` with ctx = types.SimpleNamespace(
             vehicle, base_module, crm_collect, argv, wrapper, source_root, chrono_data, ov_vehicle)
         It must patch what it needs (as ag_vehicle does) and return either None, a (scene_hook, frame_hook) tuple, or
         a dict with optional keys "scene_hook", "frame_hook", "argv" (argv = the arguments for the base collector
         after removing M113-only flags).  If ov_m113.py is absent the dispatcher stops with a clear error.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))       # crm_collect.py does the same at import

import ov_vehicle as ovv  # noqa: E402  (numpy only at import)

BASES = ("crm_collect", "crm_collect_ext")


def _pop_base(argv):
    argv, base = list(argv), "crm_collect"
    for i, a in enumerate(argv):
        if a == "--base":
            base = argv[i + 1]
            del argv[i:i + 2]
            break
        if a.startswith("--base="):
            base = a.split("=", 1)[1]
            del argv[i]
            break
    if base not in BASES:
        raise ValueError(f"--base must be one of {BASES}")
    return base, argv


def _value(argv, flag):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _source_paths(rest):
    source = Path(_value(rest, "--source-root")).resolve()
    # the base collector's main() inserts these two in this order before it imports nedm; importing nedm here after the
    # same insertion resolves the same files (duplicate sys.path entries are harmless)
    sys.path.insert(0, str(source / "src"))
    sys.path.insert(0, str(source / "scripts"))
    return source


def install_hmmwv(base_module, crm_collect, rest):
    record = ovv.HmmwvRecord("soil", wrapper=__file__)
    base_module.dump = ovv.make_dump(base_module.dump, record)
    if base_module is not crm_collect:
        crm_collect.dump = ovv.make_dump(crm_collect.dump, record)


def install_gator(base_module, crm_collect, opts, rest):
    """ag_crm_collect.install_gator, statement for statement, with this file as the recorded wrapper."""
    import ag_vehicle as agv
    _source_paths(rest)
    import nedm.hmmwv_data as hd
    import nedm.traverse.scene as scene
    wheel = agv.soil_wheel(opts)
    record = agv.VehicleRecord("soil", opts, chrono_data=_value(rest, "--chrono-data"), soil_wheel_geom=wheel, wrapper=__file__)
    hd.create_hmmwv = agv.make_create_vehicle(hd.create_hmmwv)
    scene.create_hmmwv = agv.make_create_vehicle(scene.create_hmmwv)
    scene.build_config = agv.make_build_config(scene.build_config)
    crm_collect.build_crm = agv.make_build_crm(crm_collect.build_crm, wheel)
    base_module.dump = agv.make_dump(base_module.dump, record)
    if base_module is not crm_collect:
        crm_collect.dump = agv.make_dump(crm_collect.dump, record)
    state = {}

    def scene_hook(**k):
        state["vehicle"], state["tmap"] = k["hmmwv"], k["tmap"]
        record.belly = agv.BellyClearance(k["tmap"], chrono_data=_value(rest, "--chrono-data"))

    def frame_hook(frame, **k):
        record.belly.on_frame(frame, state["vehicle"].GetChassisBody())

    return scene_hook, frame_hook


def install_polaris(arm, base_module, crm_collect, rest):
    import ag_vehicle as agv
    _source_paths(rest)
    import nedm.hmmwv_data as hd
    import nedm.traverse.scene as scene
    record = ovv.PolarisRecord(arm, "soil", chrono_data=_value(rest, "--chrono-data"), wrapper=__file__)
    hd.create_hmmwv = ovv.make_create_vehicle(hd.create_hmmwv, record.state)
    scene.create_hmmwv = ovv.make_create_vehicle(scene.create_hmmwv, record.state)
    scene.build_config = ovv.make_build_config(scene.build_config, arm)
    crm_collect.build_crm = agv.make_build_crm(crm_collect.build_crm, record.soil_wheel)
    base_module.dump = ovv.make_dump(base_module.dump, record)
    if base_module is not crm_collect:
        crm_collect.dump = ovv.make_dump(crm_collect.dump, record)
    state = {}

    def scene_hook(**k):
        state["vehicle"], state["tmap"] = k["hmmwv"], k["tmap"]
        record.belly = ovv.make_belly(k["tmap"], chrono_data=_value(rest, "--chrono-data"))

    def frame_hook(frame, **k):
        record.belly.on_frame(frame, state["vehicle"].GetChassisBody())

    return scene_hook, frame_hook


def install_m113(vehicle, base_module, crm_collect, rest):
    try:
        import ov_m113
    except ImportError as exc:
        raise RuntimeError(f"--vehicle {vehicle} needs scripts/ov_m113.py (module M2) next to this dispatcher; it is "
                           f"missing or failed to import ({type(exc).__name__}: {exc})") from exc
    if not hasattr(ov_m113, "install"):
        raise RuntimeError("scripts/ov_m113.py has no install(ctx) function (interface: see ov_crm_collect.py docstring)")
    source = _source_paths(rest)
    ctx = types.SimpleNamespace(vehicle=vehicle, base_module=base_module, crm_collect=crm_collect, argv=list(rest),
                                wrapper=str(Path(__file__).resolve()), source_root=str(source),
                                chrono_data=_value(rest, "--chrono-data"), ov_vehicle=ovv)
    res = ov_m113.install(ctx)
    scene_hook = frame_hook = None
    argv = list(rest)
    if isinstance(res, tuple):
        scene_hook, frame_hook = res
    elif isinstance(res, dict):
        scene_hook, frame_hook = res.get("scene_hook"), res.get("frame_hook")
        argv = list(res.get("argv", argv))
    elif res is not None:
        raise RuntimeError(f"ov_m113.install returned {type(res).__name__}; expected None, a tuple or a dict")
    return scene_hook, frame_hook, argv


def main(argv=None):
    base, argv = _pop_base(sys.argv[1:] if argv is None else argv)
    vehicle, opts, rest = ovv.parse_switch(argv)
    if opts.runtime_fingerprint:
        raise ValueError("--runtime-fingerprint is a rigid-collector option (the soil collectors have no fingerprint gate)")
    import crm_collect
    base_module = crm_collect if base == "crm_collect" else __import__("crm_collect_ext")
    if vehicle == "hmmwv":
        install_hmmwv(base_module, crm_collect, rest)
        return base_module.main(rest)
    if vehicle == "gator":
        scene_hook, frame_hook = install_gator(base_module, crm_collect, opts, rest)
    elif vehicle in ovv.POLARIS_NAMES:
        scene_hook, frame_hook = install_polaris(vehicle, base_module, crm_collect, rest)
    elif vehicle in ovv.M113_NAMES:
        scene_hook, frame_hook, rest = install_m113(vehicle, base_module, crm_collect, rest)
    else:
        raise ValueError(f"unknown vehicle {vehicle!r}")
    return base_module.main(rest, scene_hook=scene_hook, frame_hook=frame_hook)


if __name__ == "__main__":
    main()
