#!/usr/bin/env python3
"""Soil (CRM) collector with a vehicle switch (arena_gator_20260925, E2): scripts/crm_collect.py or
scripts/crm_collect_ext.py + ``--vehicle``.  The file name contains ``crm_collect`` so crm_worker.py forwards
``--crm-config``.

  --vehicle hmmwv|gator            else NEDM_VEHICLE, else hmmwv
  --base crm_collect|crm_collect_ext
                                   which unmodified collector runs the episode (default crm_collect; the _ext base
                                   gives the branch / pid_held / pid_perturbed / policy modes and its native mode)
  --gator-soil-radius-front/-rear, --gator-soil-width-front/-rear
                                   override the calibrated soil wheel cylinders (calibration runs only)
Every other argument goes to the base collector unchanged.

hmmwv: nothing is patched; the base collector's main(argv) runs as when its file is executed directly.
gator: before the base collector runs, nedm.hmmwv_data.create_hmmwv (-> veh.Gator for model "Gator"),
nedm.traverse.scene.build_config (-> Gator vehicle block, spawn ground + 0.35 m), crm_collect.build_crm (-> one
cylinder per axle instead of the HMMWV tyre mesh; every soil/SPH statement is the original's) and the base module's
``dump`` (-> "vehicle" block in outcome.json, collection_request.json, f104_episode.json; vehicle_extra.npz next to
outcome.json) are replaced, and the collector's scene/frame hooks record the belly clearance.  The collector's own
tyre-mesh config key is left as it is (unused for the Gator) and recorded next to the cylinders.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))       # crm_collect.py does the same at import

import ag_vehicle as agv  # noqa: E402  (numpy only at import)

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


def install_gator(base_module, crm_collect, opts, rest):
    source = Path(_value(rest, "--source-root")).resolve()
    # the base collector's main() inserts these two in this order before it imports nedm; importing nedm here after the
    # same insertion resolves the same files (duplicate sys.path entries are harmless)
    sys.path.insert(0, str(source / "src"))
    sys.path.insert(0, str(source / "scripts"))
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


def main(argv=None):
    base, argv = _pop_base(sys.argv[1:] if argv is None else argv)
    vehicle, opts, rest = agv.parse_switch(argv)
    if opts.runtime_fingerprint:
        raise ValueError("--runtime-fingerprint is a rigid-collector option (the soil collectors have no fingerprint gate)")
    import crm_collect
    base_module = crm_collect if base == "crm_collect" else __import__("crm_collect_ext")
    if vehicle == "hmmwv":
        return base_module.main(rest)
    scene_hook, frame_hook = install_gator(base_module, crm_collect, opts, rest)
    return base_module.main(rest, scene_hook=scene_hook, frame_hook=frame_hook)


if __name__ == "__main__":
    main()
