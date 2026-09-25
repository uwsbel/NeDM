#!/usr/bin/env python3
"""Rigid collector with a vehicle switch (arena_gator_20260925, E2): scripts/gen_collect_ext.py + ``--vehicle``.

  --vehicle hmmwv|gator      else NEDM_VEHICLE, else hmmwv
  --runtime-fingerprint F    Gator: the runtime fingerprint to bind (overrides FDM_RUNTIME_FINGERPRINT; must list
                             data/vehicle/gator/* and the vehicle library; see ag_runtime_fingerprint.py)
Every other argument is gen_collect_ext.py's and is passed through unchanged (all modes, --local, ...).

hmmwv: nothing is patched; gen_collect_ext.main() runs exactly as when the file is executed directly (same sys.path
order, same modules, same outputs).
gator: after gen_collect.import_runner() has loaded the frozen loop and before gen_collect_ext builds its adapted
function, the vehicle factory and config builder of nedm.traverse.scene are replaced (ag_vehicle.make_create_vehicle /
make_build_config: veh.Gator, spawn ground + 0.35 m), the frozen module's ``dump`` and gen_collect_ext's ``dump`` add a
"vehicle" block to the JSON files, and the rich-telemetry observer is wrapped to record the belly clearance
(vehicle_extra.npz).  The frozen run_chrono source, its hook locations and every gate of gen_collect_ext are
unchanged.  branch_auto children are re-launched through this wrapper with the same vehicle arguments.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT / "src"), str(HERE)):      # the same insertion gen_collect_ext.py does, so its own loop is a no-op
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ag_vehicle as agv  # noqa: E402  (numpy only at import)


def _value(argv, flag):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def install_gator(gen_collect, gen_collect_ext, opts, rest):
    record = agv.VehicleRecord("rigid", opts, chrono_data=_value(rest, "--chrono-data"), wrapper=__file__)
    fp = opts.runtime_fingerprint or os.environ.get("FDM_RUNTIME_FINGERPRINT")
    if fp:
        os.environ["FDM_RUNTIME_FINGERPRINT"] = str(Path(fp).resolve())
        record.fingerprint = agv.check_gator_fingerprint(fp)
    elif "--local" not in rest and "--check-only" not in rest:
        raise ValueError("Gator rigid runs need --runtime-fingerprint (or FDM_RUNTIME_FINGERPRINT) listing data/vehicle/gator")
    state = {"frozen": None}
    original_import_runner = gen_collect.import_runner

    def import_runner(source):
        module = original_import_runner(source)
        import nedm.hmmwv_data as hd
        import nedm.traverse.scene as scene
        if not hasattr(scene.create_hmmwv, "ag_original"):
            scene.create_hmmwv = agv.make_create_vehicle(scene.create_hmmwv)
            hd.create_hmmwv = agv.make_create_vehicle(hd.create_hmmwv)
            scene.build_config = agv.make_build_config(scene.build_config)
        if not hasattr(module.dump, "ag_original"):
            module.dump = agv.make_dump(module.dump, record)   # before the adapter copies the module namespace
        state["frozen"] = module
        return module

    original_make_observer = gen_collect.make_observer

    def make_observer(out, case):
        inner = original_make_observer(out, case)
        from nedm.traverse.terrain import TerrainMap
        module = state["frozen"]
        tmap = TerrainMap.from_dir((module.ROOT / gen_collect.read(case)["arena"]).resolve())
        belly = agv.BellyClearance(tmap, chrono_data=_value(rest, "--chrono-data"))
        record.belly = belly            # the patched dump writes vehicle_extra.npz with outcome.json
        return GatorObserver(inner, belly)

    gen_collect.import_runner = import_runner
    gen_collect.make_observer = make_observer
    gen_collect_ext.dump = agv.make_dump(gen_collect_ext.dump, record)

    original_subprocess_cmd = gen_collect_ext.subprocess_cmd

    def subprocess_cmd(args, mode, out, horizon_s, extra):   # branch_auto children run through this wrapper as well
        cmd = original_subprocess_cmd(args, mode, out, horizon_s, extra)
        assert cmd[3] == str(Path(gen_collect_ext.__file__).resolve())
        cmd[3] = str(Path(__file__).resolve())
        cmd += ["--vehicle", "gator"]
        if opts.runtime_fingerprint:
            cmd += ["--runtime-fingerprint", str(Path(opts.runtime_fingerprint).resolve())]
        return cmd

    gen_collect_ext.subprocess_cmd = subprocess_cmd
    return record


class GatorObserver:
    """Forwards to the frozen rich-telemetry observer; adds the belly-clearance measurement per recorded frame."""

    def __init__(self, inner, belly):
        self.inner, self.belly = inner, belly

    def on_frame(self, scene, frame, *a, **k):
        self.belly.on_frame(frame, scene.hmmwv.GetChassisBody())
        return self.inner.on_frame(scene, frame, *a, **k)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def main(argv=None):
    vehicle, opts, rest = agv.parse_switch(sys.argv[1:] if argv is None else argv)
    if opts.runtime_fingerprint and vehicle != "gator":
        raise ValueError("--runtime-fingerprint is a Gator option; HMMWV runs use FDM_RUNTIME_FINGERPRINT as before")
    import gen_collect_ext
    sys.argv = [gen_collect_ext.__file__] + rest
    if vehicle == "gator":
        install_gator(gen_collect_ext.gen_collect, gen_collect_ext, opts, rest)
    gen_collect_ext.main()


if __name__ == "__main__":
    main()
