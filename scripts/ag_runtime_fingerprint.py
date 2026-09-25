#!/usr/bin/env python3
"""Gator runtime fingerprint (arena_gator_20260925, E2).

The rigid collectors bind their runtime through FDM_RUNTIME_FINGERPRINT = a JSON {"runtime_sha256": {path: sha256}}
(the HMMWV campaign file experiments/fdm_f104_50h_20260909/pilot_runtime_412394.json lists 108 files: the pychrono
modules and .so files, libChrono*.so, the grass texture and data/vehicle/hmmwv/*).  gen_collect(_ext).py require an
entry containing "_vehicle.so" and one containing "/vehicle/hmmwv/"; ag_gen_collect_ext.py --vehicle gator also
requires "/vehicle/gator/".

This script re-hashes every file of the base fingerprint (all must still match unless --allow-changed; changed files
are recorded with their new hash) and adds every file under <chrono-build>/data/vehicle/gator/.  The result is a
superset fingerprint: HMMWV entries kept (so gen_collect_ext's own HMMWV check still passes), Gator entries added.

usage (cluster login node, no Chrono import):
  python3 ag_runtime_fingerprint.py --base /work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/pilot_runtime_412394.json \
      --chrono-build /work1/dannegrut/harry/nrd/chrono-build --out <G3>/runtime/gator_runtime_fingerprint.json
"""
import argparse
import hashlib
import json
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--chrono-build", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-changed", action="store_true")
    a = p.parse_args()
    base = json.loads(Path(a.base).read_text())["runtime_sha256"]
    runtime, changed, missing = {}, {}, []
    for name, value in sorted(base.items()):
        f = Path(name)
        if not f.is_file():
            missing.append(name)
            continue
        now = sha(f)
        runtime[name] = now
        if now != value:
            changed[name] = {"base": value, "now": now}
    if (changed or missing) and not a.allow_changed:
        raise SystemExit(f"base fingerprint no longer matches the runtime: changed {list(changed)}, missing {missing}")
    gator_dir = Path(a.chrono_build) / "data" / "vehicle" / "gator"
    added = {str(f): sha(f) for f in sorted(gator_dir.rglob("*")) if f.is_file()}
    if not added:
        raise SystemExit(f"no Gator data under {gator_dir}")
    runtime.update(added)
    out = {"runtime_sha256": runtime, "schema": "ag_gator_runtime_fingerprint_v1",
           "base_fingerprint": str(Path(a.base).resolve()), "base_fingerprint_sha256": sha(a.base),
           "base_entries": len(base), "base_entries_rehashed_equal": len(base) - len(changed) - len(missing),
           "base_entries_changed": changed, "base_entries_missing": missing,
           "gator_entries_added": len(added), "chrono_build": str(Path(a.chrono_build).resolve()),
           "created_unix": time.time(), "generator": "scripts/ag_runtime_fingerprint.py", "generator_sha256": sha(__file__)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "runtime_sha256"}, indent=1))


if __name__ == "__main__":
    main()
