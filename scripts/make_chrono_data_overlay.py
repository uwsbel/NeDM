#!/usr/bin/env python
"""Build a Chrono data tree that satisfies the compiled HMMWV hull mesh name.

Chrono's compiled HMMWV model asks for `vehicle/hmmwv/hmmwv_chassis_col.obj`.
pychrono 9.0.0 ships `HMMWV_chassis_col.obj`. The names differ only in case, so
the load succeeds on a case-insensitive filesystem (stock macOS) and fails on
Linux, where it produces no chassis collision at all and therefore a silent
zero-contact result. The string lives in libChrono_vehicle, so it cannot be
fixed from this repository or from the JSON data.

This builds a symlink overlay: everything points back at the installed tree,
except the one directory that needs an extra alias. Nothing is copied and the
installed environment is not modified.

  "$NEDM_PY" scripts/make_chrono_data_overlay.py
  export NEDM_CHRONO_DATA_ROOT=$(pwd)/artifacts/chrono_data_overlay

Re-run it after changing environments; it is idempotent and cheap.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

WANTED = "hmmwv_chassis_col.obj"


def shadow(src_dir: Path, dst_dir: Path, expand: str | None) -> None:
    """Symlink every entry of src_dir into dst_dir, except `expand`, made real."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src_dir.iterdir()):
        target = dst_dir / entry.name
        if entry.name == expand:
            continue
        if target.is_symlink() or target.exists():
            continue
        target.symlink_to(entry)


def main() -> int:
    import pychrono as chrono

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="artifacts/chrono_data_overlay")
    parser.add_argument("--force", action="store_true", help="rebuild from scratch")
    args = parser.parse_args()

    src = Path(chrono.GetChronoDataPath()).resolve()
    if not src.is_dir():
        print(f"FAIL: installed Chrono data path {src} is not a directory")
        return 1

    out = Path(args.out).resolve()
    if out.exists() and args.force:
        shutil.rmtree(out)

    # Three levels deep, each shadowing its parent so only the leaf is a real
    # directory: data/ -> vehicle/ -> hmmwv/.
    shadow(src, out, expand="vehicle")
    shadow(src / "vehicle", out / "vehicle", expand="hmmwv")
    shadow(src / "vehicle" / "hmmwv", out / "vehicle" / "hmmwv", expand=None)

    hmmwv_out = out / "vehicle" / "hmmwv"
    alias = hmmwv_out / WANTED
    if alias.exists() or alias.is_symlink():
        print(f"ok: {alias} already present")
    else:
        candidates = [p for p in (src / "vehicle" / "hmmwv").iterdir()
                      if p.name.lower() == WANTED.lower()]
        if not candidates:
            print(f"FAIL: no case-variant of {WANTED} in {src / 'vehicle' / 'hmmwv'}")
            return 1
        alias.symlink_to(candidates[0])
        print(f"ok: {alias.name} -> {candidates[0].name}")

    print(f"\nexport NEDM_CHRONO_DATA_ROOT={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
