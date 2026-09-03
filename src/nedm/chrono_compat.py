"""Names that moved between the Chrono this repo was written against and 9.0.0.

The pipeline was developed on `newton`, which ran Chrono 10. Both machines Kyle
can currently reach run conda-forge pychrono 9.0.0, where some symbols live in
`pychrono.vehicle` rather than `pychrono` core. `VisualizationType_*` is the
case that surfaced: absent from core in 9.0.0, present in `vehicle`.

Resolving at import keeps a call site correct on either version. Hardcoding the
`vehicle` form would fix the reachable boxes and break `newton` if it returns,
which is the same trap `configure_chrono_data_paths` already sidesteps for
`SetDataPath` / `SetVehicleDataPath`.

This is deliberately narrow. A scan of every `chrono.*` and `veh.*` name used by
`hmmwv_data.py` and `traverse/scene.py` against 9.0.0 found exactly these three
missing and nothing else, so this is one moved concept, not a porting layer.
Note that resolution only proves the names exist; it does not prove signatures
match across versions.
"""

from __future__ import annotations

import pychrono as chrono
import pychrono.vehicle as veh

__all__ = [
    "VisualizationType_NONE",
    "VisualizationType_MESH",
    "VisualizationType_PRIMITIVES",
]


def _resolve(name: str):
    for module in (chrono, veh):
        value = getattr(module, name, None)
        # `is not None`, not truthiness: these are enum values and NONE is 0.
        if value is not None:
            return value
    raise AttributeError(
        f"{name} is in neither pychrono nor pychrono.vehicle. "
        "Chrono build is too old or too new for this shim; see "
        "docs/state/machines/ for the versions in use."
    )


VisualizationType_NONE = _resolve("VisualizationType_NONE")
VisualizationType_MESH = _resolve("VisualizationType_MESH")
VisualizationType_PRIMITIVES = _resolve("VisualizationType_PRIMITIVES")
