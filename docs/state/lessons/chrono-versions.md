# Chrono 9 vs 10, and the OptiX driver coupling

Everything here was found on 2026-09-02, moving this repo from the Chrono 10 it
was written against onto the conda-forge 9.0.0 both reachable boxes had, and then
onto the `nedm` environment the repo specifies. It cost most of a day.

The single most useful habit to take from it: **verify the thing you care about,
not a proxy for it.** Every entry below is a case where a check passed and the
thing it stood in for did not.

## Do not conclude a subsystem works because it imports

**Cost:** a wrong doc commit, later reverted · **Found:** 2026-09-02 · **Applies to:** any optional Chrono module

**Expected:** `import pychrono.sensor` succeeding, `ChSensorManager` constructing,
and `AddDirectionalLight` being present meant rendering worked under 10.0.0.
**Happened:** all three passed, and the first actual render died in the OptiX
engine constructor. A blocker was marked resolved on the strength of the probes.
**Cause:** module import, class construction and attribute presence are all
resolved before any GPU backend is touched. `ChOptixEngine` is built lazily.
**Fix:** render one frame. For FSI, step one coupled timestep. A capability probe
is a necessary condition, never a sufficient one.
**Evidence:** `AddDirectionalLight: True` followed by
`OPTIX_ERROR_UNSUPPORTED_ABI_VERSION` at `ChOptixEngine.cpp:86`.

## Chrono::Sensor under 10.0.0 needs a newer driver than R580

**Cost:** the fleet has no renderer under 10 · **Found:** 2026-09-02 · **Applies to:** `envs/nedm` on both boxes

**Expected:** the `nedm` environment would render, since newton did all of Study
3's RGB-D collection under a conda env of that name.
**Happened:** `OPTIX_ERROR_UNSUPPORTED_ABI_VERSION` on both boxes.
`kyle-N7-B650E` swapped only the Chrono build on one machine, one driver, one
GPU: 10.0.0 fails, 9.0.0 renders. So it is not the GPU, not Blackwell versus
Ampere, and not machine-local.
**Cause:** Chrono's changelog for 10.0.0 states ray-tracing sensor models now
require *"OptiX 9.0 or 9.1 (and corresponding NVIDIA driver versions)"*. The
driver ships its own OptiX, and R580 identifies itself as **OptiX 9.0002, ABI
110** (`libnvoptix.so.580.126.09`). It meets the stated floor and still fails,
so the build almost certainly wants 9.1 specifically and an ABI above 110. OptiX
9.1 requires an **R590** driver.
**Fix:** upgrade to R590 or newer. No rebuild, no source compile. The prediction
is falsifiable: R590 ships a higher ABI, and if the build's requested ABI is at
or below it, rendering starts working untouched.
**Risk before doing it:** 9.0.0 *does* render on R580 and is currently the only
working renderer in the fleet. If R590 fixes 10 and breaks 9.0.0's older OptiX
expectations, the result is zero renderers instead of one. **Upgrade one box and
leave the other on R580** until the fix is confirmed.
**Evidence:** `~-~-~ OptiX Version: [ 9.0002.0.0.0.0 ] Branch: [ r582_12 ] ABI
Version: [ 110 ] CUDA Version: [ 13.0.0.0 ] ~-~-~`. The ABI the build *requests*
is still unmeasured: the env ships no OptiX SDK headers, so there is no
`OPTIX_ABI_VERSION` to read.

## Four names moved between 9 and 10

**Cost:** three separate blocked runs · **Found:** 2026-09-02 · **Applies to:** anything run outside `envs/nedm`

| | Chrono 9.0.0 | Chrono 10.0.0 |
|---|---|---|
| `VisualizationType_*` | `pychrono.vehicle` | `pychrono` core |
| `SetDataPath` / `SetVehicleDataPath` | `SetDataPath` | `SetVehicleDataPath` |
| `pychrono.fsi` (CRM) | absent | present |
| HMMWV hull mesh | `HMMWV_chassis_col.obj` | `hmmwv_chassis_col.obj` |
| `ChScene.AddDirectionalLight` | absent | present |
| `pychrono.parsers` | broken both ways | works |

**Fix:** [`src/nedm/chrono_compat.py`](../../../src/nedm/chrono_compat.py) resolves
the moved names against core then vehicle, so a call site is correct on either.
Do not hardcode one module.

The mesh row is nastier than it reads. The compiled `HMMWV_Chassis` model requests
the lowercase name, so under 9.0.0 the load fails on Linux and **succeeds on
macOS**, whose filesystem is case-insensitive by default. A chassis with no
collision geometry produces zero contacts silently, which is exactly the vacuous
gate result WP0c was trying to eliminate.

## A gate that cannot fail is worse than no gate

**Cost:** two near-miss 49-minute vacuous runs · **Found:** 2026-09-02 · **Applies to:** G0a and any contact-based gate

**Expected:** guarding `chassis_collision != NONE` on the presence of a chassis
collision mesh would stop a repeat of WP0c's vacuous zero-contact result.
**Happened:** the guard matched case-insensitively, saw `HMMWV_chassis_col.obj`,
and passed, while Chrono requested the lowercase name and failed. The check
written to prevent the failure reproduced it one layer down.
**Fix:** assert the exact path the consumer will open, not a case-folded
substring of it. See `require_chassis_hull_mesh` in `src/nedm/hmmwv_data.py`.
