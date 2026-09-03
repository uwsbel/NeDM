# Chrono 9 vs 10, and the OptiX driver coupling

Everything here was found on 2026-09-02, moving this repo from the Chrono 10 it
was written against onto the conda-forge 9.0.0 both reachable boxes had, and then
onto the `nedm` environment the repo specifies. It cost most of a day.

The single most useful habit to take from it: **verify the thing you care about,
not a proxy for it.** Every entry below is a case where a check passed and the
thing it stood in for did not.

## CRM contact does not dissipate impact energy: any drop starts a limit cycle

**Cost:** the larger half of a session · **Found:** 2026-09-03 · **Applies to:** any rigid body landing on `CRMTerrain`

Measured on a single 0.025 m sphere, 3.75 kg, weight 36.79 N, soil depth 0.15 m.
No robot, no legs, no controller.

| drop | mean z | mean Fz | Fz peak-to-peak | oscillation |
|---|---|---|---|---|
| **0.000 m** | 0.1740 | 36.79 N = **1.00x weight** | 19.9 N | 21.8 Hz |
| 0.010 m | 0.2047 | ~1.00x | 32.6 N | 22.8 Hz |
| 0.050 m | 0.2340 | ~1.00x | 44.3 N | 10.7 Hz |
| 0.095 m | 0.2136 | 35.97 N = **0.98x weight** | **161.1 N** | 6.3 Hz |

**The mean force is correct in every case**, so the soil is not pushing too hard.
What fails is *dissipation*: force spikes to 160 N, 4.4x weight, and falls to
**-0.73 N**, so the body fully leaves contact each cycle and is still oscillating
two seconds later.

**No threshold; it is binary.** Placed exactly at rest it settles steady to
0.3 mm. Dropped **one centimetre**, it enters a limit cycle it never leaves.

**This resolves the "restitution above 1" paradox.** A bouncing body spends more
time near the top of its cycle, so mean *height* rises above static equilibrium
while mean *force* stays at weight. There is no energy surplus anywhere in the
force balance. Everything recorded earlier as "restitution 0.78" or "the soil
injects energy" was a limit cycle read as a rebound.

**Not timestep chatter:** 6.3 and 21.8 Hz against an exchange interval of 2000 Hz
and a CFD step of 10,000 Hz, FFT-confirmed.

**On a quadruped it becomes directional.** Per-foot FSI force, standing, no
policy: the impact spike totals **1056 N against a ~150 N robot**, rear-biased
(291 N per rear foot vs 237 front). By t=0.8 the rear feet read **exactly
0.0 N** while their height climbs 0.284 to 0.590 m and the front pair carries
everything. That is the forward pitch measured directly rather than inferred
from the base quaternion.

*Contributing, not causal:* the commanded stand pose is front/rear asymmetric
(`GENESIS_DEFAULTS` thigh 0.8 front, 1.0 rear). On rigid ground that same pose
settles all four feet within 2 mm and holds for 2 s, so the asymmetry is harmless
until the contact pathology amplifies it.

**CAUSE FOUND: `artificial_viscosity` is too low.** Monotonic dose-response on
the dropped sphere, only that value changed:

| artificial_viscosity | Fz peak-to-peak | oscillation |
|---|---|---|
| **0.5** (Viper demo value) | 161.1 N | 6.3 Hz |
| 1.0 | 111.6 N | 6.8 Hz |
| 2.0 | 69.7 N | 8.0 Hz |
| **5.0** | **1.0 N** | **none** |

At 5.0 the dropped body is dead steady, height varying 0.1 mm, mean force
36.79 N = 1.00x weight exactly. At rest it is zero to printed precision. The
limit cycle is gone, not reduced.

0.5 comes from `demo_ROBOT_Viper_CRM.py`, and **Viper is a wheeled rover rolling,
not a legged robot landing 15 kg on four small contact patches.** The demo's
value was never exercised against impact.

**`shifting_method` was refuted**, having been the obvious candidate: XSPH shaves
11% off the force swing and changes nothing structural, `PPST_XSPH` likewise, and
`DIFFUSION_XSPH` actively destabilises it to 11 cm of excursion at 88.7 Hz. Mean
force stays 0.99-1.02x weight throughout, which was the pre-declared refutation
condition.

**The usable window is narrow and bounded on BOTH sides.** Swept on the box, then
confirmed at full scale on the robot:

| av | Fz p2p (box) | full scale: 8 bodies, 725k particles |
|---|---|---|
| 0.5 | 161.1 N | robot flips, 178° |
| 1.0 | 111.6 N | — |
| **2.0** | **69.7 N** | **GATE PASS, 8 s upright, max tilt 13.9°** |
| 3.0 | 32.4 N | **CRASHES** — particles leave the domain, core dump |
| 5.0 | 1.0 N | **CRASHES** |

So 2.0 is not a tuned optimum, it is very nearly **the only value that works**:
below it the robot falls over, at 3.0 and above the simulation dies. And at 2.0
the box still shows a 70 N force swing on a 37 N body, so the limit cycle is
**suppressed by 57%, not removed**.

**Three cautions for Case Study IV.** The window is narrow. Inside it the
pathology is only suppressed. And the damping needed to stand a quadruped up is a
*numerical* term, not a soil property, so sinkage and drawbar numbers are not
comparable across values of it.

**A methodology limit worth carrying:** the single-sphere box isolates the physics
cleanly and does **not** predict domain-scale stability. `av = 5.0` is perfect on
one sphere over a 1x1x0.15 patch and hard-crashes with 8 coupled bodies over
8x4x0.15. Anything characterised on the box needs confirming at scale before it
becomes a default.

**What the CRM training environment did instead, and it is the better answer.**
`chrono_crmenv.py`, the environment used for the real policy finetune, keeps
`artificial_viscosity` at 0.5 and **softens the soil**: `Young_modulus` 1e6 to
**5e5**, `cohesion` 5e3 to **2e3**, both commented "Reduced" in the source. It
also uses a 0.2 m bed, just under the heave threshold. That is the physically
honest fix, since Young's modulus and cohesion are soil properties, and it drops
the elastic wave speed from 24.3 to 17.1 m/s. `--soil training` selects it.

**But the soil change alone is NOT sufficient**, and neither is the damping.
Standing, 8 s, depth 0.2:

| soil | av | result |
|---|---|---|
| eval | 0.5 | flips at 1.4 s, 178° |
| training | 0.5 | falls at 2.4 s, 103° |
| eval | 2.0 | PASS, but drifts to 13.7° with an 11 cm front-rear split |
| **training** | **2.0** | **PASS, 6.8° peak, 4 cm split, 0.7° at t=1.0** |

The pair is better than either, because they fix **different halves**. Soft soil
fixes the *impact*: the spike falls from 1168 N to 138 N, about one robot weight.
Viscosity fixes the *ringing*: on soft soil at av 0.5 the box's force swing halves
but its **vertical excursion nearly triples**, 0.024 m to 0.069 m, and it is the
movement rather than the force that topples a quadruped.

At the working combination all four feet carry load continuously, the sum sits at
weight (127-155 N against ~150 N), and tilt at t=1.0 is **0.7°, better than the
rigid control's 1.5°**.

**Note `playground_crm.py` and `chrono_crmenv.py` disagree**, and only the latter
was used for training. Reading the playground first cost hours: it is a
visualisation scratch file, not the configuration anything was run with.

**A separate, smaller residue remains**, now cleanly measurable because the
oscillation is gone: at 5.0 a body placed at rest settles at 0.1743 and a dropped
body settles at 0.2063, a **3.2 cm permanent offset**. It no longer bounces, it
just comes to rest higher. That may be physical, a body resting on soil it
disturbed on landing, but that is not established.

## A CRM bed deeper than ~0.22 m heaves upward and carries bodies on it

**Cost:** most of a session's CRM work · **Found:** 2026-09-03 · **Applies to:** any `CRMTerrain` deeper than ~0.2 m

Separate from the limit cycle above, and ruled out as its cause. A body placed
**at rest** (no impact) on a deep bed still rises.

| terrain depth | net rise of a body at rest |
|---|---|
| 0.15 m | **-0.001 m** (settles correctly) |
| 0.20 m | +0.014 m |
| 0.25 m | **+0.106 m** |
| 0.30 m | +0.119 m |
| 0.45 m | +0.118 m |

Sharp onset between 0.20 and 0.25 m, then **saturating**: 0.25, 0.30 and 0.45 all
give ~0.12 m rather than swelling proportionally. Over 4 s the body rises,
plateaus at ~1.5 s, and slowly relaxes, so this is equilibration to a new level
rather than unbounded growth.

**Ruled out by direct test:** SPH step size across a 10x range, 5e-4 to 5e-5,
which falsifies a CFL explanation despite the arithmetic supporting one;
particle spacing at 0.02 and 0.03; `use_variable_time_step`, which made it worse;
and the robot entirely, since this is one sphere.

**Fix:** keep CRM beds at or below ~0.15 m depth until this is understood. Note
that enlarging a terrain from 0.20 to 0.30 m depth, which looks like an
improvement, crosses the threshold.

**Both of these belong in an upstream report**, alongside WP0c's `ChDepthCamera`
`ray_scale` finding.

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
**Fix:** upgrade to R590 or newer. No rebuild, no source compile.
**CONFIRMED 2026-09-02**: `kyle-sbel` upgraded to **595.84** and Chrono 10
renders. `crm_sensor_smoke.py` reports `camera: attached`, OptiX logs
`Shader compile time: 8.53`, no ABI error. Note Ubuntu's `nvidia-driver-590` is
a transitional package that pulls the 595 stack, so you land on 595, not 590.
**pychrono 9.0.0 still renders on 595 too**, verified by frame content and not
just an exit code, so this was a gain rather than a trade.

Do not verify a driver upgrade by grepping the OptiX banner: the
`OptiX Version: [...] ABI Version: [...]` string that R580 printed is **gone
from the 595 library**. The library and symlink are both fine. Use a functional
test instead, `crm_sensor_smoke.py --sim-seconds 0.2`, since a string that no
longer exists cannot verify anything.
**Risk before doing it:** 9.0.0 *does* render on R580 and is currently the only
working renderer in the fleet. If R590 fixes 10 and breaks 9.0.0's older OptiX
expectations, the result is zero renderers instead of one. **Upgrade one box and
leave the other on R580** until the fix is confirmed.
**Evidence:** `~-~-~ OptiX Version: [ 9.0002.0.0.0.0 ] Branch: [ r582_12 ] ABI
Version: [ 110 ] CUDA Version: [ 13.0.0.0 ] ~-~-~`. The ABI the build *requests*
is still unmeasured: the env ships no OptiX SDK headers, so there is no
`OPTIX_ABI_VERSION` to read.

## The C++ demo existing does not mean the Python binding does

**Cost:** a render cycle and a wrong doc claim · **Found:** 2026-09-03 · **Applies to:** any Chrono feature reached from Python

**Expected:** `src/demos/sensor/demo_SEN_CRM_Rendering.cpp` shows CRM soil
rendered through Chrono::Sensor via `manager->AttachFsiSphSystem(sysSPH, opts)`,
so the same should be reachable from pychrono.
**Happened:** `AttachFsiSphSystem` does not exist on `ChSensorManager` in
pychrono 10.0.0, and no symbol matching Attach/Fsi/Sph appears anywhere in
`pychrono.sensor`. The mechanism is in the C++ library and absent from the SWIG
surface.
**Fix:** none from Python today. Either render CRM through a **static proxy body**
with a real visual shape, which works but cannot deform and so cannot show
sinkage, or get the binding added upstream. **This matters for Case Study IV**,
which wants an ego depth camera watching a CRM pile change shape: that is exactly
what a proxy cannot provide.
**Evidence:** `ChSensorManager` exposes only AddSensor, GetDeviceList, GetEngine,
GetMaxEngines, GetNumEngines, GetRayRecursions, GetSensorList, GetVerbose,
ReconstructScenes, SetDebug, SetDeviceList, SetMaxEngines, SetRayRecursions,
SetVerbose, Update, scene.

Note SPH markers are not visual shapes, so this is not something a caller can
work around by walking the scene: there is nothing there to make visible.

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

## The default backend can change under you between a tag and HEAD

**Cost:** caught before configure, one grep · **Found:** 2026-09-03 · **Applies to:** any Chrono::Sensor build off main

**Expected:** Chrono::Sensor means OptiX, so a source build needs the OptiX SDK
and a driver new enough to serve its ABI. We upgraded a driver to R595 and
installed the 9.0.0 SDK on two boxes on that basis.
**Happened:** on the pinned commit, `src/chrono_sensor/CMakeLists.txt` reads

```cmake
set(CH_USE_SENSOR_OPTIX     OFF CACHE BOOL "Enable LEGACY OptiX-dependent ...")
set(CH_USE_SENSOR_VULKAN_RT ON  CACHE BOOL "Enable Vulkan ray-tracing ...")
```

Upstream demoted OptiX to **legacy, off by default** and made Vulkan RT the
default somewhere between tag 10.0.0 and main. A default configure yields a
sensor module with no OptiX in it, and the whole driver-and-SDK effort may have
been aimed at a backend this commit does not build.
**Fix:** read the module's own `CMakeLists.txt` for backend switches and their
defaults **before** configuring, and never infer a backend from what the module
required at the last tag. Enable both backends when the headers allow it; they
coexist behind separate `#ifdef`s in `ChSensorManager.h`, and different machines
may only be able to run different ones.

The general form, and the third instance of it in this file: **a version pin
fixes the source, not the assumptions built on an earlier version of it.**
Version drift moved four class names 9→10, removed the FSI-SPH render feature
from tag 10.0.0 entirely, and has now swapped a rendering backend. Each was
found by reading the tree rather than by trusting a memory of it.

## A locator variable's default can reach outside the tree you pinned

**Cost:** none, caught by reading `FindOptiX.cmake` · **Found:** 2026-09-03 · **Applies to:** any `-D` path we leave unset

**Expected:** leaving `OptiX_INSTALL_DIR` unset means CMake searches system
locations, and a wrong result would announce itself as "not found".
**Happened:** `cmake/FindOptiX.cmake:32` defaults it to `"${CMAKE_SOURCE_DIR}/../"`,
the **parent of the source tree**. Both boxes hold multiple unrelated Chrono
forks as sibling directories, one box fourteen of them. An unset value could
silently resolve against a fork we had explicitly decided not to reuse, and the
build would succeed and look correct.
**Fix:** set every path locator explicitly, and prefer building outside any
directory that holds sibling checkouts so the default cannot reach them.
