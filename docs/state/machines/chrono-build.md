# Building Chrono from source

**Written:** 2026-09-03 · **Status:** in progress, unblocked but not yet built

## Why build at all

The conda `pychrono` cannot render CRM terrain through Chrono::Sensor, so every
CRM clip so far shows a **static proxy floor that does not deform**. Case Study
IV wants an ego camera watching a CRM pile *change shape*, and a proxy cannot
show that. See [`../lessons/chrono-versions.md`](../lessons/chrono-versions.md).

## THE PINNED COMMIT

```
6982828952a920bb4e857625e74cedcf46d3573a
```

`projectchrono/chrono`, `main`, 2026-09-02, *"Eliminate the SWIG warnings in the
language bindings."*

**Both boxes build this exact SHA, checked out detached.** A source build that
differs between machines is worse than no source build: results silently stop
being comparable and nothing would reveal it.

### Why not the 10.0.0 tag

**The capability does not exist at 10.0.0.** Verified: at tag `10.0.0`
(`9faf13dd8`, 2026-04-07), `demo_SEN_CRM_Rendering.cpp`,
`ChSensorManager::AttachFsiSphSystem` and `ChFsiSphRender.h` are *all absent*.
Not present-but-unbound. The C++ feature postdates the release, landing in
`cb72352505f5a8f8a1a81adb5ea6a95c32fc417d`, 2026-05-09, *"Implement rendering
capability for FSI-SPH in Chrono::Sensor (#738)"*, 272 commits after the tag.

That also corrects an earlier reading: the missing binding was described as a
pychrono 10.0.0 *binding gap*. It is not one at that version, because the C++
never had the method. The gap is real only against `main`.

`main` HEAD was chosen over the feature commit because HEAD is 384 commits
further on and its subject is binding cleanup, which is exactly the layer being
patched. Starting behind that risks conflicting with work already done.

> **This puts the project on an UNRELEASED Chrono.** Results up to 2026-09-03
> came from a released 10.0.0 conda build; anything after comes from a snapshot
> of `main`. That is a change in footing and should be stated in any write-up,
> not left implicit in a SHA.

## Modules

`PYTHON`, `VEHICLE`, `FSI`, `SENSOR`, `PARSERS`. Not Irrlicht or VSG.

## Environment on kyle-sbel, measured

| | state |
|---|---|
| CUDA | full toolkit 12.6.85 at `/usr/local/cuda`, **`nvcc` not on `PATH`** |
| Eigen | 3.4.0, `/usr/include/eigen3` |
| CMake | 3.22.1; Chrono requires **3.18**, so this is sufficient |
| ROS 2 | Humble, supplying `libament_index_cpp.so` and `liburdfdom_model.so.3.0` |
| OptiX | **9.0.0**, installer in `~/sync/sbel/ops/`, sha256 `72a03c27…` |
| SWIG | absent |
| Disk | 1.3 T free |

The ROS 2 presence matters: those are the exact two libraries unresolved in the
conda `_parsers.so`, so a source build gets `ChParserURDF` working properly
rather than by luck. Chrono requires OptiX 9.0 or 9.1; the 7.7.0 in `~/Downloads`
is too old.

**Never build in `~/Documents/sbel/chrono_fork`.** That is Jason Zhou's HIL fork
at 9.0.1 on `feature/hil_new`, 3.8 GB with a populated `CMakeCache.txt` and a
last commit reading *"seg fault on demo run"*. Build in a fresh `chrono-src`.

## The FSI module switch is split

`CH_ENABLE_MODULE_FSI` is an umbrella and `CH_ENABLE_MODULE_FSI_SPH` is a
sub-switch. **Both are required**; enabling FSI alone yields no SPH.

## Same SHA does NOT mean same binary

The two boxes have materially different toolchains, measured:

| | kyle-sbel | kyle-N7-B650E |
|---|---|---|
| CUDA | 12.6.85 | **13.0.88** |
| gcc | 11.4.0 | **13.3.0** |
| CMake | 3.22.1 | 3.28.3 |
| SWIG | 4.5.0 (conda, installed today) | 4.2.0 (system) |
| **ROS 2** | **Humble** | **Jazzy** |
| OptiX SDK | 9.0.0 (installed today) | 9.0.0 (already present) |
| **GPU / compute cap** | **RTX 3090, 8.6** | **12.0 (Blackwell)** |
| `tinyxml2` | **conda** (Ubuntu 22.04 ships no CMake config) | **system 10.0.0** (24.04 ships one) |
| Vulkan RT | OFF (no `glslangValidator`) | OFF (same) |

Both configured successfully with OptiX ON and every module confirmed by name.
Sensor flags are **identical** across the fleet, which is the part of Kyle's
same-commit constraint that actually protects anything.

**Two divergences land in the same module, `parsers`:** ROS distro *and* now
`tinyxml2` provenance. Ubuntu 24.04 ships
`/usr/lib/x86_64-linux-gnu/cmake/tinyxml2/tinyxml2-config.cmake` (10.0.0) and
22.04 does not, so `kyle-sbel` links a **conda** tinyxml2 into an otherwise
system build while `kyle-N7-B650E` links the **system** one. Correct on both
boxes given what each has; not comparable. Parsers is now the module to suspect
first for any cross-box disagreement, on two independent grounds.

**The GPU gap is the widest divergence and the least discussed.** Compute 8.6
against 12.0 is two architecture generations. Any timing, throughput, or
realtime-factor number is a property of the box and not of the code, and must
carry the GPU when reported. Numerical results should agree; performance
results have no reason to.

### Do not let a source build overwrite the conda `pychrono`

Both boxes have a working conda `pychrono` 10.0.0, which today is the **only
working renderer** on either. A source build must not overwrite it: that trades
a known-good for an unproven one, with no way back short of a reinstall.

`kyle-N7-B650E` uses a separate `CMAKE_INSTALL_PREFIX`
(`~/chrono-build/install`) against the `nedm` interpreter. `kyle-sbel` uses a
cloned env (`nedm-src`). Different mechanics, same property, and both are fine.

**Selecting the source build must be an explicit act, never a default.** And
verify the selection by reading `pychrono.__file__` after import — not by
observing that `PYTHONPATH` is set. Both envs contain a conda `pychrono` as
well, so a path that *looks* right and an import that *resolves* right are
separate claims. Same rule as the configure summary block.

## Configure gotchas, all three hit on `kyle-sbel`

Pinning the SHA is necessary and **not sufficient**. The ROS divergence matters
most: `parsers` links against ROS, distros differ in `rmw` and `urdfdom`, and
`ChParserURDF` is precisely the component that loads the Go2. So that module is
**not comparable across boxes** and cannot be made so without matching ROS. It
also cannot simply be disabled, since the Go2 needs it.

**Record the toolchain alongside any result that depends on the source build**,
and validate that the Go2 URDF loads to the same body and motor counts on both
before treating cross-box results as comparable.

**A hypothesis worth testing once a build exists:** `kyle-N7-B650E` is still on
the R580 driver, whose OptiX is 9.0.02 at ABI 110, and it has the OptiX 9.0.0
SDK locally. A source build linked against that SDK should request an ABI the
driver can serve, which would make Chrono::Sensor work on R580 **without the
driver upgrade**. If that holds, the 595 upgrade becomes optional rather than
required, which is worth knowing before the second box is rebooted.

**`kyle-N7-B650E` has FOURTEEN existing Chrono trees** under `~/Documents`,
including `chrono_fork` at 22 G and `chrono-HIL` at 22 G, several with populated
`CMakeCache.txt`. Build outside `~/Documents` entirely so the pinned tree cannot
be confused with that set: a future agent grepping for "chrono" there finds
fifteen candidates and no way to tell which is which.

## The API moved between 10.0.0 and the pinned SHA

**Enumerated member-by-member across both builds, not discovered by breakage.**

**`CRMTerrain` — 2 removed, 9 added:**

| 10.0.0 | pinned SHA |
|---|---|
| `SetElasticSPH(...)` | `SetCrmSPH(SoilProperties)` |
| `SetActiveDomainDelay(d)` | `SetFreeFlowDuration(d)` — **same signature, pure rename** |

Added: `AddFeaMesh`, `IsFsiSolid`, `SetActiveDomainBody`, `SetActiveDomainMesh`,
`SetBcePattern1D/2D`, `SetCrmSPH`, `SetFreeFlowDuration`, `UseNodeDirections`.

**`pychrono.fsi` — 11 removed, 7 added.** `ElasticMaterialProperties` →
`SoilProperties`. The other ten removals are all VSG visualisation classes
(`ChSphVisualizationVSG`, the `Particle*ColorCallback` family,
`MarkerVisibilityCallback` and friends) — **expected, our build has VSG off**.
Added: `SoilProperties`, `ChFsiSphMarkerDeviceView`, `FsiMesh1D/2D`,
`FsiMeshForce`, `FsiMeshState`.

**`pychrono.sensor`:** `manager.scene` is now `ChOptixScene`; `ChScene` does not
exist. And `Background` is **no longer constructible** while still being the
parameter type of `SetBackground`/`GetBackground` — `GetBackground` returns a raw
`SwigPyObject` and SWIG warns *"memory leak of type 'Background *', no destructor
found"*. **This is a genuine upstream binding defect**, not a rename: a type
referenced by the public API but not properly wrapped. Two call sites hit it.

### Two results that make the migration verifiable rather than hopeful

**`SPHParameters`: zero changes.** All 28 fields identical in both builds. The
`sph_set()` guard therefore reports nothing to catch — which is the guard working,
not the guard being idle.

**`SoilProperties` is field-for-field identical to `ElasticMaterialProperties`.**
All 13 members match (`density`, `Young_modulus`, `Poisson_ratio`, `mu_I0`,
`mu_fric_s`, `mu_fric_2`, `average_diam`, `cohesion_coeff`, `mcc_M`,
`mcc_kappa`, `mcc_lambda`, `mcc_v_lambda`, `rheology_model`). **A pure rename**,
so the soil block migrates faithfully *by inspection* rather than by hope.

### Retroactive: the source build fixes this morning's first bug

`ChOptixScene` on the pinned SHA **has `AddDirectionalLight`**:

```
AddDirectionalLight(self, color: ChColor, elevation: float, azimuth: float) -> unsigned int
```

That is the method whose absence broke the first video render of the day and
forced the `AddPointLight` key/fill fallback still carried in
the RoboSimian gait script, now removed. Also present: `AddDiskLight`, `AddRectangleLight`,
`AddSpotLight`, `AddEnvironmentLight`, `AddSprite`, `Modify*` for each, and
`AddFsiSphSystem`/`RemoveFsiSphSystem`/`GetFsiSphSources` on the scene directly.

**A workaround we are still carrying is already obsolete on the build we now
have.** Worth checking the other workarounds against the new API before assuming
any of them are still needed.

### Decision: a compatibility shim, with a sunset condition

**Eight call sites**, three renames plus `Background`
(`crm_sensor_smoke.py:68,77,106`; `quadruped_go2_crm.py:251,255,346,477`;
the removed RoboSimian gait script).

`src/nedm/chrono_crm_compat.py` serves both environments from one codebase.
Chosen over a hard cut for one reason specific to today: **every number we have
came from the conda API**, so a shim is what makes the cross-API comparison
possible *at all*. Under a hard cut we could never check whether the migration
was faithful, because nothing would remain to compare against.

But `hasattr` dispatch is the silent-fallback pattern we have spent the day
cataloguing, so it is constrained:

1. **Detect once at import, not per call.** Resolve the API generation, bind the
   functions, and expose the result as a module constant.
2. **Raise when neither name is present.** Never no-op.
3. **Raise when BOTH are present** — that means a build we do not understand, and
   guessing would be the whole failure mode.
4. **Stamp the API generation on every output artifact.** A shim that makes
   numbers comparable while leaving no record of which side produced them
   defeats its own purpose.

**Sunset condition, so this does not become permanent architecture:** once the
Go2 has been re-run on the source build and the physics is confirmed to agree
across APIs, the shim is deleted and the conda path goes with it. **The shim is a
migration instrument, not a compatibility layer.**

## How we patch the pinned tree

**The tree stays a clean checkout; the patch is the artifact.** Never commit into
`chrono-src`, and never let a hand edit be the only record of a change.

Procedure:

1. Author the edit in place, iterate until it works.
2. Export it: `git -C chrono-src diff > NeDM/patches/NNNN-<name>.patch`, with the
   pinned SHA in the patch header.
3. **Verify it round-trips**: `git -C chrono-src checkout .`, re-apply with
   `git apply`, rebuild. If the exported patch does not reproduce the working
   state, the patch is wrong and the working tree was the only copy.
4. The build procedure is *checkout SHA → apply patch series → build*.

Why, given the same bytes end up on disk either way:

- **`git rev-parse HEAD` only means something if the tree is otherwise a clean
  checkout.** "Both boxes build the same SHA" is unverifiable against a tree
  carrying undocumented hand edits.
- **`kyle-N7-B650E` has to reproduce this.** A patch applies identically; a hand
  edit described in prose does not.
- Upstream submission wants the diff anyway, so producing it early costs nothing.
- On a future re-pin, a patch either applies or **conflicts loudly**. That is the
  behaviour we want from anything carrying our changes across a version bump.

## Building a standalone consumer against the source tree

*Written by `kyle-N7-B650E`, 2026-09-03, from four obstacles hit in sequence.*

The pinned build tree is **not installed**, so a program linking against it — the
CRM render gate, or anything else outside `chrono-src` — needs four things the
in-tree demos get for free. All four are silent or misleading in different ways.

### 1. Component names are not library names

`find_package(Chrono COMPONENTS ...)` takes `FSI_SPH`, `SENSOR`, `VEHICLE`,
`PARSERS`, `POSTPROCESS`. **There is no `ROBOT` component**, even though
`libChronoModels_robot.so` exists and `Chrono::ChronoModels_robot` is a valid link
target. Requesting `Robot` fails with *"Chrono was not configured with support for
the REQUIRED component ROBOT"*, which reads as a build-configuration problem and
is a naming one.

### 2. Imported targets from an uninstalled tree carry no include directories

`Chrono::Chrono_core` and friends resolve and link, but
`#include "chrono/physics/ChSystemNSC.h"` fails. Name them explicitly:

```cmake
target_include_directories(<tgt> PRIVATE
  /home/kyle/chrono-src/src      # headers
  /home/kyle/chrono-build        # generated ChConfig.h
  /usr/include/eigen3)
```

`ChronoConfig.cmake` supplies compile definitions and the OptiX include path
itself, so those need not be repeated.

### 3. Parsers drags in transitive finds the consumer must resolve

`ChronoTargets.cmake` exports `Chrono_parsers` with `urdfdom::urdfdom_model`,
`urdfdom::urdfdom_sensor` and `tinyxml2::tinyxml2` in its link interface and does
**not** `find_package` them. Because the targets file defines every module, this
fails **even for a consumer that does not use Parsers**. Each missing package
surfaces only after the previous is fixed — four configure cycles, not one:

```cmake
find_package(tinyxml2 REQUIRED CONFIG PATHS /usr/lib/x86_64-linux-gnu/cmake/tinyxml2 NO_DEFAULT_PATH)
find_package(urdfdom_headers REQUIRED CONFIG PATHS /opt/ros/jazzy/lib/x86_64-linux-gnu/urdfdom_headers/cmake NO_DEFAULT_PATH)
find_package(console_bridge QUIET CONFIG PATHS /opt/ros/jazzy/lib/x86_64-linux-gnu/console_bridge/cmake)
find_package(urdfdom REQUIRED CONFIG PATHS /opt/ros/jazzy/lib/x86_64-linux-gnu/urdfdom/cmake NO_DEFAULT_PATH)
# ...then find_package(Chrono ...)
```

**This is the ROS divergence demonstrated rather than predicted.** Paths are
written as literals *on purpose*: on `kyle-N7-B650E` they resolve to Jazzy's
urdfdom and the **system** tinyxml2 (24.04 ships
`/usr/lib/x86_64-linux-gnu/cmake/tinyxml2`; 22.04 does not), and on `kyle-sbel`
to Humble's urdfdom and a **conda** tinyxml2. **The recipe above is correct for
one box and wrong for the other**, and a templated version would hide exactly the
fact the section exists to record.

### 4. `CHRONO_DATA_DIR` is relative, and getting it wrong segfaults

It expands to `../data/`, so the working directory decides whether a run works.
The wrong directory does **not** produce a clean error — it prints

```
tiny_obj error message: Cannot open file [../data/robot/viper/col/viper_chassis.obj]
```

and then dies with **SIGSEGV**.

**This matters more than it looks.** On a box where OptiX or the driver is under
suspicion, a segfault out of a rendering program reads as a GPU problem. It is a
missing file, and it nearly was read as one. Run from a directory whose parent
holds a Chrono `data/` tree; a scratch directory with a symlink to
`chrono-src/data` works and avoids writing into the pinned tree. Prefer
`chrono-src/data`, which is the complete tree.

## MEASURED: the soil response is millimetres, and it goes UP

**Renderer-independent, from SPH particle z-coordinates.** 95th-percentile z
within 5 cm of a foot's XY, against an undisturbed control patch at the same
instant. Walking Go2, t > 2 s.

| foot | peak stance force | surface height at peak |
|---|---|---|
| FR | 194.5 N | **+0.00267 m** |
| FL | 191.0 N | **+0.00401 m** |
| RR | 183.8 N | **+0.00465 m** |
| RL | 178.7 N | **+0.00171 m** |

Mean relative height across the run is about **+1 mm**. The deepest depression
anywhere is RL at −0.018 m, and it occurs at **Fz = 12 N** — a foot lifting out
of a hole, not pressing into one.

**The sign is the physical result: under load the surface goes UP.** Every foot at
peak stance shows positive relative height. The soil **piles** around and under
the foot rather than bowling beneath it — seen first in a static proxy and
confirmed under 5× the load.

**So the renderer is not the limiting factor and neither is the load.** A few
millimetres of displacement on a bed discretised at 20 mm is a tenth of one
particle diameter. **There is essentially nothing there to render**, and no sprite
setting will change that. If Case Study III needs visible foot-soil interaction
the lever is **soil stiffness, particle size, or robot mass** — a physics-design
choice.

### The estimator was valid for a static foot and invalid for a walking one

The first walking measurement returned **+0.2798 m** of surface rise on a bed
whose top is at 0.20 m. **A footfall throws particles into the air**, and a
95th-percentile z then tracks ejecta rather than the bed.

Same estimator, same code, different validity: the static proxy generates no
ejecta and the walking robot does. **It was caught only because 0.48 m is
absurd** — a subtler contamination would have passed and been reported.

Fixed by excluding particles more than 3 cm above the control surface before
taking the percentile. The control patch needs no filter and its z95 is stable to
**sd 0.00000** across the run, which is itself evidence the estimator is sound
where there is no ejecta.

**Rule: an estimator validated in one regime is not validated in another.**
Re-derive its assumptions when the regime changes — here, from static loading to
impact.

## Soil softness: what moves sinkage, and the stability cliff

**Standing sweep, 2026-09-03, `--no-policy`, 6 s.** Sinkage is body descent.

**Cohesion is not a lever.** Held `young` at 5.0e5 and swept cohesion 2000 → 0:
the response moves by a few mm and the body sits, if anything, *slightly higher*
at zero cohesion. It never needs touching.

**Young's modulus is the only lever, and the cliff is sharp:**

| `young` | body sinkage | outcome |
|---|---|---|
| 5.0e5 *(training preset)* | 0 mm | stands, 10.2° |
| 1.0e5 | 1.2 mm | stands, 3.9° |
| **5.0e4** | **52.8 mm** | stands, 12.2° |
| 4.0e4 | 62.8 mm | stands, 20.5° — marginal |
| 3.0e4 | 425.6 mm | **falls**, face down |
| 1.0e4 | 842.0 mm | falls, sinks below the bed floor |

Nothing crashed at any setting — no domain-boundary loss, no core dump. **The
failure mode is the robot foundering, not the solver.** A 25% reduction from
4.0e4 takes it from standing to face-down, so nearly all the useful range sits
inside one factor of two.

### Body sinkage and surface displacement are different quantities

At 5.0e4 the **body** descends 52.8 mm — two to three particle diameters, plainly
visible. The **surface** under the feet never exceeds about 12 mm at any stable
setting.

**The robot descends into the bed and the soil closes around its legs**, rather
than a crater opening beneath each foot. Overhead frames show the calves
submerged and the feet no longer visible. *That* is the visible foot-soil
interaction — a robot standing **in** soil rather than on it, the soil
accommodating it continuously rather than leaving a static impression.

### Standing and walking move the surface in OPPOSITE directions

Settled by running **one** extraction over both trajectories — same code, same
convention, so no labelling artifact is possible:

| | mean surface height vs control |
|---|---|
| walking (policy on) | **+1.032 mm** — surface rises |
| standing (no policy) | **−6.219 mm** — surface settles |

**Sustained load consolidates; impact ejects.** This does not correct the
published walking result, which was always scoped to walking — it adds the static
case, which has the opposite sign and is the more intuitive of the two.

### The walking figures survive the autocorrelation correction

| | N | sd | lag-1 ρ | N_eff | SE_eff | significance |
|---|---|---|---|---|---|---|
| walking | 299 | 1.809 mm | +0.560 | 84.3 | 0.197 mm | **5.24 σ** |
| standing | 199 | 3.845 mm | +0.230 | 124.6 | 0.344 mm | 18.05 σ |

Autocorrelation is substantial for walking — ρ = 0.56 cuts the effective sample
count from 299 to 84 — and the effect survives comfortably. **Per-sample spread
is not the standard error of a mean**, and the raw spread (−18.6 to +7.2 mm)
bears on individual samples only, not on the published means.

---

## Where the rest of this went

The render investigation that produced these notes is settled, and its
conclusions live where a reader will actually need them:

- **How to reproduce a render, and every setting with the measurement that fixed
  it** → [`crm-rendering-handoff.md`](crm-rendering-handoff.md)
- **Why the SWIG binding needed no patch, and the API delta 10.0.0 → pinned SHA**
  → this file, *"The API moved…"* below
- **Contact-mode extraction, the foot-vs-kernel finding, soil response**
  → [`../decisions/quadruped-contact-mode.md`](../decisions/quadruped-contact-mode.md)

What is kept here is what a person **building** Chrono needs: the pinned commit,
the environment, the configure gotchas, the patch policy, and the standalone-consumer
recipe. The narrative of how each was discovered has been removed; the reasoning
that is still load-bearing has not.
