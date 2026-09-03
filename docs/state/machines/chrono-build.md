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

## OptiX is LEGACY on this commit. Vulkan RT is the default backend.

`src/chrono_sensor/CMakeLists.txt` on the pinned SHA:

```cmake
set(CH_USE_SENSOR_OPTIX     OFF CACHE BOOL "Enable LEGACY OptiX-dependent Chrono::Sensor features")
set(CH_USE_SENSOR_VULKAN_RT ON  CACHE BOOL "Enable Vulkan ray-tracing Chrono::Sensor features")
```

Between tag 10.0.0 and this commit, upstream made **Vulkan RT the default sensor
backend and demoted OptiX to legacy, off by default**. `ChSensorManager.h`
carries both `#ifdef CHRONO_HAS_OPTIX` and `#ifdef CHRONO_HAS_VULKAN_RT`
branches, so the two can coexist in one build.

**A default configure would produce a sensor module with no OptiX at all.**
Anyone assuming otherwise builds for an hour and gets the wrong backend.

This also means the entire OptiX chain — the R595 driver upgrade, the 9.0.0 SDK
installs — **may be unnecessary for this build**. Whether it is depends on which
backend the FSI-SPH rendering path is actually implemented under, which is the
one thing that must be checked before configuring.

**Build both backends where the headers allow it.** They coexist, the marginal
build cost is far below the cost of a wrong guess plus a rebuild, and the two
boxes have different drivers (R595 vs R580) so the backend that works may not be
the same one on each.

**`OptiX_INSTALL_DIR` defaults to the PARENT of the source tree**
(`cmake/FindOptiX.cmake:32`, `"${CMAKE_SOURCE_DIR}/../"`). On `kyle-sbel` that
is `/home/kyle/Documents/sbel/`, which contains `chrono_fork` and `chrono_hil`.
An unset value could silently find an OptiX belonging to a fork we deliberately
refused to reuse. Set it explicitly whatever the backend decision.

### RESOLVED: the feature is OptiX-only, so `CH_USE_SENSOR_OPTIX=ON` is mandatory

`ChSensorManager.cpp`:

```cpp
#ifdef CHRONO_FSI_SPH
CH_SENSOR_API int ChSensorManager::AttachFsiSphSystem(...) {
    int handle = -1;
    #ifdef CHRONO_HAS_OPTIX
    handle = scene->AddFsiSphSystem(sys, options);
    ReconstructScenes();
    #endif
    return handle;
}
```

**Built Vulkan-only, the body is empty.** `AttachFsiSphSystem` still compiles,
links, and is callable — it returns `-1` having done nothing. That is worse than
the method being absent, because a Python binding over it would **silently
no-op**: exactly the failure class this project keeps hitting.

`src/chrono_sensor/optix/ChOptixScene.h:211` carries `AddFsiSphSystem`,
`RemoveFsiSphSystem`, `ChFsiSphRenderSource` and `m_fsi_sph_sources`. Grepping
`src/chrono_sensor/vulkan/` for `FsiSph` returns **nothing**.

`ChFsiSphRender.h` sits at the top level of `chrono_sensor/` rather than under
`optix/`, so file layout alone does not reveal this. The implementation does.

The demo is gated the same way, three conditions deep
(`src/demos/sensor/CMakeLists.txt`):

```cmake
if(CH_USE_SENSOR_OPTIX)
  if(CH_ENABLE_MODULE_FSI_SPH AND CH_ENABLE_MODULE_VEHICLE)
    set(DEMOS ${DEMOS} demo_SEN_CRM_Rendering)
```

So `demo_SEN_CRM_Rendering` is **not built at all** without OptiX. The driver
upgrade and the SDK installs were not wasted.

Vulkan RT is still enabled where headers allow (system Vulkan 1.3.204 on
`kyle-sbel`), because it is free and `demo_SEN_vulkan_validation` does a 1:1
Vulkan-vs-OptiX camera comparison — a useful cross-check given `kyle-N7-B650E`
may never get OptiX working. **A Vulkan configure failure must not block the
build**; drop the flag, since the feature we need does not use it.

## The FSI module switch is split

`CH_ENABLE_MODULE_FSI` is an umbrella and `CH_ENABLE_MODULE_FSI_SPH` is a
sub-switch. **Both are required**; enabling FSI alone yields no SPH.

## Correction: the SWIG guard already exists

My earlier diagnosis, that `CHRONO_FSI_SPH` is invisible to the SWIG
preprocessor and a CMake change is needed to define it, **was wrong**. It is
already wired, in `src/chrono_swig/chrono_python/CMakeLists.txt:121-124`:

```cmake
if(CH_ENABLE_MODULE_FSI_SPH)
  set(CMAKE_SWIG_FLAGS "${CMAKE_SWIG_FLAGS};-DCHRONO_FSI_SPH")
  set(CMAKE_SWIG_FLAGS "${CMAKE_SWIG_FLAGS};-DCHRONO_CRM")
endif()
```

The vehicle bindings already depend on it: `interface/vehicle/ChTerrain.i` and
`ChModuleVehicle.i` both use `#ifdef CHRONO_FSI_SPH`, and that is how
`CRMTerrain` reaches Python today.

**Superseded: there was no gap at all.** See the resolution below —
`ChModuleSensor.i` `%include`s the header unguarded and both macros already
reach SWIG, so the methods generate without any change. The reasoning in this
section was a plausible mechanism for an absence whose real cause was the conda
build's version. There is a working pattern in the same repo to
copy, so the patch makes the sensor interface *consistent with* the vehicle one
rather than inventing a mechanism — smaller, and far more likely to be accepted
upstream.

It also explains the conda build: pychrono 10.0.0 predates the sensor FSI
feature entirely, so no binding could have existed regardless of flags.

### Configure gotchas

**1. A working CUDA compiler with no architecture list counts as NO toolchain.**
The trap of this build. CMake reported success at every step:

```
The CUDA compiler identification is NVIDIA 12.6.85
Detecting CUDA compiler ABI info - done
Found CUDAToolkit: /usr/local/cuda/include (found version 12.6.85)
```

and then:

```
CUDA archs (filtered):
CUDA architectures not found. Set CHRONO_CUDA_ARCHITECTURES
Building for NVIDIA, but no usable NVIDIA GPU toolchain was found.
  All GPU-dependent Chrono modules will be disabled.
GPU toolchains available: CUDA=FALSE HIP=FALSE
```

That disabled **FSI::SPH, Vehicle SCM GPU, and the Sensor OptiX renderer** —
precisely the three things this build exists for — and reported two of them as
*warnings*. Configure exits 0. Reading "configure succeeded" and starting the
build costs an hour and yields a Sensor module with no OptiX and no FSI.

Fix: `-DCHRONO_CUDA_ARCHITECTURES=86`, taken from the hardware rather than
assumed — `nvidia-smi --query-gpu=compute_cap --format=csv` reports 8.6 for the
RTX 3090. **Query the GPU on each box**; the value is not portable.

**2. `CUDAToolkit_ROOT` is not sufficient.** It satisfies Thrust, but CMake's
CUDA *language* check needs the compiler on `PATH` or named explicitly. `nvcc`
is not on `PATH` here. Fix: `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.

**3. `tinyxml2` has no CMake config on Ubuntu.** `libtinyxml2-dev` ships only
`pkgconfig/tinyxml2.pc`. Configs exist at `/opt/ros/humble/share/tinyxml2_vendor/`
(wrong package name) and in the conda env. Fix: `-Dtinyxml2_DIR=` pointed at the
conda one **surgically** — adding the conda prefix to `CMAKE_PREFIX_PATH` would
let CMake find dozens of conda libraries and mix ABIs against a system build.
*Known risk:* one conda library links into an otherwise-system build. **Suspect
this first if anything odd surfaces at link time.**

**4. Vulkan RT needs `glslangValidator`, not just headers.** Present headers
(1.3.204), absent shader compiler. Dropped, per the rule that Vulkan must never
block: the feature is OptiX-only.

### Confirmed enabled by reading the summary block, not by inferring from exit 0

```
CUDA archs (filtered):   86
GPU toolchains available: CUDA=TRUE HIP=FALSE
Chrono::FSI::SPH GPU backend: CUDA
OptiX include directory: /home/kyle/opt/optix/include
Building Chrono::Sensor with OptiX support
Add python CORE / FEA / POSTPROCESS / FSI / VEHICLE / SENSOR / ROBOT / PARSERS
```

The OptiX path is **our 9.0.0**, not the 7.7.0 in `~/Downloads`. And
`demo_SEN_CRM_Rendering.dir/` was generated, so the three-deep gate was met.

## The shipped demo writes nothing to disk

`demo_SEN_CRM_Rendering.cpp` on the pinned SHA:

```cpp
line  92:  bool snapshots = false;                              // DEFAULT OFF
line 314:  cam->PushFilter(ChFilterVisualize(1280, 720, ...));  // ALWAYS
line 316:  if (snapshots) cam->PushFilter(ChFilterSave(...));   // CONDITIONAL
line 295:  manager->AttachFsiSphSystem(sysSPH, fsi_render_options);  // RETURN DISCARDED
```

As shipped it opens a window and saves nothing. **On a headless box the
validation gate would produce no evidence at all**, and `ChFilterVisualize` may
fail or no-op with no display. It does attach the FSI system properly (sprite
meshes, jitter 0.005, `render_particle_spacing` 0.01), so the mechanism is
exercised — we just cannot see the result.

**Decision: copy it into NeDM as our own validation program**, rather than
patching the pinned tree. A gate that lives in our repo is versioned with the
results it produces, survives a re-pin, and can assert things the upstream demo
does not. Our copy must:

- push `ChFilterSave` and **no** `ChFilterVisualize` (headless);
- **capture the return of `AttachFsiSphSystem` and assert it is not `-1`** —
  that integer is the only in-band signal separating a working attach from the
  silent no-op described above, and the shipped demo discards it;
- support running with the attach call removed, to produce the negative control.

**Pass criterion, fixed before running:** a saved `img_SEN` frame with CRM
particles visibly present over the terrain region, **and** a second frame from an
otherwise identical run without the attach, lacking them. The *difference* is the
gate. A frame that merely renders proves nothing, since the failure mode under
test is a call that returns `-1` and does nothing.

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

## RESOLVED: there is no SWIG binding to add. It already works.

**Verified against the built artifact, 2026-09-03:**

```python
>>> mgr.AttachFsiSphSystem(terrain.GetFluidSystemSPH())
0
```

`ChSensorManager` exposes `AttachFsiSphSystem`, `DetachFsiSphSystem` and
`ClearFsiSphSystems`, with the default argument translated:

```
AttachFsiSphSystem(self, sys: shared_ptr<chrono::fsi::sph::ChFsiFluidSystemSPH>,
                   options: ChFsiSphRenderOptions const& = ChFsiSphRenderOptions()) -> int
```

**Both macros already reach SWIG**, conditional on the flags we set anyway:

```cmake
chrono_python/CMakeLists.txt:122   -DCHRONO_FSI_SPH    if CH_ENABLE_MODULE_FSI_SPH
chrono_python/CMakeLists.txt:136   -DCHRONO_HAS_OPTIX  if CH_USE_SENSOR_OPTIX
```

`ChModuleSensor.i` `%include`s `ChSensorManager.h` unguarded, so with both
macros defined SWIG parses the `#ifdef` block and generates all three methods.
**Building from a SHA that postdates the feature *is* the fix.** Nothing to
patch, nothing to send upstream.

### How this diagnosis was wrong twice, and it was the same fact both times

The conda `pychrono` 10.0.0 lacks the method. We had **already established** the
C++ feature postdates the 10.0.0 tag by 272 commits. So the absence was always
about the version — and it was then carried forward as a *separate* fact, "the
bindings are missing it", and a three-part patch designed against it. **It was
one fact wearing different clothes.**

The compounding step was reading `ChModuleSensor.i` and **inferring a cause**
instead of building and looking. The decisive check was one line against the
built artifact, and the artifact existed for an hour before anyone ran it.

**Rule: when a symbol is missing, check the artifact you have, not the source
you think produced it.** Reading source to explain an absence produces a
plausible mechanism whether or not it is the real one — and a plausible
mechanism is exactly what stops you running the one-line check.

**The patch series is empty**, and that is a finding rather than an absence. The
build procedure (checkout SHA → apply patches → build) stands as policy with
zero patches in it. `chrono-src` remains a clean checkout at the pinned SHA.

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
`quadruped_wp0_gait.py`. Also present: `AddDiskLight`, `AddRectangleLight`,
`AddSpotLight`, `AddEnvironmentLight`, `AddSprite`, `Modify*` for each, and
`AddFsiSphSystem`/`RemoveFsiSphSystem`/`GetFsiSphSources` on the scene directly.

**A workaround we are still carrying is already obsolete on the build we now
have.** Worth checking the other workarounds against the new API before assuming
any of them are still needed.

### Decision: a compatibility shim, with a sunset condition

**Eight call sites**, three renames plus `Background`
(`crm_sensor_smoke.py:68,77,106`; `quadruped_go2_crm.py:251,255,346,477`;
`quadruped_wp0_gait.py:207`).

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

## RESULT: CRM renders through Chrono::Sensor. The blocker is cleared.

**2026-09-03, `kyle-sbel`, pinned SHA, source build.** 847,714 SPH particles.

```
attached: handle = 0   frames_written=401 sim_frames=801 final_time=2.0025  exit 0
control:  handle = -1  frames_written=401 sim_frames=801 final_time=2.0025  exit 0
```

Dominant colour, frame 200: **attached `[0,0,0]` at 49.4%**, **control
`[216,230,243]` at 73.4%**. A dense granular field covers the terrain region in
the attached frame and is absent in the control. Rover geometry is
pixel-identical in both.

Five independent lines, none of which depends on the numeric bounds below:
handle ≥ 0 rules out the silent no-op; both arms writing 401 frames rules out
the void condition; matched step count and final time rule out drift; identical
rover geometry rules out a scene difference; and the difference is confined to
where terrain is.

### The pre-registered criterion REJECTED this true positive

Reported as a criterion failure rather than silently widened, which is the right
call and the reason this record is trustworthy.

```
changed pixels 66.10%          bound was 5-60%   FAILS
41.2% of changed above mid-height   bound was 0%      FAILS
```

**The criterion was wrong, and its error is the same class we catalogued all
day, one level up: a check whose passing condition encoded an unverified
assumption.** The bounds assumed a wide third-person view where terrain occupies
a lower band. The actual camera sits at `(0.5, 1, 1)` rotated 60° looking down,
essentially on the rover wheel, so terrain fills most of the frame and extends
well above the midline in perspective. **There is no sky above the horizon in
most of this frame** — the region being guarded as sky is largely terrain. The
camera pose was in the file that had already been read.

It **failed safe**: it rejected a true positive rather than accepting a false
one. That is the correct direction to fail, and it still failed.

**This run is validated by the five lines above, NOT by the numeric bounds.**
The revised criterion below was written after seeing this data and therefore
cannot validate it — it applies to future runs only.

### Revised criterion, for future runs only

Keep: both arms must render (void check), ≥5% changed pixels, handle ≥ 0 in the
attached arm and the call demonstrably absent in the control. Drop: the 60%
upper bound and the horizon test. Replace with: the changed region is contiguous
and coincides with the terrain's projected extent **computed from the camera
pose**, not assumed from a mental picture of the framing.

### Three defects found on the way

**1. `render_frame` is always 0** in the gate's own trace — the counter is
incremented inside a `#ifdef CHRONO_VSG` block that is compiled out. The
comparability check happened to use `t` and `sim_frame`, which are correct.
*"I would not have caught it if the two arms had disagreed — I would have been
comparing on a field that is constant."* An instrument that reads zero always
agrees with itself.

**2. The gate segfaults if run outside `chrono-build/bin`**, because Chrono data
paths resolve relatively. Needs an explicit `SetChronoDataPath`.

**3. The conda contamination predicted at configure time happened exactly as
described.** The imported `tinyxml2` target drags in the whole conda env include
directory, which contains a *complete competing Chrono* — the gate compiled
against `envs/nedm-src/include/chrono/` and died on a missing HACD header.

**The built library is NOT contaminated**, verified rather than assumed: source
`-I` paths precede the conda `-isystem` path in every module, and core, sensor
and fsisph carry no conda include at all. Only `chrono_parsers` sees it, source
path first. Fixed for the gate with `target_include_directories(... BEFORE ...)`.

The risk was written down at configure time, sized correctly, and landed exactly
where predicted. **That is the argument for recording risks you decide to
accept.**

### Two things noted, not chased

**Particles render black `[0,0,0]`, not shaded regolith.** May be correct for the
shipped sprite meshes and default material, or may mean lighting is not applied
to the sprite path. Irrelevant to presence-versus-absence, but **black
silhouettes are probably not the eventual deliverable** if the goal is a camera
watching a CRM pile change shape.

**Do not quote the realtime factors from this run.** `rtf_cfd` median 236.1
against mean 964.1 is wildly skewed, and both are ~2 orders of magnitude from
the 2.8-5.7 measured earlier the same day on the Go2 at up to 1.29 M particles.
Startup transients, different accessor semantics in this build, or the 200 fps
save filter perturbing timing. **Until the discrepancy is explained these
measure nothing**, and publishing them beside the earlier numbers would put a
contradiction in the record.

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

## A fourth defect: a confidently wrong absolute path

The `SetChronoDataPath` fix was **wrong on the first attempt, and failed
identically to the bug it was fixing.** `CHRONO_DATA_DIR` was set to
`chrono-build/bin/data/` by analogy with a vehicle define already in the build
flags. That directory does not exist — the data is at `chrono-build/data/`, and
the demo worked from `bin` only because its relative `../data/` happened to
resolve there.

**So the absolute path was confidently wrong in a way the relative one was not**,
and it segfaulted exactly as before. A fix that reproduces the original symptom
is indistinguishable from no fix at all, which is how it survived one round.

Same error class as the framing bounds: *an assumption encoded instead of the
filesystem read*. Inferring a path from a sibling variable is not reading it.

(A speculatively-added `vehicle::SetDataPath` call failed at **compile** time —
that function does not exist in this version. The good outcome, and the contrast
worth noticing: the wrong-but-plausible path failed at runtime and silently, the
nonexistent function failed at build time and loudly.)

## RESOLVED: CRM soil renders from Python. The cause was the sprite shape type.

**2026-09-03, `kyle-sbel`, source build + `0001-expose-fsi-sph-render-options.patch`.**

| run | sprite shapes | attach order | dominant % | dark % |
|---|---|---|---|---|
| A | none (defaults) | after camera | 86.1 | 0.2 |
| B | `ChVisualShapeSphere` | after camera | 86.1 | 0.2 |
| **C** | **regolith meshes** | **after camera** | **20.3** | **59.3** |
| D | regolith meshes | before camera | 20.3 | 59.3 |

**`sprite_shapes` must hold triangle meshes.** A `ChVisualShapeSphere` is
accepted and draws nothing. The demo loads three regolith OBJs as
`ChVisualShapeTriangleMesh` with a white `ChVisualMaterial`; reproducing that
exactly — same files, same material, jitter 0.005, spacing 0.01 — is what made
particles appear.

### Two corrections to the record

**The attach ordering was NOT the fix.** Run C renders with the attach still
*after* `AddSensor`, which is the pre-`9005507` sequence. The
"`ReconstructScenes` does not retrofit an existing camera pipeline" mechanism was
plausible and is **falsified**. `9005507` is kept because matching upstream is
right on its own terms and costs nothing, but its commit message asserts a cause
we now know to be wrong. **Recorded here so the history is not believed.**

**And the inference that led there was mine and was wrong.** I argued that
byte-identical frames ruled out the sprite path executing, since a shape the
renderer ignored would still be a path that ran. Wrong: **the path ran and had
nothing drawable, so it added zero pixels.** "Ran and drew nothing" and "did not
run" are indistinguishable from pixels alone. Two hypotheses collapsed into one,
then reasoned from confidently, which foreclosed the candidate that was correct.

It cost one run rather than an evening only because `kyle-sbel` had already
started run C before that reasoning arrived. **Second time in one day a control
run caught an error the argument had already settled** — the rigid-terrain run
caught the determinism overclaim the same way. Both controls existed because
someone ran one when the practical question looked answered.

### The third silent no-op, and the worst of them

1. **Method compiled without OptiX** — empty body, returns `-1`. Detectable via
   the handle.
2. **Default options** — a null configuration, but the header *documents* both
   fields as required.
3. **A primitive sprite shape** — a valid object, of a type the API's own
   signature accepts, producing no output and no diagnostic.

The third is the worst because **there is no document to have read more
carefully.** `sprite_shapes` is typed `std::vector<std::shared_ptr<ChVisualShape>>`
and `ChVisualShapeSphere` *is* a `ChVisualShape`. Nothing states that primitives
are unsupported.

**Upstream:** the field should either accept primitives or reject them loudly.
Silently accepting a valid subtype and drawing nothing is the defect, not the
missing feature. Report order: options struct unbound, primitive-sprite silence,
`Background`, `ChDepthCamera` `ray_scale`.

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

## CLOSED: Python and C++ render equivalently, within the solver's own noise

**2026-09-03, `kyle-N7-B650E`, patched source build.** Agreed threshold: mean
RGB, dark < 40, bright > 180, mid-run frame.

| arm | language | handle | dark | bright |
|---|---|---|---|---|
| attached | C++ | 0 | 53.8% | 38.8% |
| attached, repeat | C++ | 0 | 54.3% | 38.6% |
| **py_attached** | **Python** | **0** | **54.8%** | **38.3%** |
| noattach | C++ | −1 | 0.1% | 93.9% |
| py_noattach | Python | −1 | 0.1% | 94.0% |

**The measurement that makes this a result rather than a number:**

| comparison | pixels differing >2 | mean abs |
|---|---|---|
| **C++ vs C++ (noise floor)** | **21.3%** | 21.00 |
| C++ vs Python, rep 1 | 22.3% | 22.03 |
| C++ vs Python, rep 2 | **17.8%** | 19.67 |

**Two runs of the identical binary differ from each other more than one of them
differs from Python.** The cross-language difference sits inside the solver's own
run-to-run variation, so at the resolution this measurement reaches, the paths
are indistinguishable. The three dark fractions span less than one point.

`kyle-N7-B650E` ran the second C++ arm specifically because the first comparison
returned "22.3% of pixels differ", which reads as a real discrepancy with nothing
to compare it against — and the SPH solver is nondeterministic while
`sprite_position_jitter` is stochastic. **Without the floor, "22.3% differ" would
have entered the record as a binding discrepancy.** This is
[the noise-floor rule](../lessons/experiment-design.md) applied unprompted rather
than quoted.

### The honest limit of the claim

It rests on summary statistics and a whole-frame diff at **one timestamp**. Two
images can share a dark fraction and differ structurally, and a 21% baseline is
wide enough to hide a real but modest cross-language effect. **What is shown is
that any difference is smaller than the solver's own variation** — the useful
practical statement, and weaker than "the images match."

### Porting the gate needed five Python API fixes, none behavioural

- `veh.SetDataPath` → `veh.SetVehicleDataPath`
- `BoxSide_*` lives in `pychrono.fsi`, not `pychrono.vehicle`
- **`rover.GetWheels()[i]` fails** — the `std::array` binds as an opaque
  `SwigPyObject` and is not subscriptable. Use `GetWheel(robot.V_LF)`.
- `scene.SetAmbientLight` takes `ChVector3f`, not `ChColor` — though
  `AddPointLight` accepts `ChColor`
- everything else transferred verbatim

The render options needed **no** workaround once the patch was applied:
constructed, three mesh sprites appended, spacing and jitter assigned and
**read back before the run** (0.0 → 0.0100; `len(sprite_shapes)` 0 → 3).

### Still open, unchanged

Whether the soil **deforms**, whether particle positions track the physics,
whether the bed reads as a **surface** rather than a scatter of sprites.
2.0 s per arm, one frame each inspected quantitatively. **Presence proven,
fidelity not.**
