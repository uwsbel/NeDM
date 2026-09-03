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

**So the real gap is much narrower than reported.** `ChModuleSensor.i` simply
never `%include`s or references the guarded methods, while the vehicle interface
does the equivalent successfully. There is a working pattern in the same repo to
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

## The SWIG binding to add

`AttachFsiSphSystem` is **absent** from `src/chrono_swig/interface/sensor/ChModuleSensor.i`,
not `%ignore`d. The header *is* fully `%include`d (line 243), but on `main` the
three FSI methods sit inside `#ifdef CHRONO_FSI_SPH`, which SWIG's preprocessor
does not have defined, so it skips the whole block. `AttachFsiSphSystem`,
`DetachFsiSphSystem` and `ClearFsiSphSystems` are all missing together, which
confirms a guard problem rather than a per-method oversight.

So this is **not a one-line `%rename`**. Three things are needed:

1. `CHRONO_FSI_SPH` defined for the SWIG preprocessor when sensor is built with
   FSI enabled — a CMake change.
2. `ChFsiSphRenderOptions` made a known type (`ChFsiSphRender.h`), or the default
   argument will not translate.
3. `ChFsiFluidSystemSPH` made known to the sensor module, which it is not today.
   `ChModuleSensor.i` has **no FSI awareness at all**: no `%import` of the fsi
   module. This creates a build-order dependency from the sensor bindings onto
   the fsi bindings.

Item 3 is the structural one: it touches the module dependency graph rather than
being a local edit. The type already exists on the Python side
(`terrain.GetFluidSystemSPH()` works), so the work is making the sensor module
aware of it, not creating it.

**Patch versus PR.** `.i` files are version-specific, so any patch must record
its target SHA beside it, and `main` moved 656 commits in five months with
binding work landing the same day we looked. An upstream PR is the only form that
does not rot. File the issue regardless of whether a patch is sent: the finding
stands on its own, alongside WP0c's unreported `ChDepthCamera` `ray_scale` bug.
