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
