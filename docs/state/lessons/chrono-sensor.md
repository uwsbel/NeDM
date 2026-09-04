# Chrono::Sensor gotchas

Sourced from `docs/vision/double_pen/implementation_notes.md` and
`docs/vision/hmmwv_traverse/wp0b_implementation_notes.md`. All independently
measured, not inferred.

## ChDepthCamera casts rays 1.20× wider than its HFOV implies

**Found:** 2026-09-01 (WP0b) · **Applies to:** any depth sensor use, this Chrono build

**Expected:** depth rays follow the constructor HFOV, as the RGB camera does.
**Happened:** depth→elevation error 1.5 m median, 3 m at edges, with a radial
signature. **Cause:** an upstream ray-tangent scale bug; the RGB
`ChCameraSensor` honors HFOV (proven by 0.97 px alignment over 201 targets),
the depth camera does not. **Fix:** fitted scalar `ray_scale = 1.200` in
`CameraModel.depth_to_world` (`nedm/traverse/camera.py`), recorded in the
dataset manifest. Corrected error: 6.3 mm median, 16.9 mm at image edges.

**This is upstream-worthy — report it to Project Chrono.**

## Do not push a second depth access filter

`ChDepthCamera` installs its own access filter internally. Pushing a
`ChFilterDepthAccess` on top fails `AddSensor` validation. Just don't.

## Default sensor lag is one full period

Frame data only becomes available once sim time passes `launch + lag`, so a
reader that blocks at the control boundary **deadlocks**. `SetLag(0)`, and
`SetCollectionWindow(0)` for a true snapshot with no motion blur.

## The sensor scheduler misses boundaries

Launch times are float32 accumulations of `k/rate`; around some boundaries
(first seen at t = 2.14 s) the launch slips one substep late — 1 ms of content
error, or a deadlock for a blocking reader. **Bypass:** set nominal camera rate
to `1/dt_sim` so the schedule is always behind the clock, and call
`manager.Update()` *only* at control boundaries — each call then fires exactly
one render of the current state. **Associate frames by `LaunchedCount`, never by
timestamp.**

## Teleporting bodies leaves stale state

After `SetPos` / `SetRot` / `SetPosDt` / `SetAngVelParent`, the next step depends
on the *previous* episode's motion — measured up to 0.47 rad/s one-step
deviation, identical across solver choices, so not solver noise. **Fix:**
`system.Setup(); system.Update()` after the reset makes one-step replay bitwise
deterministic. This is what lets episodes share one system and one sensor
manager, avoiding the OptiX scene re-creation instability class.

## Alignment must be measured under near-zenith light

With a 55° collection sun, color centroids shift ~1 px toward the lit side and
inflate the alignment median to ~3 px. Render alignment probes at
`light_elevation_deg=80`; keep 55° for collection.

## Rendered colors, not material diffuse

Blob-detection references must be the *rendered* colors. An orange marker with
0.4× emissive saturated to sand-white and became undetectable; it is now blue
with 0.1× emissive. Marker detection still only succeeded on 6/10 layouts at
256² (~5×3 px) — an open risk for vehicle-center probes.

## Collision types default to NONE — verify contact can actually fire

`create_hmmwv` left Chrono's default `ChassisCollisionType = NONE`, and TMEASY
tires only query the terrain. A pass with a rock buried 0.62 m inside the vehicle
footprint recorded **0 N**. This made a "zero collisions" gate result *vacuously
true*. **Before claiming zero contact, prove the contact channel can fire.**

## A green Chrono::Sensor build may not contain the backend you meant to test

**Cost:** one full 220-target build, discarded · **Found:** 2026-09-04 · **Applies to:** verifying any Chrono::Sensor backend on Linux

Verifying "the OptiX path" on a `feature/sensor_metal_rt` checkout produced a clean
build, eleven `demo_SEN_*` binaries and a passing test suite — **with OptiX compiled
out entirely.** Two independent defaults, neither of which announces itself:

**1. `CH_USE_SENSOR_OPTIX` defaults to `OFF`** (upstream, `chrono_sensor/CMakeLists.txt`),
described as *"legacy"*; Vulkan RT is the default renderer. **Passing `OptiX_INSTALL_DIR`
does not enable it and does not warn that it was ignored.**

**2. With it explicitly `ON`, it can still be forced back OFF — by a warning.**

```
CMake Warning: Chrono::Sensor OptiX renderer requires CUDA, but none is available
               (vendor=NVIDIA, CUDA=FALSE, HIP=FALSE)
-- Building Chrono::Sensor with NO OptiX support
```

`CUDA=FALSE` on a box with CUDA 13.0.88 and a working `nvcc`, purely because
`/usr/local/cuda/bin` was not on `PATH` and `CMAKE_CUDA_COMPILER` resolved `NOTFOUND`.
Configure succeeds, build succeeds, tests pass.

**The severities are inverted relative to the consequences, which is the general
lesson.** In the same file, a missing `glslangValidator` is a `FATAL_ERROR` that stops
the build dead — and the cost of proceeding would have been *one disabled optional
feature*. A missing CUDA toolchain is a **warning** — and the cost of proceeding is
**silently measuring a different renderer than the one under test**. The loud failure
guards the cheap mistake; the quiet one guards the expensive mistake.

Related asymmetry in the same `if()`: Vulkan **absent** takes an `else()` that warns and
disables gracefully, while Vulkan **present without `glslang-tools`** is fatal. On Ubuntu
`libvulkan-dev` arrives with the graphics stack while `glslang-tools` is a separate
package, so the **more**-installed machine fails where the **less**-installed one builds.

**Fix / checklist before trusting any sensor verification:**

```bash
grep -E "^CH_USE_SENSOR_(OPTIX|VULKAN_RT|VULKAN_RT_GPU|METAL_RT):" CMakeCache.txt
ldd bin/demo_SEN_camera | grep -E "nvrtc|cuda"   # OptiX build links these
```

Assert the backend from the **cache and the linkage**, never from "it configured and the
tests passed". Required on Linux for OptiX:
`-DCH_USE_SENSOR_OPTIX=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.

**And `CMAKE_CUDA_ARCHITECTURES` is not the knob.** `cmake/ChronoGPUDetect.cmake`
overwrites `CHRONO_CUDA_ARCHITECTURES` from `CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR` with
`FORCE`, so `-DCMAKE_CUDA_ARCHITECTURES=120` is accepted and discarded, leaving
`60-real;70-real;80-real;90` on an sm_120 part — which *runs*, via PTX JIT from the
virtual `90` arch, while no longer measuring native code. Use
`-DCHRONO_CUDA_ARCHITECTURES=<arch>`.

**Evidence:** `chrono_sensor/CMakeLists.txt` (OptiX default, glslang `FATAL_ERROR`);
`cmake/ChronoGPUModule.cmake` (CUDA warning path); `cmake/ChronoGPUDetect.cmake` (arch
`FORCE`). Symptom that exposed it: `demo_SEN_Gator` missing from `bin/`, and only
because it happens to sit behind the same `if(CH_USE_SENSOR_OPTIX)`.

## Sensor demos resolve data by a RELATIVE path, so cwd decides whether they crash

**Cost:** two false "Linux segfault" results, nearly reported · **Found:** 2026-09-04 · **Applies to:** every `demo_SEN_*`

`demo_SEN_camera` and `demo_SEN_Gator` both **segfaulted (exit 139)** when run as
`./bin/demo_SEN_camera` from the build root. Not a crash in the code:

```
tiny_obj error message: Cannot open file [../data/vehicle/audi/audi_chassis.obj]
Segmentation fault
```

The demos load assets through a **relative** `../data/`, which resolves against the
**current working directory**, not the executable. From the build root that is
`<build>/../data`, which does not exist; the loader returns a null mesh and the demo
faults on it. Run from `bin/`, where `../data` resolves, both are fine —
`demo_SEN_camera` exits **0**.

**A missing asset that surfaces as a segfault is indistinguishable from a real crash
in a CI log or a bug report.** Check the working directory before filing one.

**Corollary: exit 124 is not a hang.** `demo_SEN_Gator` hit a 120 s cap with output
frozen after *"Add sensor 'GPS'"* and an empty `DEMO_OUTPUT` — which looks exactly like
a deadlock. Process state said otherwise: **`Rl`, 114% CPU, 8 threads.** It was
simulating with no per-step console output. Sample `ps -o stat=,pcpu=` before calling
anything hung; silent is not stopped.
