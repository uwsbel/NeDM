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
