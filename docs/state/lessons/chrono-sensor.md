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
