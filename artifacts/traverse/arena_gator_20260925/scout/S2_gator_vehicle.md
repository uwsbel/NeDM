# S2: can the Chrono Gator replace the HMMWV on f104? (scouting, 2026-09-25)

Scope: task (B) of the overnight plan. Keep the f104 arena, swap the HMMWV for the Chrono Gator, collect the same
amount of data, train the same planner, and say whether it works. This file is read-only scouting. No repository
file was changed, no cluster job was submitted and no collection was run. What I did run:
- short local Chrono smokes (flat and tilted rigid patches, one tiny 14 x 3 m CRM patch), using scratch scripts in
  `/tmp/s2_gator/` (the main two are copied into the appendix);
- `import`-only checks on the cluster login node.

Anything I did not measure is marked **UNVERIFIED**.

## 0. Bottom line

1. **The Gator is available everywhere we need it.** `veh.Gator` and its 18 component classes import in three places:
   - the local source build (`/usr/bin/python3.12`, `PYTHONPATH=/home/harry/chrono/build/bin`);
   - the local `nedm` conda pychrono;
   - both cluster builds (`chrono-build`, `chrono-build-fsi`).

   It instantiates and drives locally. Only `chrono-build-fsi` has `CRMTerrain` and `pychrono.fsi`, the same as for
   the HMMWV. The Gator source files and tyre meshes are byte-identical between the local and cluster checkouts.
2. **It is a very different vehicle.** All figures below are measured in the smoke unless stated.

   | | Gator | HMMWV collector setup |
   |---|---|---|
   | Mass | 906 kg | 2,573 kg |
   | Wheelbase (real) | 1.94 m | 3.30 m |
   | Track | 1.12 m front / 1.24 m rear | 1.82 m |
   | Tyre radius | 0.286 m front / 0.318 m rear (unequal) | 0.470 m |
   | Driven wheels | rear only; front driveline torque measured 0 | all four |
   | Brakes | rear axle only | all wheels |
   | Engine and gearbox | fixed simple engine: 200 N·m, 14 kW, one gear | shafts engine, 3-speed automatic, torque converter |
   | Top speed, flat | 8.2 m/s | 20.2 m/s |
   | Full-lock turn radius | 2.8 m (curvature 0.35 /m) | 6.8 m (0.147 /m) |
   | Settled ground clearance | about 0.14 to 0.17 m | about 0.51 m |

3. **The feared power limit is not what the numbers show.** The worry was that the Gator could not reach the 6 m/s
   speed profiles. On flat rigid ground it reaches 6 m/s in 1.5 s. Climbing from rest it holds these speeds:

   | Slope | Gator | HMMWV collector setup |
   |---|---|---|
   | 10° | 7.0 m/s | 4.5 m/s |
   | 15° | 5.9 m/s | 4.2 m/s |
   | 20° | 4.4 m/s | 4.0 m/s |
   | 25° | 3.4 m/s | 2.2 m/s |
   | 30° | 2.8 m/s | stalls and rolls back |

   The HMMWV result comes from the collector's setup: it stays in first gear on these slopes.

   Two Chrono modelling artefacts inflate the Gator's rigid-ground ability:
   - its simple engine has maximum torque at all low speeds;
   - its tyres are scaled to 1.5 times their nominal grip on the 0.9-friction terrain, against 1.125 times for the
     HMMWV. The Gator's tyre friction is set to 0.6 against 0.8 for the HMMWV, and Chrono scales grip by terrain
     friction divided by tyre friction.
4. **The real rigid-ground risks are clearance, rear-only braking and rollover.**
   - Clearance angles are estimated from the collision hull at the settled ride height:

     | | Gator | HMMWV |
     |---|---|---|
     | Departure angle | about 21° | about 42° |
     | Breakover angle | about 22° | about 49° |

     With chassis collision on (as in the rigid collector), crater lips and hill crests can ground the Gator's body.
   - The default brake without wheel locking creeps downhill at full brake. In 6 s it moves 1.2 m on 15°, 3.0 m on
     20° and 8.6 m on 30°. The shafts brake type runs away on 30° (5.5 m/s after 6 s).
   - Flat-ground braking averages only 3.3 m/s², against 5.8 for the HMMWV.
   - The static rollover margin is smaller: about 43° tilt against about 49°.
5. **The existing path follower works on the Gator without changes, but tracks about 15 to 30% worse.** On a flat
   test route with two 8 m-radius 90° bends, the unchanged follower gives RMS cross-track error 0.64 to 0.74 m,
   against 0.49 to 0.63 m for the HMMWV. Lowering the steering gain to 0.35 gives 0.35 to 0.45 m. The HMMWV needs
   0.87 to 0.95 of full steering on those bends; the Gator needs 0.40.
6. **Most of the stack needs no change.**
   - The frozen loop only calls wrapper methods that `veh.Gator` also has: `GetVehicle`, `GetSystem`, `GetChassis`,
     `GetChassisBody`, `Synchronize`, `Advance` and the visualisation setters.
   - The HMMWV is actually hard-coded in:
     - one vehicle factory (`src/nedm/hmmwv_data.py:290-336`);
     - one config builder (`src/nedm/traverse/scene.py:54-106`);
     - the start height of +0.75 m at three call sites;
     - the HMMWV tyre mesh in the soil collector;
     - three provenance checks that demand HMMWV data files.
   - I ran the repo's own row-capture function, the 17-column state preset and the rich-telemetry snapshot on a live
     Gator. The state is finite and in the same column order. 24 suspension telemetry fields are NaN because the
     Gator's suspension is not a double wishbone; the labels do not use them.
7. **Soil (CRM) is feasible but the tyre geometry must change.**
   - The Gator tyre meshes are not watertight: 68 and 72 boundary edges, and 11 non-manifold edges each.
   - At the production particle spacing of 0.08 m they become 94 and 116 boundary markers with a lumpy outline (the
     HMMWV tyre gets 207). Along the outer edge, the median marker radius is 0.17 m and 0.23 m, against nominal radii
     of 0.286 and 0.318 m, and some 10° sectors have no markers at all.
   - Chrono's own marker counts equal my numpy re-implementation: 94, 116 and 207 at 0.08 m; 817 and 951 at 0.04 m.
   - A tiny soil drive works with the mesh at 0.08 m (0.5 and 1 ms steps), with a cylinder at 0.08 m, and with the
     mesh at 0.04 m.
   - The Gator's body clearance (0.14 to 0.17 m) is below the 0.24 m soil layer, and the chassis is not coupled to
     the soil. A bogged Gator can therefore sink its belly through soil that should hold it up.
8. **No Chrono demo couples a Gator to CRM.** The wheeled-vehicle soil demos use the Polaris RZR: a rigid-tyre
   collision mesh per spindle, added with `AddRigidBody(spindle, geometry, false)`, at 0.04 m spacing and a 0.5 ms
   step. That is the same pattern as our soil collector.
9. **Verdict: Gator on rigid ground is a small code change.** I would run a rigid pilot first, because the failure
   mix will differ from the HMMWV's (grounding and braking rather than wheel lift and stalls), and the fraction of
   failed routes may be too low or too high for useful labels.
10. **Verdict: Gator on soil needs a one-hour calibration first**, choosing the tyre representation and optionally
    coupling the chassis. The rear-wheel-drive traction on soil is the unknown that decides whether the 97.5%-style
    goal-reaching result is even attainable.

## 1. Availability checks

| Environment | Gator symbols | `CRMTerrain` / `pychrono.fsi` | How checked |
|---|---|---|---|
| Local source build: `/usr/bin/python3.12`, `PYTHONPATH=/home/harry/chrono/build/bin`; Chrono `a92c6f72` (2026-09-12, branch `project/nrd`) | 19: `Gator`, `Gator_Vehicle`, `Gator_EngineSimple(Map)`, `Gator_AutomaticTransmissionSimple(Map)`, `Gator_SimpleDriveline`, `Gator_Driveline2WD`, `Gator_RackPinion`, `Gator_SingleWishbone`, `Gator_RigidSuspension`, `Gator_TMeasyTire_Front/Rear`, `Gator_RigidTire_Front/Rear`, `Gator_Brake{Simple,Shafts}`, `Gator_Chassis`, `Gator_Wheel` | yes / yes | Import, `Gator().Initialize()`, driving smokes (section 2) |
| Local `nedm` conda: `/home/harry/miniconda3/envs/nedm/bin/python` (pychrono 10.0.0 per memory notes) | Same 19 | yes / yes | Import plus `Initialize()` (mass 906.2 kg). The vehicle data path setter is `veh.SetVehicleDataPath`; there is no `veh.SetDataPath`. `configure_chrono_data_paths` already handles both names. |
| Cluster `/work1/dannegrut/harry/nrd/chrono-build` (rigid; used by `gen_runner.py:27`) | Same 19 | **no** / **no** | Login-node import only (command in the appendix) |
| Cluster `/work1/dannegrut/harry/nrd/chrono-build-fsi` (used by `crm_collect.sbatch:9`) | Same 19 | yes / yes (`fsi.SoilProperties` present) | Login-node import only |

- **Data files.** `data/vehicle/gator/` exists in both cluster builds. It contains the chassis meshes, the four
  wheel meshes, the coarse and fine tyre meshes (front and rear), `json/` and `Textures/`.
- **Identical copies.** The tyre meshes have identical md5 locally and on the cluster:
  - `gator_tireF_coarse.obj` `580d356f…`
  - `gator_tireR_coarse.obj` `ff17e09b…`
  - `hmmwv_tire_coarse_closed.obj` `3c7c4ddf…`
- **Same source.** The cluster source (`/home1/harry/chrono`, commit `f54254fa`, 2026-09-07) and the local source
  have the same aggregate md5 over all `src/chrono_models/vehicle/gator/*.cpp` (`be3e9fd1…`).

## 2. Gator against the collector's HMMWV (measured)

### 2.1 How the smoke was set up

The smoke settings mirror `scene.build_config` (`scene.py:54-106`):
- SMC contact;
- RigidTerrain patch with friction 0.9, restitution 0.01, Young's modulus 2e7;
- 2 ms physics step, 1 ms tyre step;
- 0.8 s settle at full brake before any command.

Vehicle setups:
- **HMMWV**, exactly the collector's: `HMMWV_Full`, shafts engine, automatic shafts transmission, all-wheel drive,
  Pitman-arm steering, TMEASY tyres.
- **Gator**: `veh.Gator()` with SMC contact, TMEASY tyres, and the default simple driveline and simple brakes.
- Chassis collision was off in the dynamics tests and on (convex hulls, as in the collector) in the settle test. The
  patch is a plane, so this makes no difference there.

Wall time for the whole Gator scenario list: 17.3 s. The same list on the HMMWV: 44.7 s.

### 2.2 Static properties

| | Gator | HMMWV (collector setup) |
|---|---|---|
| `GetMass()` | 906.2 kg (chassis 800 kg) | 2573.1 kg (chassis 2086.5 kg) |
| Wheelbase, measured from spindle positions | **1.94 m** (axles at ±0.97 m, `Gator_Vehicle.cpp:125,127`) | 3.30 m |
| `GetWheelbase()` as declared | 2.776 (**wrong**, `Gator_Vehicle.h:53`) | 3.378 |
| Track, `GetWheeltrack(0/1)` | 1.12 / 1.24 m | 1.819 / 1.819 m |
| Tyres (TMEASY) | Front 22.5x10-8: r 0.28575 m, w 0.254 m. Rear 25x12-9: r 0.3175 m, w 0.3048 m. 9.3 kg each. Tyre friction 0.6 (`Gator_TMeasyTire.cpp:67,115`). | 37x12.5R16.5: r 0.4699 m, w 0.3175 m, 37.6 kg. Tyre friction 0.8 (default). |
| TMEASY grip scaling on friction-0.9 terrain (terrain friction divided by tyre friction, `ChTMeasyTire.cpp:121`) | **1.5** | 1.125 |
| `GetRadius()` after the first `Synchronize` | Loaded radius: 0.275 / 0.306 m at 7 m/s. The collectors capture radii *before* the first `Synchronize`, so they store the unloaded values. | 0.4647 loaded |
| Settled chassis reference height | 0.272 m (front spindle 0.252, rear 0.286) | 0.588 m (spindles 0.454) |
| Collision hull size (length x width) | 3.45 x 1.50 m; hull spans x from -2.21 to +1.24 m around the reference (asymmetric, long cargo bed) | 4.67 x 2.56 m |
| Clearance (lowest hull vertex; between axles) | **0.14 m; 0.17 m** | 0.51 m; 0.51 m |
| Approach / departure / breakover angle (estimate from static hull vertices) | 54° / **21°** / **22°** | 49° / 42° / 49° |
| Chassis centre-of-mass height at rest; static stability (track divided by twice the height) | ≈ 0.63 m; ≈ 0.94 (≈ 43° tilt) | ≈ 0.80 m; ≈ 1.14 (≈ 49°) |
| Front suspension / rear suspension | Single wishbone / **rigid (no suspension)** | Double wishbone / double wishbone |
| Steering | Rack-and-pinion, `m_maxAngle = 1` (pinion rad, `Gator_RackPinion.cpp:37`) | Pitman arm, 30° |
| Declared maximum steering angle / minimum turn radius | 25° / 7.6 m (**both wrong**, `Gator_Vehicle.h:54-55`) | 30.23° / 7.62 m |
| Drive layout | **Rear axle only** (`Gator_Vehicle.cpp:131`: driven suspension index 1). Measured spindle torques [front L, front R, rear L, rear R] = [0, 0, 281.5, 281.5] N·m at 2 s. | All wheels (`ShaftsDriveline4WD`) |
| Driveline options (`SetDrivelineType`) | `SIMPLE` (default: Gator custom driveline, a limited-slip split up to 2:1 once the left-right speed difference exceeds 0.5 rad/s), or `RWD` (`ShaftsDriveline2WD`, open differential, lockable). Any other value falls back to `SIMPLE` (`Gator_Vehicle.cpp:100`). | All-wheel drive, open differentials |
| Engine and gearbox | Hard-wired `Gator_EngineSimple`: 200 N·m, 14 kW, cut-off 3500 rpm = 366.5 rad/s, torque x throttle. One-speed simple gearbox with overall ratio 0.07 and no torque converter (`Gator.cpp:113-123`, `Gator_EngineSimple.cpp:25-27`, `Gator_AutomaticTransmissionSimple.cpp:23`). A map-based engine and 5-speed gearbox exist as classes but the wrapper never selects them. The Gator wrapper has no `SetEngineType` or `SetTransmissionType`. | Shafts engine, shafts automatic gearbox (3 forward gears) |
| Brakes | **Rear axle only** (`Gator_Vehicle.cpp:82`), 800 N·m each. Type `SIMPLE` (default) or `SHAFTS`. `EnableBrakeLocking` available. | All four wheels |
| Tyre models | Only `RIGID`, `RIGID_MESH` and `TMEASY` (`Gator.cpp:127-164`). Any other type hits `default: break` and leaves **no tyres** (from source, not run). | TMEASY (rigid), rigid mesh (CRM) |

### 2.3 Dynamics

**Flat ground, full throttle from rest.**

| | Gator | HMMWV |
|---|---|---|
| Time to 2 / 4 / 6 / 8 m/s (0.5 s log resolution) | 0.5 / 1.0 / 1.5 / 3.0 s | 1.0 / 1.5 / 2.0 / 2.5 s |
| Speed at 5 / 10 / 20 / 30 s | 8.30 / 8.28 / 8.16 / 8.25 m/s | 12.0 / 18.0 / 20.2 / 20.2 m/s |
| Top-speed limit | Engine cut-off. Engine at 390 rad/s with 0 N·m, gear 1. | Engine at 271 rad/s, gear 3 |
| Peak engine power (motor speed x torque) | 14.0 kW | 81 kW |

**Flat ground, full brake.** The Gator stops from 8.2 m/s in 2.5 s (mean 3.3 m/s²). The HMMWV stops from 19.0 m/s in
3.25 s (5.8 m/s²).

**Full-lock left turn at a held speed (last 8 s of 20 s).**

| | Gator | HMMWV |
|---|---|---|
| Radius at 2 m/s (circle fit) | **2.84 m** | 6.85 m |
| Radius at 4 m/s | 2.82 m; the Gator could only hold 3.6 m/s | 6.82 m |
| Road-wheel angles FL / FR | 41.6° / 46.5° (anti-Ackermann: the outer wheel steers more) | 29.5° / 25.1° |
| Maximum roll | 1.6° | 0.8° |

**Climbing from rest, full throttle, rigid friction 0.9.** Speed at 4 / 8 / 11 s:

| Slope | Gator | HMMWV |
|---|---|---|
| 10° | 6.59 / 7.04 / 7.03 m/s | 4.51 / 4.52 / 4.51 (gear 1, engine 270 rad/s) |
| 15° | 5.21 / 5.75 / 5.93 | 4.22 |
| 20° | 4.08 / 4.29 / 4.41 | 3.99 / 4.01 |
| 25° | 3.30 / 3.40 / 3.44 | 2.17 / 2.22 / 2.32 |
| 30° | 2.80 / 2.82 / 2.84 | 1.43 / 0.60 / -0.11 (stalls, rolls back) |

For reference, f104 slopes over a 2 m baseline: median 7°, 90th percentile 19.6°, 99th percentile 26.6°, maximum
29.9°. 21% of cells are steeper than 15°, 9% steeper than 20° and 2% steeper than 25°.

**Held on a downhill with full brake, facing down, for 6 s after the settle.** Distance slid (final speed):

| Setup | 15° | 20° | 25° | 30° |
|---|---|---|---|---|
| Gator, `SIMPLE` brake (default) | 1.20 m (0.25 m/s) | 2.99 m (0.50) | 3.33 m (0.54) | 8.63 m (1.09) |
| Gator, `SIMPLE` + `EnableBrakeLocking(True)` | 0.03 m | 0.03 m | 0.14 m | 2.69 m |
| Gator, `SHAFTS` brake | 0.03 m | 0.04 m | 0.31 m | **16.7 m, 5.46 m/s runaway** |
| HMMWV, collector setup | 1.30 m (0.22) | 1.35 m | 1.56 m | 1.77 m (0.29) |

The HMMWV also creeps: this is TMEASY's behaviour at standstill.

**Settling at the frozen spawn height.** The collectors spawn the chassis reference at ground + 0.75 m
(`traverse_fdm_rgbd_diverse_chrono.py:162`, `crm_collect.py:177`, `crm_collect_ext.py:114`). At the 0.8 s anchor:

| Spawn height | Vehicle | Vertical velocity | Total tyre load / weight |
|---|---|---|---|
| +0.75 m | Gator | **-1.23 m/s (still bouncing)** | 1.01 |
| +0.75 m | HMMWV | +0.17 m/s | 1.11 |
| +0.40 m | Gator | +0.05 m/s | 0.78 |
| +0.35 m | Gator | -0.01 m/s (settled) | 0.87 |

With chassis collision on, the Gator's chassis contact force at rest is 0.

**API check.** Using the Gator, I ran the repo's `capture_row(..., include_tires=True)`, then built
`STATE_FIELD_PRESETS["tire_normal_force_omega_pt"]` from it, plus the engine columns. The result is a finite 17-column
state. Example at 5.8 m/s: tyre vertical loads 1.8 kN front and 2.7 kN rear per wheel; spindle speeds 20.9 rad/s
front and 17.9 rad/s rear (unequal radii); engine 284 rad/s at 29.6 N·m.

`RichTelemetry._snapshot` produces 196 keys, 24 of them NaN: the `*_suspension_{spring,shock}_*` fields, because
`CastToChDoubleWishbone` returns `None` for single-wishbone and rigid suspensions. The collector catches this at
`fdm_rich_telemetry.py:202-215`.

The external-control collector reads engine speed from the gearbox before `Synchronize` and asserts it equals the
recorded value (`gen_collect_ext.py:283`, checked at 460-475). On the Gator the two agree exactly (max difference
0.0 rad/s), as on the HMMWV.

**What the vehicle API exposes** (all used above):
- **Wheels:** `GetVehicle().GetSpindleOmega(axle, side)` (native sign negative for forward rolling),
  `GetSpindleAngVel`, `GetSpindleLinVel`, `GetSpindlePos`, `GetSpindleRot`.
- **Tyres:** `GetTire(a, s).GetRadius()`, `GetWidth()`, `GetMass()`, `ReportTireForce(terrain)`,
  `GetLongitudinalSlip()`, `GetSlipAngle()`, `GetDeflection()`.
- **Engine and gearbox:** `GetEngine().GetMotorSpeed()`, `GetOutputMotorshaftTorque()`;
  `GetTransmission().GetOutputMotorshaftSpeed()`, `GetCurrentGear()` (always 1), `GetMaxGear()` (= 1).
- **Driveline:** `GetDriveline().GetSpindleTorque(a, s)`, `GetOutputDriveshaftSpeed()`.
- **Steering and suspension:** `GetSteering(0)` (rack-and-pinion); `GetSuspension(a).GetAxleSpeed(side)`.
- **Chassis:** `GetSpeed`, `GetRoll`, `GetPitch`, `GetRollRate`, `GetYawRate`, `GetSlipAngle`.
- **Driver input:** `DriverInputs.m_steering` in [-1, 1] (+ = left), `m_throttle`, `m_braking`.
- **Wrapper-only methods:** `SetDrivelineType`, `SetBrakeType`, `EnableBrakeLocking`, `LockAxleDifferential`,
  `SetTireType`, `SetTireCollisionType`, `SetInitWheelAngVel`, `SetAerodynamicDrag`.

**UNVERIFIED quirk:** the Gator's engine speed is only approximately rear-wheel speed / 0.07. The measured ratio was
off by 3 to 9% at different moments. Not investigated.

### 2.4 Path-follower tracking on flat ground

Setup:
- Test route: 15 m straight, a 90° left bend of radius 8 m (curvature 0.125, the route validator's limit), 10 m
  straight, a 90° right bend of radius 8 m, then 25 m straight.
- Follower: `ChPathFollowerDriver` with 5 m look-ahead and speed gains (0.6, 0.05, 0). Waypoints thinned to one per
  2 m, path z + 0.5 m, per-substep steering clamp of 2 full-scale units per second. Metrics from 1 s after the start.
- Script: `/tmp/s2_gator/follow.py`.

| Vehicle | Steering gain | Speed | RMS / 95th pct / max cross-track | Steering reversals per s | Max \|steer\| | Completed |
|---|---|---|---|---|---|---|
| Gator | 0.8 (frozen) | 2 m/s | 0.74 / 1.41 / 1.44 m | 2.15 | 0.58 | yes |
| Gator | 0.8 | 4 m/s | 0.72 / 1.36 / 1.39 m | 0.85 | 0.40 | yes |
| Gator | 0.8 | 6 m/s | 0.64 / 1.19 / 1.22 m | 0.55 | 0.40 | yes |
| Gator | 0.5 | 2 / 4 / 6 m/s | RMS 0.60 / 0.59 / 0.50 m | 1.31 / 0.60 / 0.63 | 0.42 / 0.39 / 0.40 | yes |
| Gator | 0.35 | 2 / 4 / 6 m/s | RMS 0.45 / 0.44 / 0.35 m; max 0.89 / 0.87 / 0.67 m | 1.77 / 0.48 / 0.45 | 0.41 / 0.39 / 0.40 | yes |
| HMMWV | 0.8 (frozen) | 2 / 4 / 6 m/s | RMS 0.63 / 0.61 / 0.49 m; max 1.23 / 1.17 / 0.91 m | 2.02 / 1.43 / 0.45 | **0.89 / 0.87 / 0.95** | yes |

- The follower uses the look-ahead PID steering controller (`ChPathFollowerDriver.cpp:139`). That controller does not
  read the declared wheelbase or maximum steering angle. The pure-pursuit and XT variants do (`:176`, `:296`), and
  they would use the wrong Gator values.
- A full steering command gives about 2.4 times more path curvature on the Gator:
  - full lock: curvature 0.35 against 0.147 /m;
  - road-wheel steering rate at the frozen clamp: about 88°/s against 60°/s.

### 2.5 Tiny soil (CRM) smoke

Setup:
- Flat 14 x 3 m box, 0.24 m soil depth, soil and particle parameters of `crm_main.json`, 2 x 2 x 1 m active domain.
- Vehicles spawned at surface + 0.55 m (Gator) or + 0.75 m (HMMWV).
- 0.8 s braked settle, then 2 s full throttle, on the local RTX 5090.
- Sinkage = radius - (spindle z - 0.24). A negative value means the tyre rides above the undisturbed surface.
- Script: `/tmp/s2_gator/crm_smoke.py` (appendix).

| Wheel geometry | Spacing, step | Markers per wheel | Speed at 1 s / 2 s of throttle | Sinkage at end (FL, FR, RL, RR) | Real-time factor (tiny patch) |
|---|---|---|---|---|---|
| Gator, per-axle meshes `gator_tire{F,R}_coarse.obj` | 0.08 m, 0.5 ms | 94, 94, 116, 116 | 2.02 / 2.88 m/s | -0.03, -0.02, -0.06, -0.03 | 0.39 |
| same | 0.08 m, 1 ms (production step) | 94, 94, 116, 116 | 1.68 / 2.75 | -0.03, -0.02, -0.03, -0.02 | 0.77 |
| Gator, cylinder of nominal radius and width | 0.08 m, 0.5 ms | 212, 212, 295, 295 | 3.17 / 4.18 | **-0.10, -0.10, -0.12, -0.12** | 0.38 |
| same | 0.08 m, 1 ms | 212, 212, 295, 295 | 2.83 / 4.27 | -0.10, -0.10, -0.12, -0.12 | 0.76 |
| HMMWV, `hmmwv_tire_coarse_closed.obj` (production) | 0.08 m, 0.5 ms | 207 x 4 | 2.77 / 3.99 | -0.03, -0.03, 0.00, -0.01 | 0.38 |
| same | 0.08 m, 1 ms | 207 x 4 | 1.96 / 3.59 | -0.02, -0.03, +0.03, 0.00 | 0.76 |
| Gator, per-axle meshes | 0.04 m, 0.5 ms (186,732 particles against 27,456) | 817, 817, 951, 951 | 0.87 / 2.71 | -0.01, -0.01, -0.02, -0.02 | 0.30 |

Reading:
- **The coupling itself works** on this build: `RegisterVehicle`, `AddRigidBody(spindle, geometry, False)` with
  either meshes or cylinders, `terrain.Advance`, and `GetFsiBodyForce`. The total soil force at the end of the settle
  is 8.0 to 10.1 kN against 8.9 kN of weight (still bouncing, like the HMMWV's 27.2 to 29.2 kN against 25.2 kN).
- **Markers sit on the exact radius for a cylinder, but the soil stays about one spacing away.** The cylinder's
  effective rolling radius therefore becomes about r + 0.1 m, which makes the tyre 35% larger. The mesh markers sit
  inside the mesh, so they partly cancel that offset.

**Tyre mesh quality** (`/tmp/s2_gator/bce.py`; marker counts equal Chrono's own `GetNumBCE`):

| Mesh | Watertight? | Radius span | Markers at 0.08 m | Outer marker radius per 10° sector at 0.08 m (min / median) | Empty sectors at 0.08 / 0.06 / 0.05 / 0.04 m |
|---|---|---|---|---|---|
| `hmmwv/hmmwv_tire_coarse_closed.obj` | yes (918 edges, all shared by 2 faces) | 0.27 to 0.468 m | 207 | 0.315 / 0.418 | 0 / 0 / 0 / 0 |
| `gator/gator_tireF_coarse.obj` | **no** (68 open edges, 11 non-manifold) | 0.096 to 0.277 m (nominal 0.286) | 94 | **0.000 / 0.170** | 13 / 5 / 0 / 0 |
| `gator/gator_tireR_coarse.obj` | **no** (72 open, 11 non-manifold) | 0.106 to 0.301 m (nominal 0.318) | 116 | **0.000 / 0.228** | 12 / 4 / 4 / 0 |
| `Polaris/meshes/Polaris_tire_collision.obj` (Chrono demo) | yes | 0.199 to 0.343 m | 83 | – | 8 at 0.08 m (the demo runs at 0.04 m) |

With `check_embedded=False` (our setting), CRM only uses a two-ray parity inside-test to place markers
(`ChFsiFluidSystemSPH.cpp:2933`). The flood-fill that requires a watertight mesh (`ChFsiProblemSPH.cpp:473-485`)
does not run, so the open meshes do not throw an error; they just produce the ragged marker sets above.

## 3. Every place the HMMWV is assumed

### 3.1 Vehicle construction and scene (rigid and soil)

**Vehicle factory: `src/nedm/hmmwv_data.py:290-336` `create_hmmwv(config)`.**
- It refuses anything but `HMMWV_Full` (line 294-295) and calls `veh.HMMWV_Full()` (line 297).
- It calls HMMWV-only setters: `SetEngineType` (308), `SetTransmissionType` (309), `SetDriveType` (310) and
  `SetSteeringType` (311).
- It then sets the tyre type, tyre step and chassis collision (312-326), and turns all visualisation off (329-334).
- Callers: `scene.build_scene:354`, `crm_collect.run:184`, `crm_collect_ext:121`, `crm_smoke.py:68`, plus older
  scripts.
- `WHEEL_SPECS` (54-59), `capture_row` (438-534) and `tire_field_names` are generic for 2-axle vehicles and work on
  the Gator.

**Config builder: `src/nedm/traverse/scene.py:54-106` `build_config`.**
- It sets `"model": "HMMWV_Full"` (76), the shafts engine, shafts automatic gearbox, all-wheel drive and Pitman-arm
  steering (85-88), TMEASY tyres (89) and chassis collision `HULLS` (93).
- `TraverseScene.hmmwv` (232) and `build_scene` (353-372) re-enable the mesh visuals and add the blue roof marker at
  chassis +0.95 m. The Gator's roof is at about +1.97 m, so the marker would sit inside the cab. That only affects
  rendering: the planner's map is vehicle-free.

**Frozen runner: `scripts/traverse_fdm_rgbd_diverse_chrono.py`** (imported by `gen_collect.py` and
`gen_collect_ext.py`).
- `DRIVER` constants (35-39), `make_driver` (98-112) and the spawn at +0.75 m (162).
- Tyre radii are captured before the first `Synchronize` (189).
- The loop uses the name `hmmwv` for the wrapper at lines 172-173, 214, 227, 229, 246, 289, 306, 327 and 328.
- Only wrapper methods that `veh.Gator` also has are called, so **no edit is needed if the factory returns a Gator
  under the same attribute name.**

**Hooks and provenance gates.**
- The hook strings that `gen_collect.adapted_function` (166-190) and `gen_collect_ext.replacement_adapter` (123-146)
  assert by count include the literal `hmmwv, system, terrain = scene.hmmwv, ...`. Keeping the attribute name keeps
  the hooks valid.
- `gen_collect.SOURCE_FILES` (27-30) hash-gates `scene.py` and `hmmwv_data.py` against `source/source_manifest.json`,
  so a Gator campaign needs a new source snapshot and manifest.
- Runtime fingerprint gates require `"/vehicle/hmmwv/"`:
  - `gen_collect.py:274`
  - `gen_collect_ext.py:545`
  - `collect_traverse_f104.py:272`

  The cluster fingerprint `experiments/fdm_f104_50h_20260909/pilot_runtime_412394.json` lists 108 files, including
  HMMWV data, and must be regenerated with the Gator data.

**Other runners that build the HMMWV scene directly:**
- `gen_mission_runner.py:79-159`;
- `nav_runner.py:107-135`;
- `ga_approach.py` (hard-codes HMMWV JSON paths at line 78).

The single-goal closed-loop runners go through the collectors, so the rigid closed loop inherits whatever
`scene.py` does:
- `gen_runner.py:24-27` and `gen_runner_g.py:21` go through `gen_collect.py` / `gen_collect_ext.py`;
- `n2_runner.py:6` and `n2_ext_runner.py:6` go through `source_v1/scripts/collect_traverse_f104.py`.

### 3.2 Soil (CRM) coupling

**Soil defaults, `scripts/crm_collect.py:36-46`.**
- `tire_mesh = "hmmwv/hmmwv_tire_coarse_closed.obj"`: **one mesh for all four wheels**.
- The production config (cluster `crm_f104_20260916/configs/crm_main.json`) uses 0.08 m spacing, 0.24 m soil depth
  and a **1 ms** step. The code default is 0.5 ms.

**Soil construction, `build_crm` (`crm_collect.py:69-132`).**
- Barzilai-Borwein solver, Euler implicit linearised stepper, 4 multibody threads.
- `CRMTerrain(system, 0.08)`, `RegisterVehicle`, soil properties (density 1700, E 1e6, friction 0.8, cohesion 5 kPa).
- **Lines 111-116:** one `ChBodyGeometry` carrying the tyre mesh, with interior point `VNULL`, added to **every**
  spindle via `terrain.AddRigidBody(wheel.GetSpindle(), geometry, False)`.
- Active domain 2 x 2 x 1 m; `Construct` from the same arena image, 0.24 m deep, with side walls.
- Only the wheel spindles are coupled to the soil (**the chassis is not**, line 180).
- `crm_collect_ext.py:125` reuses `base.build_crm`, so one change covers both soil collectors.

**Vehicle setup in the soil collectors.**
- `crm_collect.py:177-184` and `crm_collect_ext.py:114-121`: spawn at +0.75 m, rigid-mesh tyres, no chassis
  collision, `create_hmmwv`.
- Wheel forces come from `terrain.GetFsiBodyForce(spindle)` (`crm_tire_fields`, 135-153). This is generic.

**Soil breakthrough rule** (`crm_collect.py:285-293`, `crm_collect_ext.py:306-313`).
- A wheel counts as sunk when radius - (spindle z - image height) > 0.24 + 0.06 m for 5 consecutive frames.
- Geometrically this means "tyre bottom 6 cm below the soil floor", which does not depend on the vehicle. For the
  Gator the belly reaches the original surface at about 0.14 m sinkage, long before 0.30 m.

**Launch check** (`crm_collect.py:421-422`): chassis reference 0 to 1.2 m above the terrain image. OK for the Gator,
whose reference sits at 0.27 m.

### 3.3 Route follower and speed controller

- **Follower construction** (`traverse_fdm_rgbd_diverse_chrono.py:98-112`, twin in `gc_control.make_follower:652-697`
  used by the branch and continuation modes): waypoints thinned to 2 m, z = image height + 0.5 m, `ChBezierCurve`,
  look-ahead 5 m, steering PID (0.8, 0, 0), speed PI (0.6, 0.05, 0).
  - The 3-D closest-point search runs from a sentinel at chassis-reference height. The Gator's reference is at
    ground + 0.27 m, the HMMWV's at ground + 0.59 m, so the vertical offset to the path grows from 0.09 m to 0.23 m.
    This is minor.
- **Steering rate clamp, 2 full-scale units per second:**
  - frozen runner line 224;
  - `crm_collect.py:236`, `crm_collect_ext.py:252`;
  - `gc_control.STEER_RATE_PER_FRAME = 0.1` (line 62).
- **Parking** when within 3 m of the last waypoint:
  - frozen runner 217;
  - `crm_collect.py:229` / 318;
  - `crm_collect_ext.py:219` / 349.

  The goal radius is 2.5 m (frozen runner 308).
- **Throttle and brake split** inside Chrono (`ChPathFollowerDriver.cpp:94-108`, throttle threshold 0.2): the same
  for both vehicles.

### 3.4 The 17-column state and its readers

- **Layout.** `src/nedm/training/constants.py:3-54`, preset `tire_normal_force_omega_pt`. Columns:

  | Columns | Content |
  |---|---|
  | 0-6 | Forward and sideways speed, roll, pitch, roll rate, pitch rate, yaw rate |
  | 7-10 | Tyre vertical load per wheel |
  | 11-14 | Spindle speed per wheel |
  | 15 | Engine speed |
  | 16 | Engine output torque |

  The wheel order FL, FR, RL, RR comes from `WHEEL_SPECS`. **The layout survives unchanged.**
- **Scales change:**
  - vertical loads 1.8 to 3.1 kN against 6.1 to 7.6 kN;
  - spindle speeds about 1.5 to 1.65 times higher at the same ground speed, and front and rear differ by 11%;
  - engine speed 0 to 390 rad/s with no gear information;
  - engine torque 0 to 200 N·m, and 0 above 366 rad/s;
  - recorded power peaks at 14 kW against 81 kW.
- **Readers** (all index-based, so the format is compatible):
  - `gc_control.OBSERVABLE_COLS` (68);
  - `gen_collect_ext.HIST_STATE_COLS` (102), `crm_collect_ext.HIST_STATE_COLS` (65);
  - `f104_n2_dataset.one` (84-110): column 0 is forward speed, action column 1 is throttle, and the anchor state goes
    into the context vector;
  - the history encoders of the generalist and CRM-improvement models (`ga_build_mixed.py`, `ci_train.py`).
- **Retraining.** The deployed rigid model (`gen_planner.geom_ctx:149`) uses only 5 geometric context numbers and no
  state. Every state-history model needs retraining or refitted normalisers. Anything that uses past actions also
  needs retraining, because the steering command maps to about 44° on the Gator against 30° on the HMMWV.

### 3.5 Stop rules, termination and labels

**Stop policy (`gen_collect.StopPolicy`, lines 54-154).**
- Arena exit when the chassis reference passes \|x\| or \|y\| > 40 m (123-125).
- Prolonged-blockage stop: 2 s windows with maximum pairwise distance ≤ 0.25 m and throttle > 0.3; minimum 24 s,
  2 s confirmation, 8 s tail.
- Launch checks: speed ≤ 1 m/s, \|roll\| and \|pitch\| ≤ 20°, yaw error ≤ 10°, position error ≤ 1 m (109-110).
- None of these depends on vehicle size.

**Other terminations:**
- Rollover at \|roll\| or \|pitch\| > 60° (frozen runner 314; `crm_collect.py:298`). The Gator's static tilt limit
  is about 43°, so the "past 30° tilt" label of the tilt-aware model will fire more often.
- Soil breakthrough: see 3.2.

**Labels (`f104_n2_dataset.py:84-110`).**
- `unsafe` = did not reach the goal, OR (forward speed < -0.10 m/s with throttle > 0.3), OR forward speed < -0.30 m/s
  at any time after 1 s.
- `event_idx` also uses 1 s of speed below 0.3 m/s with throttle > 0.3.
- These are absolute speed thresholds. The Gator's creep on downhill parking stretches (rear-only brakes) can cross
  -0.30 m/s. This can happen during the goal-approach parking band (3 m) when the goal radius is not yet reached.

**Launch footprint gate** (`generate_traverse_f104_collection.py:47-64`): a 6 x 3 m rectangle with height range
≤ 0.65 m, maximum grade ≤ 12°, fitted grade ≤ 6°. HMMWV-sized, and conservative for the Gator. Keep it, so the start
positions stay identical.

### 3.6 Route constraints

**Route validator** (`src/nedm/traverse/fdm_mppi.py:19-37`, `validate_reference` 121-168).
- Curvature limit 0.125 /m, acceleration 1.5 m/s², deceleration 2.0 m/s².
- Footprint used for the arena-boundary check: half-width 1.3 m, half-length 2.6 m (HMMWV-sized).
- `gc_control.planner_cfg:421-424` uses maximum speed 6, curvature 0.125 and arena half-width 40.

**Route sampler** (`scripts/f104_n2_sampler.py`).
- Acceleration 1.5 and deceleration 2.0 m/s², speed bounds 0.5 and 6.0 m/s (23-24).
- Curvature cap 0.125 in `lateral_profile` and `sample_one` (58, 67).
- Fixed anchor routes at 2 / 4 / 6 m/s with lateral offsets 0 / -4 / +4 m (81).

**Designed routes** (`generate_traverse_f104_collection.speed_profile:67-77`): constant 2 / 4 / 6 m/s or a smooth
2-6-2 profile, capped by a stopping cone sqrt(4 (L - s)).

**Continuation speed floor** (`gc_control.speed_floor_from:520`): maximum speed 6.

**Against the Gator:**
- Curvature 0.125 uses 36% of the Gator's lock, against 85% of the HMMWV's.
- 6 m/s is below its 8.2 m/s top speed.
- Acceleration 1.5 m/s² is well below its roughly 4 m/s² traction limit on flat ground.
- **Deceleration 2 m/s² is not achievable on steep descents.** Downhill gravity is 3.4 m/s² on 20° and 4.9 m/s² on
  30°, while the rear-only brakes give 3.3 m/s² on flat ground.

### 3.7 Risk-model inputs that assume vehicle geometry

- **Corridor** (`f104_n2_dataset.py:12`, `sensor_dataset.py:22`): 96 stations x 32 lateral samples over ±6 m, a
  0.39 m lateral step.
  - The HMMWV's wheel tracks are 1.82 m apart (about 4.7 samples); the Gator's are 1.12 to 1.24 m (about 3 samples).
  - The format is unchanged. A narrower ±4 m corridor would resolve the Gator's wheel paths better, but it would no
    longer be "the same planner".
- **Vehicle footprint mask** (`scripts/vehicle_corridor.py:15`, half-length 2.6 m and half-width 1.3 m, plus a 1.5 m
  margin in `nav_*`): only matters for the sensor-driven navigation line. It is conservative for the Gator. The
  Gator's body is asymmetric (-2.21 to +1.24 m around the reference), but the mask covers it.
- **Model context** (`gen_planner.geom_ctx`): goal offset, distance, start heading and route length. No vehicle
  parameters.

## 4. Chrono's own soil demos with wheeled vehicles

Demos live in `/home/harry/chrono/src/demos`. **No demo couples a Gator to CRM.** The Gator appears only in:
- `python/vehicle/demo_VEH_Gator.py`: TMEASY tyres, `BrakeType_SHAFTS`, rigid terrain;
- `vehicle/wheeled_models/demo_VEH_Gator_Incline.cpp`: a 20° incline acceleration test, TMEASY tyres;
- sensor, VSG and checkpoint demos.

| Demo | Vehicle | How it is attached to CRM | Spacing, step |
|---|---|---|---|
| `vehicle/terrain/demo_VEH_CRMTerrain_WheeledVehicle.cpp` (lines 70-74, 115, 135-144, 205, 402-460) | Polaris RZR from JSON; rigid tyre JSON (a deformable tyre option exists) | Tyre collision mesh `Polaris/meshes/Polaris_tire_collision.obj` (closed, 0.20 to 0.34 m); `terrain.AddRigidBody(wheel->GetSpindle(), geometry, false)` per wheel. Deformable tyres use `terrain.AddFeaMesh(mesh, false)`. `RegisterVehicle`; active domain 0.8 m. | 0.04 m; 5e-4 s (1e-4 s for deformable tyres) |
| `python/vehicle/demo_VEH_CRMTerrain_WheeledVehicle.py` (lines 28-55, 78-100, 179-231) | Same Polaris, Python twin | Same | 0.04 m, 5e-4 s, soil density 1700, cohesion 5 kPa, E 1e6 (**identical soil to ours**) |
| `vehicle/terrain/demo_VEH_CRMTerrain_TrackedVehicle.cpp` (411-510) | M113 (tracked) | Each track shoe is a rigid FSI body | 0.02 m, 5e-4 s |
| `vehicle/terrain/demo_VEH_CRMTerrain_MovingPatch.cpp` | Generic body | `ConstructMovingPatch` | 0.04 m, 5e-4 s |
| `vehicle/test_rigs/demo_VEH_WheelTestRig_CRM.cpp` (56-60, 154) | Polaris rigid-mesh tyre on a test rig | Single wheel | 0.02 m |

The pattern is exactly our `build_crm`. The differences are that the demos run at a finer spacing, 0.04 m against
our 0.08 m, and use a watertight tyre mesh.

## 5. Adaptation plan

### 5.1 Minimum code changes

Make them in a new branch and a new source snapshot. Every file named below except the soil collectors is under the
source-hash gate.

1. **`src/nedm/hmmwv_data.py` `create_hmmwv`, or a new `create_vehicle` it delegates to.** Add a `model == "Gator"`
   branch:
   - `veh.Gator()` → `SetContactMethod(SMC)`, `SetChassisFixed`, `SetInitPosition`, `SetInitFwdVel`;
   - `SetChassisCollisionType(HULLS)` on rigid, `NONE` on soil;
   - `SetDrivelineType` (open-differential `RWD` or limited-slip `SIMPLE`; see 5.2), `SetBrakeType(SIMPLE)`,
     `EnableBrakeLocking(...)`;
   - `SetTireType(TMEASY or RIGID_MESH)`, `SetTireStepSize`, `Initialize()`;
   - then the same visualisation-off and collision-system lines.

   Return the wrapper unchanged, so every `scene.hmmwv` / `hmmwv.X()` call site keeps working.
2. **`src/nedm/traverse/scene.py` `build_config`.** Pick the vehicle from one switch and record it in
   `simulation_provenance`. Options: an environment variable such as `NEDM_VEHICLE=gator`, or a `vehicle` key in
   `case.json`. The environment variable avoids touching the frozen runner's signature.

   Add a per-vehicle **spawn offset**: +0.35 m for the Gator, 0.75 m for the HMMWV. The +0.75 m literal sits at the
   three call sites (`traverse_fdm_rgbd_diverse_chrono.py:162`, `crm_collect.py:177`, `crm_collect_ext.py:114`), so
   either change those lines or make `build_config` add `spawn_dz - 0.75`.

   Optionally move the roof marker height (`build_scene:371`); rendering only.
3. **Provenance gates.** Let `gen_collect.py:274`, `gen_collect_ext.py:545` and `collect_traverse_f104.py:272`
   accept `"/vehicle/gator/"`. Write a Gator runtime fingerprint JSON (same format as `pilot_runtime_412394.json`,
   listing `data/vehicle/gator/*`), then a new `source/` + `source_manifest.json` for the campaign root.
4. **Soil: `crm_collect.py` defaults (line 45) and `build_crm` (111-116).** Replace the single `tire_mesh` with a
   per-axle wheel geometry. Two candidates:
   - **(a) Cylinder markers, cost-neutral** (recommended for the night): `chrono.CylinderShape(VNULL,
     ChVector3d(0, 1, 0), r_bce, width)` per axle, with r_bce chosen so the settled sinkage matches the HMMWV's -0.02
     to -0.04 m. From the smoke, roughly r - 0.07 m: about 0.215 m front and 0.245 m rear. **UNVERIFIED; needs a
     5-minute flat-patch calibration.**
   - **(b) Per-axle Gator meshes at 0.04 m spacing**: the faithful Chrono-demo setting. About 28 M particles for the
     80 x 80 m arena against 4.0 M. The step probably has to drop to 0.5 ms. The tiny-patch real-time factor fell
     from 0.77 to 0.30. **Full-arena cost on the MI GPUs is UNVERIFIED** and likely several times the HMMWV's.
   - 0.06 m spacing with meshes is not enough: 4 or 5 empty 10° sectors per wheel remain.

   Optional, and recommended if soil bogging is the question: couple the chassis underbody to the soil. Add a box
   geometry to `terrain.AddRigidBody(vehicle.GetChassisBody(), ...)`, or at least log the lowest hull point against
   the terrain image in `crm_extra.npz`, plus a "high-centred" diagnostic.
5. **Follower, `make_driver` and `gc_control.make_follower`.**
   - First pass: **keep the frozen gains**. They work and keep "same planner, same controller" clean.
   - Optional second arm: steering gain 0.35 to 0.4, which gives about 40% lower cross-track error.
   - Optionally set path z = ground + 0.27 m.
6. **Nothing to change** in the route sampler, the validator limits, the corridor, the labeller, the trainer
   (`f104_n2_train.py` / `f104_n2_final.py` / `f104_n2_deploy.py`, or `crm_train.py`) or the planner module
   (`gen_planner.py`). Retrain from scratch on Gator rows. Recompute normalisers for any history or state model.

### 5.2 What to choose or re-tune, in priority order

1. **Spawn height: +0.35 m.** At +0.75 m the anchor state is taken mid-bounce: vertical velocity -1.23 m/s at 0.8 s.
2. **Soil tyre geometry and marker offset** (5.1 item 4).
3. **Brake setup.** The default brake without locking creeps downhill (1.2 to 8.6 m in 6 s on 15 to 30° slopes).
   Locking holds up to 25°. The shafts brake runs away on 30°. Pick `SIMPLE` with locking, or keep the default and
   accept creep. The HMMWV also creeps 1.3 to 1.8 m.
4. **Driveline.**
   - `RWD` (open differential, like the HMMWV's differentials): tested on flat and 20° rigid ground (top speed
     8.0 m/s, 4.4 to 4.7 m/s up 20°).
   - `SIMPLE` (default, limited slip up to 2:1): less one-wheel spin on soil.

   Pick one before collecting. For "same failure physics" as the HMMWV study, `RWD` is the closer analogue.
5. **Follower gains:** optional, see 5.1 item 5.
6. **Keep all route constraints unchanged**: curvature 0.125, knots at 2 / 4 / 6 m/s, acceleration 1.5 and
   deceleration 2.0 m/s², footprint gate. Record that on descents steeper than about 15 to 20° the Gator cannot
   follow a 2 m/s² slow-down.

### 5.3 Matching "the same amount of data"

**How much the HMMWV had.**
- Rigid: 36,199 driven routes, about 200 simulated hours, over 2,700 start/goal groups (`docs/progress.md:366-367`).
  The deployed model trained on 31,851 routes.
- Soil: 15,235 validated episodes, 91.5 simulated hours (`collect_v1`).

**Proposal: re-drive the identical route files with the Gator.**
- Rigid: route files from waves `production_v2`, `v3` and `v4` on the cluster.
- Soil: `collect_v1` tasks.
- Every route then has an HMMWV and a Gator outcome, which gives a paired comparison.
- Match on route count. Report simulated hours too; they will differ, because the Gator accelerates faster and fails
  differently.
- Note that the `v4` routes were proposals of the *HMMWV* planner, so they are off-policy for the Gator.

**Cost.**
- Rigid: the Gator smoke ran 2.6 times faster than the HMMWV's on a flat patch. On the heightmap arena this is
  **UNVERIFIED**. For reference, the HMMWV's 66.4 h wave took 13.6 wall-minutes on about 2,000 workers.
- Soil at 0.08 m spacing: expect about the HMMWV's cost (2 h 35 min wall for 91.5 h on about 111 GPUs), because cost
  is set by the per-step overhead and the active domain.
- Soil at 0.04 m spacing: **UNVERIFIED**, likely several times more.

**Test.** Use the existing held-out closed-loop suites driven by the Gator:
- rigid: the 300 hill/crater groups of `night2_v1/closed_haz`;
- soil: the 200 pairs of `crm_f104_v1/eval_v1`, or the crm_improve suite.

Arms: Gator-trained planner, frozen HMMWV-trained planner, straight 6 m/s, fixed 2 m/s. Keep the paired arms in one
cluster job, because rigid runs on the AMD cluster reproduce only on the same node.

### 5.4 Risks

1. **Grounding the body on rigid ground.** Clearance is 0.14 to 0.17 m, departure about 21° and breakover about 22°,
   with chassis collision on. Crater lips and crests on f104 (2% of cells above 25°, large grade changes) can
   ground the Gator. The failure mechanism then differs from the HMMWV's (wheel lift + open differential +
   stalls), so the risk model's slope and elevation channels may explain less of it. Watch the recorded maximum
   chassis contact force in the rigid pilot.
2. **Rear-only brakes.**
   - Standstill creep and weak downhill braking.
   - Speed overshoot on descents, which feeds rollover and exiting the arena.
   - Parking creep inside the 3 m parking band can produce "slid backwards" labels.
3. **Rear-wheel-drive traction on soil.** Only the rear axle is driven, carrying about 57% of the weight. The
   HMMWV's soil failure mechanism was a stall at full throttle, then one wheel spinning and digging out the layer.
   With rear-wheel drive this may happen at much lower grades. Expect a higher soil failure rate than the HMMWV's
   68% on the designed routes, possibly so high that every route fails and the labels are useless. **UNVERIFIED;
   the soil pilot is essential.**
4. **Soil tyre representation.** At 0.08 m the Gator meshes are ragged (94/116 markers, empty sectors). Cylinders
   inflate the effective radius by about 0.1 m unless calibrated. Going to 0.04 m costs particles and time.
5. **Belly not coupled on soil.** Clearance is below the 0.24 m layer depth, so a bogged Gator can sink its belly
   through the soil, and the "breakthrough" rule fires only at 0.30 m. The labels on soil are optimistic.
6. **Grip inflated by a Chrono artefact on rigid ground.** The TMEASY scale is 1.5 for the Gator against 1.125 for
   the HMMWV. Together with the full-torque simple engine, the rigid Gator climbs 30° at 2.8 m/s. Rigid failures may
   become rare (weak labels) or dominated by grounding and rollover.
7. **Rollover.** The static tilt limit is about 43° against about 49°, and the vehicle is lighter. Crossing slopes
   at 6 m/s is riskier.
8. **Declared getters are wrong.** `GetWheelbase` (2.776 against a real 1.94), `GetMaxSteeringAngle` (25° against a
   measured about 44°) and `GetMinTurningRadius` (7.6 against 2.8 m). Harmless for the PID follower. They break any
   pure-pursuit or XT controller, and any analytic model that trusts them.
9. **Engine and energy columns have a different meaning:** one gear, no torque converter, torque cut above
   366 rad/s. Energy heads and the analytic energy model need recalibrating. They are not transferable across
   vehicles.
10. **Rich telemetry has 24 NaN suspension fields.** Harmless, unless a downstream script asserts that all telemetry
    is finite.
11. **Unequal front and rear tyre radii.** Any feature that compares wheel speeds, such as slip from speed ratios,
    must use per-axle radii. `capture_row` already uses a per-wheel `tire_radii`.
12. **The premise "6 m/s is unreachable" does not hold on flat rigid ground.** It holds on climbs above about 15°
    for 6 m/s and above about 20° for 4 m/s. The HMMWV in its collector setup is similarly capped on climbs from
    rest (about 4.5 m/s in first gear).

### 5.5 Pilot sequence (before committing the overnight budget)

1. **Local rigid check (minutes).** Run `gen_collect_ext.py --local --mode native` with `NEDM_VEHICLE=gator` on 3
   f104 cases x the 4 designed profiles. Check `native_height_check.json`, `initial_state_validation.json`,
   `outcome.json`, and that the recorded state has the right shape and is finite.
2. **Local soil calibration (minutes, RTX 5090).** On the flat-patch smoke, compare cylinder markers at r - 0.05 /
   0.07 / 0.09 m with the mesh at 0.04 m and the HMMWV reference: settled sinkage, then the speed after a 2 s
   full-throttle drive. Then pick the geometry.
3. **Cluster rigid pilot.** 24 groups x 12 designed routes, HMMWV and Gator on one node. Measure the failure rate by
   speed profile. Aim for a spread between about 20% and 60%; the HMMWV's rigid pilot rates are in
   `fdm_f104_50h_20260909`. Also measure chassis-contact frequency, creep events and cost per simulated hour.
4. **Cluster soil pilot.** 24 groups x 6 routes, as in the CRM night-1 pilot. Measure the failure rate by profile,
   the breakthrough rate, the "belly below surface" fraction and the real-time factor. If failure is above about 90%
   everywhere, consider the limited-slip `SIMPLE` driveline, or state that the Gator cannot do this arena on this
   soil.

## 6. Not verified, or open

- Gator behaviour on the actual f104 heightmap (rigid or soil); all rigid runs used a plane patch, and the soil run a
  14 x 3 m box.
- Full-arena soil cost and memory at 0.04 m, and at 0.08 m with cylinders.
- Whether the Gator is numerically stable at the production 1 ms soil step over a full 120 s episode (a 2 s run was
  fine).
- Soil failure rates and slope limits for rear-wheel drive.
- The small inconsistency between engine speed and rear-wheel speed / 0.07 (3 to 9%).
- The clearance and ramp angles are static hull-vertex estimates, not measured scraping events.
- Tyre types other than `RIGID`, `RIGID_MESH` and `TMEASY` leave no tyres: read from source, not run.
- The HMMWV slope numbers are from rest, where it stays in first gear. With momentum from a flat approach, as in real
  routes, it may upshift and do better.

## Appendix A. Commands

```bash
# local availability (source build and conda)
PYTHONPATH=/home/harry/chrono/build/bin /usr/bin/python3.12 -c "import pychrono.vehicle as veh; print([n for n in dir(veh) if n.startswith('Gator')], 'CRMTerrain' in dir(veh))"
/home/harry/miniconda3/envs/nedm/bin/python -c "import pychrono.vehicle as veh; print([n for n in dir(veh) if n.startswith('Gator')])"

# cluster, login node, import only (no jobs)
ssh amd 'bash -l -s' <<'EOF'
source /work1/dannegrut/harry/nrd/env.sh; cd /work1/dannegrut/harry/nrd
for B in chrono-build chrono-build-fsi; do export CHRONO_BUILD=/work1/dannegrut/harry/nrd/$B; nrd_pychrono
  $NRD_PYTHON - <<'PY'
import pychrono.vehicle as veh; print(sorted(n for n in dir(veh) if n.startswith("Gator")), "CRMTerrain" in dir(veh))
try:
    import pychrono.fsi as fsi; print("fsi ok", hasattr(fsi, "SoilProperties"))
except Exception as e: print("fsi import failed:", e)
PY
done
EOF
# note: running python from /tmp on the login node fails with a circular 're' import; cd to $WORK first.

# local smokes (scratch copies in /tmp/s2_gator/)
cd /tmp/s2_gator
PYTHONPATH=/home/harry/chrono/build/bin /usr/bin/python3.12 smoke.py gator    # 17 s wall; smoke_gator.json
PYTHONPATH=/home/harry/chrono/build/bin /usr/bin/python3.12 smoke.py hmmwv    # 45 s wall; smoke_hmmwv.json
PYTHONPATH=/home/harry/chrono/build/bin /usr/bin/python3.12 follow.py         # 38 s wall; follow.json
PYTHONPATH=/home/harry/chrono/build/bin /usr/bin/python3.12 crm_smoke2.py gator mesh 0.08 0.001   # also: gator cyl, hmmwv mesh, gator mesh 0.04 0.0005
/home/harry/miniconda3/envs/nedm/bin/python bce.py /home/harry/chrono/data/vehicle/gator/gator_tire{F,R}_coarse.obj /home/harry/chrono/data/vehicle/hmmwv/hmmwv_tire_coarse_closed.obj
```

## Appendix B. `smoke.py` (rigid smoke, verbatim core)

```python
import json, math, sys, time
import numpy as np
import pychrono as ch
import pychrono.vehicle as veh
DATA = '/home/harry/chrono/data/'
ch.SetChronoDataPath(DATA); (getattr(veh, 'SetVehicleDataPath', None) or veh.SetDataPath)(DATA + 'vehicle/')
STEP, TIRE_STEP = 2e-3, 1e-3

def build(kind, slope_deg=0.0, heading_up=True, tire='TMEASY'):
    s = math.radians(slope_deg); patch_rot = ch.QuatFromAngleY(-s)          # +x climbs
    rot = patch_rot * ch.QuatFromAngleZ(0.0 if heading_up else math.pi)
    z0 = 0.75 if kind == 'hmmwv' else 0.55
    pos = ch.ChVector3d(-z0 * math.sin(s), 0, z0 * math.cos(s))
    if kind == 'gator':
        v = veh.Gator(); v.SetContactMethod(ch.ChContactMethod_SMC); v.SetChassisCollisionType(veh.CollisionType_NONE)
        v.SetTireType(getattr(veh, 'TireModelType_' + tire)); v.SetTireStepSize(TIRE_STEP)
    else:  # the collectors' HMMWV (scene.build_config)
        v = veh.HMMWV_Full(); v.SetContactMethod(ch.ChContactMethod_SMC); v.SetChassisCollisionType(veh.CollisionType_NONE)
        v.SetEngineType(veh.EngineModelType_SHAFTS); v.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SHAFTS)
        v.SetDriveType(veh.DrivelineTypeWV_AWD); v.SetSteeringType(veh.SteeringTypeWV_PITMAN_ARM)
        v.SetTireType(getattr(veh, 'TireModelType_' + tire)); v.SetTireStepSize(TIRE_STEP)
    v.SetChassisFixed(False); v.SetInitPosition(ch.ChCoordsysd(pos, rot)); v.Initialize()
    sysm = v.GetSystem(); sysm.SetCollisionSystemType(ch.ChCollisionSystem.Type_BULLET)
    terrain = veh.RigidTerrain(sysm)
    mat = ch.ChContactMaterialSMC(); mat.SetFriction(0.9); mat.SetRestitution(0.01); mat.SetYoungModulus(2e7)
    terrain.AddPatch(mat, ch.ChCoordsysd(ch.ChVector3d(0, 0, 0), patch_rot), 400.0, 60.0); terrain.Initialize()
    return v, sysm, terrain
# run(kind, T, ctrl, slope, up): 0.8 s braked settle, then ctrl(t, speed) -> (steer, throttle, brake) every 2 ms;
# logs speed, pose, yaw rate, engine speed/torque, spindle omega (angvel . spin axis), driveline spindle torque,
# road-wheel steer angle (spindle Y axis in chassis frame), tyre Fz.
# Scenarios: A flat full throttle 30 s; B full lock at 2 and 4 m/s (P speed hold); C climbs 10-30 deg from rest 12 s;
# D full brake facing downhill 15-30 deg 6 s; E full throttle 12 s then full brake.
```

## Appendix C. `crm_smoke.py` (tiny soil instantiate, core)

```python
# vehicle as in Appendix B but RIGID_MESH tyres, tyre step = CRM step; spawn z = 0.24 + 0.55 (Gator) / 0.75 (HMMWV)
system.SetSolverType(ch.ChSolver.Type_BARZILAIBORWEIN); system.SetTimestepperType(ch.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
terrain = veh.CRMTerrain(system, SP); terrain.SetStepSizeCFD(DT); terrain.RegisterVehicle(vehicle)
# soil + SPH parameters = crm_collect.CRM_DEFAULT / crm_main.json
for ia, axle in enumerate(vehicle.GetAxles()):
    for wheel in axle.GetWheels():
        g = ch.ChBodyGeometry()
        if wheel_geom == 'mesh':
            f = 'hmmwv/hmmwv_tire_coarse_closed.obj' if kind == 'hmmwv' else ('gator/gator_tireF_coarse.obj' if ia == 0 else 'gator/gator_tireR_coarse.obj')
            g.coll_meshes.append(ch.TrimeshShape(ch.VNULL, ch.QUNIT, veh.GetVehicleDataFile(f), ch.VNULL))
        else:
            t = wheel.GetTire(); g.coll_cylinders.append(ch.CylinderShape(ch.VNULL, ch.ChVector3d(0, 1, 0), t.GetRadius(), t.GetWidth()))
        terrain.AddRigidBody(wheel.GetSpindle(), g, False)
terrain.SetActiveDomain(ch.ChVector3d(2., 2., 1.)); terrain.SetFreeFlowDuration(0.)
terrain.Construct(ch.ChVector3d(14., 3., 0.24), ch.ChVector3d(7., 0, 0), fsi.BoxSide_ALL & ~fsi.BoxSide_Z_POS)
terrain.Initialize()   # then terrain.GetNumBCE(spindle); 0.8 s brake, 2 s full throttle via terrain.Advance(DT)
```
