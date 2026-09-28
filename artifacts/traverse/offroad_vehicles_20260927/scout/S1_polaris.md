# S1: bringing Chrono's Polaris into the f104 soil pipeline (scout, 2026-09-27)

Scope: scout S1 of the offroad-vehicles study. It covers how the Gator was switched in, where a new vehicle must be
added, what Chrono's Polaris model is, how to build it, local smokes, and an integration recipe with checklist.

What I touched:
- No repository file and no artefact folder was changed.
- Everything I wrote is under `artifacts/traverse/offroad_vehicles_20260927/scratch/S1/`: scripts, JSON variants,
  outputs and a small private vehicle-data folder `vehdata/`.
- No cluster job was submitted. On the login node I only ran listings, hashes and `import` checks. Two temporary
  hash lists I briefly wrote to the login node's `/tmp` were deleted straight away.
- Local Chrono runs:
  - rigid-ground smokes, each under 90 s;
  - soil runs under `flock /tmp/luffy_crm.lock`, about 7 min of soil wall time in total (time spent waiting for the
    lock not counted).
- The soil smokes built their small test arenas in `/tmp` (Python temporary folders); I deleted them afterwards.
- `NEDM_VEHICLE` was never set.

Build used for the soil runs: the local source build (`/usr/bin/python3.12`, `PYTHONPATH=/home/harry/chrono/build/bin`,
Chrono `a92c6f72`). The Gator calibration used the same build, so the numbers are comparable. The rigid runs used the
conda `nedm` python. Anything I did not measure is marked **UNVERIFIED**.

Plain-language names used below:
- **CRM**: Chrono's particle (SPH) deformable soil.
- **Markers**: the soil-side particles that stand for a solid body, such as a wheel.
- **Reference point**: the chassis point whose position the collectors record and test.
- **Launch check**: the collector's test, after the 0.8 s braked settle, that the vehicle started validly.
- **Soil breakthrough**: the stop rule for a wheel that has dug through the whole soil layer.

## 0. Bottom line

1. **The Gator switch is a run-time swap and can be copied.** `scripts/ag_vehicle.py` plus two wrapper collectors
   replace a few module functions (vehicle factory, config builder, soil-wheel builder, JSON writer) before the
   unchanged collectors call them (section 1).
   - A Polaris needs a new switch module and wrapper (`ov_*`), because `ag_vehicle.py` is frozen and hard-codes
     `("hmmwv", "gator")`.
   - About 20 downstream scripts hard-code `'gator'` or `'gator__'` (section 2).
2. **Chrono's Polaris is a JSON model** (`data/vehicle/Polaris/Polaris.json`, a Polaris MRZR).
   - 1,378 kg, 4-wheel drive with limited-slip differentials, 0.330 m tyres, 2.715 m wheelbase, 1.232 m track.
   - Engine map peak about 77 kW (HMMWV 81 kW, Gator 14 kW); 6-speed automatic with no torque converter; brakes on
     all four wheels.
   - The data files are byte-identical locally (conda and source build) and in both cluster builds. All needed
     Python bindings import in both cluster builds. Only `chrono-build-fsi` has soil (`CRMTerrain`, `pychrono.fsi`),
     the same as for the Gator.
3. **Blocker 1: the Polaris reference point is at the front axle and about 4-9 cm below the ground.** This is the
   JSON's own chassis frame.
   - The soil launch check requires the reference to be 0 to 1.2 m above the terrain (`crm_collect.py:422`,
     `crm_collect_ext.py:589`). **Every Polaris soil episode would be refused.** Measured on flat soil: -0.037 to
     -0.087 m.
   - **Fix (verified):** a re-framed copy of the two top-level JSON files puts the reference at mid-wheelbase,
     2.3 cm above the axle line. That is the HMMWV's convention (mid-wheelbase, 2.6 cm above the axle line; the
     Gator's is also mid-wheelbase).
   - The physics does not change: the centre-of-mass and wheel paths match the stock model to 1e-6 m through the
     settle, and to 1.6 cm after 75 m at 24 m/s.
   - The launch check then reads +0.385 m on f104 soil. The frozen path follower then tracks as well as on the HMMWV:
     RMS cross-track 0.49-0.63 m, against 0.80-0.90 m with the front-axle reference.
4. **Blocker 2 (modelling): Chrono's generic "simple" driveline, used by `Polaris.json`, does not conserve power.**
   - It multiplies wheel speed by the final-drive ratio 0.25 on the way to the engine but divides torque by it
     (`ChSimpleDriveline.cpp:106,115`, commit `dfff7a809`, Dec 2024; the cluster source has the same code). The engine
     sees 1/16 of its true speed. Measured: engine speed / (wheel speed x overall ratio) = 0.065.
   - Consequences: the gearbox never leaves first gear, the engine sits near 190 N·m, and the model reaches 58 m/s
     after 10 s on flat ground.
   - Inside our 0-6 m/s band this acts like a one-gear vehicle with about 2.8-3.0 kN·m at the wheels (0.63-0.67 g),
     about 50 kW at 6 m/s. That is below the map's 77 kW, so it is plausible here but not the engine map's physics.
   - The engine-speed, engine-torque and power/energy columns are not physical. Power is under-reported 16 times.
   - A power-correct alternative is already in Chrono's Polaris folder: `Polaris/Polaris_4WD.json` (shafts driveline,
     open differentials). On rigid climbs of 20° and more it shows wheelspin, then upshifts to 6th gear, then rolls
     back. This is a gearbox artefact: no torque converter, and shifting reads engine speed only
     (`ChAutomaticTransmissionSimpleMap.cpp:47,55`). On soil it climbed about as well as the stock model.
5. **Soil wheel: use a calibrated cylinder of radius 0.25 m (stock 0.330 m minus 0.08 m, stock width 0.2121 m).**
   - Chrono's own Polaris tyre collision mesh is watertight: 0 open edges, 0 non-manifold edges, consistent
     orientation, 1 part.
   - But at our 0.08 m particle spacing it gets only 83 markers per wheel, and 8 of 36 10° sectors are empty: a lumpy
     wheel. Chrono's demo runs at 0.04 m.
   - The 0.25 m cylinder (152 markers) reproduces the HMMWV production mesh's sinkage under the Gator calibration
     protocol:

     | | Polaris 0.25 m cylinder | HMMWV production mesh |
     |---|---|---|
     | At the settle (axle means) | +0.004 m | +0.006 m |
     | While driving (axle means) | -0.012 m | -0.014 m |

   - The unmodified `crm_collect.build_crm` already takes the mesh with a config change only (same mesh on all four
     wheels). The cylinders go through `ag_vehicle.make_build_crm` unchanged.
6. **Spawn height: ground + 0.40 m in the re-framed model.** It is settled at the 0.8 s start of data: vertical speed
   0.004 m/s.
   - At the frozen +0.75 m it is still bouncing (-0.11 m/s).
   - At +0.2 m or less the tyres start inside the ground and the simulation blows up.
7. **Smoke against the Gator (local, indicative, one seed each): the Polaris is far better on soil.**
   - **Four f104 routes.** The Gator stalled on all four (`NOTES_E2.md:198-203`); the HMMWV reached the goal on 3 of 4.
     The Polaris reached the goal on 4 of 4: in 12.35, 7.55, 25.0 and 6.95 s, against the HMMWV's 13.1, 8.9,
     breakthrough and 9.0 s. The re-framed model also reached the goal on the first of these routes (12.35 s).
   - **20° soil ramp at 4 m/s:**

     | | how far up the ramp | speed |
     |---|---|---|
     | Polaris | 11.8-12.0 m | still 3.1 m/s |
     | HMMWV | 5.7 m | slowing to 1.0 m/s |
     | Gator | 2.5 m | stalled |

   - **25° ramp:** Polaris 10-11 m at 2.6-3.0 m/s; HMMWV 5.0 m, slowing to 0.3 m/s.
   - **Flat soil, 2 s full throttle:** Polaris 8.7 m/s; HMMWV 4.15; Gator 3.6.
   - The power-correct driveline gives the same picture, so the soil result does not rest on the driveline defect.
8. **Main study risk: the Polaris may leave little to learn.**
   - If it reaches the goal on almost every designed route, failure labels become rare.
   - The straight route alone may already pass the 90 % bar, so the planner test would sit at a ceiling, as rigid
     f104 did.
   - The pilot must measure the straight 6 m/s failure rate on the Polaris before the 15,235-id collection is
     committed.

## 1. How the Gator was switched in (the mechanism to copy)

**One switch** (`scripts/ag_vehicle.py:63-87`):
- `--vehicle hmmwv|gator` on the wrapper's command line;
- else the environment variable `NEDM_VEHICLE`;
- else `hmmwv`.

The allowed names are fixed at `ag_vehicle.py:39`. HMMWV mode patches nothing and calls the original `main()`; this
was checked bit-identical (`NOTES_E2.md:95-111`). Production task rows always carry the vehicle explicitly
(`NOTES_E2.md:288-289`).

**What Gator mode swaps** (module attributes only, before the unchanged collectors read them):

| Swapped attribute | Replacement | file:line | Effect |
|---|---|---|---|
| `nedm.traverse.scene.build_config` | `make_build_config` | `ag_vehicle.py:161-169` (spawn constants `:40-41`) | Gator vehicle block (`:107-113`); spawn moved from ground + 0.75 m to + 0.35 m. The +0.75 literal sits at `crm_collect.py:177`, `crm_collect_ext.py:114` and `traverse_fdm_rgbd_diverse_chrono.py:162`. |
| `nedm.hmmwv_data.create_hmmwv`, `nedm.traverse.scene.create_hmmwv` | `make_create_vehicle` → `create_gator` | `ag_vehicle.py:116-158` | Dispatches on `vehicle.model == "Gator"`; mirrors `create_hmmwv` (`hmmwv_data.py:290-336`, which refuses anything but `HMMWV_Full` at `:294-295`). |
| `crm_collect.build_crm` (soil) | `make_build_crm` | `ag_vehicle.py:206-230` (proxies `:173-200`) | The original solver, soil and SPH statements, with only the wheel geometry passed to `terrain.AddRigidBody` replaced: one cylinder per axle (axle 0 front, axle 1 rear). Asserts all four swaps. The original puts one mesh on every wheel (`crm_collect.py:111-116`). |
| the collectors' `dump` | `make_dump` | `ag_vehicle.py:338-348`; record block `:301-335` | Adds a `vehicle` block to outcome.json, collection_request.json, f104_episode.json (plus simulation_provenance.json and collection_meta.json on rigid). Writes `vehicle_extra.npz`. |
| soil `scene_hook` / `frame_hook` | belly diagnostic | `ag_crm_collect.py:77-86`; `BellyClearance` `ag_vehicle.py:241-297` | Lowest of 549 chassis-hull underside points (`scripts/ag_gator_belly.json`, from `gator_chassis_col.obj`) minus the undisturbed surface, every frame. |
| rigid: `gen_collect.import_runner`, `make_observer`, `gen_collect_ext.subprocess_cmd` | wrappers | `ag_gen_collect_ext.py:145-196` | Installs the same swaps after the frozen loop loads; `branch_auto` children are re-launched with `--vehicle gator` (`:186-193`). |

**Soil path** (`scripts/ag_crm_collect.py`):
- `main` (`:89-99`) picks the base collector (`crm_collect` or `crm_collect_ext`) and parses the switch.
- For the Gator, `install_gator` (`:60-86`) inserts the source tree's `src`/`scripts` in the base collector's order,
  then patches.
- The file name contains `crm_collect`, so `crm_worker.py:98` forwards `--crm-config`.

**Rigid path** (`scripts/ag_gen_collect_ext.py:213-221`):
- Gator rigid runs must bind a runtime fingerprint that lists `data/vehicle/gator/` (`ag_vehicle.py:351-358`,
  `ag_gen_collect_ext.py:147-152`).

**Why the soil wheels are cylinders.**
- The Gator tyre meshes are not watertight: 68 and 72 open edges, 11 non-manifold edges.
- At 0.08 m they give 94 and 116 markers with 11 and 8 empty 10° sectors. My re-implementation below reproduces
  Chrono's counts.
- A cylinder at the stock radius rides about one particle spacing (~0.1 m) too high. So the radius was cut until the
  sinkage matched the HMMWV mesh: stock - 0.09 m (`ag_vehicle.py:42-47`; `NOTES_E2.md:113-154`, protocol
  `scripts/ag_soil_calibrate.py`).

**Vehicle body in soil.**
- The chassis is not coupled to the soil (`crm_collect.py:180`).
- A diagnostic only, the belly-in-soil flag, records hull points below the original surface. Prior result: 8.5 % of
  Gator drives, limit 10 % (`REPORT.md:19`).

**Launch check** (soil):

| Condition | Limit |
|---|---|
| speed | at most 1 m/s |
| roll and pitch | at most 25° |
| yaw error | at most 10° |
| start position error | at most 1 m |
| chassis reference above the terrain image | 0 to 1.2 m |

Source: `crm_collect.py:411-432` (the height window at `:422`), `crm_collect_ext.py:589`. This is a class defined
inside `main()`, so it **cannot be swapped from a wrapper**.

**Other stop rules:**
- soil breakthrough: a wheel more than soil depth + 0.06 m below the surface for 5 frames (`crm_collect.py:285-293`);
- rollover at 60° (`:298`);
- fell through: reference more than 1 m below the surface (`:280-281`).

**Telemetry that became NaN** (rigid only). `src/nedm/traverse/fdm_rich_telemetry.py:204-215` reads spring and shock fields through
`CastToChDoubleWishbone`. The Gator's single-wishbone and rigid suspensions give `None`, so 24 fields are NaN. The
soil collectors do not use rich telemetry.

## 2. Every place a new vehicle must be added

Rule for the study: do not edit the `ag_*` files. Add new `ov_*` files that generalise the vehicle → id-prefix map.

| Stage | file:line | What is vehicle-specific | Needed for the Polaris |
|---|---|---|---|
| Switch and factory | `ag_vehicle.py:39` `VEHICLES`, `:65-72` Gator-only flags, `:79-85` parse, `:107-158` block and factory, `:161-169` spawn, `:206-230` soil wheel, `:241-297` belly (`BELLY_FILE` `:49`), `:306,313` record hard-codes `vehicle/gator` and `"gator"`, `:351-358` fingerprint needs `/vehicle/gator/` | everything | New `ov_vehicle.py`: model "Polaris", JSON factory + adapter (section 4), spawn 0.40, cylinder 0.25 m, belly points from the visual chassis mesh, record naming the Polaris JSON files and the private data folder |
| Soil wrapper | `ag_crm_collect.py:60-99` | installs Gator swaps | New `ov_crm_collect.py`. The name must contain `crm_collect` (`crm_worker.py:98`, `ag_soil.sbatch:17`, `ag_soil_launch.sh:15`) |
| Soil collectors (frozen; read-only) | `crm_collect.py:177` spawn +0.75, `:179-180` rigid-mesh tyres / no chassis collision, `:184` `create_hmmwv`, `:186` engine/transmission, `:195` tyre radii before the first `Synchronize`, `:422` launch window, `:285-298` stop rules; `crm_collect_ext.py:114-130`, `:196` engine speed from the gearbox, `:589` | HMMWV defaults | Covered by the swaps. The factory must map `RIGID_MESH` to the Polaris rigid tyre JSON; the reference must be re-framed to pass `:422` |
| Rigid wrapper (only if rigid Polaris data is wanted) | `ag_gen_collect_ext.py:145-196`; `gen_collect_ext.py:545` / `gen_collect.py:274` require `/vehicle/hmmwv/` in the runtime fingerprint | Gator fingerprint | New fingerprint listing `data/vehicle/Polaris/` + the private JSON. Note: the Polaris JSON has no chassis collision shapes, so the chassis cannot touch rigid ground |
| Calibration tool | `ag_soil_calibrate.py:44` (`choices=("hmmwv","gator")`), `:75-84` | Gator only | My scratch `polaris_soil_calib.py` does the same for the Polaris |
| Cluster launch | `ag_soil.sbatch:21` `CRM_ROOT`, `ag_soil_launch.sh:9` `G3` | study root hard-coded | `ov_` copies with the new root |
| Soil task rows | `ag_soil_tasks_v2.py:49-66` (`id='gator__'+id`, `extra=['--vehicle','gator']`, seeds = the HMMWV twins'), asserts `:124-134` (15,235 rows); pilot `ag_pilot_tasks.py:70-96` | prefix, extra, asserts | `polaris__<collect_v1 id>`, `pair_id`, `extra ['--vehicle','polaris']`, same seed and tier |
| Progress | `ag_soil_tier_status.py:37` key `arena:vehicle`; `ag_collect_status.py` | reads the `vehicle` field | works if rows carry `vehicle='polaris'` |
| Dataset builder | `ag_build_ds.py:51` `VEHICLE_PREFIX`, `:272` choices, `:294-295` id pattern, `:330` task vehicle, `:345-347` vehicle block must be `'gator'` or absent | two-vehicle logic | Needs a `polaris` prefix and a vehicle-block name check |
| Eval-only rows | `ag_build_evalonly.py:52,86,95` | same two-vehicle logic | same |
| Id lists | `ag_e5a_ids.py:24` default prefix `gator__`; `ag_subset.py:76` strips `'gator__'` | prefix | pass `--prefix polaris__`; the subset strip needs a new option |
| Eval drives | `ag_eval_tasks.py:53-54` (`GATOR_FP`, `VPREFIX`), `:97-101` arm `@vehicle` parse, `:166-170` run id and extra | prefix and extra | Add `polaris`; soil extra `['--vehicle','polaris']` |
| Eval index | `ag_eval_index.py:156` asserts the vehicle block is `'gator'` exactly when the row is a Gator row | Gator equality | generalise to name equality |
| Planner picks and trainer | `ag_picks.py` (vehicle-agnostic; drops `NEDM_VEHICLE` at `:222`); `ci_train.py` | none | Retrain from scratch on Polaris rows; the normalisers follow the data |
| Analysis | `ag_analyze.py:72-77` (task-B arms), `ag_bf_spec.py`, `ag_vsr_*.py`, `ag_s2b_*.py`, `ag_e6b_*` | arm names fixed to Gator/HMMWV | new spec files for the Polaris arms |
| Labels | `f104_n2_dataset.py:84-110` | absolute speed thresholds, no vehicle constants | unchanged |
| Route limits | `fdm_mppi.py:19-37` (curvature 0.125, footprint half-length 2.6 m and half-width 1.3 m), `f104_n2_sampler.py:23-24,58,67` | HMMWV-sized | Keep. Curvature 0.125 is 67 % of the Polaris's full lock (HMMWV 85 %); its body (3.4 x 1.5 m) fits inside the footprint |

## 3. Chrono's Polaris model

### 3.1 Files

`share/chrono/data/vehicle/Polaris/` (33 files; md5 list in `scratch/S1/polaris_md5_conda.txt`):
- The md5 of the whole list is `2693f88c…` for the local conda data, the local source build data, and both cluster
  builds (`chrono-build`, `chrono-build-fsi`).
- Files: `Polaris.json` (vehicle), `Polaris_Chassis.json` (+ `_Variant`, a 1,836 kg chassis with near-zero inertias,
  unused), `Polaris_Front_DoubleWishbone.json`, `Polaris_Rear_TrailingArm.json` (template ThreeLinkIRS),
  `Polaris_Antirollbar.json`, `Polaris_PitmanArm.json`, `Polaris_Wheel.json`, `Polaris_BrakeShafts.json` /
  `_BrakeSimple.json`, `Polaris_DrivelineSimple.json` (used) / `Polaris_4WD.json` (shafts, unused by
  `Polaris.json`), `Polaris_EngineSimpleMap.json`, `Polaris_AutomaticTransmissionSimpleMap.json`, and tyres
  `Polaris_TMeasyTire.json`, `_RigidTire.json`, `_RigidMeshTire.json`, `_Pac02Tire.json` (+ `.tir`),
  `_ANCF4Tire_Lumped.json`, `_ANCF8Tire_Lumped.json`.
- `meshes/`: chassis (9.7 MB OBJ + DAE), `Polaris_tire.obj` (visual), `Polaris_wheel.obj`,
  `Polaris_tire_collision.obj`.
- `README.md`: data based on SEA Ltd. 2016 measurements of a 2013 Polaris MRZR.

### 3.2 Specifications against the other two vehicles

Polaris values are measured on the constructed model or read from the JSON.

| | Polaris (stock JSON) | Gator (`S2_gator_vehicle.md`) | HMMWV collector set-up |
|---|---|---|---|
| Mass | **1,378 kg** (chassis 1,105.5 kg) | 906 kg | 2,573 kg |
| Wheelbase / track | **2.715 m / 1.232 m** both axles | 1.94 m / 1.12-1.24 m | 3.30 m / 1.82 m |
| Tyre | **r 0.3302 m, w 0.2121 m**, 15.1 kg + wheel 18.8 kg, all four equal | 0.286 / 0.318 m, unequal | 0.470 m |
| Suspension | front double wishbone, rear three-link independent (trailing arm) + anti-roll bar | single wishbone / rigid | double wishbone both |
| Driven wheels | **all four**: `SimpleDriveline`, 50/50 front/rear, limited-slip bias 2.0 front and rear, final ratio 0.25 | rear only, limited slip | all four, open differentials |
| Engine | `EngineSimpleMap`: 185 N·m at idle, **397 N·m at 1,500 rpm**, 250 N·m at the 2,700 rpm limit; map peak about 76-78 kW | 200 N·m, 14 kW | shafts engine, 81 kW |
| Gearbox | `AutomaticTransmissionSimpleMap`: 6 forward gears 0.267-1.577, reverse -0.171, up-shift at 2,375 rpm, down-shift below 1,000 rpm (gears 2-6), **no torque converter** | 1 gear | 3-speed automatic with torque converter |
| Brakes | `BrakeShafts` **2,000 N·m on all four wheels** | 800 N·m, rear only | all four |
| Steering | Pitman arm | rack and pinion | Pitman arm |
| Declared getters | `GetWheelbase` 0, `GetMaxSteeringAngle` 0, `GetMinTurningRadius` 20: not in the JSON, **wrong**; unused by the PID follower | also wrong | right |
| Reference point | **front axle, 0.397 m below the axle line: about 4-9 cm below the ground at rest** | mid-wheelbase | mid-wheelbase, 0.026 m above the axle line |
| Ground clearance | about 0.31 m (lowest visual-mesh point 0.356 m above the reference; there is **no chassis collision shape** in the JSON) | 0.14-0.17 m | 0.51 m |
| Breakover angle (rough, visual mesh) | about 26° | about 22° | about 49° |
| Tyre grip scale on the friction-0.9 rigid terrain (TMEASY, tyre friction 0.75) | 1.2 | 1.5 | 1.125 |
| Top speed | **no real limit in Chrono.** Stock: 58 m/s after 10 s (driveline defect below). With the shafts driveline the gearbox limit is 29.4 m/s, but the engine keeps 250 N·m above its rev limit (`ChEngineSimpleMap.cpp:39` clamps speed, not torque), reaching 50.8 m/s after 30 s | 8.2 m/s | 20.2 m/s |

### 3.3 Two powertrain defects in Chrono's generic templates

1. **`ChSimpleDriveline` (used by `Polaris.json`).**
   - Driveshaft speed = wheel speed x 0.25 (`ChSimpleDriveline.cpp:106`), while wheel torque = driveshaft torque / 0.25
     (`:115`). Power at the wheels is therefore 16 times the engine's power.
   - Introduced by commit `dfff7a809` ("Add missing reductions…", 2024-12-10). The cluster source `f54254fa` has the
     same code.
   - Measured: at 2 s full throttle, 4 x 709 N·m x 21.1 rad/s = 60 kW at the wheels against 20 rad/s x 189 N·m =
     3.8 kW at the engine. On speed-held climbs, engine speed / kinematic engine speed = 0.065 ≈ 1/16.
   - HMMWV's own `HMMWV_SimpleDriveline` has the same flaw, but our HMMWV set-up uses the shafts all-wheel drive.
2. **`ChAutomaticTransmissionSimpleMap` + `ChEngineSimpleMap` (both Polaris variants).**
   - There is no torque converter. The gearbox shifts on engine speed alone (`ChAutomaticTransmissionSimpleMap.cpp:47,55`).
   - Once the wheels spin, the engine speed rises and the gearbox climbs to 6th gear. It does not come back down,
     because the spinning keeps the engine fast.
   - With the power-correct shafts driveline, this makes the Polaris roll back on rigid 20° at 2 m/s held, and on 25°
     and 30° at full throttle.

### 3.4 Tyre collision mesh for soil (Chrono's own soil demos)

**The demos.**
- C++ `vehicle/terrain/demo_VEH_CRMTerrain_WheeledVehicle.cpp`: files at `:70-74`, spacing 0.04 m at `:115`, start
  height 0.25 m at `:125`, 5e-4 s step at `:144`, one mesh per spindle through `AddRigidBody` at `:434-460`.
- Its Python twin `python/vehicle/demo_VEH_CRMTerrain_WheeledVehicle.py`: `:26-55`, `:89`, `:92-95`, `:113-116`,
  `:157`.
- Both use `Polaris_RigidTire.json` as the vehicle tyre and `Polaris/meshes/Polaris_tire_collision.obj` as the soil
  geometry. Their soil parameters equal ours (density 1,700, E 1e6, friction 0.8, cohesion 5 kPa).
- The test rig `test_rigs/demo_VEH_WheelTestRig_CRM.cpp` uses the same tyre at 0.02 m.
- I found no `demo_FSI_Polaris` in the local Chrono source (`a92c6f72`).
- Deformable ANCF tyres exist (`AddFeaMesh`), but the demo needs a 1e-4 s step for them. Too costly.

**Mesh check** (`scratch/S1/mesh_check.py`; numpy re-implementation of `ChFsiFluidSystemSPH::CreatePointsMesh`,
`ChFsiFluidSystemSPH.cpp:2933-3021`: a point is kept if two rays both cross the mesh an odd number of times).
- It reproduces Chrono's counts exactly: HMMWV 207 at 0.08 m and 1,885 at 0.04 m; Gator 94 / 116 and 817 / 951;
  Polaris 83 at 0.08 m (the same as Chrono's `GetNumBCE` in my soil runs).
- The "outer radius" columns are the outermost marker per 10° sector.

| Mesh | Watertight? | Radius span | Markers at 0.08 / 0.06 / 0.04 m | Outer marker radius min / median at 0.08 m | Empty 10° sectors at 0.08 / 0.06 / 0.04 m |
|---|---|---|---|---|---|
| `Polaris_tire_collision.obj` (640 vertices, 1,280 triangles) | **yes**: 0 open, 0 non-manifold, consistent orientation, 1 part, volume 0.050 m³ | 0.199-0.343 m (tread lugs beyond the 0.330 nominal) | **83** / 216 / 759 | **0.000 / 0.269** | **8** / 0 / 0 |
| `hmmwv_tire_coarse_closed.obj` (production) | yes | 0.270-0.468 | 207 / 485 / 1,885 | 0.315 / 0.418 | 0 / 0 / 0 |
| `gator_tireF_coarse.obj` | no (68 open, 11 non-manifold) | 0.096-0.277 | 94 / 238 / 817 | 0.000 / 0.170 | 11 / 0 / 0 |
| `gator_tireR_coarse.obj` | no (72 open, 11 non-manifold) | 0.106-0.301 | 116 / 274 / 951 | 0.000 / 0.228 | 8 / 4 / 0 |

**Reading.**
- Watertightness is no longer the problem. Resolution is: the Polaris tyre is a ring only 0.14 m thick and 0.23 m
  wide, so at 0.08 m it gets 2-3 marker layers.
- It is adequate at 0.06 m (factor 2.4 more particles) or at the demo's 0.04 m (factor 8). Either would make the soil
  finer than the Gator and HMMWV data at 0.08 m, so the data would not be comparable.

## 4. How to build it in Python (works locally; bindings present on the cluster)

```python
import pychrono as ch, pychrono.vehicle as veh
vehicle = veh.WheeledVehicle(<abs path of the top-level vehicle JSON>, ch.ChContactMethod_SMC)  # sub-files resolve against the vehicle data path
vehicle.Initialize(ch.ChCoordsysd(ch.ChVector3d(x, y, z), ch.QuatFromAngleZ(yaw)), fwd_vel)
vehicle.GetChassis().SetFixed(False)
eng = veh.ReadEngineJSON(veh.GetVehicleDataFile("Polaris/Polaris_EngineSimpleMap.json"))
trn = veh.ReadTransmissionJSON(veh.GetVehicleDataFile("Polaris/Polaris_AutomaticTransmissionSimpleMap.json"))
vehicle.InitializePowertrain(veh.ChPowertrainAssembly(eng, trn))
for axle in vehicle.GetAxles():
    for wheel in axle.GetWheels():
        t = veh.ReadTireJSON(veh.GetVehicleDataFile("Polaris/Polaris_RigidTire.json"))   # soil; TMeasy on rigid ground
        t.SetStepsize(tire_step)
        vehicle.InitializeTire(t, wheel, ch.VisualizationType_NONE)
for f in (vehicle.SetChassisVisualizationType, vehicle.SetSuspensionVisualizationType, vehicle.SetSteeringVisualizationType,
          vehicle.SetWheelVisualizationType, vehicle.SetTireVisualizationType):
    f(ch.VisualizationType_NONE)
vehicle.GetSystem().SetCollisionSystemType(ch.ChCollisionSystem.Type_BULLET)      # as create_hmmwv
```

- There is no wrapper class like `veh.Gator`. The collectors call `hmmwv.GetVehicle()`, `GetSystem()`, `GetChassis()`,
  `GetChassisBody()`, `Synchronize(t, inputs, terrain)`, `Advance(dt)` and the five visualisation setters. A 40-line
  adapter covers them: `scratch/S1/polaris_lib.py` `PolarisModel`.
- `capture_row`, `crm_tire_fields`, `WHEEL_SPECS`, the frozen `make_driver` (which takes the `WheeledVehicle`),
  `terrain.RegisterVehicle` and `GetFsiBodyForce` all worked unchanged with it.
- **Availability:**
  - Local conda pychrono and the local source build: every symbol present, the model built and driven.
  - Cluster `chrono-build` and `chrono-build-fsi` (login-node import only): `WheeledVehicle`, `ReadEngineJSON`,
    `ReadTransmissionJSON`, `ReadTireJSON`, `ChPowertrainAssembly`, `RigidTire`, `TMeasyTire`, `ShaftsDriveline4WD`,
    `SimpleDriveline`, `ChPathFollowerDriver`, all needed `WheeledVehicle` methods, `ChBodyGeometry`,
    `CylinderShape`, `TrimeshShape`.
  - `CRMTerrain` and `pychrono.fsi` (with `SoilProperties`) exist only in `chrono-build-fsi`, as for the Gator.
  - **UNVERIFIED:** constructing the Polaris on a cluster compute node.
- **Gotcha:** a missing JSON file makes the Chrono loader **abort the process** ("ERROR: Could not open JSON file",
  core dump). There is no Python exception, so no `collection_failure.json` is written.
- **Re-framed variant:** the modified top-level vehicle and chassis files must be found through the vehicle data path.
  - In the scratch wrapper, the factory points `veh.SetVehicleDataPath` at a private folder (`vehdata/` = new
    `Polaris_ov/` + a symlink `Polaris/` → the stock folder) only while constructing, then restores it.
    `veh.GetVehicleDataFile("")` returns the current path.
  - Tyres are headless and use the rigid (cylinder) tyre, so no mesh is loaded later from the private path.

**Re-framed JSON** (`scratch/S1/vehdata/Polaris_ov/Polaris_ovc_{stock,shafts4WD}.json` + `Polaris_ovc_Chassis.json`).
- Every chassis-frame location gets +1.35763 m in x and -0.42 m in z:
  - both suspension locations;
  - the steering location;
  - the anti-roll-bar location;
  - the chassis centre-of-mass location;
  - the driver position.
- The subsystem files are the stock ones. The visual chassis mesh was dropped from the chassis copy; it would sit
  0.42 m too low (rendering only).
- Result: spindles at (±1.358, ±0.616, -0.023 / -0.015) relative to the reference, like the HMMWV's
  (±1.65, ±0.91, -0.026).
- An earlier variant `Polaris_ov_*` (z only, +0.55 m, reference still at the front axle) was used for 4 of the 5
  f104 soil runs.

## 5. Local smokes

### 5.1 Rigid ground

Settings as `scene.build_config`: SMC contact, patch friction 0.9, 2 ms step, 1 ms tyre step, TMEASY tyres, 0.8 s
braked settle. Files: `scratch/S1/rigid_polaris_{stock,shafts4wd}.json`, `climb_hold_*.json`,
`rigid_polaris_centered_*.json`. About 1 min wall per full scenario list.

**Settle.**
- Stock frame: settled reference height -0.036 to -0.051 m. Spawn +0.05 m gives vertical speed 0.03 m/s at 0.8 s.
  At +0.75 m the tyres carry only 0.59 of the weight at 0.8 s.
- Re-framed: the reference settles at +0.365 m.

  | Spawn (re-framed) | Vertical speed at 0.8 s | Largest vertical speed, 0.6-0.8 s |
  |---|---|---|
  | +0.40 m | 0.004 m/s | 0.008 m/s |
  | +0.75 m | -0.11 m/s | 0.49 m/s |
  | +0.20 m | 0.25 m/s | (bouncing hard) |
  | +0.10 m or less | blows up | |

**Physics identity of the re-framing.**
- Open-loop: settle, 4 s full throttle, then steering 0.5 at throttle 0.3.
- Centre-of-mass and spindle paths match the stock model to 1.4e-6 m at 0.8 s, 1.2e-3 m at 2 s and 1.1e-2 m after
  about 75 m at 24 m/s. After that the run rolled over and diverged chaotically.
- The z-only variant: 9e-6 m and 3e-3 m.
- Speed readings differ only while pitching, because the reference point moved.

**Flat, full throttle.**
- Stock: 58 m/s after 10 s, still in 1st gear (the defect).
- Shafts driveline: 5.6 / 8.9 / 12.0 / 17.8 / 26.6 / 39.1 / 50.8 m/s at 1 / 2 / 3 / 5 / 10 / 20 / 30 s. It is in 6th
  gear by 7.5 s and at the engine limit from 13 s.
- Braking (shafts): 26.3 m/s to 0 in 2.95 s, 8.9 m/s² (HMMWV 5.8, Gator 3.3).
- Full brake facing downhill 15-30°: at most 1.2 cm of slide in 6 s, both variants. The Gator slid 1.2-8.6 m.

**Full-lock turn.** Front-axle frame: 5.85 m radius at 2 m/s, 6.15-6.25 m at 4 m/s. Measured at mid-wheelbase:
5.36 m at 2 m/s (curvature 0.187 /m) and 5.86 m at 4 m/s.

**Climbs, speed held** by the follower's speed controller (gains 0.6 / 0.05). Mean speed over the last 5 s of 15 s,
in m/s:

| Slope | Stock (targets 2 / 4 / 6) | Shafts (2 / 4 / 6) | HMMWV (2 / 4 / 6) |
|---|---|---|---|
| 10° | 1.84 / 3.88 / 5.95 | 1.93 / 3.91 / 5.94 | 1.92 / 3.94 / 4.51 |
| 15° | 1.76 / 3.81 / 5.90 | 1.90 / 3.88 / 5.98 | 1.90 / 3.91 / 4.22 |
| 20° | 1.68 / 3.74 / 5.86 | **-9.7 (rolled back, 6th gear)** / 3.20 / 3.20 | 1.89 / 3.89 / 4.01 |
| 25° | 1.60 / 3.68 / 5.86 | **rolled back** | 1.88 / 2.38 / 2.38 |

**Path follower, frozen gains** (look-ahead 5 m, steering 0.8, speed 0.6 / 0.05, steering-rate clamp). Test route as
in S2: two 8 m-radius 90° bends.

| Frame | RMS cross-track at 2 / 4 / 6 m/s | Largest \|steering\| | All completed? |
|---|---|---|---|
| front-axle (stock) | 0.90 / 0.87 / 0.80 m | | yes |
| **mid-wheelbase (re-framed)** | **0.63 / 0.59 / 0.49 m** | 0.77-0.89 | yes |
| HMMWV (S2) | 0.63 / 0.61 / 0.49 m | 0.87-0.95 | yes |

### 5.2 Soil, flat patch

Protocol identical to `ag_soil_calibrate.py`: 16 x 16 m flat arena, production `crm_main.json` (0.08 m, 1 ms),
0.8 s braked settle, then 2 s full throttle. Sinkage = stock radius - (spindle height - surface); negative means the
wheel rides high. File: `scratch/S1/soil_calib.jsonl` and `soil_calib_refs.jsonl`. My HMMWV reference run repeats
`e2/calib/calib.jsonl` digit for digit.

Soil launch-check height is the reference height above the surface at 0.8 s; it is -0.04 to -0.09 m for every
stock-frame Polaris row except the oversize cylinder.

| Wheel on soil | Markers per wheel | Settled sinkage F / R (m) | Driving sinkage F / R (m) | Speed after 1 / 2 s (m/s) | Launch-check height (stock frame) |
|---|---|---|---|---|---|
| HMMWV production mesh | 207 | +0.017 / -0.004 | -0.016 / -0.012 | 2.14 / 4.15 | +0.584 (passes) |
| Gator calibrated cylinders | 88 / 170 | -0.001 / +0.001 | -0.005 / -0.018 | 2.69 / 3.63 | +0.305 (passes) |
| Polaris collision mesh (demo geometry) | 83 | -0.010 / -0.003 | -0.039 / -0.036 | 3.67 / 6.81 | **-0.037** (fails) |
| Polaris cylinder, stock radius 0.330 m | 244 | -0.097 / -0.094 | -0.098 / -0.096 | 3.91 / 7.52 | +0.051 |
| Polaris cylinder 0.26 m | 156 | -0.007 / 0.000 | -0.018 / -0.022 | 4.47 / 8.62 | -0.038 |
| **Polaris cylinder 0.25 m (chosen)** | **152** | **+0.003 / +0.004** | **-0.008 / -0.016** | **4.64 / 8.73** | **-0.048** |
| Polaris cylinder 0.24 m | 144 | +0.011 / +0.014 | +0.004 / -0.006 | 4.81 / 8.82 | -0.057 |
| Polaris cylinder 0.22 m | 132 | +0.037 / +0.037 | +0.027 / +0.015 | 4.94 / 9.28 | -0.087 |
| Polaris shafts driveline, mesh / cylinder 0.24 m | 83 / 144 | -0.010 / -0.003; +0.011 / +0.014 | -0.041 / -0.032; -0.002 / -0.001 | 3.64 / 6.44; 4.23 / 7.01 (in 3rd-4th gear) | -0.037 / -0.057 |

Speed on the flat patch, real-time factor 0.74-0.81 for every row, is the same as the HMMWV's 0.79 and the Gator's
0.76.

### 5.3 Soil ramp (indicative)

Setup: a 24 x 24 m image-based arena, flat, then a constant grade in +x (checked: +1.46 m at 4 m up the 20° ramp);
production soil. Straight drive with the follower's speed controller. File: `scratch/S1/soil_ramp.jsonl`.

| Case | HMMWV | Gator | Polaris stock, 0.25 m cylinder | Polaris shafts, 0.25 m cylinder | Polaris stock, mesh |
|---|---|---|---|---|---|
| 20°, 4 m/s: distance up the ramp; speed at the end | 5.7 m; 1.0 m/s and falling | **2.5 m; stalled, rear wheels spinning** | **12.0 m; 3.1 m/s** | 11.8 m; 3.1 m/s | 10.6 m; 3.1 m/s |
| 25°, 4 m/s | 5.0 m; 0.3 m/s | – | 10.9 m; 3.0 m/s | 10.1 m; 2.7 m/s | – |
| 20°, 2 m/s | 4.3 m; stalled, one front wheel spinning at 31 rad/s | – | 8.0 m; 1.3 m/s | 9.1 m; 1.3 m/s | – |

### 5.4 Soil on f104, through the unmodified `crm_collect.py`

Wrapper: `scratch/S1/s1_polaris_crm_collect.py` (the `ag_crm_collect` swap pattern). Production config, 120 s
horizon, seed 1. Outputs: `scratch/S1/f104_soil/`.

| Route (collect_v1 id) | Polaris | Gator (`NOTES_E2.md:198-203`) | HMMWV collect_v1 |
|---|---|---|---|
| `0005_route_01` (crater, 4 m/s) | **goal 12.35 s** (front-axle frame raised 0.55 m); **goal 12.35 s** (re-framed, launch height +0.385 m) | stalled (34 s) | goal 13.1 s |
| `0000_route_02` (hill, 6 m/s) | **goal 7.55 s** | stalled | goal 8.9 s |
| `0005_route_00` (crater, 2 m/s) | **goal 25.0 s** | stalled | soil breakthrough 22.6 s |
| `0012_route_02` (hill cross-slope, 6 m/s) | **goal 6.95 s** | stalled | goal 9.0 s |

- All episodes passed the launch check: +0.50-0.52 m with the raised frame, +0.385 m re-framed.
- The 17-column state is finite. Largest wheel sinkage 0.03-0.08 m; largest pitch 18-26°.
- Real-time factor 0.49-0.56, like the Gator's 0.51-0.55 on the same routes. One re-framed run was 0.13 because other
  jobs were loading the GPU.
- Engine column: 0-42 rad/s. Recorded power: at most 2-6.6 kW. Both show the 1/16 driveline defect.
- These outcome files carry no vehicle block, and the `tire_mesh` key still names the HMMWV mesh. That key is unused
  with cylinders; the Gator runs recorded it the same way.

## 6. Integration recipe, calibration, risks, checklist

### 6.1 Recipe (new files only; the `ag_*` files stay frozen)

1. **Vehicle data.**
   - Ship `Polaris_ovc.json` (the stock `Polaris.json`, re-framed as in section 4, stock driveline) plus
     `Polaris_ovc_Chassis.json` in the study's source tree.
   - Optional second top-level file: the shafts driveline (`"Driveline": {"Input File": "Polaris/Polaris_4WD.json"}`).
   - At run time, build a private vehicle data folder holding them under `Polaris_ov/`, next to the build's own
     `Polaris/` (a symlink or a copy of the 33 stock files).
   - Hash every file in the vehicle record.
2. **`ov_vehicle.py` (switch and factory).**
   - `VEHICLES = ("hmmwv", "gator", "polaris")`. Delegate `gator` to `ag_vehicle`, so Gator rows stay byte-for-byte
     the same.
   - Model "Polaris": the factory in section 4 behind the adapter. Map `RIGID_MESH` or `RIGID` to
     `Polaris_RigidTire.json` (the collectors set `RIGID_MESH`, `crm_collect.py:179`), and `TMEASY` to
     `Polaris_TMeasyTire.json`.
   - Set and restore the vehicle data path around construction.
   - `build_config` swap: Polaris block; spawn z = collectors' value - 0.75 + **0.40**.
   - `build_crm` swap: `ag_vehicle.make_build_crm(crm_collect.build_crm, {"front": {"radius_m": 0.25, "width_m": 0.2121}, "rear": {...same}})`.
     Diagnostic arm: the watertight mesh through the unchanged `build_crm` with
     `tire_mesh = "Polaris/meshes/Polaris_tire_collision.obj"`.
   - Belly points: the lower envelope of the visual chassis mesh `Polaris_chassis.obj` on a 0.10 m grid (there is no
     collision hull). Shift z by -0.42 m and x by +1.358 m into the re-framed chassis frame. Label them "visual mesh".
     With about 0.31 m of clearance, soil breakthrough (0.30 m wheel sinkage) fires before the belly reaches the
     original surface.
   - Vehicle record: name `polaris`, JSON hashes, frame shift, spawn, wheel geometry, driveline choice, and a note on
     the driveline defect.
3. **`ov_crm_collect.py`.**
   - A copy of `ag_crm_collect.py`'s logic calling `ov_vehicle`. The name must contain `crm_collect`.
   - Copies of `ag_soil.sbatch` and `ag_soil_launch.sh` with the new study root.
   - Use `chrono-build-fsi` on the cluster.
4. **Tasks.** Build the `ag_soil_tasks_v2.gator_rows` pattern with id `polaris__<collect_v1 id>`, the same
   `episode_seed` and tier, and `extra ['--vehicle','polaris']`. Check 15,235 rows and unique seeds. Never rely on
   `NEDM_VEHICLE`.
5. **Downstream.** Generalise the prefix and vehicle-block checks listed in section 2 (`ag_build_ds.py:51,345-347`,
   `ag_eval_tasks.py:54,170`, `ag_eval_index.py:156`, `ag_subset.py:76`, `ag_build_evalonly.py:95`) in `ov_` copies.
   Retrain from scratch with fresh normalisers; do not reuse Gator or HMMWV energy or state scales.

### 6.2 What needs calibration or a decision

1. **Soil wheel radius.**
   - 0.25 m chosen here (settled +0.004 against the HMMWV's +0.006 m; driving -0.012 against -0.014 m).
   - Re-check on the cluster GPU type.
   - Run the same "+0.08 m" sensitivity arm as the Gator pilot, where the Gator's failure moved 13 points.
2. **Spawn height: 0.40 m re-framed.** Confirm on the cluster that the vertical speed at 0.8 s stays below 0.05 m/s
   and the launch height is between 0.30 and 0.45 m.
3. **Driveline: decide before collecting.**
   - Stock `SimpleDriveline`: "Chrono's Polaris as shipped" and Chrono's own soil demo. It carries the 1/16 defect:
     one gear, non-physical engine and power columns.
   - Shafts 4-wheel drive: power-correct and HMMWV-like open differentials, but it upshifts under wheelspin on steep
     rigid climbs.
   - My recommendation: stock as primary, labelled as such, with the shafts driveline as a pilot sensitivity arm.
     Both climbed the soil ramps alike.
4. **Follower gains:** keep the frozen ones. With the mid-wheelbase frame they already track like the HMMWV.

### 6.3 Risks

1. **Ceiling effect.** If Polaris failures are rare on designed routes (4 of 4 goals in the smoke; HMMWV 68 %
   failures, Gator 88 %), the labels carry little risk signal, and the straight route may already meet the 90 % bar.
   Measure the straight 6 m/s and 2 m/s failure rates in the pilot. Pre-declare what "planner works" means if the
   straight route is already at 90 % or more.
2. **Driveline defect in the stock JSON** (section 3.3): no gear changes, torque evaluated at 1/16 of the engine
   speed. Engine speed, engine torque and power/energy are not physical. Any energy head or analytic energy model does
   not carry over.
3. **Gearbox without a torque converter.** Wheelspin causes upshift runaway (shafts variant). On soil, wheelspin is
   common: the Polaris wheels spun at about twice the ground speed on flat soil. **UNVERIFIED** at scale.
4. **Reference-frame change.** Physics-identical by test, but it is a deviation from the stock files. The recorded
   position, speed and pitch are of the new point.
   - If the stock frame were kept instead, the launch check fails every soil episode.
   - Goal, parking and arena-exit rules would then act on the front axle, about 1.36 m ahead of the body centre.
5. **Wheel stand-in.** A cylinder of stock radius - 0.08 m instead of the tyre shape. The mesh is too coarse at
   0.08 m.
6. **A loader abort leaves no failure record** (missing JSON = process abort). The pilot must check that the private
   data folder exists on every node type. `crm_worker` retires a worker after 3 consecutive failures (`crm_worker.py:132`).
7. **The body is not coupled to the soil** (as for the HMMWV and Gator). Clearance of about 0.31 m makes this less
   important than for the Gator.
8. **Rigid-only issues** (if rigid data is added):
   - no chassis collision shapes (the chassis passes through obstacles);
   - 12 NaN rear-suspension telemetry fields (three-link rear);
   - a new runtime fingerprint;
   - TMEASY grip scale 1.2.
9. **The AMD cluster reproduces a run only on the same node.** Keep paired arms in one job. Local smoke outcomes are
   not references for cluster runs.
10. **Cost.** Same particle count and a similar cost per simulated second. Episodes that reach the goal end early, so
    the simulated hours per id should fall below the Gator's 141 h, possibly below the HMMWV's 91.5 h. **UNVERIFIED**
    on the cluster.

### 6.4 Polaris smoke-test checklist (before the pilot)

1. **Cluster devel job, `chrono-build-fsi`.**
   - Construct the re-framed Polaris.
   - Check the "Loaded JSON" lines: `Polaris_ov/…` from the private folder, the subsystems from `Polaris/`.
   - Mass 1,378.33 kg; spindles at (±1.358, ±0.616, -0.023 / -0.015) relative to the reference.
2. **Flat-soil calibration on MI210 and MI350** (`scratch/S1/polaris_soil_calib.py` protocol, 0.25 m cylinder).
   - Settled axle-mean sinkage within ±0.01 m of the HMMWV's.
   - Vertical speed at 0.8 s below 0.05 m/s; 152 markers per wheel.
3. **HMMWV and Gator rows through `ov_crm_collect.py`** (`--vehicle hmmwv` / `gator`): arrays bit-identical to
   `crm_collect.py` / `ag_crm_collect.py` on the same node.
4. **The four f104 routes of section 5.4** plus 2-4 dev routes.
   - `initial_state_validation.passed`; launch height 0.30-0.45 m; finite 17-column state.
   - A `vehicle` block with name `polaris`, the wheel geometry, frame shift and driveline.
5. **Pilot** (24 groups x 6 routes, as for the Gator). Arms:
   - calibrated cylinder (primary);
   - cylinder + 0.08 m;
   - stock mesh (diagnostic);
   - shafts driveline;
   - the straight route at 6 m/s and 2 m/s.

   Per arm, report:
   - failure rate by speed profile, and against the Gator and HMMWV twins;
   - breakthrough rate and belly flag;
   - gear histogram (shafts arm);
   - simulated hours per id and real-time factor per node type.

   **Go only if** soil failures leave headroom above the 10 % that the 90 % bar allows, and if the result does not
   depend on the driveline or the wheel by more than the declared threshold.
6. **Confirm the job environment:** no `NEDM_VEHICLE`; every Polaris row carries `--vehicle polaris`.

## 7. Not verified or open

- A Polaris episode on any cluster node, and the cluster GPUs' calibration.
- The re-framed variant on soil was run on one f104 route only. The other three used the z-only raised frame.
- Failure rates on the 15,235 designed and proposal routes. The four f104 routes and the ramps are single-seed smokes
  on the local build.
- Whether the shafts driveline's upshift runaway appears on f104 soil.
- The effect of the 1/16 defect on route outcomes, as opposed to the telemetry columns: the smokes suggest little
  effect at 6 m/s or below.
- The ground clearance and breakover angle are rough, from the visual mesh; there is no collision hull.
- The ANCF (deformable) Polaris tyres were not tried.

## Appendix: scratch files and commands

Scratch root: `artifacts/traverse/offroad_vehicles_20260927/scratch/S1/`.

| File | What it is |
|---|---|
| `mesh_check.py` → `mesh_check.json` | watertightness and marker check |
| `polaris_lib.py` | builder + adapter |
| `polaris_rigid_smoke.py` → `rigid_polaris_{stock,shafts4wd}.json`, `static_*.json`, `rigid_polaris_centered_{spawnsweep,follow}.json` | rigid scenarios |
| `polaris_climb_hold.py` → `climb_hold_{stock,shafts4wd,hmmwv}.json` | speed-held climbs |
| `raised_ref_check.py` → `rr_{stock,raised,centered}.json` | re-framing identity check |
| `Polaris_shafts4WD.json` | stock-frame shafts variant |
| `vehdata/Polaris_ov/Polaris_{ov,ovc}_{stock,shafts4WD}.json` + `*_Chassis.json` | re-framed JSON |
| `polaris_soil_calib.py` → `soil_calib.jsonl`; `soil_calib_refs.jsonl` (runs of the repo's `ag_soil_calibrate.py`) | flat-soil calibration |
| `soil_ramp_smoke.py` → `soil_ramp.jsonl` | soil ramp climbs |
| `s1_polaris_crm_collect.py` → `f104_soil/polaris_{stock,ovc_stock}_cyl025/<id>/` | f104 soil episodes |
| `polaris_chassis_mesh_extent.json` | chassis mesh extent |
| `polaris_md5_{conda,localsrc}.txt` | data hashes |

```bash
cd /home/harry/NeDM-traverse_mppi; S=artifacts/traverse/offroad_vehicles_20260927/scratch/S1
# rigid (conda):
/home/harry/miniconda3/envs/nedm/bin/python $S/polaris_rigid_smoke.py out.json [--vehicle-json $PWD/$S/Polaris_shafts4WD.json]
# soil (source build, under the lock):
export PYTHONPATH=/home/harry/chrono/build/bin:src:scripts OMP_NUM_THREADS=4
flock /tmp/luffy_crm.lock /usr/bin/python3.12 -P -u $S/polaris_soil_calib.py --geom cyl --radius 0.25 --out x.jsonl
flock /tmp/luffy_crm.lock /usr/bin/python3.12 -P -u $S/s1_polaris_crm_collect.py --frame ovc --driveline stock --radius 0.25 \
  --source-root . --case $C/f104_v2_group_0005.json --route $C/routes/f104_v2_group_0005/route_01.json \
  --chrono-data /home/harry/chrono/data --crm-config artifacts/traverse/crm_f104_v1/configs/crm_main.json \
  --horizon-s 120 --episode-seed 1 --out OUT        # C=artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases
```
