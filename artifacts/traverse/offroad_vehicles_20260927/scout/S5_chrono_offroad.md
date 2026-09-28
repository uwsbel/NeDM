# S5: what Chrono itself says about the Polaris and the M113 on CRM soil, and how our soil set-up treats them (scout, 2026-09-27)

Read-only scout for the Polaris / M113 soil study (branch `offroad_vehicles_v1`). Words used below:
- **soil** or **CRM** = Chrono's deformable ground made of SPH particles (the "continuum representation model").
- **markers** (Chrono: BCE markers) = the fixed points Chrono attaches to a solid body (a wheel, a track shoe, the soil
  floor) so that the soil particles can push on it. Soil only touches a body through its markers.
- **active box** (Chrono: active domain) = a box around each soil-coupled body. Soil particles inside it move; soil
  outside it is frozen in place.
- **breakthrough rule** = our stop rule that ends a soil drive once a wheel has dug through the whole soil layer.
- **Gator study** = `artifacts/traverse/arena_gator_20260925` (the study that added the Gator).

What I did:
- I changed no existing file and submitted no cluster job. On the cluster I ran only `grep` and an import check on the
  login node.
- I ran short rigid-ground smokes on the workstation CPU (each under 2.5 min).
- I ran 24 tiny soil runs (14 x 4.4 m flat soil box, production soil settings), all under
  `flock /tmp/luffy_crm.lock`, about 6 minutes of soil wall time in total.
- Everything is in `artifacts/traverse/offroad_vehicles_20260927/scratch/S5/`: the scripts, the `*.jsonl` results,
  and the corrected Polaris data folders `vehdata/` and `vehdata4wd/`. No `NEDM_VEHICLE` was set anywhere.

Labels: **VERIFIED** = read in code or files, or measured by me. **RECALLED** = from memory, not checked.
**UNVERIFIED** = my inference.

---

## 0. Bottom line

1. **Our soil layer is thin, with a rigid floor under it.** The soil is 0.24 m deep at 0.08 m particle spacing, which
   is only 4 particle layers. Below them sit 3 floor layers that follow the terrain shape. The soil's reach from any
   one point (0.19 m) is about as deep as the whole layer. A wheel or track that digs out the soil goes straight
   through the floor, because the floor pushes only on soil, never on the vehicle. That is why the breakthrough rule
   exists. For every vehicle, most of the weight is carried through the floor, so the rigid floor limits sinkage.
   This hides the M113's real advantage, its low ground pressure on deep soft soil (section 1).
2. **Chrono's own demos pair the Polaris and the M113 with our soil, but at 2 to 4 times finer spacing.**
   - The Polaris soil demo uses the same soil material as us (1700 kg/m³, 5 kPa cohesion, friction 0.8,
     stiffness 1 MPa), at 0.04 m spacing and a 0.5 ms step.
   - The M113 soil demo uses 0.02 m spacing, a 0.6 m deep pre-built soil bed, a 0.5 ms step and friction 0.6.
   - I found no Chrono data or demo that puts either vehicle on 0.08 m soil (section 2).
3. **At our 0.08 m spacing, Chrono's own track recipe produces no markers, so the tracks do not touch the soil.**
   - The demo keeps only the shoe contact boxes that are larger than 0.32 m at this spacing. The M113's pad is
     0.11 x 0.19 m, so nothing is kept. I ran it: 0 markers, and the M113 fell 16 m through the soil in 0.8 s.
   - A home-made full-width box per shoe does work, but it has to be calibrated (section 2.3).
4. **Chrono's stock Polaris files create power (VERIFIED, measured).** The Polaris's simple driveline converts the
   wheel speed the wrong way when it reports it back to the engine (`ChSimpleDriveline.cpp:106-116`, a Chrono change
   from 2024-12-10).
   - The engine sees about the wheel speed instead of 15 times it, so it never revs up. The wheels then receive about
     **16 times the engine's power**: 52.8 kW at the wheels vs 3.2 kW at the engine at 1.8 s; 1,189 vs 74 kW at 12.8 s.
   - The stock vehicle reaches 66 m/s after 12 s on flat rigid ground and climbs 30° at 12 m/s.
   - The cluster source has the same lines. Using the stock files would make any result an artefact.
   - Fix: gear-reduction factor 1.0 in the driveline and the gearbox ratios x 0.25. Power then matches (engine 75 kW,
     wheels 74 kW).
5. **The stock M113 cannot climb (VERIFIED, rigid ground, friction 0.9).**
   - With the default driveline it slides back down a 10° slope at full throttle. With the braked-differential
     driveline it holds 2.1 m/s on 10°, creeps on 15° and slides back on 20°.
   - Cause: the model has no final-drive gear reduction. The sprocket push is about 0.17 of the vehicle's weight
     (0.35 with the braked-differential driveline).
   - RECALLED: the real M113 climbs about 60 % grades (31°). About 21 % of f104 is steeper than 15°.
   - On f104 soil the stock M113 would test Chrono's gearing, not tracks versus wheels.
6. **The M113 is also numerically awkward and costly on soil (VERIFIED on the tiny box).**
   - Chrono's default solid-contact model with the production 1 ms soil step blows up (NaN) at once. It works at
     0.5 ms.
   - Chrono's alternative contact model (the one its M113 soil demo uses) runs, but the track jams: 0.4 m of
     travel in 4 s even on rigid ground.
   - Real-time factor on the tiny box: 0.07-0.12, against 0.63-0.70 for the wheeled vehicles at 1 ms. That makes the
     M113 5-10 times more expensive per simulated second.
   - The soil collector records tyre loads and wheel speeds, and applies its breakthrough rule to wheel axles. None
     of this exists for a tracked vehicle.
7. **Tiny soil slope smoke, standing start, 4 s of full throttle** (tilted gravity; not f104):

   | vehicle | 15° slope | 25° slope |
   |---|---|---|
   | Gator | stalls (0.3 m) | stalls |
   | HMMWV | crawls to 0.1 m/s; one wheel spins | stalls |
   | Polaris, corrected, Chrono's limited-slip driveline | 5.0 m/s at 3 s | 2.9 m/s |
   | Polaris, open differentials | 4.5 m/s | stalls and digs 0.16-0.19 m in 4 s (the HMMWV's failure) |
   | M113 (braked differential) | – | on 10°: 1.0 m/s |

   **The Polaris should clearly beat the Gator.** Whether it beats the HMMWV on steep ground depends on which
   driveline model it gets.
8. **Physical expectation: the Polaris beats the Gator by a wide margin; the stock M113 does not.**
   - Polaris vs Gator: 56 vs 15.5 kW/t, all wheels driven vs 57 % of the weight on driven wheels, a 6 m turning circle
     like the HMMWV's, 0.32 m belly clearance.
   - A physically faithful M113 should be the best of all on soil. Chrono's stock one is limited by its gearing and
     should fail on hills at least as often as the Gator.
9. **Recommendation.**
   - Run the Polaris smoke and, if it passes, the full collection, with the corrected power and a calibrated soil
     wheel. Add an open-differential arm in the smoke only, to measure how much the driveline model decides.
   - Run the M113 only as a small, explicitly caveated smoke with a declared gearing fix, or drop it for tonight
     (section 5).

---

## 1. Our soil set-up on f104

### 1.1 Parameters and where they are set (VERIFIED)

The production config is `artifacts/traverse/crm_f104_v1/configs/crm_main.json`. It is byte-identical to the Gator
study's copy (`arena_gator_20260925/verify_e2/worker_root/configs/crm_main.json`). The Gator soil launch uses it too
(`scripts/ag_soil_launch.sh:11`).

| item | value | where |
|---|---|---|
| particle spacing | 0.08 m | `crm_main.json:3` |
| soil depth | 0.24 m, uniform, following the terrain surface | `crm_main.json:4`; `crm_collect.py:128-130` (`uniform_depth=True`) |
| physics step | 1 ms (the code default is 0.5 ms) | `crm_main.json:5`; `crm_collect.py:38` |
| active box | 2 x 2 x 1 m around every soil-coupled body; no free-settling period | `crm_main.json:6-11`; `crm_collect.py:118-122` |
| side walls | yes (all sides except the top) | `crm_main.json:12`; `crm_collect.py:124` |
| soil material | density 1700 kg/m³, stiffness 1 MPa, Poisson 0.3, friction 0.8 (static = dynamic, so no rate dependence), grain size 5 mm, cohesion 5 kPa | `crm_main.json:14-22`; `crm_collect.py:82-92` |
| soil behaviour model | Chrono default: friction depends on shear rate ("mu(I)"); with static = dynamic friction it is a cohesive Mohr-Coulomb soil, strength = 5 kPa + 0.8 x pressure | default in `ChFsiFluidSystemSPH.cpp:580` |
| SPH numerics | smoothing length 1.2 x spacing (0.096 m), free-surface threshold 0.8, artificial viscosity 0.5, particle-shifting method PPST (push 3, pull 1), 4 steps between neighbour searches, Adami boundary, RK2 time stepping, cubic-spline kernel (default), 3 floor/body marker layers (default) | `crm_main.json:23-31`; `crm_collect.py:94-109`; defaults `ChFsiFluidSystemSPH.cpp:641,646` |
| multibody solver | Barzilai-Borwein, linearised implicit Euler, 4 threads, Bullet collision | `crm_collect.py:71-74` |
| wheel-to-soil geometry | HMMWV: the closed tyre mesh on all 4 wheels. Gator: one cylinder per axle at the stock radius minus 0.09 m | `crm_main.json:32`, `crm_collect.py:111-116`; `ag_vehicle.py:46-47,206-230` |
| vehicle on soil | rigid-mesh tyres; chassis not coupled to the soil; spawn at ground + 0.75 m (Gator: + 0.35 m) | `crm_collect.py:177-180`; `ag_vehicle.py:41,161-169` |
| horizon | 120 s | `crm_worker.py:94`; `crm_collect.py:388` |

**Stop rules.**

| rule | condition | where |
|---|---|---|
| goal | within 2.5 m | `crm_collect.py:294,301-303` |
| rollover | roll or pitch above 60° | `:298-300` |
| arena exit | beyond ±40 m | `gen_collect.py:123-125` |
| prolonged blockage | throttle above 0.3 while every position of a 2 s window stays within 0.25 m of each other; not before 24 s, then 2 s to confirm and an 8 s tail | `gen_collect.py:126-153`, `crm_collect.py:389-391` |
| breakthrough (soil only) | some wheel's sinkage exceeds 0.24 + 0.06 m for 5 frames in a row | `crm_collect.py:282-293` |
| fell through | chassis more than 1 m below the ground; treated as a crash, never a label | `:280-281` |
| launch check (soil) | chassis reference 0 to 1.2 m above the ground, roll and pitch within 25°, speed at most 1 m/s | `:419-422` |

### 1.2 What the 0.24 m layer looks like to the solver (VERIFIED from Chrono source)

- **Four particle layers.** `Construct` makes round(0.24 / 0.08) + 1 = 4 soil particles in each column
  (`ChFsiProblemSPH.cpp:847,884`). Under them come 3 layers of floor markers that follow the terrain shape
  (`:890`). The floor sits one spacing below the last soil layer, about 0.32 m below the surface.
- **The soil's reach is as deep as the layer.** Each particle interacts with neighbours up to 2 x 0.096 = 0.19 m
  away. So every particle is within one reach of either the surface or the floor.
- **The floor never pushes on the vehicle.** Floor markers and body markers interact only through soil particles.
  Once the soil under a wheel is gone, nothing holds the wheel up. That is why the breakthrough rule exists
  (`crm_collect.py:282-284`, the comment on why).
- **The active box turns with the wheel.** It is defined in the body's own frame, turned with the body, and then
  expanded to an upright box around it (`SphDataManager.cu:630-653,675`). Soil outside it is frozen: its velocity
  is set to 0 (`SphFluidDynamics.cu:255`).
  - A wheel at its starting angle (every settle and every launch) has active soil only down to 0.5 m below its axle.
    For the HMMWV (radius 0.47 m) that is 3 cm below the tyre. For the Gator's soil cylinders it is about 0.2 m,
    for the Polaris 0.17 m.
  - While the wheel turns, the box grows to about 1.1 m below the axle.
  - I checked whether this matters on the HMMWV. With a 2 m tall active box on the 15° slope it still stalls
    (1.7 m of travel vs 1.4 m in 4 s). Its settled sinkage changes by up to 3 cm on the rear axle. So the effect
    is small for this case. It also means the Gator's wheel-radius calibration (NOTES_E2 §3), which matched the
    HMMWV's *settled* sinkage, compared the two on different active depths. The effect is small, but worth knowing.
  - The M113's ground-run shoes do not turn, so their boxes cover the whole layer.

### 1.3 Is 0.24 m deep enough?

**Physics (UNVERIFIED hand estimates).**
- Under a load of width B, the vertical stress falls to about half by a depth of about B for a long strip (a track)
  and to about a third for a square patch (a tyre). A tyre is 0.21-0.32 m wide and the floor is about 0.32 m down.
  So the floor carries roughly 30-50 % of the surface pressure under a tyre and about 60 % under an M113 track.
  A long strip's stress reaches much deeper (several track widths).
- The textbook consequence of a rigid layer this shallow ("hard pan"): static sinkage is smaller and the ground
  looks stronger than deep soil of the same material. The effect is largest for the widest, longest footprint,
  the M113's.

**For the M113 (about 11.3 t).**
- Its advantage on soft ground is low ground pressure: about 55 kPa (RECALLED; about 64 kPa over the model's pads)
  against about 60-140 kPa under the rigid tyres (section 4).
- On a floor-supported 0.24 m layer, every vehicle has small static sinkage, so that advantage is largely switched
  off.
- This soil is also strong: with 5 kPa cohesion and friction 0.8 the standard bearing-capacity formula gives roughly
  0.5 MPa under a 0.38 m wide load, far above any of these pressures. The failures come from **digging under wheel
  spin**, not from the ground giving way under the weight.
- A spinning track can dig out 0.24 m too; the breakthrough rule would need a track version.
- **Verdict:** deep enough for the collection to run, but not deep enough to show tracks' flotation.

**For the Polaris.**
- Static and driving sinkage on flat soil is small: -0.01 to +0.05 m in my runs, against 0.24 m of soil. Belly
  clearance is about 0.32 m (section 3.1).
- In a stall with open differentials, one wheel dug 0.16-0.19 m in 4 s on 25°, the same dig-through pattern as the
  HMMWV (memory: the whole layer in 10-15 s).
- The breakthrough threshold (0.30 m sinkage) leaves the axle 0.03 m above the floor for its 0.33 m radius. That is
  consistent with the HMMWV and the Gator.
- **Verdict:** as adequate as it was for the HMMWV.

**What the floor does at the bottom.** It stops sinkage for anything that does not dig. Anything that digs passes
through it, and the breakthrough rule then ends the drive as bogged. Earlier HMMWV checks
(`crm-f104-night-state` memory) found 0.06 m spacing and a 0.5 ms step changed no outcomes (25/25) or few (96.5 %
the same). **Layer depth was never varied.**

---

## 2. What Chrono ships for the Polaris and the M113 on soil

### 2.1 Where I looked (VERIFIED)

- Conda `share/chrono/data` and its `yaml/`: `yaml/fsi` has only fluid cases, and no vehicle-on-soil YAML exists.
- Conda `include/chrono_{fsi,vehicle,models}` and the conda Python demos (`site-packages/pychrono/demos`).
- The local source tree `~/chrono` (commit `a92c6f72`, 2026-09-12; the cluster source is `f54254fa`) and its
  demos, `data/vehicle/{Polaris,M113,terrain/sph,cosim}`.

Findings:
- **Polaris** (`data/vehicle/Polaris/`, 23 JSON files and meshes). Its README says the data comes from SEA Ltd test
  reports on a 2013 Polaris MRZR (2016).
- **M113** (`data/vehicle/M113/` and `M113_RS/`). Its README cites Wallin et al. 2013 and other public sources.
- **Soil demos that use them:**
  - `demo_VEH_CRMTerrain_WheeledVehicle` (C++ and Python), the Polaris;
  - `demo_VEH_CRMTerrain_TrackedVehicle.cpp`, the M113;
  - `demo_VEH_WheelTestRig_CRM`, a single Polaris tyre;
  - `demo_VEH_Cosim_WheeledVehicle_SPH`, the Polaris in co-simulation.
- **One pre-built soil bed:** `data/vehicle/terrain/sph/S-lane_RMS/`, 4.70 M particles at 0.02 m on integer grid
  coordinates. It is 30 x 7 m and 0.58 m deep, with an S-shaped path (a 3 m lateral shift) and its own
  `sph_params.json` (friction 0.9).
- The cluster FSI build has `veh.M113`, `WheeledVehicle`, `TrackedVehicle`, `CRMTerrain` and the same Polaris files
  (md5 checked for `Polaris.json`, the driveline file and the tyre collision mesh).

### 2.2 Settings in Chrono's own soil demos vs ours (VERIFIED)

| | Polaris demo (C++) | Polaris demo (Python) | M113 demo | co-simulation `granular_sph.json` | **ours** |
|---|---|---|---|---|---|
| spacing | 0.04 m (0.02 m on the S-lane bed) `:115` | 0.04 `:89` | **0.02** `:134` | 0.04 | **0.08** |
| soil depth | 0.25 m box `:214` | 0.25 | 0.58 m (S-lane bed) | 0.5 | **0.24** (on a relief) |
| density / cohesion / friction / stiffness | 1700 / 5 kPa / 0.8 / 1 MPa `:105-106` | the same | 1700 / 5 kPa / **0.6** / 1 MPa `:111-115` | 1700 / 5 kPa / 0.7 / 2 MPa | 1700 / 5 kPa / 0.8 / 1 MPa |
| step | 0.5 ms (0.1 ms with flexible tyres) `:144` | 0.5 ms `:157` | 0.5 ms, variable CFD step `:119,194` | 0.1 ms | **1 ms** |
| smoothing / free-surface threshold | 1.0 / 2.4 `:190-191` | 1.2 / 0.8 `:204-205` (= ours) | 1.3 / 2.4, Holmes boundary, Wendland kernel, no shifting `:183-194` | – | 1.2 / 0.8, Adami, cubic spline, PPST |
| active box | 0.8 m cube `:111` | 0.8 m cube | 0.3 x 0.4 x 0.4 m per shoe `:120` | 0.8 m cube | 2 x 2 x 1 m |
| vehicle contact model / spawn | smooth-contact (SMC) / ref 0.25 m on a 0.25 m bed | the same | **NSC contact** `:416`, spawn 0.7 m | – | SMC |
| wheel or track to soil | `Polaris_tire_collision.obj` on each axle `:433-460` (flexible tyres through `AddFeaMesh`) | the same `:34` | the shoes' ground contact boxes, **kept only if longer and wider than 2 x (3 - 1) x spacing** `:486-493` | rigid tyre mesh | HMMWV mesh / Gator cylinders |
| test | 20 x 3 m straight at 7 m/s, stop 4.5 m before the end | the same | S-lane path at 7 m/s | – | f104 routes, 2-6 m/s |

- NSC (non-smooth contact) is Chrono's alternative to its smooth, spring-like contact model (SMC).
- The table's line numbers refer to `~/chrono/src/demos/vehicle/terrain/demo_VEH_CRMTerrain_{WheeledVehicle,TrackedVehicle}.cpp`
  and `site-packages/pychrono/demos/vehicle/demo_VEH_CRMTerrain_WheeledVehicle.py`.
- Neither demo drives a slope, a relief or a heightmap. The Polaris C++ demo has a `bump64.bmp` height-map option
  (0-0.3 m) that is off by default.
- Our soil material equals the Polaris demo's exactly. Our SPH settings equal the Python twin's. Only the spacing
  (twice as coarse) and the step (twice as long) differ.

### 2.3 Chrono's M113 recipe at our spacing (VERIFIED, measured)

The M113's single-pin shoe has these ground-contact boxes (`M113/track_shoe/M113_TrackShoeSinglePin.json`; the same in
the C++ model):
- a centre pad, 0.11 x 0.19 x 0.03 m;
- two side pads, 0.1315 x 0.0542 x 0.02 m each, at ±0.162 m.

The shoe pitch is 0.154 m and the shoe mass 18.02 kg.

What happens at each spacing:
- **The demo's size filter needs x and y above 2 x (3 - 1) x spacing:** 0.08 m at 0.02 m spacing (the centre pad
  passes, the side pads fail), 0.16 m at 0.04 m and 0.32 m at 0.08 m (nothing passes).
- **Even at 0.02 m** the demo couples only the 0.19 m centre pad of the model's 0.298 m of pad width (0.38 m overall).
  That roughly doubles the ground pressure the soil sees (UNVERIFIED estimate: about 140 kPa vs about 64 kPa over the
  model's pads).
- **Measured at 0.08 m:**

  | shoe geometry | markers per shoe | result |
  |---|---|---|
  | the demo's filter | 0 (0 bodies coupled) | the M113 fell 16 m in 0.8 s |
  | centre pad without the filter | 6 (Chrono fills small boxes with a 2 x 3 x 1 grid, `ChFsiFluidSystemSPH.cpp:2274`) | a sparse "bed of nails" |
  | a home-made full-width box per shoe, 0.154 x 0.38 x 0.06 m | 36 | 127 shoe bodies, 4,572 markers; it drives (section 3.5) |

- The home-made box's bottom markers sit 3 cm below the real pad. Settled, the pad ends up 0.6 cm above the surface
  (1 cm on 10°). Like the Gator's cylinders, it needs calibrating against a fine-spacing reference.
  - A 0.02 m reference is 64 times the particles per volume: about 208 M for the full f104 arena, not practical.
  - A small patch is practical (the demo's own bed has 4.7 M particles).

### 2.4 What Chrono's authors have published (RECALLED, not verified)

- The Chrono team at UW-Madison (Hu, Serban, Unjhawala, Negrut and co-authors, about 2021-2025) has shown the Polaris
  MRZR and the M113 on CRM granular terrain, with the soil values above. These appeared as demos and in papers or
  talks on the CRM model and its GPU speed.
- The S-lane bed and the 7 m/s target in both demos look like the course used there.
- I do **not** reliably recall any published sinkage, speed or slope number for either vehicle on CRM, so I quote
  none.
- The Chrono CRM validation papers I remember best are about single wheels and rovers (Curiosity/VIPER-type wheel
  tests, slope and gravity studies), not about full-vehicle gradeability.
- **So Chrono gives no benchmark for "can this vehicle climb an f104 hill on this soil".**

---

## 3. The vehicles as shipped: what I measured

Scripts are in `scratch/S5/`:

| script | what it does | results |
|---|---|---|
| `static_props.py` | vehicle geometry and masses | `static_props.json` |
| `rigid_smoke.py`, `polaris_fixed_smoke.py` | Polaris rigid-ground runs | `rigid_smoke_results.jsonl` |
| `polaris_debug*.py` | powertrain logs, second by second | printed |
| `m113_smoke.py` | M113 rigid-ground runs | `m113_smoke_results.jsonl` |
| `crm_smoke.py`, `wheeled_batch.sh` | the tiny soil runs | `crm_smoke_results.jsonl` |

The rigid ground copies S2's Gator smoke (smooth contact, friction 0.9, stiffness 2e7, 0.8 s braked settle). All runs
used the conda pychrono 10.0.0.

### 3.1 Static properties

| | Gator (S2) | HMMWV (S2) | Polaris (JSON, as the demo builds it) | M113 (`veh.M113`, stock) |
|---|---|---|---|---|
| mass | 906 kg | 2,573 kg | **1,378 kg** (chassis 1,105.5) | **11,343 kg** (chassis 7,819) |
| drive | rear axle only, limited slip | all 4 wheels, open differentials, 3-speed automatic, torque converter | all 4 wheels; `SimpleDriveline`: fixed 50/50 front/rear torque split, and each axle sends up to 2x torque to the slower wheel (a Torsen-like limited slip; `ChSimpleDriveline.cpp:62-89,94-130`); 6-speed automatic without torque converter | tracks; default `SIMPLE` driveline: equal torque to both sprockets, steering cuts the inner sprocket's torque (`ChSimpleTrackDriveline.cpp:103-109`); or `BDS`: gear reduction 0.5, steering by braking the inner track (`ChTrackDrivelineBDS.cpp:110-120`); 4-speed + torque converter |
| engine power | 14 kW | 81 kW | about 77.7 kW peak (map: 400 N·m at 1,500 rpm, 2,700 rpm max) | about 160 kW (map: 610 N·m peak, 380 lb-ft at 3,000 rpm; `M113_EngineShafts.cpp:35-43`) |
| power per tonne | 15.5 kW/t | 31.5 | **56** | 14 |
| wheels / tracks | r 0.286 / 0.318 m, w 0.254 / 0.305 | r 0.47, w 0.3175 | r 0.330 m, **w 0.212 m** (rigid tyre); collision mesh r ≤ 0.343 | 5 road wheels per side, r 0.305 m; 63 / 64 shoes per side (asymmetric); about 20 on the ground spanning 2.9 m; track centres 2.159 m apart |
| wheel loads, flat (measured) | about 1.9 / 2.5 kN per wheel | about 6.3 kN | **2.9 front / 4.1 rear kN** | 111 kN over two tracks |
| settled chassis reference | 0.27 m | 0.59 m | **-0.04 m (below the ground)** | 0.58 m |
| belly clearance | 0.14-0.17 m | 0.51 m | about 0.32 m (lowest chassis-mesh point at +0.356 m) | about 0.40 m (hull mesh at -0.18 m) |
| centre-of-mass height; tip-over angle, estimated | 0.63 m; 43° | 0.80 m; 49° | chassis 0.713 m (whole vehicle about 0.64 m); about 44° | chassis 1.02 m (whole vehicle about 0.87 m); about 51° |
| full-lock turn radius | 2.8 m | 6.8 m | **6.1-6.4 m** at 2-4 m/s (measured) | not measured (skid steering) |
| chassis collision shape in the data | hulls | hulls | **none** (`Polaris_Chassis.json` has no contact block) | hulls file, off by default |

Pipeline notes:
- The Polaris is built from JSON as a `WheeledVehicle`. It has no `GetVehicle()` wrapper method, so the frozen loop
  needs a thin adapter. Its settled chassis reference below the ground **fails the soil launch check**
  (clearance 0 to 1.2 m, `crm_collect.py:419-422`): -0.075 m measured on soil.
- The M113's contact model defaults to NSC in the wrapper (`M113.cpp:42`); the demos set SMC.

### 3.2 The Polaris power fault and its fix (VERIFIED)

`ChSimpleDriveline::Synchronize` (`ChSimpleDriveline.cpp:106-116`, `git blame`: commit dfff7a809, 2024-12-10, "Add
missing reductions ...") handles the gear reduction inconsistently:
- it reports the driveshaft speed as wheel speed **x** 0.25;
- it applies wheel torque as driveshaft torque **÷** 0.25.

Torque is multiplied by 4 but the reported speed is divided by 4, so power at the wheels is 1/0.25² = 16 times the
engine's. The cluster source (`/home1/harry/chrono`, f54254fa) has the same lines. The Gator's own simple driveline
has no such factor, and its top speed matches its gearing (S2).

| Polaris variant (rigid ground) | flat, speed after 10 s | 25° climb from rest, speed at 10 s | 30° | 35° | power at wheels vs engine |
|---|---|---|---|---|---|
| stock Chrono files | 57.9 m/s (66 m/s at 12 s) | 20.3 m/s | 12.4 m/s | slides back | about 16x |
| **corrected** (`scratch/S5/vehdata/Polaris/`: gear-reduction factor 1.0, gearbox ratios x 0.25) | 29.6 m/s (no aerodynamic drag in the model) | 2.4 m/s, steady | 1.5 m/s | about 0 | about 1x (engine 75 kW, wheels 74 kW) |
| open differentials (`vehdata4wd/`: Chrono's own `Polaris_4WD.json`, an open-differential shaft driveline) | not run (about 23 m/s at 8 s) | – | – | – | about 1x |

Reading:
- On the 25° climb the corrected Polaris is held back by its **front wheels spinning at about 10 times the ground
  speed**. The fixed 50/50 split keeps feeding torque to the lightly loaded front axle, the spin raises the engine
  speed, and the gearbox shifts up.
- The front wheels spin because of that fixed split, not because of power or grip limits.

### 3.3 M113 on rigid ground (VERIFIED)

Stock `veh.M113`, single-pin shoes, SMC contact, 0.5 ms step. The time is the wall time for the 8-10 s drive.

| driveline | flat, speed at 2 / 5 / 10 s | 10° | 15° | 20° | 30° |
|---|---|---|---|---|---|
| SIMPLE (default) | 2.3 / 4.9 / 6.9 m/s; turns 37° off its heading in 10 s with the steering held straight | **slides back at 1.3 m/s** | – | slides back | slides back and spins round |
| BDS | 3.2 / 5.4 at 6 s | 2.1 m/s | 0.6 m/s (creeping) | slides back at 1.5 m/s | – |

- The engine sits at the torque converter's stall point (163 rad/s, 541 N·m). The sprocket torque is about 2 x 2,400
  N·m (SIMPLE) or about 2 x 5,000 N·m (BDS). With the sprocket radius of about 0.26 m that is a push of 0.17 or 0.35
  of the weight. Neither setting has a final-drive reduction.
- A final-drive ratio of about 3.9 (RECALLED for the real vehicle) would add that factor. The JSON version of the M113
  (`M113/vehicle/M113_Vehicle_SinglePin.json` + `powertrain/M113_AutomaticTransmissionShafts.json`, gearbox ratios
  0.240-0.962) allows it without code. **Untested.**
- **Speed and stability:**
  - The real-time factor on the CPU was 0.29 with SMC.
  - With NSC contact the track barely moves: 0.4 m in 4 s at full throttle with the default solver, 0.66 m in 2 s
    with 100 solver iterations.
  - With NSC the real-time factor was 0.02-0.04.

### 3.4 Wheel and track geometry against the soil at 0.08 m (VERIFIED)

| body | markers | settled sinkage (flat soil) | driving |
|---|---|---|---|
| HMMWV production tyre mesh | 207 per wheel | +0.007 / -0.004 m, run with a 2 m tall active box (NOTES_E2, production box: +0.017 / -0.004) | – |
| Gator cylinders (r - 0.09 m) | 88 / 170 | -0.005 / +0.010, settled on the 15° box (NOTES_E2 flat: -0.001 / +0.001) | – |
| Polaris, Chrono's tyre collision mesh | 83 per wheel | +0.006 / +0.003 | rides **3-5 cm above** the surface; 6.5 m/s at 2 s |
| Polaris cylinder, r - 0.09 = 0.240 m, w 0.212 m (the Gator's offset, **not calibrated for the Polaris**) | 144 per wheel | +0.019 / +0.024 (+0.02 / +0.04 on slopes) | -0.01 to +0.04 |
| M113 home-made shoe box | 36 per shoe, 127 bodies | pad -0.006 m (above the surface) | – |

- **The soil also sees wheels as wider than they are.** S2 found that soil stays about one spacing (0.08 m) away from
  the markers; that is why the stock-radius cylinders needed shrinking. The same happens at the sides, so a wheel
  looks about 0.16 m wider (UNVERIFIED).
- In proportion this helps the narrow Polaris tyre most: 0.21 -> 0.37 m, x 1.75. The Gator gets x 1.5-1.6, the HMMWV
  x 1.5 and the M113 track x 1.4.

### 3.5 Tiny soil smoke: standing start on a tilted-gravity slope (VERIFIED)

Set-up:
- A flat 14 x 4.4 x 0.24 m box with the production soil and SPH settings.
- Gravity tilted about the side axis by the slope angle, so the flat box behaves like a slope for both soil and
  vehicle (Chrono applies the gravity to both, `ChFsiSystem.cpp:110-115`).
- 0.8 s braked settle, then full throttle and straight steering. The time is measured from the throttle.
- Rigid-mesh tyres (or the M113), the production soil-contact geometry, 1 ms step unless marked.

| vehicle (soil geometry) | slope | speed at 1 / 2 / 3 / 4 s (m/s) | travel in 4 s | what happens | real-time factor |
|---|---|---|---|---|---|
| Gator (production cylinders) | 15° | 0.16 / 0.04 / -0.04 / 0.00 | 0.33 m | stalls: rear wheels spin at about 4 m/s, fronts stop | 0.63 |
| Gator | 25° | 0.06 / 0.01 / -0.05 / -0.04 | 0 | stalls; rear sinkage grows to 0.077 m | 0.64 |
| HMMWV (production mesh) | 15° | 0.65 / 0.39 / 0.11 / 0.09 | 1.39 m | crawls to a stop; **one wheel spins at 8-12 m/s**, the others about 0-4 (open differentials) | 0.69 |
| HMMWV, active box 2 m tall | 15° | 0.61 / 0.46 / 0.39 / 0.25 | 1.69 m | the same pattern | 0.73 |
| HMMWV | 25° | stuck; one wheel at 15 m/s | 0 | stalls | 0.68 |
| **Polaris corrected** (cylinder r - 0.09) | 15° | 2.8 / 4.5 / **5.0** / (drove off the box) | more than 10 m | climbs; wheels slip 50-100 % | 0.19 |
| **Polaris corrected** | 25° | 0.73 / 1.87 / 2.65 / **2.94** | 6.7 m | climbs, still accelerating | 0.36 |
| Polaris, stock (power fault) | 25° | 0.42 / 0.49 / 1.32 / 1.98 | 3.0 m | climbs, more slowly | 0.63 |
| Polaris, open differentials | 15° | 1.6 / 3.5 / 4.5 / (off the box) | more than 10 m | climbs | 0.15 |
| Polaris, open differentials | 25° | 0.27 / 0.03 / 0.03 / 0.14 | 0.38 m | **stalls and digs: sinkage 0.16-0.19 m in 4 s, one wheel at 28 m/s** | 0.50 |
| Polaris corrected (Chrono's tyre mesh) | 0° | 4.0 / 6.5 | – | fine on flat ground | 0.64 |
| M113, home-made box, SMC, **0.5 ms** | 0° | 0.68 / 1.23 / 1.70 | 2.7 m in 3 s | moves; chassis rises 0.15-0.2 m during the drive (unexplained: pitching or riding onto pushed-up soil?) | 0.12 |
| M113 BDS, home-made box, SMC, 0.5 ms | 10° | 0.56 / 1.05 / 1.01 | 1.7 m in 3 s | climbs slowly | 0.067 |
| M113, home-made box, **NSC**, 1 ms | 0° / 10° | about 0 | 0.2 / 0.08 m in 3 s | track jammed | 0.09 |
| M113, home-made box, SMC, 1 ms | any | – | – | **NaN at once** (spawn 0.62 or 0.80 m, one or two step sizes) | – |

Caveats:
- A uniform slope with a standing start is harsher than an f104 route, which approaches with momentum over varying
  slopes.
- 4 s is short, the box is tiny, and there is one run per cell.
- The real-time factors are for a 30 k-particle box. Moving vehicles cost more there because their active boxes
  move.
- Even so, the ordering is stark and matches the collection: the Gator failed 88 % on soil, the HMMWV 68 %.

---

## 4. Physical expectation against the Gator and the HMMWV

### 4.1 Key numbers (hand calculations, UNVERIFIED; inputs from the tables above)

| | Gator | HMMWV | Polaris | M113 |
|---|---|---|---|---|
| power per tonne | 15.5 kW/t | 31.5 | **56** | 14 (model: engine held at 88 kW at the converter stall point on climbs) |
| share of weight on driven wheels or tracks | 0.57 | 1 | 1 | 1 |
| grip limit on flat soil (friction x driven share + cohesion x contact area / weight) | about 0.50 | about 0.84 | about 0.84 | about 0.88 |
| rough contact pressure (rigid tyre, 3 cm sinkage) | 60-70 kPa | about 120 kPa | about 100 (front) - 140 (rear) kPa | about 55 kPa (RECALLED) / 64 kPa (model pads) |
| relative wheel "mobility number" (width x diameter / load, rigid-wheel form; higher = floats better) | **0.062** front and rear | 0.041 | 0.042 front / **0.029** rear | – (tracks) |
| gear-limited push on climbs, model | torque ample below 2 m/s; power-limited above: 6 m/s is possible only up to about 10-15° | first gear, torque converter | 0.6-1.3 of the weight in first gear (corrected) | **0.17 / 0.35 of the weight (SIMPLE / BDS)** |
| how the model splits drive torque | rear axle only, limited slip | open differentials (one wheel spinning leaves the others without torque) | fixed split plus limited slip (spinning wheels do not starve the others) | equal sprocket torque |

### 4.2 Prediction

**Polaris vs Gator: much better.** All four wheels are driven, it has 3.6 times the power per tonne, several gears,
and it keeps driving every wheel even when one spins. The Gator's soil failures came from rear-wheel spin and its
14 kW (it stalls at 15° in the smoke).
- The Polaris's weak points are its narrow, more heavily loaded tyres: a rear mobility number of 0.029 vs the
  Gator's 0.062. That means more sinkage and slip-sinkage on flat soft ground.
- At 0.08 m spacing this is partly masked by the extra apparent width.
- **Expected: well below the Gator's 88 % failure on the Gator routes.**

**Polaris vs HMMWV: probably better on climbs; the driveline model decides.** The HMMWV's documented failure is
"stall, then one wheel spins with open differentials, then it digs through". The Polaris's Chrono driveline cannot do
that, because torque is split by fixed fractions with no link to wheel speed. With open differentials the Polaris
reproduces exactly the HMMWV's failure at 25°.
- Plausible range (UNVERIFIED): below the HMMWV's 68 % with Chrono's driveline, near it with open differentials.
- For the 90 % planner bar the Polaris is the most promising vehicle available. The HMMWV already reaches 95.6 %
  with its own planner on the 800-pair suite.

**Physically faithful M113 vs both: the best on soil.** It has low ground pressure, a thrust area of about 2 m², and
little rolling resistance. The 0.24 m floor-supported layer hides much of that advantage (section 1.3).

**Stock Chrono M113 vs the Gator: no better, probably no worse.** It is limited by its gearing to about 10° (SIMPLE)
or about 15° (BDS), below the f104 slopes: 21 % of the arena is steeper than 15° and 9 % steeper than 20°. On hill
and crater routes it should stall on most climbs. Its result would measure the missing final drive in Chrono's M113,
not tracks.

**Turning.** The Polaris's turn radius (6.1-6.4 m) is close to the HMMWV's (6.8 m), so the frozen path follower and
the 0.125 1/m route curvature limit fit it (about 78 % of full steering, HMMWV 85 %). The M113 skid-steers: steering
means cutting or braking the inner track's torque, so the follower's steering gain has a different meaning. On soil
skid-steering also meets high sideways resistance. Its tracking quality is untested.

### 4.3 Model-fidelity issues that could make a result an artefact (ranked)

1. **Polaris power fault** (section 3.2). Must be fixed before any drive; the fix is two JSON values and is
   power-consistent (measured). *If unfixed:* optimistic, and meaningless as a vehicle comparison.
2. **Which driveline model the Polaris gets** (section 3.5): Chrono's simple limited-slip driveline climbs 25° soil;
   open differentials dig in. The real MRZR/RZR runs on-demand all-wheel drive with a locked or limited-slip rear
   (RECALLED), which is closer to the simple model than to open differentials. *Effect:* decides steep-slope
   outcomes, and so "better than the HMMWV" is a driveline-model result. Declare it; measure it once in the smoke.
3. **M113 missing final drive** (section 3.3). *Effect:* the M113 looks as weak as the Gator for a gearing reason.
   Fix with JSON gearbox ratios (untested), or report it as "stock Chrono M113".
4. **M113 track-to-soil coupling at 0.08 m** (section 2.3). Chrono's recipe gives 0 markers. The home-made
   full-width box is uncalibrated: it sits 3 cm low, markers overlap at the shoe hinges, and there are no grousers.
   The chassis rising 0.15-0.37 m during the soil drives is unexplained. Treat the tracks as a model under test,
   not a reference.
5. **M113 numerics and cost**: SMC needs the 0.5 ms step (NaN at 1 ms); NSC jams; 127 soil-coupled bodies. Cost is
   5-10 times the wheeled vehicles' (tiny-box real-time factor 0.07-0.12 vs 0.63-0.70). The collector's 17-value
   state, tyre-force fields, spindle-height breakthrough rule and wheel-speed features are all wheel-specific
   (`crm_collect.py:135-153,195,217-223,285-287`).
6. **Soil-wheel geometry calibration.**
   - Chrono's own Polaris tyre mesh gives 83 markers and rides 3-5 cm high.
   - The r - 0.09 cylinder was borrowed from the Gator; settled sinkage +0.02/+0.04 m vs the HMMWV's about +0.01/0.
   - Calibrate the Polaris cylinder as NOTES_E2 §3 did for the Gator (5 minutes). The 13-point swing the Gator showed
     for 8 cm of radius (REPORT §3.4) shows that this matters.
7. **Apparent wheel width** at 0.08 m spacing (section 3.4). Favours the narrow Polaris tyre most. It cannot be
   fixed without finer spacing; state it.
8. **Rigid floor at 0.32 m** (section 1.3). Lowers every vehicle's sinkage, removes most of the M113's flotation
   advantage, and caps digging at breakthrough. Never varied so far.
9. **Chassis not coupled to the soil** (all vehicles).
   - Lower concern for the Polaris (0.32 m belly, beyond the breakthrough threshold) and the M113 (0.40 m) than for
     the Gator (0.14-0.17 m).
   - The M113's long nose can still pass into crater walls without resistance. For all vehicles that makes the
     results optimistic.
10. **Active box set at the wheel's starting angle** (section 1.2). A 0.5 m active depth below the axle at the settle.
    Small in the one test I ran (HMMWV 15°).
11. **Launch check and spawn height.** The Polaris's chassis reference settles below the ground (-0.04 m rigid,
    -0.075 m on soil) and fails the soil launch gate (`crm_collect.py:421-422`). It needs a vehicle-specific
    reference offset and a spawn at about ground + 0.10 m; the frozen +0.75 m would drop it about 0.8 m.

---

## 5. Recommendations

**The Polaris is the vehicle to run. Before the smoke:**
1. Use a corrected powertrain in a versioned data folder that the source-hash and runtime fingerprint gates cover,
   not the conda or cluster Chrono data:
   - `Polaris_DrivelineSimple.json` with both gear-reduction factors set to 1.0;
   - `Polaris_AutomaticTransmissionSimpleMap.json` with the ratios x 0.25 (reverse -0.04275; forward 0.06675, 0.125,
     0.186, 0.25, 0.3235, 0.39425).

   Templates are in `scratch/S5/vehdata/Polaris/`. Declare it as a deviation from Chrono's stock files, with the
   16x power evidence.
2. Build the vehicle from Chrono's JSON: `WheeledVehicle(Polaris.json, SMC)`, the simple-map engine and gearbox,
   rigid tyres (`Polaris_RigidTire.json`) on soil and TMEASY tyres on rigid ground. Wrap it in a small adapter class
   that provides `GetVehicle`, `GetSystem`, `GetChassis`, `GetChassisBody`, `Synchronize(t, inputs, terrain)` and
   `Advance`, so that the frozen loop and the `ag_vehicle`-style patch points keep working.
3. Soil wheel: one cylinder per wheel, width 0.212 m. Calibrate the radius on flat soil against the HMMWV's settled
   and driving sinkage (NOTES_E2 §3 protocol). Start from r - 0.07 and r - 0.09: r - 0.09 settled 1-3 cm deep.
4. Spawn at ground + 0.10 m. Change the soil launch gate to a vehicle-specific reference, for example the axle height
   above the ground in about [0.2, 0.5] m.
5. Keep the production soil settings unchanged (0.08 m, 1 ms, 0.24 m, 2 x 2 x 1 m) for comparability with the Gator
   and the HMMWV.
6. **In the smoke only**, add an open-differential Polaris arm (`Polaris_4WD.json`, template in `vehdata4wd/`) on
   the same routes. If the Polaris then passes only with Chrono's simple driveline, report that the gain belongs to
   the driveline model.
7. Optional, once the Polaris passes: a 0.48 m-deep-soil spot check (7 particle layers, about 1.75x the particles;
   memory says cost is set by the active box, not the particle count) on about 24 routes, to show that the floor does
   not drive the result.

**For the M113:**
8. Do not start a full M113 collection tonight. Cost is 5-10 times per simulated second (S3 already rules out 3x
   overnight), and the collector needs a tracked-vehicle state and rules.
9. If the user still wants an M113 data point, run a small, declared smoke on a subset of sample A (for example 24
   routes). The M113 settings for it:
   - SMC contact at a 0.5 ms step for both soil and vehicle;
   - one home-made box per shoe;
   - the BDS driveline;
   - a declared gearing fix (gearbox ratios ÷ about 4, through the JSON `TrackedVehicle`); the stock M113 is
     expected to fail on gearing.

   Report it as "tracks at 0.08 m with a home-made shoe geometry", not as Chrono's validated tracked model.
10. Otherwise write down that the stock Chrono M113 cannot climb f104's hills on rigid ground either (section 3.3).
    That answers "is it better than the Gator" without soil runs.

**For the report:** use the ranked fidelity list in 4.3 as the caveat section. The power fault and the M113's
gearing are worth reporting to Chrono upstream (the power fault comes from the 2024-12-10 change to the simple
driveline).

---

## 6. Not verified or open

- The corrected Polaris on the f104 relief (rigid or soil), and any Polaris calibration.
- The M113 with a gearing fix; M113 turning and path-following. Why its chassis rose during the soil drives.
- Why SMC with a 1 ms step gives NaN for the M113 (not diagnosed; 0.5 ms works).
- Full-arena soil cost for either vehicle. The tiny-box real-time factors are indicative only.
- The published Chrono numbers for either vehicle on CRM (section 2.4 is from memory).
- Real-vehicle figures marked RECALLED: MRZR drive layout; M113 ground pressure 55 kPa, 60 % gradeability, final
  drive about 3.9, track 0.381 x 2.67 m.
- The Polaris's braking and downhill holding (not measured). Its brakes are Chrono shaft-type brakes, 2,000 N·m per
  wheel, the brake type that let the Gator run away on 30° in S2.

## Appendix: commands

```bash
cd artifacts/traverse/offroad_vehicles_20260927/scratch/S5; P=/home/harry/miniconda3/envs/nedm/bin/python
$P static_props.py                                    # geometry and masses
$P rigid_smoke.py polaris flat 20                     # stock Polaris: 57.9 m/s at 10 s
$P polaris_debug.py                                   # wheel vs engine power (16x)
$P polaris_fixed_smoke.py climb 25 10                 # corrected gearing
$P m113_smoke.py SIMPLE 10 8 ; $P m113_smoke.py BDS 15 8   # stock M113 climbs (SMC, 0.5 ms)
S5_M113_NSC=1 S5_ITERS=100 $P m113_smoke.py SIMPLE 0 2      # NSC track jam
# soil, always under the lock:
flock /tmp/luffy_crm.lock $P crm_smoke.py m113_simple demo 0 0.2           # demo filter: 0 markers, falls through
flock /tmp/luffy_crm.lock $P crm_smoke.py m113_simple proxy 0 3 0.0005     # SMC needs 0.5 ms
S5_RUNS="gator cyl 15 4;hmmwv mesh 15 4;polaris_fixed cyl 25 4;polaris_4wd cyl 25 4" \
  flock /tmp/luffy_crm.lock bash wheeled_batch.sh
```
