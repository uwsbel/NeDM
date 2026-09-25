# NOTES E2: the Gator vehicle switch (arena_gator_20260925)

Module E2 of PLAN.md (sections 1.5, 1.6, 3; scout S2 section 5). Built 2026-09-25 01:20-01:55 on luffy. All local runs
use the source-built Chrono (`PYTHONPATH=/home/harry/chrono/build/bin:src:scripts /usr/bin/python3.12`), the closest
local build to the cluster's; soil runs ran under `flock /tmp/luffy_crm.lock`. Records are under `e2/` in this folder.

## 0. Summary

- **One switch:** `--vehicle gator|hmmwv` on the new wrapper collectors. If the argument is missing, the environment
  variable `NEDM_VEHICLE` is used, and if that is missing too, the default is `hmmwv`. The argument always wins over
  the variable.
- **No frozen file was changed, and no existing script was edited.** The wrappers swap a few module-level functions at
  run time, before the unmodified collectors call them. The 9 hash-checked source files, the frozen loop's source and
  every hook location stay the same, so the existing `source_manifest.json` gate still passes.
- **The HMMWV path is bit-identical.** Rigid and soil episodes run through the original collectors and through the
  wrappers in HMMWV mode produce identical arrays in every `.npz` file. The JSON files differ only in wall time.
- **Soil wheel calibration:** one cylinder per axle, radius = stock tyre radius - 0.09 m (front 0.19575 m, rear
  0.2275 m), stock widths.
  - At the 0.8 s settle, the axle-mean sinkage is 0.000 m for the Gator against +0.006 m for the HMMWV mesh.
  - Under full throttle it is -0.011 m against -0.014 m (verifier correction: (-0.0046 - 0.0183) / 2 = -0.0115).
- **Local checks on f104:**
  - **Rigid, 12 routes, Gator against the HMMWV:** launch and ground-height checks pass 12/12, all states are finite,
    and both vehicles reach the goal on 11/12 routes (the same route fails for both). The chassis never touches the
    ground, the lowest belly point stays 0.04 to 0.11 m above the ground, and the vertical speed at the settle is
    small.
  - **Soil, 4 Gator episodes:** all four stall on 19-24° climbs, with the rear wheels spinning. The HMMWV reached the
    goal on 3 of those 4 routes. The stall does not come from the wheel geometry: the stock Gator tyre meshes stall at
    the same spot. **This is the main risk for task B:** the Gator's soil failure rate may saturate.

## 1. The switch and the files

New files, all additive:

| File | Role |
|---|---|
| `scripts/ag_vehicle.py` | the switch: argument parsing, Gator factory, Gator config block and spawn height, soil wheel geometry, belly-clearance diagnostic, "vehicle" provenance block, fingerprint check |
| `scripts/ag_gen_collect_ext.py` | rigid collector = `gen_collect_ext.py` + switch (all its modes and flags pass through) |
| `scripts/ag_crm_collect.py` | soil collector = `crm_collect.py` (default) or `crm_collect_ext.py` (`--base crm_collect_ext`) + switch. The name contains `crm_collect`, so `crm_worker.py` forwards `--crm-config` |
| `scripts/ag_gator_belly.json` | 549 sample points on the underside of the Gator's chassis collision hull, in the chassis frame |
| `scripts/ag_gator_belly_points.py` | makes that file (needs scipy, local only; the collectors need numpy only) |
| `scripts/ag_soil_calibrate.py` | flat-soil wheel calibration (section 3) |
| `scripts/ag_runtime_fingerprint.py` | Gator runtime fingerprint builder (section 6) |

Helpers in `e2/`:
- `bitid/compare.py`: exact comparison of two run folders;
- `summarize_rigid.py`;
- the `run*.sh` scripts that produced every record below.

**HMMWV mode.**
- The wrappers install nothing and call the original `main()`:
  - `gen_collect_ext.main()` for rigid;
  - `crm_collect.main(argv)` or `crm_collect_ext.main(argv)` for soil.
- The `sys.path` order is the same as when those files run directly.
- No "vehicle" key is written. The convention: **a run with no `vehicle` block is an HMMWV run**, which covers all
  earlier data.

**Gator mode: what is swapped** (module attributes only, before the collector reads them):

| Attribute | Replaced by | Effect |
|---|---|---|
| `nedm.traverse.scene.build_config` | `ag_vehicle.make_build_config` | Returns the Gator vehicle block and moves the spawn height from ground + 0.75 m to ground + 0.35 m. All three call sites pass ground + 0.75; the wrapper subtracts 0.40. |
| `nedm.traverse.scene.create_hmmwv`, `nedm.hmmwv_data.create_hmmwv` | `create_vehicle` | Builds `veh.Gator` when the model is "Gator"; otherwise calls the original. |
| `crm_collect.build_crm` (soil) | the same function with the wheel geometry swapped | Every solver, soil, SPH and build statement is the original's. A stand-in `veh` object hands a terrain wrapper to the original function; that wrapper replaces only the geometry given to `AddRigidBody` (front: axle 0, rear: axle 1). All four swaps are asserted. |
| the collectors' module-level `dump` | a wrapper | Adds a `vehicle` block to `outcome.json`, `collection_request.json`, `f104_episode.json` (both worlds) and to `simulation_provenance.json` and `collection_meta.json` (rigid). Writes `vehicle_extra.npz` when `outcome.json` is written, so both files are covered by `episode_complete.json`. |
| rigid: `gen_collect.import_runner`, `gen_collect.make_observer` | wrappers | Install the swaps above after the frozen loop is loaded and before `gen_collect_ext` copies its namespace. Wrap the rich-telemetry observer with the belly measurement. |
| rigid: `gen_collect_ext.subprocess_cmd` | a wrapper | `branch_auto` children are re-launched through the wrapper with `--vehicle gator`. |
| soil: `scene_hook` / `frame_hook` | the existing optional hooks of `crm_collect.main` | Only record the belly clearance; no bodies or contacts are added. |

**The Gator (`create_gator` mirrors `create_hmmwv` line for line):**
- `veh.Gator()`, SMC contact.
- Stock driveline `SIMPLE` (rear axle driven, limited-slip split) and stock brakes `SIMPLE` (rear axle only), no brake
  locking. The engine and gearbox are fixed by the wrapper: simple engine, one gear.
- Tyre step = the config's. The same visualisation-off lines, then Bullet collision.
- Rigid: TMEASY tyres and chassis collision hulls (`gator_chassis_col.obj`), as the HMMWV.
- Soil: the collectors set RIGID_MESH wheel bodies and no chassis collision, as for the HMMWV. The chassis is not
  coupled to the soil. The follower, its gains, the route limits, the stop rules, the labels and the file schemas are
  the frozen ones.

**Belly-clearance diagnostic** (both worlds, every recorded frame): the lowest of 549 underside sample points of the
chassis hull, minus the undisturbed arena surface under that point.
- The 549 points are 471 lower-envelope grid points at 0.10 m plus 78 hull vertices, computed from the hull Chrono
  builds (the one-object OBJ gives one convex hull).
- The surface is the BMP, read in Chrono's node-on-edge convention.
- Negative = the hull is below the original surface. On soil this can happen because the chassis is not coupled.
- The mesh sha256 is checked against the runtime data folder (`647358e1…`, identical locally and in both cluster
  builds).
- The point transform is checked against Chrono's own `TransformPointLocalToParent` (difference 0.0).
- The recorded chassis vertical speed equals the rich-telemetry value.
- Output:
  - `vehicle_extra.npz`: `belly_clearance_min_m`, `belly_argmin_point`, `belly_points_below_surface`,
    `chassis_vz_mps` per frame, plus the sample points;
  - `outcome.json` `vehicle.belly`: anchor clearance, minimum, frames below the surface, time of the first one,
    vertical speed at the anchor, largest |vertical speed| in the first second.

## 2. HMMWV bit-identity (task 2)

- **Case:** `f104_v2_group_0000` `route_00`, 12 s horizon.
- **Rigid:** `--local`, single thread.
- **Soil:** production `crm_main.json` (0.08 m, 1 ms), `--episode-seed 1`, under the lock.
- **Records:** `e2/bitid/run_rigid.sh`, `run_soil.sh`, `cmp_*.json`.

| Pair | Arrays compared | Result |
|---|---|---|
| `gen_collect_ext.py` vs `ag_gen_collect_ext.py` (no switch) | every array in trajectory, anchor_state, command_reference, rich_telemetry, rich_intervals | identical |
| same vs `ag_gen_collect_ext.py --vehicle hmmwv` run with `NEDM_VEHICLE=gator` in the environment | same | identical (the argument wins) |
| `crm_collect.py` vs `ag_crm_collect.py` (no switch) | trajectory, anchor_state, command_reference, crm_extra | identical |
| `crm_collect_ext.py` vs `ag_crm_collect.py --base crm_collect_ext --vehicle hmmwv` | same | identical |
| rigid re-check on the final `ag_vehicle.py` (`rigid_ag_final`) | all | identical |

In every pair, the JSON files differ only in wall time and in the hashes of the files that contain wall time. The
local soil run is also repeatable: `crm_collect.py` and `crm_collect_ext.py` in native mode give identical arrays.

## 3. Soil wheel calibration (task 3)

**Protocol** (`scripts/ag_soil_calibrate.py`, records in `e2/calib/calib.jsonl`):
- A flat 16 x 16 m arena (constant-grey BMP), built by the production `crm_collect.build_crm` with
  `crm_main.json`: 0.08 m spacing, 1 ms step, 0.24 m soil.
- The vehicle is built exactly as the soil collectors build it, and spawned at ground + 0.75 m (HMMWV) or
  + 0.35 m (Gator).
- 0.8 s at full brake (the settle), then 2 s at full throttle, straight.
- Sinkage per wheel = stock tyre radius - (spindle height - ground), as in the collectors. Negative = the tyre rides
  above the surface.
- "Anchor" = the last 0.2 s of the settle; "driving" = the mean over the throttle phase.

| Wheel geometry | Markers per wheel | Anchor sinkage F / R (m) | Driving sinkage F / R (m) | Speed after 1 s / 2 s (m/s) | Vertical speed at 0.8 s (m/s) |
|---|---|---|---|---|---|
| HMMWV production mesh | 207 | +0.017 / -0.004 | -0.016 / -0.012 | 2.14 / 4.15 | +0.10 (still settling; +0.054 / +0.026 at 2 s) |
| Gator cylinders, stock radius (0.286 / 0.318) | 212 / 295 | -0.037 / -0.018 | -0.091 / -0.094 | 2.25 / 3.87 | -0.005 |
| Gator, radius - 0.05 | 140 / 200 | -0.029 / -0.015 | -0.047 / -0.055 | 2.82 / 3.78 | -0.004 |
| Gator, radius - 0.07 (S2's guess) | 128 / 180 | -0.016 / -0.012 | -0.026 / -0.036 | 2.80 / 3.97 | -0.005 |
| **Gator, radius - 0.09 (chosen: 0.19575 / 0.2275)** | 88 / 170 | **-0.001 / +0.001** | **-0.005 / -0.018** | 2.69 / 3.63 | +0.004 (fully settled: 0.0001 at 2 s) |
| Gator, radius - 0.11 | 80 / 150 | +0.017 / +0.017 | +0.015 / +0.001 | 2.51 / 3.59 | +0.037 |

**Choice:** radius - 0.09 m on both axles, stock widths (0.254 m front, 0.3048 m rear). It matches the HMMWV in both
phases, as axle means:
- anchor: 0.000 against +0.006 m;
- driving: -0.011 against -0.014 m (verifier correction, was -0.012).

Per-axle differences are within the HMMWV's own front/rear spread (about 0.02 m). Real-time factor on this patch:
0.72 (HMMWV 0.75).

**Deviation from the task text.** The task named the band -0.02 to -0.04 m as the HMMWV's production value. That band
is S2's end-of-throttle reading on its small patch, not a settled value. Measured here:
- HMMWV on flat soil: +0.017 / -0.004 m at the anchor, -0.016 / -0.012 m while driving.
- 300 f104 `collect_v1` anchors: mean +0.03 m.

Matching the literal band would need radius - 0.05 to - 0.07. The Gator would then ride 1.5-3 cm higher than the
HMMWV when settled and 2-4 cm higher while driving. So the calibration matches the HMMWV measured under the same
protocol. The radii can be overridden per run with `--gator-soil-radius-front/-rear` and `--gator-soil-width-front/-rear`.

Why the stock-radius cylinder is wrong:
- Soil keeps about one spacing away from the markers, so the effective radius grows by about 0.1 m (S2).
- On the f104 crater route below, stock-radius cylinders did not move the Gator at all: the terminal pose was
  within 0.14 m of the start.

## 4. Local checks on the real f104 heightmap (task 4)

### 4.1 Rigid

Setup:
- `e2/rigid_f104/run.sh`, summary in `e2/rigid_f104/summary.json`.
- 4 groups: hill 0000, crater 0005, hill cross-slope 0012, crater cross-slope 0017. Each at constant 2 / 4 / 6 m/s
  with lateral offset 0, for 12 routes.
- 120 s horizon, `--local`.
- The Gator runs through `ag_gen_collect_ext.py --vehicle gator`; the HMMWV runs through the unmodified
  `gen_collect_ext.py`.

| | Gator | HMMWV |
|---|---|---|
| Launch check / ground-height check | 12/12 / 12/12 | 12/12 / 12/12 |
| Finite states and terminal states | 12/12 | 12/12 |
| Goal reached | 11/12 (0017 at 2 m/s: prolonged blockage) | 11/12 (the same route) |
| Time to goal, other routes | 0.2-0.5 s faster than the HMMWV; 0017 at 4 m/s: 14.8 s against 39.1 s | |
| Largest chassis contact force | 0 N on every route | 0 N |
| Lowest belly point over the episode | +0.04 to +0.11 m (anchor +0.14 to +0.15) | n/a |
| Vertical speed at the anchor / largest in the first 1 s | -0.035…+0.14 / ≤ 0.21 m/s | -0.17…+0.08 / ≤ 0.35 m/s |
| Forward speed at the anchor | -0.39…+0.18 m/s | -0.10…+0.04 m/s |
| Largest roll / pitch | 8-31° / 17-27° | 6-28° / 17-25° |
| Wall time, 8 single-thread runs in parallel on 16 cores | **19 s + 1.15 s per simulated s** | 22 s + 1.28 s per simulated s |

Notes:
- **Backward creep at the anchor.** On group 0000 the Gator rolls back at 0.39 m/s at the anchor, against 0.10 for
  the HMMWV. The settle brake command comes from the follower's speed controller and is small, and the Gator brakes
  only its rear axle.
  - The launch limit is 1 m/s, so the check passes.
  - The value enters the anchor state that the models read as context.
  - The label rule "slid backwards" counts only after 1 s, so this creep does not trigger it.
- **Grounding.** The chassis never touched the ground on these 12 routes, so grounding (S2 risk 1) did not appear
  here. The cluster pilot should keep watching `max_chassis_contact_resultant_n`.

### 4.2 Soil

Setup:
- `e2/soil_f104/run.sh`.
- Production config, 120 s horizon, under the lock.
- Designed routes with a known HMMWV `collect_v1` outcome.

| Route | Gator | Where the Gator stalled | HMMWV (`collect_v1`) |
|---|---|---|---|
| 0005 crater, 4 m/s (`route_01`) | prolonged blockage at 34 s | 31.6 of 46.7 m; 24° climb out of the crater; peak speed 5.5 m/s before it | goal at 13.1 s |
| 0005 crater, 2 m/s (`route_00`) | prolonged blockage at 34 s | 30.3 m; 19° climb | soil breakthrough at 22.6 s, at almost the same spot |
| 0000 hill, 6 m/s (`route_02`) | prolonged blockage at 34 s | 16.6 of 40.1 m; 19° climb; peak speed 2.9 m/s | goal at 8.9 s |
| 0012 hill cross-slope, 6 m/s (`route_02`) | prolonged blockage at 34 s | 13.4 of 37.4 m; 19° climb; peak speed 3.3 m/s | goal at 9.0 s |

In every stall the rear wheels spin at about 9-13 rad/s (median 9.4-10.4; verifier correction, was 10-15) while the front wheels stand still. The largest wheel sinkage,
by the collectors' per-wheel measure, is 0.17-0.18 m, below the 0.30 m breakthrough threshold. All four Gator soil
episodes:
- pass the launch check: chassis reference 0.29-0.33 m above the surface; vertical speed at the anchor -0.008 to
  +0.056 m/s;
- have finite states throughout;
- carry the belly block. The belly falls below the original surface on 0 %, 8 %, 25 % and 20 % of the frames
  (lowest -0.036 m), once the Gator is bogged.

Cost:
- The episode loop takes 1.87-1.97 wall s per simulated s on the RTX 5090, against 1.85 for the HMMWV. Both use the
  same 4.0 M particles, and the build takes 3-5 s.
- A 34 s Gator episode took 69-74 s of wall time in total.

**The stall does not come from the wheel geometry** (`e2/soil_f104/sens/`, `route_01`):
- With the stock Gator tyre meshes (diagnostic `--gator-soil-mesh`), the Gator stalls at the same point (x 15.89 m
  against 15.86 m) and digs through the soil (soil breakthrough at 27.8 s).
- With stock-radius cylinders it does not move at all.

This is rear-wheel-drive traction on this soil: 906 kg, about 60 % of the weight on the driven axle, climbs of 19° or
more. **Consequence for task B:** designed hill and crater routes may fail almost always on soil with the Gator. The
cluster soil pilot (PLAN section 3: 24 groups x 6 routes; best drawn from every stratum) must measure the failure rate before the
15,235-id collection is committed. PLAN risk 6 already accepts a saturated result as an answer.

**Soil checks still valid for the Gator:**
- **Breakthrough rule** (stock tyre radius - (spindle height - surface) > 0.24 + 0.06 m for 5 frames): it uses the
  stock radius, which the calibration made the effective contact radius, so it means the same as for the HMMWV: the
  tyre bottom is 6 cm below the soil floor. It fired in the stock-mesh diagnostic run.
- **Launch check** (chassis 0 to 1.2 m above the surface, roll and pitch ≤ 25°): the Gator sits at 0.29-0.33 m and
  passes.
- **Belly.** On the Gator the belly reaches the original surface long before the breakthrough depth. The belly block
  records this; it is not a stop rule.

### 4.3 Other modes (`e2/modes/`, `e2/gates/`)

- **Rigid `branch_auto` as the Gator.** The replay and both continuations were re-launched through the wrapper with
  `--vehicle gator`. The replayed pose matched the recorded Gator run exactly (0.000 m).
- **Soil, `crm_collect_ext` as the Gator.** Native and pid_held modes run. Its native mode is bit-identical to the
  `crm_collect` base as the Gator.
- **The two switches agree.** A run with `--vehicle gator` and a run with `NEDM_VEHICLE=gator` give identical arrays.
- **Gates without `--local`** (a temporary source root with the `crm_improve` manifest, whose 9 frozen-file hashes
  equal this worktree's):
  - The Gator is refused with a fingerprint that has no Gator files, and without any fingerprint.
  - With the Gator fingerprint it runs, and both gates read "checked".
  - HMMWV runs with a Gator-only option are refused.
- **Determinism.** The final `ag_vehicle.py` reproduced an earlier Gator soil episode bit for bit.

## 5. How to run each collector as the Gator

Common local environment:

    cd /home/harry/NeDM-traverse_mppi
    export PYTHONPATH=/home/harry/chrono/build/bin:src:scripts     # or the conda nedm python with PYTHONPATH=src:scripts
    PY="/usr/bin/python3.12 -P -u"
    C=$PWD/artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases
    CFG=$PWD/artifacts/traverse/crm_f104_v1/configs/crm_main.json

Rigid (any `gen_collect_ext` mode and flag passes through):

    OMP_NUM_THREADS=1 $PY scripts/ag_gen_collect_ext.py --vehicle gator --source-root . \
      --case $C/f104_v2_group_0000.json --route $C/routes/f104_v2_group_0000/route_00.json \
      --chrono-data /home/harry/chrono/data --horizon-s 120 --local --out OUT
    # on the cluster: no --local; add  --runtime-fingerprint $G3/runtime/gator_runtime_fingerprint.json

Soil:

    OMP_NUM_THREADS=4 flock /tmp/luffy_crm.lock $PY scripts/ag_crm_collect.py --vehicle gator --source-root . \
      --case $C/f104_v2_group_0005.json --route $C/routes/f104_v2_group_0005/route_01.json \
      --chrono-data /home/harry/chrono/data --crm-config $CFG --horizon-s 120 --episode-seed <md5 id> --out OUT
    # branch / pid modes: add  --base crm_collect_ext --mode ...  (crm_collect_ext's own arguments)

**Cluster task rows.**
- **Soil** (`crm_worker.py`, `CRM_COLLECTOR=$G3/source/scripts/ag_crm_collect.py`):
  - Gator row: `"extra": ["--vehicle", "gator"]`.
  - HMMWV row: `"extra": ["--vehicle", "hmmwv"]` (recommended explicitly), or no extra.
  - The worker adds `--crm-config` because the file name contains `crm_collect`.
- **Rigid** (`gen_runner_g.py`, `GEN_COLLECTOR=$G3/source/scripts/ag_gen_collect_ext.py`):
  - Gator row:
    `"extra": ["--vehicle", "gator", "--runtime-fingerprint", "/work1/dannegrut/harry/experiments/arena_gator_20260925/runtime/gator_runtime_fingerprint.json"]`.
    `gen_array_g.sbatch` hard-sets `FDM_RUNTIME_FINGERPRINT` to the HMMWV file; the row argument overrides it for the
    Gator.
  - HMMWV rows: `["--vehicle", "hmmwv"]` or nothing. They then run the unmodified code path, so one collector serves a
    mixed task file.
- **Do not set `NEDM_VEHICLE` in production job environments.** It would silently turn every row without
  `--vehicle` into a Gator row. Always put the vehicle in the row.
- **Ids.** The Gator re-drives the HMMWV task ids. If Gator and HMMWV rows share an output folder
  (`CRM_OUT/runs/<id>`, `GEN_OUT/runs/<id>`), give the Gator rows distinct ids, for example
  `gator__f104_v2_group_0000_route_00`, and keep the HMMWV id in a `pair_id` field for the per-route comparison.

## 6. What must be staged on the cluster (for E3)

1. **Gator runtime fingerprint: done.**
   - File: `/work1/dannegrut/harry/experiments/arena_gator_20260925/runtime/gator_runtime_fingerprint.json`, sha256
     `599c514e84a28800e567d3b6260402ebce5d7ed6efeb1561168e2d81cc12ed77`. Local copy in `e2/runtime/`.
   - Built by `ag_runtime_fingerprint.py` (a copy is in `$G3/e2_tools/`).
   - Contents: all 108 entries of `fdm_f104_50h_20260909/pilot_runtime_412394.json`, re-hashed on the login node and
     all unchanged, plus 63 files under `chrono-build/data/vehicle/gator/`. 171 entries in total.
   - It keeps the HMMWV entries, so `gen_collect_ext`'s own check (it needs an `/vehicle/hmmwv/` entry) still passes;
     the wrapper additionally requires `/vehicle/gator/` entries.
2. **Source tree: one tree serves both vehicles.** No frozen file changed, so the HMMWV rows and the Gator rows can
   share `G3/source`.
   - rsync `src/` and `scripts/` from this worktree. They must include `ag_vehicle.py`, `ag_gen_collect_ext.py`,
     `ag_crm_collect.py`, `ag_gator_belly.json`, and E1's `gen_arenas.json`. Also rsync `assets/traverse/<arenas>`.
   - Add a fresh `source_manifest.json`. The 9 frozen files equal the `crm_improve_20260922` manifest hashes (checked
     again today). Four non-frozen scripts differ from that manifest because E1 is editing them: `ci_train.py`,
     `ci_a5data.py`, `ga_build_mixed.py`, `gen_arenas.json`.
   - Check on the login node: `ag_gen_collect_ext.py --check-only ...` (no Chrono import).
3. **Soil launcher.**
   - `crm_collect.sbatch` and `crm_launch.sh` fix `CRM_ROOT=crm_f104_20260916`, which is read-only here. Use an
     `ag_` copy with `CRM_ROOT=$G3`, so that `--source-root` is `$G3/source` and the wrapper, `crm_collect` and `nedm`
     all come from one tree.
   - Copy `crm_main.json` to `$G3/configs/crm_main.json`. It is used unchanged: its `tire_mesh` key names the HMMWV
     mesh, which the Gator ignores, and the Gator run records the cylinders in `vehicle.soil_wheel_geometry`.
4. **Builds.** Both cluster builds have `veh.Gator`, `CylinderShape`, `ChBodyGeometry` and `ChBody.GetIdentifier`
   (checked on the login node, import only). No Gator episode has run on the cluster yet; the cluster pilot is the
   first. Remember that rigid runs on AMD reproduce only within one node.

## 7. Deviations from the stock Gator, and other caveats

- **Soil coupling geometry.** Cylinders of stock radius - 0.09 m, stock widths, replace the stock tyre meshes. The
  stock meshes are not watertight and give ragged marker sets at 0.08 m (S2 2.5). The wheel bodies (RIGID_MESH:
  mass, inertia, collision mesh) are stock.
- **Scene settings, not vehicle changes:**
  - spawn height ground + 0.35 m;
  - chassis not coupled to the soil (as the HMMWV).
- **Rendering only.** The frozen blue roof marker (chassis + 0.95 m) sits inside the Gator cab. The collectors add no
  camera, so it has no effect on data.
- **Stock parts kept, with their known effects (S2):**
  - rear-only brakes: anchor creep up to 0.39 m/s;
  - limited-slip rear-wheel drive;
  - one-gear 14 kW engine;
  - TMEASY grip scaled 1.5 times on rigid ground;
  - wrong declared wheelbase and steering getters: unused by the PID follower.
  - Every earlier study's route limits and follower gains are unchanged, as the PLAN fixes.
- **Data columns.** 24 suspension telemetry fields are NaN for the Gator, because its suspensions are not double
  wishbones. The collectors accept this; any downstream code that asserts all telemetry is finite must skip those
  fields. The 17-column state is finite.
- **Cost.**
  - Rigid: the Gator runs about 10 % cheaper per simulated second than the HMMWV.
  - Soil: equal per simulated second. However, Gator soil episodes that stall run to the 34 s stop rule, against
    9-13 s for HMMWV successes, so Gator soil hours per task will be higher than the HMMWV's.
