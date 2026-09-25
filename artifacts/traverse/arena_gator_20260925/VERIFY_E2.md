# VERIFY E2: the Gator vehicle switch (independent check, 2026-09-25 01:56-02:10)

**Verdict: PASS.** Every claim I re-tested holds. I found no defect in the switch. There are three small fixes and
notes (section 8) and one item E3 must act on before any Gator row runs: the cluster source tree holds an older copy
of `ag_vehicle.py` (section 7).

All my scripts and records are in `verify_e2/`. Local Chrono is the source build
(`PYTHONPATH=/home/harry/chrono/build/bin:src:scripts /usr/bin/python3.12 -P -u`). Every soil run held
`flock /tmp/luffy_crm.lock`. I wrote my own comparison and check scripts instead of reusing the builder's helpers.
The only builder code I ran is the code under test: the wrappers and `ag_soil_calibrate.py`.

## 1. The HMMWV path is bit-identical (re-run on routes the builder did not use)

| Pair | Episode | Arrays compared | Result |
|---|---|---|---|
| `gen_collect_ext.py` vs `ag_gen_collect_ext.py --vehicle hmmwv`, run with `NEDM_VEHICLE=gator` in the environment | rigid, crater cross-slope `f104_v2_group_0017/route_01`, 4 m/s, 25 s, `--local` | 262 arrays in 5 npz files | all byte-identical |
| `gen_collect_ext.py` vs `ag_gen_collect_ext.py` with no switch and no environment variable | same | 262 | all byte-identical |
| `crm_collect.py` vs `ag_crm_collect.py --vehicle hmmwv`, run with `NEDM_VEHICLE=gator` | soil, hill cross-slope `f104_v2_group_0012/route_02`, 6 m/s, production `crm_main.json`. The episode ends at the goal at 9.0 s, so goal handling and the final files are covered too. | 31 arrays in 4 npz files | all byte-identical |

- **JSON files.** The only differences are wall time, the soil real-time factor and build time, and the two
  completion-file hashes of the files that contain wall time. `collection_request.json` is identical, including its
  collector hashes.
- **Tools.** Comparison: `verify_e2/cmp_runs.py`, which checks dtype, shape and raw bytes of every array and diffs
  every JSON leaf. Run scripts: `run_bitid_rigid.sh` and `run_bitid_soil.sh`. Results: `bitid_*/cmp_*.json`.
- **Explicit argument beats the environment variable.** Both runs with `NEDM_VEHICLE=gator` still produced HMMWV
  episodes, because `--vehicle hmmwv` was given.
- **An HMMWV run through the wrapper leaves no trace of the wrapper.** That follows from the design: the wrapper
  installs nothing in HMMWV mode.

## 2. Gator mode really builds the stock Gator

`verify_e2/probe_gator.py` runs the real wrapper collectors with their normal arguments. The only addition is a hook
around `ag_vehicle.create_gator` and `make_build_crm` that records what was built.

Probe runs:
- **Rigid:** `0012/route_01` for 15 s. The Gator was selected through the environment variable only (no `--vehicle`).
- **Soil:** `0021/route_05` for 12 s, with `--vehicle gator`.

Results are in `gator/*.probe.json`.

| Check | Rigid | Soil |
|---|---|---|
| Wrapper class, `isinstance(veh.Gator)` | `Gator`, True | `Gator`, True |
| Total mass / chassis mass | 906.2 kg / 800 kg | 906.2 kg / 800 kg |
| Driveline | `GatorCustomDriveline` (stock SIMPLE), driven axles [1] = rear only | same |
| Brakes | rear axle only (`BrakeSimple` RL, RR); front none | same |
| Engine / gearbox | `EngineSimple` / `AutomaticTransmissionSimpleMap` (stock one-gear class) | same |
| Spawn height | config z = arena surface at the start + **0.350 m** | + **0.350 m** |
| Tyres / chassis collision | TMEASY (0.28575 / 0.3175 m), 1 hull shape, collision on | RigidTire, chassis has no collision model (not coupled to the soil) |
| Soil wheel geometry handed to `AddRigidBody` | n/a | front cylinder r 0.19575 m, length 0.254 m; rear r 0.2275 m, length 0.3048 m; both on the spindle y axis, centred; swap order [0, 0, 1, 1] |
| Soil marker count per wheel (CRM's own count) | n/a | 88, 88, 170, 170 (the builder's calibration counts) |
| Evidence from the recorded data | front driveline torque 0 on every frame, rear up to 1,291 N·m; summed tyre load at the start ≈ 8.5 kN (906 kg; an HMMWV would read ~25 kN) | rear wheels at 21-22 rad/s against 12 rad/s at the front (rear-wheel drive slipping) |

The builder's records for `branch_auto` (replay + 2 continuations) and soil `crm_collect_ext` (native and pid_held)
all carry the Gator vehicle block and `vehicle_extra.npz`. The builder's report says "branch and policy modes still
work"; policy mode itself was not exercised, only branch_auto, native and pid_held. Nothing in the plan needs it.

## 3. The soil worker forwards the soil configuration to the wrapper

- **By reading the code:** `crm_worker.py:98` appends `--crm-config` when the string `crm_collect` appears in
  `CRM_COLLECTOR`. `ag_crm_collect.py` contains it.
- **By running it:** I ran one Gator row through the unmodified `crm_worker.py` (`run_worker_gator.sh`: a temporary
  CRM_ROOT, `CRM_COLLECTOR=scripts/ag_crm_collect.py`, `CRM_CONFIG=configs/crm_main.json`, row id
  `gator__f104_v2_group_0005_route_02`, row arguments `["--vehicle", "gator", "--horizon-s", "10"]`).
  - The episode completed.
  - `collection_request.json` holds step 0.001 s. That value comes from `crm_main.json`; the code default is
    0.0005 s, so the configuration arrived.
  - The outcome carries the Gator block with the calibrated cylinders.
  - The prefixed id works as a run folder.
  - Its start clearance, 0.19631634 m, equals the builder's episode from the same start to every digit, which shows
    the build is repeatable.

## 4. Soil wheel calibration (flat patch, re-run once)

`run_calib.sh` re-runs `ag_soil_calibrate.py` with the default (calibrated) Gator wheels and with the HMMWV. Results
are in `calib.jsonl`.

| | Settled, last 0.2 s: front / rear / axle mean (m) | Full throttle: front / rear / axle mean (m) | Speed after 1 s / 2 s (m/s) |
|---|---|---|---|
| Gator, re-run | -0.0007 / +0.0010 / **+0.000** | -0.0046 / -0.0183 / **-0.011** | 2.69 / 3.63 |
| HMMWV, re-run | +0.0166 / -0.0041 / **+0.006** | -0.0159 / -0.0123 / **-0.014** | 2.14 / 4.15 |

- **Gator:** all 56 recorded rows are bit-identical to the builder's chosen setting (`gator_rm09`).
- **HMMWV:** equal to 4 decimals (end speed 4.1475 against 4.1483 m/s).
- **The builder's claim holds:** the calibrated cylinders put the Gator at the HMMWV's ride height both when settled
  and when driving.
- **Deviation from the plan, accepted.** PLAN 1.5 names a band of -0.02 to -0.04 m, but the HMMWV does not sit in
  that band when settled:
  - on flat soil it sits at +0.006 m;
  - on real f104 starts it is higher still. On 300 random `collect_v1` HMMWV episodes I measured the start sinkage
    at +0.030 m mean (+0.029 median). This is a rough measure, using the ground height under the chassis, and it
    agrees with the builder's "+0.03 m".

  The plan's intent is "match the HMMWV", and the builder matched it under the same protocol.
- **Small correction.** The throttle-phase axle mean is -0.0115, which rounds to -0.011; the notes said -0.012. I
  fixed it in NOTES_E2.

## 5. Belly-clearance diagnostic

`verify_e2/check_belly.py` uses scipy and my own code; results are in `check_belly.json`.

**The sample points.**
- The collision OBJ has one object. Chrono builds one convex hull per object (checked in the source:
  `Gator_Chassis.cpp:73`, `ChBodyGeometry::CreateCollisionShapes`).
- Collision shapes are placed in the chassis reference frame (`ChBody::GetCollisionModelFrame` returns
  `GetFrameRefToAbs`). That is the frame the diagnostic uses.
- All 549 points lie on the hull surface (largest distance 5e-7 m).
- All 471 grid points are on the lower envelope: 1 mm below each is outside the hull, 1 mm above is inside.
- The 78 extra points are exactly the hull's vertex set.
- The lowest point is -0.1332 m. A finer 0.05 m probe grid finds nothing lower.
- The OBJ sha256 matches the local file and both cluster builds (`647358e1…`).

**Per-frame values.** I recomputed them from the recorded chassis position and orientation, with my own rotation
and the arena heightmap.

| Episode | Recomputed vs recorded | Recorded minimum in `outcome.json` matches the npz |
|---|---|---|
| My rigid Gator episode | 7e-9 m | yes |
| Builder rigid `0000/route_00` | 7e-9 m | yes |
| Builder rigid `0017/route_01` | 7e-9 m | yes |
| My soil Gator episode | 5e-8 m (float32 inputs) | yes |

- On the rigid episodes, the recorded chassis vertical speed equals the rich-telemetry value exactly.
- `vehicle_extra.npz` is listed in `episode_complete.json` with the correct hash, in both worlds.

**Surface convention.** The diagnostic samples the heightmap at 511/512 of the world position.
- Against Chrono's own raycast heights (`native_height_check.json`, 50 points), this convention gives an rms error
  of 3.5 mm; plain sampling gives 10 mm. So the convention is the right one.
- Plain sampling would shift the belly values by 1.3-2.8 cm.

## 6. Provenance fields and gates

**Vehicle block.**
- It is present in all five rigid JSON files named in the notes and in the three soil ones. It records:
  - the switch name, `ag_vehicle.py` sha, wrapper path and sha;
  - spawn offset 0.35 against the frozen 0.75;
  - driveline, brakes, engine, tyres, chassis coupling;
  - the sha of all 63 Gator data files;
  - the runtime fingerprint (gated runs);
  - the soil wheel geometry with the nominal tyre next to it;
  - the belly summary with the sha of the points file and of the OBJ.
- My three Gator runs record the current `ag_vehicle.py` (`072716ee…`) and wrapper hashes.
- The 63 local Gator data hashes equal the 63 cluster entries of the fingerprint.

**Cluster fingerprint.**
- On the login node I re-hashed all 171 entries of `G3/runtime/gator_runtime_fingerprint.json` (file sha
  `599c514e…`, matches the notes): 0 mismatches, 0 missing.
- All 108 base entries equal `pilot_runtime_412394.json`.
- All 63 Gator files on disk are listed.

**Refusal paths.** 11 tests (`run_gate_tests.sh`, `gates/gate_tests.out`). Every one fails before any Chrono work and
creates no run folder:
- a Gator rigid run with no fingerprint and no `--local`;
- a Gator rigid run with an HMMWV-only fingerprint, given as an argument, through the environment variable, and
  through the environment variable with `--local`;
- an HMMWV run given `--runtime-fingerprint`;
- a soil run given `--runtime-fingerprint`;
- an HMMWV soil run with a Gator radius option;
- `NEDM_VEHICLE=polaris`;
- `--vehicle polaris`;
- a bad `--base`;
- an implausible radius (0.9 m).

The positive path with both gates on is the builder's `e2/gates/G3`. Both gates read "checked", and the fingerprint
recorded is the one above.

**No hash-gated frozen file changed.**
- The 9 files that `gen_collect.SOURCE_FILES` hash-checks equal the `crm_improve_20260922/source/source_manifest.json`
  hashes (read from the cluster).
- Of all 814 files in that manifest, only E1's four additive edits differ from the worktree: `ci_train.py`,
  `ci_a5data.py`, `ga_build_mixed.py`, `gen_arenas.json`.
- `git status` shows no tracked file modified by E2. The collectors (`crm_collect*.py`, `gen_collect*.py`,
  `crm_worker.py`, `gc_control.py`, `gen_runner_g.py`) are unchanged, and are the same in `G3/source`.
- No file in the read-only artefact folders is newer than PLAN.md.

## 7. For E3 (not changed by me: E3's live staging tree)

- **Stale copy on the cluster.** `G3/source/scripts/ag_vehicle.py` is the 01:30 copy (sha `3e828142…`, 20,029
  bytes). The final local file is `072716ee…` (21,233 bytes).
  - The difference is only the diagnostic `--gator-soil-mesh` option and one provenance field
    (`calibrated_default`). The physics of production Gator rows would be the same.
  - Their provenance would, however, record a file that is not the verified one.
  - Re-sync `ag_vehicle.py` before freezing the dispatcher for the Gator relaunch (NOTES_E3a / e3/README step 4).
    The other E2 files in `G3/source` (`ag_gen_collect_ext.py`, `ag_crm_collect.py`, `ag_gator_belly.json`) are
    identical to the worktree.
  - Nothing currently running imports it: the soil HMMWV rows call `crm_collect.py` directly, and the rigid rows call
    `gen_collect_ext.py`.
- **HMMWV rows need an explicit vehicle once they share the dispatcher.** The HMMWV rows in `soil_v1.json` and
  `rigid_hmmwv_v1.json` carry no `extra`. Once those rows run through the dispatcher (`ag_crm_collect.py` /
  `ag_gen_collect_ext.py`), a stray `NEDM_VEHICLE=gator` in a job environment (`--export=ALL`) would silently turn
  them into Gator runs in the HMMWV folder. Either:
  - give new HMMWV rows `["--vehicle", "hmmwv"]`, or
  - `unset NEDM_VEHICLE` in the job script.

  E4 should also assert that every `gator__*` run has `vehicle.name == "gator"` and that no HMMWV run has a vehicle
  block.

## 8. Small fixes and notes

1. **NOTES_E2 wording (fixed, marked "verifier correction").**
   - The throttle-phase Gator sinkage is -0.011 m, not -0.012 (two places).
   - The rear-wheel spin during the soil stalls is about 9-13 rad/s (median 9.4-10.4), not 10-15.
2. **The builder's soil stall finding is consistent with the records.**
   - On all 4 local f104 soil episodes the Gator stops for good at 7.5-18.9 s.
   - The front wheels stand still (median |spin| < 0.1 rad/s) while the rear wheels spin.
   - The episodes end on the 34 s stop rule. The stock-mesh diagnostic ends in soil breakthrough at 27.8 s.
   - My two extra soil runs were shorter (10-12 s) and still moving at the end, so they neither add to nor contradict
     it.
   - The cluster soil pilot should measure this failure rate first, as the builder says.
3. **Suspension telemetry is not finite for the Gator.** In the rich telemetry, 24 suspension fields are NaN on every
   Gator frame (the HMMWV has only the 2 end-of-episode command fields NaN, as the Gator also has). The 17-column
   state and the terminal state are finite. Any dataset step that requires all rich telemetry to be finite must skip
   those 24 fields.

## Records (`verify_e2/`)

| What | Where |
|---|---|
| Bit-identity | `bitid_rigid/`, `bitid_soil/` (runs, logs, `cmp_*.json`), `run_bitid_*.sh`, `cmp_runs.py` |
| Gator build probe | `gator/` (rigid + soil runs, `*.probe.json`), `probe_gator.py`, `run_gator_probes.sh` |
| Worker forwarding | `worker_root/` (task file, run, logs), `worker_gator.log`, `run_worker_gator.sh` |
| Calibration | `calib.jsonl`, `calib_*.log`, `run_calib.sh` |
| Belly checks | `check_belly.py`, `check_belly.json` |
| Gates | `run_gate_tests.sh`, `gates/` |
