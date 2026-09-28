# VERIFY M2: independent check of the M113 module (2026-09-28, 00:27-00:55 CDT)

**Verdict: PASS. GO for admitting `m113` and `m113_g4` to the smoke. I found no blocking defect.** Admission still
needs the orchestrator decisions at the end, above all the plan amendment for the brake and a stage-0 rule the M113
can pass.

Scope: I read `scripts/ov_m113.py` (sha `a61f29ad`), `scripts/ov_m113_check.py` (`62d59d98`), `configs/crm_m113.json`
(`0bba40c0`), `assets/traverse/vehicles/ov_m113/` and the builder's notes and job outputs. I re-ran the key checks and
tried edge cases.
- No cluster job; the cluster was only read over ssh.
- Local soil: 72 s of wall time, both runs under `flock /tmp/luffy_crm.lock`.
- Local rigid: 137 s on the CPU.
- My files are in `verify_M2/`. `NOTES_M2.md` is the builder's file, so I left it untouched; this file is my notes file.

## What I re-checked, with results
1. **Hard rules: all hold.**
   - `git diff HEAD` is empty on tracked files. The only files changed since 23:43 are new `ov_*` files,
     `configs/crm_m113.json` and `assets/traverse/vehicles/ov_*`.
   - `NEDM_VEHICLE` appears only in the job scripts' refusal guard.
   - sacct shows exactly two M2 jobs, 441588 and 441589: mi3501x, 1 node, 45-minute limit, 8 min 48 s and 5 min 46 s.
   - `PLAN.md` hash is unchanged (`e166e1fb`).
2. **Soil config.**
   - `crm_m113.json` differs from `crm_f104_v1/configs/crm_main.json` in one field only: `step_s` 0.001 -> 0.0005.
   - That source file has sha `90cd049e`, the same as `G3/configs/crm_main.json` on the cluster, i.e. the file the
     K3 soil jobs used.
   - Its `name` field still says "step1ms". No script reads that field.
3. **Rigid checks, re-run locally with the final file: bit for bit the builder's numbers.**
   - Poses equal `veh.M113`'s (difference 0.0 m).
   - The JSON transmission with the stock ratios equals the C++ one (0.0 m).
   - Braked on 10° rigid ground, the stock brake slides at 0.89 m/s (the sprocket turns 6.7 rad).
   - With the shafts brake the vehicle slides at 0.24 m/s with the sprocket locked (0.24 rad). It slides as a sled,
     so rigid M113 rows are rightly excluded.
4. **Chrono source: the builder's diagnosis holds.**
   - Single-pin shoes always get `M113_BrakeSimple` (`M113_TrackAssemblySinglePin.cpp:62,71`).
   - Chrono's simple brake (`ChLinkLockBrake.cpp:66-120`) sets its "stick" flag but never reads it. So it is a Coulomb
     torque that flips sign every step and never locks.
   - The build sequence matches `M113::Initialize` (`M113.cpp`), including the wrapper defaults.
   - The ratios-/4 JSON copies every value of the C++ transmission except the ratios.
   - The pad bottom sits at the shoe's pad bottom face (z = -0.03 m). The belly points match `M113_Chassis.cpp`.
   - The steering-to-brake split matches `ChTrackDrivelineBDS::CombineDriverInputs`. The powertrain reads only the
     throttle, so passing it zero steering and braking is safe.
5. **The cluster drives, copied read-only and QA re-run: all pass.**
   - 4/4 pass `crm_qa`, with a finite (n, 17) state, a passed launch check and the vehicle block.
   - Wall seconds per simulated second: M113 7.78 / 7.78 / 7.90, Gator 1.89, i.e. 4.1-4.2 times the Gator.
   - Route tracking, which the builder did not report: M113 cross-track RMS 0.66 / 0.22 / 0.20 m, against 0.49 m for
     the Gator on the same route. Steering reverses only 0.06-0.45 times per second, so there is no oscillation.
   - The cluster ran file `94177c4e`. The final file differs only in one provenance field, which my local runs
     exercised.
6. **Interface edge cases: all pass.**
   - `install()` returns None, and patches nothing, for gator, hmmwv, polaris or no name.
   - It refuses `crm_collect_ext`.
   - It strips its own flags; a duplicated flag reaches the collector and fails loudly.
   - Every swapped function keeps a link to the original.
   - The dispatcher contract of M1 (a dict with scene_hook, frame_hook, argv) matches.
7. **Two adversarial f104 drives through the dispatcher (local, final file): both pass.**
   - Worst sample-A start for ground under the track: group 0594, whose ground rises 0.19 m, so the rear pads spawn
     inside the soil. The `m113` launch check passed (0.13 m/s, pitch 1.9°) and QA passed. It reached 2.0 m/s 1.2 s
     after the settle and 3.1 m/s at 4 s.
   - Most uphill-facing sample-A start: group 0362, +3.8°. The `m113_g4` launch check passed (0.21 m/s, pitch -6.4°)
     and QA passed.
   - Across all 144 sample-A starts, the rise under the track is median 0.07 m and at most 0.19 m (`start_rise.json`).

## Findings (none blocking)
- **F1 (minor): the brake is not capped the way Chrono's is.**
  - `ov_m113.py:280` sets the torque limit to 10,000 N·m x (braking + steering). Chrono's shafts brake caps its input
    at 1 (`ChShaftsClutch.h:62`).
  - So when braking + |steering| > 1, the inner brake reaches up to 20,000 N·m.
  - How often: 0 frames in the 3 cluster M113 drives, and 0.013 % of 198,688 Gator pilot frames (same follower).
  - Fix: `min(per_side, 1.0)`. It changes nothing when the sum is <= 1.
- **F2 (fairness caveat; review R1 S7(c)): the soil-breakthrough stop is effectively off for the M113.**
  - At rest the pads float 0.09-0.16 m above the undisturbed surface. The HMMWV tyre bottom rests 0.05 m below it,
    the Gator's 0.01 m below.
  - So a pad must sink about 0.4 m below its resting height before the rule fires, which is deeper than the 0.24 m
    soil layer.
  - Also, only the lowest pad of each quarter is tested. On a slope that is the downhill end, not the most sunk pad.
  - Effect so far: in the 3 full cluster drives (90 simulated s, including stalls at 19-31° pitch) the pads never went
    below the surface (maximum sinkage 0.0). No stop would have changed.
  - Options: declare it as a caveat, or fix it in one round: measure from each quarter's height at the end of the
    settle and test the most sunk pad (about 15 lines and one short local soil run).
- **F3 (caveat; R1 S7(d)): the front-sprocket reference gives the M113 a small head start.**
  - Bound from stored drives: of the sample-A failures, 0 of 135 (Gator) and 1 of 100 (HMMWV) ended 2.5-4.5 m from
    the goal; over all of collect_v1, 53 of 10,368.
  - So the 2 m head start flips at most about 1 of the 144 routes.
  - **Do not adopt R1's "recompute from the centre of mass" as worded.** Every M113 drive stops as soon as its front
    is 2.5 m from the goal, so the centre never gets there, and every M113 goal would count as a failure.
- **F4 (plan): stage 0 has not been run for the M113.** That means no flat-soil 2 / 4 / 6 m/s drive, no half circle,
  and no full brake from 6 m/s.
  - With the fixed pass marks of PLAN 2.1 the M113 would be excluded: its top speed on soil is about 4 m/s.
  - Under review R1-B2's exclusion rule the existing evidence passes:
    - launch check passed on 7 of 7 f104 drives;
    - state finite on 7 of 7;
    - the 10° hold drifts at -0.009 m/s, 0.03 m in 2 s;
    - 2 m/s is reached, though on f104 ground rather than on a flat patch.
- **F5 (cosmetic).**
  - `X_SPLIT_M` = -1.989 m is the left track's middle wheel. The right track's is at -2.074 m, so the right-track
    quarters are split 0.085 m off-centre.
  - The config's `name` says "step1ms".
  - `--m113-pad` given without a value raises IndexError instead of a clean message (diagnostic flag only).
- **F6 (caveat): only the M113 has a brake that can lock.** The Gator and the HMMWV keep Chrono's simple brake (the
  same non-locking kind; `Gator.cpp:46`, `ag_vehicle.py:133`). Braking occurs on about 0.7 % of Gator frames, and the
  Gator holds on 10° soil anyway.

## Contracts with the other modules
- **M3.**
  - M113 rows carry `config: configs/crm_m113.json` and `timeout_s` 4000.
  - The launcher raises the timeout and the claim margin to match.
  - The vehicle-block name equals the arm, and `vehicle_extra.npz` holds `belly_clearance_min_m`.
- **The M113 data folder is final.** The ratios-/4 JSON sha `d086aab8` equals the cluster-tested copy, so it may be
  frozen with the Polaris staging (this answers VERIFY_M3 point 3).
- **G4 is not staged yet** (it holds only `checks/` and `stage_dryrun/`).

## For the orchestrator to decide
1. **Amend PLAN 1.3 (section 9).**
   - The build is Chrono's C++ `M113_Vehicle_SinglePin`, identical by test to `veh.M113`.
   - The brake is replaced by the shafts construction (the only physics change). The pads are flat.
   - Wording: "an M113 with a locking brake" and, for `m113_g4`, "with 4x lower gearbox ratios".
2. **Adopt R1-B2 as the stage-0 rule for the M113.** Otherwise it cannot enter stage 1.
3. **F2: declare it or fix it.** F1 is an optional one-line fix. After any change: a new sha, one local soil run, then
   `ov_stage.sh --add scripts/ov_m113.py configs/crm_m113.json`.
4. **Time step:** decide whether to drive the Gator at 0.5 ms as a control (R1 S7(a); `ov_smoke_tasks.py
   --gator-half-step` exists).
5. **F3:** use the bound above rather than R1 S7(d) as worded.

Files in `verify_M2/`:
- scripts and results: `start_rise.py/.json`, `near_goal.py/.json`, `unit_checks.py/.json`, `rigid_rerun.jsonl`,
  `cluster_qa_rerun.jsonl`;
- the cluster drives, copied: `cluster_runs/`;
- the two local drives and their QA: `local/`.
