# VERIFY M1: independent check of the Polaris vehicle and the new soil dispatcher (2026-09-28 00:40 CDT)

Verdict: **pass, nothing blocking.** The Polaris path is ready for the cluster smoke. Three things need a decision
by the orchestrator (end of this file). No cluster job submitted. Local soil wall time used: 33 s, under
`flock /tmp/luffy_crm.lock`. My files: `checks/VERIFY_M1/` (tests, run folders, results).

## What I read

- `scripts/ov_vehicle.py`, `scripts/ov_crm_collect.py`, `scripts/ov_polaris_check.py`;
- the asset folder `assets/traverse/vehicles/ov_polaris/`;
- the builder's job script and outputs (`checks/M1/`);
- the frozen collector and worker (`crm_collect.py`, `crm_collect_ext.py`, `crm_worker.py`), and the Gator files
  `ag_vehicle.py` and `ag_crm_collect.py`, for comparison;
- the code of the other modules that talks to M1: M2 `ov_m113.install`, and the M3/M4 scripts `ov_stage.sh`,
  `ov_smoke_tasks.py`, `ov_eval_tasks.py`, `ov_smoke_analyze.py`, `ov_eval_index.py`, `ov_build_ds.py`.

## Checks I re-ran and their results

1. **Hard rules: all hold.**
   - `git diff` on tracked files is empty.
   - No file outside the study folder changed since 21:00, apart from other modules' new `ov_*` files and M2's
     `configs/crm_m113.json`.
   - `NEDM_VEHICLE` is never set. It appears only in refusals and `unset` / `env -u` lines.
   - The builder ran one cluster job, 441578 (mi3501x, 45 min limit, 9 min 35 s), with outputs under
     `G4/checks/M1/`.
   - The worktree `ov_vehicle.py` (3bc549ee), `ov_crm_collect.py` (4cc116ec) and asset MANIFEST (82cf309e) match
     byte for byte what job 441578 ran and what M3's staging dry run put on the cluster.
   - The check script differs from the cluster copy by three lines only: the temp-folder cleanup.
2. **Asset files, compared with Chrono's stock Polaris files.**
   - Every location in the vehicle file and the chassis file is shifted by exactly (+1.35763, 0, -0.42) m. That
     covers both suspensions, the steering, the anti-roll bar, the centre of mass and the driver position.
   - Mass and inertia are unchanged.
   - The three driveline variants differ only in the driveline line. The power-corrected gearbox has every ratio
     x 0.25 and conical ratios of 1.0.
   - The wheel has no offset, so a cylinder centred on the spindle is right.
   - The belly points (431) are in the re-framed chassis frame: the lowest point is 0.041 m below the axle line, as
     S1 reports.
3. **Rigid build of all four arms (local conda build): pass.**
   - Mass 1,378.334 kg.
   - Spindles at (+-1.358, +-0.616, -0.023 / -0.015) m from the reference point.
   - Launch height 0.362 m; vertical speed 0.0035 m/s at 0.8 s. Same numbers as the builder.
4. **The moved reference point does not change the physics (my own test).** Stock vehicle file against the
   re-framed one, built at the same pose on a rigid plane:
   - after the 0.8 s braked settle, positions agree to 2.5e-5 m;
   - on a gentle drive (throttle 0.3, then a turn), they agree to 0.3 mm at 3.8 s and 2.4 mm after 19 m;
   - at full throttle (18 m/s, then steering 0.5) the runs agree to 1.4 cm, until the vehicle rolls over. After the
     rollover they separate, as any chaotic run does.
5. **Edge cases of the switch (no Chrono needed; `checks/VERIFY_M1/v_unit.txt`): all behave correctly.**
   - Plain refusals, each with a clear error:
     - `NEDM_VEHICLE` set;
     - an unknown or upper-case vehicle name;
     - a Gator-only flag given with the Polaris;
     - a bad `--base`.
   - Asset folder: a changed file or a missing file is caught against the MANIFEST.
   - The file list for every arm resolves.
   - M113 dispatch:
     - if M2's file is missing, the run stops with a clear message;
     - a stand-in M2 module gets all 8 inputs;
     - its returned arguments and hooks are used;
     - a return value of the wrong type is refused.
   - The vehicle block is written only to the declared files.
   - The Polaris spawn height is ground + 0.40 m.
6. **Two short local soil runs through the dispatcher (3 s each): both pass.** Passing means `crm_qa` ok, launch
   check passed, a finite 60 x 17 state, and the vehicle block in all three JSON files.
   - Polaris on `0005_route_01`: launch height 0.385 m. Every array of the first 3 s is **identical** to the
     builder's local full run, so Polaris rows repeat exactly on one machine.
   - Polaris with `--base crm_collect_ext`, which the builder had not tried, on `0000_route_02`: launch height
     0.328 m, same as the cluster.
7. **Builder's cluster results: confirmed from the files.**
   - HMMWV and Gator rows through the new dispatcher have identical arrays to the frozen dispatchers on the same GPU.
   - The Gator rows are also identical to the stored run from another node.
   - The worker-driven Polaris row got `--vehicle polaris` and the production soil settings: 1 ms step, 0.08 m.
   - The Polaris file hashes recorded on the cluster equal the local ones (13 files).
8. **Contracts with the other modules: consistent.**
   - M2 returns the dict form the dispatcher expects.
   - Task rows (M3 and M4) carry exactly `['--vehicle', v]` and run through the default soil collector.
   - Staging copies and freezes the two M1 files, the asset folder, `ag_vehicle.py` and `ag_gator_belly.json`.
   - M3/M4 read the vehicle block's `name` and the belly data file, which M1 writes for the Polaris.

## Findings, none blocking

1. **`polaris_w08` never moved.** On `0005_route_01` it did not stall in the crater, as the builder's summary
   suggests: it sat at the start for 34 s at full throttle, 0.07 m of travel in total.
   - Evidence: the wheels are not turning, the engine is at near-zero speed giving its 185 N m idle torque, and the
     wheel forces are front -0.7/-1.4 kN, rear +0.7/+1.0 kN.
   - On flat soil it was slow to start: 0.5 s to reach 0.5 m/s, against 0.15 s for the primary.
   - Reading: at a standing start, the stock engine map and the reduction defect give about 2.8 kN m at the wheels,
     whatever the gear. The same torque on a 0.33 m wheel pushes 24 % less than on 0.25 m.
   - So in this model the wheel-size arm mixes a standing-start torque limit with soil behaviour. The primary
     0.25 m cylinder also gives about 1.3x the push that the real 0.33 m tyre would give at the same torque. This is
     the same approach as the Gator's cylinders; it belongs in the report's caveats.
2. **The soil breakthrough stop is stricter for the Polaris.** The frozen collector measures wheel sinkage with the
   vehicle tyre radius (0.330 m), not the 0.25 m soil cylinder.
   - The Polaris breakthrough stop therefore fires at 0.22 m of true cylinder sinkage (the HMMWV at 0.30 m). The
     slip ratios in the extra per-frame arrays are biased the same way.
   - This is the Gator precedent (about 0.21 m), so the Polaris-Gator comparison stays like for like. Say it in the
     caveats. For `polaris_w08` the threshold is the full 0.30 m.
3. **The flat-soil check does not cover all of PLAN 2.1.**
   - No module ran the braked hold on 10 and 15 deg tilted soil for the Polaris or the HMMWV. M2's tilt check handles
     the M113 and the Gator only.
   - The "settle, then 1.2 s at rest, creep reported" phase was not run on its own. Its numbers come from the anchor
     frame of the drives.
4. **Small wording and robustness points.**
   - `NEDM_VEHICLE=""` (set but empty) is accepted by the switch, though its note says "set at all". M3's job script
     refuses it anyway.
   - A missing or changed asset folder fails before the collector starts. That gives a traceback in the worker log
     but no `collection_failure.json`, contrary to NOTES_M1. The worker still retires after three failures.
   - The copy of the vehicle block in `collection_request.json` has no file hashes or build numbers: it is written
     before the vehicle is built. `outcome.json` has them all.
   - A killed run leaves a small folder of two links in `/tmp`.

## For the orchestrator to decide

1. **PLAN 2.1's turn rule cannot be met; I agree with the builder.**
   - The frozen follower steers towards a point 5 m ahead. On a 10 m radius, that point stays on the path only when
     the vehicle runs about 0.8 m inside it: sqrt((10 + 0.5)^2 - 5^2) = 9.23 m.
   - Measured RMS cross-track: HMMWV 0.77, Polaris arms 0.79-0.81, Gator 0.95 m.
   - Amend the rule (e.g. "within 0.1 m of the HMMWV"), or report the turn without gating.
2. **Judge the speed rule on the window mean, not frame by frame.**
   - Every vehicle sits 5-7 % under 2 m/s and swings about +-0.07 m/s. `polaris_pc` misses on one frame (1.796 m/s
     at 4.15 s).
   - That frame is in the first speed step, so the 5 s steps, instead of S3's 8 s, did not cause it.
   - R1's proposed amendment B2 makes this moot. Under it, the missing tilted-soil hold (finding 3) is the one
     exclusion test left open for the Polaris.
3. **The `polaris_w08` result is a standing-start push limit, not a crater stall.** Label it that way if the
   wheel-size arm differs in stage 1.
