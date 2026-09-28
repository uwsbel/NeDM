# VERIFY M3: independent check of staging, task files, launcher, smoke read-out, billing, status (2026-09-28 00:12-00:30 CDT)

Verdict: **pass, nothing blocking.** The files do what the PLAN (as hashed, e166e1fb, no amendments yet) asks. Nothing I
found would make the Polaris smoke run wrong, unsafe or unfair. There are 3 things to settle before launch and some
small points (below).

What I did not do: I submitted no job, ran no local soil, set NEDM_VEHICLE nowhere and edited no existing or builder file.
My only cluster writes were a scratch folder in the login node's /tmp, which I have deleted. My test scripts and outputs
are in `K4/verify_M3/`.

## Checks I ran myself
- **Hard rules.**
  - `git diff HEAD` is empty: no tracked file changed.
  - No file newer than 22:30 in any older artefact folder, or in configs/, scripts/ or src/ apart from the new ov_* files.
  - No M3 file sets NEDM_VEHICLE. They only refuse to run when it is set (launcher, job script, dispatcher).
  - G4 holds only `checks/` and `stage_dryrun/`.
- **Same code as the Gator study.**
  - The local `src/nedm` (74 files) and the 9 frozen collector files are byte-identical to G3/source.
  - The 7 frozen hashes in `ov_stage.sh` match `K3/e3/gator_collector_sha256.txt`.
  - The staged f104 arena assets are identical to G3's.
  - G3's `crm_main.json` is 90cd049e and its `source_manifest.json` is c9e4ff01.
- **Dry-run staging tree.** `sha256sum -c DISPATCHER.sha256` passes on the cluster (103 entries). The configs are
  read-only and there are 0 `__pycache__` folders.
- **Smoke task file** (`K4/tasks/smoke_v1.json`).
  - Rebuilding it gives a byte-identical file (sha 93bc69a5): 916 rows, unique ids, every row names its vehicle, no
    per-row config.
  - All 144 `gatorctl` rows equal the Gator pilot rows of `K3/e3/tasks/pilot_gator_soil.json` (same case, route and seed).
  - The B rows point at the stored Gator (192) and HMMWV (192) runs, with equal route content.
  - A new cluster path check found all 1,134 paths present.
  - An M113 superset (`--m113 quick --gator-half-step`) passes as a superset: +144 rows, timeout 4,000 s.
- **Collection file.**
  - Built from the smoke file with tiers 0-6: 16,007 rows, 8,739 set to run, 144 moved back to their original tiers.
  - Next version (`--run-tiers all --rerun-tiers`): 15,575 set to run, 6,836 switched on.
  - A duplicate appended evaluation id is refused.
- **Launcher dry runs on the login node** (9 cases).
  - smoke_v1: timeout 2,400 s, margin 1,000 s, budget 13,400 s. With M113 rows: 4,000 / 4,000.
  - Refused as intended: NEDM_VEHICLE set, even to an empty value; a real submit from the dry-run tree; an output folder
    outside G4; a lowered claim margin; a 1 h job with M113 rows.
  - Queue count: 4 of harry's tasks now (2 rg_*, 2 ov_m2_*). All 8 mi3501x slices are k007-005-v*, which matches the
    analyzer's MI350X host test.
- **Billing.**
  - Weights equal slurm.conf x10 for every partition.
  - Since 09-27 22:00: 0.50 billed (ov_ 0.04, rg_ 0.46).
  - Since 09-25: 113.79 (ag_ 113.29, which matches the recorded 113.3).
- **Analyzer, run the production way** (no test options). My fake "polaris" = HMMWV runs relabelled, with a vehicle
  block and zero belly contact. The Gator re-drive = the stored pilot runs.
  - Verdict BETTER. Failure 69.4 % against 93.8 % (-24.3 points). Routes where only the Gator failed / only the fake
    polaris failed: 36 / 1. p 2.76e-10. 95 % interval [-34.0, -15.3].
  - Sample B: straight 6 m/s fails 34.4 %; the H_full picks reach the goal 96.9 %.
  - These are the builder's numbers.
- **12 edge cases** (`verify_M3/edge_results.txt`), all as the PLAN says:

  | case | what I changed | verdict |
  |---|---|---|
  | 1 | 3 drives missing | INCOMPLETE |
  | 2 | 2 crashes (1.4 %) | NOT BETTER (crash gate) |
  | 3 | 8 launch failures | NOT BETTER (launch and validity gates) |
  | 4 | one run with the wrong vehicle name | NOT BETTER (vehicle-record gate) |
  | 5 | belly flag 10.4 % / 9.7 % | not better / better |
  | 6 | 8 / 7 Gator re-drive end states changed (94.7 % / 95.3 %) | BLOCKED / better |
  | 7 | fake polaris = the Gator with +0.08 m wheels (exactly 116/144 fail) | NOT BETTER (criterion 5 needs strictly below 80.6 %) |
  | 8 | one sample-B drive missing | "better on A; B not complete" |

- **Login node** (python 3.9.21, numpy 2.0.1): the analyzer and the status tool both run.

## Settle before launch (orchestrator)
1. **Plan-review amendments would change M3's files.** REVIEW_R1 proposes several changes, and the current files follow
   the PLAN without them. If any is adopted before 01:30, update the files and re-check them first (about 20-30 min):
   - S8: new tier order, 2 Polaris canary rows, B straight 6 m/s ahead of the sensitivity arms;
   - S2: at least 8 of the 9 stored Gator goals reproduced, plus Gator B control rows;
   - N2: Holm family fixed at 3;
   - B1: M113 in its own task file (quick look = tier 0 only). The launcher already sets the timeout per task file.
     The task builder has no "M113 only" mode.
2. **Freeze the decision rule.** `ov_smoke_analyze.py` is staged as a tool that can still be refreshed, not frozen.
   - Record its sha256 in LOG.md before the smoke starts. It is d7b9141e... now, and would change if amended. Or stage
     it frozen with `--add`.
   - The dry-run tree holds an older copy (66070608). On an empty run it prints BLOCKED where the current file prints
     INCOMPLETE. The real staging copies the current file.
3. **The M113 data folder gets frozen with the Polaris staging.** `assets/traverse/vehicles/ov_m113/` goes into
   DISPATCHER.sha256 at the first real staging, although `ov_m113.py` is held back. M2 is still changing things:
   `ov_m113.py` changed from c65ddf55 to 94177c4e since the dry run. Any later change to those three data files could
   only come in under new file names. Either confirm they are final, or accept that.

## Small points (not blocking)
- **Bit-identity rows that crash are left out.** If they crash or never finish, they drop out of the pooled agreement
  and do not hold the verdict at INCOMPLETE. Test: all 3 HMMWV rows crashed and the verdict was still BETTER, with
  147/147 agreeing. Read the per-row bit-identity lines of the text report by hand.
- **"B not complete" is ambiguous.** The same wording appears when B drives are finished but invalid.
- **The launcher accepts any `*crm_collect*` file under G4/source/scripts.** The plain `crm_collect.py` would fail
  loudly on `--vehicle`, not silently, but always pass `ov_crm_collect.py`.
- **`--run-tiers all` without `--rerun-tiers`** leaves the earlier rows' run flags unchanged. Check `run_rows` in the
  meta file.
- **M113 claim margin.** It equals the timeout (4,000 s). The worker checks the timeout only every 120 s, so a late
  claim can overrun the job's time limit by about 2 min. It is then re-taken 25 min later. Timeout + 300 s would be
  safer. This does not affect smoke_v1.
- **Collection file example.** Its docstring switches the sensitivity arms off. Do that only after they have been
  driven.
- **Two workers per GPU.** The PLAN 3.3 test of this has no tool (an overlap step, as in `ag_s2_soil_step.sh`).
- **Contracts with the other modules hold:**
  - M1 writes the vehicle-block name = arm and `vehicle_extra.npz:belly_clearance_min_m`, as the analyzer reads them.
  - M1's compute-node job 441578 already drove the Polaris through `crm_worker`, which covers the builder's point 4.
  - M4's dataset builder matches `^polaris__f104_v2_group_NNNN_(route|op)_NN$` at tier >= 0. It therefore skips B
    rows, sensitivity arms and `gatorctl`. With the smoke file it would silently skip the 144 sample-A rows, so it must
    be given the collection file (builder's point 6).
  - M4's `ov_eval_tasks --existing` can reuse the smoke's B drives if the arm is named `straight6_polaris`.
- **Scratch to delete.** `G4/stage_dryrun` can be deleted after the real staging.
