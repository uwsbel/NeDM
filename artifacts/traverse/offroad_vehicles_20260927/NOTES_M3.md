# NOTES M3: staging, task files, launch, smoke read-out, billing, status (2026-09-28 00:12 CDT)

No cluster job submitted. Nothing existing edited. No local soil run. NEDM_VEHICLE never set.
Only cluster writes: the throw-away rehearsal tree `G4/stage_dryrun/` (G4 = /work1/dannegrut/harry/experiments/offroad_vehicles_20260927).

## What I built (all new files)
- `scripts/ov_stage.sh` (run locally). `--dry-run | --real` stages G4:
  - `source/` = src/ + scripts/ (no `__pycache__`, no `ov_*` drafts) + the dispatcher set by name + 2 login-node tools + `assets/traverse/{arena_f104_50h_v1,vehicles}` + a byte copy of G3's `source_manifest.json`;
  - `configs/crm_main.json` = byte copy of G3's (sha 90cd049e);
  - `tasks/ soil_v1/ runtime/ launch/submissions.tsv checks/`.
  - It checks the 7 unchanged collector files against G3's frozen hashes, then writes `source/DISPATCHER.sha256` (dispatcher, both configs, src/nedm, arena and vehicle assets) and checks it on the cluster. It then freezes every listed file (chmod a-w) and records to `runtime/stage_record.txt` + `K4/stage_records/`.
  - Other modes: `--tasks F...` (read-only task files, never replaced); `--add F...` (new files only; appends to DISPATCHER.sha256); `--tools F...` (refresh non-frozen `ov_*` tools; refuses frozen files); `--with-m113`.
  - The M113 module and `crm_m113.json` are left out unless asked. A frozen draft could not be replaced, and the dispatcher imports `ov_m113` by name.
- `scripts/ov_soil.sbatch`: copy of `ag_soil.sbatch` with CRM_ROOT = G4. The job stops if any of these fail:
  - NEDM_VEHICLE must not be set;
  - DISPATCHER.sha256 must still match;
  - CRM_EPISODE_TIMEOUT_S must be passed in.
- `scripts/ov_soil_launch.sh`: copy of `ag_soil_launch.sh` pointed at G4.
  - Refuses a launch if the task file or collector is still writable, if the collector is outside G4/source, if NEDM_VEHICLE is set, or for mi3501x > 4 h.
  - The 50 - 5 queue guard counts ALL of harry's tasks and prints them by name.
  - `OV_DRY_RUN=1` prints the commands and submits nothing. Records go to `G4/launch/submissions.tsv`.
- **No `ov_crm_worker.py`: it is not needed.**
  - Rows carry `timeout_s`. The launcher passes the largest value among run:true rows as the worker's own `CRM_EPISODE_TIMEOUT_S`: 2,400 s without M113 rows, 4,000 s with them.
  - The claim margin is set the same way: 1,000 s, or 4,000 s when M113 rows are in the file.
  - Wheeled rows take about 230-700 s, so a job-wide timeout never changes their outcome.
  - The frozen `crm_worker.py` (f856b998) stays as used in K3.
- `scripts/ov_smoke_tasks.py` builds the PLAN 2.2 file. Built: `K4/tasks/smoke_v1.json` (+ .meta.json), sha 93bc69a5..., 916 rows:
  - tier -4: 6 bit-identity rows (3 ids outside sample A as `bitid_hmmwv__` and `bitid_gator__`);
  - tier -3: 48 `polaris__` + 48 `gatorctl__`;
  - tier -2: 96 + 96;
  - tier -1: 3 x 144 sensitivity rows, plus 190 sample-B rows `polaris__<group>__straight6_polaris` / `__Hfull_free_polaris`. In 2 groups both routes are identical, so one row carries both arms.
  - `--m113 quick|all` (+ optional `--gator-half-step` Gator-at-0.5 ms control) builds a checked superset. M113 rows carry `config: configs/crm_m113.json` and `timeout_s: 4000`.
  - 1,134 cluster paths checked: all present.
- `scripts/ov_soil_tasks.py`: the 15,235-row collection file per vehicle (Gator rows, prefix and flag changed).
  - Supersets with `--previous`. The smoke's 144 `polaris__` rows are moved to their collect_v1 tiers ('retiered_from' kept), so they are reused, not re-driven: same id, same output folder `G4/soil_v1`.
  - Other options: `--run-tiers 0-6|all|none`, `--disable/--enable-prefix`, `--append eval rows --append-tier -5`.
- `scripts/ov_smoke_analyze.py`: the frozen PLAN 2.3/2.4 rule (numpy only; run it with `python3 -B`). Test options exist but are refused without `--test-label`.
- `scripts/ov_bf_billed.py`: weights for every partition, checked against slurm.conf (`--check-weights`); splits the total by job-name prefix.
- `scripts/ov_collect_status.py`: counts per arm and tier, failed ids, launch failures, stalled claims, retired workers.

## Checks run
- **Staging rehearsal** into `G4/stage_dryrun`: 101 files hashed (103 after `--add` of ov_m113.py + crm_m113.json). The cluster copy matches the local hashes. After freezing, 0 of those files are writable and there are 0 `__pycache__` folders. `--add` refuses an existing file; `--tools` refuses a frozen file.
- **Launcher dry runs** on the cluster:
  - smoke_v1 gives timeout 2,400 s and margin 1,000 s; the M113 superset gives 4,000 / 4,000.
  - Refusals work: NEDM_VEHICLE set, 2 + 44 tasks over the cap, mi3501x 6 h, a real submit from the rehearsal tree.
- **Collection builder**: from smoke_v1, 16,007 rows, 144 moved to their collect_v1 tiers, 8,739 rows to run for tiers 0-6 (9,885 paths all present). The append, all-tier, and M113 queue-behind versions pass. A changed seed is rejected.
- **Billing**: since 09-27 22:00, 0.33 billed (the other session's rg_* jobs). Since 09-25, 113.62, which matches the recorded 113.3 plus rg.
- **Analyzer tests on stored runs** (read-only rsync pull to /tmp; outputs in `K4/checks_M3/`):
  - T1: HMMWV runs fed in as a fake "polaris" fail 69.4 % against the Gator's 93.8 %, i.e. -24.3 points. Discordant routes 36/1, one-sided p 2.8e-10, group interval [-34.0, -15.3]. Verdict NOT BETTER, as it should be: HMMWV runs have no belly or vehicle record.
  - T2: the same with those two gates relaxed gives BETTER. Sample B straight 6 m/s fails 34.4 %; the H_full picks reach the goal 96.9 %.
  - T3: with the Gator +0.08 m wheel runs posing as the stored runs, only 122/144 end states agree, and the verdict is BLOCKED.
  - T4 (M113 path): Holm over 3 arms works, and B = 83.3 % gives MIXED.
  - T5: sample B not driven gives "better on A; B not complete".
  - T0: nothing driven gives INCOMPLETE.
  - The stored pilot numbers come back exactly: 93.8 %, belly flag 4.2 %, +0.08 m wheels 20/1 discordant.

## Not done
- `ov_driveability.py`: not built (LOG: stage 0 was folded into the M1/M2 check jobs).
- No real staging and no real task-file staging (not asked).
- The analyzer has not seen a real new-dispatcher run yet.

## For the orchestrator to decide or know
1. **Order of steps:**
   - `bash scripts/ov_stage.sh --real` (after the M1 verifier), then `--real --tasks $K4/tasks/smoke_v1.json`;
   - then on the login node: `env -u NEDM_VEHICLE bash $G4/source/scripts/ov_soil_launch.sh $G4/tasks/smoke_v1.json $G4/soil_v1 $G4/source/scripts/ov_crm_collect.py mi3501x:24:4:6`;
   - M113 later: `--real --add scripts/ov_m113.py configs/crm_m113.json`, then smoke_v2 = `ov_smoke_tasks.py --m113 quick --check-superset smoke_v1.json`.
2. **Reference check is pooled.** PLAN 2.3's ">= 95 % same end state" is applied over gatorctl + bit-identity rows together (150 rows). The 6 bit-identity rows alone could not pass on noise.
3. **Gates use sample A only.** Sample B's validity is reported, not gated.
4. **Holm family** = the primary arms actually driven. The p value is also checked at alpha/3.
5. **M113 with only the quick look** is judged on 48 routes.
6. **No early Polaris build proof.** PLAN tier -4 has no Polaris rows (CRITIC 3a.4). A Polaris build that fails on the cluster would retire workers after 3 failures in a row, so M1's compute-node check should come before the smoke.
7. **Misleading config name (M2's file).** `configs/crm_m113.json` has step 0.5 ms but its "name" still says `crm_main_step1ms_spacing008`. The name is recorded in every M113 run's request file.
8. **Dataset input.** M4's `ov_build_ds.py` must read the collection file (tiers >= 0), not the smoke file (negative tiers).
9. **Cleanup.** `G4/stage_dryrun` can be deleted.
