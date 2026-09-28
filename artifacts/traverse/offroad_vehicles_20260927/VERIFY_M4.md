# VERIFY M4: independent check of the planner training and evaluation chain (2026-09-28 00:30-00:50 CDT)

I am the verifier of module M4. I read all seven new files and the frozen scripts they reuse. I re-ran the main
checks myself, tried edge cases, and checked the hard rules and the contracts with M1 and M3.
- **Machine and cluster use:** one GPU process at a time, no soil (Chrono) run, no cluster job, only read-only ssh.
- **What I wrote:** this file and small check records in `verify_M4/`. I edited nothing.
- I did not write a NOTES_M4.md, because that file is the builder's; this file is my notes file (as for M3).
- Short names: K3 / G3 = the Gator study's local / cluster folders, K4 / G4 = this study's.

## Verdict
**No blocking problem.** The chain does what the builder says, and every headline number reproduces. The findings
below are minor. Four points need a decision by the orchestrator (last section).

## Checks I re-ran (records in `verify_M4/`)
1. **Dataset builder** (`ov_build_ds.py`, Gator runs tiers 0-6, from the builder's read-only copy in `/tmp/ov_m4/gator_runs`):
   - result: 8,399 episodes and 30,827 rows in 79 s;
   - file sha256 dfd1975c..., the same as K3's cluster record `e4/soil_s1/f104_gator/f104_gator_crm_record.json`;
   - the frozen `ag_e5a_ids.py` gives the validated-id list ed248f52, and `ag_subset.py` gives the G subset b3938949.
     Both equal K3's.
   - Limit: this input held only the 8,399 selected runs. K3's cluster build also skipped 11,182 runs with other ids
     and 98 runs outside tiers 0-6. That selection code is the frozen one and is unchanged.
2. **Edge cases of the builder** (12-run link folders; every case behaved as it should):
   - Gator runs renamed `polaris__` with `--vehicle polaris`: refused ("vehicle block 'gator'"). Adding
     `--require-vehicle-block no` does not help: still refused;
   - the same runs renamed to plain HMMWV ids with `--vehicle hmmwv`: refused;
   - a `polaris_pc` build over `polaris__` folders: selects nothing and stops ("no episode selected");
   - `gatorctl__` ids with `--vehicle gator`: builds (48 rows);
   - a run whose case names a suite pair (`f104_pair_group_0001`): refused ("suite ids");
   - a run whose case group differs from its id: refused.
   - The prefix tools also work with a non-Gator prefix: `ag_e5a_ids --prefix polaris__` and `ag_subset
     --ids-strip-prefix polaris__` both ran on a test set.
3. **Training job**:
   - the `ov_train_jobs.py` lines are byte-identical to K3's `soil_bf_G.tsv`;
   - a local quick run of `ov_train.sbatch` (both lanes on one GPU) exited 0, with round-trip difference 0.0;
   - the job refuses to start unless the trainer hash begins 7a4f2d67.
   - K3's 5-seed runs took about 20 min, so the default 1 h 30 limit is ample.
4. **Gradient picks** (`e6/picks/f104/G_full_grad`, `H_full_grad`):
   - I recomputed the route locks from the route files: bf10f41d and 95f2483c. The set lock is cf0396c4; manifest
     hashes 1e4738a8 and c0840f73;
   - model hashes equal K3's deploy SHA256SUMS;
   - ci_grad's CEM stage matched the recorded picks on 800/800 for both models, and its own checks are all true.
   - **Independent re-plan:** I re-planned the last 3 groups of the md5 order on their own. They are outside the
     builder's 20-group rerun. All 6 route files are byte-identical, so the result does not depend on group order.
   - Refinement changed / abstained: 734 / 66 (Gator), 643 / 157 (HMMWV).
   - Picks above 5 m/s: 120 and 73; top pick speed 5.75 m/s.
   - Settings are the ci_grad defaults: 17 starts, 60 steps, pessimistic keep, abstain 0.3. crm_improve used the same.
5. **Drive rows** (`e6/tasks/`):
   - `_noreuse`: 1,600 rows (800 Gator + 800 HMMWV);
   - `_reuse`: 1,377 rows.
   - Rows in both files:
     - ids and seeds are unique;
     - no id clash with the smoke file;
     - each row's extra is exactly `--vehicle <v>`;
     - 0 staging-hash mismatches and 0 route-content mismatches.
   - The `_reuse` mapping, checked against K3's stored index:
     - every one of the 66 / 157 abstained groups points to the stored CEM run;
     - all four stored arms point to K3's run ids.
   - Staging dry run (read-only): 2,166 files listed, none on G4 yet.
   - **Sample-B reuse:** straight route + H_full picks on the Polaris with `--existing smoke_v1.json`. It found 192
     arm-group pairs, i.e. 190 smoke rows, plus 1,393 new rows.
6. **Outcome index** (`ov_eval_index.py` on K3's two Bfull mappings + `K3/e6/runs_soil`):
   - 6,400 rows, 0 field differences from the stored index;
   - belly flags: G_full 12/800, straight 89/800.
7. **Analysis**:
   - the self-test passes, all 16 checks, including G_full 67.4 % on the 800 pairs;
   - the spec writer gives the PLAN 4.4 family (4 tests for one vehicle, 7 for two), takes the REVIEW_R1 options, and
     refuses to overwrite a spec.
   - **Mock run:** K3 rows relabelled as Polaris arms, with QA fields faked:
     - verdict "works", 95.6 / 95.5 %, clearly above;
     - with 8 % random belly flags: "not physically trustworthy", because the bar with belly-as-failure drops below 90.
8. **Hard rules:**
   - `git diff`: no tracked file changed;
   - no file in K3, generalist, crm_f104_v1 or crm_improve changed since 22:37;
   - the new files are only `ov_*` and M2's config;
   - NEDM_VEHICLE is only unset, popped or asserted unset;
   - G4 holds only M1/M2 checks and M3's rehearsal: M4 wrote nothing there;
   - the queue holds 2 jobs, both from the other session.

## Minor findings (none changes a result as the chain stands)
1. **Validity needs every drive present and its checks synced.**
   - A drive that fails twice counts as missing, and one missing drive makes the vehicle "not physically trustworthy".
   - Every drive also needs its QA files synced (`--list-files`).
   - K3's local runs have no QA files. So the Gator gradient arm's 66 reused old drives read "QA not checked", unless
     their QA files are synced from G3. That arm's validity is only reported.
2. **Reuse ignores soil config and dispatcher.** The `--existing` reuse compares group, vehicle, case and route, but
   not the per-row soil config or the dispatcher.
   - This is safe with today's files: the 0.5 ms rows sit only on sample-A training groups.
   - REVIEW_R1 N9 (same dispatcher hash) is not enforced by the tool.
3. **The spec's note text does not change with the options.** It always describes the PLAN defaults. With REVIEW_R1
   options set, only the `options` field is correct.
4. **The 704-pair set is not locked into the spec.** It reads `scratch/S3/sample_B.json`, and the spec does not hash
   that file.
5. **The ref-B check has no partial override.** If a new vehicle's CEM stage differs from its recorded picks on even
   one group, `ov_grad_picks.py` stops. That is a safe failure, but it could cost time.
6. **Two small tools are not built:**
   - a deploy-sync copy for K4/G4 (`ag_deploy_sync.sh` is hard-wired to K3/G3; about 10 lines);
   - the evaluation-file subset needed for the offline AUC.
7. **Wording and a docstring:**
   - the verdict "not physically trustworthy" also covers "bar met only if belly-flagged drives are ignored". The
     `c_validity` field shows which case it was;
   - the `ov_build_ds.py` docstring uses `$G4/soil/runs`, but M3 writes `G4/soil_v1/runs` (the chain in NOTES_M4 is
     right).

## For the orchestrator
1. **Freeze the spec (with sha256 in LOG) before any gradient-arm drive**, including the Gator and HMMWV rows that are
   ready now. So REVIEW_R1 S1/S3/S4/S6 must be decided before those rows are launched.
2. **Name the arms by stage** (REVIEW_R1 S5; PLAN is silent). The spec judges the arms named `polaris_grad`,
   `polaris_cem` and `straight6_polaris`. If stage-1 evaluation drives come first, give them other names (e.g.
   `polaris_s1_grad`) so that the frozen family is judged on stage 2.
3. **Choose the rows file:**
   - `_reuse` only if the Gator re-drive passes. REVIEW_R1 S2 notes that the "95 % same end state" rule is weak.
   - Otherwise use `_noreuse`, re-drive G_full CEM under a new arm name, and pass `--gator-cem`.
   - Re-tier the rows when appending: they sit at tier -1, like the smoke.
4. **Disk:** 98 % full. My temporary files are deleted. `/tmp/ov_m4/gator_runs` (1.9 GB) can go now.
