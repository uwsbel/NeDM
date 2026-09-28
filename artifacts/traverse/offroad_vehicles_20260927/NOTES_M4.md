# NOTES M4: planner training and evaluation chain for any vehicle (2026-09-28 00:35 CDT)

No cluster job submitted, nothing written on the cluster (read-only ssh: file copies to this machine, one staging dry
run). No existing file edited. No local soil run. NEDM_VEHICLE never set. At most 2 GPU processes at any time.
Short names: K3 / G3 = the Gator study (local / cluster), K4 / G4 = this study.

## What I built (all new files)
- `scripts/ov_build_ds.py`: the training-file builder for any vehicle. It imports the frozen `ag_build_ds.py` and
  reuses its checks. What changed:
  - it knows every vehicle name (hmmwv, gator, polaris and its 3 variants, m113, m113_g4), with the id prefix
    `<vehicle>__` ('' for the HMMWV), or any prefix via `--id-prefix` (e.g. `gatorctl__`);
  - the vehicle check is generic: every run must carry a vehicle record naming the vehicle. Old HMMWV runs may have no
    record;
  - it still refuses test-suite and evaluation groups, and any group outside the arena's training pool;
  - new `--compare-subset` option.
- Subsets and validated-id lists: no new copies needed. The frozen `ag_subset.py` (`--ids-strip-prefix <v>__`) and
  `ag_e5a_ids.py` (`--prefix <v>__`) already work for any vehicle (tested below).
- `scripts/ov_train.sbatch` + `scripts/ov_train_jobs.py`: training on one MI350X with the PLAN 1.7 recipe.
  - 2 lanes per GPU, as in the Gator study.
  - The G4 root; the job refuses to start unless the trainer's hash begins 7a4f2d67.
  - `ov_train_jobs.py` writes the two-lane job list (deploy with round-trip check, and holdout).
- `scripts/ov_grad_picks.py`: the default planner from a standing start. It runs `ci_grad.py` unchanged, without
  `--poses`: CEM 4 x 64, then gradient refinement with the defaults. Around that it:
  - checks the map;
  - checks the models (their record and hash);
  - requires that the CEM stage equals the recorded CEM picks (`--ref-b-picks`, `--require-ref-all`);
  - plans N groups a second time and requires identical files (`--rerun-check N`);
  - writes the `ag_picks.json` manifest the row builder reads, a lock file, and `OV_GRAD_LOCK.json`.
- `scripts/ov_eval_tasks.py`: soil drive rows for any vehicle.
  - Arms are given as `NAME=<pick dirs>@<vehicle>`; the vehicle is required.
  - Row ids are `<v>__<group>__<arm>`, and every row carries `--vehicle`.
  - M113 rows get the 0.5 ms config and a 4,000 s timeout.
  - Paths are relative to G4; files outside K4 go under `ext/`.
  - Arms of the same vehicle with identical routes are merged into one drive.
  - `--existing` reuses an identical earlier drive (same group, vehicle, case file and route content). This works for
    this study's task files and for the Gator study's (G3). `--avoid` blocks id and seed clashes.
  - `stage` copies the files and checks their hashes (dry run unless `--execute`).
- `scripts/ov_eval_index.py`: the outcome index. It imports the frozen `ag_eval_index.py` and adds:
  - a generic vehicle-record check (new-dispatcher drives must carry a record);
  - the soil QA flags, launch check and belly-in-soil flag of every drive;
  - `fail_belly` / `unsafe_belly` (belly-flagged drives count as failures);
  - `suite_part` (the 600 fresh pairs vs the 200 tuning pairs);
  - `--list-files`, the list of files to copy back from the cluster.
- `scripts/ov_analyze.py`: the declared analysis, a thin layer over the frozen `ag_analyze.py`. It adds:
  - the sets f104_600, f104_200 and f104_800_notB (the 704 pairs outside sample B);
  - the 90 % bar test;
  - the validity checks of PLAN 4.3(c);
  - the "planner works" verdict;
  - `--reuse INDEX=arms`;
  - `--write-spec`: the PLAN 4.4 family, which is 3 tests per vehicle (`<v>_grad` against straight6, against the
    Gator's gradient arm, and against `<v>_cem`) plus the Gator's gradient vs CEM;
  - `--selftest`.

  The options `--bar-label`, `--rule` and `--clearly-above-bound` implement REVIEW_R1 S4, S3 and S6, if they are
  adopted. Both verdicts are always computed.

## Checks and results
1. **Dataset builder, full Gator stage 1.** I copied the 8,399 stored Gator soil runs (tiers 0-6) from G3 and rebuilt
   the file with `ov_build_ds.py`.
   - Result: 8,399 episodes, 30,827 rows, every array equal to K3's file.
   - The file itself is byte-identical: sha256 dfd1975c, the same as K3's record.
   - Validated ids: sha ed248f52, equal to K3's.
   - `ag_subset.py` G subset: sha b3938949, equal to K3's; 28,057 / 22,284 training rows.
   - Records: `scratch/M4/build_gator_s1/`.
2. **Id and vehicle-record checks.**
   - Gator runs renamed `polaris__` are refused (vehicle record 'gator').
   - `polaris_pc__` ids are not picked up by a `polaris` build.
   - `gatorctl__` ids with `--vehicle gator` build.
   - An HMMWV build over Gator runs selects nothing.
3. **Training job script.**
   - The generated job lines are byte-identical to K3's `soil_bf_G.tsv`.
   - A local quick run of `ov_train.sbatch` (1 epoch, 2 lanes on one GPU): both exit 0, deploy round trip max diff 0.0
     (`scratch/M4/ov_train_local_quick.out`).
4. **Gradient picks, 20 Gator suite pairs.**
   - The CEM stage equals the recorded G_full picks on 20/20.
   - A re-plan of 5 pairs is identical.
   - 17 picks changed, 3 abstained.
5. **Full 800-pair picks, locked.** Both runs checked the CEM stage against the recorded picks on 800/800, and a
   re-plan of 20 pairs was identical. Set lock `e6/picks/LOCK_gradref_v1.sha256`, all = cf0396c4.

   | picks | refinement changed / abstained | predicted fail, CEM -> gradient | mean speed, CEM -> gradient | picks faster than 5 m/s | route lock |
   |---|---|---|---|---|---|
   | `e6/picks/f104/G_full_grad` | 734 / 66 | 0.181 -> 0.114 | 3.34 -> 4.10 m/s | 120 | bf10f41d |
   | `e6/picks/f104/H_full_grad` | 643 / 157 | 0.0045 -> 0.0001 | 3.23 -> 3.59 m/s | 73 | 95f2483c |

   Planning took 2.6 s per pair, 35 min per model.
6. **Drive rows, ready (tier -1, not staged).**
   - `e6/tasks/soil_eval_gradref_v1_noreuse.json`: 1,600 new drives (800 Gator + 800 HMMWV).
   - `e6/tasks/soil_eval_gradref_v1_reuse.json`: 1,377 new drives. It reuses the stored Gator-study drives where the
     route is identical (66 Gator and 157 HMMWV abstentions). Its mapping also carries the stored G_full CEM, H_full
     CEM and both straight-route arms (800 each, all reused).
   - The staging dry run lists 2,400 files; none is on G4 yet.
   - Reuse of the smoke's sample-B rows was tested: 96 pairs x 2 arms resolve to the 190 smoke rows.
7. **Index.** Rebuilt from the Gator study's Bfull mappings and synced runs, it reproduces K3's stored index: 6,400
   rows, 0 field differences (`scratch/M4/check_eval_index_k3_reproduction.json`). Its belly flags equal K3's extras
   (G_full 12/800, straight 89/800).
8. **Analysis self-test passed.**
   - Every K3 Bfull family test and contrast is reproduced (decision, difference, intervals, Holm).
   - G_full reaches the goal on 67.4 % of the 800 pairs (66.7 % on the 600 fresh pairs).
   - The declared spec was run end to end on relabelled stored rows (a mock, in `scratch/M4/mock_analysis`).

## Not done
- No Polaris or M113 dataset, model or picks: they need that vehicle's collection.
- No spec frozen.
- No rows staged or added to a task file.
- The optional wide-shape arm (A4b) was not built.

## For the orchestrator to decide
1. **Freeze the analysis spec before the first gradient-arm drive**, and record its sha256 in LOG.
   - As PLAN is written: `ov_analyze.py --write-spec e6/analysis/spec_ov_v1.json --vehicle polaris`.
   - With REVIEW_R1 S3/S4/S6/S1 adopted, add `--rule r1 --bar-label unsafe_belly --clearly-above-bound cluster95
     --extra-bar polaris_pc:polaris_grad_pc`.
   - PLAN 4.3(b) "95 % group-bootstrap lower bound" does not say two-sided or one-sided. I used the lower end of the
     two-sided interval; the cluster version is computed beside it.
2. **Which rows file.** PLAN 4.2 reuses the stored Gator CEM / straight drives only if the sample-A Gator re-drive
   passes. If it passes: `_reuse`. If not: `_noreuse`, plus re-drives of G_full CEM under a new arm name (then
   `--gator-cem <name>` in the spec).
3. **Tier and staging.**
   - Re-tier when appending: M3's `ov_soil_tasks.py --append ... --append-tier`. The rows are at tier -1, which is
     below collection tier 0 but level with the smoke's tier -1.
   - Stage the files with `ov_eval_tasks.py stage --tasks <file> --execute` (writes to G4).
4. **Caveat.** The gradient makes the Gator's picks faster (120/800 above 5 m/s). On the Gator, 6 m/s fails often
   (S4 risk 1). This is a known risk, not a reason to change the declared planner.

## Chain for a passing vehicle V (after its collection)
```
ov_build_ds.py --world crm --vehicle V --arena f104 --crm-runs $G4/soil_v1/runs --tasks-crm <collection file> --tiers 0-12 \
    --map-root $G3/e4/map_roots/f104 --source-root $G4/source --out $G4/e4/soil_bf/f104_V --workers 16   (login node)
ag_e5a_ids.py --world crm --prefix V__ --ci .../ci_f104_V_crm.npz --record .../f104_V_crm_record.json --tasks <file> --runs $G4/soil_v1/runs --out-ids <ids>
ag_subset.py --world crm --tiers 0-12 --eval-rows filtered --groups f104=all --ds <ci> --ids-file <ids> --ids-strip-prefix V__ --out $G4/e4/subsets/V_full_f104_V_soil.npz
ov_train_jobs.py --tag V_full_soil --ds <subset> --out <tsv>
env -u NEDM_VEHICLE sbatch -p mi3501x -t 01:30:00 --export=ALL,OV_JOBS=<tsv> $G4/source/scripts/ov_train.sbatch
ag_picks.py --arena f104 --world crm --mode free --cases $SUITE --groups all --models "<deploy>/V_full_soil_deploy_s*.pt" --model-tag V_full --out $K4/e6/picks/f104/V_full_free --rerun-check
ov_grad_picks.py --arena f104 --world crm --cases $SUITE --groups all --models "<same>" --model-tag V_full --ref-b-picks $K4/e6/picks/f104/V_full_free --require-ref-all --rerun-check 20 --out $K4/e6/picks/f104/V_full_grad
ov_eval_tasks.py build --tier <t> --arm V_grad=$K4/e6/picks/f104/V_full_grad@V --arm V_cem=$K4/e6/picks/f104/V_full_free@V \
    --arm straight6_V=$K3/e6/picks/crm/f104/straight6@V --existing $K4/tasks/<smoke/collection file> --out ...
ov_eval_index.py --mapping <maps> --runs-g4 <sync> --runs-g3 $K3/e6/runs_soil --out <index>
ov_analyze.py --index <index> --spec <frozen spec> --out <results>
```
The arm names must match the spec: `V_grad` / `V_cem` stand for `<v>_grad` / `<v>_cem`, e.g. `polaris_grad`.
