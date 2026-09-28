# NOTES U1: Polaris on the 8 unseen arenas - everything up to the full drive launch (2026-09-28, 13:55-15:15 CDT)

PLAN amendment 9.2, milestone "the planner generalises to arenas it never trained on". Short names: K3 / G3 = the Gator
study (local / cluster, read-only), K4 / G4 = this study.

What was not done or touched:
- The full drives were NOT launched.
- No existing file was edited; every file is new.
- Nothing was written into G3 or any older root. The only G3 access was a read-only copy of 8 stored runs.
- NEDM_VEHICLE was never set; every launch went through `env -u NEDM_VEHICLE`.
- No local soil run.
- At most one extra GPU process from me at any time: the planning ran one arena and one stage after another. Every
  planning call ran under `timeout`; none hung.
- One cluster job: the pilot, 442088. It finished in 2 min 11 s and cost 0.03 billed node-hours.

## 1. Read-out spec, frozen before any drive

- **`e6/analysis/spec_unseen_v1.json`**
  - sha256 `bdc2a7385b5dc7fa403aa0e50af4a4eb80f52841252aa85617c3ca41627bdcb4`.
  - Written 14:03:04, when G4 held no drive of any of the three arms (checked: 0 run folders).
  - Frozen: read-only, with `e6/analysis/spec_unseen_v1.sha256` beside it.
  - It records PLAN.md's sha256 at that time (dc278146...).
- **Tool `scripts/ov_unseen_analyze.py`**, sha256 `ce15dc28...`, read-only.
  - It is a thin layer over `ov_analyze.py` / `ag_analyze.py`. Every paired statistic is `ag_analyze.contrast`,
    unchanged.
- **What the read-out computes:**
  - **Label.** "Goal reached safely" = not `unsafe_belly`: the goal is reached, there is no roll-back, and there is no
    belly-in-soil flag.
  - **Pooled rate** over the 1,000 declared pairs, with four 95 % intervals:
    - Wilson;
    - pair bootstrap;
    - cluster bootstrap over (arena, nearest feature);
    - bootstrap over the 8 arenas.

    A t interval over the 8 arena rates is shown beside them.
  - **Verdict:**
    - "meets the bar" if the pooled rate is at least 90 %;
    - "clearly above the bar" if, in addition, the lower end of the two-sided 95 % interval over arenas is at least
      90 %;
    - otherwise "below the bar";
    - "not physically trustworthy" if the primary arm fails validity: any crash or collector failure, 5 % or more
      launch-check failures, or belly flags above 10 %;
    - "incomplete" while any declared pair has no drive of the primary arm.
  - **Failed drives.** A pair whose drive failed twice (`--failed-ids`) counts as not reached safely, and as a crash.
  - **Per arena:** the rate, its Wilson and pair-bootstrap intervals, and above / below / not distinguishable from 90 %.
  - **Declared family** (Holm at 0.05 over the one-sided cluster p-values), both on the declared label:
    - `polaris_u_grad` vs `straight6_polaris_u`;
    - `polaris_u_grad` vs `polaris_u_cem`.

    Each comes with exact McNemar (two- and one-sided), the pair bootstrap and the cluster bootstrap.
  - **Secondary (unadjusted):**
    - the same two comparisons on goal only;
    - CEM vs straight.
  - **Context, not tests:** K3's HMMWV planners on the same 1,000 pairs (M1a, M1b, M3a, A3, the composites M1 and M3,
    and the HMMWV straight route), plus Polaris-minus-HMMWV paired differences.
- **Self-test** (`e6/unseen/selftest/`): PASSED.
  - Wilson matches known values.
  - A mock that relabels K3's M3a / A3 / straight6 drives as the three Polaris arms returns exactly 90.3 % and the
    same paired numbers as `ag_analyze`.
  - The context rows reproduce K3's stored goal rates: M1 87.95 %, M3 90.3 %, straight6 60.0 %.
  - Planted cases give the expected verdicts: clearly above / below / meets.
  - A missing drive gives "incomplete".
  - A drive that failed twice counts as a failure and as a crash.
- **One change after the spec was frozen** (a reporting label only; no declared number changed). The "interim" label on
  a partial run used to call a partial run untrustworthy just because drives were missing. It now leaves out the
  "all driven" check, since "incomplete" already says that. Before the change the tool's sha256 was `c4345148...`. The
  self-test passed again after the change.

## 2. Picks: standing start, each arena's own map, 125 declared pairs per arena, locked

- **Ensemble:** `e5/deploy/polaris_full_soil/polaris_full_soil_deploy_s0-4.pt`. `SHA256SUMS` verified: 5/5 OK.
- **Driver script:** `e6/unseen/run_picks_u1.sh`. Logs are in `e6/picks/logs/u1_*`.
- **CEM step:** `ag_picks.py --mode free --model-tag polaris_u --rerun-check`, map root `K3/map_roots/<arena>`.
  - The map check passed on all 8 arenas.
  - The re-plan was identical on 8/8.
  - About 105 s per arena (0.4 s per pair).
- **Gradient step:** `ov_grad_picks.py --ref-b-picks <CEM> --require-ref-all --rerun-check 10`.
  - The CEM stage equals the recorded CEM picks on 125/125 in every arena.
  - The re-plan of 10 groups per arena was identical.
  - 2.65 s per pair.
- Across the 1,000 pairs, gradient refinement changed 620 picks and left 380 unchanged (abstained).
- Mean predicted failure is about 1.3e-5 before and after refinement. The f104-trained model is extremely confident on
  arenas it never saw; it did not help separate routes on f104 either (CEM 100 % vs straight 99.5 %).

| arena | CEM lock | gradient lock | changed / abstained | mean speed CEM -> gradient (m/s) |
|---|---|---|---|---|
| g260 | f29255281201b08b | 615c62fe2170728e | 65 / 60 | 3.07 -> 3.08 |
| g271 | 54246e5a6046eda0 | a3cd5db444e05e63 | 79 / 46 | 3.08 -> 3.07 |
| g251 | d65070cd26fc581a | 22bd5e0cae4cab79 | 69 / 56 | 3.25 -> 3.31 |
| g247 | d1280032c23e9ee5 | 909f0e31a7ebc003 | 80 / 45 | 3.27 -> 3.37 |
| g258 | 8fbe0a496a78ebab | 23734e6b3b828718 | 70 / 55 | 3.11 -> 3.13 |
| g268 | 2a24de5403b2b09e | 0df78b4f737f6479 | 91 / 34 | 3.20 -> 3.11 |
| g263 | d9e383cf9b7bb773 | 8d4cda9d79a09f9c | 75 / 50 | 3.25 -> 3.53 |
| g241 | 8d5505ba33f86f94 | 847fe1bf4a9505c7 | 91 / 34 | 3.28 -> 3.48 |

- **Set lock over all 16 folders:** `e6/picks/LOCK_unseen_v1.sha256` / `.json`, ALL = `255958f013a3c017...`.
  - Folders: `e6/picks/<arena>/polaris_u_free` (CEM) and `e6/picks/<arena>/polaris_u_grad`, each with its own
    `PICKS_LOCKED.sha256`, `ag_picks.json` and, for the gradient folders, `OV_GRAD_LOCK.json`.

## 3. Drive rows

- **`e6/tasks/soil_eval_polaris_unseen_v1.json`** (sha256 `44edd114...`, plus `.mapping.json` `e1384589...`,
  `.staging.tsv`, `.meta.json`).
  - Built with `ov_eval_tasks.py build --tier -10 --existing tasks/soil_v6_polaris.json`, vehicle polaris.
  - It works for any arena as it is: the case and arena come from each pick folder's manifest, so no wrapper was needed.
- **Arms**, each covering 1,000/1,000 pairs:
  - `polaris_u_grad` = `e6/picks/<a>/polaris_u_grad`;
  - `polaris_u_cem` = `e6/picks/<a>/polaris_u_free`;
  - `straight6_polaris_u` = `K3/e6/picks/crm/<a>/straight6`.
- **2,618 new drives** from 3,000 arm-pair entries.
  - 380 abstentions share the CEM drive.
  - 2 straight routes are identical to a planner route.
  - Per arena: 315-341 rows.
- **Paths.**
  - Cases are written as `ext/artifacts/traverse/arena_gator_20260925/cases/test_<a>/cases/<g>.json`.
  - K3 straight routes go under `ext/`; K4 routes keep their K4-relative path.
- **Staged:** `ov_eval_tasks.py stage --execute` copied the files. The cluster check found 3,618/3,618 files present
  with equal sha256.

## 4. Arena staging on G4

- **What the soil collector reads for an arena.** `crm_collect.py` sets `arena = <--source-root>/<case["arena"]>`,
  i.e. `G4/source/assets/traverse/arena_<a>`.
  - `TerrainMap.from_dir` and `scene.build_config` read `arena_meta.json` and the BMP it names (`arena_000.bmp`).
  - `build_crm` builds the soil from that BMP.
  - Nothing else is arena-specific: no mesh, and no soil or map file. The overhead map roots are used only by the
    planner, locally. The Polaris / Gator vehicle code has no hard-coded arena.
- **How K3 did it.** K3's G3 drives found the same two files in `G3/source/assets/traverse/arena_<a>`. The local copies
  are byte-identical to G3's (16/16 sha256 equal).
- **What was missing.** G4/source had only `arena_f104_50h_v1`.
- **Added:** `bash scripts/ov_stage.sh --real --add` with the 16 files (the 8 arenas x 2), new files only.
  - They are frozen, and appended to DISPATCHER.sha256 (now `6b34007b...`). The cluster check (`sha256sum -c`)
    passes.
  - Record: `stage_records/added_files.tsv`.
  - The orchestrator's job 442067 started after this, and its check passed.
- **Case files** needed no rewrite. Their `"arena": "assets/traverse/arena_<a>"` resolves against G4/source, as the
  pilot proved.

## 5. Pilot (job 442088, mi2104x x 2 tasks, `ov_soil_launch.sh`, output G4/soil_v1)

- **Task file.** `tasks/pilot_unseen_v1.json` (sha256 `e22dd85d...`), built by the new `scripts/ov_unseen_pilot.py`
  and staged read-only to G4/tasks.
- **Groups.** Per arena, the lowest-md5 group of the declared subset: g260_0171, g271_0125, g251_0000, g247_0191,
  g258_0068, g268_0098, g263_0070, g241_0179.
- **Rows** (16, all complete, 0 failed):
  - 8 HMMWV reference rows `<g>__u1ref_M1a_free`: the same case and the same K3 M1a_free route file as the stored G3
    drive `G3/soil_v1/runs/<g>__M1a_free`;
  - 8 Polaris `polaris_u_grad` rows, copied unchanged from the full row file.
- **Result:** `e6/unseen/pilot_compare_v1.{json,txt}`.
  - **HMMWV reference: the end state matches the stored run on 8/8** (rule: at least 7). Exact arrays are identical on
    7/8.
  - g251_0000 differs from 12.8 s of its 16.2 s drive onward. The end pose is within 1.7 cm and the end state is the
    same (goal reached at the same time). It re-ran on node k003-004, while the stored run was on k004-002. This is
    the known small node-to-node non-determinism, not a staging problem.
  - Caveat: all 8 stored reference drives were goal-reached drives, so the end-state test alone is weak; the identical
    arrays are the stronger evidence.
  - **Polaris: 8/8 reached the goal safely.** No belly flag, launch check passed, drives 9-27 s simulated. The goal
    only rate is also 8/8.
- **Full chain run on the pilot drives:**
  - `ov_eval_index.py` read the new arenas' cases and checked the vehicle block;
  - `ov_unseen_analyze.py` then ran (`e6/unseen/pilot_index_v1.json`, `pilot_results_v1.*`);
  - verdict "incomplete" (8/1000), as it should be.
- **Speed.** About 3.4 wall seconds per simulated second, about 55-65 s per drive on an MI210 GPU.

## 6. Superset task file (staged, not launched)

- **`tasks/soil_v7_polaris.json`** (sha256 `3330fd3e3337b54a092b251793eca9aa71b2cf6776d5cae5c16b53a4c820d13a`).
  - `--previous tasks/soil_v6b_polaris.json`: v6b is the newest version. The orchestrator wrote it at 14:20 with the
    1,324 f104 gradient rows at tier -11.
  - `--append soil_eval_polaris_unseen_v1.json pilot_unseen_v1.json.ref_rows.json --append-tier -10 --check-cluster`.
- **Contents:** 25,512 rows, all run: true. Tier -10 = 2,626 rows (2,618 Polaris + 8 HMMWV reference); 16 of them are
  already complete from the pilot.
- **Checks:** all 27,371 case and route paths exist on G4; no seed clash among the new rows.
- **Staged:** `ov_stage.sh --real --tasks` put it at `G4/tasks/soil_v7_polaris.json`, read-only.
- **Launcher dry run** passes (queue 8 + 12 = 20).
- **Run order.** Tier -11 comes first (54 of v6b's 1,324 rows were still open at 15:08), then tier -10, then rows that
  are already complete.

**Launch command for the full drives (login node):**
```
G4=/work1/dannegrut/harry/experiments/offroad_vehicles_20260927
env -u NEDM_VEHICLE bash $G4/source/scripts/ov_soil_launch.sh $G4/tasks/soil_v7_polaris.json $G4/soil_v1 \
    $G4/source/scripts/ov_crm_collect.py mi2104x:128:3:8 mi3501x:24:3:4
```
- **Size and cost.** About 2,610 drives x about 70 s is about 50 GPU-hours, i.e. about 1.5 h on 36 GPUs.
  - About 5-6 billed node-hours (mi2104x 0.4 per node-hour).
  - Billed since 09-27 22:00: 53.6.
  - Idle jobs end by themselves when the rows run out.

**After the drives (local):**
```
export PYTHONPATH=src:scripts; PY=/home/harry/miniconda3/envs/nedm/bin/python; K4=artifacts/traverse/offroad_vehicles_20260927
$PY scripts/ov_eval_index.py --mapping $K4/e6/tasks/soil_eval_polaris_unseen_v1.json.mapping.json --list-files /tmp/u1_files.txt
cut -f2 /tmp/u1_files.txt > /tmp/u1_rel.txt
rsync -a --files-from=/tmp/u1_rel.txt amd:$G4/soil_v1/runs/ /tmp/u1_runs/      # exit 23 for absent optional files is fine
# drives that failed twice without a completed run -> one id per line in /tmp/u1_failed.txt (G4/soil_v1/failed/<id>.json, attempts >= 2)
$PY scripts/ov_eval_index.py --mapping $K4/e6/tasks/soil_eval_polaris_unseen_v1.json.mapping.json --runs-g4 /tmp/u1_runs \
    --out $K4/e6/index/unseen_polaris_v1.json
$PY scripts/ov_unseen_analyze.py --index $K4/e6/index/unseen_polaris_v1.json --spec $K4/e6/analysis/spec_unseen_v1.json \
    [--failed-ids /tmp/u1_failed.txt] --out $K4/e6/analysis/results_unseen_v1.json
```

## 7. For the orchestrator

1. **Launch** with the command above once the verifier agrees.
2. **v7 is built on v6b**, not v6, because v6b is the newest file on G4. Any later version must use `--previous
   soil_v7_polaris.json`.
3. **The reference check passed** (8/8 end states, 7/8 identical arrays). The stored references were all goal-reached
   drives.
4. **The spec's context numbers** are K3's goal-only rates (M1 87.95 %, M3 90.3 %). Safe rates for the HMMWV are 87.9 /
   90.3 %.
5. **The model's predicted risk is about 1e-5 everywhere**, so "planner vs straight route" may again be decided mostly
   by the straight route's own failures. On K3's HMMWV, the straight 6 m/s route reached the goal on 60.0 % of these
   pairs.
