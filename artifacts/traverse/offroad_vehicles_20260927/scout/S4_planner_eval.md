# Scout S4: how to train and evaluate a planner for a new vehicle (Polaris, M113) on f104 soil

Written 2026-09-27 by scout S4. Read-only on every existing file. The only things written: this report and one local
smoke run under `artifacts/traverse/offroad_vehicles_20260927/scratch/S4/` (gradient refinement from a standing start
on 3 pairs; no Chrono, no cluster job).

Short names used below:
- **K3** = `artifacts/traverse/arena_gator_20260925` (the Gator study), **G3** = its cluster root
  `/work1/dannegrut/harry/experiments/arena_gator_20260925`.
- **K4 / G4** = this study's local root `artifacts/traverse/offroad_vehicles_20260927` and a proposed cluster root
  `/work1/dannegrut/harry/experiments/offroad_vehicles_20260927`.
- **V** = the new vehicle's tag (`polaris` or `m113`).
- **Risk model** = the CNN-GRU network that reads the terrain along a candidate route (a corridor cut from the overhead
  depth map) and predicts the chance that the drive fails. "`--cond none --ctx geom`" means it sees only the route's
  terrain, the commanded speeds and five geometry numbers (distance and direction to the goal, remaining length). It
  sees no vehicle state and no history.
- **CEM 4 x 64** = the planner's iterated sampling: 4 rounds of 64 candidate routes, each round re-centred on the best
  15 %. **Standing start** = the planner decides once, with the vehicle at rest on the start pad.
- **Gradient refinement** (`scripts/ci_grad.py`) = take the CEM pick and 16 other good candidates and push their shape
  and speed downhill in predicted risk for 60 steps through the network; keep the result only if it beats the CEM pick
  by a margin.
- **The 800 suite** = the 800 f104 soil start/goal pairs (600 `f104_pair_group_*` + 200 `f104_crm_eval_group_*`),
  cases in `artifacts/traverse/generalist_20260921/A_adapt/suite/cases`.

---

## 0. Summary

1. **The Gator chain is fully reproducible for a new vehicle, but not by re-running the scripts as they are.** About
   15 scripts hard-code the vehicle name (`hmmwv` / `gator`), the `gator__` id prefix or the Gator study's roots
   (section 1.12). None needs a deep change; each needs a new generalised copy (existing files are frozen).
2. **Best planner formulation.** The 97.5 % soil result of `crm_improve_20260922` decomposes into two parts:
   - deciding after a 0.5 s approach with a history model: **about 0 points** when the world is known. The soil
     specialist from a standing start (95.8 %) equals the history model after 0.5 s (95.8 %), and the oracle-tag
     model reaches 96.1 % under both protocols (crm_improve REPORT table);
   - **gradient refinement: +1.7 points** (97.5 vs 95.8 %, p 0.016, 800 pairs). Night 2 saw +1.0 from a standing start
     (99.5 vs 98.5 %, 200 pairs, not significant).
   - For one vehicle on one known ground type, the moving-start machinery buys nothing and costs new tooling, a
     tracked-vehicle substitute for the wheel-speed history columns, and an approach drive per pair.
   - **Recommendation: the default "best planner" = the vehicle's own soil model + CEM 4 x 64 + gradient refinement
     (ci_grad defaults), from a standing start.** I checked locally that the ci_grad command line does exactly this
     with the Gator's models (section 2.4): its CEM pick equals the recorded pick on 3 of 3 pairs, at 2.6-3.9 s per
     pair.
3. **Wider route shapes (search probe arm A4)** are the most promising but least certain upgrade. They were never
   driven, and 47 of their 71 "rated safe" routes swing more than 10 m sideways, which no training route does.
   - Cheap fix: add 2 wide-shape routes per group (2,400 soil drives per vehicle, about 6-8 billed node-hours) to the
     collection.
   - Run them as a **secondary arm**, not the default. If the wide rows are not collected, use the probe's A4b variant
     (5 bend terms, 10 m limit), which stays inside the training range.
4. **Declared evaluation (section 3).** The 800 suite from a standing start, per vehicle:
   - the same planner as the Gator (for comparability), the best planner, and the straight route;
   - the Gator re-driven with the best planner (fair comparison);
   - a cheap HMMWV "bridge" arm that confirms the standing-start gradient default reproduces about 97 % on the HMMWV.
   - The 90 % bar applies to the best-planner arm, on "reaches the goal safely".
   - One Holm family of 7 paired tests.
   - About 14 billed node-hours for evaluation drives without the wide arm, about 41 with it. The collections come on
     top.

---

## 1. How the Gator planners were trained and evaluated (arena_gator_20260925)

### 1.1 Data

- **"Same amount of data" = the same task ids** (PLAN.md:46-48): the 15,235 HMMWV `collect_v1` soil ids (1,200
  training-pool groups of `f104_v2`, tiers 0-12; 9,168 designed + 6,067 on-policy routes).
  - Gator rows are `gator__<collect_v1 id>`, with `extra ['--vehicle', 'gator']`, in the soil task files
    `G3/tasks/soil_v2.json` / `soil_v3.json`, interleaved by tier (LOG.md:57).
  - They are driven by the frozen dispatcher `ag_crm_collect.py` (sha `b52e1fa6`), launched with `scripts/ag_soil_launch.sh`
    (`ag_soil.sbatch`, output `G3/soil_v1/runs`).
- **Result:** 15,235 / 15,235 validated, 141.1 simulated hours against the HMMWV's 91.5 (REPORT.md:264-270).
- **Collection read-out:** `scripts/ag_s1_gator_soil_qa.py` compares each Gator drive with its HMMWV twin: validated
  share, launch failures, non-finite states, belly flag, fail rate (LOG.md:112).

### 1.2 Dataset build

All three steps ran on the login node from the tool tree `G3/tools/bf`; the commands are in `scripts/ag_bf_build.sh` and
`scripts/ag_bf_subsets.sh`.

1. **`scripts/ag_build_ds.py`** `--world crm --crm-runs G3/soil_v1/runs --tasks-crm G3/tasks/soil_v3.json --tiers 0-12
   --arena f104 --vehicle gator --map-root G3/e4/map_roots/f104 --source-root G3/source --workers 16`
   (`scripts/ag_bf_build.sh:16-19`).
   - It selects runs by id pattern and task row, then checks the completion marker, the launch check, soil QA
     (`crm_qa.check`), the map, and the vehicle block.
   - It then runs the unchanged `f104_n2_dataset.py` (labels and corridors), `n2_reanchor_dataset.py` (re-anchored
     rows: a standing-start row plus a row every 2 s of the drive) and the one-world cutter (`ga_build_mixed.cut_episode`:
     history windows and privileged context).
   - Output: 55,826 rows, `G3/e4/soil_bf/f104_gator/ci_f104_gator_crm.npz` (LOG.md:108).
2. **`scripts/ag_e5a_ids.py`** `--world crm --prefix gator__ ...` writes the list of validated ids
   (`ag_bf_subsets.sh:28-30`).
3. **`scripts/ag_subset.py`** `--world crm --tiers 0-12 --eval-rows filtered` (`ag_bf_subsets.sh:21,32-36`):
   - `G_full`: `--groups f104=all --ids-file <validated>`, 1,089 training groups, 50,822 fitted rows;
   - `H_full`: `--preset H` on the HMMWV f104 file with the same ids. It is byte-identical to the full HMMWV file,
     because every id validated;
   - evaluation files: `--groups f104=0 --keep-dev-fold`.

### 1.3 Training

- **Trainer:** `scripts/ci_train.py` (sha `7a4f2d67`), recipe from the job lists `K3/e5/jobs/soil_bf_G.tsv` / `soil_bf_H.tsv`:
  `--ds <subset> --mode deploy --arch gru --cond none --domain-filter crm --ctx geom --split-eval val --bs 256 --epochs 30
  --seeds 5 --seed0 0 --roundtrip-check`. A second lane runs the same with `--mode holdout` (and no round-trip check),
  for the offline scores.
- **Hyper-parameters:** learning rate and weight decay are the trainer defaults (2e-3, 1e-4; `K3/e5/README.md` section 1).
- **Job script:** `scripts/ag_train_gpus.sbatch` (one lane per GPU). Submitted as `sbatch -p mi3501x -t 3:00:00
  --export=ALL,AG_JOBS=<list> ag_train_gpus.sbatch` from `G3/tools/bf`.
- **Cost:** jobs 439361 / 439362 took 19.9 / 19.2 min on one MI350X each, 2 lanes, 0.08 billed together (LOG.md:116, 122).
- **Platform caveats:** the stage-1 soil models were trained on MI250X. pytorch/2.10.0 crashes on MI210 compute nodes.

### 1.4 Deploy and offline scores

- **Deploy:** `scripts/ag_deploy_sync.sh <model> <train subdir> <tag>` copies the 5 checkpoints, the trainer summary and
  the logits to `K3/e5/deploy/<model>/` with a `SHA256SUMS` file that must match the cluster. Result:
  `K3/e5/deploy/G_full_soil/G_full_soil_deploy_s{0..4}.pt`; manifest `K3/e5/deploy/soil_bf_models.json`.
- **Offline:** `scripts/ag_bf_offline.sh` runs `scripts/ag_offline_auc.py` with `--model name=glob --eval name=file
  --check-trainer`, TF32 off.
  - It gives the within-pair AUC of "unsafe" on the dev fold + val groups.
  - G_full scored 0.946 (0.950 on val only) (LOG.md:117).

### 1.5 Picks

- **Tool:** `scripts/ag_bf_picks.sh` runs `scripts/ag_picks.py --arena f104 --world crm --mode free --cases <800 suite>
  --groups all --models ".../G_full_soil_deploy_s*.pt" --model-tag G_full --out K3/e6/picks/crm_bfull/f104/G_full_free
  --rerun-check`.
- **What "free" mode runs:** `ci_planner.py --family free --world crm --arms B`, i.e. CEM 4 x 64 from the case pose at rest
  (`scripts/ag_picks.py:6,207-216`). The map check runs first. `--rerun-check` plans everything twice and requires
  identical files.
- **Straight route:** `--mode straight6` builds the straight route at 6 m/s. It is vehicle-independent; the existing set is
  `K3/e6/picks/crm/f104/straight6`.
- **Set lock:** `scripts/ag_e6b_lock.py --dirs <globs> --out LOCK_*.sha256`, written before any drive (lock `52f510b7`,
  LOG.md:120).

### 1.6 Frozen analysis spec

- `scripts/ag_bf_spec.py` writes `K3/e6/analysis/spec_soil_v1_Bfull.json`; its sha256 is recorded before any model,
  pick or drive exists.
- Settings: `margin_pts 2.0, alpha 0.05, boot 4000, seed 0, cluster_key 'cluster', min_groups 50`.
- Family F1-F4 (fail = goal not reached, set `f104_800`): F1 declared primary G_full vs H_full on the Gator, F2/F3 vs
  straight 6 m/s, F4 G_full vs stage-1 G. Secondaries follow.

### 1.7 Drive rows

- **Rows:** `scripts/ag_bf_rows.sh` runs `scripts/ag_eval_tasks.py build --world crm --tier -9 --arm
  Gfull_free_gator=".../G_full_free@gator" ... --existing K3/e3/tasks/soil_v3.json`.
  - Identical routes are merged into one drive.
  - Routes already driven with the same case and content are reused, not re-driven.
  - The vehicle is always explicit on the row (`ag_eval_tasks.py:165-170`).
- **Task file:** `scripts/ag_bf_soil_v4.py` makes `soil_v4.json`, a strict superset of soil_v3 plus the new rows.
  `ag_eval_tasks.py stage --execute` copies the files and checks their hashes.
- **Launch:** from the e3/README recipe with `ag_soil_launch.sh` (439414 mi3501x x 7, 439415 mi2104x x 5, 439426
  mi2101x x 10, 4 h). This was 2,318 new drives, 15.4 simulated h and 5.04 billed, i.e. **about 0.33 billed per
  simulated hour** (LOG.md:120-122).

### 1.8 Sync, index, analysis

- **Sync and index:** `scripts/ag_bf_sync.sh` copies each run's outcome, trajectory, completion marker, case and
  vehicle_extra, then runs `scripts/ag_eval_index.py --mapping <build>.mapping.json --runs K3/e6/runs_soil --out
  K3/e6/index/soil_eval_bfull.json`. The index has 6,400 rows (8 arms x 800).
- **Labels** come from `ga_analyze.safe_labels`:
  - fail = goal not reached;
  - unsafe = not (goal reached, less than 0.05 s rolling backwards under throttle, and minimum forward speed above
    -0.30 m/s).
- **Analysis:** `scripts/ag_analyze.py --index ... --spec ... --out results_soil_v1_Bfull.json`.
  - Paired group bootstrap, a cluster bootstrap over the nearest terrain feature (one-sided p), and exact McNemar.
  - Holm over the family, and the ±2-point "no meaningful difference" rule.
  - Headroom closed = (planner - straight) / (1 - straight). Time ratio on pairs both arms complete.
- **Drive QA:** `scripts/ag_s2_extras.py` gives QA flags, launch failures, non-finite states and the belly flag.
- **Verification:** an independent verifier recomputed everything (VERIFY_gator_full.md).

### 1.9 The 800 suite

- 600 `f104_pair_group` + 200 `f104_crm_eval_group`, standing start, case cases above.
- The index set name is `f104_800` (`ag_eval_index.py:64-75`; `ag_analyze.py:125-138`).
- Caveat carried from REVIEW_R1-6: CEM 4 x 64 was tuned on the 200 `f104_crm_eval` groups, so report the 600 fresh pairs
  separately.

### 1.10 Criteria

- **PLAN 3 (task B)**, declared "works":
  1. G beats the straight route on the Gator (lower bound > 0);
  2. G beats H on the Gator, or is within 2 points of it;
  3. offline within-group AUC on held-out groups >= 0.95;
  4. headroom closed, compared with the HMMWV's (reported, no threshold).
- **PLAN 7.7**, "collects the same data":
  - >= 95 % of ids validated, < 1 % crashed or NaN, < 5 % launch-check failures;
  - the belly-in-soil flag (lowest body point more than 0.05 m under the surface for more than 1 s) on <= 10 % of soil
    drives, else "not physically trustworthy";
  - H trained on exactly the validated ids;
  - a wheel-radius sensitivity pilot; a change of more than 15 points means the result depends on the wheel model.
- **PLAN 7.10:** stage-1 cut at tiers 0-6; H cut to the Gator's validated ids. The all-tier "Bfull" retrain was run once
  the collection finished (the 67.4 % result).

### 1.11 Exact command chain for a new vehicle tag V

Scripts marked **[new]** must be generalised copies; the existing files stay frozen. Everything else runs unchanged.

```bash
# --- local env
cd /home/harry/NeDM-traverse_mppi; export PYTHONPATH=src:scripts; unset NEDM_VEHICLE
PY=/home/harry/miniconda3/envs/nedm/bin/python; K4=artifacts/traverse/offroad_vehicles_20260927
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925; G4=/work1/dannegrut/harry/experiments/offroad_vehicles_20260927
SUITE=artifacts/traverse/generalist_20260921/A_adapt/suite/cases

# 0. collection (other scouts): rows "<V>__<collect_v1 id>" for the 15,235 ids, tiers 0-12, extra ['--vehicle', V],
#    frozen [new] dispatcher (ag_crm_collect.py generalised), launched with a [new] ag_soil_launch.sh / ag_soil.sbatch
#    whose G3 / CRM_ROOT is G4.  QA: [new] copy of ag_s1_gator_soil_qa.py with the prefix as an argument.

# 1. build (login node, numpy env): [new] ag_build_ds.py with V in VEHICLE_PREFIX and a generic vehicle-block check
$NRD_PYTHON -u scripts/ov_build_ds.py --world crm --crm-runs $G4/soil/runs --tasks-crm $G4/tasks/soil_V.json --tiers 0-12 \
   --arena f104 --vehicle V --map-root $G3/e4/map_roots/f104 --source-root $G4/source --out $G4/e4/f104_V --workers 16
# 2. validated ids (unchanged; --prefix exists)
$NRD_PYTHON -u scripts/ag_e5a_ids.py --world crm --prefix V__ --ci $G4/e4/f104_V/ci_f104_V_crm.npz \
   --record $G4/e4/f104_V/f104_V_crm_record.json --tasks $G4/tasks/soil_V.json --runs $G4/soil/runs \
   --out-ids $G4/e5/ids/V_soil_validated.txt --out-json $G4/e5/ids/V_soil_validated.json
# 3. subsets (unchanged; --ids-strip-prefix exists)
TC="--world crm --tiers 0-12 --eval-rows filtered"
$NRD_PYTHON -u scripts/ag_subset.py $TC --groups f104=all --ds $G4/e4/f104_V/ci_f104_V_crm.npz --ids-file $G4/e5/ids/V_soil_validated.txt \
   --ids-strip-prefix V__ --out $G4/e4/subsets/V_full_f104_V_soil.npz
$NRD_PYTHON -u scripts/ag_subset.py $TC --groups f104=0 --keep-dev-fold --ds $G4/e4/f104_V/ci_f104_V_crm.npz \
   --ids-strip-prefix V__ --out $G4/e5/eval/EV_f104_V_full_crm.npz
#    (H_V = --preset H on $G3/e4/f104_hmmwv/ci_f104_hmmwv_both.npz only if V misses ids; if all 15,235 validate, H_V = H_full)
# 4. training (cluster, MI350X): job list lines  "lane<TAB>tag<TAB>out<TAB>args"
#    1  V_full_soil_deploy   e5/train/V_full_soil   --ds $G4/e4/subsets/V_full_f104_V_soil.npz --mode deploy --arch gru --cond none
#       --domain-filter crm --ctx geom --split-eval val --bs 256 --epochs 30 --seeds 5 --seed0 0 --roundtrip-check
#    2  V_full_soil_holdout  e5/train/offline    (same, --mode holdout, no --roundtrip-check)
env -u NEDM_VEHICLE sbatch -p mi3501x -t 03:00:00 -J ov_train_V -o $G4/e5/logs/%x_%j.out \
   --export=ALL,AG_JOBS=$G4/e5/jobs/soil_V.tsv,AG_G3=$G4 $G4/tools/scripts/ag_train_gpus.sbatch    # needs $G4/source with ci_train.py 7a4f2d67
# 5. deploy sync ([new] copy of ag_deploy_sync.sh with K4/G4), then offline AUC (unchanged)
NVIDIA_TF32_OVERRIDE=0 $PY -u scripts/ag_offline_auc.py --model "V_full_deploy=$K4/e5/deploy/V_full_soil/V_full_soil_deploy_s*.pt" \
   --model "V_full_holdout=$K4/e5/train/offline/V_full_soil_holdout_s*.pt" --eval f104_V_all=$K4/e5/eval/EV_f104_V_full_crm.npz \
   --check-trainer --out $K4/e5/offline/V_auc.json
# 6. picks (local 5090): Gator protocol (unchanged)
$PY scripts/ag_picks.py --arena f104 --world crm --mode free --cases $SUITE --groups all \
   --models "$K4/e5/deploy/V_full_soil/V_full_soil_deploy_s*.pt" --model-tag V_full --out $K4/e6/picks/f104/V_full_free --rerun-check
#    best planner: ci_grad from a standing start (no --poses; checked in section 2.4) + [new] manifest writer (ag_picks.json)
$PY scripts/ci_grad.py --cases $SUITE --map-root artifacts/traverse/crm_f104_v1/map_root \
   --models "$K4/e5/deploy/V_full_soil/V_full_soil_deploy_s*.pt" --world crm --domain crm --task-root $K4/e6/picks \
   --out $K4/e6/picks/f104/V_full_grad --ref-b-picks $K4/e6/picks/f104/V_full_free/picks
#    straight 6 m/s: reuse K3/e6/picks/crm/f104/straight6 (vehicle-independent routes)
$PY scripts/ag_e6b_lock.py --dirs "$K4/e6/picks/f104/*" --out $K4/e6/picks/LOCK_f104.sha256
# 7. spec: [new] ov_spec.py -> $K4/e6/analysis/spec_ov.json, sha256 written BEFORE any drive
# 8. rows: [new] ag_eval_tasks.py with V in VPREFIX (and G4 roots)
$PY scripts/ov_eval_tasks.py build --world crm --tier -9 --arm Vfull_free_V="$K4/e6/picks/f104/V_full_free@V" \
   --arm Vfull_grad_V="$K4/e6/picks/f104/V_full_grad@V" --arm straight6_V="K3/e6/picks/crm/f104/straight6@V" \
   --existing $G4/tasks/soil_V.json --out $K4/e6/tasks/soil_eval_V.json
$PY scripts/ov_eval_tasks.py stage --tasks $K4/e6/tasks/soil_eval_V.json [--execute]
#    superset task file ([new] copy of ag_bf_soil_v4.py) -> launch with the [new] ov_soil_launch.sh (same recipe as 1.7)
# 9. after the drives: sync ([new] copy of ag_bf_sync.sh), index ([new] ag_eval_index.py with a generic vehicle-block
#    assert), extras ([new] ag_s2_extras.py with belly/body flags for V), analysis (unchanged)
$PY scripts/ag_analyze.py --index $K4/e6/index/soil_eval_ov.json --spec $K4/e6/analysis/spec_ov.json --out $K4/e6/analysis/results_ov.json
```

Timings to expect:
- CEM picks: 0.4 s per pair.
- ci_grad: 2.6-3.9 s per pair on a free 5090, so about 35-50 min per model for 800 pairs.
- Training: about 20 min per 5-seed ensemble (2 lanes on one MI350X).
- Soil drives: about 0.33 billed per simulated hour.

### 1.12 Every place the vehicle (or the Gator study) is hard-coded

| file:line | what is hard-coded | effect for V |
|---|---|---|
| `scripts/ag_vehicle.py:39,65,79-81` | `VEHICLES = ("hmmwv", "gator")`, argparse choices, NEDM_VEHICLE check | V refused; the switch needs a new vehicle factory (other scouts) |
| `scripts/ag_crm_collect.py:60-98` | `install_gator` only (Gator cylinders, spawn, belly hooks) | new dispatcher; also needed for `--base crm_collect_ext` branch mode (moving start) |
| `scripts/ag_build_ds.py:51,272` | `VEHICLE_PREFIX = dict(hmmwv='', gator='gator__')`, `--vehicle` choices | V refused |
| `scripts/ag_build_ds.py:346` | vehicle-block check `vehicle != 'gator' if a.vehicle == 'gator' else vehicle is not None` | a V run with a vehicle block would be rejected as "HMMWV" |
| `scripts/ag_build_ds.py:85-86,332` | f104 tier = `crm_tasks` per-group shuffle of `route_00-11` / `op_00-07`, asserted equal to the task tier | extra (wide-shape) routes on f104 raise KeyError: needs a builder variant (section 2.5) |
| `scripts/ag_eval_tasks.py:53-54,97-100,170` | `GATOR_FP`, `VPREFIX` (hmmwv / gator), `parse_arm` assert, the rigid-only fingerprint for the Gator | V arms refused |
| `scripts/ag_eval_index.py:156` | `assert (vb == 'gator') == (vehicle == 'gator')` | does not check that a V run really carries a V block (it would pass on an HMMWV run); needs `(vb or 'hmmwv') == vehicle` |
| `scripts/ag_subset.py:76,89`; `scripts/ag_e5a_ids.py:24` | default prefix `gator__` | fine: overridable (`--ids-strip-prefix`, `--prefix`) |
| `scripts/ag_s1_gator_soil_qa.py:47,50` | `startswith('gator__')` | collection read-out needs a copy |
| `scripts/ag_s2_extras.py:46,68,78` | belly flag and simulated hours only for `gator` / `hmmwv` | no belly flag for V |
| `scripts/ag_bf_soil_v4.py:24-25,76,107` | K3/G3, `vehicle in ('hmmwv', 'gator')` | copy |
| `scripts/ag_bf_spec.py:36,58-78` | arm names, `GA = 'gator'` | new spec script |
| `scripts/ag_analyze.py:59-86` | only the template's arm/vehicle names | fine: spec-driven, vehicle strings free, `any` pools vehicles |
| `scripts/ag_tasklib.py:14-16` | `K3`, `G3` = arena_gator roots (used by ag_eval_tasks / ag_eval_index / ag_picks for non-f104 map roots and staging) | rows and staging go to G3 unless copied |
| `scripts/ag_soil.sbatch:21`, `scripts/ag_soil_launch.sh:9` | `CRM_ROOT` / `G3` = arena_gator | copy for G4 |
| `scripts/ag_train_gpus.sbatch:14,16,23` | `#SBATCH -o` G3 log path; G3 default (`AG_G3` overrides); `cd $G3/source` | pass `-o` and `AG_G3`; G4 needs a source tree |
| `scripts/ag_deploy_sync.sh:6`; `ag_bf_*.sh` (`ag_bf_build.sh:10`, `_subsets.sh:13`, `_picks.sh:12`, `_offline.sh:14`, `_sync.sh:10`) | K3/G3 constants, model names G_full/H_full | copies |
| `scripts/sp_search_probe.py:58-63` | K3, `REF_DIR` = G_full picks, `INDEX_ARM 'Gfull_free_gator'`, OUT folder | cannot plan for another model without a wrapper (section 2.1) |
| `scripts/crm_collect.py:135-153,195,220` | the soil state fields: `vehicle.GetWheel/GetTire/GetSpindle*` for 4 wheels | fine for Polaris (4 wheels); M113 has no wheels or spindles (section 2.3) |
| `src/nedm/traverse/fdm_mppi.py:34-35`; `scripts/gen_planner.py:30` | planner footprint = HMMWV size (half 2.6 x 1.3 m), arena ±40 m | only matters at the arena edge; conservative for Polaris |

Vehicle-independent, reusable as is:
- the planner and the pick tool;
- the risk model's inputs (`--cond none --ctx geom`);
- the labels (from forward speed, throttle and the parked flag);
- the straight-route picks;
- `ag_analyze.py`.

The dataset builders only require the collector contract:
- `trajectory.npz` state `(n, 17)`, action `(n, 3)`, pose `(n, 3)` (`ga_build_mixed.py:54`; `crm_qa.py:30`);
- `crm_extra.npz` with `pos_z_m`, `bmp_ground_z_m`, `slip_ratio`, `spindle_z_m` (`crm_qa.py:24,35`;
  `ga_build_mixed.py:56-59`);
- `initial_state_validation.json`, `collection_request.json`, and `positive_work_kj_per_interval` (energy targets).

---

## 2. The best planner formulation

### 2.1 Candidate (a): wider route shapes + gradient refinement, standing start

**What exists.**
- `scripts/sp_search_probe.py` arm A4 is 5 sideways sine bends instead of 3, a sideways limit of ±20 m instead of ±10 m,
  and CEM 16 x 512 (`sp_search_probe.py:77-79`).
  - It is implemented by swapping `f104_n2_iter`'s `MODES / PRIOR_SD / from_params / draw_prior / project` inside a
    context manager (`sp_search_probe.py:183-236`).
  - The original family is 3 modes and 10 m (`f104_n2_iter.py:20-21`).
- **Offline result** (the Gator's G_full model; RESULTS.md): on 71 of the 134 pairs the planner knew were risky, A4
  finds a route rated safe (P < 0.05). Gradient refinement within the old family (A3) finds 40, CEM 16 x 512 alone
  (A2) finds 27.

**What does not exist.**
- **Nothing is driven.**
- **The wide family has no gradient version.** `ci_grad.py` ties its route chain to the 3-mode, 10 m family:
  - the import-time assert (`ci_grad.py:70`);
  - the `lat_clip` default bound at definition (`ci_grad.py:238-244`);
  - `np_route` uses `IT.caps`, `IT.MODES`, `IT.LAT_CLIP` (`ci_grad.py:256-273`);
  - `family_patch` does not patch `LAT_CLIP` or `caps`.
  - A "wide + gradient" arm therefore needs a new wrapper and a self-test (steps 0 must reproduce the A4 CEM pick).
- **The probe cannot plan for a new model.** It is hard-wired to G_full and to the probe folder
  (`sp_search_probe.py:58-63`), so V needs a wrapper (e.g. `ov_picks.py` importing its `make_family` / `family_patch`).
  That wrapper must write the `ag_picks.json` manifest that `ag_eval_tasks.py build` requires (`ag_eval_tasks.py:113-128`).
  `ci_grad.py` does not write one either.

**Out of the training range (the main problem).**
- Designed training routes stay within 4 m of the straight line; planner-proposed training routes stay within 10 m
  (search probe checks/training_route_extent.json).
- 47 of A4's 71 safe routes go beyond 10 m, so the model is extrapolating exactly where the gain is claimed.
- The collection for V uses the same 15,235 task ids, hence the **same route files**, so V's model will have the same
  blind spot.

### 2.2 Candidate (b): the crm_improve final configuration

**What it is.**
- **Model:** shared CNN-GRU with a 2 s history encoder, `deploy_v1/deploy_a1_haux_gru_s*.pt`
  (`crm_improve_20260922/deploy_v1/deploy_a1_haux_gru.json`).
  - Flags: `--arch gru --hist-enc gru --cond hist_aux --domain-filter both --crm-batch-frac 0.5 --hist-drop 0.2
    --aux-weight 0.5 --ctx geom`, 5 seeds x 30 epochs.
  - Training data: three files, both worlds, HMMWV only:
    - `mixed_reanchor_plus_branch_both.npz`: re-anchored rows plus the generalist moving-prefix branch rows;
    - `short_anchor.npz`: rows at frames 10/20/30 of every training drive, i.e. 0.5/1/1.5 s after the start;
    - `anchor_k60.npz`: rows at frame 60.
- **Protocol:**
  - Pass 1 drives a 0.5 s straight approach at 3 m/s on every suite pair (`ci_a5data.py --stage pass1-tasks`, native
    mode, `--horizon-s 0.5`).
  - `ci_a5data.py --stage analyze --length 0.5` reads the decision state and the 40-frame history window at frame 10
    (`--vx-min 0.3` default below 3 s).
  - The planner plans from that state: `ci_planner.py --family free --poses poses_crm_all.json`, then
    `ci_grad.py --poses ...`.
  - Pass 2 replays the approach and switches to the pick at frame 10: `ga_a5_pass2_tasks.py --branch-frame 10`, rows
    with `--mode branch --branch-frame 10 --branch-route <abs path>` (`ga_a5_pass2_tasks.py:38-40`), collector
    `crm_collect_ext.py`.
  - Scripts: `crm_improve_20260922/s2/run_s2_new.sh`, `s4/run_grad_picks.sh`, `s4/run_grad_drives.sh`.

**What it would take for V.**

- **Training rows (no extra drives).**
  - Re-anchored rows with history are already in the `ag_build_ds.py` file (it cuts history with
    `ga_build_mixed.cut_episode`).
  - Short-anchor rows come from the same 15,235 drives via `scripts/ci_short_anchors.py --ks 10 20 30`. It cannot read an
    `ag_build_ds` file as `--ref` directly: that file has the extra `arena / vehicle / tier` columns, and
    `ci_short_anchors.py:281,286` build every reference key from the rows, so the build stops at the first unknown key.
    A stripped reference copy or a wrapper is needed.
  - Its defaults point at the HMMWV f104 files (`ci_short_anchors.py:39-40`); runs are found by episode id
    (`ci_short_anchors.py:146`), so `--crm-runs` must name V's run folder.
  - The model would be `--cond hist --domain-filter crm --hist-drop 0.2`. The auxiliary head of `hist_aux` predicts
    rigid vs soil, which is meaningless with one world.
  - `ci_train.py` merges several `--ds` files on their common keys (`ci_train.py:265`), so the extra columns are simply
    dropped.
  - The branch rows (2,256 soil, HMMWV) and the S3 continuation data are **not needed**: S3 showed they add nothing
    offline (REPORT item 7).
- **Drives.**
  - Pass 1: 800 short approach drives per vehicle, dominated by set-up time (K1: 800 drives = 1.65 GPU-h, about 0.25
    billed).
  - Pass 2: one drive per arm per pair, the same count as a standing-start arm plus 0.5 s.
  - Roughly **+0.3 billed per vehicle** over the standing-start protocol.
- **Tooling (the real cost).**
  - `ci_a5data.py` is hard-wired to the K1/HMMWV cluster roots: `CRM_ROOT = crm_f104_20260916`, the K1 case copies, the
    K1 H models (`ci_a5data.py:72-90`). Its rows carry no vehicle switch.
  - `ga_a5_pass2_tasks.py` rows have no `--vehicle` / `--base crm_collect_ext`.
  - The branch mode of `crm_collect_ext.py` has never been driven with a non-HMMWV vehicle: E2 checked native and
    pid_held only (LOG.md:23).
  - Prefix-replay determinism must be re-checked per vehicle (K1 found 9 of 10 byte-identical).
- **Hard-wired to f104.**
  - `ci_a5data` (suite and twin groups), `ci_short_anchors` (map root), `ci_grad` (`--arena-tag f104`,
    `ARENA_SOFT = 37 m` from `f104_n2_grad.py:37`), and the planner map root.
  - Since this study **is** on f104 with the same suite and map, none of that matters. Only the HMMWV/K1 roots and the
    vehicle switch matter.

### 2.3 The history state fields, and whether they exist for a tracked vehicle

- **Columns.** The history window is 40 frames (2 s) of 12 state columns + 3 actions (`ga_build_mixed.py:27`,
  `gc_control.OBSERVABLE_COLS`):
  - vx, vy, roll, pitch, roll rate, pitch rate, yaw rate (columns 0-6);
  - **4 spindle (wheel) angular speeds** FL/FR/RL/RR (columns 11-14);
  - engine speed (column 15);
  - steering, throttle, brake.
- **Privileged context** (training only) = 4 wheel normal forces, engine torque, soil slip and sinkage
  (`ga_build_mixed.py:28,63-72`).
- **Polaris.** 4 wheels on 2 axles, JSON engine map (`vehicle/Polaris/Polaris_EngineSimpleMap.json`): every field
  exists with the same code (`crm_collect.py:135-153`).
- **M113.** No wheels, tyres or spindles. `GetWheel/GetTire/GetSpindle*` do not exist on a tracked vehicle, so the
  collector's state row (`crm_collect.py:195,220`) and the soil extras must be redefined.
  - Available substitutes: left/right sprocket speed (`driveline.GetSprocketSpeed(LEFT/RIGHT)`, already used in
    `src/nedm/tracked_vehicle_data.py:287-288`).
  - The engine exists (`vehicle/M113/powertrain/M113_EngineSimpleMap.json` etc.).
  - Normal force, slip and sinkage per wheel have no direct equivalent. Road-wheel or track-shoe quantities would have
    to be chosen.
- **What this means for the default.** With the default model (`--cond none`) **none of these columns is read by the
  model**. The builders only assert the (n, 17) shape and finite values. So the default planner needs no tracked-vehicle
  field mapping; formulation (b) would.

### 2.4 Local check: gradient refinement from a standing start works with the Gator's models

Command (scratch; about 11 s including start-up):

```bash
$PY scripts/ci_grad.py --cases artifacts/traverse/generalist_20260921/A_adapt/suite/cases \
  --map-root artifacts/traverse/crm_f104_v1/map_root --models "$K3/e5/deploy/G_full_soil/G_full_soil_deploy_s*.pt" \
  --world crm --domain crm --groups f104_crm_eval_group_0000,f104_pair_group_0120,f104_pair_group_0440 \
  --task-root <scratch> --out <scratch>/grad_standing_smoke --ref-b-picks $K3/e6/picks/crm_bfull/f104/G_full_free/picks
```

Result (`scratch/S4/grad_standing_smoke.log`):
- 5 members, conds `none`, defaults (17 starts, 60 steps, pessimistic keep, abstain 0.3).
- **The CEM pick equals the recorded Gator pick on 3 of 3 pairs** (`ref B 3/3`).
- All 17 finals are valid on each pair. The pick changed on 3 of 3, with gains of 0.69-1.68 logit.
- 2.6-3.9 s per pair, 0.44 GB GPU.

So the command line takes the standing start when `--poses` is omitted (`ci_grad.py:534,567-570`; decision state from
`ga_planner.decision_for`, `ga_planner.py:288-293`: layout pose, route_00, all-masked history). The search probe's A3
had already used the same path in-process.

What is missing for the evaluation: an `ag_picks.json` manifest next to ci_grad's `picks/`, `routes/` and `tasks.json`,
so that `ag_eval_tasks.py build` can read the folder. That is a small [new] writer.

### 2.5 The training range for wider shapes: a cheap fix and its cost

**Option W0: no new data.** Search with the probe's A4b family (5 bend terms, 10 m limit). Its sideways range stays
inside the planner-proposed training routes. Offline: 47 of 134 risky Gator pairs rated safe, against 71 for A4, 27 for
A2 and 40 for A3. Cost 0.

**Option W1 (recommended if the wide arm is run): add wide-shape routes to the collection.**
- **Which routes.** 2 extra routes per group on all 1,200 groups (training, val and test splits, so that held-out wide
  routes exist for an offline check).
  - Each is a draw from the A4 prior (5 modes, ±20 m, the same curvature caps and speed knots, the unchanged route
    checker).
  - Draws are rejected until the route swings more than 10 m sideways at its widest point, to fill exactly the missing
    range.
  - The seed is md5(group + 'wide' + k), for reproducibility.
- **Collection.** Tiers 13-14, so the Gator-comparable model trains with `--tiers 0-12` and the wide model with
  `--tiers 0-14`. The rows can be queued with the standard ids from the start; no model is needed to make them.
- **Cost.** 2,400 drives per vehicle at about 25-37 simulated s each (HMMWV mean 21.6 s, Gator 33.3 s; wide routes are
  about 1.13x longer). That is **about 17-25 simulated hours, about 6-8 billed node-hours per vehicle**, and about 1.5-2 h
  of wall time at the earlier 15 simulated h per wall hour.
  - The Gator needs the same 2,400 rows for the fair comparison: about 7 billed.
- **Tooling.**
  - A [new] route writer (from `sp_search_probe.make_family`, written in the collector's route format).
  - A [new] builder variant: `ag_build_ds.py` raises KeyError on f104 ids outside `route_00-11 / op_00-07`
    (`ag_build_ds.py:85-86,332`). Use ids `..._op_20` / `_op_21` and take the tier from the task file.
- **Offline check before any drive.** Score V_full (trained without wide rows) on the held-out wide rows. If its
  within-group AUC there is close to its AUC on normal routes, extrapolation was not the problem.

**Option W2 (not recommended tonight).** Collect the A4 picks of a first V model on the training groups, 1,200 more
drives. It is more targeted, but serial: train, plan 1,200 x about 10-20 s of wide CEM, collect, retrain.

### 2.6 Recommendation: which planner arms

| role | arm (plain label) | model | planner | why |
|---|---|---|---|---|
| **primary / default ("best planner")** | V, own model + gradient | V_full (all 15,235 V soil ids, `--cond none --ctx geom`, 5 seeds x 30 epochs) | CEM 4 x 64 + ci_grad defaults (17 starts, 60 steps, pessimistic keep, abstain 0.3), standing start | the only upgrade with driven evidence (+1.7 points HMMWV at 0.5 s, +1.0 at a standing start); the tooling exists and is checked (2.4) |
| comparability | V, own model, Gator protocol | V_full | CEM 4 x 64, standing start | identical to the Gator's 67.4 % protocol |
| anchor | straight route 6 m/s on V | none | straight | headroom; the planner must beat it |
| secondary (exploratory) | V, wider shapes | V_full+W (15,235 + 2,400 wide rows, `--tiers 0-14`) | A4 CEM 16 x 512 | biggest offline gain, never driven; only with W1 rows; else A4b with V_full |
| not run | V, 0.5 s approach + history + gradient | history model | two-pass | expected about 0 over the primary for one known world and vehicle (0 summary); costs tooling and M113 field mapping |
| not run unless built and self-tested before the spec freeze | V, wide + gradient | V_full+W | A4 + gradient | needs a new chain (2.1) |

---

## 3. Proposed declared evaluation

### 3.1 Suite

- **The 800 suite on f104 soil, standing start**, the Gator's exact cases and map (`crm_f104_v1/map_root`).
- Every result is reported on all 800 and also on the **600 fresh `f104_pair_group` pairs** (CEM was tuned on the 200
  `f104_crm_eval` pairs).
- Soil only: rigid f104 was at the ceiling for both earlier vehicles, and the question is soil.

### 3.2 Arms

Only for the vehicles that pass the smoke gate against the Gator. Drive counts are before identical-route merges.

| # | arm | vehicle | model / planner | drives | status |
|---|---|---|---|---|---|
| 1 | `Vgrad_V` (best, default) | V | V_full + CEM + gradient | 800 | new |
| 2 | `Vcem_V` (Gator protocol) | V | V_full + CEM 4 x 64 | 800 | new |
| 3 | `straight6_V` | V | straight, 6 m/s | 800 | new (routes reused) |
| 4 | `Ggrad_gator` (Gator, best planner) | Gator | G_full + CEM + gradient | 800 | new |
| 5 | `Gcem_gator`, `straight6_gator` | Gator | existing | 0 | reuse the K3 drives (67.4 % / 14.3 %) if the new dispatcher reproduces 3-5 old Gator rows bit for bit; otherwise re-drive (about 2 billed each) |
| 6 | `Hgrad_hmmwv` (bridge) | HMMWV | H_full + CEM + gradient | 800 | new; confirms the standing-start default against crm_improve's 97.5 % (H_full CEM is 95.6 %, reused) |
| 7 | `Vwide_V` (secondary) | V | V_full+W, A4 CEM 16 x 512 | 800 | only with W1 rows |
| 8 | `Gwide_gator` (secondary) | Gator | G_full+W, A4 | 800 | only with the Gator's W1 rows |
| 9 | `Hfull_V` (transfer, optional) | V | H_full + CEM (picks reused) | 800 | cheap; shows how vehicle-specific the planner is |

### 3.3 Metrics

- **Goal reached** (fail = not reached): the paired family label, comparable with every earlier result.
- **Reached safely** = 1 - unsafe (unsafe = not reached, or reached after rolling backwards under throttle or below
  -0.3 m/s). This is the user's bar. On soil the two labels almost coincide (soil short-anchor rows: fail 0.672 vs
  unsafe 0.673).
- **Also reported:**
  - the rate with belly/body-in-soil-flagged drives counted as failures;
  - rollover and non-finite counts;
  - the time ratio on pairs both arms complete;
  - headroom closed against the same vehicle's straight route;
  - per terrain stratum;
  - mean speed of the picks and the share with 3 or more speed knots at the limit (the gradient's known exploitation
    sign, search probe section 4);
  - offline within-pair AUC on V's held-out groups (dev fold + val).

### 3.4 The 90 % bar (declared on the primary arm `Vgrad_V`)

- **Meets the bar:** reached-safely rate >= 90.0 % on the 800 pairs **and** >= 90.0 % on the 600 fresh pairs (point
  estimates).
- **Clearly above the bar:** additionally, the one-sided 95 % lower bound of the cluster bootstrap over terrain-feature
  clusters (B = 4000) is >= 90 %. At n = 800 and p ≈ 0.9 one standard error is about 1.1 points, so this needs about
  91.8 % or more.
- **Below the bar:** a point estimate under 90 %. Report the gap and the 95 % interval.
- `ag_analyze.py` has no one-sample test, so the bound needs a small [new] add-on that reuses its cluster assignment.

### 3.5 Paired tests (one Holm family at 0.05, one-sided cluster-bootstrap p, exact McNemar alongside)

For each vehicle V that goes ahead (2 x 3 tests, plus F4):

| test | contrast (label fail, set f104_800) | question |
|---|---|---|
| F1_V | `Vgrad_V` on V vs `Ggrad_gator` on the Gator (vehicle `any`, paired by pair id) | does V with the best planner beat the Gator with the best planner? (the headline "better than the Gator") |
| F2_V | `Vgrad_V` vs `straight6_V` | does the planner work on V? |
| F3_V | `Vgrad_V` vs `Vcem_V` | does the default upgrade help on V? |
| F4 | `Ggrad_gator` vs `Gcem_gator` | does the upgrade help the Gator? |

- **Decision rule:** the ag_analyze rule. "Improves" if Holm rejects. "No meaningful difference" if the 90 % cluster
  interval lies inside ±2 points. Otherwise "inconclusive".
- **Secondary, unadjusted:**
  - `Vcem_V` vs `Gcem_gator` (like-for-like, the Gator protocol);
  - `Vwide_V` vs `Vgrad_V` and `Gwide_gator` vs `Ggrad_gator` (wide shapes);
  - `Hgrad_hmmwv` vs `Hcem_hmmwv` (bridge);
  - `Vgrad_V` vs `Hgrad_hmmwv` (gap to the HMMWV);
  - `Hfull_V` vs `Vcem_V` (vehicle-specific training);
  - the same contrasts on "unsafe"; the 600 fresh pairs; per stratum; time ratios.
- **Order of events** (as in K3): models, then picks locked with sha256 (with `--rerun-check`), then the spec frozen
  with sha256, then the drives. An independent verifier recomputes everything from the raw drives.

### 3.6 What counts as "the planner works" on V (declared)

The planner **works** if all three hold:
1. **Beats the straight route:** F2_V improves (Holm).
2. **Meets the 90 % bar** (3.4).
3. **Valid drives:** on the primary arm, 0 crashed or non-finite runs, fewer than 5 % launch-check failures, the
   body/belly-in-soil flag on <= 10 % of drives, and the bar still met when flagged drives count as failures. Otherwise
   the result is reported as "not physically trustworthy".

Other outcomes and read-outs:
- **Helps but below the bar:** 1 and 3 hold, 2 does not.
- **Reported without a threshold:** offline AUC (the Gator plan asked for >= 0.95), headroom closed compared with the
  HMMWV's 86 %, and the time ratio.

### 3.7 Training noise

- Single ensembles differed by up to about 1 point pooled in K3.
- For the vehicle nearest the bar, train a second V_full ensemble (seeds 5-9) and drive its gradient picks (about 800
  drives, 2-3 billed). Otherwise state the noise as unmeasured.

### 3.8 Costs of the evaluation (at about 0.33 billed per simulated hour; the collections are extra)

| item | drives | billed node-hours |
|---|---|---|
| per new vehicle: arms 1-3 | about 2,400 | about 5-6 (at 20-30 simulated s per drive) |
| Gator best planner (arm 4) | 800 | about 2 (26.6 simulated s per Gator evaluation drive in K3) |
| HMMWV bridge (arm 6) | 800 | about 1.4 |
| **subtotal, two vehicles, no wide arm** | about 6,400 | **about 14** |
| wide rows (W1), per vehicle + Gator | 2,400 each | about 6-8 each |
| wide arms 7-8 | 800 each | about 2 each |
| **total with the wide arm, two vehicles** | about 15,600 | **about 41** |
| training (each 5-seed ensemble, deploy + holdout) | - | about 0.1 |

Local 5090 time:
- gradient picks: about 35-50 min per model (V_full x 2, G_full, H_full);
- A4 CEM 16 x 512: probe measurement about 20 s per pair on a busy machine, about 4.5 h per 800 pairs in one process;
- run at most 3 planning processes at once, and release GPU memory between searches (the probe caused a GPU
  out-of-memory incident with 6).

---

## 4. Open risks

1. **The gradient exploits speed on weak vehicles.** The probe's gradient picks averaged above 5 m/s on 36 of 134 pairs,
   and the Gator fails 83 % of constant 6 m/s soil drives. The ci_grad defaults (pessimistic keep, 0.3 margin, 60 steps)
   are gentler than the probe's A3 (mean keep, 300 steps), but watch the pick speeds (3.3).
2. **Calibration.** On the Gator, picks rated below 1 % failed 9.6 % of the time, and picks rated 1-5 % failed 52 %. Any
   stronger search selects the model's mistakes. Only driving settles it.
3. **Near-saturated labels.** If V fails most training routes (the Gator failed 88 %), the model learns little. Check V's
   collection fail rate by tier first; the headroom gate belongs to the smoke test (other scouts).
4. **The collector contract for the M113.** Builders assert a 17-column state and soil extras per wheel. The M113 needs a
   declared substitute mapping (e.g. sprocket speeds in columns 11-14) even for the history-free default, or the
   builder stops.
5. **The H_full byte-identity shortcut** (H_full = the full HMMWV file) holds only if V validates all 15,235 ids.
   Otherwise train an H_V on V's validated ids (`--preset H`).
6. **Reusing old Gator drives** requires that the new dispatcher reproduce the Gator path bit for bit (do what E2 did
   for the HMMWV: 3-5 re-drives with array comparison). If not, re-drive `Gcem_gator` / `straight6_gator` (about 4
   billed).
7. **The wide-shape arm extrapolates** without W1 rows (47 of 71), and wide rows need a builder variant (the f104 tier
   KeyError). A "wide + gradient" arm needs new code.
8. **Single ensembles:** the training noise is unmeasured unless 3.7 is done.
9. **The planner footprint** is HMMWV-sized (half 2.6 x 1.3 m). It is conservative for the Polaris; check the M113's
   size (about 4.9 x 2.7 m, i.e. half 2.45 x 1.35 m, similar).
10. **Ordering and the queue:** 50 tasks per user (keep 5 free); soil throughput swung between 5 and 15 simulated hours
    per wall hour in K3. The session died twice there, so keep a LOG.md that allows a resume.

## 5. Files

- This report: `artifacts/traverse/offroad_vehicles_20260927/scout/S4_planner_eval.md`.
- Scratch (local smoke of ci_grad from a standing start): `artifacts/traverse/offroad_vehicles_20260927/scratch/S4/`
  (`grad_standing_smoke/`, `grad_standing_smoke.log`).
- Sources read:
  - `artifacts/traverse/arena_gator_20260925/{PLAN,REPORT,LOG}.md`, `NOTES_E4.md`, `NOTES_E6a.md`, `e5/README.md`,
    `e5/jobs/soil_bf_*.tsv`, `e6/analysis/spec_soil_v1_Bfull.json`;
  - `artifacts/traverse/crm_improve_20260922/{REPORT,PLAN,LOG}.md`, `NOTES_ci_a5data.md`, `NOTES_ci_grad.md`,
    `NOTES_ci_planner.md`, `datasets/short_anchor_build.md`, `deploy_v1/deploy_a1_haux_gru.json`, `s2/run_s2_*.sh`;
  - `artifacts/traverse/search_probe_20260927/RESULTS.md`; `artifacts/traverse/crm_night2_v1/REPORT.md` section 6;
  - the scripts cited above.
