# Evidence, code and release status

For each milestone this lists:

- the compact table in this folder;
- the study records behind it;
- the code that produced it;
- what is not yet public.

Links go to the experiment commit [`901d6c9`][commit] on GitHub. Files that are **local-only** exist on the
workstation where the studies ran but are not in git. They are identified by repository path and SHA256 so they can
be matched once they are released. The complete copies of some raw data exist only on the AMD HPC Fund cluster.

Checks behind these documents:

- **Every compact table was rebuilt twice.** Two independent passes derived each table from the per-drive records;
  their results agreed and match the study's own analysis. The per-drive records behind the headlines are local-only,
  except for the tracker and, for the shared model, the standing-start and 3 s-approach arms.
- **Hash checks.** All 12 index hashes recorded by the terrain and vehicle analyses (final, supplementary and add-on,
  plus one interim Polaris analysis whose index was kept in a temporary folder outside the repository) match the
  local index files. The drive records of the Gator, Polaris and tracker studies record hashes of the collectors and
  vehicle adapters, and these equal the committed files. The tracker drives also record the hash of the deployed
  actor, which equals the local-only actor file listed under milestone 3.
- **Code at the experiment commit.** For every headline, the code between the study's own closing commit and the
  experiment commit changes by at most one-line blacklist additions or one unused launcher option.

## Milestone 1: depth camera in the loop

- **Table:** [`m1_navigation_missions.csv`](../results/m1_navigation_missions.csv). Each row carries the SHA256 of its
  raw per-mission outcome file.
- **Records:**
  - tracked: [report][nav-report] (section 5a is the corrected run), [plan][nav-plan], [commands][nav-running],
    [task list][nav-tasks], [planner timing][bench];
  - the model's training record, [matched_Dabs.json][dabs], and its three checkpoints, tracked through Git LFS;
  - local-only: `artifacts/traverse/fdm_f104_50h_20260909/nav_v1/local_luffy/summary.json`
    (`f8d03b267b7bcc096d7c64f5d683cf0fc8dece143fdda73fa1f1753a1c6aa85b`) and the 120 corrected per-mission outcome
    files.
- **Beware:** the per-mission outcomes tracked in git are from the *pre-fix* runs (`nav_v1/main/`,
  `nav_v1/local_luffy_v0_foldback/`). A reader of git alone would find only the results affected by the rescue-route
  defect.
- **Code:**
  - the runner, [`nav_runner.py`][nav_runner] and [`nav_online.py`][nav_online];
  - the batch driver, [`nav_local_batch.py`][nav_local_batch];
  - sensing, [`sensor_map_v2.py`][sensor_map_v2] and [`vehicle_corridor.py`][vehicle_corridor];
  - model training, [`sensor_dataset_v2.py`][sensor_dataset_v2] and [`sensor_train_v2.py`][sensor_train_v2];
  - analysis, [`nav_analyze.py`][nav_analyze].

  The static import closure is 53 files (10,128 lines).
- **Commands:** in the tracked commands file. The cluster job files that built the model's training corridors and
  trained it are not in git; a planning-time reconstruction of those steps is [here][labels-recon].

## Milestone 2: shared rigid/soil model and early decision

- **Tables:** [`m2_shared_risk_soil.csv`](../results/m2_shared_risk_soil.csv) and
  [`m2_shared_risk_rigid.csv`](../results/m2_shared_risk_rigid.csv).
- **Tracked records:**
  - reports: [shared-model report][gen-report], [plan][gen-plan], [soil-improvement report][ci-report],
    [plan][ci-plan];
  - the frozen 800-pair [suite][suite];
  - per-pair results: standing start ([soil][a3-crm], [rigid][a3-rigid]) and after the 3 s approach ([soil][a5-crm],
    [rigid][a5-rigid]);
  - the early-decision [text summary][s2-txt];
  - the baseline model's [training record][h-deploy].

**Local-only records**, all under `artifacts/traverse/crm_improve_20260922/`:

| File | SHA256 | Holds |
|---|---|---|
| `s4/results_s4.json` | `d4b2007f3885bdc45b542addda6f8ce6f6323320f0dde58fe0d9c5477658695a` | final configuration and comparison arms, soil, per pair |
| `s4/results_s4_rigid.json` | `438ce5c84a5b4e9b910c2ca5e2a385da93474f3bf3e88fd5ae912b1a6ef75575` | same, rigid |
| `s2/results_s2_crm_vs3s.json` | `da58edcada7d71199d5fa2ab4d8f9fcdf1ce6cc22768c499ebb10c70b7ecebc4` | 1 s and 0.5 s arms against the 3 s baseline |
| `s2/results_s2_crm_vsstand.json` | `123617e68648371a5670e22349aa7c07f0736143be808000f543dcd90d63755c` | 1 s and 0.5 s arms against standing start |
| `deploy_v1/deploy_a1_haux_gru.json` | `99bea5299d07a5bb8ca92e87dec84675075ff05d79ccedbb72cbf2e2c494ef8f` | final model's training arguments and rows |

- **Code:**
  - training rows: [`ga_build_mixed.py`][ga_build_mixed] and [`ci_short_anchors.py`][ci_short_anchors];
  - training: [`ga_train.py`][ga_train] (baseline model) and [`ci_train.py`][ci_train] (final model);
  - suite and approach: [`ga_suite.py`][ga_suite], [`ga_approach.py`][ga_approach] and [`ci_a5data.py`][ci_a5data];
  - planning: [`ga_planner.py`][ga_planner] and [`ci_planner.py`][ci_planner] (iterated sampling),
    [`ci_grad.py`][ci_grad] (gradient refinement);
  - drives: [`crm_collect_ext.py`][crm_collect_ext] (soil) and [`gen_collect_ext.py`][gen_collect_ext] (rigid);
  - analysis: [`ga_analyze.py`][ga_analyze].

  The static closure is 68 files (20,981 lines).
- **Commands:** recorded in [`run_A3_deploy.sh`][run_a3], [`run_a5_planning.sh`][run_a5], [`run_s2_new.sh`][run_s2],
  [`run_grad_picks.sh`][run_grad_picks] and [`run_grad_drives.sh`][run_grad_drives]. The final analysis command was not
  saved. It is reconstructed from the arms, contrasts, margin, bootstrap settings and seed recorded inside the results
  file.

## Milestone 3: learned tracker against PID

- **Table:** [`m3_tracker_routes.csv`](../results/m3_tracker_routes.csv). This is the one headline whose per-route
  results are tracked, so it can be recounted from git alone.
- **Tracked records:**
  - per-route results: [rigid][b0v2-rigid], [soil][b0v2-crm], and round 1 ([rigid][b0-rigid], [soil][b0-crm]);
  - the [route suite][track-suite];
  - the [policy record][policy-meta], with observation layout, reward, NumPy parity and the hashes of the NRD, the
    cache and the terrain grid;
  - the NRD [config][nrd-config] and [gates][nrd-metrics];
  - the PPO [command][run-ppo].

**Local-only records**, all under `artifacts/traverse/generalist_20260921/B_tracker/`. The NRD checkpoint and the cache
manifest are hash-linked to the tracked policy record; the actor's hash is recorded only in the local-only drive
records:

| File | SHA256 | Linked from |
|---|---|---|
| `ppo_v2/actor.npz` (deployed actor) | `ea9388204a952d0a096a59aecfb7d27a54e4fe1f1cb406f1c6772d212baae3d6` | all 846 learned-tracker drive records |
| `nrd_tag_v3/ckpt_best.pt` (NRD) | `9a97a9bf4ab470a89c29fe23c5685c3a0e957100f721f476f89ff58437d0eaba` | policy record |
| `cache_v3/cache_manifest.json` (training episodes) | `0393e8c5d22c6229a989a22cd05347d02048b7edf512573af789fecf662ec064` | policy record |

- **Code:**
  - NRD data and training: [`gb_build_cache.py`][gb_build_cache], [`gb_crop.py`][gb_crop],
    [`gb_nrd_common.py`][gb_nrd_common] and [`gb_train_nrd.py`][gb_train_nrd];
  - PPO: [`gb_tracker_env.py`][gb_tracker_env] and [`gb_train_tracker.py`][gb_train_tracker];
  - the torch-free actor and follower modes: [`gc_control.py`][gc_control];
  - tasks and analysis: [`gb_track_tasks.py`][gb_track_tasks] and [`gb_track_analyze.py`][gb_track_analyze].

  The static closure is 50 files (14,595 lines).
- **Commands:** the NRD job file is not in git; its arguments are in the tracked config. The round-2 task build and
  analysis commands were not saved; they are reconstructed from the settings recorded in the results files.

## Milestone 4a: unseen arenas (HMMWV)

- **Tables:** [`m4_unseen_arenas_hmmwv_soil.csv`](../results/m4_unseen_arenas_hmmwv_soil.csv) and
  [`m4_unseen_arenas_hmmwv_rigid.csv`](../results/m4_unseen_arenas_hmmwv_rigid.csv).
- **Tracked records:**
  - write-ups: [report][ag-report], [plan][ag-plan], [soil results][ag-soil], [rigid results][ag-rigid];
  - frozen-spec analyses: [soil][ag-soil-json], [rigid][ag-rigid-json]; the specs, [soil][ag-soil-spec] and
    [rigid][ag-rigid-spec]; the [Holm family][ag-family];
  - arena selection: [near][arena-sel], [spread][arena-spread];
  - training job lines: [`e5/jobs/`][ag-jobs].
- **Local-only per-drive indexes**, which the analyses hash, under `artifacts/traverse/arena_gator_20260925/e6/index/`:

  | File | SHA256 |
  |---|---|
  | `soil_eval_v1.json` | `d9c86320694f352d24826c4d828fabe1a2f0cd7b7b359891eba76bdde4cd4029` |
  | `rigid_eval_v1.json` | `7c09e4d3abfaf3da205adca2ee672813c93eb96481df24e3c1d4de405567b900` |

- **Code:**
  - arenas: [`traverse_wp7_arenas.py`][wp7_arenas], [`ag_arena_rank.py`][ag_arena_rank] and
    [`ag_spread_select.py`][ag_spread_select];
  - datasets: [`ag_build_ds.py`][ag_build_ds] and [`ag_subset.py`][ag_subset];
  - training: [`ci_train.py`][ci_train];
  - planning and drives: [`ag_picks.py`][ag_picks] and [`ag_eval_tasks.py`][ag_eval_tasks];
  - analysis: [`ag_eval_index.py`][ag_eval_index] and [`ag_analyze.py`][ag_analyze].

  The static closure is 100 files (22,082 lines).

## Milestone 4b: Gator, Polaris, M113

- **Tables:** [`m4_vehicles_f104_soil.csv`](../results/m4_vehicles_f104_soil.csv),
  [`m4_polaris_unseen_soil.csv`](../results/m4_polaris_unseen_soil.csv) and
  [`m4_vehicle_smoke.csv`](../results/m4_vehicle_smoke.csv).
- **Tracked records:**
  - Gator: [results][ag-gator], [analysis][ag-bfull];
  - Polaris and M113: [report][ov-report], [plan][ov-plan], [smoke test][ov-smoke], [collection][ov-collection],
    [planner][ov-planner];
  - analyses: [f104][ov-f104], [unseen arenas][ov-unseen], [smoke Polaris][ov-smoke-p], [smoke M113][ov-smoke-m];
  - the Polaris planner build chain, [`NOTES_M4.md`][ov-notes-m4].

**Local-only records**, all under `artifacts/traverse/`:

| File | SHA256 | Holds |
|---|---|---|
| `arena_gator_20260925/e6/index/soil_eval_bfull.json` | `ce1a5ee1618b9e132ff1fb457a36fc78c30e50d3a69d988cf588b193f158d000` | Gator and HMMWV f104 soil drives |
| `offroad_vehicles_20260927/e6/index/soil_eval_ov_final.json` | `bb5e35ecb29fd20e139238a7b373778aa22ae55c2c45e8a4f5bedb931a6fc3f5` | Polaris f104 drives |
| `offroad_vehicles_20260927/e6/index/unseen_polaris_v1.json` | `49c5b4741423f13bf8555a0e9d5358e24baea51f6961bcebd8e184d100afb4f5` | Polaris unseen-arena drives |
| `offroad_vehicles_20260927/analysis/k4_extract.jsonl` | `511cfaae7af7dea8a5a2919e96e229e1477a0447e317fc7ed73f4c616d3da496` | per-route extract of the collection and the smoke test |
| `offroad_vehicles_20260927/e5/deploy/polaris_full_soil/SHA256SUMS` | `71b5d4a34dd7cbc6d6a94e41e098c757d1669bead1fa47194f63bbb2e375106a` | checksums of the 5 Polaris networks |

- **Code:**
  - Gator adapter and wrapper: [`ag_vehicle.py`][ag_vehicle] and [`ag_crm_collect.py`][ag_crm_collect];
  - Polaris adapter and wrapper: [`ov_vehicle.py`][ov_vehicle] and [`ov_crm_collect.py`][ov_crm_collect];
  - M113 adapter: [`ov_m113.py`][ov_m113];
  - Polaris gradient picks: [`ov_grad_picks.py`][ov_grad_picks];
  - analysis: [`ov_analyze.py`][ov_analyze] and [`ov_unseen_analyze.py`][ov_unseen_analyze];
  - the frozen smoke-test rule: [`ov_smoke_analyze.py`][ov_smoke_analyze].

  The static closures are 70 files (Gator) and 79 files (Polaris).
- **Figure:** the Polaris unseen-arena figure's plotting script was not saved. The data to redraw it are in the two
  unseen-arena tables here.

## Release status

| Item | Size | In git at the experiment commit | Local only | Cluster only |
|---|---|---|---|---|
| Per-task outcomes for every headline | 1 MB | – (this PR adds them) | | |
| Full per-drive evaluation indexes and result files | about 83 MB local-only | tracker; shared-model standing-start and 3 s arms | navigation, shared-model early-decision and final arms, arenas, vehicles | |
| Navigation risk ensemble (3 networks) | 3 MB | yes (Git LFS pointers; upload to GitHub's LFS store not yet verified) | | |
| Final shared-model ensemble (5 networks) | 5 MB | | yes, **no checksum recorded** | copy in the study's cluster root |
| Tracker actor, PPO checkpoint, NRD | 28 MB | | yes; NRD hash in the tracked policy record, actor hash in local drive records, PPO checkpoint hash not recorded | |
| Multi-arena and Gator-study ensembles (16 x 5 networks) | 84 MB | checksums only | yes, checksums match | |
| Polaris ensemble (5 networks) | 5 MB | | yes, with a checksum file | |
| HMMWV f104 labels and datasets, shared-model training files, NRD cache | about 17 GB | | yes | complete episode folders |
| Raw Gator, Polaris and g203/g228 episodes; g203/g228 soil training subsets; Polaris training files; the navigation model's depth-corridor training datasets (exact cluster paths not recorded) | not measured | | | yes |

Collection totals:

- HMMWV on f104: 24,000 rigid routes (203 simulated hours) and 15,235 soil routes (91.5 h).
- HMMWV on g203 and g228: 24,000 rigid routes each; on soil, 8,482 validated routes over both arenas (4,240 + 4,242),
  from only the first 7 routes of each start/goal group.
- Gator on f104: 15,235 soil routes (141.1 h) and 24,000 rigid routes (165.7 h).
- Polaris on f104: 15,235 soil routes (93.0 h).

The planned release puts datasets and all trained model files (including the NumPy actors) in new Hugging Face
repositories. Their manifests, checksums, split definitions and a download helper go in this folder. A manifest will
say `unreleased` for anything without a verified public copy. The paper's Hugging Face dataset is a separate, stable
release and will not change.

## Gaps in the record

- **The original HMMWV soil collection's collector.** The recorded hashes of `crm_collect.py` and `crm_worker.py` do
  not match any version on the branch. Both versions survive in local, unpushed checkpoint commits. Their differences
  from the committed files are name shims for pychrono 10, optional hooks that are off by default, and worker dispatch
  changes. On the cluster's Chrono build the collection physics is unchanged.
- **Per-drive code provenance.** Soil drives carry the collector hash in their synced outcome files. The fuller
  per-drive record (helper-module, wrapper, rigid-collector and per-file source hashes) and the soil worker's version
  were not synced from the cluster, so rigid drives have no local code hash.
- **Unsaved commands.** The cluster job files for the navigation model's datasets and training, the NRD round-2
  training and the final shared model's training are not in git. Several analysis commands were not saved and are
  reconstructed from parameters inside their result files. How one of the final model's three training files was cut
  from a larger build is not recorded.
- **The final shared-model ensemble has no checksum of record.** One should be generated and checked against the
  cluster copy before release.
- **Absolute paths.** About 2,500 absolute symlinks in the recent study folders point into the workstation's home
  directory. Many scripts hard-code cluster roots or workstation paths.

## Notes for the code pull requests

- **Minimal scope.** The static closures above (53-100 files per headline, overlapping heavily) are a conservative
  starting list, not a strict upper bound. They count only `scripts/` and `src/`, so the code PRs must also add the
  tracked driver scripts inside the study folders and a few files the walk missed (the first label builder behind the
  navigation data, and the multi-GPU training job script actually used for the Gator models). They include some older
  modules from the project's earlier learned-dynamics planner that the current planner imports only for route shapes,
  the route validator and terrain utilities. The frozen rigid collector is loaded by path.
- **Two shared files that the published studies also use.** The branch modified both. Every headline depends on the
  first, and every headline with rigid-ground drives on the second:
  - `src/nedm/training/constants.py` gains the 17-number state preset that every collector selects;
  - `src/nedm/hmmwv_data.py` gains a configurable chassis collision, which rigid-ground traversal scenes set to hull
    collision. Main's version silently ignores the setting, which would change rigid physics.

  To leave the published code untouched, the plan is to keep traversal-specific copies of these two pieces inside the
  traversal package rather than editing the shared files. The other three shared files the branch modified are in no
  headline closure and will not be carried over.

[commit]: https://github.com/uwsbel/NeDM/tree/901d6c9423a16c0fafc3d60056065415d5a725f2
[nav-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/fdm_f104_50h_20260909/nav_v1/REPORT.md
[nav-plan]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/fdm_f104_50h_20260909/nav_v1/PLAN.md
[nav-running]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/fdm_f104_50h_20260909/nav_v1/RUNNING.md
[nav-tasks]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/fdm_f104_50h_20260909/nav_v1/local_luffy/tasks_main.json
[bench]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/fdm_f104_50h_20260909/nav_v1/bench_cuda.json
[dabs]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/matched/matched_Dabs.json
[labels-recon]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/crm_f104_v1/scout/labels_dataset_training.md
[nav_runner]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/nav_runner.py
[nav_online]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/nav_online.py
[nav_local_batch]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/nav_local_batch.py
[nav_analyze]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/nav_analyze.py
[sensor_map_v2]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/sensor_map_v2.py
[vehicle_corridor]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/vehicle_corridor.py
[sensor_dataset_v2]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/sensor_dataset_v2.py
[sensor_train_v2]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/sensor_train_v2.py
[gen-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/REPORT.md
[gen-plan]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/PLAN.md
[ci-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/crm_improve_20260922/REPORT.md
[ci-plan]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/crm_improve_20260922/PLAN.md
[suite]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/A_adapt/suite/suite.json
[a3-crm]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/A_adapt/a3/results_crm_A0A3.json
[a3-rigid]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/A_adapt/a3/results_rigid_A0A3.json
[a5-crm]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/A_adapt/a5/results_crm_A5.json
[a5-rigid]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/A_adapt/a5/results_rigid_A5.json
[s2-txt]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/crm_improve_20260922/s2/results_s2_crm_vs3s.txt
[h-deploy]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/A_adapt/train/deploy_v1/H_deploy.json
[ga_build_mixed]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ga_build_mixed.py
[ci_short_anchors]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ci_short_anchors.py
[ga_train]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ga_train.py
[ci_train]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ci_train.py
[ga_suite]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ga_suite.py
[ga_approach]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ga_approach.py
[ci_a5data]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ci_a5data.py
[ga_planner]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ga_planner.py
[ci_planner]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ci_planner.py
[ci_grad]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ci_grad.py
[crm_collect_ext]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/crm_collect_ext.py
[gen_collect_ext]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/gen_collect_ext.py
[ga_analyze]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ga_analyze.py
[run_a3]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/A_adapt/run_A3_deploy.sh
[run_a5]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/A_adapt/a5/run_a5_planning.sh
[run_s2]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/crm_improve_20260922/s2/run_s2_new.sh
[run_grad_picks]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/crm_improve_20260922/s4/run_grad_picks.sh
[run_grad_drives]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/crm_improve_20260922/s4/run_grad_drives.sh
[b0v2-rigid]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/B_tracker/b0/results_rigid_b0v2.json
[b0v2-crm]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/B_tracker/b0/results_crm_b0v2.json
[b0-rigid]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/B_tracker/b0/results_rigid_b0.json
[b0-crm]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/B_tracker/b0/results_crm_b0.json
[track-suite]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/B_tracker/suite/tracking_suite.json
[policy-meta]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/B_tracker/ppo_v2/policy_meta_999.json
[nrd-config]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/B_tracker/nrd_tag_v3/config.json
[nrd-metrics]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/B_tracker/nrd_tag_v3/metrics.json
[run-ppo]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/B_tracker/run_ppo_v2.sh
[gb_build_cache]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/gb_build_cache.py
[gb_crop]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/gb_crop.py
[gb_nrd_common]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/gb_nrd_common.py
[gb_train_nrd]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/gb_train_nrd.py
[gb_tracker_env]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/gb_tracker_env.py
[gb_train_tracker]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/gb_train_tracker.py
[gc_control]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/gc_control.py
[gb_track_tasks]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/gb_track_tasks.py
[gb_track_analyze]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/gb_track_analyze.py
[ag-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/REPORT.md
[ag-plan]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/PLAN.md
[ag-soil]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/RESULTS_soil.md
[ag-rigid]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/RESULTS_rigid.md
[ag-soil-json]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/e6/analysis/results_soil_v1.json
[ag-rigid-json]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/e6/analysis/results_rigid_v1.json
[ag-soil-spec]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/e6/analysis/spec_soil_v1.json
[ag-rigid-spec]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/e6/analysis/spec_rigid_v1.json
[ag-family]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/e6/analysis/family_v1_S2.json
[arena-sel]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/arenas/selection.json
[arena-spread]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/arenas/selection_spread.json
[ag-jobs]: https://github.com/uwsbel/NeDM/tree/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/e5/jobs
[wp7_arenas]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/traverse_wp7_arenas.py
[ag_arena_rank]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ag_arena_rank.py
[ag_spread_select]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ag_spread_select.py
[ag_build_ds]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ag_build_ds.py
[ag_subset]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ag_subset.py
[ag_picks]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ag_picks.py
[ag_eval_tasks]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ag_eval_tasks.py
[ag_eval_index]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ag_eval_index.py
[ag_analyze]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ag_analyze.py
[ag-gator]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/RESULTS_gator_full.md
[ag-bfull]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/e6/analysis/results_soil_v1_Bfull.txt
[ov-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/REPORT.md
[ov-plan]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/PLAN.md
[ov-smoke]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/RESULTS_smoke.md
[ov-collection]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/RESULTS_collection.md
[ov-planner]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/RESULTS_planner.md
[ov-f104]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/e6/analysis/results_ov_v1.txt
[ov-unseen]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/e6/analysis/results_unseen_v1.txt
[ov-smoke-p]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/analysis/smoke_polaris_v1_final.txt
[ov-smoke-m]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/analysis/smoke_m113_v1_final.txt
[ov-notes-m4]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/NOTES_M4.md
[ag_vehicle]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ag_vehicle.py
[ag_crm_collect]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ag_crm_collect.py
[ov_vehicle]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ov_vehicle.py
[ov_crm_collect]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ov_crm_collect.py
[ov_m113]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ov_m113.py
[ov_grad_picks]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ov_grad_picks.py
[ov_analyze]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ov_analyze.py
[ov_unseen_analyze]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ov_unseen_analyze.py
[ov_smoke_analyze]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/ov_smoke_analyze.py
