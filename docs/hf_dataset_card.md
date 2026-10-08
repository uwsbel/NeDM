---
license: bsd-3-clause
pretty_name: NeDM Neural Reduced Dynamics Datasets
tags:
  - robotics
  - vehicle-dynamics
  - project-chrono
  - simulation
  - time-series
  - reduced-order-model
  - reinforcement-learning
  - terramechanics
task_categories:
  - time-series-forecasting
  - robotics
size_categories:
  - 100M<n<1B
configs:
  - config_name: hmmwv_flat
    default: true
    data_files:
      - split: train
        path: raw/hmmwv_flat/train/*.parquet
      - split: val
        path: raw/hmmwv_flat/val/*.parquet
  - config_name: hmmwv_bumpy
    data_files:
      - split: train
        path: raw/hmmwv_bumpy/train/*.parquet
      - split: val
        path: raw/hmmwv_bumpy/val/*.parquet
  - config_name: hmmwv_crm
    data_files:
      - split: train
        path: raw/hmmwv_crm/train/*.parquet
      - split: val
        path: raw/hmmwv_crm/val/*.parquet
  - config_name: arm
    data_files:
      - split: train
        path: raw/arm/train/*.parquet
      - split: val
        path: raw/arm/val/*.parquet
  - config_name: tracked
    data_files:
      - split: train
        path: raw/tracked/train/*.parquet
      - split: val
        path: raw/tracked/val/*.parquet
  - config_name: hmmwv_flat_episodes
    data_files: raw/hmmwv_flat/episodes.parquet
  - config_name: hmmwv_bumpy_episodes
    data_files: raw/hmmwv_bumpy/episodes.parquet
  - config_name: hmmwv_crm_episodes
    data_files: raw/hmmwv_crm/episodes.parquet
  - config_name: arm_episodes
    data_files: raw/arm/episodes.parquet
  - config_name: tracked_episodes
    data_files: raw/tracked/episodes.parquet
---

# NeDM datasets: the paper, the traversing study and the contact NRD

This repository holds three separate releases. **Part A** is the set of datasets behind the NeDM paper, unchanged since
its release (Hub tag `paper-v1`). **Part B**, everything under `traversing/`, holds the drives, training files, models
and evaluation records of the traversing study, follow-on work that is not part of the paper. **Part C**, everything
under `contact_nrd/`, holds the training data, test sets and models of the contact NRD, also follow-on work.

Contents:

- [Part A: datasets of the paper](#part-a-datasets-of-the-paper): `raw/`, `processed/`, `assets/` and
  `release_manifest.json`
- [Part B: traversing study](#part-b-traversing-study): `traversing/`
- [Part C: contact NRD](#part-c-contact-nrd): `contact_nrd/`

## Part A: datasets of the paper

### NeDM — Neural Reduced Dynamics Datasets

High-fidelity [Project Chrono](https://projectchrono.org) trajectories used to train the
neural reduced dynamics models (NN-ROMs) in

> **Learning the Right Abstraction: Neural Reduced Dynamics for Complex Robot Control**
> Harry Zhang and Dan Negrut, 2026 (preprint).
> Project page: <https://uwsbel.github.io/NeDM/> · Code: <https://github.com/uwsbel/NeDM>

Every dataset here is exactly what the paper's models were trained and validated on. Two tiers
are published (70 GB in total):

* **`raw/`** — every recorded channel of every episode (Parquet, float32), plus a per-episode
  index and the byte-exact collection metadata (driver profiles, seeds, terrain, termination
  causes). This is the reusable resource: build your own reduced states from it.
* **`processed/`** — the four training caches the deployed models read (`.npy`), so the
  paper's training configs run without touching the raw data.

### Datasets

| Config | System | Terrain / task | Rate | Episodes (train / val) | Rows | Raw Parquet | Columns |
|---|---|---|---|---|---|---|---|
| `hmmwv_flat` | HMMWV (`HMMWV_Full`, TMEASY tires, SMC contact) | flat rigid, μ = 0.9, 900 × 900 m | 100 Hz | 32,768 (26,124 / 6,644) | 160,551,861 | 44.0 GB, 128 shards | 105 |
| `hmmwv_bumpy` | HMMWV (same vehicle) | rigid heightmap, 100 random 500 × 500 m fields, ±0.6 m | 100 Hz | 1,360 (1,104 / 256) | 4,511,778 | 1.3 GB, 4 shards | 105 |
| `hmmwv_crm` | HMMWV (rigid-mesh tires) | CRM deformable soil (SPH), 150 × 150 × 0.25 m | 100 Hz | 2,000 (1,582 / 418) | 2,884,961 | 0.8 GB, 4 parts | 105 |
| `arm` | 4-DOF LRV arm mounted on an M113 (base held) | free-space joint motion, PD torque control | 50 Hz | 15,000 (12,716 / 2,284) | 920,640 | 0.09 GB, 15 shards | 47 |
| `tracked` | M113 tracked vehicle, arm welded at home | flat rigid drive, 10 manoeuvre families | 50 Hz | 2,160 (1,808 / 352) | 1,683,484 | 0.2 GB, 60 shards | 42 |

Roles in the paper: `hmmwv_flat` + `hmmwv_crm` train the terrain-conditioned HMMWV NN-ROM
(Study Case I); `hmmwv_bumpy` is the zero-shot out-of-distribution test regime and never enters
training, model selection, normalisation or reward tuning; `tracked` and `arm` train the two
Study Case II NN-ROMs. All five were collected with PyChrono 10.0.0 (conda `projectchrono`
channel) using the collectors in the code repository (`src/nedm/hmmwv_data.py`,
`scripts/collection/collect_hmmwv_crm_dataset.py`, `src/nedm/arm_data.py`,
`src/nedm/tracked_vehicle_data.py`).

#### Splits

Train/val is decided **per episode at collection time** and stored in the `split` column:
`sha1(episode_id)[:8] / 0xFFFFFFFF < validation_ratio → val` (ratio 0.20 for the HMMWV sets,
0.15 for `arm` and `tracked`). Whole episodes stay together; the assignment depends only on the
episode id, so it is stable under re-sharding. `train` and `val` files never share an episode.

### Layout

```
raw/<config>/train/<shard>.parquet     transitions, one file per raw collection shard
raw/<config>/val/<shard>.parquet
raw/<config>/episodes.parquet          one row per episode: index entry + JSON sidecar (see below)
raw/<config>/metadata.tar.gz           byte-exact originals: dataset_index.json, collector_config.resolved.json,
                                       episodes/<id>.json sidecars, shard-plan manifests
processed/<cache>/                     .npy training caches + metadata.json (state layout, normalisation)
assets/bumpy_terrain/bumpy_field_NNN.bmp   the 100 heightmaps behind hmmwv_bumpy (256×256, 8-bit, gray 128 = 0 m)
release_manifest.json                  sha256 / size / row count of every file, tool versions, source commit
```

Rows are ordered by episode (collection order) then `sample_index`; each episode is contiguous
inside exactly one file. Column names and order are the collector's CSV columns, unchanged. All
physical channels are `float32` (`time_s` is `float64`; `sample_index`, `collision` are `int32`;
identifiers are dictionary-encoded strings). Files are zstd-compressed with BYTE_STREAM_SPLIT
float encoding and ≤ 262,144-row row groups.

#### Column groups

**HMMWV (`hmmwv_flat`, `hmmwv_bumpy`, `hmmwv_crm` — identical 105 columns).** Units are in the
names (`_m`, `_mps`, `_mps2`, `_rad`, `_radps`, `_n`, `_nm`); world frame is Chrono's ISO
(x forward, z up), body frame is the chassis frame.

| Group | Columns |
|---|---|
| identifiers | `episode_id`, `scenario_name`, `scenario_family`, `split`, `sample_index`, `time_s` |
| driver command (the action) | `driver_steering` ∈ [−1, 1], `driver_throttle` ∈ [0, 1], `driver_braking` ∈ [0, 1] |
| chassis pose | `pos_{x,y,z}_m`, `quat_e0..e3`, `roll_rad`, `pitch_rad`, `yaw_rad` |
| chassis motion | `vel_world_{x,y,z}_mps`, `vel_body_{x,y,z}_mps`, `acc_world_*`, `acc_body_*`, `ang_vel_world_{x,y,z}_radps`, `ang_vel_body_{x,y,z}_radps`, `speed_mps`, `body_slip_rad`, `roll_rate_radps`, `yaw_rate_radps` |
| per-tire block, prefix `tire_{fl,fr,rl,rr}_` (16 × 4) | `longitudinal_slip`, `slip_angle_rad`, `camber_angle_rad`, `force_world_{x,y,z}_n`, `moment_world_{x,y,z}_nm`, `force_wheel_{fx,fy,fz}_n`, `spindle_omega_radps`, `wheel_vx_mps`, `slip_ratio`, `deflection_m` |

`force_wheel_*` and `slip_ratio` are derived from spindle state and the world-frame force so they
are computed identically on rigid and CRM terrain (on CRM the tire force comes from the FSI
solver, `tire_force_source: crm_fsi`). The paper's 15-D HMMWV state is
`vel_body_x_mps, vel_body_y_mps, roll_rad, pitch_rad, roll_rate_radps, ang_vel_body_y_radps, yaw_rate_radps`
+ `tire_*_force_wheel_fz_n` (4) + `tire_*_spindle_omega_radps` (4); action is the driver triple;
pose for open-loop rollout scoring is `pos_x_m, pos_y_m, yaw_rad`. Recording starts after a
settle/warm-up window (`warmup_s` 2.5 s rigid, 0.2 s CRM), so `time_s` does not start at 0.

**Arm (`arm`, 47 columns).** Each row is one 50 Hz control step written as a transition
`(s, a, s')`: `q_0..3`, `qd_0..3` (joint angle rad / rate rad/s), `qcmd_0..3` (current joint
command), `act_0..3` (Δq_cmd), `qcmd_next_0..3` (command applied over this step — the paper's
action), `q_next_0..3`, `qd_next_0..3`, end-effector position in world (`ee_{x,y,z}`,
`ee_next_*`) and in the vehicle base frame (`ee_base_{x,y,z}`, `ee_next_base_*`), plus
`collision` (0/1), `collision_kind` (`ground` / `track` / `joint_limit` / empty), `contact_force_n`.
Episodes start from the home pose with random command increments and terminate on the first
contact or joint-limit hit, so lengths are 9–500 steps (mean ≈ 58). The paper's 8-D state is
`[q, qd]` with the end effector recovered by forward kinematics.

**Tracked (`tracked`, 42 columns).** The HMMWV chassis block without `body_slip_rad` and without
tire channels, plus `left_sprocket_speed_radps`, `right_sprocket_speed_radps`. The paper's 3-D
state is `vel_body_x_mps, vel_body_y_mps, yaw_rate_radps`; action is the driver triple.

#### `episodes.parquet` and the metadata bundle

`episodes.parquet` flattens each episode's `dataset_index.json` entry and its JSON sidecar
(nested values are JSON strings): `episode_id`, `split`, `scenario_family`, `rows`,
`duration_s`, `warmup_s`, `source_shard`, `parquet_file`, and per dataset e.g.
`height_map_index` / `height_map` / `terminated_out_of_bounds` (bumpy), `terminated_near_boundary`,
`crm_particles`, `crm_force_summary`, full `driver` profile (CRM), `collision_kind`,
`collision_links`, `start_q` (arm), `diverged` (tracked), `tire_nominal_radius_m`.

`metadata.tar.gz` is the untouched original metadata: per shard `dataset_index.json` and
`collector_config.resolved.json` (every materialised scenario: driver profile, seed, family,
terrain and solver settings), every per-episode sidecar, and the shard-plan manifests. It is what
lets the release be turned back into the collectors' original directory tree (below).

### Loading

**Streaming with 🤗 `datasets`** (no download of the 44 GB flat set required):

```python
from datasets import load_dataset
ds = load_dataset("harryzhang1018/NeDM", "hmmwv_crm", split="val", streaming=True)
for row in ds.take(3):
    print(row["episode_id"], row["time_s"], row["vel_body_x_mps"], row["tire_fl_force_wheel_fz_n"])
episodes = load_dataset("harryzhang1018/NeDM", "hmmwv_bumpy_episodes", split="train")
```

**Arrow / DuckDB** — one shard at a time, with row-group statistics for pushdown:

```python
import pyarrow.parquet as pq
t = pq.read_table("raw/hmmwv_flat/train/shard_017.parquet",
                  columns=["episode_id", "time_s", "vel_body_x_mps", "yaw_rate_radps"],
                  filters=[("scenario_family", "==", "chirp_steer")])
```

**Reproducing the paper** with the code repository (`conda env create -f environment.nedm.yml`):

```bash
# training caches -> artifacts/training_datasets/, then any config in configs/ runs verbatim
PYTHONPATH=src python scripts/release/download_nedm_datasets.py --dataset all --no-raw --processed
PYTHONPATH=src python scripts/training/train_hmmwv_dynamics.py --config configs/tracked_transformer_v1.json

# raw Parquet -> the collectors' original per-episode CSV tree under artifacts/datasets/,
# so scripts/preprocess/* and the RL reference builders run unchanged
PYTHONPATH=src python scripts/release/download_nedm_datasets.py --dataset arm --rehydrate
```

The rehydrated CSVs carry the float32 values the trainer uses; caches rebuilt from them are
bit-identical to the ones in `processed/` (this is checked in the release validation).

### Processed caches

| Cache | Trained model | State | Action | Transitions (train / val) | Size |
|---|---|---|---|---|---|
| `hmmwv_tire_rigid_300g_normal_force_omega_seq_v1` | terrain-conditioned HMMWV NN-ROM (flat share) | 15-D | 3-D | 128,043,338 / 32,475,755 | 23.1 GB |
| `hmmwv_crm_2000_normal_force_omega_seq_v1` | terrain-conditioned HMMWV NN-ROM (CRM share) | 15-D | 3-D | 2,280,431 / 602,530 | 0.4 GB |
| `arm_dyn_v3_8d_seq16_v1` | arm NN-ROM | 8-D `[q, q̇]` | 4-D `q_cmd` | 763,886 / 141,754 | 87 MB |
| `tracked_drive_v2_seq16_v1` | tracked-base NN-ROM | 3-D `[vx, vy, r]` | 3-D | 1,407,465 / 273,859 | 81 MB |

Each cache holds contiguous `float32` arrays `{train,val}_{states,actions,targets,rollout}.npy`
(`targets = states[t+1] − states[t]`, `rollout` = pose per recorded row), `episode_starts` /
`episode_lengths`, `{train,val}_episodes.json` (episode ids and provenance) and `metadata.json`
(`state_fields`, `action_fields`, `dt_s`, train-split mean/std used for normalisation). Values are
raw physical units; the model applies the statistics.

### Known limitations

* `hmmwv_bumpy` episodes are short (mean 3.3 k rows) because 78 % end on the 0.9 × 500 m
  keep-in guard; the regime is meant as a test set.
* The arm collection is restricted to free-space motion (episodes end at first contact) and
  under-samples the lower/rear workspace.
* CRM episodes are 12–18 s long (SPH cost) and use rigid-mesh tires; the CRM tire "force" is the
  fluid–solid interaction force.
* Simulation is deterministic and noise-free; there is no sensor model.

### Citation

```bibtex
@article{zhang2026abstraction,
  title   = {Learning the Right Abstraction: Neural Reduced Dynamics for Complex Robot Control},
  author  = {Zhang, Harry and Negrut, Dan},
  journal = {Preprint},
  year    = {2026}
}
```

License: BSD-3-Clause (same as the code). Simulation assets are Project Chrono's HMMWV and M113
models; the LRV arm geometry is in the code repository (`src/arm_model/`).

## Part B: traversing study

### What the study is

Follow-on work to the paper, not part of it. A vehicle in [Project Chrono](https://projectchrono.org) has to reach a
goal across generated hill-and-crater terrain, on rigid ground and on CRM deformable soil. A neural network trained on
thousands of recorded Chrono drives predicts, for a candidate route and speed profile, where along the route the
vehicle is likely to fail: roll back on a climb, stall, dig into the soil or tip over. A route search picks the route
with the lowest predicted risk, and Chrono's stock PID path follower drives it. Separately, a route tracker is trained
with PPO inside a learned neural reduced dynamics model (NRD) and compared with that PID follower. The milestones, how
they were measured and what they do not show are documented in
[`traversing/` of the code repository](https://github.com/uwsbel/NeDM/tree/main/traversing). This part holds what
reproduces those results: the recorded drives, the exact training files built from them, every trained model behind a
result-table column, and the evaluation inputs and per-drive records. Every file is a byte-exact copy of the study's
record, or a deterministic archive of such copies; nothing was re-encoded.

Terms used below:

- **Milestones** (the `milestone` labels of each item):
  - `m1`: a Chrono depth camera in the planning loop (live multi-waypoint missions, rigid ground);
  - `m2`: one risk model for rigid ground and soil, not told which one it is on (HMMWV on f104);
  - `m3`: a learned route tracker against Chrono's PID follower;
  - `m4a`: the HMMWV planner on arenas it never trained on;
  - `m4b`: the pipeline on other vehicles (Gator, Polaris; the M113 only in a smoke test).
- **Arenas.** f104 is the 80 m x 80 m hill-and-crater arena where most training drives were recorded. g203, g217 and
  g228 are sibling arenas from the same terrain generator: the depth-camera model of m1 trained on all three (g216 and
  g231 were its held-out arenas), the m4a planners on g203 and g228 (g217 was their development arena). The unseen
  arenas are eight more made with new seeds and never used in training.
- **Ground.** Rigid means a rigid heightmap. Soil means Chrono's CRM particle soil: a 0.24 m layer over a rigid floor,
  one setting throughout.
- **Groups and route ids.** A start/goal pair is a group. Each group has several routes: designed routes (sideways
  offsets x speed profiles, ids ending `_route_NN`) and planner-style routes (`_op_NN`), for example
  `f104_v2_group_0000_route_03`. The soil collections and the other vehicles reuse these ids; the Gator and Polaris
  drive folders carry a `gator__` or `polaris__` prefix.
- **Drive.** One Chrono run of one route with the stock PID path follower, after a 0.8 s braked settle, recorded every
  50 ms. It stops at the goal (2.5 m radius), on rollover, on leaving the terrain, on prolonged blockage, on soil when a
  wheel digs through the whole layer, or at the 120 s horizon.

### Layout

```
traversing/README.md                   short index: every item with bundle, milestone, packing, files, bytes, restore path
traversing/release_manifest.json       every file (bytes, sha256, item, kind) and every item (bundle, milestone, ...)
traversing/models/<item>/...           trained networks with their training records
traversing/evaluation/<item>/...       evaluation suites, locked route picks, task lists, per-drive records and results
traversing/processed/<item>/...        the exact training files the models were fitted on
traversing/raw/<item>/...              the recorded Chrono drives the training files were built from
traversing/assets/<item>/...           arena heightmaps, the f104 terrain grid, the soil setting and vehicle variants
```

Each item is packed in one of two ways:

- **`files`.** Every file is uploaded as is, at `traversing/<bundle>/<item>/<path>`, where `<path>` is its path under
  the item's source folder. Used for models, single large training files and small sets of evaluation files.
- **`tar`.** Folders of many small files (mostly drive folders) go into `part-00000.tar.gz`, `part-00001.tar.gz`, ...
  of about 1 GiB of uncompressed files each; a drive folder never spans two shards. Each member is stored under its
  restore path relative to the code repository's root, so extracting a shard at the root of a checkout puts it in
  place. The archives are deterministic (members sorted by path, time stamps 0, owner 0, mode 0644, regular files
  only, PAX format, gzip level 6 without name or time). Next to the shards:
  - `index.csv.gz`: one row per member, with columns `path, bytes, sha256, shard`;
  - `ids.txt.gz`: the list of drive folders, for items selected by such a list;
  - `episodes.csv.gz`: raw collections only, one row per drive (see below).

`release_manifest.json` is the reference for everything under `traversing/`:

- `source_commit`: the experiment commit the files come from,
  [`901d6c9`](https://github.com/uwsbel/NeDM/tree/901d6c9423a16c0fafc3d60056065415d5a725f2);
- `items`: per item its bundle, milestones, description, `needed_for` (`rerun`, `recount` or `retrain`), packing,
  `restore_root`, member count and bytes, Hub files, parameter counts of its checkpoints (`known_params`), and the other
  items it needs (`requires`);
- `files`: per Hub file its bytes, SHA256, item and kind (`file`, `tar_shard`, `index` or `episodes`). Shards also
  record `tar_bytes` and `tar_sha256` of the uncompressed stream, since the compressed bytes depend on the zlib
  version;
- `path_remap`: how paths written inside released task files map to restore paths, and the symbolic links the
  planners expect (the release holds no links; `ln -s` recreates them from this list).

The code repository keeps a copy pinned to a Hub commit, `traversing/manifests/hf_release_manifest.json`, whose
`hf_revision` is `6620faead5225ac9aa5ae8ab19bc2ef2db38a863`.

### Items

<!-- BEGIN fill_card.py: items (generated from traversing/release_manifest.json; do not edit by hand) -->

**119 items, 732 files, 56.6 GB** (models 30 items, 206 MB; evaluation 33 items, 5.8 GB; processed 28 items, 33.9 GB; raw 25 items, 16.8 GB; assets 3 items, 8.1 MB). The raw collections with an episodes table hold 189,335 drives. Sizes are Hub bytes (compressed for tar shards); `release_manifest.json` has every file's exact size and SHA256. The descriptions are the manifest's: they name the column of the result tables in `traversing/results/` that each model or record stands behind, and the study's own file and model names.

#### Models

| Item | Milestone | What it is | Files | Size |
|---|---|---|---|---|
| `m1_nav_direct_depth_risk_ensemble` | m1 | Navigation risk ensemble: 3 seeds (s0-s2) of the direct-depth corridor model (channels range_abs, sec1, speed, valid) that scored every candidate route in all 120 navigation missions (all 4 decision-timing arms). | 4 | 3.1 MB |
| `m2_joint_station_history_transformer` | m2 | Joint station+history transformer trained on the same rows (deploy_a3_haux_txjoint, 5 seeds s0-s4) behind transformer_1s, transformer_0p5s and transformer_0p5s_grad (soil). | 6 | 8.4 MB |
| `m2_oracle_world_label_model` | m2 | Shared model given the true world label (T, 5 seeds s0-s4) behind oracle_tag_standing, oracle_tag_3s, oracle_tag_1s, oracle_tag_0p5s. | 6 | 5.3 MB |
| `m2_pooled_no_label_no_history_model` | m2 | Shared model with neither history nor label (P, 5 seeds s0-s4) behind pooled_3s, pooled_1s, pooled_0p5s. | 6 | 5.3 MB |
| `m2_rigid_only_planner_standing_start` | m2 | Earlier deployed rigid-only planner (night-2 N2, 5 seeds s0-s4) behind column specialist_rigid_standing (study arm Srigid, sampling arm B) in both m2 tables. | 6 | 5.2 MB |
| `m2_rigid_specialist_same_rows` | m2 | Single-world rigid model trained on the same re-anchored rows (Sp_rigid, 5 seeds s0-s4) behind specialist_rigid_3s. | 6 | 5.3 MB |
| `m2_shared_history_early_rows_final_model` | m2 | Model of the FINAL label-free configuration: the history design retrained with extra early-decision and moving-branch rows (deploy_a1_haux_gru, 5 seeds s0-s4). Behind shared_hist_early_rows_1s, shared_hist_early_rows_0p5s (soil and rigid) and shared_hist_early_rows_0p5s_grad (the 780/800 soil and 800/800 rigid headline). | 6 | 6.2 MB |
| `m2_shared_history_model` | m2 | Shared rigid/soil history model H (CNN-GRU + 2 s history encoder, 5 seeds s0-s4) behind shared_hist_standing, shared_hist_3s, shared_masked_3s (history masked at decision), shared_hist_1s and shared_hist_0p5s. | 6 | 5.5 MB |
| `m2_soil_only_planner_standing_start` | m2 | Earlier deployed soil-only planner (CRM_N2, 5 seeds s0-s4) behind column specialist_soil_standing (study arm Scrm, sampling arm B) in both m2 tables. | 6 | 5.2 MB |
| `m2_soil_specialist_same_rows` | m2 | Single-world soil model trained on the same re-anchored rows (Sp_crm, 5 seeds s0-s4) behind specialist_soil_3s and specialist_soil_1s. | 6 | 5.3 MB |
| `m3_tracker_learned_dynamics_model` | m3 | Tag-conditioned NRD vehicle-dynamics model (causal transformer, 17-D state, 50 ms steps, 8x8 terrain crop) inside which the round-2 PPO policy was trained; not used when driving. | 2 | 19.7 MB |
| `m3_tracker_round2_numpy_actor` | m3 | Deployed round-2 learned tracker: the PPO actor exported to NumPy (158-D observation, 3 actions), run without torch or the NRD model; behind column nrd_policy_v2 of m3_tracker_routes.csv. | 2 | 2.0 MB |
| `m3_tracker_round2_ppo_checkpoint` | m3 | Final PPO torch checkpoint (iteration 999) of the round-2 tracker, the source of the NumPy actor. Not needed to re-drive (the actor is enough) and not a training input, but it is the only artefact with which the actor export and parity check can be repeated. | 3 | 5.9 MB |
| `m4a_rigid_f104_only_ens1` | m4a | CNN-GRU route-risk ensemble, 5 seeds (s0-s4), M1a_rigid_deploy, behind columns f104_only_ens1_fixed2mps and _speedfree (rigid, M1a). The same bytes are aliased as H, the HMMWV-trained rigid model of the Gator study. | 8 | 7.8 MB |
| `m4a_rigid_f104_only_ens2` | m4a | CNN-GRU route-risk ensemble, 5 seeds (s5-s9), M1b_rigid_deploy, behind columns f104_only_ens2_fixed2mps and _speedfree (rigid, M1b). | 8 | 7.8 MB |
| `m4a_rigid_three_arenas_all_data` | m4a | CNN-GRU route-risk ensemble, 5 seeds (s0-s4), A3_rigid_deploy, behind columns three_arenas_all_data_fixed2mps and _speedfree (rigid, A3). | 8 | 12.5 MB |
| `m4a_rigid_three_arenas_same_total_ens1` | m4a | CNN-GRU route-risk ensemble, 5 seeds (s0-s4), M3a_rigid_deploy, behind columns three_arenas_same_total_ens1_fixed2mps and _speedfree (rigid, M3a). | 8 | 8.0 MB |
| `m4a_rigid_three_arenas_same_total_ens2` | m4a | CNN-GRU route-risk ensemble, 5 seeds (s5-s9), M3b_rigid_deploy, behind columns three_arenas_same_total_ens2_fixed2mps and _speedfree (rigid, M3b). | 8 | 8.0 MB |
| `m4a_rigid_two_arenas_same_total` | m4a | CNN-GRU route-risk ensemble, 5 seeds (s0-s4), M2_rigid_deploy, behind columns two_arenas_same_total_fixed2mps and _speedfree (rigid, M2). | 8 | 7.9 MB |
| `m4a_soil_f104_only_ens1` | m4a | CNN-GRU route-risk ensemble, 5 seeds (s0-s4), M1a_soil_deploy, behind column f104_only_ens1 (soil table, study M1a). The same bytes are aliased as H_soil, the Gator study's stage-1 HMMWV soil model. | 8 | 6.3 MB |
| `m4a_soil_f104_only_ens2` | m4a | CNN-GRU route-risk ensemble, 5 seeds (s5-s9), M1b_soil_deploy, behind column f104_only_ens2 (soil table, study M1b). | 8 | 6.3 MB |
| `m4a_soil_three_arenas_all_data` | m4a | CNN-GRU route-risk ensemble, 5 seeds (s0-s4), A3_soil_deploy, behind column three_arenas_all_data (soil, A3). | 8 | 7.1 MB |
| `m4a_soil_three_arenas_same_total_ens1` | m4a | CNN-GRU route-risk ensemble, 5 seeds (s0-s4), M3a_soil_deploy, behind column three_arenas_same_total_ens1 (soil, M3a). | 8 | 6.3 MB |
| `m4a_soil_three_arenas_same_total_ens2` | m4a | CNN-GRU route-risk ensemble, 5 seeds (s5-s9), M3b_soil_deploy, behind column three_arenas_same_total_ens2 (soil, M3b). | 8 | 6.3 MB |
| `m4a_soil_two_arenas_same_total` | m4a | CNN-GRU route-risk ensemble, 5 seeds (s0-s4), M2_soil_deploy, behind column two_arenas_same_total (soil, M2). | 8 | 6.3 MB |
| `m4b_gator_rigid_own_model` | m4b | Gator-trained rigid ensemble on the Gator's 24,000 f104 rigid drives (CNN-GRU, 5 seeds s0-s4, G_rigid_deploy). Behind the README bar-table Gator rigid row (800/800 on f104; arms G_free at free speed and the G_fixed2 addendum); not a compact-table column. | 8 | 7.6 MB |
| `m4b_gator_soil_own_model_full_data` | m4b | Gator-trained soil ensemble on all 15,235 Gator f104 soil drives (CNN-GRU, 5 seeds s0-s4, G_full_soil_deploy), behind gator_own_model_sampling (Gfull_free_gator) and gator_own_model_sampling_grad (Gfull_grad_gator). | 8 | 6.8 MB |
| `m4b_gator_soil_own_model_tiers0to6` | m4b | Gator-trained soil ensemble on only the first 7 routes of each group (8,399 drives; CNN-GRU, 5 seeds s0-s4, G_soil_deploy), behind gator_own_model_tiers0to6_sampling (G_free_gator). | 8 | 6.2 MB |
| `m4b_hmmwv_soil_model_full_data` | m4b | HMMWV-trained soil ensemble on the full HMMWV f104 soil file, the same ids as the Gator (CNN-GRU, 5 seeds s0-s4, H_full_soil_deploy). Behind gator_hmmwv_model_sampling (Hfull_free_gator), hmmwv_own_model_sampling (Hfull_free) and hmmwv_own_model_sampling_grad (Hfull_grad_hmmwv). | 8 | 6.9 MB |
| `m4b_polaris_soil_own_model_full_data` | m4b | Final Polaris soil ensemble on all 15,229 valid Polaris f104 soil drives (CNN-GRU, 5 seeds s0-s4, polaris_full_soil_deploy). Behind polaris_own_model_sampling_grad and polaris_own_model_sampling on f104 and on the 8 unseen arenas; polaris_corrected_driveline_grad_routes re-drives the polaris_grad routes. | 7 | 5.4 MB |

#### Evaluation

| Item | Milestone | What it is | Files | Size |
|---|---|---|---|---|
| `nav_corrected_mission_outcomes` | m1 | Corrected navigation read-out after the rescue-route fix: summary.json and the 120 per-mission outcome files (30 missions x 4 decision-timing arms) behind m1_navigation_missions.csv. | 121 | 377 kB |
| `nav_corrected_run_folders` | m1 | Per-mission decisions, candidate routes and trajectory of the 120 corrected navigation drives. | 3 | 33.8 MB |
| `nav_missions_and_task_list` | m1 | The 30 frozen waypoint missions and the 120-row task list (mission x decision-timing arm). | 31 | 67.1 kB |
| `f104_eval_suite_800` | m2, m4a, m4b | The frozen 800-pair f104 start/goal suite: suite definition, lock file, 800 case files, the designed route of each group and the locked standing-start task lists. | 2 | 5.7 MB |
| `planner_map_f104` | m2, m4a, m4b | Whole-arena overhead depth map of f104 on which every f104 planner scores its candidate routes. | 2 | 4.1 MB |
| `shared_model_decision_states` | m2 | Inputs of the moving-start decisions: straight approach routes, the approach drives, and the recorded decision poses and 2 s history arrays at 3 s, 1 s and 0.5 s, with the approach task files. | 2 | 53.5 MB |
| `shared_model_drive_folders_rigid` | m2 | Per-drive folders behind every cell of m2_shared_risk_rigid.csv (9,469 distinct drives for 9,600 cells). | 3 | 435 MB |
| `shared_model_drive_folders_soil` | m2 | Per-drive folders behind every cell of m2_shared_risk_soil.csv (18,027 distinct drives for 18,400 cells). | 4 | 903 MB |
| `shared_model_drive_task_lists` | m2 | Task lists that drove the milestone-2 table arms (standing start, 3 s, 1 s, 0.5 s and final), with the two standing-start lock files; those two locks cover unreleased route copies and cannot be checked from the release. | 14 | 25.8 MB |
| `shared_model_results_early_and_final` | m2 | Per-pair results of the 1 s and 0.5 s early-decision arms and of the final gradient-refined configuration on soil and rigid ground. | 4 | 16.7 MB |
| `shared_model_results_standing_and_3s` | m2 | Per-pair results of the standing-start arms and the 3 s-approach arms, soil and rigid. | 4 | 10.6 MB |
| `shared_model_route_picks_3s` | m2 | Locked planner output of the 3 s-approach arms. | 2 | 45.8 MB |
| `shared_model_route_picks_early` | m2 | Locked planner output of the 1 s and 0.5 s early-decision arms (sampling search only), including the rigid 0.5 s arm. | 2 | 52.7 MB |
| `shared_model_route_picks_final` | m2 | Locked planner output of the final gradient-refined arms after the 0.5 s approach (soil and rigid). | 2 | 38.3 MB |
| `shared_model_route_picks_standing` | m2 | Locked planner output of the standing-start arms: per-pair pick records with candidate scores, route files, task list, run summary and lock file. | 2 | 47.6 MB |
| `f104_reference_routes_and_cases` | m3, m4b | f104 start/goal case files and reference routes that released drive tasks point at: the 423 routes and 55 cases the tracker drives follow, and the 147 routes (88 designed, 59 on-policy) and 27 cases of the vehicle smoke test: the 144 sample-A routes (86 designed, 58 planner proposals) plus the 3 routes of the bit-identity check. | 2 | 1.4 MB |
| `tracker_drive_folders` | m3 | Drive folders of the 2,538 tracker table rows (native PID, held PID and learned tracker on 423 routes, both grounds). | 3 | 141 MB |
| `tracker_results_suite_tasks` | m3 | Per-route tracker results (both grounds), the 423-route tracking suite with strata, and the task lists that drove the PID and learned-tracker arms. | 7 | 6.2 MB |
| `planner_maps_new_arenas` | m4a, m4b | Overhead depth maps of the 11 other planner arenas (g203, g217, g228 and the 8 unseen test arenas) and the map checks. | 23 | 45.2 MB |
| `unseen_and_vehicle_picks_lock_files` | m4a, m4b | Top-level pick lock files and job lists of the unseen-arena, Gator and Polaris evaluations. | 23 | 567 kB |
| `unseen_arena_drive_folders_rigid` | m4a | Drive folders behind the 37,100 cells of m4_unseen_arenas_hmmwv_rigid.csv (36,653 distinct drives). | 5 | 1.9 GB |
| `unseen_arena_drive_folders_soil` | m4a, m4b | Drive folders behind the 9,000 non-empty cells of m4_unseen_arenas_hmmwv_soil.csv (8,746 distinct drives). | 3 | 411 MB |
| `unseen_arena_drive_task_lists` | m4a, m4b | Task lists that drove the HMMWV soil and rigid unseen-arena evaluations (including the Gator rigid f104 drives). | 18 | 31.4 MB |
| `unseen_arena_eval_indexes` | m4a, m4b | Per-drive evaluation indexes of the unseen-arena study (soil and rigid; the rigid index also holds the Gator rigid f104 rows) and the task-to-arm mapping files. | 11 | 80.5 MB |
| `unseen_arena_route_picks` | m4a, m4b | Locked planner picks of every HMMWV arm of the two unseen-arena tables, on the 8 unseen arenas, f104, the held-out and dev arenas (236 arm x arena folders). | 3 | 228 MB |
| `unseen_arena_suites` | m4a, m4b | Frozen start/goal suites of the 8 unseen test arenas, held-out g203/g228 and dev g217: case files, designed routes, per-suite lock files, manifests, the f104 in-distribution list, the declared soil subset and the spread-headroom group list. | 2 | 43.7 MB |
| `gator_rigid_f104_drive_folders` | m4b | Drive folders of the Gator rigid f104 read-out in the README bar table: the Gator-trained planner (800/800) and the straight route at 6 m/s. | 3 | 68.7 MB |
| `polaris_unseen_drive_folders` | m4b | Drive folders behind the 3,000 cells of m4_polaris_unseen_soil.csv (2,618 distinct drives). | 3 | 183 MB |
| `vehicle_drive_task_lists` | m4b | Task lists of the Gator/HMMWV all-data drives, Polaris on f104, the gradient-refined Gator/HMMWV drives, Polaris on the unseen arenas and the smoke test. | 16 | 6.6 MB |
| `vehicle_eval_indexes_and_smoke_extract` | m4b | Per-drive indexes of the Gator all-data comparison, Polaris on f104 and Polaris on the unseen arenas, their task-mapping files, the per-route smoke and collection extract, and the definition of smoke sample A. | 11 | 54.0 MB |
| `vehicle_f104_drive_folders` | m4b | Drive folders behind the 9,600 cells of m4_vehicles_f104_soil.csv (9,243 distinct drives; Gator, HMMWV and Polaris variants). | 3 | 702 MB |
| `vehicle_route_picks` | m4b | Locked picks of the vehicle arms: Gator own models, HMMWV-trained model, gradient refinements, Polaris on f104 and on the 8 unseen arenas, and the Gator rigid planner. | 3 | 86.2 MB |
| `vehicle_smoke_test_drives` | m4b | Complete drive folders of the vehicle smoke test on sample A (144 f104 soil routes) for the 7 variants with their own folders (three Polaris driveline/wheel variants, stock and re-geared M113, Gator at 1 ms and 0.5 ms steps), plus the failure record of the one crashed drive. The polaris_stock and hmmwv_stored columns of m4_vehicle_smoke.csv come from drives released in the raw collections polaris_f104_soil_collection_runs and hmmwv_soil_f104_collection (not required: about 3.7 GB). | 3 | 148 MB |

#### Processed training files

| Item | Milestone | What it is | Files | Size |
|---|---|---|---|---|
| `nav_depth_corridors_arena_g203` | m1 | Navigation corridors on sibling training arena g203. 4,925 rows. | 1 | 366 MB |
| `nav_depth_corridors_arena_g216_heldout` | m1 | Navigation corridors on held-out arena g216 (read by the trainer for its held-out route-choice score, not fitted). 4,956 rows. | 1 | 368 MB |
| `nav_depth_corridors_arena_g217` | m1 | Navigation corridors on sibling training arena g217. 5,008 rows. | 1 | 372 MB |
| `nav_depth_corridors_arena_g228` | m1 | Navigation corridors on sibling training arena g228. 4,937 rows. | 1 | 367 MB |
| `nav_depth_corridors_arena_g231_heldout` | m1 | Navigation corridors on held-out arena g231 (read by the trainer for its held-out score, not fitted). 4,889 rows. | 1 | 363 MB |
| `nav_depth_corridors_f104_extra_drives` | m1 | Navigation corridors from f104 gen_v1 / sensor_v1 test drives (fitted; f104 is a training arena). 3,111 rows. | 1 | 231 MB |
| `nav_depth_corridors_f104_training_routes` | m1 | Direct-depth route-corridor tensors for the navigation risk ensemble (matched_Dabs, 3 seeds): all 36,199 f104 rigid training routes (labels reused from the merged station set via R/sensor_v1/labels_station_ds_all.npz, corridors re-sampled from the back-projected depth grid). 36,199 rows. | 1 | 2.7 GB |
| `rigid_specialist_training_rows` | m2 | Training rows of the earlier rigid-only f104 planner N2 (column specialist_rigid_standing, arm Srigid): merged night-1 + night-2 rigid route set, 36,199 routes / 2,700 groups (train 33,840, val 764, test 1,595; 31,851 fitted per N2_meta.json). | 1 | 1.1 GB |
| `shared_model_baseline_reanchored_rows` | m2 | Re-anchored rigid + soil rows with a 2 s history on the f104 route ids: 115,868 rows (105,193 fitted). Training file of the generalist ensembles H, P, T, Sp_crm, Sp_rigid (columns oracle_tag_*, shared_hist_standing/3s/1s/0p5s, pooled_*, shared_masked_3s, specialist_soil_3s/1s, specialist_rigid_3s). | 1 | 2.1 GB |
| `shared_model_final_anchor_k60_rows` | m2 | Final shared-model training file 3 of 3: decision rows at frame 60 (3 s), the k=60 slice of anchor_k40_60_80.npz (14,972 rigid + 14,727 soil). 29,699 rows, 26,936 fitted. | 1 | 541 MB |
| `shared_model_final_reanchored_plus_branch_rows` | m2 | Final shared-model training file 1 of 3: mixed_reanchor rows plus 4,614 moving-prefix branch rows of both worlds. 120,482 rows, 109,244 fitted. | 1 | 2.2 GB |
| `shared_model_final_short_anchor_rows` | m2 | Final shared-model training file 2 of 3: early decision rows at frames 10/20/30 of every rigid and soil f104 episode. 89,998 rows, 81,634 fitted. | 1 | 1.6 GB |
| `soil_specialist_training_rows` | m2 | Training rows of the earlier soil-only f104 planner CRM_N2 (column specialist_soil_standing, arm Scrm): one row per validated HMMWV soil route (15,235); 13,821 fitted per CRM_N2_deploy.json. | 1 | 471 MB |
| `tracker_dynamics_training_cache` | m3 | Per-episode 50 ms state/action/pose/power cache of 43,235 f104 episodes (rigid 26,500 = 24,000 production + 1,500 perturbed + 1,000 round-1 policy harvest; soil 16,735 = 15,235 collection + 1,500 perturbed) plus cache_manifest.json and build_report.json. | 3 | 2.0 GB |
| `tracker_group_split_file` | m3 | Night-2 soil twin dataset whose group/split arrays define the 1,200-group split (1,089/56/55) used by the cache builder and cross-checked by the NRD trainer. | 1 | 169 MB |
| `terrain_rigid_subset_f104_only` | m4a | HMMWV rigid, f104 only, 1,089 groups. 93,397 rows, 84,787 fitted. Trained M1a and M1b (columns f104_only_ens1/ens2 _fixed2mps/_speedfree); byte-identical to H_f104_hmmwv_rigid.npz (Gator-study H). | 1 | 1.7 GB |
| `terrain_rigid_subset_three_arenas_all` | m4a | HMMWV rigid, all data: f104 1,089 + g203 1,083 + g228 1,063 groups. 278,735 rows, 250,490 fitted. Trained A3 (column three_arenas_all_data). | 1 | 5.0 GB |
| `terrain_rigid_subset_three_arenas_same_total` | m4a | HMMWV rigid, f104 + g203 + g228, same total. 112,726 rows, 84,481 fitted. Trained M3a and M3b (columns three_arenas_same_total_ens1/ens2). | 1 | 2.0 GB |
| `terrain_rigid_subset_two_arenas` | m4a | HMMWV rigid, f104 + g203, same total. 101,916 rows, 84,310 fitted. Trained M2 (column two_arenas_same_total). | 1 | 1.8 GB |
| `terrain_soil_subset_f104_only` | m4a | HMMWV soil, f104 only, 1,089 training groups, tiers 0-6. 32,152 rows, 29,210 fitted. Trained M1a_soil and M1b_soil (columns f104_only_ens1/ens2, soil) and is byte-identical to H_f104_hmmwv_soil.npz (Gator-study H_soil). | 1 | 583 MB |
| `terrain_soil_subset_three_arenas_all` | m4a | HMMWV soil, all data: f104 1,089 + g203 559 + g228 520 groups. 64,400 rows, 57,898 fitted. Trained A3_soil (column three_arenas_all_data). | 1 | 1.2 GB |
| `terrain_soil_subset_three_arenas_same_total` | m4a | HMMWV soil, f104 + g203 + g228, 363 groups each. 35,573 rows, 29,071 fitted. Trained M3a_soil and M3b_soil (columns three_arenas_same_total_ens1/ens2). | 1 | 644 MB |
| `terrain_soil_subset_two_arenas` | m4a | HMMWV soil, f104 + g203, 545 groups each. 33,227 rows, 29,048 fitted. Trained M2_soil (column two_arenas_same_total). | 1 | 603 MB |
| `gator_rigid_subset_all` | m4b | Gator rigid on f104, all 24,000 route ids. 93,551 rows, 84,922 fitted. Trained G, the Gator rigid planner cited in the README bar table (800/800). | 1 | 1.7 GB |
| `gator_soil_subset_all` | m4b | Gator soil on f104, all 15,235 validated ids (tiers 0-12). 55,826 rows, 50,822 fitted. Trained G_full_soil (columns gator_own_model_sampling, gator_own_model_sampling_grad). | 1 | 1.0 GB |
| `gator_soil_subset_tiers0to6` | m4b | Gator soil on f104, tiers 0-6 (8,399 drives). 30,827 rows, 28,057 fitted. Trained G_soil (column gator_own_model_tiers0to6_sampling). | 1 | 557 MB |
| `hmmwv_soil_subset_gator_matched_all` | m4b | HMMWV soil on f104, exactly the route ids the Gator validated (all tiers). 58,268 rows, 52,923 fitted. Trained H_full_soil (columns gator_hmmwv_model_sampling, hmmwv_own_model_sampling, hmmwv_own_model_sampling_grad). | 1 | 1.1 GB |
| `polaris_soil_subset_all` | m4b | Polaris soil on f104, all 15,229 valid ids (tiers 0-12). 57,228 rows, 52,021 fitted. Trained polaris_full_soil (all Polaris planner columns on f104 and the unseen arenas). | 2 | 1.0 GB |

#### Raw collections

| Item | Milestone | What it is | Files | Size |
|---|---|---|---|---|
| `hmmwv_rigid_f104_early_waves` | m1, m2 | Seven small early HMMWV rigid f104 waves whose drives are in the navigation model's 36,199-route f104 label file: prospective_v1 309, fixed_speed_v1 292, speed_c4 206, wide_v1 146, speed_c6 90, band_v1 73, showcase_v1 14. The last six make up the 821 rows the label file tags as MPPI arms. | 4 | 84.9 MB |
| `hmmwv_rigid_f104_first_overnight_collection` | m1, m2 | Earlier HMMWV rigid f104 production collection (1,500 groups, the first overnight wave). It is not part of the 24,000-route pool. It supplies 11,412 of the navigation model's 36,199 f104 routes and part of the rigid N2 specialist's training rows. | 4 | 681 MB |
| `hmmwv_rigid_f104_pool_designed` | m1, m2, m3, m4a | HMMWV rigid-ground pool on f104, designed routes: 1,200 start/goal groups x 12 routes. This is the first half of the 24,000-route rigid pool. Every later rigid f104 model and the tracker cache were built from this pool, and the Gator and Polaris re-drove its route ids. | 4 | 853 MB |
| `hmmwv_rigid_f104_pool_onpolicy` | m1, m2, m3, m4a | HMMWV rigid-ground pool on f104, planner-style (on-policy) routes: 1,200 groups x 8 routes. This is the second half of the 24,000-route rigid pool. | 5 | 1.1 GB |
| `hmmwv_rigid_multi_arena_test_drives` | m1 | HMMWV rigid test drives on f104, g203, g217 and g228 from the multi-arena study (planner, rule-based, fixed-speed and straight-route arms; 1,103 / 1,098 / 1,132 / 1,111). The navigation trainer (sensor_train_v2.py) fits every row from a training arena whatever its source, so these are training rows. | 4 | 250 MB |
| `hmmwv_rigid_sensor_study_test_drives` | m1 | HMMWV rigid drives from the sensor study's two test campaigns (sensor_v1/test 3,614 and test2 4,523) on the training arenas f104, g203, g217 and g228. They enter the navigation training rows through ds_v2_gen_&lt;arena&gt;.npz. | 4 | 408 MB |
| `hmmwv_rigid_sibling_arenas_designed_routes` | m1 | HMMWV rigid designed-route drives on the navigation model's three sibling training arenas g203, g217 and g228 (1,800 each, 5,400 total). The g216 and g231 drives in the same folder are held-out evaluation only and are excluded. | 4 | 330 MB |
| `hmmwv_rigid_f104_branch_anchor_records` | m2 | Per-anchor records of the HMMWV rigid f104 branch drives (&lt;episode&gt;\_\_a4, 800 anchor states): the automatic branch choice (branch_auto.json, 789) or the skip record (skipped.json, 11) and the completion marker. The drives themselves are hmmwv_rigid_f104_branch_continuations. | 3 | 519 kB |
| `hmmwv_rigid_f104_branch_continuations` | m2 | HMMWV rigid drives that branch from a moving state on f104: 2,367 continuation drives (&lt;anchor&gt;\_\_a4\_\_c&lt;j&gt;; 2,358 produce rows, 9 were skipped) from 800 anchor states. The per-anchor records are hmmwv_rigid_f104_branch_anchor_records. | 4 | 251 MB |
| `hmmwv_soil_f104_branch_continuations` | m2 | HMMWV soil drives that branch from a moving state on f104: 752 anchor states x 3 = 2,256 continuation drives (&lt;episode&gt;\_\_c&lt;j&gt;). | 4 | 246 MB |
| `hmmwv_soil_f104_collection` | m2, m3, m4a, m4b | HMMWV soil collection on f104 (Chrono's particle-based deformable soil, 0.24 m deep over a rigid floor): 15,235 validated route ids in 1,200 groups, tiers 0-12 (9,168 designed + 6,067 planner-style). The Gator and Polaris later re-drove the same ids. | 5 | 1.4 GB |
| `hmmwv_rigid_tracker_held_control_drives` | m3 | HMMWV rigid drives with perturbed controls on f104 training routes: controls held for a true 50 ms with brake taps added (1,500 drives, &lt;route&gt;\_\_b3). | 4 | 100 MB |
| `hmmwv_rigid_tracker_policy_harvest` | m3 | HMMWV rigid drives by the round-1 learned tracker on 1,000 training routes (&lt;route&gt;\_\_harvest). They were added to the dynamics cache. | 4 | 68.9 MB |
| `hmmwv_soil_tracker_held_control_drives` | m3 | HMMWV soil drives with the same held-control perturbations on f104 training routes (1,500 drives, &lt;route&gt;\_\_b3). | 4 | 130 MB |
| `arena_gator_collection_task_files` | m4a, m4b | Task files passed to the collection builds of the arena and vehicle studies: the Gator rigid pool (rigid_v2), the Gator soil builds (soil_v3 for tiers 0-12, soil_v2 for tiers 0-6, which also defines the HMMWV g203/g228 soil tiers) and the HMMWV g203/g228 rigid pools (rigid_hmmwv_v1). | 4 | 103 MB |
| `hmmwv_rigid_g203_pool` | m4a | HMMWV rigid 24,000-route pool on the extra training arena g203 (1,200 groups x 20 routes; 14,400 route_ ids + 9,600 op_ ids). ag_build_ds.py selected all 24,000 (92,225 rows). | 6 | 2.0 GB |
| `hmmwv_rigid_g228_pool` | m4a | HMMWV rigid 24,000-route pool on the extra training arena g228 (93,113 rows). | 6 | 2.0 GB |
| `hmmwv_soil_g203_tiers0to6` | m4a | HMMWV soil drives on g203, tiers 0-6 (the first 7 routes of each of 606 start/goal groups): 4,242 driven, of which 4,240 pass QA (2 rejected as crm_qa:unstalled_break). | 4 | 385 MB |
| `hmmwv_soil_g228_tiers0to6` | m4a | HMMWV soil drives on g228, tiers 0-6: 4,242 driven, all validated. | 4 | 377 MB |
| `gator_collection_qa_records` | m4b | Recorded QA of the Gator collections and the validated-id lists passed to the subset builder: soil all tiers, soil tiers 0-6 and rigid. | 8 | 9.1 MB |
| `gator_f104_rigid_collection_runs` | m4b | The Gator driving the 24,000-route rigid f104 pool (1,200 groups x 20: 14,400 designed routes and 9,600 planner-style routes), 165.7 simulated hours. Chain: ci_f104_gator_rigid.npz (93,551 rows), then subset G_f104_gator_rigid.npz, then the G_rigid ensemble (e5/deploy/G). | 6 | 2.0 GB |
| `gator_f104_soil_collection_runs` | m4b | The Gator driving the 15,235 HMMWV f104 soil route ids (tiers 0-12: 9,168 designed routes and 6,067 planner proposals), 141.1 simulated hours. These are the source of ci_f104_gator_crm.npz (tiers 0-12, 55,826 rows), which feeds the G_full_soil ensemble. They are also the source of the stage-1 file (tiers 0-6, 8,399 runs, 30,827 rows), which feeds the G_soil ensemble. | 6 | 2.3 GB |
| `polaris_collection_build_record` | m4b | The Polaris stage-2 build record (15,235 candidates, 6 rejected, 15,229 selected) and the validated id list passed to the subset builder. | 2 | 552 kB |
| `polaris_collection_task_file` | m4b | soil_v5_polaris.json: the task file passed as --tasks-crm to the Polaris stage-2 build. Its tier 0-12 rows are the 15,235 collection ids. | 1 | 15.9 MB |
| `polaris_f104_soil_collection_runs` | m4b | The Polaris (stock driveline) driving the same 15,235 f104 soil ids: 15,229 valid, 93.0 simulated hours. Chain: stage-2 ci_f104_polaris_crm.npz (tiers 0-12, 57,228 rows), then polaris_full_f104_soil.npz, then the polaris_full_soil ensemble. The 144 sample-A rows of the smoke arm 'polaris' are part of this collection. | 5 | 1.7 GB |

#### Assets

| Item | Milestone | What it is | Files | Size |
|---|---|---|---|---|
| `arena_heightmaps` | m1, m2, m3, m4a, m4b | Heightmap and metadata of every arena driven in a milestone: f104 and 17 generated arenas, plus the arena family description. | 37 | 4.8 MB |
| `crm_soil_config_and_vehicle_variants` | m2, m3, m4a, m4b | The single deformable-soil setting (crm_main.json), the M113 smoke-test soil setting (crm_m113.json) and the Polaris and M113 vehicle definition variants. | 14 | 42.0 kB |
| `f104_terrain_grid` | m3 | Metric terrain grid of f104 used for the terrain crops of the vehicle-dynamics model and the tracker training environment. | 1 | 3.3 MB |

<!-- END fill_card.py: items -->

### What one drive folder contains

Raw collections and evaluation drives share one folder layout: one folder per drive, named by its route id, with a
suffix for drive variants (`__b3` drives with perturbed held controls, `__a4__cN` on rigid ground and `__cN` on soil for
continuations branching from a moving state, `__harvest` drives by the first learned tracker). Raw collections release only the files the dataset builders
read:

| File | Present in | Contents |
|---|---|---|
| `trajectory.npz` | every drive | the drive, one row per 50 ms interval (below) |
| `command_reference.npz` | every drive | the route the follower was given and the speed it was asked for (below) |
| `anchor_state.npz` | every drive | the vehicle at the start of recording, with a short history array (below) |
| `outcome.json` | every drive | how the drive ended: `status` (for example `goal_reached`, `timeout`, `rollover`, `prolonged_blockage_terminated`, `soil_breakthrough_terminated`), `goal_reached`, `elapsed_s`, goal distances and progress (m), path length (m), positive engine work (kJ), frame count, the follower's settings; on soil a `crm` block (particle counts, sinkage and slip summaries); for the Gator and Polaris a `vehicle` block |
| `case.json` | every drive | the start/goal task: `id`, `arena` (the arena's asset path), start pose (in `layout`), `goal_xy`, `goal_radius_m`, `horizon_s`, the route parameters of its group (sideways offsets, speeds, speed profiles), and the train/val/test `split` of its group |
| `episode_complete.json` | every drive | completion marker: elapsed time and the SHA256 of every file the collector wrote, including files not released |
| `initial_state_validation.json` | when recorded | check of the settled start: horizontal speed, roll, pitch, heading and position error against their limits |
| `crm_extra.npz` | soil drives | per 50 ms: chassis height `pos_z_m` and heightmap ground under it `bmp_ground_z_m` (m), chassis quaternion `quat` (e0..e3), wheel-centre heights `spindle_z_m` (m), `slip_ratio` and longitudinal soil force `fsi_force_wheel_fx_n` (N) per wheel (`wheel_order` fl, fr, rl, rr), and `tire_radius_m` |
| `collection_request.json` | soil drives | the collector's inputs: case and route paths with their SHA256, soil settings, seed, stop rules, and the vehicle setup for the Gator and Polaris |
| `vehicle_extra.npz` | Gator and Polaris | per 50 ms (`frame`): `belly_clearance_min_m`, the lowest of the points sampled on the chassis underside minus the undisturbed ground below it (m, negative = inside the soil or ground), `belly_points_below_surface`, `belly_argmin_point`, chassis vertical speed `chassis_vz_mps`; and those points in the chassis frame, `belly_points_chassis_m` |
| `native_height_check.json` | Gator rigid drives | Chrono's terrain height against the heightmap at 50 points (m) |
| `branch_route.json`, `branch_reference.json` | branch continuations | the continuation route from the moving state (waypoints, speeds, stations, headings) |
| `branch_auto.json`, `skipped.json` | rigid branch anchor records | the automatic branch choice at each anchor state, or why the anchor was skipped |

Evaluation drive folders are released with every recorded file except logs, and hold at least `trajectory.npz`, `outcome.json` and
`episode_complete.json`, usually `case.json`; navigation missions instead hold `decisions.json`, `routes.json` and
`trajectory.npz`.

`outcome.json` also has a `safe_goal_reached` flag. It is the collector's own flag (no obstacle contact, no 2 s
effortful stop, no rollover), not the study's label. The study's label counts rolling back on a climb and is
recomputed from `trajectory.npz`: see the outcome codes in
[`traversing/results/README.md`](https://github.com/uwsbel/NeDM/blob/main/traversing/results/README.md).

**`trajectory.npz`.** `T` rows, one per 50 ms interval; row `i` is measured at the start of interval `i` and the
action is the one applied over it.

| Key | Shape, type | Contents |
|---|---|---|
| `state` | (T, 17) float32 | the 17-number vehicle state below |
| `action` | (T, 3) float32 | applied steering in [-1, 1], throttle in [0, 1], braking in [0, 1] |
| `pose` | (T, 3) float64 | world x (m), y (m), yaw (rad) |
| `power_kw` | (T,) float64 | engine output torque times transmission shaft speed (kW) |
| `positive_work_kj_per_interval` | (T,) float64 | positive engine work in the interval (kJ) |
| `contact_n` | (T,) float64 | largest contact force with placed obstacles (N); the arenas have none, so it is 0 |
| `parked` | (T,) bool | the vehicle is at the end of the route and asked to stop |
| `terminal_state`, `terminal_pose`, `terminal_parked` | (17,), (3,), () | the same quantities after the last interval |
| `state_fields` | (17,) str | the names of the state columns |
| `dt_s` | () float32 | 0.05 |

The 17-number state, in this order (`STATE_FIELDS` in `src/nedm/traverse/fdm_data.py`, the preset
`tire_normal_force_omega_pt` of `src/nedm/training/constants.py`, at the experiment commit
[`901d6c9`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/src/nedm/training/constants.py)):

| # | Field | Unit |
|---|---|---|
| 0-1 | `vel_body_x_mps`, `vel_body_y_mps`: chassis velocity in the body frame (x forward) | m/s |
| 2-3 | `roll_rad`, `pitch_rad` | rad |
| 4-6 | `roll_rate_radps`, `ang_vel_body_y_radps` (pitch rate), `yaw_rate_radps` | rad/s |
| 7-10 | `tire_{fl,fr,rl,rr}_force_wheel_fz_n`: normal force on each wheel | N |
| 11-14 | `tire_{fl,fr,rl,rr}_spindle_omega_radps`: wheel spin rate | rad/s |
| 15 | `engine_motor_speed_radps` | rad/s |
| 16 | `engine_motorshaft_torque_nm` | N m |

**`command_reference.npz`.** `interval_start_s` and `desired_speed_mps` (T,): the time of each interval (s) and the
speed the follower was asked for (m/s). `reference_waypoints` (N, 2, m), `reference_stations` (N, m along the route),
`reference_speeds` (N, m/s) and `reference_headings` (N, rad): the route. Rigid branch continuations add `branch_frame` and
`branch_waypoints`, `branch_stations`, `branch_speeds`, `branch_headings`.

**`anchor_state.npz`.** `state` (17,) and `pose` (3,) at the start of recording, `goal_xy` (2, m), `goal_radius_m`,
and `history` (16, 24): 16 steps of 50 ms (0.8 s), each with the 17 state numbers, the 3 previously applied
controls, the position relative to the current pose in its heading frame (x, y in m) and the sine and cosine of the
relative heading (`HISTORY_FIELDS` in `src/nedm/traverse/fdm_data.py`). At a standing start the history repeats the
first state with a braking command.

### `episodes.csv.gz`

One row per drive of a raw collection, next to its tar shards at `traversing/raw/<item>/episodes.csv.gz`:

| Column | Meaning |
|---|---|
| `run_id` | drive folder name |
| `collection` | the collection folder: the folder holding the drive's `runs/` folder, else the drive's parent folder |
| `vehicle`, `world` | `hmmwv`, `gator` or `polaris`; `rigid` or `soil` (from `outcome.json`, else the item's defaults) |
| `arena` | the arena asset path from `case.json`, for example `assets/traverse/arena_f104_50h_v1`; the item `arena_heightmaps` restores that folder (heightmap `arena_000.bmp` and `arena_meta.json`) |
| `status`, `elapsed_s` | end status and recorded time (s) from `outcome.json` |
| `n_files`, `bytes` | released files of the drive and their total size |
| `trajectory_sha256`, `outcome_sha256` | SHA256 of the drive's `trajectory.npz` and `outcome.json` |
| `shard` | the tar shard holding the drive |

These tables have no `load_dataset` configs of their own: the `datasets` library reads every config of a repository
with one file format, here the paper's Parquet. Load a table with the CSV reader and its Hub path instead:

```python
from datasets import load_dataset
eps = load_dataset("csv", split="train", data_files="hf://datasets/harryzhang1018/NeDM@6620faead5225ac9aa5ae8ab19bc2ef2db38a863/"
                   "traversing/raw/hmmwv_soil_f104_collection/episodes.csv.gz")
```

### Download and restore

The helper in the code repository downloads items at the pinned revision, checks every file against the manifest's
SHA256, and restores each item to its path in a checkout (for example `artifacts/traverse/crm_f104_v1/collect_v1/runs/`
for the HMMWV soil collection), which is where the study's scripts expect it. It needs `huggingface_hub`:

```bash
python traversing/scripts/release/download_traversing_data.py --list --all             # items, sizes, restore paths
python traversing/scripts/release/download_traversing_data.py --milestone m2 --bundle models
python traversing/scripts/release/download_traversing_data.py --items hmmwv_soil_f104_collection --verify-members
python traversing/scripts/release/verify_release.py --local artifacts/hf_release/download  # re-check downloaded files
```

The default revision is `6620faead5225ac9aa5ae8ab19bc2ef2db38a863`, the Hub commit pinned in
`traversing/manifests/hf_release_manifest.json`; `--revision` selects another one (with a warning). Items named in an
item's `requires` are added to the selection. Existing files with other content are kept unless `--overwrite` is
given. Without the helper, fetch a folder with
`huggingface_hub.snapshot_download("harryzhang1018/NeDM", repo_type="dataset", revision="6620faead5225ac9aa5ae8ab19bc2ef2db38a863", allow_patterns="traversing/models/m2_shared_history_model/*")`
and extract tar shards with `tar -xzf part-00000.tar.gz` at the root of a checkout.

### Loading notes

- **`.npz` files** contain object arrays in places: load them with `numpy.load(path, allow_pickle=True)`.
- **Models** are PyTorch checkpoints: a dict holding a state dict (under `state` for the risk networks, `model` for the
  NRD, `model_state_dict` for the PPO checkpoint) plus the architecture settings. Load with
  `torch.load(path, map_location="cpu", weights_only=True)`, allowing the NumPy arrays and paths some of them store
  (`load_checkpoint` in `traversing/scripts/release/verify_release.py` shows how). The JSON next to each ensemble is
  its training record. The network classes are in the experiment code at commit
  [`901d6c9`](https://github.com/uwsbel/NeDM/tree/901d6c9423a16c0fafc3d60056065415d5a725f2) and are not on main yet.
- **The tracker actor** (`m3_tracker_round2_numpy_actor/actor.npz`) is plain NumPy: observation normalisation
  (`obs_mean`, `obs_var`, `obs_eps`), the layer weights `W0..W3`, `b0..b3` of a 158 -> 512 -> 256 -> 128 -> 3
  network, and the action scaling (`action_center`, `action_scale`, `action_low`, `action_high`). It runs without
  torch.
- **Units** are in the key names (`_m`, `_mps`, `_rad`, `_radps`, `_n`, `_nm`, `_kw`, `_kj`, `_s`).

### Known limitations

- **Simulation only.** Every drive is a Chrono simulation; there is no real-vehicle data.
- **Whole-arena overhead camera.** Every planner map comes from a fixed overhead camera that sees the whole arena,
  with ideal depth, and the planners use the simulator's exact vehicle pose.
- **One soil setting and one terrain generator.** All soil drives use the same 0.24 m particle layer, and only the
  wheels touch the soil: the HMMWV's rigid tyre mesh, and plain cylinders for the Gator and Polaris calibrated to the
  HMMWV's sinkage. All arenas come from one hill-and-crater generator.
- **The Polaris's engine values are not physical.** Chrono's stock Polaris driveline gives the wheels about 16 times
  the engine's power, so in Polaris drives the engine speed and torque (state numbers 15 and 16), `power_kw` and the
  work values do not describe a real engine.
- **Vehicle-specific models.** Each risk model was trained on one vehicle's drives and is meant for that vehicle; the
  labels describe drives by Chrono's stock PID follower. The predicted failure probabilities rank routes but are not
  calibrated.
- **Re-drives are not always bit-identical.** Chrono runs on the cluster repeat exactly on one node but not across
  nodes. Re-driving the PID follower on soil reproduced 369 of 423 trajectories exactly and 419 of 423 end states.

### License and citation

BSD-3-Clause, the same as Part A and the code. The traversing study has no publication yet: please cite the NeDM paper
(Part A) and name the dataset revision you used (`6620faead5225ac9aa5ae8ab19bc2ef2db38a863`).

## Part C: contact NRD

### What it is

Follow-on work to the paper, not part of it. A neural reduced dynamics model (NRD) for systems whose contacts start
and stop: a bouncing ball, two pool balls, and an SO101 arm that pushes a T-shaped block. A core network, a collision
network and a contact network predict each 20 ms step, with one design and one training config for all three cases.
The design, the results and the commands are in
[`contact_nrd/` of the code repository](https://github.com/uwsbel/NeDM/tree/main/contact_nrd). The training files,
the SO101 test files and the models are byte-exact copies of the study's files. The ball and pool test files are new
conversions of the study's raw test recordings, made with the same converter as the training files. All data come from
[Project Chrono](https://projectchrono.org) simulations.

### Layout

| Path | Contents |
|---|---|
| `contact_nrd/<case>/train/` | `system.json` and `unified_data.npz`: training (split 0) and validation (split 1) episodes, plus the source campaign's own test episodes (split 2, not used) |
| `contact_nrd/<case>/test/` | The same two files for test episodes (split 2) from a separate collection with a new seed. No test episode was used for training or checkpoint selection |
| `contact_nrd/<case>/models/seed61.pt`, `seed62.pt` | The trained models (PyTorch checkpoints) |
| `contact_nrd/release_manifest.json` | Size and SHA-256 of every file; source folders, campaigns, seeds and episode counts |

`<case>` is `bouncing_ball` (5,400 / 900 training / validation episodes, 1,800 test), `pool` (19,200 / 2,400,
4,800 test) or `so101_push_t` (51,200 / 6,400, 1,000 test). Total 9.8 GB.

### What the files contain

- `system.json`: the bodies (kind, size, fixed planes or table), the contact pairs, the record step, the target
  body and time; for the arm and the T, the channel names and types.
- `unified_data.npz` (uncompressed NumPy): `contacts` [N, T-1, P] (pair p touches during record interval k),
  `lengths` [N], `splits` [N], and the states. Ball and pool: `states` [N, T, D, 9] = position, velocity, angular
  velocity of each ball. SO101 push-T: `arm` [N, T, 10] (joint angles, joint speeds), `tshape` [N, T, 13]
  (position, quaternion wxyz, velocity, angular velocity) and `action` [N, T, 5] (joint position command). Other
  arrays (`launches`; for the SO101 `contacts_link`, `t_table`, `scenario`) give more detail on each shot or push;
  the models do not use them. Units are SI. Ball and pool records are 1 ms apart, the SO101 records 10 ms. In the
  ball and pool `system.json`, `model_step_s` is 0.01 s; the released models use 0.02 s (from the training config).

### Download and use

```bash
PYTHONPATH=src python -m nedm.contact_nrd.download --case pool      # in the code repository; checks every SHA-256
```

The code repository's `contact_nrd/release_manifest.json` pins the Hub revision. The files cannot be loaded with
`datasets.load_dataset`; read them with NumPy or with `nedm.contact_nrd.data.load_data`.

### Known limitations

- **Simulation only.** There is no real-robot or real-table data.
- **One scene per case.** One ball, wall and floor; one pool table with fixed ball start positions; one arm, T and
  table. The test episodes are new shots and pushes from the same ranges as the training data.
- **Open-loop scores.** The published errors are model rollouts against recorded episodes, with the recorded arm
  commands.
- **Test sets seen during design.** The study also scored earlier design versions on these test episodes.

### License and citation

BSD-3-Clause, the same as Part A and the code. The contact NRD has no publication yet: please cite the NeDM paper
(Part A) and name the dataset revision you used (`d68fa3c4d91539bc6a079f4b2f3ff5c27d825101`).
