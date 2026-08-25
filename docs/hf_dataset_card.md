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

# NeDM — Neural Reduced Dynamics Datasets

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

## Datasets

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

### Splits

Train/val is decided **per episode at collection time** and stored in the `split` column:
`sha1(episode_id)[:8] / 0xFFFFFFFF < validation_ratio → val` (ratio 0.20 for the HMMWV sets,
0.15 for `arm` and `tracked`). Whole episodes stay together; the assignment depends only on the
episode id, so it is stable under re-sharding. `train` and `val` files never share an episode.

## Layout

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

### Column groups

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

### `episodes.parquet` and the metadata bundle

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

## Loading

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

## Processed caches

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

## Known limitations

* `hmmwv_bumpy` episodes are short (mean 3.3 k rows) because 78 % end on the 0.9 × 500 m
  keep-in guard; the regime is meant as a test set.
* The arm collection is restricted to free-space motion (episodes end at first contact) and
  under-samples the lower/rear workspace.
* CRM episodes are 12–18 s long (SPH cost) and use rigid-mesh tires; the CRM tire "force" is the
  fluid–solid interaction force.
* Simulation is deterministic and noise-free; there is no sensor model.

## Citation

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
