# Collection contract: SO101 cube grasping and G1 locomotion

The data that the two new collectors write. Contact NRD training (`src/nedm/contact_nrd/`) reads it with
`load_data`. Plan and reasons: [`../docs/contact_nrd_grasping_locomotion_plan.md`](../docs/contact_nrd_grasping_locomotion_plan.md).

| Item | Rule |
|---|---|
| Simulator | MuJoCo on the CPU (python bindings), one `MjData` per environment, worker processes. Exact contact access at every physics step. MJX-Warp (GPU) was not tried, so its physics was not compared (pending) |
| Environment | conda env `nrdsim`: [`configs/contact_nrd/collection/requirements.txt`](../configs/contact_nrd/collection/requirements.txt). No torch in collectors |
| Steps | Control period 20 ms (zero-order hold of the joint targets). Record step 10 ms. Model step 20 ms. Physics step: per task, a divisor of 10 ms, checked by a convergence test |
| Units and frame | SI, radians. World z up. Ground or table top at z = 0. Quaternions wxyz, unit, w ≥ 0 at the first record, sign continuous |
| Rates | Instantaneous rates at the record, as the simulator gives them. Linear and angular velocity of free bodies in the **world** frame (MuJoCo free-joint `qvel[3:6]` is in the body frame: rotate it). Training makes the step-average rates itself |
| Writer and checker | `nedm.contact_nrd.collect.export` (pack, shard, merge, write, provenance), `python -m nedm.contact_nrd.collect.validate DIR` |

## Files of one data directory

| File | Contents |
|---|---|
| `system.json` | Bodies, pairs, steps, target body and time, families, terminal reasons, all physics and controller settings |
| `unified_data.npz` | Arrays below (uncompressed `np.savez`) |
| `manifest.json` | Written by `export.write_dataset`: hashes, package versions, git commit, counts; plus the collector's own entries (config, asset hashes, seeds, throughput) |

| Array | Shape, dtype | Meaning |
|---|---|---|
| `<moving body name>` | [N, T, S] float32 | Body state at each record, channels as in `system.json` |
| `action` | [N, T, A] float32 | Joint position targets applied **after** the record, in the interval that starts at it. Constant inside a control period. The final applied value after scaling and clipping |
| `contacts` | [N, T-1, P] uint8 | 1 if the summed normal force of pair p was above the threshold in **any** physics step inside record interval k (between records k and k+1). MuJoCo computes the contacts and forces of a step from the state at the start of that step |
| `lengths` | [N] int32 | Valid records. Records after it repeat the last state and action; contacts after it are 0 |
| `splits` | [N] int8 | 0 train, 1 validation, 2 test (only in a separate test campaign) |
| `family` | [N] int16 | Index into `system.json` `families` |
| `episode_id`, `group_id`, `seed` | [N] int64 | Episode id, group id (branches of one parent share it and the split), episode seed |
| `terminal_reason` | [N] int8 | Index into `system.json` `terminal_reasons` (0 = reached the planned end) |
| `contact_force` | [N, T-1, P] float32 | Diagnostic: largest summed normal force of the pair over the physics steps of the interval (N) |
| `phase` | [N, T] int8 | SO101 only: scripted controller phase at each record, an index into `system.json` `phases`. The evaluation block's timing rule reads it (table events during `close` and `grip` are left out of contact timing) |
| `branch_start` | [N] int16 | Needed if a family has `"use": "core_only"`: the record index at which a core-only branch starts (the counterfactual change acts from the interval after it on; earlier records are the parent's, with their real contact labels). No contact label from it on. -1 for the episodes of the other families (G1: probe_core episodes have 6) |
| Others | free | Task diagnostics (teacher command, raw policy output, torques, task outcome). Training does not read them, except the arrays that the evaluation block names (SO101 `phase`) |

`state[t]` comes before `action[t]`; `action[t]` acts on `[t, t + 10 ms)` and is constant over its 20 ms control
period (`control_step_s`); `state[t+1]` is its result. No record,
window or contact label crosses a reset. A terminal state is recorded before any reset.

## system.json

Required: `name`, `record_step_s` 0.01, `model_step_s` 0.02, `control_step_s` 0.02, `target_time_s`,
`records_per_episode`, `frame`, `bodies`, `pairs`, `pair_metadata`, `target_body`, `contact_rule`,
`families`, `terminal_reasons`, `physics_config`; SO101 also `phases` (the phase names, with `close` and `grip`).

- **Moving body:** `name`, `kind`, `moving: true`, `state_dim`, `channels`, `channel_types`, `channel_mask` (one 0/1
  per channel; 1 = the model predicts it; collectors write all ones), and for the actuated body `action_dim`,
  `action_channels`, `joint_names`. Channel types: `position`, `quaternion`, `angle`, `velocity`,
  `angular_velocity`, `angular_rate`.
- **Channel order** (the integrator pairs pose and rate channels by order): all position and angle channels, in
  the same order as all velocity and angular-rate channels; a quaternion has 4 channels and its angular velocity 3.
- **Fixed body:** `name`, `kind`, `moving: false`, geometry (`top_z_m`, size).
- **Pairs:** `[i, j]` body indices, `i` a moving body. Two contact regions of the same two bodies are two pairs with
  the same `[i, j]` (the model gives each pair its own code). `pair_metadata[p]`: `name`, MuJoCo geom names of
  both sides.
- **contact_rule:** the label rule in words and numbers (force threshold in N, `mj_contactForce` normal force
  summed over the pair's contacts at one physics step).
- **families:** `[{name, group: nominal | difficult | probe, select: bool, note}]`, optional `use`. `select` marks
  the task families that checkpoint selection may use. The family array and the families list come together; every
  family has a boolean `select`, and the validation split has episodes of a selected family (the checker skips this
  rule for a test campaign: split 2 only).
- **use:** absent = the family serves every part of training and evaluation. `"use": "core_only"` = counterfactual
  episodes (for example G1 walking states with the floor switched off) that train only the core: train.py loads them
  apart (`load_data(core_only="separate")`: a store cut after their longest episode, so they are not padded to the
  full record count) and uses their windows only for the core's linear fit, the core loss and the core validation
  report (`core_validation.json` `by_use`). They never enter the history noise (an input noise of the collision and contact
  training: it comes from the core validation windows of the other families), the channel-mask statistics, the
  collision network (contact, near-contact and any windows), the pair input statistics, the contact fit and its step
  RMS, the contact batches, the rollout loss, the checks, the final validation report or test.py (which does not
  load them and says how many). `run_config.json` counts them apart (`core_only_episodes`). A core-only family has
  `select: false` and a `branch_start` array. The checker counts its windows apart (`core_only_*`;
  `core_fit_<body>` = all windows that the core fit draws from) and runs its frame check on them apart
  (`rate_consistency_core_only`); the other families' counts and frame check stay those without it. A test campaign
  needs no core-only episodes (nothing scores them).
- **evaluation:** written by `export.write_dataset` from `configs/contact_nrd/evaluation/<name>.json` (task
  metrics and the checkpoint-selection score; see the README there). `python -m nedm.contact_nrd.collect.export DIR`
  replaces it in an existing directory. Pair names in `pair_metadata` are unique (they key the per-pair reports).

## The two systems

| | SO101 grasping (`so101_grasp`) | G1 locomotion (`g1_locomotion`) |
|---|---|---|
| Bodies | 0 `arm` (moving, kind `articulated`), 1 `cube` (moving, kind `rigid`), 2 `table` (fixed) | 0 `robot` (moving, kind `articulated`), 1 `ground` (fixed plane) |
| State | arm 12: 6 joint angles (`angle`), 6 joint rates (`angular_rate`), jaw included. cube 13: position 3, quaternion 4, velocity 3, angular velocity 3 | 67: base position 3, base quaternion 4, 27 joint angles, base velocity 3, base angular velocity 3, 27 joint rates |
| Action | 6 joint position targets (jaw included) | 27 joint position targets: 12 policy leg targets + 15 upper-body targets |
| Pairs | `[0,1]` fixed jaw–cube, `[0,1]` moving jaw–cube, `[1,2]` cube–table | `[0,1]` left foot–ground, `[0,1]` right foot–ground |
| End the episode before | any other contact (arm–table, other links–cube, self-contact) | any non-foot contact with the ground, self-contact, base height below a stated limit |
| Episode | about 6 s (601 records) | 10 s (1001 records) |
| Target body | cube | robot |

## Families and amounts

| Group | Share | SO101 | G1 |
|---|---|---|---|
| Nominal | about 50 % | Lift, hold, lower, release over reachable cube poses and timings | Frozen teacher with varied velocity and yaw commands, standing, starts and stops |
| Difficult | about 30 % | Off-centre and one-sided touches, near misses, weak closure, slips, drops | Noisy targets, abrupt command changes, initial-state perturbations, recovery, loss of balance |
| Probe | about 20 % | Free arm motion, cube flight, isolated jaw impacts, cube–table impacts | Airborne drops (contact-free windows), single-foot support and landings; core-only branches from walking states with the floor switched off (`probe_core`, `"use": "core_only"`) |

- Training needs contact-free windows of each moving body and windows with exactly one active pair. The checker
  counts them on the training split. Targets for the pilot: at least 1,000 single-pair windows per pair, and at least
  10,000 contact-free windows per moving body on the 20 ms grid (`free_`). train.py fits and trains the core on every
  fifth record only (`core_`, about a quarter of `free_`). Core-only families add core windows only
  (`core_only_core_`); they do not count as single-pair or contact windows.
- Physical parameters are fixed (masses, friction, sizes, gains). Initial states and commands are random. No hidden
  randomisation, no unrecorded pushes.
- Splits: `export.assign_splits`, 90 % train / 10 % validation inside each family, by group. The test set is a
  separate campaign with a new master seed (split 2), sealed until the final test.
- Pilot: 512 task episodes and the probes that the coverage targets need, written under
  `artifacts/contact_nrd/collect_pilot/<system>/`. The coverage targets win over the shares above: the probe share
  may exceed 20 %. Production (18,000 task episodes cut 90/10 inside each family, about 16,200 train + 1,800
  validation, and the probes at 10 times their pilot numbers; a separate test campaign of 2,000 task episodes plus probes with its own master seed) runs on newton, not
  on this workstation.
