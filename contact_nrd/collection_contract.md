# Data format of the G1 locomotion case

The data of the G1 locomotion case. Contact NRD training (`src/nedm/contact_nrd/`) reads it with `load_data`.

| Item | Rule |
|---|---|
| Simulator | MuJoCo on the CPU (python bindings), one `MjData` per environment, worker processes. Exact contact access at every physics step |
| Steps | Control period 20 ms (zero-order hold of the joint targets). Record step 10 ms. Model step 20 ms. Physics step: per task, a divisor of 10 ms, checked by a convergence test |
| Units and frame | SI, radians. World z up. Ground at z = 0. Quaternions wxyz, unit, w ≥ 0 at the first record, sign continuous |
| Rates | Instantaneous rates at the record, as the simulator gives them. Linear and angular velocity of free bodies in the **world** frame (MuJoCo free-joint `qvel[3:6]` is in the body frame: rotate it). Training makes the step-average rates itself |
| Writer and checker | The collection code (not in this repository) packs, splits, writes and checks the data |

## Files of one data directory

| File | Contents |
|---|---|
| `system.json` | Bodies, pairs, steps, target body and time, families, terminal reasons, all physics and controller settings |
| `unified_data.npz` | Arrays below (uncompressed `np.savez`) |
| `manifest.json` | Written by the collection code: hashes, package versions, git commit, counts; plus the collector's own entries (config, asset hashes, seeds, throughput) |

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
| Others | free | Task diagnostics (teacher command, raw policy output, torques, task outcome). Training does not read them, except the arrays that the evaluation block names |

`state[t]` comes before `action[t]`; `action[t]` acts on `[t, t + 10 ms)` and is constant over its 20 ms control
period (`control_step_s`); `state[t+1]` is its result. No record,
window or contact label crosses a reset. A terminal state is recorded before any reset.

## system.json

Required: `name`, `record_step_s` 0.01, `model_step_s` 0.02, `control_step_s` 0.02, `target_time_s`,
`records_per_episode`, `frame`, `bodies`, `pairs`, `pair_metadata`, `target_body`, `contact_rule`,
`families`, `terminal_reasons`, `physics_config`.

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
- **families:** `[{name, group: nominal | difficult | probe, select: bool, note}]`. `select` marks
  the task families that checkpoint selection may use. The family array and the families list come together; every
  family has a boolean `select`, and the validation split has episodes of a selected family (the checker skips this
  rule for a test campaign: split 2 only).
- **evaluation:** written by the collection code from `configs/contact_nrd/evaluation/<name>.json` (task
  metrics and the checkpoint-selection score; see the README there). Pair names in `pair_metadata` are unique (they
  key the per-pair reports).

## The system

| Item | G1 locomotion (`g1_locomotion`) |
|---|---|
| Bodies | 0 `robot` (moving, kind `articulated`), 1 `ground` (fixed plane) |
| State | 67: base position 3, base quaternion 4, 27 joint angles, base velocity 3, base angular velocity 3, 27 joint rates |
| Action | 27 joint position targets: 12 policy leg targets + 15 upper-body targets |
| Pairs | `[0,1]` left foot–ground, `[0,1]` right foot–ground |
| End the episode before | any non-foot contact with the ground, self-contact, base height below a stated limit |
| Episode | 10 s (1001 records) |
| Target body | robot |

## Families and amounts

| Group | Share | G1 |
|---|---|---|
| Nominal | about 50 % | Frozen teacher with varied velocity and yaw commands, standing, starts and stops |
| Difficult | about 30 % | Noisy targets, abrupt command changes, initial-state perturbations, recovery, loss of balance |
| Probe | about 20 % | Airborne drops (contact-free windows), single-foot support and landings |

- The shares are design targets. The released production data have 11,250 nominal, 6,750 difficult and 16,098 probe
  episodes (13,121 drop, 2,977 landing). The probes are short (1.2 s and 1.5 s), so they hold 10 % of the records
  (nominal 57 %, difficult 33 %).
- Training needs contact-free windows of each moving body and windows with exactly one active pair. The checker
  counts them on the training split. Targets for the pilot: at least 1,000 single-pair windows per pair, and at least
  10,000 contact-free windows per moving body on the 20 ms grid (`free_`). train.py fits and trains the core on every
  fifth record only (`core_`, about a quarter of `free_`).
- Physical parameters are fixed (masses, friction, sizes, gains). Initial states and commands are random. No hidden
  randomisation, no unrecorded pushes.
- Splits: 90 % train / 10 % validation inside each family, by group. The test set is a separate campaign with a new
  master seed (split 2). Training and checkpoint selection do not read it.
