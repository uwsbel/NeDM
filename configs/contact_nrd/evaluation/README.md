# Task metrics of the contact NRD (evaluation blocks)

One file per system. The collection writer copies `<system name>.json` into the data's `system.json` as
`"evaluation"`, so the metrics travel with the data. Training uses the block to pick the best checkpoint. The test
script uses it to score the test campaign. Data without a block (bouncing ball, pool, SO101 push-T) keep the
original free-rollout metrics and score, unchanged.

| File | System | Rollouts in the score | Rollouts in the report only | Outcome flags | Score terms |
|---|---|---|---|---|---|
| `g1_locomotion.json` | G1 walking | Windows (with an end window) | Whole episode from the first state, long windows (2 s, 5 s), event windows | `fall` | 7 |
| `so101_grasp.json` | SO101 cube grasping | Whole episode from the first state, windows | Event windows | `lifted`, `dropped` | 10 |

The scales, weights and flag thresholds are frozen as used in the full training runs and the sealed test
(`artifacts/contact_nrd/runs/selection_rule.md`: "Evaluation blocks ... are frozen as used in training"; the two
files have the hashes listed in `artifacts/contact_nrd/runs/code_full_sha256.txt` and `code_full2_sha256.txt`,
verified on 2026-10-09). A later change needs new runs. Data written before a change keep the old block in their
`system.json`; `python -m nedm.contact_nrd.collect.export DIR` puts the current block in.

## Data that the blocks read

Besides the arrays of the collection contract, a block can name more data. Loading stops with an error at the start
of training or test if they are missing.

| Block | More data | Checked by |
|---|---|---|
| `so101_grasp.json` | `phase` [N, T] integer array (the controller phase of each record) and the `system.json` list `phases` (its names, with `close` and `grip`); the timing rule reads them | `load_data`; `tests/contact_nrd/test_pilot_data.py` on the pilot data. Not yet in the collection contract or in `validate.py` |
| `g1_locomotion.json` | None | |

## Rollouts

| Key | Meaning |
|---|---|
| `free` | One rollout for each episode from its first recorded state, with the recorded commands, as for push-T. Errors at the target time of `system.json` (`free/target/...`) and at the last valid record (`free/final/...`, the whole episode). Path RMS of the target body up to the target time (`free/path_rmse_m`). An episode that ends before the target time gives its error at its last valid record, and its path RMS up to that record (`data._free_raw`: step min(target, last valid step)). |
| `windows` | Rollouts that start inside each episode. The model gets the recorded last 4 states as its history. Starts at `first_start_s`, then every `start_every_s`, while the shortest horizon fits. A horizon counts only if it ends before the episode ends. Errors at each horizon (`windows/<horizon>/...`). The 0.02 s horizon is one model step from recorded history (the one-step error). With `end_window: true`, each episode also gets one window that ends at its last valid record: for an episode that ends in a failure, the last second before it. |
| `long_windows` | The same as `windows`, with their own starts and horizons (G1: 2 s and 5 s, for multi-second drift). |
| `event_windows` | The same rollouts, but they start `lead_s` before each recorded contact onset (and offset, if `events` has `"offset"`) of any pair, at least `min_gap_s` apart (default 0). An event less than `lead_s` after the episode start gets a window that starts at the first record. |
| `flags` | Outcome conditions, see below. |
| `timing_exclude` | Contact events that the timing statistics leave out, see below. |
| `contact_tolerance_steps` | A predicted contact change matches a recorded one if they are at most this many 20 ms steps apart. |
| `batch` | Rollouts that run together on the GPU. |
| `score` | List of `{stat, scale, weight}`. Score = sum of weight × statistic / scale. Lower is better. |

A rollout mode whose statistics the score does not use is a report:

| Rule | Effect |
|---|---|
| The checks during training skip it | A report adds no time to training |
| Its non-finite rollouts do not count in `nonfinite_fraction` | A report cannot change the score |
| The final report of training and the test compute it | Also per family |

Families with `"use": "core_only"` (G1 `probe_core`: walking states with the floor switched off) are never scored.
`data.evaluate`, the checks, the final report and the test leave their episodes out (train.py keeps them apart for
the core, the test does not load them); the test prints how many. The G1 test campaign collects none. The pilot facts
below are about the scored families only.

Errors of each moving body (statistics `mean`, `median`, `p95`, `max`; non-finite rollouts: see below):

| Name | Error | Channels |
|---|---|---|
| `pos_m` | Distance between predicted and recorded position | `position` (robot base, cube) |
| `angle_rad` | Rotation angle between predicted and recorded orientation | `quaternion` |
| `joint_rms_rad` | RMS of the joint angle errors | `angle` (all joints) |
| `vel_mps` | Size of the linear velocity error, both as 20 ms averages | `velocity` |

`<mode>/by_start_contact/<pairs>/<horizon>/...` gives the same errors for the windows that start in one contact
phase: the pairs that touch during the first step (`none`, `left_foot_ground`, `left_foot_ground+right_foot_ground`,
`fixed_jaw_cube+moving_jaw_cube`, ...). For G1 this is flight, single or double support; for SO101 a free cube, a
cube on the table, a grasp on the table or a lifted grasp.

## Contact events

Per pair and for all pairs (`.../contacts/<pair name or all>/...`):

| Name | Meaning |
|---|---|
| `onset`, `offset` | Recorded, predicted, matched, missed and false events; `timing_s` (absolute timing error of matched events); `bias_s` (mean signed error, + = late); `timing_left_out` (matched events that the timing leaves out, see below) |
| `event_error_rate` | (missed + false) / recorded events, onsets and offsets together |
| `timing_s` | Absolute timing error of all matched events (0 if none matched) |
| `gate_accuracy` | Fraction of steps where the model's contact switch equals the label |

Two views of the same rollouts:

| Key | Recorded label of a 20 ms step | Predicted | Use |
|---|---|---|---|
| `contacts` | The pair touched in this step or in the step before (the model's switch convention) | The switch | Score; what the model learns |
| `contacts_physical` | The pair touched in this step | On if the switches of this step and the next are on | Physical timing |

In the switch view an onset keeps its physical time and an offset comes one step later. A switch cannot show a
one-step gap between two contacts, so in the physical view such a gap counts as one missed offset and one missed
onset. So even a perfect model has physical event errors. Pilot task episodes (512 per system), whole episodes from
the first state, replay of the records:

| System | Physical onsets | Switch onsets | Onsets after a one-step gap | Physical event error rate of a perfect model |
|---|---|---|---|---|
| G1 | 13,324 | 11,845 | 11 % | 0.116 (windows: 0.118) |
| SO101 | 1,624 | 1,514 | 7 % | 0.069 (windows: 0.069) |

The score uses the switch view.

Matching keeps the time order and finds the most matches, then the smallest summed timing error. Events in the
first step of a rollout do not count (the start state already shows them). Windows overlap, so the same event can
count in two windows.

`timing_exclude` leaves some matched events out of `timing_s` and `bias_s` (both views). They still count as
recorded, matched, missed or false events, and in `event_error_rate`:

| Key | Meaning |
|---|---|
| `array` | A per-record integer array of the data [N, T] (for example the controller phase) |
| `values` | The values to leave out. Integers, or names if `names` is given |
| `names` | Optional: a list in `system.json` that names the values (for example `"phases"`) |
| `pairs` | Optional: the pairs that the rule applies to (default: all pairs) |

An event is left out if the array has one of the values at the record where the recorded event happens (the first
record of its 20 ms step).

## Outcome flags

A flag is a condition on one step. It holds for a rollout if it holds at any valid step after the start. The test
evaluates it twice with the same rule: on the recorded states and labels, and on the predicted states and switches.
So a perfect model has no disagreement.

| Condition | Holds when |
|---|---|
| `{"body": B, "channel": C, "above": x}` (or `"below"`) | Channel C of body B (a name of its `channels`) is above (below) x |
| `{"body": B, "tilt_above_rad": x}` | The angle between the body's z axis and the world z axis is above x (from its quaternion) |
| `{"pair": P, "on": true}` | The switch of pair P is on (`"on": false`: off) |

`"all"`: every condition holds. `"any"`: at least one holds. A flag can have both.

Statistics (`.../flags/<name>/...`):

| Name | Meaning |
|---|---|
| `rollouts`, `nonfinite`, `recorded` | All rollouts; rollouts that are not finite up to their last valid step; rollouts flagged in the record |
| `predicted`, `matched` | Finite rollouts flagged by the model; finite rollouts flagged by the model and in the record |
| `missed`, `false` | Finite rollouts flagged in the record only; flagged by the model only |
| `disagreement` | (missed + false + nonfinite) / rollouts: a non-finite rollout counts as wrong |
| `first_s` | Timing error of the first flagged step, rollouts flagged by both |

## Non-finite rollouts and missing rollouts

| Item | Rule |
|---|---|
| Error at a step (window horizon, last record) | Uses every rollout that reaches the step. A rollout that is not finite at that step or before counts as an error of 1000 (m, rad or m/s). So the p95 reaches 1000 when more than about 5 % of the rollouts fail |
| Error at the target time (`free/target`) | Uses every rollout. A rollout that ends before the target time gives its error at its last valid record (the same value as `free/final`). Non-finite rollouts as above |
| Cap | Every error counts as 1000 at most. So a rollout that diverges but stays finite cannot count worse than a non-finite one. An error that is itself not finite (an overflow of a far-diverged state) counts as 1000 |
| `nonfinite` next to the errors | The number of rollouts that are not finite at that step or before. A rollout that fails late keeps its errors at the earlier steps |
| `free/path_rmse_m` | As the error at the target time |
| Outcome flags | A non-finite rollout counts as a disagreement (see above) |
| Contact events | Use every rollout. From the first non-finite step on, the switches are off (the switch rule `logit >= 0` is false for NaN), so later recorded events count as missed |
| `nonfinite_fraction` | Episodes with a non-finite rollout in a mode that the score uses (G1: windows; SO101: first state and windows). The score uses it directly. `finite_fraction` = 1 − this; the test output calls it "finite in scored modes". The lines of the report-only modes give their own non-finite counts |
| Per-episode records | `scored_finite`: every rollout of the episode in the scored modes is finite. `nonfinite_modes`: every mode (reports too) with a non-finite rollout of the episode. The errors are raw (no cap), `null` if not finite |
| No rollout reaches the step (for example a horizon longer than every episode of a family) | Every statistic of that step is `null` (also the contact rates, timings and accuracies and the flag disagreement of a mode without any rollout). The test output leaves these lines out |
| The score needs a `null` statistic | Family with `select: false`: its score and that term are `null` ("not available" in the test output). All episodes, the selected families and each family with `select: true`: an error (checkpoint selection must not compare a placeholder) |

## G1 walking

Windows (in the score): horizons 0.02, 0.2, 0.5 and 1.0 s; starts at 0.1 s, then every 0.5 s, plus the end window.
A 10 s episode gives 18 starts with a full 1 s horizon, plus the end window. Reports: the whole episode from the first
state; long windows with horizons 2 s and 5 s, starts at 0.1 s, then every 1 s (8 starts with 2 s and 5 starts with
5 s in a 10 s episode); event windows 0.1 s before each foot touchdown (from the first record for a touchdown in the
first 0.1 s), horizons 0.2, 0.5 and 1.0 s.

The plan asks for one-step, 0.2, 0.5 and 1 s errors and for full-episode errors and multi-second drift. Its
provisional targets are at 1 s. So the score uses the windows up to 1 s, and the whole episode and the 2 s and 5 s
windows are reports. The rollouts use the recorded commands. The teacher computed them from the recorded states,
not from the predicted ones, so they do not correct a predicted drift. We expect the errors of a balancing robot to
grow fast over several seconds (not measured yet); a score term on them would mostly measure that growth.

Flag `fall`: base height below 0.35 m, or base tilt above 45°.

Pilot (512 task episodes, 1,599 probe episodes). The recorded `fall` flag of the whole episode from the first
state:

| Pilot fact | Recorded `fall` flag |
|---|---|
| The collector ends an episode when the pelvis goes below 0.3 m (18 probe episodes). Their last records are at 0.30 to 0.32 m, with 39.5° to 57.6° tilt | 18 of 18 |
| 5 probe episodes end at a non-foot ground contact, with 0.38 to 0.52 m base height and 36° to 66° tilt at the end | 4 of 5 |
| 130 probe episodes end at a self-contact (0.30 to 0.74 m, 1° to 59° at the end) | 16 of 130 |
| 1,446 probe episodes reach their planned end | 5 of 1,446 |
| Task episodes stay at 0.48 m or higher and below 28° tilt (squats, recoveries) | 0 of 512 |

| Statistic | Scale | Weight | Why |
|---|---|---|---|
| `windows/1.0/robot/pos_m/p95` | 0.05 m | 1 | Plan target: root position p95 below 5 cm at 1 s |
| `windows/1.0/robot/angle_rad/p95` | 5° (0.0873 rad) | 1 | Plan target: root orientation p95 below 5° at 1 s |
| `windows/1.0/robot/joint_rms_rad/p95` | 0.1 rad | 1 | Plan target: joint RMS below 0.1 rad |
| `windows/contacts/all/event_error_rate` | 0.1 | 1 | Missed or false touchdowns and liftoffs; 10 % wrong events cost as much as one error at its target |
| `windows/contacts/all/timing_s/p95` | 0.02 s | 1 | Plan target: onset and offset p95 within one 20 ms step |
| `windows/flags/fall/disagreement` | 0.01 | 1 | Fall or no fall must agree with the record; 1 % of windows wrong costs one unit |
| `nonfinite_fraction` | 0.01 | 1 | A non-finite rollout is a failure; 1 % costs one unit |

Tolerance: 2 steps (40 ms). An event 3 or more steps away counts as one missed and one false event.

A model at every target, with no wrong fall, scores about 5 (one unit for each of the five error terms).

In the pilot no task episode falls. The 16 task episodes that end early end at a self-contact, with the pelvis at
0.62 m or higher and less than 26° tilt. So on the pilot the fall term counts only falls that the model invents.
Self-contact needs the robot's link geometry, which the NRD state does not have.

## SO101 cube grasping

Whole-episode rollout from the first state (as push-T); windows at 0.02, 0.2, 0.5 and 1.0 s (starts at 0.1 s, then
every 0.5 s); event windows 0.1 s before each recorded touch and release (onsets and offsets of the three pairs),
a report.

| Flag | Rule | Meaning |
|---|---|---|
| `lifted` | Cube centre above 0.025 m (bottom 10 mm above the table, 30 mm cube), both jaw switches on, table switch off | The collector's lift rule |
| `dropped` | Cube centre above 0.025 m, both jaw switches off | The cube is in the air without the jaws: a drop (planned or not) or a throw |

On the 512 pilot task episodes the recorded `lifted` flag equals the collector's outcome (410 lifted: 408 held and
2 dropped; 102 missed). The recorded `dropped` flag holds in 33: 31 of the 32 episodes of the family `drop` (planned
drops) and the 2 episodes that the collector marks as dropped (family `weak_closure`). The 32nd `drop` episode ends
at an arm self-contact at 3.9 s, before its planned drop.

Timing rule: cube–table events during the controller phases `close` and `grip` are left out of the timing. There the
closing jaws lift the cube off the table for short times. In the pilot task episodes a cube–table contact break
that starts in these phases lasts 3 steps (60 ms) at the median, 25 of the 48 breaks 3 steps or less; in the other
phases 99 steps at the median (413 breaks; 1 more lasts to the episode end). These events still count in
`event_error_rate`. In the pilot task episodes the rule leaves out 75 of the 899 recorded cube–table events of the
whole-episode rollouts (switch view: 36 of 449 onsets, 39 of 450 offsets) and no jaw event.

| Statistic | Scale | Weight | Why |
|---|---|---|---|
| `free/target/cube/pos_m/p95` | 5 mm | 1 | Plan target: cube position p95 below 5 mm; at the target time of the episode, as for push-T |
| `windows/1.0/cube/pos_m/p95` | 5 mm | 1 | The same target over 1 s from any phase (approach, grasp, lift, hold, release) |
| `free/target/cube/angle_rad/p95` | 5° | 1 | Plan target: cube orientation p95 below 5° |
| `windows/1.0/cube/angle_rad/p95` | 5° | 1 | The same over 1 s from any phase |
| `windows/1.0/arm/joint_rms_rad/p95` | 0.02 rad | 1 | 0.02 rad moves the jaw by about 5 mm at a 0.25 m lever, the size of the cube target |
| `free/contacts/all/event_error_rate` | 0.1 | 1 | Missed or false touches, grasps, releases and table contacts over the whole episode |
| `free/contacts/all/timing_s/p95` | 0.02 s | 1 | Plan target: onset and offset p95 within one 20 ms step (cube–table events in `close` and `grip` left out) |
| `free/flags/lifted/disagreement` | 0.02 | 1 | Lift or no lift must agree with the record; 2 % of episodes wrong costs one unit |
| `free/flags/dropped/disagreement` | 0.02 | 1 | Drop or hold must agree with the record; 2 % of episodes wrong costs one unit |
| `nonfinite_fraction` | 0.01 | 1 | A non-finite rollout is a failure |

Tolerance: 2 steps (40 ms).

The probe families `cube_flight` (650 episodes, 0.32 to 1.0 s) and `cube_table` (100 episodes, 1.0 s) are too short
for a 1 s window. Their 1 s window errors are `null` and their score is "not available". They do not take part in
checkpoint selection (`select: false`).

## Pilot data of these numbers

| System | `unified_data.npz` SHA-256 | Written |
|---|---|---|
| G1 | `5c70f3f3686ff55d1a0be2dd6a0a421aae58b7184e2ec222a4d67e36d6989599` | 2026-10-08 21:31 |
| SO101 | `b7b55a614b817f7285ff9be3b406e85aa921cc2a34b2b7b9f713a004a6faa42c` | 2026-10-08 23:47 |

## Cost

One check during training uses up to 1,500 validation episodes of the families with `select: true`. The checks skip
the report-only rollouts; the final report and the test compute everything, also per family. Measured on one RTX
5090 with the blocks of this folder, a full-size network (a smoke-trained model with random contact-switch heads) and
1,500 production-length synthetic episodes (`tests/contact_nrd/synthetic.py`), two runs each, while other jobs used
the CPU:

| Run | Rollouts | Time | Peak GPU memory |
|---|---|---|---|
| SO101 push-T check today (free rollouts, 401 records; released model, measured earlier) | 1,500 | 12.4 / 12.7 s | |
| G1 check | 30,164 windows | 11.1 / 10.6 s | 3.6 GiB |
| G1 final report or test | the same + 1,500 whole episodes + 11,416 long windows + 39,696 event windows | 61.6 / 67.8 s | 3.6 GiB |
| SO101 grasping check | 1,500 whole episodes + 17,193 windows | 17.3 / 16.4 s | 1.3 GiB |
| SO101 grasping final report or test | the same + 56,934 event windows | 48.2 / 47.5 s | 1.3 GiB |

The G1 check and the final report give the same score (22.85 here): the report-only rollouts do not enter it. The
synthetic contacts switch far more often than real ones: the SO101 pilot has 2,600 event windows for its 512 task
episodes (about 5 per episode, against 38 here).

## Not in these blocks

| Plan item | Why not | Where it could go |
|---|---|---|
| Foot paths, stance-foot slip and penetration; cube motion relative to the gripper | They need the robot's forward kinematics and link geometry; the NRD state has joint angles only | The evaluation of the collection package, or extra recorded channels |
| G1 failure at a self-contact | Needs link geometry, as above | As above |
| Duty factor, single- and double-support time as separate numbers | The per-pair switch accuracy and the start-contact split cover them in part | A later report |
| Instantaneous-rate errors | The metrics compare 20 ms average rates, as the model predicts | A later report |
| Teacher-forced against own-history errors at the same horizon | The 0.02 s horizon is teacher-forced; longer horizons use the model's own history | A later report |
| Paired bootstrap intervals between models | `test.py` scores each model alone | A comparison script |
