# Task metrics of the contact NRD (evaluation blocks)

One file per system. The collection writer copies `<system name>.json` into the data's `system.json` as
`"evaluation"`, so the metrics travel with the data. Training uses the block to pick the best checkpoint. The test
script uses it to score the test campaign. Data without a block (bouncing ball, pool, SO101 push-T) keep the
original free-rollout metrics and score, unchanged.

| File | System | Rollouts in the score | Rollouts in the report only | Outcome flags | Score terms |
|---|---|---|---|---|---|
| `g1_locomotion.json` | G1 walking | Windows (with an end window) | Whole episode from the first state, long windows (2 s, 5 s), event windows | `fall` | 7 |

The pilot data, the released G1 training data and the G1 test data all carry this block in their
`system.json`. Its `note` still says "provisional", but the block did not change after the pilot.

## Data that the blocks read

Besides the arrays of the collection contract, a block can name more data. Loading stops with an error at the start
of training or test if they are missing.

| Block | More data | Checked by |
|---|---|---|
| `g1_locomotion.json` | None | |

## Rollouts

| Key | Meaning |
|---|---|
| `free` | One rollout for each episode from its first recorded state, with the recorded commands, as for push-T. Errors at the target time of `system.json` (`free/target/...`) and at the last valid record (`free/final/...`, the whole episode). Path RMS of the target body up to the target time (`free/path_rmse_m`). |
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

Errors of each moving body (statistics `mean`, `median`, `p95`, `max`; non-finite rollouts: see below):

| Name | Error | Channels |
|---|---|---|
| `pos_m` | Distance between predicted and recorded position | `position` (robot base) |
| `angle_rad` | Rotation angle between predicted and recorded orientation | `quaternion` |
| `joint_rms_rad` | RMS of the joint angle errors | `angle` (all joints) |
| `vel_mps` | Size of the linear velocity error, both as 20 ms averages | `velocity` |

`<mode>/by_start_contact/<pairs>/<horizon>/...` gives the same errors for the windows that start in one contact
phase: the pairs that touch during the first step (`none`, `left_foot_ground`, `left_foot_ground+right_foot_ground`,
...). For G1 this is flight, single or double support.

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
onset. So even a perfect model has physical event errors. Pilot task episodes (512), whole episodes from
the first state, replay of the records:

| System | Physical onsets | Switch onsets | Onsets after a one-step gap | Physical event error rate of a perfect model |
|---|---|---|---|---|
| G1 | 13,324 | 11,845 | 11 % | 0.116 (windows: 0.118) |

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
| Error at a step (horizon, target time, last record) | Uses every rollout that reaches the step. A rollout that is not finite at that step or before counts as an error of 1000 (m, rad or m/s). So the p95 reaches 1000 when more than about 5 % of the rollouts fail |
| Cap | Every error counts as 1000 at most. So a rollout that diverges but stays finite cannot count worse than a non-finite one. An error that is itself not finite (an overflow of a far-diverged state) counts as 1000 |
| `nonfinite` next to the errors | The number of rollouts that are not finite at that step or before. A rollout that fails late keeps its errors at the earlier steps |
| `free/path_rmse_m` | As the error at the target time |
| Outcome flags | A non-finite rollout counts as a disagreement (see above) |
| Contact events | Use every rollout. From the first non-finite step on, the switches are off (the switch rule `logit >= 0` is false for NaN), so later recorded events count as missed |
| `nonfinite_fraction` | Episodes with a non-finite rollout in a mode that the score uses (G1: windows). The score uses it directly. `finite_fraction` = 1 − this; the test output calls it "finite in scored modes". The lines of the report-only modes give their own non-finite counts |
| Per-episode records | `scored_finite`: every rollout of the episode in the scored modes is finite. `nonfinite_modes`: every mode (reports too) with a non-finite rollout of the episode. The errors are raw (no cap), `null` if not finite |
| No rollout reaches the step (for example a horizon longer than every episode of a family) | Every statistic of that step is `null` (also the contact rates, timings and accuracies and the flag disagreement of a mode without any rollout). The test output leaves these lines out |
| The score needs a `null` statistic | Family with `select: false`: its score and that term are `null` ("not available" in the test output). All episodes, the selected families and each family with `select: true`: an error (checkpoint selection must not compare a placeholder) |

## G1 walking

Windows (in the score): horizons 0.02, 0.2, 0.5 and 1.0 s; starts at 0.1 s, then every 0.5 s, plus the end window.
A 10 s episode gives 18 starts with a full 1 s horizon, plus the end window. Reports: the whole episode from the first
state; long windows with horizons 2 s and 5 s, starts at 0.1 s, then every 1 s (8 starts with 2 s and 5 starts with
5 s in a 10 s episode); event windows 0.1 s before each foot touchdown (from the first record for a touchdown in the
first 0.1 s), horizons 0.2, 0.5 and 1.0 s.

The design of the G1 case asks for one-step, 0.2, 0.5 and 1 s errors and for full-episode errors and multi-second drift. Its
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
| `windows/1.0/robot/pos_m/p95` | 0.05 m | 1 | Design target: root position p95 below 5 cm at 1 s |
| `windows/1.0/robot/angle_rad/p95` | 5° (0.0873 rad) | 1 | Design target: root orientation p95 below 5° at 1 s |
| `windows/1.0/robot/joint_rms_rad/p95` | 0.1 rad | 1 | Design target: joint RMS below 0.1 rad |
| `windows/contacts/all/event_error_rate` | 0.1 | 1 | Missed or false touchdowns and liftoffs; 10 % wrong events cost as much as one error at its target |
| `windows/contacts/all/timing_s/p95` | 0.02 s | 1 | Design target: onset and offset p95 within one 20 ms step |
| `windows/flags/fall/disagreement` | 0.01 | 1 | Fall or no fall must agree with the record; 1 % of windows wrong costs one unit |
| `nonfinite_fraction` | 0.01 | 1 | A non-finite rollout is a failure; 1 % costs one unit |

Tolerance: 2 steps (40 ms). An event 3 or more steps away counts as one missed and one false event.

A model at every target, with no wrong fall, scores about 5 (one unit for each of the five error terms).

In the pilot no task episode falls. The 16 task episodes that end early end at a self-contact, with the pelvis at
0.62 m or higher and less than 26° tilt. So on the pilot the fall term counts only falls that the model invents.
Self-contact needs the robot's link geometry, which the NRD state does not have.

## Not in these blocks

| Item | Why not | Where it could go |
|---|---|---|
| Foot paths, stance-foot slip and penetration | They need the robot's forward kinematics and link geometry; the NRD state has joint angles only | The evaluation of the collection code (not in this repository), or extra recorded channels |
| G1 failure at a self-contact | Needs link geometry, as above | As above |
| Duty factor, single- and double-support time as separate numbers | The per-pair switch accuracy and the start-contact split cover them in part | A later report |
| Instantaneous-rate errors | The metrics compare 20 ms average rates, as the model predicts | A later report |
| Teacher-forced against own-history errors at the same horizon | The 0.02 s horizon is teacher-forced; longer horizons use the model's own history | A later report |
| Paired bootstrap intervals between models | `test.py` scores each model alone | A comparison script |
