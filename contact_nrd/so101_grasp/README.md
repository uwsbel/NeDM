# SO101 cube grasping: simulated data for the contact NRD

A scripted SO101 arm picks up a 30 mm cube in MuJoCo on the CPU. The collector writes contact NRD training data in
the format of the [collection contract](../collection_contract.md). There is no learned teacher: every motion
comes from a script.

| Item | Result (pilot, 2026-10-09) |
|---|---|
| Pilot data | 1,562 episodes, 477,632 records (1.32 h), `artifacts/contact_nrd/collect_pilot/so101_grasp/` |
| Contract check (`collect.validate`) | Passed: no fails, no warnings. The production and sealed test folders pass too (test folder checked again on 2026-10-09: `artifacts/contact_nrd/results/checker_rerun_20261009/`) |
| Lift-hold-release, 320 normal episodes | 320 lifted and held, 0 dropped, 0 missed |
| Slip of the cube in the jaws during the hold | Median 0.030 mm, p95 0.053 mm, max 0.067 mm |
| Physics step | 0.5 ms (20 steps per 10 ms record), chosen by a convergence check over all 12 families |
| Cube inside a gripper part that has no collision shape | Ends the episode (13 of 1,562). Stored overlap at most 0.99 mm. The limit of 1 mm holds to about 0.2 mm: a finer measure finds 5 records above 1 mm, the largest 1.16 mm |
| Records with a joint past its limit | 0 of 477,632 (first pilot: 1,930, 0.40 %) |
| Bounces on the table that gain energy | 0 of 1,193 |
| Deepest cube-table penetration | 4.9 mm (landings up to 2.7 m/s); 0.6 mm in normal grasps |
| Contact threshold | 0.01 N (the cube weighs 0.29 N) |
| Repeat runs, same seeds | Identical arrays (same file hash `b7b55a61…`) in 6 runs: 6 workers with 16 episodes per shard (5 runs) and 4 workers with 7 |
| Replay of the stored commands | The same states and contact labels, bit for bit, in 71 of 71 episodes (all 59 early endings and one of each family) |
| NRD training check (`train --smoke`) | Passed on this pilot: every stage ran, all values finite, 43 s, 13.9 GB GPU memory at peak |
| Speed | 1.28 episodes and 194 saved 20 ms steps per CPU second at load 14 (stored manifest); 1.02-1.72 episodes per CPU second in two other runs (load 21 and 8) |

The fixed NRD (2026-10-10) is in [Fix of the prediction offset](#fix-of-the-prediction-offset-2026-10-10). The
results of the released recipe on the production data are in [Results](#results-the-contact-nrd-on-this-data). The
data description follows.

## Fix of the prediction offset (2026-10-10)

The released recipe drifted from the simulator (cube 18-22 mm at 1 s). There was no indexing, alignment, pose
integration or evaluation bug. The networks (core, collision, contact) and the step rule stay as they are. The fix
changes the state that the networks see, the training rules and the training data.

Sealed test, task families (2,000 episodes), seed 61 / seed 62, p95:

| Statistic | Released recipe | Fixed NRD | Plan target |
|---|---|---|---|
| Cube position, 1 s windows | 18.5 / 22.0 mm | 4.7 / 4.3 mm | Below 5 mm: met |
| Cube orientation, 1 s windows | 14.7 / 15.2° | 3.6 / 3.4° | Below 5°: met |
| Contact onset and offset timing | 40 / 40 ms | 20 / 20 ms | Within one 20 ms step: met |
| Cube position, whole episode at 4 s | 48.5 / 51.4 mm | 13.2 / 11.3 mm | Report only |
| Wrong contact events per recorded event | 9.34 / 6.31 | 0.57 / 0.59 | |
| Lift wrong / drop wrong | 9.5 / 11.9 % ; 3.5 / 7.0 % | 3.3 / 2.5 % ; 2.6 / 1.8 % | Agree with the simulator: better, not met |
| Arm joint RMS, 1 s windows | 5.4 / 4.9 mrad | 4.6 / 4.3 mrad | |
| Selection score | 125.1 / 98.6 | 17.5 / 16.1 | |

Per family (sealed test; 1 s cube p95 / whole-episode cube p95 at 4 s / lift wrong; seed 61, seed 62):

| Family | Released recipe | Fixed NRD |
|---|---|---|
| nominal | 14.6, 18.2 mm / 39.8, 37.8 mm / 1.1, 0.1 % | 3.2, 2.7 mm / 6.7, 5.4 mm / 0.2, 0.2 % |
| off-centre | 11.9, 16.2 / 35.2, 43.5 / 0.8, 0.0 | 4.0, 3.5 / 6.6, 5.2 / 0.8, 1.6 |
| one-sided | 40.0, 35.5 / 52.6, 71.5 / 24.0, 42.4 | 3.1, 3.4 / 6.3, 6.2 / 0, 0 |
| near miss | 42.6, 42.2 / 65.6, 69.1 / 74.4, 73.6 | 10.7, 11.0 / 35.3, 22.8 / 9.6, 11.2 |
| drop | 13.0, 13.7 / 39.9, 32.7 / 2.4, 0.0 | 7.3, 6.9 / 9.7, 9.5 / 0, 0 |
| push | 27.4, 40.0 / 48.6, 66.1 / 18.4, 53.6 | 8.7, 8.1 / 15.5, 15.2 / 11.2, 4.8 |
| weak closure | 36.6, 47.2 / 69.3, 73.7 / 20.0, 20.0 | **60.2, 55.9 / 97.2, 85.0 / 28.8, 20.8 (worse)** |

### Causes (measured on validation episodes)

| Cause | Evidence | Fix |
|---|---|---|
| A held cube is not tied to the gripper | In lifted holds the predicted arm is right (grasp point 0.1 / 1.2 mm at 0.5 s), but the cube slides against its own predicted jaws (6 / 27 mm at 1 s). A 3 mm shift of the cube into a jaw gets a 0.03 mm/s response | State: the grasp point pose and twist (forward kinematics) as arm channels; the cube pose and velocity in the grasp-point frame as cube channels |
| The collision network over-reads rollout drift | 99.2-99.8 % right on recorded histories, 7-10 predicted contact events per recorded event in rollouts. A resting cube that tilts by about 1 degree turns a jaw switch on | Noise floor on the history; collision network trained on in stage 3 with own histories |
| Loss weights ignore the cube | Rollout loss 81-84 % arm (shoulder-pan rate alone 50 %); cube z, vz and spin about 1 % each | One loss scale per body and channel type, from the one-step change on all contact steps |
| Own-history training is almost off | 0-17 % of own histories kept (push-T 84-95 %): the guard unit of the shoulder pan was 6.6 µrad | Guard unit from all contact steps; only touched bodies judged |
| The cube's history noise is ~0 | 1e-6 to 1e-8 (its free motion is a flight), 1e4 times its own one-step error | Noise floor: 5 % of the one-step change on contact steps |
| A few outlier steps dominate the contact loss; one-jaw contacts are rare | Top 1 % of windows carry 85 % of the loss; one jaw plus table is 0.9 % of contact windows | Huber contact loss; contact batches drawn per set of touching pairs (square root of the counts) |
| Rare hard cases | near miss, weak closure, one-sided and push: about 1,000 training episodes each | Campaign `extra`: 2,250 more training episodes of each |

### Recipe

- Config: [`configs/contact_nrd/train_so101_grasp.json`](../../configs/contact_nrd/train_so101_grasp.json) (the
  short schedule: 3,000 collision updates, 15,000 co-training updates; about 1 h on one MI350X). New keys: `derived_state`
  ([`configs/contact_nrd/derived/so101_grasp_fk.json`](../../configs/contact_nrd/derived/so101_grasp_fk.json)),
  `relative_state`, `loss_scale`, `push_guard`, `history_noise` (`floor_step_rms`, `rate_kick`), `contact_huber`,
  `contact_sampling`, `collision_cotrain`. Every key is optional; without them `train.py` gives the released results
  (regression test: 6 tests OK, the three released cases bit-identical).
- Data: production + campaign `extra` of the four hard families (9,000 episodes, all split 0; the validation episodes
  stay those of production):

```bash
PYTHONPATH=src ~/anaconda3/envs/nrdsim/bin/python -P -m nedm.contact_nrd.collect.so101_grasp collect --campaign extra \
    --families one_sided near_miss weak_closure push --workers 24 --out DIR/so101_grasp_extra
python scripts/contact_nrd/merge_data.py DIR/so101_grasp_plus DIR/so101_grasp_production DIR/so101_grasp_extra
PYTHONPATH=src python -m nedm.contact_nrd.train --data DIR/so101_grasp_plus --seed 61 \
    --config configs/contact_nrd/train_so101_grasp.json --output RUNS/so101_fix_s61
PYTHONPATH=src python -m nedm.contact_nrd.test --data DIR/so101_grasp_test --model seed61=RUNS/so101_fix_s61 \
    seed62=RUNS/so101_fix_s62 --output RESULTS/test_so101_fix.json
```

- Selection: a rule written before the test (`artifacts/contact_nrd/fix_20261010/results/selection_rule.md`): the
  lowest mean validation score over seeds 61 and 62 among runs of the reviewed code. Mean validation score 17.15; the
  sealed test was scored once. Runs, test output and validation videos: `artifacts/contact_nrd/fix_20261010/`.

### What did not work, and what is left

| Item | Result (validation) |
|---|---|
| Output mask: the fixed jaw does not drive the jaw joint | No gain; the near-miss lifts came from false moving-jaw switches |
| Pose shifts of the whole history | Arm joint error 8-28 mrad (worse) |
| The core's next state as an extra input token of the pair networks | No gain |
| Step-command probes (servo saturation without contact) | Arm joint error 24-27 mrad (worse); the core fit is dominated by saturation |
| Weak closure (left) | The cube slips out of a weak grip too often in the NRD (lift wrong 21-29 %). The grip holds or slips by about 1 mrad of jaw command |
| Jaw after a false switch (left) | After a one-step false jaw switch in a near miss, the jaw stays at the cube width (0.16 rad) without contact. The core never saw a joint far from its command without contact |

## Results: the contact NRD on this data

Two full runs with the released training config, unchanged (`configs/contact_nrd/train.json`, SHA-256 `be849ac7…`, the
same file as in the release commit `41be3ef`; seeds 61 and 62, one AMD MI350X each). The training code is not the
released code. It is extended for data of the collection contract and is not committed (see "Code version" in
"Reproduce the results"). On this data the extension changes two things. It selects the checkpoint by the task score
of the SO101 evaluation block (`configs/contact_nrd/evaluation/so101_grasp.json`, also new). And the checks during
training use only the 7 task families. Verified: each run's `run_config.json` lists the selection families, and its
training log has the task-score terms; the released `train.py` has neither. Training data: the production campaign
(28,500 episodes, 34.5 h, `unified_data.npz` SHA-256 `64c9c775…`; 25,647 training and 2,853 validation episodes). Each
run keeps its best validation check. The rule was written before the test
(`artifacts/contact_nrd/runs/selection_rule.md`, 17:06 UTC; both SO101 runs ended after it, at 17:26 and 17:42 UTC,
`complete.json` file times). The sealed test campaign (3,050 episodes, master seed 20261034, SHA-256 `657e3b47…`) was
scored by one test job, after training. Training and selection never read it. Later diagnostic rollouts on test
episodes (the timing check, the near-miss check and the videos below) took part in no choice of a model, a checkpoint
or a setting. The video picks use seed-61 values of the test file, by a rule fixed before the rollouts.

"Task families" are the 7 families that checkpoint selection uses: nominal, off-centre, one-sided, near miss, weak
closure, drop and push (2,000 test episodes). "All families" adds the 5 probe families (1,050 episodes): free arm,
thrown cube, cube sliding on the table, cube on the fixed jaw and cube on the moving jaw. This section writes the
family names as words; the data files use identifiers such as `near_miss` and `jaw_fixed`. Every rollout uses the
recorded commands. Sources: `artifacts/contact_nrd/results/test_so101_recipe.json` (test),
`artifacts/contact_nrd/runs/full_so101_recipe_s6{1,2}/` (`validation.json`, `complete.json`, `train_log.jsonl`).
All headline numbers are also in `artifacts/contact_nrd/results/so101_report.json`.

### Plan targets (sealed test)

| Plan target | Seed 61 | Seed 62 | Status |
|---|---|---|---|
| All rollouts finite | 0 non-finite in 3,050 episodes, all rollout modes | 0 in 3,050 | Met (verified) |
| Cube position p95 below 5 mm at 1 s | 18.5 mm (task families), 21.7 mm (all) | 22.0 mm, 24.6 mm | Not met (failed) |
| Cube orientation p95 below 5° at 1 s | 14.7° (task families), 16.9° (all) | 15.2°, 17.2° | Not met (failed) |
| Contact onset and offset timing p95 within one 20 ms step | 40 ms (median 0 ms) | 40 ms (median 0 ms) | Not met (failed) |
| Lift and drop outcomes agree with the simulator, for successful and failed grasps (whole episode, task families) | Lift wrong in 9.5 %, drop wrong in 3.5 % of episodes | 11.9 %, 7.0 % | Not met (failed). Failed grasps: 170 / 237 of the 376 episodes without a recorded lift get a predicted lift (45 % / 63 %). Of the episodes without a recorded lift: near miss 92 of 101 (91 %, both seeds), weak closure 25 of 25 (100 %, both seeds), push 23 / 67 of 125 (18 % / 54 %), one-sided 30 / 53 of 125 (24 % / 42 %). Successful grasps (verified): 19 / 1 of the 1,624 recorded lifts are missed (1.2 % / 0.1 %); lift and drop wrong in at most 3.2 % of nominal, off-centre and drop episodes |

The timing statistic uses matched events only. A predicted event matches a recorded event if they are at most 2 steps
(40 ms) apart. So 40 ms is the largest value that it can have. Two label rules give the same 40 ms, for the task
families and for all families: the switch rule (contact in the step or in the step before; the model's switch learns
this rule and the score uses it) and the physical rule (contact in the step itself).

The statistic also leaves out cube-table events in the close and grip phases (the timing rule of the evaluation
block). There the closing jaws lift the cube off the table for short times. These are 62 / 33 of the 9,048 / 8,001
matched events (seed 61 / 62, switch rule; physical rule: 65 / 29). The event counts and the wrong-event rate still
include them; only the timing leaves them out. With them included, the p95 is still 40 ms, in both seeds and with both rules.
Verified by new whole-episode rollouts of the 2,000 task episodes on the RTX 5090 of this workstation
(`artifacts/contact_nrd/results/so101_timing_exclusion_check/`, script and `result.json`; 56 s; there 62 / 31 events
are left out).

### Sealed test: cube and arm errors

Median / p95, task families (2,000 episodes). A window starts from the recorded 4-state history, at 0.1 s and then
every 0.5 s. The whole episode starts from the first recorded state (6 s).

| Rollout | Cube position (mm), seed 61 | Seed 62 | Cube orientation (°), seed 61 | Seed 62 | Arm joint RMS (mrad), seed 61 | Seed 62 |
|---|---|---|---|---|---|---|
| Window, 0.02 s (one step) | 0.02 / 0.43 | 0.02 / 0.43 | 0.03 / 0.29 | 0.04 / 0.36 | 0.03 / 0.26 | 0.04 / 0.27 |
| Window, 0.2 s | 0.18 / 3.97 | 0.15 / 5.14 | 0.61 / 4.40 | 0.46 / 4.99 | 0.19 / 1.93 | 0.21 / 1.98 |
| Window, 0.5 s | 0.69 / 9.58 | 0.69 / 11.91 | 1.57 / 9.27 | 1.82 / 10.40 | 0.31 / 2.65 | 0.34 / 2.75 |
| Window, 1 s | 1.27 / 18.55 | 1.28 / 22.03 | 2.62 / 14.72 | 3.43 / 15.23 | 0.51 / 5.38 | 0.56 / 4.92 |
| Window that starts 0.1 s before a contact event, 1 s (report only) | 2.35 / 29.0 | 2.89 / 45.9 | 3.00 / 17.5 | 5.49 / 21.0 | 0.65 / 4.01 | 0.75 / 4.09 |
| Whole episode, at 4 s (or at the end, if shorter) | 7.52 / 48.51 | 8.02 / 51.35 | 6.02 / 33.05 | 9.78 / 30.31 | 0.76 / 15.26 | 0.92 / 10.37 |
| Whole episode, at the end | 4.72 / 42.95 | 3.81 / 40.21 | 5.68 / 36.26 | 10.00 / 37.25 | 1.28 / 12.26 | 1.67 / 11.63 |
| Whole episode, cube path RMS up to 4 s | 4.29 / 30.26 | 5.43 / 29.53 | | | | |

All families (3,050 episodes), p95, seed 61 / seed 62:

| Rollout | Cube position (mm) | Cube orientation (°) | Arm joint RMS (mrad) |
|---|---|---|---|
| Window, 0.02 s | 0.49 / 0.51 | 0.40 / 0.47 | 0.24 / 0.24 |
| Window, 0.2 s | 5.26 / 6.29 | 6.56 / 7.02 | 1.77 / 1.84 |
| Window, 0.5 s | 18.42 / 22.35 | 20.04 / 19.67 | 2.50 / 2.63 |
| Window, 1 s | 21.68 / 24.64 | 16.86 / 17.20 | 5.29 / 4.87 |
| Whole episode, at 4 s (or at the end of a shorter episode) | 87.65 / 81.47 | 127.2 / 124.8 | 14.38 / 10.41 |
| Whole episode, at the end | 86.74 / 79.53 | 127.2 / 124.9 | 12.94 / 11.34 |

Validation and test agree (task families). The checkpoint selection did not fit the validation episodes:

| Item | Validation, seed 61 | Test, seed 61 | Validation, seed 62 | Test, seed 62 |
|---|---|---|---|---|
| Selection score | 124.5 | 125.1 | 97.7 | 98.6 |
| Cube position p95 at 1 s (mm) | 18.66 | 18.55 | 21.87 | 22.03 |
| Cube orientation p95 at 1 s (°) | 14.42 | 14.72 | 15.17 | 15.23 |
| Cube position p95, whole episode at 4 s (mm) | 47.65 | 48.51 | 49.27 | 51.35 |
| Wrong contact events per recorded event | 9.36 | 9.34 | 6.36 | 6.31 |
| Lift wrong / drop wrong | 9.8 % / 2.9 % | 9.5 % / 3.5 % | 11.6 % / 5.5 % | 11.9 % / 7.0 % |

### Sealed test: contacts (whole episode from the first state, task families)

| Item | Seed 61 | Seed 62 |
|---|---|---|
| Recorded onsets / offsets (switch rule) | 6,014 / 5,993 | 6,014 / 5,993 |
| Predicted onsets and offsets | 118,194 | 79,800 |
| Predicted events per recorded event | 9.84 | 6.65 |
| Missed / false events | 2,959 / 109,146 | 4,006 / 71,799 |
| Wrong events per recorded event (missed + false) / recorded | 9.34 | 6.31 |
| Same with the physical rule (contact in the step itself) | 6.66 | 4.33 |
| Timing of matched events, median / p95 | 0 / 40 ms | 0 / 40 ms |
| Switch accuracy (steps where the switch equals the label), all pairs | 0.880 | 0.893 |
| Switch accuracy: fixed jaw / moving jaw / cube-table | 0.831 / 0.905 / 0.905 | 0.854 / 0.930 / 0.893 |

Almost all wrong events are false events: the model's switches turn on and off far more often than the recorded
contacts (see "Why the contact score is high").

### Sealed test: per family

Seed 61 / seed 62. "Lift" is the outcome flag of the whole episode: cube centre above 25 mm, both jaw switches on,
table switch off.

| Family | Episodes | Cube position p95 at 1 s (mm) | Cube orientation p95 at 1 s (°) | Cube position at the end, median; p95 (mm) | Wrong contact events per recorded event | Lift: recorded; predicted | Lift wrong |
|---|---|---|---|---|---|---|---|
| nominal | 1,250 | 14.6 / 18.2 | 12.1 / 13.2 | 3.8 / 2.9; 38.2 / 32.6 | 7.92 / 5.39 | 1,250; 1,236 / 1,249 | 1.1 % / 0.1 % |
| off-centre | 125 | 11.9 / 16.2 | 11.3 / 12.8 | 3.1 / 2.9; 34.9 / 34.2 | 6.50 / 4.32 | 125; 124 / 125 | 0.8 % / 0 % |
| one-sided | 125 | 40.0 / 35.5 | 12.7 / 14.0 | 12.8 / 9.3; 53.7 / 66.0 | 31.0 / 16.8 | 0; 30 / 53 | 24.0 % / 42.4 % |
| near miss | 125 | 42.6 / 42.2 | 29.8 / 25.7 | 10.7 / 9.7; 31.2 / 31.7 | 14.4 / 9.95 | 24; 115 / 116 | 74.4 % / 73.6 % |
| weak closure | 125 | 36.6 / 47.2 | 8.5 / 11.2 | 2.3 / 2.7; 18.2 / 17.2 | 12.9 / 8.95 | 100; 125 / 125 | 20.0 % / 20.0 % |
| drop | 125 | 13.0 / 13.7 | 13.7 / 16.3 | 5.1 / 5.6; 34.7 / 28.6 | 6.75 / 4.06 | 125; 122 / 125 | 2.4 % / 0 % |
| push | 125 | 27.4 / 40.0 | 38.5 / 37.4 | 26.7 / 25.2; 63.3 / 83.5 | 19.0 / 14.8 | 0; 23 / 67 | 18.4 % / 53.6 % |
| free arm (probe) | 100 | 0.6 / 0.3 | 3.3 / 2.8 | 1.9 / 0.6; 3.0 / 1.4 | no recorded events; 70 / 90 false | 0; 0 / 0 | 0 % |
| thrown cube (probe, up to 1 s) | 650 | no 1 s window | no 1 s window | 39.1 / 36.8; 92.3 / 79.9 | 0.42 / 0.44 | 0; 0 / 0 | 0 % |
| cube sliding on the table (probe, 1 s) | 100 | no 1 s window | no 1 s window | 16.5 / 16.2; 47.3 / 34.1 | 0.66 / 0.88 | 0; 1 / 1 | 1.0 % |
| cube on the fixed jaw (probe, 2 s) | 100 | 179.2 / 162.8 | 131.9 / 151.4 | 60.6 / 71.3; 331.8 / 333.2 | 1.14 / 1.61 | 4; 2 / 5 | 6.0 % / 5.0 % |
| cube on the moving jaw (probe, 2 s) | 100 | 231.4 / 206.2 | 157.2 / 160.8 | 98.6 / 98.7; 339.0 / 251.3 | 0.98 / 0.98 | 11; 1 / 4 | 12.0 % / 15.0 % |

Drop flag in the drop family: recorded in 124 of 125, predicted in 121 (both seeds), 2.4 % wrong. The arm joint
RMS p95 at 1 s is 0.4-6.8 mrad in every family except near miss: 95.7 / 95.3 mrad (see below).

1 s windows by the contacts at the window start (all families):

| Contacts at the start | Windows | Cube position p95 (mm), seed 61 / seed 62 |
|---|---|---|
| None (cube in the air) | 22 | 8.2 / 10.2 |
| Cube on the table only | 12,815 | 12.6 / 11.7 |
| Both jaws and the table (grasp on the table) | 1,284 | 13.2 / 21.6 |
| Both jaws (lifted grasp) | 6,316 | 21.8 / 24.6 |
| Moving jaw and the table | 222 | 40.7 / 64.0 |
| Fixed jaw and the table | 242 | 61.5 / 79.3 |
| Fixed jaw only (cube on the jaw, probes) | 185 | 179.8 / 164.3 |
| Moving jaw only (cube on the jaw, probes) | 155 | 225.7 / 204.4 |

### Validation: the same tables

The final validation of each run (`validation.json`, `complete.json`) on the best checkpoint. Task families: 1,803
validation episodes; all families: 2,853. Median / p95, seed 61 and seed 62. Last column: p95, all families.

The checks during training use fewer episodes. Every 10,000 updates, `train.py` scores an evenly spaced 1,500 of the
1,803 task validation episodes (`run_config.json`: `check_episodes_used` 1500). Every check value in this report
(from the training logs) uses these 1,500 episodes. So the same checkpoint has two values. Cube position p95 at
1 s: seed 61 at 120,000 updates, 18.94 mm (check) and 18.66 mm (final validation); seed 62 at 110,000 updates,
22.05 and 21.87 mm. Verified from `run_config.json`, `train_log.jsonl` and `validation.json`.

| Rollout | Cube position (mm), seed 61 | Seed 62 | Cube orientation (°), seed 61 | Seed 62 | Arm joint RMS (mrad), seed 61 | Seed 62 | Cube position p95, all families (mm), seed 61 / 62 |
|---|---|---|---|---|---|---|---|
| Window, 0.02 s (one step) | 0.02 / 0.42 | 0.02 / 0.43 | 0.03 / 0.27 | 0.04 / 0.36 | 0.03 / 0.25 | 0.04 / 0.28 | 0.49 / 0.51 |
| Window, 0.2 s | 0.17 / 4.15 | 0.13 / 5.01 | 0.59 / 4.37 | 0.45 / 4.97 | 0.19 / 1.91 | 0.21 / 1.90 | 5.48 / 6.37 |
| Window, 0.5 s | 0.67 / 9.30 | 0.68 / 11.96 | 1.55 / 9.07 | 1.77 / 10.15 | 0.31 / 2.69 | 0.35 / 2.63 | 19.52 / 24.27 |
| Window, 1 s | 1.25 / 18.66 | 1.28 / 21.87 | 2.62 / 14.42 | 3.38 / 15.17 | 0.49 / 5.33 | 0.56 / 4.90 | 22.12 / 25.14 |
| Window that starts 0.1 s before a contact event, 1 s (report only) | 2.32 / 33.84 | 2.97 / 46.02 | 3.02 / 18.87 | 5.54 / 21.63 | 0.65 / 4.24 | 0.75 / 4.02 | 48.80 / 56.09 |
| Whole episode, at 4 s (or at the end, if shorter) | 7.40 / 47.65 | 7.97 / 49.27 | 5.84 / 30.88 | 9.65 / 29.94 | 0.74 / 17.79 | 0.95 / 12.10 | 88.58 / 89.58 |
| Whole episode, at the end | 4.82 / 41.56 | 4.21 / 39.65 | 5.79 / 32.09 | 9.91 / 33.95 | 1.25 / 12.66 | 1.54 / 11.07 | 85.29 / 88.41 |
| Whole episode, cube path RMS up to 4 s | 4.23 / 32.43 | 5.23 / 30.23 | | | | | |

Validation contacts and outcomes (whole episode from the first state, task families):

| Item | Seed 61 | Seed 62 |
|---|---|---|
| Recorded onsets / offsets (switch rule) | 5,375 / 5,351 | 5,375 / 5,351 |
| Predicted events per recorded event | 9.87 | 6.71 |
| Missed / false events | 2,613 / 97,747 | 3,483 / 64,736 |
| Wrong events per recorded event: switch rule / physical rule | 9.36 / 6.61 | 6.36 / 4.35 |
| Timing of matched events, median / p95 | 0 / 40 ms | 0 / 40 ms |
| Switch accuracy: all pairs; fixed jaw / moving jaw / cube-table | 0.879; 0.830 / 0.904 / 0.904 | 0.893; 0.854 / 0.930 / 0.895 |
| Lift: recorded; predicted; wrong | 1,455; 1,589; 9.8 % | 1,455; 1,662; 11.6 % |
| Drop: recorded; predicted; wrong | 123; 150; 2.9 % | 123; 205; 5.5 % |
| Non-finite rollouts (all 2,853 episodes) | 0 | 0 |

Validation per family, seed 61 / seed 62:

| Family | Episodes | Cube position p95 at 1 s (mm) | Cube orientation p95 at 1 s (°) | Cube position at the end, median; p95 (mm) | Wrong contact events per recorded event | Lift: recorded; predicted | Lift wrong |
|---|---|---|---|---|---|---|---|
| nominal | 1,125 | 14.1 / 18.2 | 11.9 / 13.0 | 4.1 / 3.3; 36.5 / 32.9 | 7.84 / 5.40 | 1,125; 1,109 / 1,125 | 1.4 % / 0 % |
| off-centre | 113 | 12.1 / 19.3 | 10.8 / 12.8 | 3.5 / 3.3; 30.6 / 21.7 | 7.14 / 4.58 | 113; 113 / 113 | 0 % / 0 % |
| one-sided | 113 | 36.6 / 31.8 | 11.3 / 13.5 | 10.4 / 10.0; 52.2 / 57.1 | 29.0 / 17.0 | 0; 17 / 37 | 15.0 % / 32.7 % |
| near miss | 113 | 42.1 / 43.3 | 25.2 / 24.3 | 9.4 / 7.0; 30.8 / 32.7 | 17.1 / 11.6 | 18; 106 / 106 | 77.9 % / 77.9 % |
| weak closure | 113 | 45.7 / 60.6 | 11.1 / 13.3 | 3.5 / 2.8; 23.0 / 25.5 | 12.8 / 8.96 | 85; 113 / 113 | 24.8 % / 24.8 % |
| drop | 113 | 11.0 / 11.7 | 11.9 / 14.3 | 4.4 / 4.0; 18.4 / 25.7 | 6.55 / 3.88 | 113; 109 / 113 | 3.5 % / 0 % |
| push | 113 | 28.2 / 38.5 | 43.6 / 42.1 | 27.1 / 25.1; 76.0 / 107.0 | 18.9 / 14.0 | 1; 22 / 55 | 20.4 % / 49.6 % |
| free arm (probe) | 100 | 0.6 / 0.3 | 3.1 / 2.5 | 1.8 / 0.7; 2.9 / 1.3 | no recorded events; 94 / 38 false | 0; 0 / 0 | 0 % |
| thrown cube (probe) | 650 | no 1 s window | no 1 s window | 40.0 / 39.0; 97.2 / 89.8 | 0.47 / 0.55 | 0; 2 / 2 | 0.3 % |
| cube sliding on the table (probe) | 100 | no 1 s window | no 1 s window | 14.8 / 16.4; 60.6 / 38.8 | 0.32 / 0.32 | 0; 0 / 0 | 0 % |
| cube on the fixed jaw (probe) | 100 | 174.0 / 155.2 | 152.4 / 168.9 | 53.8 / 64.1; 281.5 / 285.5 | 1.32 / 1.68 | 3; 2 / 4 | 3.0 % / 5.0 % |
| cube on the moving jaw (probe) | 100 | 283.9 / 227.1 | 146.1 / 167.2 | 98.3 / 110.1; 381.5 / 409.6 | 1.02 / 1.10 | 10; 2 / 5 | 12.0 % / 13.0 % |

The validation pattern is the test pattern: the same families fail (near miss, weak closure, push, one-sided, the
jaw probes), and the arm joint RMS p95 at 1 s is again large only in near miss (97.7 / 97.2 mrad).

### What works and what does not

| Claim | Evidence (sealed test unless stated) | Label |
|---|---|---|
| Rollouts stay finite | 0 non-finite rollouts, 3,050 episodes, all modes, both seeds | Verified |
| Short horizons are accurate | Cube p95: one step 0.43 mm; 0.2 s 3.97 / 5.14 mm (task families) | Verified |
| The arm follows its commands | Joint RMS p95 at 1 s: 5.4 / 4.9 mrad (task families); near miss excepted | Verified |
| Successful grasps keep their outcome | 19 / 1 of the 1,624 recorded lifts are missed. Nominal, off-centre and drop: lift and drop wrong in at most 3.2 % of episodes; cube at the end median 2.9-5.6 mm | Verified |
| Validation predicts the test | Table "Validation and test agree" above: cube p95 differs by at most 0.16 mm at 1 s and 2.1 mm at 4 s | Verified |
| Failed grasps keep their outcome | 170 / 237 of the 376 task episodes without a recorded lift get a predicted lift (45 % / 63 %). The next three rows give the families | Failed |
| A near miss stays a miss | 92 of the 101 near-miss episodes without a recorded lift get a predicted lift (91 %, both seeds) | Failed |
| A weak grasp slips | All 25 weak-closure episodes where the cube slips out at lift-off get a predicted lift (100 %, both seeds) | Failed |
| Pushes and one-jaw touches do not lift the cube | Predicted lifts: push 23 / 67 of 125 (18 % / 54 %), one-sided 30 / 53 of 125 (24 % / 42 %); none recorded | Failed |
| Cube within 5 mm / 5° at 1 s | 18.5 / 22.0 mm and 14.7 / 15.2° | Failed |
| Contact timing within one step | p95 40 ms; 6.6 to 9.8 predicted contact changes per recorded change | Failed |
| A cube that rests on one jaw | 1 s windows that start with one jaw only: cube p95 164-226 mm (jaw probes) | Failed |
| Whole episodes | Cube p95 at 4 s: 48.5 / 51.4 mm (task families), 87.7 / 81.5 mm (all) | Failed |
| The 1 s error mostly levels off after 50,000 updates | Validation checks (1,500 episodes, see "Validation: the same tables"), 1 s cube p95: 49.3 / 45.6 mm at 10,000 updates, 21.4 / 23.2 mm at 50,000, 18.9 / 21.2 mm at 120,000. From 50,000 to 120,000 updates it fell only 2.5 / 2.0 mm (12 % / 9 %). Between them 17.6-25.7 mm, except 32.0 / 38.5 mm at 70,000 and the unstable check below | Verified (training logs) |

**Near miss.** The jaws close 15-35 mm off centre and miss the cube. The model closes them on the cube. Checked on
8 sealed-test near-miss episodes without a lift (rows 1500, 1506, 1513, 1522, 1531, 1537, 1548, 1555; collector
outcome "missed"; seed 61; whole episode from the first state):

| Item | Value (range over the 8 episodes) |
|---|---|
| Arm joint with the largest error | The jaw, in all 8 |
| Smallest recorded jaw angle (the simulated jaw closes to) | -0.12 to +0.01 rad |
| Predicted jaw angle at that step | 0.159 to 0.170 rad |
| Predicted jaw angle in the grip, lift and hold phases (median per episode) | 0.167 to 0.168 rad |
| Jaw angle on a held cube in the record (nominal test grasps that are lifted and held; median per episode over the record intervals with both jaws on the cube) | 0.167 rad (5-95 %: 0.167 to 0.168 rad), the same for the first 300 and for all 1,250 |
| Largest jaw error | 0.16 to 0.29 rad |

So the model stops the jaw at the angle of a held cube. Verified on 2026-10-09 by a new rollout of the 8 episodes
(RTX 5090, about 10 s). Script, rollout file and output:
`artifacts/contact_nrd/results/so101_near_miss_jaw_check/` (`check.py`, `near_miss_s61_first.npz`, `result.json`).
This rollout took part in no choice.

### Why the contact score is high: switch flicker

The contact-event part of the score is 75 % (seed 61) and 64 % (seed 62) of the task-family test score (93.4 of
125.1; 63.1 of 98.6). Verified from the test file. The learned switches turn on and off in short bursts where the
recorded contact is steady.

A separate check on 2026-10-09 measured the effect on validation: seed 61 at 60,000 updates (the `best.pt` of the
run at that time), 400 evenly spaced task validation episodes, whole episodes from the first state. Script and
output: `artifacts/contact_nrd/results/so101_switch_diagnostic/` (`gatetest.py`, `flicker.py`, `results.json`). The
numbers below are those of `results.json` (verified against the file). The check ran once. The checkpoint of
60,000 updates is not kept (the run folder has only the final `best.pt` and `last.pt`), so the same check cannot run
again on it.

| Switches in the rollout | Cube position at the end, median / p95 | Predicted switch onsets per recorded onset |
|---|---|---|
| Learned (the model's own) | 3.66 / 39.68 mm | 9.56 |
| Recorded contact labels in place of the switch | 3.02 / 33.55 mm | 1.0 |
| Learned, with hysteresis of 1 / 2 / 4 steps | p95 41.73 / 48.45 / 50.63 mm | 8.35 / 7.26 / 6.21 |

| Claim | Evidence | Label |
|---|---|---|
| The flicker causes little of the cube error | Perfect switches lower the end p95 only from 39.7 to 33.6 mm (table above) | Verified once (`results.json`) |
| The contact network's response (the speed change while a pair is on) causes most of the cube error | Conclusion of that check. The test replaced only the switches with the recorded labels. It did not separate the contact network from the free-motion network (the core) or from the arm | Pending: not isolated |
| Hysteresis on the switch does not help | p95 41.7 / 48.5 / 50.6 mm against 39.7 mm | Verified once (`results.json`) |
| Where the flicker is (300 validation episodes, `flicker.py`) | Off gaps of 1 or 2 steps between two on runs: fixed jaw 1,148 predicted against 2 recorded, moving jaw 575 against 0, cube-table 1,912 against 7. The phases with the most predicted gaps: cube-table in the retreat (635), fixed jaw in the close (331), the retreat (317) and the descent (287), cube-table in the descent (271), moving jaw in the close (270) | Verified once (`results.json`, `flicker_by_pair_300_episodes`; the file keeps the 6 largest phase counts) |
| Where the flicker is (the five video episodes) | See the list below | Verified from the rollout file |
| The wrong-event rate does not fall with training | Validation checks (1,500 episodes): seed 61 6.35 at 10,000 updates, 9.32 at 120,000; seed 62 5.88 and 6.50. Over the same checks the 1 s cube p95 fell (49.3 to 18.9 mm; 45.6 to 21.2 mm) | Verified from the training logs |
| The flicker did not change which check each run kept | The score without its contact-event part keeps the same checks: 120,000 updates (seed 61) and 110,000 (seed 62) | Verified (recomputed from the training logs) |
| Within a run, the score follows the contact-event part | Correlation over the 12 checks (1,500 episodes each): seed 62 0.90; seed 61 0.67 without the unstable check at 80,000 updates, -0.48 with it | Verified (recomputed); strong for seed 62 only |
| Across the seeds, the contact-event part decides the order | Seed 62 has the lower score (98.6 against 125.1, sealed test), but the larger 1 s cube error and more wrong lift and drop outcomes. Without the contact-event part, seed 61 scores better: 31.7 against 35.5 (sealed test), 30.9 against 34.1 (final validation) | Verified |
| One unstable check | Seed 61 at 80,000 updates: 1 s cube p95 104 mm, lift wrong in 80 % of the check episodes. It recovered by 90,000 updates | Verified from the training log |

Flicker in the five video episodes (verified: labels and switches in `so101_test_s61_first.npz`; times are the start
of the first and the end of the last 20 ms step that is on; the first 0.1 s left out):

- Cube-table switch, steps where it differs from the label, by the recorded phase of the step: in nominal,
  off-centre and drop, 30 of 48 are in the descent and the close (5 more in the grip). In push, 14 of 31 are in the
  push and 9 in the retreat. In weak closure, 88 of 120 are in the lift, hold, loosen and lower phases, while the
  model carries the cube.
- In the four grasps (nominal, off-centre, drop, weak closure), the jaw switches first flicker on before the recorded
  touch: the fixed jaw 1.04-1.40 s before, the moving jaw 0.30-1.18 s before.
- In the three grasps that lift the cube (nominal, off-centre, drop), both jaw switches go off at the recorded
  release (4.12, 4.72 and 3.98 s). Then the fixed-jaw switch comes on again in all three, and the moving-jaw switch
  in two:

| Episode, recorded release | Fixed-jaw switch on again | Moving-jaw switch on again | Recorded jaw touch after the release |
|---|---|---|---|
| nominal 330, 4.12 s | 4.18-4.24, 4.26-4.28 and 4.32-4.54 s (last off 0.42 s after the release) | 4.22-4.26 s | Fixed jaw, 4.38-4.42 s (40 ms) |
| off-centre 1362, 4.72 s | 4.76-4.90, 4.92-5.00 and 5.04-5.10 s (last off 0.38 s after) | 4.82-4.84 s | None |
| drop 1833, 3.98 s | 4.12-5.06 s (last off 1.08 s after) | None | None |

### Videos and contact timelines (sealed test, seed 61)

Folder `artifacts/contact_nrd/results/videos/so101/`: one video (`.mp4`, 50 frames per second), one contact
timeline (`_contacts.png`) and stills at 1.5, 3, 4.5 and 5.98 s (`_t*.png`, with a simulator replay panel) per
episode. Checkpoint `full_so101_recipe_s61/best.pt`; whole episodes from the first state (6 s). Pick rule (fixed
before the rollouts): in each family, among the episodes that run to the end and meet the condition, the episode at
the median of the seed-61 cube error at the end. The fifth episode is an extra failure example.

| Family, row | Condition | Cube error at the end | Largest cube error in the episode | Wrong contact events (recorded events) | Lift: recorded / predicted |
|---|---|---|---|---|---|
| nominal, 330 | Lifted and held | 3.80 mm | 5.9 mm at 2.58 s | 72 (8) | yes / yes |
| off-centre, 1362 | Lifted and held | 3.11 mm | 5.1 mm at 3.42 s | 42 (6) | yes / yes |
| drop, 1833 | Released in the air | 5.06 mm | 5.3 mm at 5.12 s | 44 (6) | yes / yes; drop yes / yes |
| push, 1913 | Pushed, not lifted | 27.3 mm | 29.3 mm at 3.54 s | 82 (4) | no / no |
| weak closure, 1647 (extra) | The cube slips out at lift-off | 2.58 mm | 47.8 mm at 2.76 s | 124 (14) | no / yes |

Verified. End error and wrong events: the test file (MI350X). Largest error and recorded events: the video rollouts
(RTX 5090 of this workstation). The times below are from the video rollout file (verified, same rule as in the
flicker list above).

- **nominal (verified):** the predicted cube stays within 6 mm of the record. The fixed-jaw switch flickers on during
  the descent (0.74-1.78 s), before the touch at 1.78 s; the cube-table switch flickers off at the same time. Both
  jaw switches go off at the recorded release (4.12 s). The fixed-jaw switch then comes on again in three bursts
  from 4.18 to 4.54 s; the record has one 40 ms fixed-jaw touch at 4.38-4.42 s. The moving-jaw switch comes on
  again at 4.22-4.26 s.
- **off-centre (verified):** both jaw switches go off at the recorded release (4.72 s). The fixed-jaw switch comes
  on again in three bursts from 4.76 to 5.10 s; the record shows no touch.
- **drop (verified):** the release and the landing come at the right step (release 3.98 s, cube on the table
  4.04 s). The fixed-jaw switch goes off at the release, then comes on again from 4.12 to 5.06 s; the record shows
  no touch. The moving-jaw switch stays off.
- **push (verified):** the moving-jaw switch flickers on from 1.68 s and stays on from 1.82 s, 0.48 s before the
  recorded touch (2.30 s). It goes off at the end of the touch (record 3.36 s, switch 3.40 s), then comes on 3 more
  times up to 4.14 s. The fixed-jaw switch is on in 43 of the 45 steps of 1.76-2.66 s (runs 1.76-2.42, 2.44-2.52
  and 2.54-2.66 s) and in 26 of the 27 steps of 3.02-3.56 s (runs 3.02-3.10 and 3.12-3.56 s). It also flickers on
  for 1-4 steps at a time from 1.18 s and up to 4.16 s. The record has one fixed-jaw touch, 3.26-3.40 s. The
  predicted cube ends 27.3 mm away, turned by 41.8° (test file, MI350X); the video rollout gives 27.28 mm and 39.2°.
- **weak closure (failed):** in the simulator the cube slips out at lift-off and stays on the table. The model keeps
  both jaw switches on from 2.30 to 4.08 s and carries the cube up (predicted centre height 62.7 mm at 2.76 s;
  recorded 15.0 mm, on the table). It puts the cube down at the release, so the error at the end is only 2.6 mm.
  The error at the end can hide such a failure; the lift flag and the path RMS show it.
- **Checks (verified):** a simulator replay from the recorded start state with the recorded commands differs from
  the record by at most 0.067 mm at the 7 times that the check samples in each episode (0.02, 0.2, 0.5, 1, 2, 4 and
  6 s; largest: off-centre at 6 s; `so101_test_s61_first_replay_check.json`). The check does not look between these
  times. The RTX 5090 rollouts give the test
  file's end error and wrong-event count for rows 1362, 1833 and 1647. Row 330: 3.72 against 3.80 mm (the same 72
  wrong events). Row 1913: 27.28 against 27.31 mm, 88 against 82 wrong events. A different GPU changes the digits
  of a long contact rollout.
- Time: rollouts 5 s, rendering of the five episodes with stills and replay 8 min 51 s on this workstation (an
  observation; not in a file, pending).

### Training cost

The training stages are those of [`../README.md`](../README.md): stage 1 is a linear fit of the core, stage 2
trains the collision network alone, stage 3 trains the core and the contact network together. Verified from the
training logs and the file times unless a row says otherwise.

| Item | Seed 61 | Seed 62 |
|---|---|---|
| GPU | One AMD MI350X (torch 2.10.0, ROCm 7.1) | The same |
| Jobs | 3 chained 4 h jobs; the training log shows 2 | The same |
| Job 1 (`457145` / `457148`) | Stages 1 and 2, then stage 3 up to 90,000 updates (last check at 3.87 h); stopped by the 4 h limit (inferred) | Last check at 3.63 h |
| Job 2 (`457146` / `457149`) | Resumed at 90,000; 120,000 updates and the final validation in 1.86 h | 1.62 h |
| Time from start to finish | 5.87 h | 6.14 h (31 min in the queue between the jobs) |
| GPU time (job 1 counted as 4 h) | 5.86 h | 5.62 h |
| Work lost at the job 1 limit (after the last check) | 8 min | 22 min |
| Stage 2: collision network alone | 30,000 updates at 144 updates/s | 146 updates/s |
| Stage 3: core and contact network together, mean with the checks | 120,000 updates at 5.9 updates/s | 6.4 updates/s |
| Stage 3, first / last 10,000 updates | 14.7 / 4.4 updates/s | 15.0 / 5.1 updates/s |
| Best check | 120,000 updates | 110,000 updates |
| Peak GPU memory | Not logged | Not logged |

- Stage 3 slows down while the horizon of the rollout loss grows from 0.1 s to 1 s (first half of the updates).
- That job 1 ran to the 4 h limit is inferred: its next check would have come after 4 h. The third job left no line
  in the log (pending: check the cluster job record).
- Peak GPU memory: the estimate before the run was about 30 GB ([`../README.md`](../README.md)). It was not
  measured.

### Earlier runs

| Run | Data | Schedule | Cube position p95 at 1 s (validation) | Label |
|---|---|---|---|---|
| Pilot, short (`artifacts/contact_nrd/pilot_runs/so101_grasp_short_s61`) | Pilot, 1,562 episodes | 3,000 updates of stage 2 + 15,000 of stage 3 | 98.5 mm | Verified (`complete.json`) |
| Production, short (`short_runs/prod_so101_base_s61`: only `best.pt` kept) | Production | 3,000 updates of stage 2 + 15,000 of stage 3 | 18.50 mm at the best check (12,000 updates; a check scores up to 1,500 validation episodes); 18.19 mm in the final validation (score 49.26) | Verified (log `artifacts/contact_nrd/short_runs/logs/prod_so101_base_s61.stdout.log`) |
| Learning rate 3e-4 (`prod_so101_lr3_s61`) | Production | Short | 20.09 mm (final validation; score 58.67 against 49.26): no gain | Verified (log in the same folder) |
| Ridge 0.1 in the core's linear start fit, together with learning rate 3e-4 (`prod_so101_r01lr3_s61`) | Production | Short | 85.24 mm (final validation; score 201.50): worse. Ridge 0.1 alone was not tried on SO101 | Verified (log in the same folder) |
| Full runs (this section) | Production | Full | 18.66 / 21.87 mm (task families) | Verified |

At a short schedule, the production data lowered the 1 s error from 98.5 to 18.2 mm (final validation; each on its
own validation episodes). The full schedule did not lower it further (18.66 / 21.87 mm).

### Reproduce the results

All outputs go to a new folder, `N`, in the layout of `artifacts/contact_nrd`. The commands do not write over the
reported data, runs, test file or videos.

```bash
N=artifacts/contact_nrd/rerun; mkdir -p $N/production $N/test_sealed $N/runs $N/results/videos/so101
# 1. Data: the production and sealed test campaigns (on newton, commands in "Reproduce" below, with
#    DIR/so101_grasp_production = $N/production/so101_grasp and DIR/so101_grasp_test = $N/test_sealed/so101_grasp)
# 2. Training, one seed per GPU (about 5.6-5.9 GPU hours on an MI350X). After a 4 h job limit, run the same command
#    again: it continues from the last check (every 10,000 updates)
PYTHONPATH=src python -m nedm.contact_nrd.train --data $N/production/so101_grasp --seed 61 \
    --output $N/runs/full_so101_recipe_s61
PYTHONPATH=src python -m nedm.contact_nrd.train --data $N/production/so101_grasp --seed 62 \
    --output $N/runs/full_so101_recipe_s62
# 3. Sealed test, both seeds
PYTHONPATH=src python -m nedm.contact_nrd.test --data $N/test_sealed/so101_grasp \
    --model seed61=$N/runs/full_so101_recipe_s61 seed62=$N/runs/full_so101_recipe_s62 \
    --output $N/results/test_so101_recipe.json
# 4. Videos and timelines: rollouts (env nedm), then the render (env nrdsim)
PYTHONPATH=src ~/miniconda3/envs/nedm/bin/python -P -m nedm.contact_nrd.rollouts \
    --data $N/test_sealed/so101_grasp --model $N/runs/full_so101_recipe_s61/best.pt \
    --episodes 330 1362 1833 1913 1647 --mode first --out $N/results/videos/so101/so101_test_s61_first.npz
MUJOCO_GL=egl PYTHONPATH=src ~/miniconda3/envs/nrdsim/bin/python -P -m nedm.contact_nrd.collect.render \
    $N/results/videos/so101/so101_test_s61_first.npz --out $N/results/videos/so101 --stills 1.5 3 4.5 5.98 --replay
```

- To score the reported checkpoints again, give `--model` the folders
  `artifacts/contact_nrd/runs/full_so101_recipe_s6{1,2}` and keep `--output` below `N`.
- **Code version.** Job 1 of each run imported `train.py` `a6ab99a8…`, `data.py` `e99ebdf9…`, `model.py` `ac52d8f5…`
  and `test.py` `9168d41b…` (`run_config.json`, `source_sha256`; verified). These are not the files of the release
  commit `41be3ef` (there: `train.py` `20a856f6…`, `data.py` `b02f393b…`, `model.py` `12ae3268…`, `test.py`
  `a53fcb33…`; verified with `git show`). The extended code is not committed. Only the training config is the released
  file. So "released recipe, unchanged" in `selection_rule.md` (written before the test) holds for the config only.
  Job 1 ran from the cluster folder `code_full` (path in its warning lines in `*.stdout.log`). These hashes cover job
  1 only: `train.py` writes `run_config.json` only if the file does not exist yet, and a resumed job logs only its job
  ID. So no file records the code that job 2 ran (updates 90,000-120,000, the final validation and the choice of
  `best.pt`). It was probably the same `code_full` folder (pending: not on record). This worktree has later versions
  of `train.py`, `data.py` and `test.py` (`b662acbe…`, `5479b60a…`, `9bf49b8d…`; the same as
  `artifacts/contact_nrd/runs/code_full2_sha256.txt`). Since the runs, `model.py`, the training config, the evaluation
  block and the SO101 collector files are unchanged; the data checker (`collect/validate.py`) and the G1 collector
  changed (verified: both hash lists against this worktree). The test output does not record the code version that
  wrote it (pending).
- A new run with the same seed does not give the same digits (see [`../README.md`](../README.md)).

## What changed since the first pilot

| Change | Why | Effect in the pilot |
|---|---|---|
| An episode ends when the cube enters a part of the gripper or moving jaw that has no collision shape | The cube passed 3-6 mm into such parts, and these intervals were stored as contact-free | 13 episodes end early. No stored record keeps more than about 1 mm of such an overlap (see "The cube inside gripper parts") |
| Every scripted joint target stays 50 mrad inside the joint limits | The wrist ran up to 25 mrad past its limit in 383 of 512 task episodes | No record has a joint past its limit (see "Joint limits") |
| The stored command is the float32 value that the simulation applies | The applied float64 target differed from the stored value by up to 1e-7 rad, so replays drifted in contact-rich episodes | Replays of the stored commands match bit for bit |
| Campaigns (pilot, production, test) with their own master seeds, shard files, a merge step and the production commands | Production and the sealed test run on newton | See "Campaigns and seeds" |
| `pair_metadata` names the contact regions of the two jaw pairs | Pair 0 also covers the palm blocks and the wrist-servo box | See "The jaw" |

## The scene

| Part | Setting | Reason |
|---|---|---|
| Arm | MuJoCo Menagerie SO101, pinned revision `0059d43`, file unchanged | `fetch_sources.py` fetches it and checks every file hash |
| Arm base | At the origin, on the table top. Collision shapes: the convex hulls of its 4 meshes (added) | The Menagerie base has no collision shapes, so a falling cube went through it. Now a cube that touches the base ends the episode. Base-table and base-shoulder contacts are excluded |
| Table | Box 1.2 m x 1.2 m, top at z = 0, friction 0.5, contact softness 5 ms | A large table: a thrown cube stays on it. The softness sets how deep a landing cube sinks in |
| Cube | 30 mm, 30 g, inertia 4.5e-6 kg m² on each axis, friction 0.5, softness 5 ms | A small plastic cube. The same values in every episode |
| Jaw contacts | The Menagerie gripper settings: torsional and rolling friction on (6 friction directions), friction 1, softness 10 ms | A pinch has small contact areas. Torsional friction stops the cube turning about the jaw axis |
| Jaw shapes | The Menagerie jaw meshes and pads, unchanged | A good grasp without extra pads (table above) |
| Friction model | Elliptic friction cones, friction-to-normal ratio 10 | Less creep in sustained grasps |
| Slip removal pass (noslip) | Off | The cube moves the same in the grasp with it on (see "Physics step") |
| Solver | Newton, tolerance 1e-10, at most 100 iterations | At 1e-8 the solver stopped too early in impacts (see "Physics step"). The same CPU time |
| Physics step | 0.5 ms | See "Physics step" |
| Joint servos | Menagerie position servos: 998 N m/rad, damping 2.7 N m s/rad, torque limit 2.94 N m | One servo per joint, jaw included. The torque limit gives a grip of about 40 N |
| Joint limits | Menagerie ranges, soft MuJoCo limits (see "Joint limits") | Written to `system.json` (`physics_config.joint_limits`) |
| Command | The 6 joint targets, float32, held for 20 ms, at least 50 mrad inside the joint limits | The stored command is exactly the applied one |

The collector adds the table, the cube, the solver settings, names for the unnamed collision shapes and the base
collision shapes in code. The fetched model files are not edited.

**The jaw.** The SO101 has one moving jaw. The fixed jaw is part of the gripper link. Contact pair 0 is any
collision shape of the gripper link against the cube. Pair 1 is the moving jaw against the cube. Pair 2 is the cube
against the table. `pair_metadata` lists the shapes of each region:

| Pair | Region | Collision shapes |
|---|---|---|
| 0 `fixed_jaw_cube` | Fixed finger | 3 tip spheres, 5 pads, the finger hull |
| 0 | Palm blocks (at the finger base) | `fixed_jaw_box1`, `fixed_jaw_box2` |
| 0 | Wrist-servo box | `gripper_collision0` |
| 1 `moving_jaw_cube` | Moving finger | 3 tip spheres, 2 pads, 2 finger hulls |
| 1 | Hinge block | `moving_jaw_box1` |

One pair code covers all regions of a jaw. The per-interval `palm_force` gives the force on the blocks. Any other
contact ends the episode at the record before it: an arm part on the table, another arm link, the arm base or the
camera mount on the cube, or the arm touching itself. A cube that leaves the table area also ends the episode.

**Reach.** The wrist-flex limit (1.66 rad, targets up to 1.61 rad) keeps the gripper vertical only up to a grasp
height of about 7 cm (a little less with the target margin; not measured again). Above that, the scripts let the
gripper tilt by up to 45 degrees; the gripper position is still exact.

## The cube inside gripper parts without collision shapes

The Menagerie gripper has collision shapes for the fingers, pads, palm blocks and servo box. Some solid parts have
none, for example the hinge of the moving jaw and the sides of the gripper body. In the first pilot a falling cube
could pass up to 6 mm into them. These intervals were stored as contact-free.

The rule now, at every record:

1. Measure how deep the cube overlaps the visual meshes of the gripper and of the moving jaw. Two ways: mesh
   surface points inside the cube, and cube surface points inside the solid mesh.
2. Subtract how deep that body's collision shapes are in the cube. In a firm grip a pad sinks about 1.3 mm into the
   cube (soft contact), and the visual mesh sinks with it.
3. If more than 1 mm remains, the episode ends at the record before. Terminal reason: `cube_visual_overlap`.

Each record stores the remaining overlap (`visual_overlap`, in metres). The meshes are closed: the inside tests along
x, y and z agree in every grid cell.

| Family | Episodes that end this way | Largest stored overlap |
|---|---|---|
| near_miss | 2 of 32 | 0.88 mm |
| push | 1 of 32 | 0.89 mm |
| jaw_fixed | 3 of 100 | 0.81 mm |
| jaw_moving | 7 of 100 (5 before the throw, while the cube rests or slides on the sloped jaw) | 0.99 mm |
| All other families | 0 | 0.47 mm (weak_closure); 0.001 mm in nominal |

In the task episodes, the cube edge meets the side of a finger next to its 8 mm wide pad. The remaining overlap
there reaches 0.9 mm. With a 0.5 mm threshold, 5 near_miss, 6 push, 21 jaw_fixed and 11 jaw_moving episodes more
would end (independent script below).

| Check on the stored data | First pilot | Now |
|---|---|---|
| Arm mesh vertices more than 1 / 2 / 3 mm inside the cube, no jaw label on either side (all jaw probes and a sample of the other families, 590 episodes) | 34 / 26 / 19 records in 8 episodes | 0 / 0 / 0 |
| Remaining overlap, measured again at every stored record with an independent script (13 cube points per edge, contacts from the stored state) | 17 episodes above 1 mm, largest 6.2 mm | 0 records above 1 mm, largest 0.99 mm |
| The collector's measure after every 0.5 ms physics step, the stored commands replayed (off_centre, near_miss, weak_closure, push, jaw_fixed, jaw_moving: 328 episodes, 111,503 intervals) | | Largest 0.996 mm. No physics step between two records goes above 1 mm |
| A finer measure at every stored record (every mesh point 0.5 mm apart, 41 instead of 11 cube points per edge; the same families and one_sided: 360 episodes, 131,063 records) | | 5 records in 4 episodes above 1 mm (jaw_moving 4, near_miss 1), largest 1.16 mm |

**Precision of the measure.** To save time, the collector keeps one mesh point per 0.5 mm cell and 11 points per
cube edge. Where either measure finds an overlap (5,822 records), the finer measure reads more by 0.04 mm (median),
0.14 mm (p95), 0.17 mm (p99) and 0.34 mm at most. So the 1 mm limit holds to about 0.2 mm. The replays give the
stored values again (largest difference 4e-8 mm). An independent measure of the checker (cube corners, edge
midpoints and face centres; mesh vertices) found one record above 1 mm in 378 episodes: 1.06 mm (near_miss, the
same record as the finer measure).

The probe throws were not changed: only 10 of 200 jaw probes end this way. The moving-jaw probes still give
5,051 single moving-jaw training windows, 5 times the target.

The check costs 17-19 % CPU time in normal grasps and jaw_fixed probes, 2 % in jaw_moving probes (single process,
12 episodes each: nominal 0.98 s against 0.83 s per episode).

## Joint limits

| Setting | Value |
|---|---|
| Joint ranges (rad) | shoulder_pan ±1.920, shoulder_lift ±1.745, elbow_flex ±1.690, wrist_flex ±1.658, wrist_roll -2.744..2.841, gripper -0.175..1.745 |
| Limit type | Soft MuJoCo limit constraint: solref 0.02 s / 1, solimp 0.9 0.95 0.001 0.5 2, margin 0 (MuJoCo defaults) |
| Servo control ranges | The joint ranges (within 3 µrad) |
| Scripted targets | At least 50 mrad inside every limit (`joint_target_margin_rad`). The inverse kinematics uses the same range |

**Why the wrist ran past its limit.** A vertical gripper above a grasp height of about 7 cm needs the wrist flex at
its limit. The inverse kinematics stopped 1 mrad inside the limit, so in 29 % of the task records the wrist target
sat at the limit. In fast lifts and retreats the wrist overshoots its target by up to 44 mrad. It then ran into the
soft limit constraint.

| | First pilot | Now |
|---|---|---|
| Records with a joint past its limit | 1,930 of 478,825 (0.40 %); task records 0.63 % | 0 of 477,632 |
| Wrist flex | 1,926 records, up to 25.3 mrad past (lift 869, retreat 892, shake 139) | Closest 3.3 mrad inside (a weak_closure wrist shake) |
| Wrist roll, shoulder lift | 3 records up to 8.3 mrad past (push); 1 record 0.1 mrad past | Closest 42 and 49 mrad inside |

The wrist shake of weak_closure can still come within a few mrad of the limit. Production (35 times more task
episodes) may have a few such records past the limit.

## Physics step

The check simulates 106 episodes of all 12 families at the chosen settings. Then it replays their recorded
commands with other settings. It compares each run with a reference: a 0.125 ms step and a tight solver (tolerance
1e-12, up to 200 iterations). Values are p95 over the episodes. "Landing" is 50 ms after a thrown cube first lands
(32 throws).

| Setting | Cube at hold end | Thrown cube after landing: position / spin | Contact events within 20 ms of the reference: fixed jaw / moving jaw / table (198 / 172 / 320 events) | Bounces that gain energy |
|---|---|---|---|---|
| **0.5 ms (chosen)** | 0.24 mm | 1.6 mm / 3.6 rad/s | 80 % / 86 % / 89 % | 0 of 73 |
| 1 ms | 0.46 mm | 4.7 mm / 7.3 rad/s | 74 % / 76 % / 87 % | 0 of 87 |
| 0.25 ms | 0.14 mm | 0.5 mm / 1.3 rad/s | 85 % / 81 % / 87 % | 0 of 61 |
| 0.5 ms, tight solver | 0.24 mm | 1.6 mm / 3.6 rad/s | 80 % / 86 % / 89 % | 0 of 73 |
| First build: 2 ms, tolerance 1e-8, table softness 10 ms (against its own reference) | 0.71 mm | 10.6 mm / 11.8 rad/s | 68 % / 68 % / 80 % | 0 of 36 here; 1 of 766 in its data |

Three settings changed after the first build (an earlier version of the collector, before the first pilot; its
data is not kept):

| Setting | First build | Now | Why |
|---|---|---|---|
| Physics step | 2 ms | 0.5 ms | At 2 ms one stored landing gained energy: 70 mJ before, 74 mJ after, spin 6 → 147 rad/s (the reference: 6 mJ, 13 rad/s). Landings and the table and fixed-jaw timing were not converged |
| Solver tolerance | 1e-8 | 1e-10 | At 1e-8 the solver stops early in impacts. At 1 ms, thrown cubes after landing were off by 6.2 mm p95 with 1e-8 and 1.5 mm with 1e-10. 1e-10 and 1e-12 give the same result |
| Cube and table softness | 10 ms | 5 ms | A cube landing at 2.4 m/s sank 9-10 mm into the table. Now it sinks at most 4.9 mm. The softness must be at least 10 physics steps: at 1 ms, a 5 ms softness gave 4.6 mm / 7.3 rad/s after landing |

By family, chosen settings against the reference (events within 20 ms, reference events in brackets):

| Family (episodes) | Fixed jaw | Moving jaw | Table |
|---|---|---|---|
| nominal (16) | 100 % (40) | 100 % (32) | 90 % (50) |
| off_centre (4) | 80 % (10) | 100 % (8) | 75 % (16) |
| one_sided (4) | 62 % (8) | 100 % (8) | 100 % (4) |
| near_miss (4) | 100 % (6) | 100 % (4) | 100 % (6) |
| weak_closure (8) | 81 % (52) | 82 % (34) | 100 % (22) |
| drop (8) | 100 % (16) | 100 % (16) | 80 % (30) |
| push (4) | 48 % (48) | 69 % (48) | 72 % (18) |
| cube_flight (32) | | | 90 % (114) |
| cube_table (8) | | | 93 % (28) |
| jaw_fixed (8) | 100 % (18) | 0 % (2) | 83 % (12) |
| jaw_moving (8) | | 95 % (20) | 94 % (18) |

**Landings in every family.** The 1.6 mm above is for thrown cubes only. Stored pilot episodes, their commands
replayed; cube position error 50 ms after the reference's first landing:

| Family | Landings | 0.5 ms: median / p95 / max | 0.25 ms: median / p95 / max |
|---|---|---|---|
| cube_flight (64 episodes) | 64 | 0.50 / 1.7 / 2.7 mm | 0.21 / 0.67 / 1.6 mm |
| cube_table (32) | 21 | 0.18 / 0.78 / 0.84 mm | 0.09 / 0.17 / 0.20 mm |
| drop (32) | 31 | 0.68 / 1.7 / 2.2 mm | 0.26 / 1.0 / 1.1 mm |
| jaw_fixed (100) | 54 | 2.3 / 33 / 300 mm | 1.0 / 8.8 / 167 mm |
| jaw_moving (100) | 60 | 10 / 37 / 70 mm | 8.7 / 38 / 65 mm |

In the jaw probes the cube first rests or slides on a jaw. That does not converge (next point), so the landing
comes at a different time and place. A smaller step does not help for jaw_moving. Do not apply 1.6 mm to these
landings.

- **Limit: sliding and sticking contacts do not converge at any step.** Examples: the closed jaws sliding along the
  cube (push), and a cube that rests on a sloped jaw near the sticking limit (jaw probes). A 0.25 ms step agrees no
  better. The replays do not stop at contacts that end an episode.
- The cube moves 0.6 mm (median) in the gripper while it is carried, 3.8 mm at most. This is the give of the soft
  jaw contacts. It is the same at every step (0.52-0.58 mm median) and, in the first build, with the slip removal
  pass on.

## Contact labels

A pair is on in a 10 ms record interval if its summed normal force is above **0.01 N** in any of the 20 physics
steps. MuJoCo `mj_contactForce` gives the normal force. MuJoCo computes the contacts and forces of a step from the
state at the start of that step. This is the contract's rule. The stored `contact_force` is the largest of the 20
values.

| Pair | Intervals on | Intervals at 0.005-0.01 N | Intervals at 0.01-0.02 N | Labels that change at 0 N | Labels that change at 0.02 N |
|---|---|---|---|---|---|
| Fixed jaw - cube | 116,147 | 10 | 15 | 32 | 15 |
| Moving jaw - cube | 111,850 | 3 | 20 | 5 | 20 |
| Cube - table | 338,949 | 8 | 29 | 25 | 29 |

Few intervals have a force near the threshold, but the histogram has no empty gap there.

**The table label while the jaws close and grip.** The closing moving jaw drags the cube up a little by friction.
The cube then rests partly on the jaws and partly on the table, and the table force can go below the threshold and
back. The table label goes off and on again during close and grip in:

| Family | Episodes | First build (2 ms) |
|---|---|---|
| nominal | 56 of 320 | 127 of 320 |
| off_centre | 9 of 32 | 16 of 32 |
| near_miss | 5 of 32 | |
| drop | 4 of 32 | 17 of 32 |
| weak_closure | 1 of 32 | |

The label rule stays. The stored per-record `phase` marks these records, so evaluation can leave them out. The
evaluation block in `system.json` does this: its `timing_exclude` leaves cube-table onsets and offsets in the close
and grip phases out of the timing statistics.

## Contact depth

The stored `contact_depth` is the deepest penetration of each pair in each interval.

| Contact | Usual depth | Deepest |
|---|---|---|
| Each jaw in a firm grip (40 N) | 1.33 mm | 2.6 mm (normal), 2.9 mm (off_centre), 3.5 mm (push) |
| Cube resting on the table | 0.01 mm | |
| Cube on the table in normal grasps | | 0.6 mm |
| Cube landing on the table | | 4.9 mm (table below) |

| Family | Landings | Landing speed, median / max | Deepest table penetration |
|---|---|---|---|
| cube_flight | 1,557 | 0.51 / 2.56 m/s | 4.7 mm |
| cube_table | 222 | 0.54 / 1.15 m/s | 1.4 mm |
| drop | 39 | 1.07 / 1.50 m/s | 2.6 mm |
| jaw_fixed | 103 | 1.04 / 1.91 m/s | 3.6 mm |
| jaw_moving | 155 | 0.67 / 2.73 m/s | 4.9 mm |

The jaw contacts keep the Menagerie softness (10 ms). In the 40 N grip each jaw sinks 1.3 mm into the cube, like a
soft pad. A landing is a table touch after an interval in which the cube touches nothing. Of the 1,193 bounces
(landings with a free cube after them), none gains energy; the smallest loss is 0.13 mJ.

## Episode families

Each episode is 6 s (601 records), except where shown. The cube start, cube yaw, start pose, timings and offsets
are random. Grasps come from inverse kinematics (damped least squares with `mj_jac`) on the 5 arm joints. The
gripper points down and the jaws line up with the cube faces. Joint moves use smooth minimum-jerk profiles. Arm
joints are limited to 3 rad/s and the jaw to 4.5 rad/s.

| Family | Group | Pilot episodes | What happens | Outcome in the pilot |
|---|---|---|---|---|
| nominal | normal | 320 | Approach, descend, close, lift 3-12 cm, hold, carry to a new place (40 %), lower, release, retreat | 320 lifted and held |
| off_centre | difficult | 32 | Grasp 4-12 mm off centre along the jaw width, ±3 mm across, up to 12 mm high | 32 held, hold slip up to 0.41 mm |
| one_sided | difficult | 32 | One jaw only: fixed-jaw side push (11), moving-jaw swat (12), fingertip tap on the top (9) | 32 not lifted (as planned) |
| near_miss | difficult | 32 | Jaws close 15-35 mm off centre, then lift, lower, release | 29 missed, 3 caught an edge; 2 ended at a gripper overlap, 2 with the arm on the table |
| weak_closure | difficult | 32 | Jaw target set at the measured touch angle -2..+4 mrad; fast lift; shake (5); slow jaw opening (7) | 21 held, 9 slipped out at lift-off, 2 slipped and fell from the air; 1 ended with the arm on the table |
| drop | difficult | 32 | Grasp, lift 5-15 cm, open the jaw in the air | 32 lifted, then released in the air as planned; 1 ended with the arm touching itself |
| push | difficult | 32 | Closed jaws push the cube 6-12 cm across the table | 32 pushed; 1 ended at a gripper overlap |
| free_arm | probe | 100 | Random arm motion; the cube rests out of reach | |
| cube_flight | probe | 650 (1 s) | Cube thrown up at 1-2.2 m/s with spin; flight, landing, tumbling until it rests | 11 ended early (9 left the table, 2 hit an arm link) |
| cube_table | probe | 100 (1 s) | Cube released just above the table with sliding speed and spin | |
| jaw_fixed | probe | 100 (2 s) | Fixed jaw face held up in the air; the cube lands on it, rests or slides, then a fast joint turn tips the face over | 53 cubes fell to the table; 41 stayed on the gripper (39 against a palm block); 6 ended early (3 on another arm link, 3 at a gripper overlap) |
| jaw_moving | probe | 100 (2 s) | The same with the moving jaw | 58 cubes fell to the table; 4 stayed on the gripper, against a palm block; 3 in the air at the end; 35 ended early (23 at the arm base, 7 at a gripper overlap, 5 on another arm link) |

Most episodes run to the end: 1,503 of 1,562. The other 59 ended early: 23 with the cube on the arm base, 13 at a
gripper overlap, 10 with the cube on another arm link, 9 with the cube leaving the table area, 3 with the arm on
the table and 1 with the arm touching itself. There were no numerical failures. No recorded interval holds an
unsupported contact, even below the threshold.

**The jaw probes.** The throw does not always clear the gripper. In 41 of 100 jaw_fixed episodes the cube stays on
the gripper to the end. In jaw_moving the throw is mostly a fast backward turn of the shoulder. It swings the
gripper over the arm base, and in 23 episodes the cube then hits the base and the episode ends. The jaw contacts
before the throw are kept. In the first pilot two other throws were tried on the same 200 probes and were worse
for jaw_moving:

| Throw (first pilot) | Ends at the arm base | Ends on another arm link | Cube on the table at the end | CPU time per episode |
|---|---|---|---|---|
| **Used: as above** | 27 | 11 | 55 | 0.63 s |
| Rejected if the jaw face comes within 12 cm of the pan axis | 0 | 39 | 26 | 3.05 s |
| Stopped where the jaw face comes within 10 cm of the pan axis | 17 | 60 | 6 | 0.63 s |

**Extra arrays.** Training does not read these:
- `phase`: scripted phase of each record.
- `cube_in_grasp`: cube centre in the gripper frame.
- `visual_overlap`: remaining overlap of the cube with the gripper and moving-jaw meshes at each record (m).
- `contact_force`, `contact_impulse`, `contact_depth`: per pair and interval.
- `palm_force`: the force of the palm blocks and the servo box (pair 0) and of the moving jaw's hinge block (pair 1).
- `outcome`, `max_lift_m`, `hold_slip_mm`, `carry_slip_mm`: task outcome of each episode.
- `variant`: the sub-type of the episode.
- `unsupported_max_n`: the largest force below the threshold from a contact that would end the episode.
- `family_index`, `plan_tries`: episode bookkeeping.

## Pilot coverage

Windows on the 20 ms training grid, counted by `collect.validate`. "Core grid": every fifth record, where
`train.py` fits and trains the core.

| Count | Training split: all / core grid | Validation split: all / core grid | Target (all) |
|---|---|---|---|
| Contact-free arm windows | 154,624 / 31,014 | 17,062 / 3,417 | 10,000 |
| Contact-free cube windows | 11,637 / 2,745 | 1,318 / 311 | 10,000 |
| Fixed jaw only | 6,499 | 808 | 1,000 |
| Moving jaw only | 5,119 | 499 | 1,000 |
| Cube-table only | 142,987 | 15,744 | 1,000 |
| Two or more pairs | 48,177 | 5,231 | |

The splits are 90/10 by family. The test split is a separate campaign with its own seed (see "Campaigns and seeds").

**Palm blocks.** Pair 0 includes the palm blocks and the servo box of the gripper, and pair 1 the hinge block of
the moving jaw. In the jaw probes the cube can slide onto them. Of the single-jaw training windows, these touch such
a block: fixed jaw 1,385 of 6,499 (1,212 in jaw_fixed, 170 in jaw_moving), moving jaw 447 of 5,119. The windows on
the fingers alone (5,114 and 4,672) still meet the target. No task episode touches a block.

**Why there are many cube throws.** A contact-free cube window needs the cube in the air. The tasks rarely give
one: a drop gives about 4, the other tasks almost none. A throw gives about 17. 650 short throws reach the target.

**Family shares.** The contract lets the coverage targets win over its shares (about 50 / 30 / 20 %):

| Group | Episodes | Share of records |
|---|---|---|
| Normal | 320 | 40.3 % |
| Difficult | 192 | 23.7 % |
| Probe | 1,050 | 36.1 % |

**Rate check.** The checker compares the pose-derived 20 ms average rates with Simpson's rule over the recorded
rates (both ends and the 10 ms midpoint). Relative RMS: arm 0.09-0.13, cube velocity 0.015-0.05, cube angular
velocity 0.03-0.10. The warning level is 0.5. In free flight, the stored cube angular velocity matches the rate from
the quaternions to a relative RMS of 7e-6 (24,625 contact-free cube_flight intervals; float32 storage). A body-frame
value would be off by 137 %.

## Campaigns and seeds

| Campaign | Master seed | Task episodes | Probe episodes | All | Splits |
|---|---|---|---|---|---|
| pilot | 20261008 | 512 | 1,050 | 1,562 | 0 / 1 by family (90 / 10) |
| production | 20261021 | 18,000 (16,197 train + 1,803 validation) | 10,500 (10 x pilot; 9,450 + 1,050) | 28,500 | 0 / 1 by family (90 / 10) |
| test (sealed) | 20261034 | 2,000 | 1,050 (1 x pilot) | 3,050 | all 2 |
| convergence check | 20261085 | | | 106 | |

- The task families keep the pilot ratio: nominal 11,250 and each difficult family 1,125 in production; 1,250 and
  125 in the test.
- The production row of the contract says 18,000 task episodes cut 90/10 inside each family, about 16,200 train +
  1,800 validation. `export.assign_splits` gives 16,197 + 1,803 (nominal 10,125 + 1,125; each difficult family
  1,012 + 113), as in the G1 collector. (The plan's first number, 16,000 + 2,000, cannot come from the 90/10 rule.)
- Each episode seed comes from the master seed, the family and the episode number. A run therefore does not depend
  on the number of workers or the shard size.
- Before it simulates, the collector stops if two master seeds are equal, or if an episode seed of the run is also
  an episode seed of another campaign or of the convergence check. Checked: no overlap among the 1,562 + 28,500 +
  3,050 + 106 episode seeds.

## Speed and the production run

CPU time per episode on this 16-core workstation. Three pilot runs wrote the same file (hash `b7b55a61…`) at
different machine loads (load average at the end of the run; 6 of it from the run's own workers). The stored
`manifest.json` (`throughput`) is from the last one:

| Family | CPU s per episode, stored manifest (load 14) | Saved 20 ms steps per CPU s, stored manifest | CPU s per episode, other runs (load 8 / load 21) | CPU s per episode, first pilot (load 4-6) |
|---|---|---|---|---|
| nominal | 1.67 | 179 | 1.31 / 2.05 | 0.93 |
| difficult (6 families) | 1.17-3.13 | 95-238 | 0.86-1.98 / 1.57-3.53 | 0.77-1.31 |
| Task mix (all 7 task families) | 1.72 | 173 | 1.26 / 2.07 | 0.93 |
| free_arm / cube_flight / cube_table | 0.96 / 0.11 / 0.15 | 312 / 450 / 337 | 0.67 / 0.09 / 0.11 and 1.27 / 0.17 / 0.21 | 0.61 / 0.08 / 0.10 |
| jaw_fixed / jaw_moving | 0.43 / 1.18 | 227 / 73 | 0.36 / 0.92 and 0.57 / 1.65 | 0.26 / 0.64 |
| All 1,562 pilot episodes | 1,225 s in all (1.28 episodes per CPU s) | 194 | 911 s / 1,535 s | 689 s |

Machine load changes the speed by almost a factor of 2. In a single-process test, the overlap check added 17-19 % in
grasps. These runs do not separate the other causes. drop is the slowest family: with the joint-target margin, its
high lifts need more re-planning (4.1 plans per episode, 2.7 before).

Before the production run, estimates used the per-family CPU time of each pilot run and the campaign counts. The
production and test campaigns then ran on newton on 2026-10-09 with 24 workers. The measured values are in the data
manifests (`artifacts/contact_nrd/production/so101_grasp/manifest.json` and
`artifacts/contact_nrd/test_sealed/so101_grasp/manifest.json`, `throughput`; verified):

| Run | Episodes | Estimate before the run: CPU time, stored manifest (load 8 / load 21); time on 32 workers | Measured on newton: worker CPU time; time on 24 workers; load at the end | Data (`unified_data.npz`) | RAM for the merge |
|---|---|---|---|---|---|
| production | 28,500 | about 34,000 s (25,000 / 42,000 s); about 18 min (13 / 22 min) | 21,589 s; 922 s (15.4 min); 24.5 | 3.22 GB (measured) | about 6.5 GB (estimate; not measured) |
| test | 3,050 | about 3,800 s (2,800 / 4,600 s); about 2 min (1.5 / 2.5 min) | 2,318 s; 113 s; 19.9 | 0.34 GB (measured) | about 0.7 GB (estimate; not measured) |

- Both campaigns ran 1.32 episodes per CPU second on newton (`episodes_per_cpu_s`).
- Nominal episodes: 1.06 CPU s per episode on newton (production; 11,980 s for 11,250 episodes), against 1.31 to
  2.05 s on this workstation (table above). The family mix and the machine load differ, so this is not a clean
  measure of the speed of one core.

newton has 32 cores, 62 GB RAM and the conda env `~/anaconda3/envs/nrdsim` (mujoco 3.15.0, numpy 2.5.3,
scipy 1.18.1, checked 2026-10-09). The shards go to a temporary folder next to the output (about the data size)
and are removed after the merge.

**GPU memory (NRD side).** The loader keeps every episode padded to 601 records in float64 on the GPU, so a 1 s
throw costs as much as a 6 s grasp. First pilot: loading needed 0.44 MB per episode; `train --smoke` needed
8.3 GB plus 0.22 MB per episode. This pilot: `train --smoke` peaked at 13.9 GB (the process's GPU memory, sampled
every second, PyTorch cache included; the cause of the difference is not checked). For production, loading alone
is about 12.5 GB. A full training run uses larger batches; its peak is not measured yet.

## Reproduce

```bash
# 1. Fetch and check the SO101 model (data only; nothing in it is run)
~/miniconda3/envs/nrdsim/bin/python -I contact_nrd/so101_grasp/fetch_sources.py
# 2. Pilot (6 workers, about 3 min; 5 min on a busy machine) and the contract check
PYTHONPATH=src ~/miniconda3/envs/nrdsim/bin/python -P -m nedm.contact_nrd.collect.so101_grasp collect \
    --campaign pilot --out artifacts/contact_nrd/collect_pilot/so101_grasp --workers 6
PYTHONPATH=src ~/miniconda3/envs/nrdsim/bin/python -m nedm.contact_nrd.collect.validate \
    artifacts/contact_nrd/collect_pilot/so101_grasp --output artifacts/contact_nrd/collect_pilot/so101_grasp/validation.json
# 3. Physics-step check (106 episodes of all families, 6 workers, about 3 min), outcome summary, videos
PYTHONPATH=src ~/miniconda3/envs/nrdsim/bin/python -P -m nedm.contact_nrd.collect.so101_grasp convergence \
    --out artifacts/contact_nrd/collect_pilot/so101_grasp/convergence.json --workers 6
PYTHONPATH=src ~/miniconda3/envs/nrdsim/bin/python -P -m nedm.contact_nrd.collect.so101_grasp report \
    artifacts/contact_nrd/collect_pilot/so101_grasp
MUJOCO_GL=egl PYTHONPATH=src ~/miniconda3/envs/nrdsim/bin/python -P -m nedm.contact_nrd.collect.so101_grasp render \
    artifacts/contact_nrd/collect_pilot/so101_grasp --episodes 2 442 448 504 613 1362 1462 1495
# 4. NRD training check on the GPU (env nedm, about 1 min)
PYTHONPATH=src ~/miniconda3/envs/nedm/bin/python -P -m nedm.contact_nrd.train \
    --data artifacts/contact_nrd/collect_pilot/so101_grasp --output /tmp/so101_grasp_smoke --smoke
```

Production and the sealed test, on newton, from a checkout of this branch (`DIR` is the output folder; the reported
runs used 24 workers, as here):

```bash
~/anaconda3/envs/nrdsim/bin/python -I contact_nrd/so101_grasp/fetch_sources.py
PYTHONPATH=src ~/anaconda3/envs/nrdsim/bin/python -P -m nedm.contact_nrd.collect.so101_grasp collect \
    --campaign production --workers 24 --out DIR/so101_grasp_production
PYTHONPATH=src ~/anaconda3/envs/nrdsim/bin/python -m nedm.contact_nrd.collect.validate \
    DIR/so101_grasp_production --output DIR/so101_grasp_production/validation.json
PYTHONPATH=src ~/anaconda3/envs/nrdsim/bin/python -P -m nedm.contact_nrd.collect.so101_grasp collect \
    --campaign test --workers 24 --out DIR/so101_grasp_test
PYTHONPATH=src ~/anaconda3/envs/nrdsim/bin/python -m nedm.contact_nrd.collect.validate \
    DIR/so101_grasp_test --output DIR/so101_grasp_test/validation.json
```

- **What `collect` does:** each pool task simulates 16 episodes (`--chunk`) and writes them as one shard file. The
  main process merges the shards in episode order and writes the folder with `export.write_dataset`
  (`system.json` with the evaluation block, `unified_data.npz`, `manifest.json`). It prints the progress and, at the
  end, the throughput. `manifest.json` keeps the throughput per family.
- **Test campaign check:** a test folder has only split 2 by design. The checker skips its "validation episode in
  a family with select: true" rule for such a folder (`src/nedm/contact_nrd/collect/validate.py`, `da704c34…`).
  Verified: the sealed test folder's `validation.json` has no fails, and on 2026-10-09 `collect.validate` on it
  exited 0 with no fails and no warnings (`artifacts/contact_nrd/results/checker_rerun_20261009/`). An earlier
  checker version (before this change) failed test folders with that one message.
- **Smoke runs:** `--scale S` multiplies all counts of the campaign, `--probe-scale` the probe counts, `--families`
  picks families.
- **Environment:** conda env `nrdsim` ([requirements](../../configs/contact_nrd/collection/requirements.txt)). The
  collector needs numpy, scipy (the overlap grids) and mujoco; rendering also needs imageio-ffmpeg.
- **Settings:** all of them are in
  [`configs/contact_nrd/collection/so101_grasp.json`](../../configs/contact_nrd/collection/so101_grasp.json),
  including the campaigns and the episode counts of the physics-step check.
- **Flags:** `-P` keeps the working folder off the import path.
- **Provenance:** `manifest.json` holds the campaign, the master seed, the episode counts, the sha256 of the
  collector's source files (`collector_source_sha256`), the config, the fetched model files and the built scene,
  besides the git commit.
- **First pilot:** kept in `artifacts/contact_nrd/collect_pilot/so101_grasp_round1/` for the before/after numbers.

## Limits

- **Grip force.** It is about 40 N, set by the Menagerie servo torque limit. Firm grasps never slip. Slips and
  in-air drops come only from the weak-closure family.
- **Sliding and sticking contacts.** Their timing does not converge at any step (see "Physics step"). Landings after
  the cube slid on a jaw are not converged either.
- **Gripper parts without collision shapes.** Overlaps up to about 1 mm beyond the pad penetration stay in the
  data. The measure is accurate to about 0.2 mm: the stored values reach 0.99 mm, a finer measure 1.16 mm. The check
  runs at the records only. Measured after every physics step with the collector's measure, the overlap stays below
  1 mm too (largest 0.996 mm, 328 episodes).
- **The table label while gripping.** It can go off and on before the lift (see "Contact labels").
- **Wrist shake near the limit.** It can come within a few mrad of the wrist limit (see "Joint limits").
- **Padding.** The probe episodes are shorter than 6 s, but each is padded to 601 records.
- **Jaw-probe throws.** 23 of 100 jaw_moving probes end when the cube hits the arm base, and 41 of 100 jaw_fixed
  probes end with the cube on the gripper (see "The jaw probes").
- **Script bias.** Steps that do not fit in 6 s are left out. They are therefore less frequent than drawn:
  - moves while holding: 129 of 320 (40 %, drawn 50 %);
  - slow jaw openings: 7 of 32 (drawn 50 %);
  - wrist shakes: 5 of 32 (drawn 60 %).

  Plans with an unreachable pose are drawn again: 1.0-4.1 tries per episode on average (drop 4.1, jaw_moving 3.1).
