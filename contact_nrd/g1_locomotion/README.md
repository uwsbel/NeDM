# G1 locomotion

A frozen walking policy (HOMIE) drives the 27-joint Unitree G1 humanoid on a flat floor in MuJoCo. Every 20 ms it
sets the 12 leg joint targets from a velocity command (forward, sideways, turn) and a height command. Design, training
and commands: [`../README.md`](../README.md).

| Item | Value |
|---|---|
| Moving body | Robot, state 67: base position 3, base quaternion 4, joint angles 27, base velocity 3, base angular velocity 3, joint rates 27; action: 27 joint position targets (12 leg targets of the policy, 15 upper-body targets) |
| Fixed body | Ground plane at z = 0 |
| Contact pairs | Left foot–ground, right foot–ground. A pair is on in a 10 ms record step if its summed normal force is above 1 N in any physics step of it |
| Speeds the model predicts | Base velocity, base angular velocity and joint rates, all as averages over the 20 ms step |
| Pose from integration | Base position, base quaternion, joint angles |
| Record step / model step | 10 ms / 20 ms (physics step 1 ms, control step 20 ms) |
| Episode | 10 s (1,001 records) for the task families; drop probes 1.2 s, landing probes 1.5 s; shorter if the robot touches the ground with another part, touches itself or the base drops below 0.3 m |
| Families | `nominal` and `difficult` (task families, used for checkpoint selection); `probe_drop` and `probe_landing` (the robot is released above the ground from a walking state) |
| Error measured | Rollouts from the recorded 4-state history (a window every 0.5 s from 0.1 s, plus one that ends at the last record): root position, root orientation and joint RMS at 1 s, contact events, falls |

## Data

| Split | Episodes | Source campaign (MuJoCo 3.15.0, step 1 ms) | Size |
|---|---|---|---|
| Train / validation | 30,685 / 3,413 | `production`, master seed 731948205: task 16,200 / 1,800 (nominal 10,125 / 1,125, difficult 6,075 / 675), probes 14,485 / 1,613 (drop 11,808 / 1,313, landing 2,677 / 300), split 90/10 inside each family; 54.4 h of records | 19.3 GB |
| Test | 3,608 | `test`, master seed 586120437: task 2,000 (nominal 1,250, difficult 750), probes 1,608 (drop 1,310, landing 298) | 2.0 GB |

The policy is `deploy.onnx` of [OpenHomie](https://github.com/InternRobotics/OpenHomie) at commit `33071f79`.
OpenHomie has its own license; its files are not in this repository or on the Hub. The `manifest.json`
next to each data file gives the collector settings, seeds, package versions and model-file hashes.

## Training

- **Own config.** `configs/contact_nrd/train_g1_locomotion.json` is `train.json` with `"ridge": 0.1` (the ridge term
  of the core's linear start fit). Give it with `--config`. With the shared value 1e-6 (two runs, not released) the
  mean validation score of seeds 61 and 62 is 6.08 against 4.71 (lower is better), and the test gives 84.6 / 83.5 mm
  at 1 s (root position p95).
- **Selection by the task score** of the `evaluation` block in `system.json` (see `configs/contact_nrd/evaluation/`)
  instead of the free-rollout score, on up to 1,500 validation episodes of the families with `select: true`. In both
  runs the best check is the last one (120,000 co-training updates).
- **Data items.** The data carry `channel_mask` and `family`: [`../collection_contract.md`](../collection_contract.md).

## Results (test, task families: 2,000 episodes, median / p95, seed 61 | seed 62)

| Rollout | Root position (mm) | Root orientation (°) | Joint RMS (mrad) |
|---|---|---|---|
| One step (20 ms) | 0.05 / 0.42 \| 0.05 / 0.43 | 0.03 / 0.24 \| 0.03 / 0.25 | 0.39 / 2.7 \| 0.39 / 2.7 |
| Windows, 1 s | 6.9 / 59.8 \| 7.2 / 60.8 | 0.68 / 6.8 \| 0.68 / 6.7 | 2.5 / 18.1 \| 2.6 / 18.2 |
| Windows, 5 s | 45.6 / 488 \| 46.4 / 450 | 1.6 / 17.7 \| 1.6 / 16.5 | 2.8 / 19.9 \| 2.8 / 19.3 |
| Whole episode from the first state, at its end (10 s for a full episode) | 92.3 / 1,063 \| 93.4 / 934 | 2.5 / 29.0 \| 2.5 / 29.7 | 3.1 / 20.8 \| 3.1 / 21.2 |

- The `nominal` family alone (1,250 episodes): 1 s windows, p95 16.9 / 17.6 mm and 1.8 / 1.8°.
- Contacts (task windows): wrong events (missed and false) per recorded event 0.100 / 0.097. A predicted event
  matches a recorded one if they are at most 40 ms apart. The timing p95 of matched events is 20 ms (one model step)
  in every rollout mode, pair and family that has matched events.
- Falls (base below 0.35 m or tilted more than 45°): the whole-episode rollouts predict no false fall (0 / 0 of
  3,608) and find 12 / 23 of the 36 recorded falls (all in drop probes). No rollout is non-finite.

Source: `results/sealed_test_seed61_seed62.{json,out}` on the Hub (`test.py` of this repository, one RTX 5090, about
3.5 min). The first test on an MI350X gives the same numbers at this precision (summary statistics within 5e-4
relative). The command (repository root, after `download --case g1_locomotion --part test models`):

```bash
PYTHONPATH=src python -m nedm.contact_nrd.test --data artifacts/contact_nrd/g1_locomotion/test \
    --model seed61=artifacts/contact_nrd/g1_locomotion/models/seed61.pt \
    seed62=artifacts/contact_nrd/g1_locomotion/models/seed62.pt \
    --output sealed_test_seed61_seed62.json > sealed_test_seed61_seed62.out
```

The footstep policy in [`../../g1_footstep_rl/`](../../g1_footstep_rl/README.md) trains inside these two checkpoints.
