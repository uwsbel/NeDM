# SO101 push-T

A 5-joint SO101 arm (jaw locked) pushes a T-shaped block (bar 100 × 25 mm, stem 25 × 75 mm, 30 mm high, 78.75 g) on a
table. A joint PD controller tracks the joint position command q_cmd, which changes every 20 ms. Design, training and
commands: [`../README.md`](../README.md).

| Item | Value |
|---|---|
| Moving bodies | Arm, state 10: joint angles 5, joint speeds 5; action q_cmd 5. T, state 13: position 3, quaternion 4, velocity 3, angular velocity 3 |
| Fixed body | Table (friction with the T is part of the core) |
| Contact pairs | Arm–T |
| Speeds the model predicts | Arm joint speeds; T vx, vy, yaw rate. All are averages over the 20 ms step. The model keeps T z, roll and pitch at their start values |
| Pose from integration | Joint angles; T x, y and yaw (quaternion, normalised after each step) |
| Record step / model step | 10 ms / 20 ms |
| Episode | 4.0 s. The T starts at the same pose in all episodes; the arm starts from a random pose. Then the arm makes one push (linear or rotating), two pushes, a near miss or a free motion away from the T |
| Error measured | T position and T angle at 4 s |

## Data

| Split | Episodes | Source campaign (Chrono, step 83.3 µs) | Size |
|---|---|---|---|
| Train / validation | 51,200 / 6,400 | `full_v1`, seed 20261100: scripted episodes of five types (linear push, rotating push, two pushes, near miss, free motion), split 80/10/10 inside each type | 3.12 GB |
| Test | 1,000 | `fresh_v1`, seed 20261200: new episodes with the same types and mix | 49 MB |

The training file also holds the campaign's own 6,399 test episodes (split 2). Training does not use them. The arm
state has no end-effector channels; the npz also holds three arrays that training does not read (`contacts_link`,
`t_table`, `scenario`).

## Results (test, median / p95)

| Checkpoint | T position at 4 s | T angle at 4 s | T path RMS up to 4 s | Episodes with all contact events right |
|---|---|---|---|---|
| `models/seed61.pt` | 0.43 / 2.70 mm | 0.23 / 1.61° | 0.30 / 1.63 mm | 708 / 1,000 |
| `models/seed62.pt` | 0.37 / 3.07 mm | 0.20 / 1.61° | 0.27 / 1.71 mm | 697 / 1,000 |

- In 181 of the 1,000 test episodes the T does not move (it moves less than 5 mm and turns less than 0.05 rad). These
  are the 100 near misses, the 80 free motions and one short push. On the 819 other episodes the position error at
  4 s is 0.56 / 2.96 mm (seed 61) and 0.51 / 3.26 mm (seed 62).
- The largest error (44 to 55 mm) is in a near miss: the model starts a contact that does not occur.
- An episode has all contact events right when the model starts the arm–T contact as many times as the record.
