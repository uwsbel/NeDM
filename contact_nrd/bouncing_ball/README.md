# Bouncing ball

One ball (radius 0.1 m, mass 1 kg) starts at x = 0, z = 1 m and moves in the x–z plane. Its start speed is vx 4 to
7 m/s and vz −10.5 to −9 m/s (down), with no spin. It bounces once on the floor, hits a wall at x = 5 m once and flies
back. Design, training and commands: [`../README.md`](../README.md).

| Item | Value |
|---|---|
| Moving body | Ball, state 9: position 3, velocity 3, spin 3 |
| Fixed bodies | Floor (z = 0), wall (x = 5 m) |
| Contact pairs | Ball–floor, ball–wall |
| Speeds the model predicts | vx, vz (averaged over the 20 ms step), spin wy (as recorded). The model keeps y, vy, wx, wz at their start values |
| Pose from integration | x, z |
| Record step / model step | 1 ms / 20 ms (`system.json` lists 10 ms; training and test use 20 ms) |
| Episode | From the launch to 0.02 s before the second floor contact, 1.9 to 2.2 s |
| Error measured | Ball position at 1.7 s, and its RMS over the path up to 1.7 s |

## Data

| Split | Episodes | Source campaign (Chrono, step 0.125 ms) | Size |
|---|---|---|---|
| Train / validation | 5,400 / 900 | `ball_span_v1_20260930`, seed 202609301: a 30 × 15 grid of start speeds (vx, vz), 12 train + 2 validation shots per cell | 598 MB |
| Test | 1,800 | `ball_fresh_v5_20261004`, seed 1321133355: the same grid, 4 new shots per cell | 149 MB |

The training file also holds the source campaign's own 900 test shots (split 2). Training does not use them.

## Results (test, median / p95)

| Checkpoint | Position at 1.7 s | Path RMS up to 1.7 s | Shots with all contact events right |
|---|---|---|---|
| `models/seed61.pt` | 0.90 / 2.09 mm | 0.60 / 1.08 mm | 1,800 / 1,800 |
| `models/seed62.pt` | 0.90 / 2.09 mm | 0.61 / 1.10 mm | 1,800 / 1,800 |

A contact event is the start of a contact of one pair. A shot is right when the model starts as many floor contacts
and as many wall contacts as the record, over the whole shot. The check compares the counts, not the times. Each
recorded shot has one floor contact and one wall contact.
