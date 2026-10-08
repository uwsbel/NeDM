# Pool

Two equal balls (radius 28.575 mm, mass 0.17 kg) are on a 2.54 m × 1.27 m table. Ball A starts at (−0.635, 0) m. It
is shot without spin at ball B, which is at rest at (0, 0). Both balls roll, slide and spin. They hit each other and
the cushions. Design, training and commands: [`../README.md`](../README.md).

| Item | Value |
|---|---|
| Moving bodies | Ball A, ball B, state 9 each: position 3, velocity 3, spin 3 |
| Fixed bodies | Four cushion planes |
| Contact pairs | 9: A–B, and each ball with each cushion. No shot reaches the cushion behind ball A (x = −1.27 m), so those two pairs are never on |
| Speeds the model predicts | vx, vy (averaged over the 20 ms step), spin wx, wy, wz (as recorded). The model keeps z and vz at their start values |
| Pose from integration | x, y of each ball |
| Record step / model step | 1 ms / 20 ms (`system.json` lists 10 ms; training and test use 20 ms) |
| Episode | 2.5 s from the shot |
| Error measured | Ball B position at 2 s |

## Data

| Split | Episodes | Source campaign (Chrono, step 12.5 µs) | Size |
|---|---|---|---|
| Train / validation | 19,200 / 2,400 | `pool_span_v1_20261002`, seed 202610021: a 30 × 40 grid of shot speed (1.5 to 3.0 m/s) and aim (up to ±4.68° from the line A–B, so the cut angle is 65° or less), each shot at a random point in its cell, 16 train + 2 validation shots per cell | 4.86 GB |
| Test | 4,800 | `pool_fresh_v5_20261004`, seed 202610055: the same grid, 4 new shots per cell | 0.97 GB |

The training file also holds the source campaign's own 2,400 test shots (split 2). Training does not use them.

## Results (test, median / p95)

| Checkpoint | Ball B at 2 s | Path RMS of B up to 2 s | Shots with all contact events right |
|---|---|---|---|
| `models/seed61.pt` | 0.69 / 2.41 mm | 0.41 / 1.31 mm | 4,787 / 4,800 |
| `models/seed62.pt` | 0.64 / 2.52 mm | 0.38 / 1.44 mm | 4,783 / 4,800 |

A contact event is the start of a contact of one pair. A shot is right when each pair has the recorded number of
events in the full 2.5 s.
