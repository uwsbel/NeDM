# Gravity arms, corrected checkpoints: INTERIM at two seeds

**2026-09-08.** First results computed from `last.pt` (epoch 80, uniform across all
runs) after [`go2-checkpoint-selection-bug.md`](go2-checkpoint-selection-bug.md).
**Interim: two seeds per arm, which cannot certify any contrast.** Recorded now so
the corrected numbers are on the record beside the buggy ones, not to be cited as a
result.

## The correction moved the numbers, and not in one direction

```
  arm    best_val epoch   OLD rate   EP80 rate   change
  A_s1        26            54.1%      57.8%      +3.7
  A_s2         1            49.6%      40.9%      -8.8
  B_s1        73            52.4%      55.8%      +3.4
  B_s2         1            17.4%      45.9%     +28.5
  C_s1        26            33.2%      28.9%      -4.3
  C_s2        60            56.0%      37.1%     -18.9
```

**B_s2 moves +28.5 points** when its epoch-1 surrogate is replaced by epoch 80 --
the single clearest confirmation that the old numbers were reading checkpoint
maturity. But **the change is not monotone in epoch**: A_s2 and C_s2 both got
*worse* with a more-trained surrogate. A better surrogate is not automatically a
better teacher, which is the same shape as the earlier v6 result where the
conditioned surrogate produced a worse policy than the cruder one.

## Corrected comparison, seed as the unit

```
  arm    n   mean%    sd     per-seed
  A      2   49.3    12.0    57.8  40.9
  B      2   50.8     7.0    55.8  45.9
  C      2   33.0     5.8    28.9  37.1

  contrast   delta        p     floor    reading
  B - A      + 1.5    1.0000   0.3333    not separated
  C - A      -16.3    0.3333   0.3333    at the floor: maximally lopsided, uncertifiable
  C - B      -17.8    0.3333   0.3333    at the floor: maximally lopsided, uncertifiable
```

**Read the floor column before the p column.** C sits below A and B on *every* seed --
the permutation test is as lopsided as it can be -- and 0.3333 is still the smallest p
two seeds per arm can produce. The design, not the data, is what fails here.

## UPDATE, same day: a third seed of arm A broke the pattern

Arm A's third seed scored **29.9%** -- below *both* C seeds. One additional seed, and:

```
              n=2 (A: s1,s2)          n=3 (A: s1,s2,s3)
  arm A       49.3  sd 12.0           42.8  sd 14.1   (57.8  40.9  29.9)
  C - A      -16.3  p at floor        -9.8  p 0.4000
  B - A       +1.5                    +8.0  p 0.5000
```

**A's range is now 29.9-57.8 and contains C's range (28.9-37.1) entirely.** The
"C sits below everything on every seed" reading below was an artifact of *which two
seeds* arm A happened to have, and it did not survive the third.

This is the clearest demonstration available of why the two-seed design was
unreportable: the pattern was maximally lopsided, perfectly consistent across every
cell measured, and **wrong**. Nothing about the data at n=2 signalled that -- only the
floor did, which is why the floor is printed beside every contrast.

**Arm A's between-seed spread is 27.9 points at matched epoch 80.** That is the real
error bar on any single-seed claim about these arms.

## What the pattern suggests, and why it is not yet a finding

- **B - A is +1.5 points.** Real gravity channels bought essentially nothing over the
  baseline that has none.
- **C is 16-18 points below both.** C is the control whose gravity channels are
  permuted across episodes and carry no tilt information.

Taken at face value that reads: **three extra input channels HURT when they carry
noise and are NEUTRAL when they carry the true signal** -- so the model is not
extracting usable tilt information from them, merely tolerating them. That would be a
real, if negative, result about the arm design, and it is the opposite of what B was
built to show.

**It is not established.** Two seeds, p at the floor, and the earlier corrected/uncorrected
comparison shows single arms moving by up to 28 points from a checkpoint change alone.
A/B/C are being taken to four seeds (B_s3/B_s4, C_s3/C_s4 configs committed, datasets
staging) which is the first count at which any of this can reach p<0.05.

## val_loss is NOT comparable across these arms

```
  run     val_loss@80    state dim   note
  A_s1    5.652e-04       34         baseline, no gravity channels
  A_s2    5.699e-04       34
  B_s1    5.300e-04       39         + 3 real gravity channels
  B_s2    5.586e-04       39
  C_s1    5.272e-04       39         + 3 permuted gravity channels
  C_s2    5.435e-04       39
  D_s1    1.598e-03       43         + contact channels: a DIFFERENT problem
```

Two traps here, both of which would produce a confident wrong reading:

1. **B and C sit below A**, but they predict three extra channels that are *constant
   within an episode* and therefore nearly free to predict. A mean-over-channels loss
   falls when you add easy channels. That is arithmetic, not a better model.
2. **D looks 3x worse.** It predicts 43 channels including four contact booleans. It is
   not solving the same prediction problem, and its loss is not on the same scale as
   anything else in the table.

`checkpoint_metric = val_loss` is still correct, because selection happens **within a
run** where the channel set is fixed. **Never rank arms by it.** The Chrono completion
rate is the cross-arm metric, and it is the only one.

## Scope

Rigid terrain, 536 val episodes, `--target-dw 4.0` (all seven fine-tunes stopped within
0.007 of it, over 281-456 updates), fine-tune `--seed 0` throughout. Scoring distributed
across the fleet with the host recorded per file; the measured cross-machine effect is
0.2 points on the rate, and a3's pychrono build was not among the two probed.
