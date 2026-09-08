# Every surrogate was selected by a noise metric, and three of seven are epoch-1 models

**2026-09-08.** Root cause of the "seed swing" reported earlier the same day. The
observation there was real; the attribution was wrong.

## What is actually happening

`checkpoint_metric` was `rollout_sel`. That metric is dominated by noise at
single-epoch granularity:

```
  run     val_loss                          rollout_sel
          argmin  median|lag-1 jump|        argmin  median|lag-1 jump|
  A_s1     e79          3.8%                 e26         27.2%
  A_s2     e79          3.6%                 e1          26.7%
  B_s1     e75          3.5%                 e73         35.1%
  B_s2     e79          4.0%                 e1          36.8%
  C_s1     e75          3.4%                 e26         38.0%
  C_s2     e79          2.7%                 e60         46.1%
  D_s1     e77          5.8%                 e1          31.0%
```

**`rollout_sel` moves 27-46% between ADJACENT epochs, with no trend.** `val_loss`
moves 3-6% and descends smoothly to a minimum at epoch 75-79 in every single run.
So `best_val.pt` is not the best checkpoint; it is **whichever epoch drew a lucky
number.**

```
  best_val.pt epoch:   A_s1 26   A_s2 1    B_s1 73   B_s2 1
                       C_s1 26   C_s2 60   D_s1 1
  last.pt epoch:       80 for all seven
```

**Three of seven arms — A_s2, B_s2 and D_s1 — are barely-trained epoch-1 models.**
Every fine-tune, every action-sensitivity gate result and every Chrono score to date
was computed on surrogates of essentially random training maturity.

## This replaces the seed-variance explanation

[`go2-surrogate-seed-variance.md`](go2-surrogate-seed-variance.md) reported arm A
completing 23.1% on seed 1 and 69.0% on seed 2 at dw6 and attributed the swing to
seed sensitivity. **The measurement stands. The cause does not.** A_s1 is an
epoch-26 model and A_s2 an epoch-1 model; the seed mattered only because it decided
which epoch won a noise lottery. That is not a property of the arms, the data, or
the training procedure -- it is a selection bug.

It also explains, without any appeal to seed sensitivity:

- the action-sensitivity gate reversing between B_s1 and B_s2 (B_s2 is epoch 1)
- D_s1 scoring 29.3% on the val split (D_s1 is epoch 1)
- the same config selecting epoch 26 on sbel and epoch 60 on north

## The fix needs no retraining

**`last.pt` is epoch 80 for every run**, so it is uniform across arms and seeds and
already on disk. The pipeline now fine-tunes from `last.pt` and writes to
`finetune_ep80_*`, leaving the earlier `best_val`-based artefacts in place so the
two can be compared rather than quietly replaced.

`checkpoint_metric` is now `val_loss` in all 15 configs. **Selecting on `rollout_sel`
was the right intent** -- rollout quality, not window loss, is what a closed-loop
surrogate is for -- and it fails only because one epoch's `rollout_sel` is noise. A
smoothed or multi-epoch rollout criterion is the better long-term answer; `val_loss`
is the correct choice now because it is the one logged metric that actually descends.

## What this invalidates, stated plainly

Every arm ordering, gate verdict and completion rate reported for the gravity arms
before this date was computed on checkpoints of arbitrary maturity, and **none of
them should be cited.** That includes the banded McNemar tables, the dw4/dw6
completion rates, and the arm-B gate reversal.

## What survives

- **The methodological finding is untouched and still binding:** the episode is not
  the experimental unit, the surrogate training run is, and at two seeds per arm the
  exact permutation test cannot return a p below 0.333. That is combinatorics, not
  data. Four seeds per arm remains the floor.
- The tooling built for it -- `seed_level_eval.py`, `gate_seed_aggregate.py` -- is
  unaffected and is what the corrected runs will be read with.
- The corpus, the pairing, the keying fixes and the harness all check out; the defect
  was upstream of all of them, in which file the word "best" pointed at.

## How this went unseen

`best_val.pt` is a name that asserts its own correctness, and nothing downstream ever
compared the epoch it carried against the epoch `val_loss` would have chosen. The
metrics were logged correctly the whole time -- the argmin of `val_loss` sits in
`metrics.jsonl` for all seven runs. **The information needed to catch this was
present from the first run and never read.**

**Scope:** the go2 gravity-world arms A/B/C/D on rigid terrain. Trainings currently
in flight (A_s3, A_s4, D_s2, D_s3, D_s4) carry the old `rollout_sel` setting in their
loaded config and will also need `last.pt`; the config fix applies to runs started
after it.
