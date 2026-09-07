# Two measured properties of this surrogate's training metrics

**Status:** established across two independent designs · **Last updated:** 2026-09-07

These are facts about *this model and this pipeline*, not general cautions. They
decide which number an experiment should register on, and both were learned by
paying for the wrong choice first.

## 1. `val_loss` is near-blind to whatever `rollout_sel` measures

| contrast | `val_loss` | downstream |
|---|---|---|
| `input_noise_sigma` on/off, seed …801 | **2.69x** | 22 → 0 surviving pairs |
| `input_noise_sigma` on/off, seed …802 | **2.83x** | 27 → 11 pairs |
| seed, at `sigma = 0.0` | 1.001x | 22 vs 27 pairs |
| seed, at `sigma = 0.05` | 1.048x | 0 vs 11 pairs |
| seed, observed-gravity arm A | **1.009x** | `rollout_sel` 0.409 vs 0.573, **1.40x** |

**It detects the flag and misses the seed.** Two designs, four seed pairs: wherever
the two metrics disagree, `val_loss` is the one that moves less.

**The mechanism, which is why this is a property rather than a coincidence.**
`val_loss` is one-step and teacher-forced: the model is given the true current
state and asked for the next one. Much of what distinguishes two surrogates only
appears over a rollout, because the current state already encodes the
consequences of whatever the model got wrong so far. This is NeRD's own argument
for why one-step accuracy and open-loop horizon come apart, and it reproduces here.

**Consequence.** Do not register thresholds on `val_loss`. Use a downstream
endpoint, or an open-loop error at a real horizon. `val_loss` is a diagnostic:
useful for "did this train at all", not for "is this surrogate better".

## 2. Training is NOT deterministic, at any seed

`seed_everything` (`trainer.py:93-98`) seeds `random`, `numpy`, `torch` and
`torch.cuda`. **Nothing in this repository sets `cudnn.deterministic`,
`use_deterministic_algorithms` or `cudnn.benchmark`** — verified by grep across
`src/nedm/` and `scripts/training/`.

So a fixed seed pins the data order and the initialisation and **nothing about
which GPU kernels are selected.** Two runs of an identical config on an identical
machine are expected to differ.

**Consequences, in order of how much they cost:**

1. **A cross-machine comparison is uninterpretable without a same-machine
   control.** The cross-box difference is machine *plus* nondeterminism; only
   subtracting a same-box repeat isolates the machine.
2. **"Seed variance" is a misnomer here** for any quantity measured by running two
   seeds once each. It is seed variance *plus* run-to-run variance, and the split
   between them has never been measured.
3. **Retrospective:** seed effects reported on the `input_noise_sigma` 2x2 — 5 and
   11 surviving pairs — are partly run-to-run. Whatever share that is, the
   surrogate-seed story there is weaker than it was stated. **Amendment pending a
   measurement now running.**

**If determinism is ever wanted**, the switches are absent rather than disabled,
so this is a change to make deliberately: it costs throughput, and every existing
result was produced without it.
