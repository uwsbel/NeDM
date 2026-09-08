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

## 2. Training is empirically deterministic, but nothing guarantees it

`seed_everything` (`trainer.py:93-98`) seeds `random`, `numpy`, `torch` and
`torch.cuda`. **Nothing in this repository sets `cudnn.deterministic`,
`use_deterministic_algorithms` or `cudnn.benchmark`** — verified by grep across
`src/nedm/` and `scripts/training/`.

**From that I claimed training could not be reproducible, and that every reported
"seed effect" was seed plus run-to-run variance. That was wrong, and the
correction is measured rather than argued:**

| | val_loss | rollout_sel |
|---|---|---|
| same box, same seed, two runs | **0.00%** | **0.00%** |
| cross box, same seed | 1.92% | 2.67% |
| machine effect (cross − same) | 1.92% | 2.67% |

Two runs of one config on one machine came back **bit-identical** — all 49 weight
tensors equal under `torch.equal`, the full metric series identical to twelve
significant figures. Not close: the same numbers.

**The absent flags did not matter here.** PyTorch happened to select deterministic
kernels for this model on this hardware. **That is a property of the kernels, not
of the code**, and it is one library upgrade or architecture change from being
false. The flags still belong in the trainer; their absence is currently harmless
rather than correct.

**What this settles:**

1. **The retrospective amendment is withdrawn.** The `input_noise_sigma` 2x2's seed
   effects of 5 and 11 surviving pairs are entirely the seed, and that attribution
   stands unchanged.
2. **A 40% gap between two seeds of one config is entirely seed variance.** Repeats
   are worthless here — replication has to be across seeds.
3. **Cross-machine *training* comparisons are sound at about 2%.**

**What it does not settle: the four physics builds.** This measured PyTorch on two
GPUs, not Chrono on four binaries. That remains bounded only by a
one-episode-in-320 per-condition figure, and this number must not be quoted as
licence for pooling the four-shard corpus.
