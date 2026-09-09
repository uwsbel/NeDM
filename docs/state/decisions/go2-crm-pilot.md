# CRM pilot: the pipeline works, and the primary metric has to change

**2026-09-08.** 14 pilot episodes on CRM (8 command families, 16 s, default 8x4 patch,
`--soil training`), collected to derisk a full corpus before committing ~10 h of fleet
time.

## The pipeline is sound

```
  episodes                       14
  rows                           median 1475  (min 1405, max 1475; ~1494 = full length)
  truncated at bed boundary       0 of 14
  solver diverged                 0 of 14
  weight-supported (0.7-1.3x)    14 of 14      settled sum Fz against 158 N
  x travel                       median 0.16 m, max 4.03 m  (usable bed ~5.7 m)
```

Cost measured directly: **7.5x real time**, 712 MiB GPU and ~40% GPU utilisation per
episode. That last number, not memory, is the throughput ceiling: 2-3 concurrent per
box, so ~10 fleet-wide, and the driver already serialises CRM. A 2000-episode corpus at
16 s is roughly **10 hours of fleet time**.

16 s fits the bed comfortably -- zero truncations with 5.7 m of usable travel against a
4.03 m worst case -- so episode length is not currently the binding constraint.

## THE FINDING THAT CHANGES THE DESIGN: nothing falls over

**Not one pilot episode fell, diverged, or ended early.** Every episode ran to full
length with the feet carrying body weight. On rigid terrain the base policy's failure
mode was *falling*, and the entire evaluation -- completion rate, the base-failure
complement, the banded McNemar tables -- was built on **survival**.

On CRM the robot survives and simply fails to go anywhere:

```
  family      cmd vx   mean body vx   fraction of commanded
  constant     0.14       +0.104              74%
  weave       -0.17       -0.144              85%
  vel_step     0.10       +0.059              59%
  arc         -0.05       -0.017              34%
  yaw_step     0.11       +0.037              34%
```

A separate probe at a larger command reached **0.21 m/s against a commanded 0.5 (42%)**.

**So a completion-rate metric will not discriminate anything on CRM.** Every arm,
the baseline, and a random control would all score ~100% and the experiment would
return a flat table that looks like a null result and is actually a metric failure.
**The primary metric on CRM must be velocity-tracking error**, which the verdict
harness already supports (the paired tracking criterion, threshold -0.020 m/s, that
went unscored on rigid for want of surviving pairs -- on CRM every pair survives, so
the criterion that was unusable there is the natural one here).

This is worth having *before* collection rather than after: it would have produced a
corpus, four arms, and a table of 100%s.

## Why CRM is a better-posed problem than rigid

The rigid conclusion was that fine-tuning helps where the base policy has headroom and
hurts where it is already optimal -- and on flat rigid the base policy sits at 96-100%,
so there was nothing to win and the method only did damage.

**On CRM the base policy achieves 34-85% of commanded velocity.** That is a large,
unambiguous deficit in exactly the regime where the method was shown to work.

## Open, not yet retired

- **Soil has memory.** Ruts and compaction persist; the surrogate state carries no
  terrain field. The transformer's 128-step (1.28 s) window covers recent interaction,
  which may suffice for a single pass over fresh soil, and does not cover revisited
  ground.
- **`foot_*_in_contact` is NaN on every CRM row** (measured 475/475). The
  contact-conditioned arm must use `foot_*_force_fz_n`, which is fully populated
  through the FSI coupling, or it trains on an all-NaN block.
- **Command envelope.** The pilot's stratified draw with 2 episodes per family gave
  small commands (|vx| <= 0.17). A corpus must span the measured envelope, and the
  tracking deficit is worse at higher speeds, so the pilot understates the headroom.


## The contact Schmitt trigger is mis-tuned for soil, and it is silent about it

`--contact-mode` derives packed per-foot contact from FORCE via a Schmitt trigger, which
is the natural way to recover contact on CRM given `foot_*_in_contact` is NaN there. But
its defaults were set on rigid terrain:

```
  CONTACT_ENGAGE_N  = 60.0     CONTACT_RELEASE_N = 5.0
```

Measured over 23,500 per-foot samples from 12 CRM corpus episodes:

```
  per-foot Fz     p50   26.1    p75   64.4    p90   97.9    p95  124.5    p99  197.1
  fraction of samples above the engage threshold
      >=  5 N   68.6%
      >= 20 N   54.0%
      >= 40 N   41.1%
      >= 60 N   27.8%   <- the rigid default
```

**A Go2 foot at rest carries about 40 N (158 N over four feet), and the rigid engage
threshold is 60 N -- above it.** On compliant soil the force distribution is shifted
down and smoothed (no impact spikes), so the median stance sample sits at 26 N and is
classified as SWING. The channel would still be populated and would still look like
data; it would simply be wrong in a way nothing downstream can detect.

**Set to engage 25 N / release 5 N for CRM**, from the measured distribution rather
than copied from the rigid constant.

**Stated honestly:** a trotting quadruped spends roughly half its time in stance per
foot, and 25 N puts the contact fraction near 50% while 60 N puts it at 28%. That duty
factor is a *sanity check*, not a target -- this robot is barely locomoting on soil
(0.21 m/s against a commanded 0.5), so its real duty factor may differ. The defensible
part is that a threshold above the resting per-foot load cannot be right; the exact
value is a judgement anchored on the measured median, and it is recorded here so it can
be revisited rather than inherited silently the way the 60 N was.


## Tracking scorer, smoke-tested before the corpus landed

`score_crm_tracking.py` had never executed. Run against 2 pilot episodes with the base
policy, on north, at concurrency 2:

```
  scored 2 of 2;  completed 2 of 2 (100.0%)
  vx: mean|err| 0.1231 m/s   mean cmd -0.147   mean achieved -0.025
  wall 283 s
```

**It works, and it immediately reproduces the two things the pilot predicted.**
Completion is 100% -- the metric that would have been used by default discriminates
nothing. And the base policy achieved -0.025 m/s against a commanded -0.147: **17% of
commanded**, an even larger deficit than the 34-85% seen across families earlier.

**Two things this probe changed.**

*The corpus has no root index.* The collector writes `dataset_index.json` PER SCENARIO
DIRECTORY, with `csv_path` relative to that directory. Consolidation cannot glob for a
root index; it has to merge the per-scenario ones and absolutise the paths. Learned from
a 4-episode probe rather than from a failed 960-episode preprocessing run.

*Scoring is expensive and needs sizing now, not later.* 283 s for 2 episodes at
concurrency 2 is ~140 s per episode. A val split of ~200 episodes is therefore ~1 hour
per policy on one box at concurrency 8, and the planned set is 8 arms plus baseline plus
three controls -- about 12 policies. Spread over the four CUDA boxes at the concurrency
their GPUs allow, that is roughly **5-6 hours of fleet time for one scoring round**,
comparable to collecting the corpus itself. Either the val split is capped for scoring
or the arm count is cut; deciding that after training would waste a night.
