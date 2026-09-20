# Multi-step rollout fidelity predicts transfer; one-step accuracy predicts it backwards

**2026-09-19.** Every Go2 surrogate in this study was selected on one-step validation
loss. That is the wrong criterion, the right one was already being computed and logged
every epoch, and switching to it is free.

## The ranking disagreement

On the data-axis dose ladder -- one corpus, nested fractions, so only data volume varies
and every arm is fine-tuned and scored under the identical protocol:

```
  hours   val_loss   5s err/dist   10s err/dist   delivered transfer
   1.30   0.023973        1.577          2.624          -36.3%
   2.61   0.017866        1.004          0.841          -40.7%
   5.15   0.013911        3.086          2.764          -26.8%
  10.5    0.011101        5.160          3.708          -30.7%
  26.1    0.007863        2.525          4.035          -19.8%
```

Spearman rank correlation against delivered policy quality:

```
  one-step val_loss        rho = -0.80      wrong sign
  5s  rollout err/dist     rho = +0.60
  10s rollout err/dist     rho = +0.90
```

`err/dist` is position error over distance travelled, so a value above 1 means the
predicted trajectory has drifted further than the robot actually moved. Only the 2.61 h
surrogate -- the transfer optimum -- stays below 1 at ten seconds.

## The selection is available inside runs already finished

`rollout_sel` is logged every epoch by every run. Comparing what each criterion would
pick out of the same 80 epochs:

```
  surrogate        best val_loss epoch -> rollout_sel    best rollout_sel epoch -> rollout_sel   val cost
  wide 26.1 h       74  ->  4.035                         32  ->  0.607                          +11.8%
  baseline 36-D     54  ->  3.830                         29  ->  0.524                           +1.5%
  dose 5.15 h       14  ->  2.764                         67  ->  0.505                           +9.2%
  dose 10.5 h       45  ->  3.708                         31  ->  0.819                           +0.7%
```

A checkpoint with four to seven times better ten-second rollout fidelity exists inside
every run, at a one-step cost between 0.7% and 11.8%. The best-rolling epoch is
consistently early (29-34 of 80) while the best one-step epoch is late (45-74): the model
goes on improving its next-step prediction long after its multi-step behaviour has started
to come apart, and selecting on val_loss reliably picks the later, worse-rolling weights.

## Why this is the quantity that matters

Policy fine-tuning never queries the model one step ahead. It rolls fifteen steps on the
model's own output and differentiates through the result. Selection on one-step error
optimises a quantity the downstream use never touches, and the dose ladder shows the two
can point in opposite directions.

This also supplies a mechanism for a choice the framework paper makes without one. It
selects checkpoints by multi-step fidelity and lists "why rollout error beats one-step
loss for checkpoint ranking" among its open questions. The answer here is that one-step
loss keeps improving through a regime where compounding behaviour degrades, so it stops
being a proxy for anything the optimiser will experience.

## Consequences

Cheap transfer prediction. `rollout_sel` costs twelve rollouts per epoch and is already
paid for. If it ranks surrogates at rho +0.90, a candidate model can be triaged before
spending a Chrono verdict on it, and Chrono verdicts are the bottleneck in this study.

## Not established

Five points on one axis. The rank correlation is suggestive rather than conclusive at
n = 5, and it has NOT been shown to hold on the channel axis: `forcez` has better
ten-second rollout than the 36-D baseline (2.044 against 3.830) and delivers a worse
policy (-34.8% against -40.6%), which is the opposite ordering. Whether rollout fidelity
ranks abstractions as well as it ranks data volumes is open, and the abstraction ladder
now in flight is the test.

The retrains selecting on `rollout_sel` for the 26.1 h and 36-D arms are running. Until
they are fine-tuned and scored, the claim that better rollout selection yields a better
delivered policy is an inference from the correlation, not a demonstration.

## Addendum: val_loss is not comparable across abstractions

A prediction recorded before the sub-floor arms ran -- that a state which propagates fewer
and easier channels would be competitive or better on one-step val_loss -- was wrong, and
wrong in a way that matters more than the prediction did.

```
  arm                      val_loss   5s err/dist   10s err/dist
  36-D baseline            0.013448        2.012         3.830
  34-D joint_grav          0.018461        1.357         1.563
  31-D joint               0.019099        1.370         2.415
  48-D force3d             0.014513        1.298         1.387
```

Both sub-floor arms come out 37-42% WORSE on one-step loss, not better. But the
comparison should never have been made: val_loss is a weighted mean over the channels an
arm actually propagates, with weights derived per dataset from that dataset's own channel
statistics. Two arms with different state dimensions are averaging different quantities
under different weights, so their val_loss values are not on a common scale and their
ordering carries no information about which model is better.

The rollout metric does not have this problem. `errdist` is planar position error over
distance travelled, computed from `pos_x_m`, `pos_y_m` and `yaw_rad`, which every arm
recovers analytically outside the propagated state. It is the same physical quantity in
the same units regardless of how many channels the model carries.

So on the channel axis there is only one comparable metric available, and it is the
multi-step one. That is a stronger statement than the dose-ladder correlation: there,
one-step loss was comparable and merely ranked badly; here it is not comparable at all.

Under the metric that does compare, every arm that departs from the 36-D baseline rolls
better than it, in both directions -- dropping channels (34-D, 31-D) and adding them
(48-D). Whether any of that reaches the delivered policy is still open and is what the
Chrono verdicts now queued will say.

## Addendum 2: the abstraction spread was mostly a selection artifact

Comparing each ladder arm at its best-val epoch against the same arm at its best-rollout
epoch:

```
  arm                 best-val  its 10s  ep      best-roll  its val  ep
  31-D joint          0.019099    2.415  44          0.614  0.019486 18
  34-D joint_grav     0.018461    1.563  24          0.543  0.018593 37
  36-D baseline       0.013448    3.830  54          0.524  0.013652 29
  40-D contact_cond   0.021953    1.639  50          0.473  0.022217 24
  40-D forcez         0.011697    2.044  49          0.632  0.030314  1
  48-D force3d        0.014513    1.387  41          0.656  0.014991 78
  52-D terrain        0.021242    3.614  50          0.547  0.023705 13
  2.61 h reference    0.017866    0.841   6          0.531  0.019410 34
```

Read down the "its 10s" column and the abstractions look very different: 1.39 to 3.83, a
factor of 2.8. Read down "best-roll" and they do not: 0.47 to 0.66, a factor of 1.4, with
no clear ordering by dimension.

For the 36-D baseline alone, moving the selection epoch takes rollout from 3.830 to
0.524. That single-arm effect is larger than the entire spread between abstractions. So a
ladder assembled from val-selected checkpoints is largely comparing which epoch val_loss
happened to land on, not which channels the model propagates.

Every arm is therefore being retrained under rollout_sel selection before any of them is
fine-tuned, since otherwise the Chrono verdicts would inherit the same artifact.

One entry deserves suspicion rather than acceptance. forcez reaches its best rollout at
EPOCH 1, with val_loss 0.030314, which is 2.6x worse than its converged value. A model
that predicts very little motion scores errdist near 1.0 for free, because its trajectory
stays put while the true one moves, so a number below 1 from an almost untrained model
needs explaining rather than crediting. It is below the static floor, so it is doing
something, but epoch-1 checkpoints should not be selected on this metric without a
guard, and the smoothing window of 3 is not enough to prevent it.

## Confirmed causally: selection changes the delivered policy

The correlational result above is now an intervention. Two surrogates were retrained
identically except for `checkpoint_metric`, fine-tuned under the same protocol to the same
stop, and scored on ONE box against ONE base:

```
  arm                                vx        vy        wz      n
  26.1 h corpus, val-selected     -22.7%    +27.7%     +7.1%     74
  26.1 h corpus, rollout-selected -27.8%     +6.6%     -9.2%     75
  36-D corpus,   val-selected     -39.5%    -10.9%    -20.1%     75
  36-D corpus,   rollout-selected -40.9%    -23.4%    -25.2%     74
```

Selecting on ten-second rollout fidelity instead of one-step loss improves ALL THREE
command channels on BOTH corpora. Nothing else differs: same data, same epochs, same
batch, same seed, same architecture, same stopping rule, same scoring subset, same Chrono
build.

The off-axis repair is the larger effect. On the 26.1 h corpus lateral tracking moves from
27.7% WORSE than base to 6.6% worse, and yaw from 7.1% worse to 9.2% better. That failure
-- a large surrogate buying forward speed by degrading heading -- is the signature this
study has been documenting on the data axis throughout, and checkpoint selection removes
most of it.

Forward tracking on the 36-D arm barely moves (-39.5% to -40.9%), which is consistent
with it already sitting near the best result available; the headroom that exists there is
off-axis, and that is where the gain appears.

## What this does not do

It does not meet the original goal. The large surrogate still delivers a worse policy than
the small one, -27.8% against -40.9%. Selection narrows the gap from 20.8 points to 13.1,
which is a substantial improvement to a negative result rather than a reversal of it.

Cross-box drift, measured incidentally: the 26.1 h val-selected arm reads -19.8% on sbel
and -22.7% on euler, the 36-D val-selected -40.6% and -39.5%. Gaps of 1 to 3 points here,
against 6.6 points seen earlier between sbel and north on a different arm. Same-box
comparison remains the only safe basis and is what the table above uses.
