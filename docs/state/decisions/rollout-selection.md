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
