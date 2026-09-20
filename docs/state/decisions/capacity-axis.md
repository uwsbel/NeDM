# Capacity helps. Abstraction expansion does not. They are separate knobs.

**Status:** w512 scored 2026-09-20. w1024 and a rollout-selected w512 in flight.

## The matched comparison

Both arms selected on `val_loss`, same corpus (`go2_crm_baseline`, 635 episodes), same 80
epochs, same 6 layers, same fine-tune budget, same box, same Chrono build, and the same
75 paired episodes. Only `n_embd` differs.

```
                       vx        vy        wz      n
  w256 val-selected  -39.5%    -10.9%    -20.1%    75
  w512 val-selected  -43.2%    -13.0%    -24.7%    75
```

Doubling the width improves ALL THREE command channels: 3.7 points of forward tracking,
2.1 of lateral, 4.6 of yaw. That last property is worth stating separately, because
almost every other lever in this study has bought forward tracking by degrading heading,
and the large-corpus arms in particular did exactly that. Width does not.

## The comparison NOT to make

Against the headline -40.9% reference, w512 looks like a 2.3 point win. That number is
wrong to quote, because the -40.9% arm is selected on `rollout_sel` and this w512 is
selected on `val_loss`. Rollout selection is worth about 1.4 points on this corpus on its
own, so a w512-vs--40.9% comparison is a width effect and a selection effect added
together and reported as one.

The val-vs-val pairing above is the sound one. Best-vs-best needs a rollout-selected
w512, which is training now; until it lands the claim is "at matched val selection".

## Why this is not in tension with the abstraction ladder

The ladder found that ADDING CHANNELS hurts, monotonically, and that the 36-D
control-interface closure is the best abstraction available. This finds that making the
MODEL bigger helps. Those are not opposing results because they move different things.

```
  what is scaled         effect on delivered policy
  corpus size            helps, with diminishing returns
  model width            helps
  state channel count    hurts, monotonically
```

The state is a specification of what must be predicted; the width is a budget for
predicting it. Widening the state adds quantities the optimiser can exploit and the
metric cannot rank, as the ladder showed when rollout fidelity inverted on that axis.
Widening the model adds capacity to fit a specification that has not changed. The design
rule that follows is compact: keep the state minimal and spend the parameters.

## Open

`w1024` says whether this saturates or keeps going, and it is the arm that could not run
at all until the memory work landed, since it OOMs a 40 GB A100 at the batch the
comparison fixes. If the trend continues to 1024 the honest framing is that this study
never found the top of the capacity curve. If it flattens, w512 is the recommendation.

Not tested: whether width trades against the horizon result. The earlier observation that
width buys horizon rather than one-step accuracy (15.7x parameters for 8% val_loss but
roughly 2x usable rollout) suggests the mechanism here may be horizon, not fit, which
would be consistent with a 15-step branch benefiting from a model that stays accurate
further out. Nothing here tests that and it should not be asserted.

## Correction: surrogates are not comparable across training machines either

The capacity ladder was extended with a rollout-selected w512 trained on euler, intended
as the selection control for the val-selected w512 trained on sbel. It is not a valid
control, and finding out why invalidates a claim made earlier the same day.

Same config file, derived from the sbel one with only the selection fields and the paths
changed. Same dataset, verified identical on both hosts at 635 train episodes / 447,372
transitions and 160 val / 108,479. Same architecture, same 80 epochs.

```
  arm                  host    best val_loss    delivered
  abl_w512  (val)      sbel       0.012329        -43.2%
  selw512   (rollout)  euler      0.016301        -33.9%
  sel_baseline w256    euler      0.017107        -40.9%
  w1024     (val)      sbel       0.012430        -33.6%
```

The euler-trained surrogates converge about 30% worse in val_loss than the sbel-trained
ones at matched architecture and identical data. That is far larger than the difference
selection makes within a single run: inside the selw512 run the val-selected epoch 40 and
the rollout-selected epoch 33 differ in val_loss by 0.09%.

So the -43.2% against -33.9% gap is a TRAINING RUN difference, not a selection-rule
difference, and nothing about rollout selection at w512 can be read from it.

### What this invalidates

The claim recorded earlier that width helps on every channel rested on the sbel
val-selected ladder. Set the two internally-consistent groups side by side:

```
  sbel, val-selected        w256 -39.5%   w512 -43.2%   w1024 -33.6%
  euler, rollout-selected   w256 -40.9%   w512 -33.9%   w1024 pending
```

They disagree about whether w512 beats w256. The first group says width helps by 3.7
points, the second says it costs 7.0. Machine and selection rule both differ between the
groups, so neither can be isolated, and THE LOCATION OF THE CAPACITY PEAK IS NOT
ESTABLISHED. The earlier claim should be read as holding within one training environment
rather than as a property of width.

### What survives

One comparison is clean, because it is matched on everything including host: `abl_w512`
and `w1024`, both val-selected, both trained on sbel, on identical data.

```
  w512    val_loss 0.012329    -43.2%
  w1024   val_loss 0.012430    -33.6%
```

Four times the parameters buy NO improvement in fit, 0.8% worse in fact, and cost 9.6
points of delivered policy. Capacity that does not improve val_loss still enlarges the
space the optimiser can exploit. That is the exploitability mechanism appearing on the
capacity axis, and it is the one capacity statement this study can presently defend.

### Operating rule

This repository already refuses to pair VERDICTS across hosts, because replay is not
machine-invariant. The same refusal is now required one stage earlier: a surrogate
trained on one host may not be compared against a surrogate trained on another, because
training is not machine-invariant either, and the effect is roughly 30% of val_loss,
large enough to swamp any result this study reports. Ladders must be trained end to end
on one host.

The `selw1024` run still in flight remains worth finishing, not as a control for the sbel
ladder but because it completes a second internally consistent ladder on euler. Whether
the w512 peak replicates within THAT group is a real test, and a disagreement between two
matched ladders would itself be the finding.
