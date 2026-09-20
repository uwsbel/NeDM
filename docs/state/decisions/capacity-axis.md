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
