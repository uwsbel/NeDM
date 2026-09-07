# Family split, confirmatory result

Pre-registration: `docs/state/decisions/go2-family-split-prereg.md`. Two corpora were
run, and only the second answers the registered question.

```
   cell3   300 episodes   NO disturbance      unregistered contrast
   cell4   255 collected, 208 paired          matched to the discovery corpus
           peak 0-120 N by i%6, prewalk U(0,3), roll and pitch U(-3,3)
```

cell3 was collected with `--perturb-peak-n 0` and zero tilt, constants inherited from
the script that had been written to exercise the parameter-recording fix. It is a
different experiment, not a replication, and its E1 "sign reversal" must not be read as
refuting anything. See the lesson in `lessons/experiment-design.md`.

## Registered endpoints, on cell4

```
   E1   straight  n=78   -0.00471  [-0.0155, -0.0001]   excludes 0
        turning   n=130  +0.00404  [-0.0004, +0.0156]   INCLUDES 0     -> NOT MET
   E2   straight upper bound -0.0001 against criterion -0.020           -> NOT MET
   E3   separation: family 0.00875, motion 0.00684, family wins by 28%  -> FAMILY
```

**E1 fails, and the reason matters.** Both signs are as predicted -- straight negative,
turning positive -- but the turning interval does not exclude zero. The direction
replicates; the registered conjunction does not.

**The discovery magnitude does not replicate.** Discovery gave straight -0.03755
[-0.0593, -0.0104] on 16 episodes. cell4 gives -0.00471 on 78. The new estimate lies
outside the discovery interval: the effect is roughly **eight times smaller** than the
number the hypothesis was built on. That is the expected shape of an effect found on 16
common survivors and tested on the 36 containing them, and it is why E2 is not merely
unmet but decisively so -- the straight-line effect is about 0.005 m/s, not the 0.038
that would have cleared a 0.020 criterion.

**E3 is the one result that reproduces cleanly.** Family beats the body-motion split by
28% on fresh episodes, against 29% on the discovery set, both outside the 20% tie band.
So "turning" is not simply a proxy for realised body-motion magnitude. Two independent
corpora agree on this, and it was operationalised before either was collected.

## The disturbance contrast (not registered)

The accidental zero-disturbance corpus turns out to answer a question neither run could
ask alone.

```
                cell3 (no disturbance)      cell4 (matched disturbance)
   straight     +0.0084 [+0.0042, +0.0128]  -0.00471 [-0.0155, -0.0001]
   turning      +0.0121 [+0.0042, +0.0166]  +0.00404 [-0.0004, +0.0156]
   aggregate    +0.0097 [+0.0059, +0.0139]  -0.00013 [-0.0011, +0.0028]
```

**Without disturbance the straight-line sign is positive; with disturbance it is
negative.** Under no disturbance arm A is uniformly and significantly worse across all
five families with no family structure at all. Under matched disturbance the aggregate
is an extremely tight null and the family structure appears, weakly.

So whatever the split is, it requires disturbance to exist. That is a genuine finding
recovered from an error, and it is unregistered -- the corpora differ in one intended
variable, but they were not designed as a pair and cell3's 300 survived where cell4 lost
15% to pitch-driven stand-up failures, so the two are not matched on survivorship.

## Standing caveat

`go2-command-realisation-deadband.md`: the cell these are measured in lies inside the
forward-velocity deadband, where the robot realises 14-30% of command. Every number
above is a difference in an error that is mostly the robot not moving. The pairing is
sound -- both arms face the same deadband on the same episodes -- but no mechanism story
about *tracking* is supported without addressing that.
