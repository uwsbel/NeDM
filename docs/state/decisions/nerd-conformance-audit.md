# Auditing our surrogate against NeRD's ablation table

**Measured 2026-09-05**, against `../reference/papers/nerd-spec.md`. NeRD is the paper
this case study takes structural direction from, so the useful question is not "what does
it do" but **"which of its ablations are we already on the wrong side of"**.

NeRD's ablations are reported as **multiples of its own error**, so 23.3x means that
variant is 23.3 times worse. Double Pendulum / Ant.

| NeRD finding | its ablation cost | ours | status |
|---|---|---|---|
| relative (delta) prediction, not absolute | **26.8x / 5.4x** | predicts `dz` | **OK** |
| base-frame anchoring, not world frame | 1.1x / **23.3x** | body-frame except `pos_z_m`, roll, pitch | **PARTIAL** |
| input normalisation | 3.3x / 3.6x | `state_mean/std`, `action_mean/std` | **OK** |
| output normalisation + inverse-std loss weighting | 1.6x / 5.3x | loss computed on normalised targets, which is the same thing | **OK** |
| contact geometry as INPUT, not predicted | **56.1x / 19.8x** | model predicts contact | **tested, and the opposite holds for us** |
| history `h = 10`; `h=5` worse than `h=1`; `h=20` explodes | 4.4/2.1/1.0 and 4.1/5.2/1.0 | **`block_size = 128`** | **FAR OUTSIDE** |
| history deque not cleared on env reset (a bug, not an ablation) | — | `reset_idx` overwrites `state_hist[env_ids]` and `action_hist[env_ids]` per env | **not present** |

Four of seven are already right. Two findings follow.

## The contact-input result went the other way, and the reason is specific

W1 proposed moving contact from a predicted output to a computed input, on the strength of
that 56.1x/19.8x row. **Measured on our surrogate it is a large loss, not a gain**
(`w1-rigid-scope.md`): 0.968 accuracy predicting contact against 0.608 computing it, at
1 s, and the gap is present at every horizon.

**The load-bearing difference is that NeRD's query is exact and ours is not.** NeRD's
analytic collision detector *defines* the contact its training data records, so the query
introduces zero error by construction. Ours approximates a different system's contact
determination with a fixed threshold on a sphere, and it has a **ceiling near 0.86 at
0.0029 rad of joint error** — before anything compounds. No surrogate accuracy fixes a
ceiling that exists at zero surrogate error.

**Generalisation worth keeping:** an ablation showing "computed beats predicted" is
evidence about a computation that is exact in that setting. Transplanting it requires
checking that the computation is exact in yours, and ours is not.

## The context length is the open gap, and it points at our measured failure

`block_size = 128` at 100 Hz is **1.28 s of history**. NeRD uses `h = 10` at 1/60 s, which
is **0.167 s**. We are at **7.7x** their setting, and past the point they report as
unstable: they say `h = 20` "will occasionally result in an exploded training loss", and
their curve is **non-monotonic** — `h=5` is *worse* than `h=1` on Ant — so intuition is no
guide and the value cannot be defended by "longer is safer".

**This connects to the failure we actually measure.** The action-sensitivity gate's own
framing states the mechanism:

> *"A surrogate can have excellent one-step accuracy and still be action-blind: it
> predicts the next state well because the history already determines it."*

That is a description of what an over-long context buys. Our gate result is corr **0.310**
at 0.5 s and a trustworthy horizon of 0.1 s, which is the shape of an action-blind model,
and our contact-mode accuracy is **0.968 at 1 s** — the model tracks the gait's own
progression very well while responding poorly to what the actions do. **A model given
1.28 s of a periodic gait can predict the next step from phase alone.**

**Hypothesis, stated before the experiment:** the context length is a cause of the
action-insensitivity, not an innocent hyperparameter. **Test:** retrain at `block_size` 8,
16, 32 against 128, everything else fixed, and run the action-sensitivity gate on each.
**Pre-registered reading:** if gate corr at 0.5 s rises monotonically as context shrinks,
the hypothesis holds; if it is flat, context length is not the mechanism and this line
closes.

**Note what makes this cheap and unusually clean:** it changes one hyperparameter, needs
no new data, no new collection, and no architecture work, and it is scored by a gate that
was declared before any of these numbers existed.

## What this audit does not cover

`pos_z_m` remains a world coordinate in a state that is otherwise body-frame. NeRD
eliminates absolute height entirely and lets the contact query's **depths** carry it. That
is a real structural difference and it sits behind the contact-query result above, so it
is not actionable until the context-length question is settled.

## Addendum: `vx` characterised, and a worry withdrawn

**Measured 2026-09-05 on rigid episode CSVs, no models.**

`vx` came back as the least predictable component in the W0 linear baseline — negative R^2
at both horizons — which raised the question of whether the fine-tune acceptance criterion,
scored on `vx` tracking, had less power than it appeared to.

**It did not, and the reason is a distinction between two different quantities:**

```
  the concern    the 0.29 s INCREMENT of vx is unpredictable
  what was scored  the MEAN achieved vx over a 10 s window
```

Averaging ~1000 rows of a **zero-autocorrelation** increment is precisely the case where
the window mean is well determined while each step is not. The criterion's measured 95%
interval, **+/-0.0114 m/s**, never depended on increments being predictable — and serially
uncorrelated increments give that mean a *smaller* standard error than correlated ones
would. **The fine-tune verdict is undisturbed.**

**Three measurements, and the second is the one that mattered:**

| | result |
|---|---|
| increment autocorrelation, lags 1-5, all intervals | flat, slightly negative. Nothing predictable from past increments |
| **sd across events vs within a cycle** | **vx 3.28x, vy 3.11x**, wz 1.23x |
| command changes explaining `dvx` | pooled corr +0.352 = **12.4%** of variance; **zero in six of eight families** |

The second was run because it could have **excused** the negative R^2: a settled limit
cycle sampled at its own period gives an increment that is pure noise, and a negative R^2
would then be correct rather than a failure. **It declined to excuse it.** At 3.28x the
within-cycle spread the section value genuinely varies, so there is something to predict.

**That converts an ambiguous result into a specific one:** not "there is no structure in
`vx`" but **"the linear fit is the wrong instrument for it"** — which is consistent with
the trained surrogate reaching 0.723 on the same family, and with the fault being
action-sensitivity rather than accuracy.

**A command channel is not the missing ingredient.** It explains an eighth of the pooled
variance and *nothing at all* in six of eight families, where the command is constant and
`sd(dvx)` is still 0.06-0.096.

**Method note worth carrying:** the informative check here was the one that could have let
the result off. Running it and reporting that it failed to is what makes the conclusion
narrow enough to act on.
