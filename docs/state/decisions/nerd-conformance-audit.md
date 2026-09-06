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
