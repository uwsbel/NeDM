# W1 on rigid: scope (not implementation)

CRM is tabled ([crm-tabled.md](crm-tabled.md)). On rigid the terrain query is a
closed-form plane query, so contact geometry can move OUT of the model and into
a deterministic `G()`. This scopes that.

---

## READ THIS BEFORE ANY NUMBER BELOW: which metric is valid on which axis

**Three axes, three different valid metrics. Two of them were got wrong in this
document before the rule was written down, and each error produced a confident,
internally consistent, wrong conclusion.**

| comparison | valid metric | why the alternative fails |
|---|---|---|
| across **horizons**, one group, one model | **raw RMSE** | R^2 and normalised RMSE divide by an increment spread that MOVES with horizon -- it dips near one gait cycle and recovers, so a later horizon appears to face a higher bar |
| across **groups**, one horizon, one model | **normalised RMSE** | raw RMSE compares rad against rad/s: a units comparison, not a quality one |
| across **models**, one group, one horizon | **raw RMSE** | the normaliser is a property of the DATA, and two models trained on differently-processed data have different denominators |

**The third is the one that is hardest to see, and it misleads in the flattering
direction.** Measured here: the `body` group reads normalised **0.764 -> 0.975**
across the wrap fix, which says the fix made it worse, while raw error went
**0.4256 -> 0.1771**, which is 2.4x better. `sd(delta pitch)` in the wrapped data
includes the 2*pi jumps; in the unwrapped data it does not. **The baseline was
being divided by a spread its own defect manufactured.**

**The metric is not the finding. The axis is.** Any table below that is read along
an axis other than the one it was computed for will give a plausible answer to the
wrong question.

---

## 1. The contact query: what `G()` can compute

The FK already exists and is Chrono-free: `trot.forward_kinematics`
(`trot.py:114`), foot position in the hip-joint frame, with geometry parsed from
the URDF joint frames by `extract_geometry` (thigh 0.2130, calf 0.2130,
abduction 0.0955). Its docstring claims validation to 0.00000 m and that holds.

**Two things had to be measured, not assumed, to lift it to world frame:**

- **`SIGN = -1`.** The dataset stores raw Chrono-order joint angles, deliberately
  un-flipped (`dataset.py:502`). FK wants the flipped convention. Raw gives 36.97 mm
  mean |z| error; flipped gives 6.38 mm. This is the fifth consumer of the
  four-ordering problem and it must be asserted, not remembered.
- **The ground plane is in the CONFIG, not the CSV**: `terrain.top_z_m = 0.05`.
  There is no soil/ground column on rigid. Any `G()` must join the episode config.
  Assuming z=0 puts every foot 50 mm off and drops contact accuracy to 0.411.

With both right, FK foot height vs the recorded foot position:

    signed bias   0.24 mm      (unbiased -- the frame lift is correct)
    |dz| median   4.52 mm      p90 14.13 mm   p99 27.65 mm
    corr(|dz|, tilt) = -0.013  (flat in tilt: not an attitude error)

**Derived contact boolean vs the recorded indicator**, 30,900 foot-samples over
10 episodes, threshold on FK foot-centre height above the plane:

    contact median   13.4 mm     swing median  161.2 mm
    best threshold   31 mm  ->   accuracy 0.906    F1 0.930

**That was measured on ground-truth state and with the threshold fit in-sample,
so it is an upper bound.** Both defects are now removed -- see the degradation
curve below, which is the number that actually decides W1.

| quantity | status |
|---|---|
| per-foot height above plane | **exists** -- FK + config plane |
| penetration depth | **exists** -- same subtraction, signed |
| contact boolean | **exists** -- threshold, F1 0.930 |
| contact normal | **free** -- constant (0,0,1) on a flat rigid plane |
| contact velocity | **needs new code** -- analytic Jacobian d(FK)/dq, then `J q̇ + base terms` |

Only contact velocity needs writing, and it is the analytic derivative of a
function already in the repo.

## 1b. The degradation curve: does `G()` survive the model's own error?

The upper bound above only matters if the query stays informative when `G()` runs
on PREDICTED joints. Threshold fit on 20 episodes (**30 mm**), evaluated on **25
held-out episodes, 70,856 foot-samples**, with Gaussian error of magnitude `e`
injected into the joint angles (3 repeats per level).

`acc`/`F1` hold the fit threshold fixed; `acc*`/`F1*` refit it at that noise level,
which measures how much the mechanism depends on a tuned constant.

| joint err (rad) | acc | F1 | refit th | acc* | F1* |
|---|---|---|---|---|---|
| 0.000 | 0.867 | 0.916 | 27 mm | 0.873 | 0.918 |
| 0.005 | 0.867 | 0.916 | 27 mm | 0.872 | 0.917 |
| 0.010 | 0.866 | 0.916 | 28 mm | 0.869 | 0.916 |
| 0.020 | 0.862 | 0.913 | 32 mm | 0.861 | 0.913 |
| 0.050 | 0.820 | 0.882 | 43 mm | 0.842 | 0.904 |
| 0.100 | 0.721 | 0.803 | 76 mm | 0.821 | 0.895 |
| 0.200 | 0.585 | 0.672 | 80 mm | 0.768 | 0.855 |
| 0.400 | 0.441 | 0.486 | 80 mm | 0.586 | 0.688 |

Held-out at `e=0` is **0.867 / F1 0.916**, below the in-sample 0.906 / 0.930. The
in-sample figure was optimistic by about 4 points, which is the reason for
re-fitting rather than quoting it.

**Three things the curve says.**

1. **It is flat to 0.02 rad** (0.867 -> 0.862). Roughly 1.1 degrees of per-joint
   error costs half a point of accuracy. The knee is between 0.02 and 0.05.
2. **The threshold does not need tuning in that flat region** -- fixed 30 mm and
   refit 27-32 mm agree to within 0.001. A mechanism that needs a tuned constant
   is more fragile than one that does not, and in the regime that matters this
   one does not.
3. **Refitting rescues a surprising amount of the tail** (at 0.1 rad, 0.721 ->
   0.821). So even past the knee the failure is partly a calibration shift rather
   than loss of signal -- the feet are still ordered correctly, the cut has moved.

**The injection curve is i.i.d. per joint. A surrogate's error is not.** So
rather than mark the surrogate's RMSE on this curve, the measurement below runs
`G()` on the surrogate's actual predicted joints, which carries the actual error
structure. The curve stays as calibration context, not as the answer.

## 1c. `G()` on the surrogate's own rollout -- the measurement that decides W1

**Dataset `go2_contact_40d`, split `val`, 39 held-out episodes, open-loop
autoregressive rollout, checkpoint `go2_contact_40d/checkpoints/best_val.pt`.**
Stated in full because "the surrogate's accuracy" must not come to mean different
things on different machines.

`G() acc` computes contact from the model's PREDICTED joints. `model acc` is the
same model's DIRECTLY PREDICTED contact channels, thresholded at 0.5 -- i.e. the
thing W1 proposed to remove. `frozen` holds contact at its last observed value.

| horizon | joint RMSE | G() acc | G() F1 | model acc | model F1 | frozen |
|---|---|---|---|---|---|---|
| 0.02 s | 0.0029 | 0.859 | 0.897 | **0.997** | 0.998 | 0.981 |
| 0.10 s | 0.0065 | 0.830 | 0.871 | **0.988** | 0.991 | 0.831 |
| 0.29 s | 0.0117 | 0.742 | 0.795 | **0.985** | 0.988 | 0.726 |
| 0.50 s | 0.0181 | 0.701 | 0.763 | **0.981** | 0.985 | 0.712 |
| 1.00 s | 0.0281 | 0.608 | 0.677 | **0.968** | 0.975 | 0.718 |

### This is a negative result for contact-as-input, and it is not close

**The model predicts its own contact channels far better than `G()` reconstructs
them from the model's own joints** -- 0.968 against 0.608 at 1 s, and the gap is
present at every horizon including the shortest. W1's premise was that moving
contact out of the model and into exact geometry would help. Measured, it costs
0.14 accuracy at 0.02 s and 0.36 at 1 s.

**`G()` also fails to beat a frozen-contact baseline.** It is worse at 0.02 s
(0.859 vs 0.981), ties at 0.10 s, wins marginally at 0.29 s (0.742 vs 0.726), and
is worse again at 0.50 and 1.00 s. A mechanism that does not beat "assume nothing
changed" is not carrying the contribution.

**Two distinct causes, and they should not be conflated:**

1. **`G()` has a ceiling of about 0.86**, visible at 0.02 s where joint RMSE is
   0.0029 rad and essentially nothing has compounded yet. That is `G()`
   disagreeing with the recorded contact indicator, not error accumulation --
   a fixed threshold on a rigid sphere is not what Chrono's contact system
   reports. This ceiling caps contact-as-input no matter how good the surrogate
   gets.
2. **Degradation below that ceiling is far faster than the i.i.d. curve predicts.**

### The correlation effect, quantified rather than asserted

Reading the injection curve at each measured RMSE and comparing:

| horizon | joint RMSE | curve predicts | measured | gap |
|---|---|---|---|---|
| 0.02 s | 0.0029 | ~0.867 | 0.859 | 0.008 |
| 0.10 s | 0.0065 | ~0.867 | 0.830 | 0.037 |
| 0.29 s | 0.0117 | ~0.866 | 0.742 | 0.124 |
| 0.50 s | 0.0181 | ~0.863 | 0.701 | 0.162 |
| 1.00 s | 0.0281 | ~0.851 | 0.608 | 0.243 |

**The direction predicted was right and the magnitude is worse than a constant
offset.** i.i.d. per-joint error partially cancels through the FK sum; correlated
error does not. But the gap is 0.008 at 0.02 s and 0.243 at 1 s, so this is not a
fixed penalty for correlation -- **the error structure becomes more damaging as it
compounds, not merely larger**. The injection curve is therefore optimistic in a
horizon-dependent way, and would have been actively misleading as a marker. That
is the argument for having measured directly.

### What this does and does not rule out

It does NOT say contact geometry is useless. It says **`G()`-as-a-replacement-for-
predicting-contact is worse than predicting contact**, on this checkpoint, this
dataset, this split, open-loop. Contact geometry as an ADDITIONAL input alongside
the predicted channels is untouched by this measurement, and is the version worth
testing next -- it cannot do worse than the model alone, because the model keeps
what it already has.

The NeRD correcting-loop analogy is what needs re-examining: NeRD's query is exact
against its own ground truth, and `G()` here is not -- it has a 0.86 ceiling
against the contact indicator the data actually records.

## 2. What leaves the model's output, and whether it still closes

Today `quadruped_contact_conditioned` (40 ch) = 36 ch + the four
`foot_*_in_contact` indicators, both predicted and conditioned on.

Under W1 those **four channels leave the output** and become `G()`-derived inputs.

**The model closes.** `G()` needs exactly: 12 joint positions (present, inside the
24 `QUADRUPED_JOINT_STATE_FIELDS`), `pos_z_m` (present), `roll_rad` and
`pitch_rad` (present, in `DEFAULT_STATE_FIELDS`), and the plane constant (config,
not predicted). **Yaw is not needed** -- `R = Rz·Ry·Rx` and `Rz` preserves the
z-component, so foot height is yaw-independent. Nothing `G()` consumes is a
quantity it was supposed to produce. No circularity.

This is also why the coordinator's "reduce the terrain, not the robot" framing
does not survive the move to rigid, and what replaces it: **full joint positions
must stay in the state**, because without them there is no FK, no foot pose, no
query, and no correcting loop. The reduction moves to the body-level channels.

## 3. The ablation

Contact-as-input (`G()`-derived) against the current predict-it model. Same data,
same width, same schedule. **Scored on the 36 shared channels only**, since the
W1 model does not emit the four indicators -- with the channel-set assertion from
`experiment-design.md` so the two cannot be silently scored on different sets.

## 4. Base-frame re-anchoring

Largely already done, which was not obvious until checked. `DEFAULT_STATE_FIELDS`
is body-frame throughout (`vel_body_x/y`, `ang_vel_body_y`, `roll_rate`,
`yaw_rate`), and gravity is `grav_body_*`. The only world-referenced quantities
are `pos_z_m` and the attitude angles -- and both must stay world-referenced,
because the ground plane is a world object. So NeRD's world-frame ablation has
little left to bite on here; the re-anchoring it recommends is already the
repo's state layout.

## Data

Only ONE dataset carries all three of measured joints, gravity, and contact:

| set | episodes | cols | measured joints | grav | contact |
|---|---|---|---|---|---|
| `go2_verdict_hostlocal` | 1493 | 171 | yes | 3 | 4 |
| `go2_comprehensive_rigid` | 2000 | 168 | yes | 0 | 4 |
| `go2_jointstate_merged` | 304 | 93 | yes | 0 | **0** |

**SCOPE LINE: W1 runs on `go2_verdict_hostlocal`, and only on it.**

Do NOT backfill gravity into `go2_comprehensive_rigid` to reach the larger set.
1493 episodes with all three channels recorded beats 2000 with one reconstructed,
and a backfill is precisely where a silent channel-definition mismatch would
enter -- the same class of defect as the `startswith("vel_body")` selection bug,
which cost a headline. `go2_jointstate_merged` is ruled out on its own: no contact
indicators, so there is nothing to move from output to input.

Anyone who later runs the ablation on the bigger set and finds it disagrees
should look here first.

## 1d. Increment R^2, and a comparator that had to be thrown out

**Dataset `go2_contact_40d`, split `val`, 30 episodes, 274 samples per horizon.**

**A framing I built and then discarded.** To get a like-for-like number against a
one-step-from-ground-truth linear baseline, I summed the model's one-step deltas
along the ground-truth trajectory ("teacher forced"). That produced TF R^2 BELOW
autoregressive R^2 -- 0.005 vs 0.879 at 0.29 s -- which is impossible for a
well-posed comparison, since autoregression is strictly harder.

It is an artifact of the construction, twice over:

- **Open-loop integration of a biased derivative diverges linearly.** Summing 100
  one-step deltas with no state feedback accumulates 100x any systematic bias.
  Autoregression feeds the drifted state back, and for oscillatory channels that
  is self-limiting. So TF-sum is not "the easier version of AR", it is a different
  and worse-conditioned estimator.
- **The aggregate was dominated by one near-constant channel.** `grav_body_z` has
  a ground-truth increment sd of 0.0014 and accumulates a +0.0050 bias, giving
  R^2 = **-60.3** on its own and dragging the unweighted 40-channel mean to
  -0.981. An unweighted mean R^2 is not robust when a channel with almost no
  variance is in the average.

**So there is no valid like-for-like number here against a direct h-step
predictor**, because this model is a one-step delta model and does not produce
one. Reporting the TF row as that comparison would have been a bad number wearing
a good number's name. Autoregressive per-group R^2, which is well-posed:

| horizon | jpos | jvel | grav | contact | body |
|---|---|---|---|---|---|
| 0.02 s | 0.998 | 0.989 | 0.649 | 0.922 | 0.882 |
| 0.10 s | 0.999 | 0.997 | 0.948 | 0.968 | 0.862 |
| 0.29 s | 0.992 | 0.993 | 0.441 | 0.883 | 0.723 |
| 0.50 s | 0.983 | 0.985 | 0.287 | 0.892 | 0.688 |
| 1.00 s | 0.951 | 0.975 | -0.680 | 0.802 | 0.203 |

Gravity is the worst group by a wide margin, consistent with the earlier finding
that the gravity channels are a plant defect rather than a reconstruction error.

## 1e. Is the error actually in contact? Stance vs swing

The premise behind W1 and W2 -- and behind the manuscript's limitations paragraph
-- is that a single continuous model must average across contact switches. If
that is right, per-leg error should concentrate while the leg is loaded.

Per-leg increment R^2, split by that leg's ground-truth contact state at the
target step. Same rollout, same dataset and split.

| horizon | jpos stance | jpos swing | diff | jvel stance | jvel swing | diff |
|---|---|---|---|---|---|---|
| 0.02 s | 0.995 | 0.999 | -0.004 | 0.981 | 0.992 | -0.010 |
| 0.10 s | 0.998 | 0.999 | -0.001 | 0.996 | 0.998 | -0.002 |
| 0.29 s | 0.981 | 0.998 | -0.017 | 0.980 | 0.997 | -0.017 |
| 0.50 s | 0.966 | 0.995 | -0.029 | 0.966 | 0.992 | -0.025 |
| 1.00 s | 0.893 | 0.992 | -0.099 | 0.941 | 0.988 | -0.047 |

**Stance is worse than swing at every horizon, in both channel groups, ten out of
ten comparisons in the same direction, and the gap grows monotonically with
horizon** (-0.004 to -0.099). The direction is not in doubt.

**But read the magnitudes before concluding anything.** Stance joint positions are
still at R^2 0.893 at one second. The error concentrates in contact, and the model
is nonetheless good in contact. "Contact is where the residual lives" is
supported; "the model fails at contact" is not.

**A limitation that bounds what this can say:** the 40-channel state carries no
contact FORCE. So this measures joint dynamics conditioned on contact state, not
contact force dynamics, and the distinction between "mode identification is fine
but within-mode forces are wrong" cannot be settled with these channels. Settling
it needs `foot_*_force_fz_n` in the state, which the raw CSVs have and the
processed dataset does not.

## 1f. The attitude channel is broken, and it -- not contact -- is what fails

Chasing why the `body` group scored worst led to a defect that changes the
attribution of everything above.

### `pitch_rad` wraps, and the delta model has no angle handling

Measured on the `go2_contact_40d` training data:

    pitch_rad  range [-3.142, +3.142]     per-step jumps > 1 rad:  2159
    roll_rad   range [-1.160, +1.067]     per-step jumps > 1 rad:     0
    largest single-step pitch jump: 6.2831 = exactly 2*pi

A Cardan ZYX middle angle is bounded to +-pi/2. This channel spans the full +-pi
and jumps by exactly 2*pi, which is angle wraparound. The model is a DELTA model,
so every wrap is a +-2*pi target it must fit as an ordinary real, 2159 times.

**It also poisons the normalisation.** The wraps inflate the channel's std:

    normalization std   pitch_rad 0.89109     roll_rad 0.03811     (23x)

The loss is computed on normalised targets, so an inflated std silently
DOWN-WEIGHTS the channel -- the one channel with a pathology is also the one the
loss cares least about.

### The channel names are swapped relative to the quaternion

Correlating each logged channel against angles derived from the stored quaternion
(`quat_e0..e3`, which are exact):

| logged | q-roll | q-pitch | q-yaw |
|---|---|---|---|
| `roll_rad` | 0.0175 | **1.0000** | -0.0475 |
| `pitch_rad` | **1.0000** | 0.0175 | -0.2231 |
| `yaw_rad` | -0.2231 | -0.0475 | **1.0000** |

`yaw_rad` matches to 0.000000, so the quaternion formulas are not globally wrong.
`roll_rad` and `pitch_rad` are transposed relative to their names. Whether that is
a field-assignment error or Chrono's `GetCardanAnglesZYX` ordering differing from
what the assignment assumes needs the API confirmed -- but the transposition
itself is not in doubt at correlation 1.0000 each way.

**This did not invalidate the FK validation in 1a**, and the reason is worth
stating so nobody re-runs it: on near-level ground both angles are small, and
`Ry(a)Rx(b)` vs `Ry(b)Rx(a)` differ only at second order, so the swap costs well
under a millimetre there. It matters exactly where the robot tilts.

### Attribution: it is the attitude, not the joints

Recomputing `G()` with one input group substituted from ground truth at a time:

| horizon | all predicted | GT attitude + z | GT joints |
|---|---|---|---|
| 0.10 s | 0.830 | 0.870 | 0.831 |
| 0.29 s | 0.742 | 0.848 | 0.744 |
| 0.50 s | 0.701 | 0.858 | 0.702 |
| 1.00 s | 0.608 | **0.849** | 0.611 |

**With ground-truth attitude, `G()` is flat across every horizon at its ceiling.
With ground-truth joints it collapses exactly as before.** The joint predictions
(RMSE 0.028 rad at 1 s) were never the problem. The whole of `G()`'s rollout
degradation is the predicted body attitude and height.

And the attitude error is large: `pitch_rad` at 1 s has median |error| 0.31 rad on
FALL-FREE episodes, with 55 of 59 episodes above 0.1 rad.

### What this changes

**The 1c conclusion stands but its cause was misattributed.** `G()` still loses to
the model's own contact prediction (0.849 at best against 0.968), so
contact-as-input is still not the win W1 claimed. But the measured collapse to
0.608 was not evidence about contact geometry -- **it was a broken attitude
channel propagating through the FK.** The honest headline is that `G()` runs at
0.85 and the model runs at 0.97, not that `G()` runs at 0.61.

**And the ordering of work changes.** Before adding force channels or dropping
gravity or sweeping context, the wrap should be fixed -- unwrap the angle, or
carry `(sin, cos)`, or take the delta modulo 2*pi. It is a preprocessing change,
it affects a channel every downstream consumer reads, and no arm's result is
clean while a state channel the model both consumes and predicts has 2159
discontinuities in it.

## 1g. Correction: the channel is CORRECT. The representation is not.

Two corrections to 1f, one from the Chrono API and one from my own arithmetic.

**The values are right; only the labels are exchanged.** Chrono's
`GetCardanAnglesZYX` packs **(pitch, roll, yaw)** into `.x/.y/.z`, and
`dataset.py:469` reads `.x` and names it `roll_rad`. So `pitch_rad` **is roll** --
an OUTER ZYX angle, which legitimately spans +-pi and legitimately wraps -- and
`roll_rad` **is pitch**, the middle angle, correctly bounded. Every value in every
episode is a correct angle. Nothing is corrupt and nothing needs re-collecting.
I called the channel broken; it is not. It is correctly labelled-wrong.

**The loss-weighting number in 1f was the wrong std.** I quoted `state_std`
(0.891 vs 0.038, 23x). The loss is computed on normalised TARGETS, so the figure
that matters is `target_std`: **0.41543 against roll's 0.00270, a 154x ratio.**
Same mechanism, same direction, wrong magnitude by a factor of seven.

### The modelling half survives, and the obvious test does not rescue it

"The data is fine" and "a delta model can consume this" are different claims. The
proposed check was to recompute this channel's R^2 with wrapping episodes
excluded, on the theory that a handful of roll-overs produced the -928.

**It does the opposite.** 10 of 59 val episodes (16.9%) contain a wrap:

| subset | R^2 | RMSE | n |
|---|---|---|---|
| all episodes | +0.419 | 1.6159 | 59 |
| non-wrapping only | **-323.4** | **0.3333** | 49 |
| wrapping only | +0.371 | 3.8551 | 10 |

Excluding the wraps makes R^2 *worse* -- because wrapping episodes carry enormous
target variance, and including them inflates the denominator. **R^2 is not usable
on this channel at any episode selection**, which is the fourth time tonight a
variance-normalised aggregate has reported something other than what it was read
to mean. RMSE is the honest metric.

**And on RMSE the failure is real and is not the roll-overs.** On the 49
non-wrapping episodes the ground-truth 1 s increment has sd 0.0185 rad, and the
model's error is 0.3333 rad -- **18x the channel's own variation**, on episodes
containing no wrap at all.

The wraps do account for roughly half the total error: a wrap-aware residual,
`atan2(sin(d), cos(d))`, gives RMSE 0.8585 against 1.6159 unwrapped. So both
things are true -- the representation costs about half, and the remainder is an
ordinary modelling failure on a channel the loss weights 154x too little.

`roll_rad` (true pitch) for contrast: RMSE 0.0574 rad, R^2 -3.686. Small error,
negative R^2, same artifact.

### A bug the new guard found in itself

The unwrap guard's first version tested every channel for a jump above pi and
fired immediately on nine joint VELOCITY channels, up to 16.85 rad/s. Those are
impact transients: a rate may jump by any amount and it says nothing about
circularity. **Testing "large jump" as though it meant "wrapped" is the same
conflation the guard exists to catch, one level up.** It now checks only
angle-valued channels, `_rad` and not `_radps`.

## 1h. The action is 95% determined by the state, and that is the confound

Every perturbation the collector offers is an external push on the body
(`--perturb-peak-n`, `--perturb-torque-peak-nm`, both at the base COG). **None
perturbs the ACTION.** The twelve joint targets are always exactly what the policy
commands from the state it sees, so the training data lies on the manifold
`a = pi(s)`, and a model can fit `f(s, a) -> ds` perfectly while ignoring `a`.
Nothing in the loss penalises ignoring it.

Measured rather than argued. 120,000 rows, states normalised, K nearest
neighbours restricted to **different episodes** -- same-episode neighbours are
temporally adjacent and would measure gait smoothness rather than action
diversity.

| K | neighbourhood radius | conditional / unconditional action variance | effective rank |
|---|---|---|---|
| 8 | 1.335 | **0.0245** | **2.04** |
| 16 | 1.486 | 0.0298 | 2.29 |
| 32 | 1.747 | 0.0396 | 2.47 |
| 64 | 1.905 | 0.0443 | 2.61 |
| 128 | 2.262 | 0.0577 | 2.71 |

**Roughly 4% of action variance survives conditioning on the state, spread over an
effectively 2-to-3-dimensional subspace of a 12-dimensional action.**

**And the trend says this is an upper bound.** Tighter neighbourhoods -- smaller K,
purer conditioning -- give consistently *less* surviving variance and *lower*
rank, monotonically across the whole sweep. Extrapolating toward zero radius, the
true figure is at or below 2.45% and rank 2.04. Two further effects push the same
way: the neighbourhood has finite width, so some of the residual is state
difference rather than action freedom; and the policy consumes a 5-step history,
so part of the residual is history variation rather than command variation.

The surviving rank of ~2-3 against a 3-dimensional command is the expected
signature: **the command is the only thing that moves the action independently of
the state, and it is constant within an episode for six of the eight families.**

This is the largest divergence from NeRD, and it was never discussed. NeRD
collects with uniform random torques resampled every step -- maximal action
variation, zero confounding. We collect exclusively from one policy's closed loop.
It also explains the fine-tune failures directly: the surrogate is only valid
where `a = pi(s)`, fine-tuning moves the policy off that manifold, and `f` is
undefined there. The action-sensitivity gate reporting corr 0.310 is the
instrument working, not failing.

### Context sweep, first cell

| run | best val_loss | selected rollout_sel |
|---|---|---|
| ctx128 (baseline) | 0.00995 | 0.4378 |
| ctx8 | 0.01869 | 0.5551 |

Shorter context is worse on both, which is the expected direction for one-step
loss and does NOT settle the pre-registered question -- that is decided by the
action-sensitivity gate, which has not been run on these checkpoints yet. ctx16
and ctx32 are still training.

## 1i. Two replay facts, one of which was assumed and is false

Before running an action-noise pilot, the new `--action-noise-sigma-rad` flag
needed checking at sigma = 0. The code path makes it a structural no-op -- the
noise RNG is not even constructed unless the flag is positive -- but "structurally
guaranteed" has been wrong twice tonight, so it was measured.

**The flag is a true no-op, verified.** Replaying one episode with the pre-patch
collector and with the patched collector at sigma = 0:

    167 numeric columns compared, 0 differ.   IDENTICAL.

**But the same run showed replay itself is not reproducible**, which is a separate
and pre-existing fact. Original recording vs a fresh replay with the UNPATCHED
collector, same seed, same spec:

    164 numeric columns, 145 differ.  foot_fr_force_fz_n max |diff| 51.3 N

This is why the control mattered. Without it, a broken-looking replay would have
been attributed to the new flag.

### It is chaos from rounding, not a structural difference

| row | t | `joint_fl_calf_pos_rad` \|diff\| | `pos_x_m` \|diff\| |
|---|---|---|---|
| 0 | 1.40 s | 4.6e-05 | 7.2e-06 |
| 50 | 1.90 s | 2.8e-04 | 3.1e-05 |
| 1000 | 11.40 s | 8.0e-04 | 3.3e-05 |
| 3984 | 41.24 s | **5.4e-02** | **2.0e-02** |

Divergence begins at the level of CSV rounding and amplifies roughly 5000x over
41 s. A contact-rich simulation with a 50 Hz policy in the loop is chaotic; two
runs that agree to float precision at the first recorded row do not stay together.

**The earlier "s2000000 episodes replay bit-identically" is false.** It was
generalised from a small sample and is now measured directly on one of exactly
those episodes.

**What it does and does not invalidate.** At the horizons the action-sensitivity
gate uses, replay divergence is 1e-4 to 1e-3 rad against a measured `d_chrono` of
about 0.27 rad for joint positions -- two to three orders of magnitude below the
signal. **The gate is not compromised.** What is compromised is any claim that
rests on exact reproduction: a stripped digest comparison, or treating a recording
as ground truth for an open-loop replay at multi-second horizons, where by 41 s
the replay differs from its own original by more than the quantity being measured.

### Refinement: the simulator IS deterministic. The divergence has a source.

The section above called this "chaos from rounding". Chaos explains the
AMPLIFICATION and not the origin, and the distinction matters.

Three replays now exist of the same episode spec: with the patched collector at
sigma = 0, with the pre-patch collector, and with the collector from before the
perturbation RNG draw-order fix.

    patched vs pre-patch            0 of 167 numeric columns differ
    pre-RNG-fix vs patched          3 differ, all grav_world_* (ground-tilt gravity)
    ANY of them vs the recording  145 of 164 differ, max |diff| 51.31091 N
                                   -- the SAME value to 7 digits from every version

**The simulator is deterministic.** Two collector versions separated by a
deliberate RNG change produce byte-equal physics. And the divergence from the
recording is identical across all three, so it originates in the RECORDING side,
not in any run-to-run instability.

**It is also not the RNG fix**, which was the obvious suspect: the pre-fix
collector diverges from the recording by exactly the same amount.

**And it is not a parameter reconstruction error.** `spec_for` rebuilds the
perturbation peak, prewalk and ground tilt from a seeded RNG rather than reading
them from the episode metadata, which records none of them -- so that was the
leading hypothesis. It is ruled out by magnitude: the reconstruction gives a
ground tilt of 1.57 deg roll, and 41 s on a slope that size would displace the
robot by metres. Measured `pos_x_m` divergence is 0.02 m.

**What remains is an environment difference between the recording (Sep 4 23:12)
and now** -- Chrono build, thread count, or host. The initial difference is
4.6e-05 rad at the first recorded row, which then amplifies about 5000x over 41 s
through ordinary contact chaos.

**This reconciles the verdict harness.** Its check is a genuine sha256 over 164
physics columns on every row and it aborts on any mismatch -- not weaker than its
docstring. It produced a verdict at n=36 because it collected and replayed within
one environment, where reproduction is exact. **The check is sound; what it
cannot detect is a recording made under a different build.** The episode metadata
recording no perturbation, prewalk or tilt parameters is a separate gap worth
closing, because it forces replay to depend on a reconstruction nobody can verify
against the episode itself.

### STATUS: UNDER TEST, not a result. Cause eliminated, not established.

The two sections above are downgraded. sbel-pc replays its own episodes at
**0.000e+00 across 164 physics columns x 3994 rows**, so replay failure is not a
general property, and its argument against the chaos reading is quantitative:

**Chaos amplifies a difference that already exists; it cannot create one at t=0.**
A float64 round-trip through CSV is ~1e-16 relative. The observed row-0
difference is 4.6e-05 rad -- eleven orders larger. And the rates do not
reconcile: 4.6e-05 to 5.4e-02 over 40 s is an e-folding time near 5 s, but
reaching 4.6e-05 from 1e-16 during the 1.4 s before recording would need one near
56 ms. **One system does not have two Lyapunov times a hundredfold apart.** The
difference existed before recording began.

**My magnitude argument against a parameter mismatch was invalid.** I compared
"reconstructed tilt versus no tilt" and concluded metres of displacement. The real
alternative is "reconstructed tilt versus slightly different tilt", which produces
exactly a small initial offset. The argument did not test what it claimed to test.

**What has actually been eliminated, by measurement:**

| candidate | verdict |
|---|---|
| the new action-noise flag | ruled out -- 0 of 167 columns differ vs unpatched |
| the perturbation RNG draw-order fix | ruled out -- pre-fix collector diverges identically |
| run-to-run nondeterminism | ruled out -- two collector versions agree byte-for-byte |
| `prewalk`, `ground-tilt-roll`, `ground-tilt-pitch` | **match** -- driver draws 0.15 / -1.57 / -1.17, reconstruction gives the same |
| `perturb-peak-n`, `ground-size-m`, `patch-y` | match -- 24.0, 200.0, 4.0 on both sides |
| `spawn`, `heading`, `command_params`, `seed`, `duration` | taken from episode metadata, not reconstructed |

**No cause is established.** Every parameter the replay passes is either read from
the episode's own metadata or reconstructed and verified to match the driver.
What remains is an environment difference between the Sep 4 recording and now, or
a parameter not yet identified -- and "the residual after elimination" is a
hypothesis, not a finding. **Marked under test.**

**Two things this does not touch, both checked rather than assumed.** The verdict
is unaffected: 3e-5 m over a 10 s scored window is 3e-6 m/s against a +-0.0114
interval, four thousandfold below it. And the action-noise verification is
unaffected, because it compared patched against unpatched on the SAME
reconstruction, so it is independent of whether that reconstruction is right.

**One real gap regardless of how this lands:** the episode metadata records no
perturbation peak, prewalk or ground tilt, so replay depends on rederiving them
from a driver RNG stream that no longer exists in the episode. That worked here
and is unverifiable in general. The harness's digest check is meaningful only for
episodes replayed on their collecting machine with a collector whose RNG stream
matches -- a condition that was nowhere stated.

### RESOLVED: it was the Chrono build. Replay IS bit-identical.

    replay with PYTHONPATH=/home/kyle/chrono-build/bin
    original vs replay: 164 numeric columns, 0 differ, 0.000e+00

**Retracted in full: "replay is not reproducible" was false.** It is exactly
reproducible when run against the build that produced the episodes.

There are two Chrono builds on this machine:

    /home/kyle/chrono-build/bin/pychrono/_core.so          md5 d1d0bd0a  Sep 3
    envs/nedm/.../site-packages/pychrono/_core.so          md5 8e9e3865  May 7

`drive_go2_collection.py` sets `PYTHONPATH` from `NEDM_CHRONO_PYTHONPATH` for every
episode it launches, so collection used the local build. My replay went through
`gate_go2_action_sensitivity.py`, whose `PY_` is the conda interpreter and which
sets **no** `PYTHONPATH` -- so it imported the conda build. Two different binaries,
two different floating-point paths, a difference present at the first step and
amplified about 5000x over 41 s by ordinary contact chaos.

**sbel-pc's rate argument was right and it is what forced this.** Chaos amplifies a
difference; it cannot create one. 4.6e-05 at row 0 against a 1e-16 float64 floor
meant something differed before recording started, and the only thing that did was
the binary.

### RETRACTED: the gate was never affected. The fault was my invocation.

I recorded, in d24d5ed, that `gate_go2_action_sensitivity.py` sets no `PYTHONPATH`
and that its apparatus check therefore mixes two builds. **That is false and is
withdrawn.** The gate is correct:

```
  CHRONO = "/home/kyle/chrono-build/bin"                     # the COLLECTION build
  def env_for(s):
      return dict(os.environ, PYTHONPATH=f"{CHRONO}:{REPO}/src", ...)
  p = subprocess.run(arm_cmd(...), env=env_for(s), ...)      # the only subprocess call
```

`PY_` being the conda interpreter is irrelevant, because `PYTHONPATH` is prepended
and the local build wins over `site-packages`. **The gate's arms and its apparatus
check both run against the collection build. No gate number is contaminated and
nothing needs re-running.**

**What I actually did** was call `arm_cmd()` to BUILD the command and then run it
with a bare `subprocess.run(cmd, cwd=REPO)` -- no `env=`. `run_arm()` is the
function that supplies the environment, and I used only its first half. **I reused
part of a two-part function and assumed it was the whole thing**, then attributed
the result to the tool rather than to my use of it.

So the lesson is about **ad-hoc replays**, not about the gate: two Chrono builds
sit on this box four months apart, both importable, and which one loads depends on
whether `PYTHONPATH` happens to be set. That silently produces plausible numbers
rather than an error. **The precondition for the digest check is not "same
machine", it is "same binary"**, and nothing in the repo states or checks it.

**The fix worth more than a docstring note:** have the collector record the md5 of
the `pychrono` `_core.so` it imported into the episode metadata, alongside the
perturbation, prewalk and tilt values. Then a build mismatch is DETECTED at
comparison time rather than diagnosed over several hours. Same argument as
recording values instead of seeds, applied to the binary.

**And the metadata gap stands regardless.** Episodes still record no perturbation
peak, prewalk or ground tilt, so replay depends on rederiving them from a driver
RNG stream the episode does not carry. That was verified correct here only by
locating the collection-time commit and re-executing the draw sequence by hand.
Recording the values -- not the seed -- would make replay checkable rather than
reconstructable.

## 1j. The wrap fix moves the gate. Paired, at 0.5 s, on every measure.

First result in this study to move the action-sensitivity gate.

**Validity before the statistic.** Both runs are `sequence_length` 128 and both used
the same cached Chrono arms, so the comparison is paired by construction:

    max |d_chrono_A - d_chrono_B| = 0.000e+00   at both horizons

Identical, not merely comparable. The between-episode variance is common to both
arms and cancels in the difference.

| 0.5 s, 16 matched episodes | value | 95% CI | |
|---|---|---|---|
| gain | A 1.451 -> B 1.046 | | |
| paired `d(log gain)` | -0.2706 | [-0.5122, -0.0248] | **excludes 0** |
| paired `d(\|log gain\|)` | -0.3184 | [-0.5338, -0.1060] | **excludes 0** |
| corr | A 0.479 -> B 0.773 | | |
| paired-bootstrap `d(corr)` | +0.2863 | [+0.0874, +0.5307] | **excludes 0** |

**The marginal intervals overlapped and I had written this off as unseparated.**
corr [-0.022, 0.788] against [0.450, 0.917] looks like nothing. Paired, it is
+0.2863 excluding zero. Pairing beat an n we could not have afforded to collect.

**Two things the analysis had to get right.**

`corr` is a **cross-episode** statistic with no per-episode value, so "take 16
per-episode differences" is not available for it. It is paired by **bootstrap** --
resample episodes, recompute corr for BOTH arms on the SAME resample, difference,
20k draws.

And `d(log gain)` alone is the wrong target, because **gain's ideal is 1.0, not
smaller** -- a run overshooting to 0.6 would score as an improvement on the signed
difference. `d(|log gain|)`, the distance from ideal, is what the verdict cares
about: 0.828 -> 0.510. That it separates too is what makes this solid rather than
an artefact of which direction was examined.

### 1.0 s is UNDERPOWERED, NOT NULL

    0.5 s  d(log gain)  [-0.5122, -0.0248]   width 0.49
    1.0 s  d(log gain)  [-0.6499, +0.5593]   width 1.21   -- 2.5x wider

Nothing separates at 1.0 s, and the apparatus is worse there (`err_over_signal`
0.164 against 0.093). **So the supported claim is "we cannot detect an effect at
1.0 s", NOT "the fix does not help at 1.0 s".** Those are different statements and
the interval width is the evidence against the second. If 1.0 s ever needs to be
readable the fix is more episodes, not a better statistic.

### What is NOT established

**The attribution.** `go2_contact_40d_unwrap_pinned` restores the pre-unwrap
`target_std` for that one channel against the same unwrapped arrays, and it is the
only thing that separates "the +-2*pi target spikes are gone" from "that channel's
effective loss weight rose 113x". Both landed in this run. The paired analysis
sharpens THAT something happened, not WHAT.

**And the stake is worth stating plainly before the pinned arm lands**, so neither
outcome gets over-read on arrival. The trustworthy horizon has been 0.1 s all
study. This cell reads gain 1.046 PASS and corr 0.773, with the interval
straddling the 0.5 threshold. If it survives attribution, the usable horizon
extends roughly fivefold and 0.5 s covers a full gait cycle -- the thing we
established the model could not do -- which would relicense the fine-tune line,
since v4's 5-step branches were chosen against a 0.1 s window. If instead the
pinned arm attributes it to the loss weight, that is equally real: one
badly-weighted channel would have been costing a fivefold horizon.

Neither is claimed. One unreplicated pair with an interval straddling the
threshold is precisely the shape of result this study has spent its time
retracting.

### Tooling

The gate now persists `per_episode` arrays with episode ids, so any two runs on
shared arms can be paired without re-instrumenting, and it applies the same
`circular_unwrapped` transform the training set got -- a model trained on
unwrapped angles fed a wrapped +-2*pi jump is being evaluated off-distribution on
the one channel that was fixed.

**And cached arms are valid only for the `sequence_length` that generated them.**
`branch_at = rowsA[L]["time_s"]`, so ctx8 and ctx16 reused L=128 arms branching at
2.68 s while their own windows start at 1.48 s -- entirely before the branch,
`d_chrono` of 1e-8, `err_over_signal` of 1.1e8, INCOMPLETE. The apparatus check
caught it and refused to emit a number. Same class as the Chrono build mismatch: a
cached artefact reused under conditions it was not generated for, producing
plausible machinery rather than an error.

## 1k. Arm C, the context sweep: a third outcome, and what it constrains

Pre-registered reading was: gate corr at 0.5 s rising monotonically as context
shrinks supports the action-blindness mechanism; flat closes the line.

**Neither branch applies. All three shortened contexts return INCOMPLETE.** Each
cell got its own regenerated arm B at its own branch time, so this is not the
stale-cache artefact of the first attempt.

| context | history | body_vel err/signal 0.5 s | 1.0 s | verdict |
|---|---|---|---|---|
| 8 | 0.08 s | 4.968 | 8.222 | INCOMPLETE |
| 16 | 0.16 s | 2.837 | 3.485 | INCOMPLETE |
| 32 | 0.32 s | 2.558 | 6.945 | INCOMPLETE |
| 128 | 1.28 s | 0.609 | 0.650 | PARTIAL |
| 128 + unwrap | 1.28 s | **0.518** | 0.802 | PARTIAL |

The surrogate's own open-loop error is 2.6x to 8.2x the between-arm signal at
every shortened context, so the gate refuses to emit gain or corr. At that ratio
the numbers would be reading its own drift.

**This is not a null result and must not be written as one.** Shortening context
degrades the surrogate faster than it could plausibly improve action sensitivity,
so the remedy destroys the instrument that would measure it. Two things follow and
only the first is about the hypothesis:

- **The mechanism is neither supported nor refuted.** Nothing here bears on
  whether 1.28 s of periodic gait lets the model predict from phase alone.
- **Shortening context is not a usable fix regardless**, because at ctx <= 32 the
  model is not accurate enough to serve as a surrogate at 0.5 s at all.

**The one pre-registered prediction that held is `val_loss`:** 0.00995 at 128,
0.01628 at 16, 0.01869 at 8, monotone, shorter is worse. The declared rule was
that val_loss and the gate moving the SAME way would mean the mechanism was
wrong. They did not move the same way -- **the gate did not move at all, because
it could not** -- so the rule does not apply.

### What the sweep constrains, and what it does not

The measurable/unmeasurable boundary lies **somewhere in (32, 128]**, a 4x gap
with no cell in it. We established that 32 fails and 128 works; **we did NOT
establish that 128 is needed.**

**Two claims must not be collapsed, and only the first is measured:**

| claim | value | status |
|---|---|---|
| what we USE | 1.28 s against NeRD's 0.167 s = **7.7x** | VERIFIED both sides |
| what we NEED | somewhere in (0.32 s, 1.28 s] | a BOUND, no cell inside it |

"Our model needs eight times the history" is the second claim quoted with the
first's number. NeRD's side is now verified at source in `nerd-spec.md`:
`frame_dt = 1/60 s` (line 65), `h = 10` for all six robots (line 282,
`num_states_history: 10`), and their own non-monotonicity -- h=5 worse than h=1 on
Ant, h=20 "occasionally exploded" (line 453). Ours is verified too. **What is not
measured is our minimum:**

    true minimum 128  ->  7.7x NeRD
                  64  ->  3.8x
                  48  ->  2.9x

**"Eight times the history on a robot of the same size" and "three times" support
different stories.** One ctx64 cell -- one training run plus one arm regeneration
-- converts the range into a number, and should be run before the ratio is used
for anything.

### The reframing this suggests, and the prediction it licenses

If more history is needed because the reduced state dropped information the
transition depends on, then **context length is a diagnostic of the abstraction
rather than a hyperparameter** -- which is the manuscript's own subject. NeRD
keeps the full generalized state and takes contact from an analytic query; we keep
a 40-channel projection.

**Pre-registered for Arm A, before it runs:** restoring a state variable the
transition provably depends on should REDUCE the history required. With foot force
and slip in the state, the apparatus ratio at ctx32 should improve materially
against the 2.558 measured here, and may become measurable at all. If it does not,
the partial-observability explanation is wrong.

**"Materially" needs a floor, and we do not have one.** The only handle is that
ctx128 baseline and ctx128 unwrap differ by 0.09 in this ratio at the SAME context
-- a real difference between two models, which means model-to-model variation is
not obviously small next to what Arm A might buy. **Arm A should therefore run
ctx128 and ctx32 with the added channels, plus one seed repeat at ctx32 without
them** -- three runs, where the third is the threshold the first two are judged
against.

### Declared next action if the context diagnosis is revisited

One re-run at `--rel-sigma 0.05`, which the gate's own docstring names as the
response to INCOMPLETE -- so it is pre-declared rather than a remedy invented
after seeing a null. **Recorded with its reason: the apparatus, not the model, was
the binding constraint.** Not run now; three hours of Chrono arms is the wrong
spend while the unwrap attribution is outstanding.

## 1l. Attribution resolved: it is the spike removal, not the loss weight

`go2_contact_40d_unwrap_pinned` trains on the SAME unwrapped arrays but restores
`pitch_rad`'s pre-unwrap `target_std` (0.41543), so it keeps the OLD effective loss
weight. It is the only thing that separates the two changes that landed together.

Its gate: **gain 1.007 [0.815, 1.692] PASS, corr 0.704 [0.320, 0.889], cosine
0.928 PASS** -- essentially matching the unpinned arm despite the old weight.

Paired, 0.5 s, 16 matched episodes, `max |d_chrono_A - d_chrono_B| = 0.0e+00` on
all three comparisons:

| comparison | isolates | `d(\|log gain\|)` | 95% CI | |
|---|---|---|---|---|
| baseline -> pinned | **spike removal** | **-0.4158** | [-0.7668, -0.0502] | **excludes 0** |
| pinned -> unwrap | **the 113x weight** | +0.0974 | [-0.1801, +0.3802] | includes 0 |
| baseline -> unwrap | both together | -0.3184 | [-0.5343, -0.1091] | excludes 0 |

**Removing the +-2*pi target spikes carries the whole effect.** Spike removal alone
excludes zero and is LARGER than the combined change. The weight change's point
estimate is **+0.0974 -- slightly worse** -- though its interval includes zero, so
the supported statement is that it does nothing measurable, not that it hurts.

**Without the pinned arm this would have been a real result with two candidate
causes and no way to choose**, and the 113x loss-weight correction is the more
interesting-sounding one.

**One limit held rather than glossed:** isolated spike removal moves corr +0.2311
with the interval touching zero at -0.0097. corr separates only when both changes
are pooled, which is an n=16 power limit and NOT evidence that the weight change
contributes. Gain is the solid result.

### State the horizon claim precisely, because it will be quoted

    0.5 s   gain 1.007 / 1.046   PASS cleanly, both arms
            corr 0.704 / 0.773   point estimate above the 0.5 threshold,
                                 lower bound 0.320 / 0.450 -- STRADDLES it

**Not "the trustworthy horizon extends fivefold".** The supported claim is: *the
0.5 s cell now passes on gain and is favourable-but-indeterminate on corr,
attributable to a data representation defect.* The fix for the indeterminacy is
more episodes, not a better statistic.

**And the cause is mundane, which is why it is credible:** a delta model was being
handed 2159 discontinuities of exactly 2*pi. Nothing about the abstraction, the
loss, or contact.

## 1m. The unwrap must reach every future dataset, and that is now checkable

The excitation collection uses the same collector and the same channel, and
everything trained on it is a delta model. **A dataset that reaches training with
the wraps intact inherits a defect already proven expensive, and does so silently
-- training runs, the loss falls, and only the gate notices.**

The unwrap is structural in `preprocess.py` from `c699338`, and the
`circular_unwrapped` metadata marker from `8633631`. **Both are on origin, so any
checkout at or past those commits gets it automatically** -- but a checkout that
predates them does not, which is the actual risk on a second machine.

`scripts/preprocess/verify_circular_unwrap.py` fails loudly on any processed
dataset whose circular channels still carry wrap-sized targets. Run it before
anything trains.

**The threshold had to be measured, not assumed.** The first version failed on
`>1.0 rad` reasoning that a real per-step attitude change is ~0.03 rad -- and it
FAILED the known-good unwrapped dataset, which legitimately contains 6 steps up to
2.8766 rad. Those are genuine large changes in near-diverged episodes, and
`np.unwrap` leaves them precisely because they are BELOW pi and therefore not
wraps. **Testing "large" when the property is "wrapped" is the same
magnitude-for-property conflation the preprocessing guard itself got wrong on
joint velocities.** The threshold is now 5.0 rad, against 2*pi = 6.283, and counts
above 1.0 rad are reported as a diagnostic rather than a failure.

Self-tested against both a known-bad and a known-good dataset, which is how the
false failure was caught:

    go2_contact_40d          pitch_rad max 6.2832   39241 wraps   FAIL
    go2_contact_40d_unwrap   pitch_rad max 2.8766       0 wraps   PASS

## 1n. RETRACTION: the ctx64 result was an artifact of its own branch point

**Section 1k's context table is void as quantities, and the "minimum <= 0.64 s,
at most 3.8x NeRD" conclusion is withdrawn.** It was measured on an apparatus that
each cell partly chose for itself.

`branch_at = rowsA[L]["time_s"]` tied the branch to `sequence_length`, so **every
context was scored against a branch point selected to suit it** -- a different
instant, a different gait phase, a different `d_chrono`. The cell that looked
measurable was the one whose own branch point flattered it.

Branching at a **fixed row** (128) instead gives every context the same arms, the
same instant, and the same `d_chrono`. A model of context L uses rows `[B-L, B)`
as history -- the last L rows of the same identical prefix.

**Verified behaviour-preserving before use.** For L=128 the change is a no-op and
the baseline reproduces to the last digit:

    gain     1.451003619926325  ->  1.451003619926325   IDENTICAL
    corr     0.47924507377254183 -> 0.47924507377254183 IDENTICAL
    d_chrono 0.04002717314039082 -> 0.04002717314039082 IDENTICAL

So the unwrap, pinned and baseline results are untouched. **One result retracted,
not a cascade.**

### The corrected curve, all cells on one apparatus

| context | history | err/signal 0.5 s (common) | (own branch, VOID) | verdict |
|---|---|---|---|---|
| 8 | 0.08 s | 4.114 | 4.968 | INCOMPLETE |
| 16 | 0.16 s | 2.184 | 2.837 | INCOMPLETE |
| 32 | 0.32 s | 1.637 | 2.558 | INCOMPLETE |
| 64 | 0.64 s | **1.572** | **0.994** | **INCOMPLETE** |
| 128 | 1.28 s | 0.609 | 0.609 | PARTIAL |

**ctx64 moved the wrong way -- 0.994 to 1.572 -- and crossed the threshold.** Its
own branch fell at t ~ 2.04 s against the common branch at t ~ 2.68 s: a different
gait phase. Every other cell moved DOWN, so the bias was not uniform; each cell's
own branch point simply suited it differently.

**The bound therefore moves UP, not down:**

    reported in 1k    minimum <= 0.64 s        ->  at most 3.8x NeRD
    corrected         minimum in (0.64, 1.28]  ->  3.8x to 7.7x

The ctx64 run tightened the LOWER bound and left the upper one where it was. "One
run halved the number" was wrong. And **1k's suggestive ctx64-beats-ctx128
direction result is void** -- it compared two different apparatuses.

The curve is also nearly flat from 32 to 64 (1.637 to 1.572) and then drops
sharply to 0.609 at 128, so the transition is somewhere in (0.64, 1.28] and is not
gradual.

### The class this belongs to

> **A comparison in which the thing being compared silently selects its own
> reference.**

| instance | how the reference was selected |
|---|---|
| the Chrono build | the run chose its own physics, via whether `PYTHONPATH` was set |
| cached gate arms | a model was scored against arms built for a different context |
| branch-at-L | each context chose the branch point it was measured at |

**All three produce plausible numbers rather than errors, and all three were found
by a change made for a different reason.** Fixed-index branching was requested for
statistical power and turned out to be a correctness fix. No amount of scrutinising
the numbers would have exposed it, because the numbers were internally consistent.

### What it enables

Arm A's prediction is now a clean paired test on a common reference rather than an
unpairable comparison: **if history is compensating for partial observability,
adding contact force and slip should make a shorter context measurable where it
currently is not.** Concretely, ctx64 + force/slip against the same arm set that
just read 1.572 INCOMPLETE without them -- a binary outcome, not a ratio
improvement to be argued about. The corrected 3.8x-to-7.7x range makes that
prediction more interesting rather than less, since there is more history to
explain away.

## 1o. CORRECTION: the per-group R^2-against-horizon table in 1d is not a degradation curve

`normalised RMSE^2 ~ 1 - R^2`, so the normalised-RMSE metric adopted to escape
R^2's denominator problem **inherits it along the horizon axis**. Both divide by
the increment spread `sd(dx)`, and that spread MOVES with horizon -- it dips near
one gait cycle, when the oscillatory part of the state has returned near its start
so the net increment is small, and recovers once drift accumulates.

**So a model at 1.00 s appears to face a higher bar than at 0.29 s not because
prediction is harder but because the divisor is larger.**

    ACROSS CHANNELS at one horizon   normalised RMSE   -- correct, keep
    ACROSS HORIZONS for one channel  RAW RMSE          -- or a horizon-independent scale

### Re-reported in raw RMSE

| horizon | jpos | jvel | grav | contact | body |
|---|---|---|---|---|---|
| 0.02 s | 0.00373 | 0.32426 | 0.00040 | 0.08963 | 0.24682 |
| 0.10 s | 0.01003 | 0.37168 | 0.00242 | 0.11482 | 0.50303 |
| 0.29 s | 0.02024 | 0.44679 | 0.00909 | 0.14626 | 0.41090 |
| 0.50 s | 0.03073 | 0.75305 | 0.01590 | 0.19136 | 0.40376 |
| 1.00 s | 0.05495 | 1.02474 | 0.02627 | 0.25885 | 0.42556 |

**Raw error grows monotonically with horizon for jpos, jvel, grav and contact,
exactly as physics demands.** The R^2 table's apparent structure was largely its
moving normaliser.

**Two readings change materially.**

`grav` in R^2 collapses 0.649 -> **-0.680**, which was read as the worst group by a
wide margin. In raw error it goes **0.00040 -> 0.02627 rad** -- among the smallest
absolute errors in the table. This is the same low-variance artefact identified in
1e, now visible on the horizon axis as well as the channel axis. **Gravity is not a
failing group; it is a near-constant one.**

`body` is the one group whose raw error is NOT monotone: 0.247 at 0.02 s, 0.503 at
0.10 s, then falling to ~0.41 and flat. **That is the `pitch_rad` wrap.** A 2*pi
error appears or does not depending on whether a wrap falls inside the window, and
it does not grow with horizon the way an accumulating error does. The body group's
numbers here are from the WRAPPED baseline and inherit the defect fixed in 1j.

**What still stands:** the within-horizon rankings. Body being worse than joints at
1.00 s is a comparison at fixed horizon and is unaffected. What does not stand is
reading any of those rows as a degradation curve.

**And the G() accuracy-against-horizon table in 1c is unaffected** -- checked
rather than assumed. Accuracy and F1 are counts over a fixed denominator (16
episodes x 4 feet) with no horizon-dependent normaliser, and its `jointRMSE`
column is already raw. That table does read correctly as a degradation curve.

## 1p. Where the error actually is, on a model without the defect

Three tables have now been used to answer "where is the error", and each was
invalid on the axis it was read along. **The metric is not the finding; the axis
is.**

| comparison | valid metric | why the other fails |
|---|---|---|
| across HORIZONS, one group, one model | **raw RMSE** | R^2 and normalised RMSE divide by an increment spread that MOVES with horizon |
| across GROUPS, one horizon, one model | **normalised RMSE** | raw RMSE compares rad against rad/s -- a units comparison |
| across MODELS, one group, one horizon | **raw RMSE** | the normaliser is a property of the DATA, and the two models' data differ |

### Normalised RMSE by group, UNWRAP model (1.0 = no skill)

| horizon | jpos | jvel | grav | contact | body |
|---|---|---|---|---|---|
| 0.02 s | 0.028 | 0.077 | 0.266 | 0.130 | 0.133 |
| 0.10 s | 0.020 | 0.030 | 0.116 | 0.118 | 0.201 |
| 0.29 s | 0.067 | 0.054 | 0.548 | 0.274 | 0.400 |
| 0.50 s | 0.095 | 0.083 | 0.792 | 0.266 | 0.431 |
| 1.00 s | 0.182 | 0.130 | **8.718** | 0.333 | **0.975** |

**At 1.0 s gravity is the worst group by an order of magnitude and body sits at
the no-skill line.** This is the first valid across-group table in the study: the
earlier R^2 version had a moving normaliser, the raw-RMSE version was a units
comparison, and both were computed on the wrapped baseline.

**This partly reverses 1o.** "Gravity is not the worst group, it is the most
nearly constant one" is true in ABSOLUTE terms and false as a statement about
where the error is. **1o used the raw table to answer a cross-group question,
which is the incomparability this section exists to name.** Both halves are true
and only one was said.

### What the unwrap changed, in raw RMSE (the valid cross-model metric)

| at 1.0 s | baseline | unwrap | |
|---|---|---|---|
| jpos | 0.0550 | 0.0481 | better |
| jvel | 1.0247 | 0.8751 | better |
| contact | 0.2589 | 0.2087 | better |
| body | 0.4256 | **0.1771** | 2.4x better |
| grav | 0.0263 | 0.0491 | 1.9x worse |

**Body's raw error is now monotone across horizons** (0.0099, 0.0239, 0.0518,
0.0799, 0.1771) where the baseline's was not -- confirming the non-monotonicity
in 1o was the wrap, since removing the wrap removed it.

### A third incomparability, which appeared inside this very table

**Normalised RMSE across MODELS is invalid**, and it misleads in the flattering
direction. Body reads 0.764 -> 0.975 normalised, "the unwrap made body worse",
while raw error went 0.4256 -> 0.1771, **2.4x better**. `sd(delta pitch)` in the
wrapped data includes the 2*pi jumps; in the unwrapped data it does not. **The
baseline was being divided by a spread its own defect manufactured.**

### Gravity: both readings are true and neither alone is complete

    absolute   RMSE 0.049 on a UNIT vector -- negligible to any consumer of the state
    relative   normalised 8.718 -- error many times the channel's own variation

**Which matters depends on the consumer.** A policy reading projected gravity sees
0.049 and does not care. A claim that the surrogate "models attitude" is refuted
by 8.718. Recording both rather than picking a verdict, because the single number
would be wrong for one of the two questions.

### What survives untouched

**The confound diagnosis never rested on this table.** `vel_body_x` is at R^2
0.822 at 1.0 s -- the surrogate predicts forward velocity WELL -- while the gate
reads corr 0.310 at 0.5 s. **The gate's failure was never a prediction-accuracy
failure**, which is why the action-blindness reading survives every revision of
the localisation. **What has been retracted and re-retracted is where the error
is, not what the gate is measuring.**

## 1q. Two instruments agree on the same horizon boundary

Neither was designed to check the other.

    the gate         0.5 s passes on gain; nothing separates at 1.0 s
    per-group table  body normalised RMSE 0.431 at 0.5 s, 0.975 at 1.0 s

**0.975 is the no-skill line.** The model has essentially no predictive skill on
body channels at one second, and it crosses into that between 0.5 s and 1.0 s --
the same boundary the action-sensitivity gate reports from an entirely different
construction (Chrono arm divergence against surrogate rollout divergence, versus
open-loop prediction error against ground truth).

**Two independent measurements agreeing on a boundary is much stronger than either
alone**, and this is the first time in the study that has happened.

It also re-establishes the localisation on the correct axis: **gravity 8.718 and
body 0.975 against 0.13-0.33 for everything else. Contact is not where the error
lives**, which was the original claim and survives every revision.

### Open observation, deliberately not chased

**Gravity is the one group the unwrap made worse in raw terms: 0.0263 -> 0.0491 at
1.0 s.** The unwrap touched only `pitch_rad`, and the gravity channels are a
backfill computed from the quaternion, so a change in how the model represents
attitude could plausibly propagate there. **But it is 1.9x on an already tiny
absolute error, and one model pair cannot distinguish a mechanism from retraining
noise.** Recorded so that if the excitation models show the same sign it becomes
worth a look; on this evidence it is not.

## 1r. Contact channels: accurate, not where the error is, and expensive to remove

Three instruments built for unrelated questions, agreeing on one channel group.

| measurement | value | what it says |
|---|---|---|
| normalised RMSE by group, 1.0 s | contact **0.333** against gravity 8.718, body 0.975 | contact is the third-best PREDICTED group |
| `G()` against the model's own prediction | **0.968** vs 0.849 | the model predicts contact better than exact geometry computes it |
| removing contact as an INPUT | err/signal **0.518 -> 1.190** | it carries 2.3x of the model's open-loop accuracy |

**"Well-predicted" and "load-bearing as an input" are independent properties of a
channel**, and this is the clearest statement of that in the study. W1 spent its
effort discovering the second from the other direction: a model that predicts
contact better than geometry computes it is a model that *uses* contact.

### The ablation is single-variable, and the obvious comparison is not

The comparison first reached for was the 40-channel WRAPPED baseline at 0.609,
which differs from the 36-channel model in **two** ways -- the channel set and the
wrap fix. The clean one is the unwrapped 40-channel model:

    40ch WRAPPED     err/signal 0.609   PARTIAL      <- two variables
    40ch UNWRAPPED              0.518   PARTIAL      <- one variable
    36ch UNWRAPPED              1.190   INCOMPLETE

**Verified rather than assumed:** the 36-channel preset is a strict subset of the
40-channel one, differing in exactly `foot_{fl,fr,rl,rr}_in_contact` and nothing
else. Same source corpus, same context, same architecture.

**Nobody designed this ablation.** It fell out of dropping the contact channels to
work around an excitation corpus that carries them as NaN.

## 1s. Declared before the numbers: how to read the excitation sweep

The 36-channel baseline is **INCOMPLETE** -- `err/signal` 1.190 at 0.5 s -- so a
paired comparison against it may be unmeasurable, as in Arm C. The pre-declared
response is one re-run at `--rel-sigma 0.05`, named in the gate's own docstring as
the answer to an apparatus-limited INCOMPLETE.

**`rel-sigma` makes a worse model measurable rather than making it better.** The
36-channel apparatus deficit is real and would persist.

**And a degraded baseline has more headroom, so excitation could help MORE at 36
channels than at 40.** A positive would then overstate what the full channel set
would show. That is the favourable-result-evades-scrutiny pattern arriving
structurally rather than through anyone's error, which is why it is written down
before the numbers exist.

**So the fallback has asymmetric value, and that decides the ordering rather than
cost:**

| outcome | reading |
|---|---|
| NEGATIVE at `rel-sigma 0.05` | **decisive.** If decorrelated actions do not help even a degraded model with headroom to improve, that is strong evidence the confound was not the binding constraint. |
| POSITIVE at `rel-sigma 0.05` | **provisional.** A licence to spend the two-hour 40-channel re-collection, not an endpoint. |

**A positive is not the result. It is permission to go and get the result** at the
channel set everything else in the study uses.

## 1t. The excitation sweep: what survives a measured floor

**The floor was placed third rather than last, and it caught the number that had
already been reported.** Two runs of one configuration, differing only in seed:

| metric | exc25 | exc25 seed 2 | base | verdict |
|---|---|---|---|---|
| `rollout_sel` | 0.2364 | **0.4167** | 0.4338 | **seed noise** -- 1.76x spread, seed 2 is the baseline |
| `val_loss` | 0.00662 | 0.00676 | 0.00811 | **real** -- 3% spread, 18% below baseline |
| `err/signal` 0.5 s | 0.287 | 0.432 | 1.401 | **real** -- spread 0.145 against a gap of ~1.0 |

**Same two runs, three metrics, opposite conclusions.** `rollout_sel` had a seed
spread that swallowed its effect; the apparatus ratio has one the effect clears
tenfold. **That is the argument for measuring a floor per metric rather than
once**, and the 1.8x `rollout_sel` improvement reported earlier is withdrawn.

### What holds

**Excitation data improves the surrogate's open-loop apparatus ratio by 3x to 5x**
and makes the 0.5 s cell measurable where the baseline is not -- base INCOMPLETE
at 1.401, every excitation cell PARTIAL. Gain passes at 0.5 s in all three:
1.156, 1.112, 1.154.

**And it does not need the volume.** The 1/100 cell reads 0.271, the best of the
three, matching the `val_loss` result and rank saturating at 2,000 rows.

### corr: a class claim, not a per-cell one

    exc25 0.466    exc25_seed2 0.641    exc25_lowvol 0.664

**Each interval straddles 0.5, so no cell passes.** But these are three
independently trained models -- different seeds, a 100x volume difference -- and
**two of three exceed the threshold with none below 0.45, mean 0.59.** That is
evidence about the class of excitation-trained models that three INDETERMINATEs
read separately would lose.

**The baseline cannot join this comparison.** Its corr reads 0.555, which looks
comparable, but it sits on an apparatus at 1.401 and is drift-dominated -- not a
number. Stated explicitly because a reader will otherwise notice it and wonder.

### The 1.0 s failure is now replicated

    40ch unwrap     corr 0.051 [-0.237, 0.330]   FAIL
    36ch lowvol     corr 0.022 [-0.263, 0.305]   FAIL

**Two independent models, different channel sets, different corpora, both
measurable at 1.0 s, both showing no action relationship at all.** The study has
been saying "usable at 0.5 s, not at 1.0 s" on the strength of 1.0 s being
UNMEASURABLE. It is now measured twice and it fails. **A positive finding
replacing an absence of evidence.**

### What is NOT established

**That this is the action decorrelation.** The 1/100 cell rules out that the
benefit scales with excitation VOLUME. It does not rule out that a small amount of
any varied data would do the same -- 1/100 of 5M is still 50,000 varied rows.
**The control that would separate them: take excitation STATES and replace their
actions with what the policy would have commanded**, re-confounding the data while
holding the state distribution identical. Nothing run so far does this.

**That action sensitivity improved.** What improved is accuracy, which makes the
gate readable. corr is indeterminate in every cell. **The gate becoming readable is
a precondition for the fine-tune question, not an answer to it** -- the endpoint is
a fine-tune inside an excitation-trained surrogate that survives transfer to
Chrono, and that is one experiment away.

## 1u. CORRECTION: the sweep's baseline comparison was confounded, and one metric was not a comparison at all

**Two of the three verdicts in 1t rested on a comparison against `base` that
differs from the excitation cells in TWO ways, not one.** The channel weighting
differs as well as the data.

### The mechanism: single-dataset weighting is an exact identity

`_build_channel_weights` computes `w_i = flat_std_i^2 / mean_d(std_{d,i}^2)`, where
`flat_std` is `self.metadata`'s `target_std`. When `channel_weight_datasets` is a
single dataset matching the metadata source, numerator and denominator are the
**same array** and `w_i = 1.0` identically, for every channel.

**The config generator set `channel_weight_datasets` to each cell's own
`train_mix`.** For the baseline that is walking alone, so the baseline trained
**unweighted** while every excitation cell trained under a real weight vector:

|  | data | weights |
|---|---|---|
| `base` | walking | 1.0 everywhere (identity) |
| `base_ownweights` | walking | 1.0 everywhere -- **same config as `base`** |
| `exc25`, `seed2`, `lowvol` | walking + excitation | `roll_rate` 0.058, `grav_body_y` 0.161, `vel_body_x` 1.163 |

**`base_ownweights` was built to remove a contamination that was never there.** It
is `base` under a different name, and its gate reproduces `base` to nine decimals
-- `gain 1.042596576`, `corr 0.554773709`, `d_model 0.051442282` -- on a
checkpoint file with a different md5. **Recorded as what it actually is: a
determinism check on the training pipeline, which we did not previously have.**
A run that reproduces another to nine decimals is either a determinism check or a
duplicate, and which one must be established before it is reported as either.

### `val_loss` is withdrawn outright, not merely confounded

The weights are mean-normalized (`weights * weights.size / weights.sum()`), so
they do not shrink the loss -- overall scale is fixed. **But mean-normalization
fixes the total, not the allocation.** `base` minimises an unweighted MSE;
`exc25` minimises one that downweights `roll_rate` 17x and `grav_body_y` 6x --
**the very channels 1p identified as where the error is.** Two different
objectives, each scored on itself.

> **WITHDRAWN: "val_loss 0.00662 vs 0.00811, real, 18% below baseline."** That is
> not one quantity measured twice. It is two quantities.

### `err/signal` is confounded differently, and less badly

The gate reads **raw** channels, so the ruler is common across cells. The confound
is in what the model learned, not in the measurement. `1.401 -> 0.287` remains a
real comparison of a real quantity between two models -- which differ in two ways.

> **WITHDRAWN: "the gap to baseline is ~1.0, far exceeding the seed spread."**
> The gap is data-plus-weighting.

**What is untouched:** the seed spread (0.145), the floor measurement, and the
low-volume result. Those compare cells that share a weight vector.

### The corrected cell, and its metrics declared before the numbers

`go2_mix36_base_bothweights`: walking-only data under exc25's exact weight vector,
**verified equal channel by channel**. It differs from `exc25` only in data.

|  | chanw walking | chanw walking+exc |
|---|---|---|
| **train walking** | `base` | `base_matchw` |
| **train walking+exc** | (not built) | `exc25` |

    base -> base_matchw     WEIGHTING effect     err/signal ONLY
    base_matchw -> exc25    DATA effect          err/signal AND val_loss
    base -> exc25           the confounded total already reported

**The two legs do not admit the same metrics.** The weighting leg compares models
trained under different weight vectors, so reporting it on `val_loss` would repeat
the error withdrawn above; it is readable only on `err/signal`. **Additivity is
likewise testable only on `err/signal`**, the one metric defined across all three
cells. The leg that carries the claim -- data -- is the one where both metrics are
admissible, and their agreement is a check worth having.

**If weighting carries most of the effect, the excitation corpus was not the lever.**

### Two instrument defects found while setting this up

**The fine-tune checkpoints were not loadable by anything that consumes them.**
The fine-tune scripts save `{"state_dict": ...}`; the collector, the gate and the
verdict harness all call `torch.jit.load`, which fails outright on it. v4 worked
only because someone exported it by hand and never captured the step. Now
`scripts/evaluation/export_finetuned_policy.py`, validated by re-exporting v4 and
reproducing the existing artifact bit-exactly in all 14 tensors.

**The verdict harness aborts on mixed-machine episode sets and had no flag for the
stratified design its own abort message recommends** -- only `--allow-foreign`,
which proceeds *including* the foreign half. Added `--own-machine-only`, which
drops foreign episodes and prints a standing warning that the result is a stratum
and not the pool. The patch is purely additive and guarded by the flag. **On this
host the cell holds 43 episodes, the denominator of the 38-of-43 control.**

**Both of the harness's path defaults are the other machine's**, including
`NEDM_CHRONO_PYTHONPATH`. That one is the dangerous member: this box carries a
second pychrono inside the conda env, so the wrong default can silently change the
physics build and break the bit-exact replay the paired design rests on. **Check
the md5, not the path.**

## 1v. The fine-tune verdict ladder: the excitation data is exonerated, and three explanations for the surviving result have died

One harness, one root, 43 episodes per arm, `--own-machine-only` on kyle-N7-B650E.
Every arm stopped on `target_dw`, so displacement is matched to three decimals.

| policy | surrogate | `\|\|dW\|\|` | surviving pairs |
|---|---|---|---|
| unmodified | -- | -- | **43 of 43**, median paired difference exactly +0.0000, CI [0,0] |
| v4 | 34-D, walking | 8.9032 | **20 of 43** |
| base36 | 36-D, walking only | 8.9003 | **0 of 43** |
| arm A | 36-D, walking + excitation | 8.9017 | **0 of 43** |

### The excitation data is not the cause

**`base36`'s surrogate never saw an excitation row and collapses identically to arm
A.** No weighting story can rescue the attribution: a walking-only 36-channel
surrogate produces total collapse on its own. The sentence *"the excitation
fine-tune destroys transfer"* was sealed as "arm A, pending arm B" for four hours
and never written.

### What survives, welded to no mechanism

> **At `||dW||` matched to three decimals, same script, same objective, same 5-step
> horizon: 34-D gives 20 of 43, and two independently trained 36-D surrogates each
> give 0.**

Three explanations for that have been proposed and withdrawn in one session. The
empirical claim outlived all three, which is the reason to state it alone.

### The failure mode is not a fall in the scored episode

The policy **cannot hold a stand**: 0.100 m against a standing 0.30-0.42 m at the
first recorded row, during prewalk, before the episode begins. Joint targets then
grow exponentially -- 4.15, 4.75, 5.58, 6.80, 9.66, 16.5, doubling about every two
rows -- to 1e30. Chrono clamps **torque** (`robot.py:130`) but nothing clamps the
**command**, which is why these episodes terminate on length rather than crashing.
**Ordering: fails to stand, then falls, then unbounded feedback through the
observation, then 1e30.** The first observable was none of the first three.

### WITHDRAWN 1: the surrogate is fooled

> ~~The surrogate is smooth and bounded where physics is not, so gradient ascent
> finds a policy with unbounded outputs that the surrogate scores as excellent.~~

The predicted **height** rises to 0.758 m, which is measured. The predicted
**reward** was inferred and is the opposite: +0.95 at 5 steps, **-1.53 at 20**.
The surrogate penalises the runaway heavily, using terms it already computes.

### WITHDRAWN 2: the window is narrower than the validity

> ~~The failure is visible inside the validity the surrogate already had, and the
> fine-tune's 0.05 s window is narrower than both.~~

Killed by the control, which should have been run before the claim:

```
  surrogate   policy        h=5      h=20                h=50
  exc25       exc25 FT     +0.95    -1.53   1.2e1     -7.94     2.4e1
  exc25       UNMODIFIED   +0.60    -4.0e8  3.4e6     -inf      3.1e19
  34-D        v4 FT        +1.06    -0.29   1.7e1     -2.5e15   1.4e10
  34-D        UNMODIFIED   +0.86    -1.5e8  1.5e6     -inf      5.2e18
```

**The policy that completes 43 of 43 in Chrono is the worst-behaved policy in every
surrogate.** The surrogate does not discriminate; its closed-loop rollout diverges
for everything past ~5-10 steps. `-1.53` is a good number, not a bad one.

**The conflation this exposed, which ran through the whole evening:**

    0.5 s validity from the gate      OPEN-LOOP, against RECORDED actions
    what the fine-tune consumes       CLOSED-LOOP, policy feeding its own actions back

Different properties. The gate certifies the first. **Closed-loop validity is under
~10 steps for every surrogate here, 34-D and 36-D alike, and nothing has ever
measured it.** So `--branch-steps 50` would optimise against a model already at
1e19 actions for the baseline policy, and the 5-step window may be about right for
the validity that actually matters.

### WITHDRAWN 3: 36-D permitted a sharper myopic optimum

v4 scores **+1.0645** at 5 steps, the *highest* of the three, not the lowest.

### A real defect, which is not the explanation

`go2_reward_terms.NOT_COMPUTABLE` drops `correct_base_height` at **-10.0**, the
largest weight in the reward, for the stated reason *"no pos_z_m in the 34-D
state"*. True for v4. **False for every 36-D and 40-D surrogate**, which carry
`pos_z_m` -- but the dict is a module-level constant encoding a per-run fact. The
34-D to 36-D difference is exactly two channels, `pos_z_m` and `vel_body_z_mps`,
and **neither is constrained**: one by the stale omission, one because `lin_vel_z`'s
converged weight is literally 0.0, so half the pose pair cannot be fixed by any
reward change.

`wrongly_omitted(state_fields)` now returns omitted terms whose required channels
are all present in the loaded state. **The rule it enforces was already written in
that file**, by whoever hit this with `torques` and `dof_power`: *name the missing
quantity and confirm it cannot be derived.* It was recorded and not re-run when the
state changed. **Recording a rule and enforcing a rule are different things.**

The term itself is **not implemented** -- `RT.terms()` takes no height argument, so
removing the dict entry would change nothing. It needs the upstream form and target
height from `go2_env.py`, and both-class validation against a policy known to stand
and one known to collapse, before any fine-tune trusts it.

### 17, 20 and 38 are one row under two predicates

    v4   42/43 recorded   COMPLETED 17, fell 4, diverged 21   PAIRS 20

**17 is a status category; 20 is surviving scored pairs; 38 is that same status
category on the unmodified policy.** Never in conflict, all from one run. This
harness reports 43 of 43 for the base policy because it counts pairs, not statuses.

### Three diagnostics that were printing all along

`reward: 10 computable terms, 4 omitted` on every fine-tune including v4's; the
2026-09-05 `constants.py` note diagnosing the height gap correctly and shipping only
the state half of the fix; and v3's own pre-registered falsification test on mean
`|raw action|`. **A diagnostic nobody reads is not a diagnostic**, and the fix is
that `wrongly_omitted` asserts rather than prints.

### The verdict is a divergence test with a tracking test bolted onto its survivors

Per-episode, using the harness's **own** `scored()` rather than a reimplementation,
with "diverged" = `max |raw policy action| > 1e3` (invariant from 1e3 to 1e6):

| policy | n | diverged | scored | diverged AND scored | neither |
|---|---|---|---|---|---|
| base | 43 | 0 | 43 | 0 | 0 |
| v4 | 43 | 23 | 20 | 0 | 0 |
| base36 | 43 | 43 | 0 | 0 | 0 |
| arm A | 43 | 43 | 0 | 0 | 0 |

**Zero exceptions in 172 episodes: an episode scores if and only if its commanded
actions stay bounded.**

So `"v4 completes 20 of 43"` means `"v4 does not diverge in 20 of 43"`. **The primary
axis has never measured control quality; it measures whether the policy blows up.**
And the paired tracking difference is then computed over the survivors -- **a sample
selected by the very failure being studied.** `go2-finetune-displacement-result.md`
already suspected the survivors were the easier episodes; this is the sharp form.

**Threshold choice is not free-floating:** base's median max is 4.547, so a bound at
10 sits inside its own operating range and produced a spurious 2/43. At 1e3 that
vanishes and the rates are identical at 1e6. **Quote a rate only where it is
invariant across decades.**

**Raw action space matters.** `targets = action * 0.25 + IMPORTED_DEFAULTS` in the
policy frame with `SIGN = -1` and a 12-element reindex, and the defaults differ per
joint. Comparing a recorded Chrono target against a surrogate-side action without
inverting all three compares different quantities -- which is how an earlier version
of this analysis produced a spurious fifteen-order-of-magnitude "separation" in which
v4 looked bounded. **v4 diverges in Chrono too, in 23 of 43.**

### ARM B: the attribution, single-variable at last

| policy | surrogate | `\|\|dW\|\|` | surviving |
|---|---|---|---|
| base | -- | -- | **43 of 43** |
| v4 | 34-D, walking | 8.9032 | **20 of 43** |
| base36 | 36-D, walking, identity weights | 8.9003 | **0 of 43** |
| arm A | 36-D, walking + excitation | 8.9017 | **0 of 43** |
| arm B | 36-D, walking, exc25's weight vector | 8.9017 | **0 of 43** |

**Arm A and arm B match to four decimals on displacement, share one channel-weight
vector, and differ only in whether the surrogate's training data included the
excitation corpus. Both are 0 of 43.**

> **The excitation data has no effect on fine-tune transfer. Not adverse, not
> beneficial -- no effect.**

Two independent walking-only 36-D arms reach the same zero, one under identity
weights and one under exc25's, so no weighting story survives either.

**Every 36-D arm is 0; the 34-D arm is 20.** Three replicates against one, at
matched displacement, same script, same objective, same 5-step horizon. That
sentence has now outlived four proposed mechanisms.

### The excitation corpus does buy what it was collected to buy

Same three checkpoints, evaluated on both validation splits:

| surrogate | walking | excitation |
|---|---|---|
| base_matchw (0%) | 0.008933 | **0.741947** |
| exc25 (25%) | 0.006657 | **0.123441** |
| exc50 (50%) | 0.011860 | **0.128190** |

**A 6x improvement on the excitation distribution.** The sweep's walking-only
validation set was structurally incapable of measuring it: with 55.7% of excitation
rows beyond walking's 99th percentile, degradation on walking is close to what the
design guarantees, and the benefit is invisible by construction.

**The two columns are NOT one instrument and the magnitudes must not be traded off
against each other.** `val_loss` comes from the **primary** loader built from
`processed_root`; `validation_datasets` only adds **extra** loaders keyed
`val_<name>_loss`. Overriding the former and reading the latter returns *the same
number for every set* -- caught only because two different validation sets produced
identical values, the same signature that exposed the `base_ownweights` duplicate.
The walking column here also **inverts** the dose-response ordering computed at
training time, so a trade curve needs both sides re-measured on one loader.

### WITHDRAWN: the walking-split dose-response

> ~~Adding excitation data costs walking-split accuracy monotonically with dose:
> 0.00613 / 0.00662 / 0.00867.~~

**`checkpoint_metric` is `rollout_sel` in all three runs, so `best_val.pt` is not the
best-by-val_loss checkpoint.** The two "disagreeing" loaders never disagreed -- they
were measuring different epochs:

| run | saved ckpt | min val_loss over 80 epochs | val_loss AT the saved epoch |
|---|---|---|---|
| base_matchw | ep37 | 0.00613 @ep78 | **0.00893** |
| exc25 | ep76 | 0.00662 @ep74 | **0.00666** |
| exc50 | ep24 | 0.00867 @ep78 | **0.01186** |

The right-hand column reproduces the extra-loader evaluation to four decimals. **The
dose-response was computed from per-run epoch-wise minima of a metric that selected
none of the saved models** -- a selected extreme of a noisy series, describing
checkpoints that do not exist.

**On the artifacts that do exist the trend is non-monotone**, with the 25% cell
lowest: 0.00893 / 0.00666 / 0.01186.

**And the selection compounds it.** `rollout_sel` is the metric withdrawn in 1t as
seed-noise-dominated (0.2364 vs 0.4167 across two seeds of one config). It set the
saved epochs to 37, 76 and 24 -- the 0% and 50% arms stopped less than half way
through a run the 25% arm nearly completed, on that noise. **Nothing about
walking-split accuracy across these three cells is currently reportable.**

**The excitation reversal is unaffected**: both columns come from the same loader on
the same artifacts, so `0.00893 -> 0.74195` against `0.00666 -> 0.12344` is one
comparison, and the 6x stands.

### What replaces it: a common epoch, from `last.pt`

All three ran 80 epochs and all three saved `last.pt` at epoch 80 -- a fixed epoch,
free of `rollout_sel` selection entirely. Verified from `metrics.jsonl` on this box:

|  | val_loss @ep80 | vs the seed floor (0.00020) |
|---|---|---|
| 0% excitation | 0.00660 | -- |
| 25% excitation | 0.00662 | difference 0.00002 = **0.1x** -> NULL |
| 50% excitation | 0.00901 | difference 0.00239 = **12x** -> REAL |

> **At 25% excitation, walking-split accuracy is indistinguishable from
> walking-only. At 50% it is materially worse.** A null followed by a penalty,
> not a monotone dose-response.

**Caveats attached:** n=1 per cell, and the floor is borrowed from the excitation
family rather than measured on the 0% arm. A second 0% seed is training.

### `best_val.pt` is not selected by val loss, in ANY run in this project

`checkpoint_metric` defaults to `val_loss` but every go2 config sets it to
`rollout_sel` -- **the metric withdrawn in 1t as seed-noise-dominated.** The
filename says otherwise and there is no error, the same class as
`val_tracking_mse` naming a negated reward.

**So every `best_val.pt` here is a model chosen by a metric the sweep's own floor
says cannot support selection**, and the saved epochs show what that costs: 37, 76
and 24 out of 80. The 0% and 50% arms were frozen less than halfway through a run
the 25% arm nearly completed.

**Prefer `last.pt` for any cross-run comparison** until selection is fixed. The
fine-tune arms were all trained inside `rollout_sel`-selected surrogates, which is a
**shared** defect rather than a differential one -- so the arm A / arm B attribution
holds, but it is stated here rather than assumed.

### The 34-D replicate, and a confound caught before the verdict

The surviving claim -- *34-D gives 20 of 43, every 36-D arm gives 0* -- rests on
**three independent 36-D checkpoints against ONE 34-D checkpoint (v4)**, trained
weeks earlier under a different config generation and merely re-scored since.
Re-running v4 confirmed the *evaluation* reproduces; it did not replicate the
*training*. That is n=1 on the arm making the claim -- the structure of the
withdrawn `rollout_sel` result.

**The first attempt at the replicate reintroduced the confound it existed to
remove.** It trained on `go2_corrected_34d_excl`:

| | `go2_corrected_34d_excl` | `go2_walking_36d` |
|---|---|---|
| processed | 2026-09-05T06:33 | 2026-09-06T20:50 |
| `circular_unwrapped` | **ABSENT** | `['roll_rad','pitch_rad']` |
| `processing_provenance` | **ABSENT** | commit `b42e3ebb` |

**38 hours apart, across the circular-unwrap fix** -- which alone moved `err/signal`
from 0.609 to 0.518. Pairing them would have confounded the channel set with the
wrap fix, *a confound this document already records about an earlier comparison*.

**The config asserted that no 36-D dataset reference survived. That was true and
insufficient**: it checked what the run pointed AT, not whether the two things being
compared were otherwise identical.

Now gated by `scripts/preprocess/assert_datasets_differ_only_by_channels.py`, which
requires matching raw roots, `dt_s`, `contact_mode`, `circular_unwrapped`,
action/rollout fields, provenance commit and split contents, and **treats `<ABSENT>`
as a failure rather than a match** -- a property one dataset does not record cannot
be asserted equal. It runs as a hard stop before training.

**Not a third variable:** the `_excl` exclusion lives in the raw corpus, and both
datasets report identical episode *and* transition counts (2382/621, 8,961,196).

**Recorded before the replicate's result exists:** v4's surrogate trained on
`go2_corrected_34d_excl`, the pre-unwrap dataset. **If the fresh replicate lands at 0
while v4 sits at 20, old preprocessing is a live explanation and must not be selected
after seeing the number.**

### Two stability measures, both anti-correlated with plant performance

**The divergence growth constant is a policy-specific eigenvalue-like number.**
Fitting `log |raw action|` against time over a fixed window (1e12 to 1e20, identical
for every episode):

| policy | verdict | lambda (1/s) | episodes with lambda <= 0 |
|---|---|---|---|
| base | 43 of 43 | -- | **43** |
| v4 | 20 of 43 | **41.8** | **20** |
| base36 | 0 of 43 | **31.0** | 0 |
| arm B | 0 of 43 | **33.5** | 0 |
| arm A | 0 of 43 | **16.8** | 0 |

**The `lambda <= 0` column is exactly the survivor count** -- a third independent
route to the same column, after `never` and `scored`.

It is **not** a fit artifact, and the check that would have shown one was run: a
suspicion that lambda measured `fixed_log_range / time_to_overflow` was tested by
refitting over a bounded window and **refuted** -- the values reproduce. It is also
**disturbance-independent** (sbel-pc measured arm A at 16.8471 across 0-80 Nm) and
**not a selection effect**: v4's lambda on its 21 diverging seeds versus arm A's on
*those same seeds* gives a paired difference of **+24.894**, against +25.0 unmatched.

**But it does not order with performance.** v4 is the fastest diverger and the best
arm. *Rare and violent* versus *reliable and gentle* -- and no account of it.

### rho(J): the same reversal a third time

`J = d action_{t+1} / d action_t` through the observation, plant held fixed, by
autodiff over 96 recorded in-distribution states:

| policy | verdict | p50 rho | max | fraction rho > 1 |
|---|---|---|---|---|
| base | 43 of 43 | 0.836 | **2.581** | **0.344** |
| v4 | 20 of 43 | 0.550 | 0.796 | 0.000 |
| arm A | 0 of 43 | 0.717 | 0.987 | 0.000 |
| base36 | 0 of 43 | 0.750 | 1.045 | 0.010 |

**The working policy is expansive in a third of sampled states; all three failing
policies are contractive almost everywhere.** Arm A never exceeds 1.

**Scope:** this is the *direct* action-feedback path only -- state and command
frozen, history fixed. The loop that actually diverges runs through the plant, which
is the leg every other elimination points at. So this is a fourth **elimination**
(the direct path cannot be the mechanism, agreeing with sbel-pc's `prev_actions`
isolation converging) rather than an explanation.

**Three local stability measures now point the wrong way:**

    surrogate closed-loop action magnitude   base worst (18.6-24.9), fine-tunes 4.9-8.4
    surrogate predicted reward               base lowest at every horizon
    rho(J) on the nominal trajectory         base the only one exceeding 1

**Every measure that says a policy is locally well-behaved says the failing
policies are the well-behaved ones.** That pattern is the result. Inventing a
mechanism to explain it away would be the fifth reframe to die in one session.


### CORRECTION: arm B's two empty episodes are the extreme, not missing data

Arm B is the only arm with empty episodes (0/0/0/0/2 across the five). Both were
re-run and both reproduce deterministically:

    vel_step_64   rc=1  rows=0
    vel_step_81   rc=1  rows=0
        ValueError: episode produced zero recorded rows: nothing to summarise

Recording begins at `warmup_s`; **these episodes never reached it.** The policy
destroyed the run during prewalk so completely the simulation ended before one row
was logged.

**So arm B's instant-divergence count is 33 of 43, not 31 of 41.** Excluding them as
"missing" biased the worst arm's instant count DOWNWARD by two. `scored()` returns
None for both, so `0 of 43` is unaffected -- but the classification was wrong.

**The general form, which the sentinel rule did not cover:** a failure path that
returns a sentinel is a silent denominator shrink, *and the dropped records are not
a random sample* -- they are the tail in the direction being measured. Count what
ran, and check which direction what did not run failed in.

### Which Chrono build produced the five verdicts

**All five, proven rather than asserted: the source build.** Each run's replay check
re-runs a baseline episode and requires a bit-identical physics digest against the
original, which was collected under `/home/kyle/chrono-build/bin` (md5 `d1d0bd0a`).
The conda env holds a different binary (`8e9e3865`). **25 of 25 replay checks passed
across the five runs, which a different build cannot do.**

This was luck backed by a check, not discipline: `NEDM_CHRONO_PYTHONPATH` is not in
any shell rc file, and the harness's default points at the *other* machine's layout.
`run_go2_finetune_verdict.py` now prints the resolved pychrono path and md5 before
the replay check, and says so loudly when the path does not exist -- turning "which
build produced these numbers?" from unanswerable-from-artifacts into a grep.

### CORRECTION: rho(J) is state-sensitive, so point evaluations of it mean little

Two boxes computed rho at "the reset pose" and disagreed across the stability
boundary the conclusion rested on -- **base 0.5572 here, 1.3652 on sbel-pc**. Both
are probably correct measurements on *different constructed vectors*: neither box
lifted the reset state from a simulation. **"The reset pose" named two different
states** -- the denominator rule applied to a state rather than a sample.

Testing whether that explains it, by perturbing joint angles around the reset pose
(40 draws per level):

| joint noise | base min/med/max | frac>1 | arm A min/med/max | frac>1 |
|---|---|---|---|---|
| 0.02 | 0.499 / 0.551 / 0.602 | 0% | 0.449 / 0.490 / 0.526 | 0% |
| 0.10 | 0.443 / 0.533 / 0.652 | 0% | 0.452 / 0.531 / 0.702 | 0% |
| 0.20 | 0.453 / 0.555 / **1.016** | **5%** | 0.457 / 0.593 / 0.841 | 0% |

**Base crosses 1 under joint perturbation alone; arm A never does.** And on nominal
walking states base spans **0.44 to 2.58** while every fine-tune sits in a narrow
band under 1. **A single point evaluation of base can land anywhere in that range**,
so 0.5572 and 1.3652 are both inside it and the disagreement is what a
state-sensitive quantity measured at two different states should produce.

> **WITHDRAWN: "nothing exceeds 1 at the reset pose, therefore entry needs the
> plant."** That was a categorical conclusion from a point measurement of a quantity
> since shown to be state-sensitive. The plant may still be required; this does not
> establish it.

**What survives, independent of which vector is right:**

> Across nominal walking, near-reset, and under perturbation, **base's
> action-feedback gain is broadly state-dependent and exceeds 1 in part of the
> space, while all three fine-tuned policies are confined to a narrow band below
> it.**

**And rho is invariant to body height** -- identical to three decimals from 0.10 to
0.40 m -- so it is not tracking postural stability at all. It measures how strongly
a policy responds to its own previous action, and **fine-tuning both damped that
response and flattened its state-dependence.**

My reset vector, published for recomputation: `sha256[:16] = 0cc8011915d3a9ff`,
36-D float32, surrogate field order, joint positions = policy defaults mapped to the
Chrono frame, everything else zero, `grav_body_z = -1.0`, `pos_z_m = 0.30`,
`cmd = zeros(3)`, `prev = zeros(12)`, fresh `initial_history`.

### REFUTED: "a fine-tune can only degrade what its surrogate can represent"

The account: v4's 34-D surrogate carries no `pos_z_m`, so the fine-tune has no
gradient on height and height is preserved by inability; 36-D surrogates can
represent height, the reward does not constrain it, so it degrades.

**v4 sinks.** `--log-warmup`, three seeds, identical through row 125 because the
pose ramp is not under policy control:

| policy | seed | r125 | r150 | r170 | r200 | verdict |
|---|---|---|---|---|---|---|
| base | 0/1/2 | 0.3026 | 0.379/0.376/0.374 | 0.382/0.379/0.378 | 0.376 | 43/43 |
| **v4** | 0/1/2 | 0.3026 | **0.2735/0.2733/0.2732** | **0.1497/0.1497/0.1504** | 0.188 | **20/43** |
| arm A | 0/1/2 | 0.3026 | 0.304/0.303/0.302 | 0.209/0.207/0.204 | 0.160 | 0/43 |

**v4 sinks FURTHER and FASTER than arm A and survives anyway** -- 0.150 against
0.207 at row 170, reproducible to four decimals across seeds. Independently
reproduced on sbel-pc, which measured v4's minimum at 0.131 against base's 0.302.

> **Every fine-tune sinks. Only the 36-D ones diverge. Height is a shared symptom,
> not the discriminator.**

**And the fine-tune degrades height with no height channel anywhere in the loop** --
v4's surrogate cannot represent it. So the degradation is a side effect of moving
the policy at all, **which means restoring `correct_base_height` may not fix it.**
That is now a live possibility to test with the two-second readout rather than
discover after implementing a signature change.

**This voids the 2x2's representability-derived prediction** (that the 34-D
replicate would preserve height by inability). The 2x2's original reading stands on
its own terms and is not re-derived from a dead account.

### PRE-REGISTERED, before the replicate's fine-tune exists

On the real first-call observation `06f73542…` (hash verified), one forward pass:

| policy | rho | max abs action | verdict |
|---|---|---|---|
| v4 | 0.4591 | 3.9550 | 20/43 |
| base | 0.4919 | 2.4569 | 43/43 |
| arm A | 0.6522 | 4.4515 | 0/43 |
| base36 | 0.7139 | 6.3754 | 0/43 |
| arm B | 1.0225 | 6.8732 | 0/43 |

**Both quantities are monotone with the outcome across five policies**, with a gap
between survivors and failures at rho 0.492-0.652 and at max abs action 3.96-4.45.

**This is causally inert** -- rows 0-125 are byte-identical across policies because
the pose ramp is not under policy control, so what a policy emits at row 0 is never
executed. It is a pure measure of how differently each policy responds to that
state. At n=5, fitted after the fact, it is suggestive and nothing more.

> **PREDICTION, recorded before the 34-D replicate's fine-tune exists:**
> compute rho and max abs action for it on `06f73542…`.
> **rho < 0.55 and max abs action < 4.0 -> predict it SURVIVES (non-zero verdict).
> rho > 0.65 or max abs action > 4.4 -> predict it gives 0 of 43.**
> Between the gaps: no prediction, and that is itself informative.

**This is the only out-of-sample test available.** It costs seconds and the answer
arrives before the verdict does.

### Registered BEFORE the replicate's verdict is read: what a 0 obliges

The 2x2 as it stands:

|  | pre-unwrap dataset | post-unwrap dataset |
|---|---|---|
| **34-D** | v4 = **20 of 43** | replicate = ? |
| **36-D** | (empty) | base36 = **0 of 43** |

An earlier registration committed to filling the empty cell if the replicate
returns 0. **That ordering is now amended, and the amendment is recorded before the
number exists.**

**If the replicate returns 0, run a DIRECT REPLICATION OF v4 first** -- 34-D trained
on v4's own dataset (`go2_corrected_34d_excl`) with today's pipeline -- and the
fourth cell second.

```
  replication ~ 20  -> the DATASET is the variable, v4 reproduces, and the fourth
                       cell then asks whether that holds at 36-D. Clean design.
  replication = 0   -> v4 does not reproduce on its own data with current code.
                       Its 20 was the pipeline vintage or a lucky checkpoint, and
                       the entire 34-vs-36 result rests on one unreproducible run.
```

**The second outcome is the one worth knowing most, and the fourth cell cannot
reach it: a replication tests reproducibility, the fourth cell assumes it.** v4's
dataset differs from the current one in more than the unwrap -- it predates the
provenance instrumentation entirely -- so "pre-unwrap" labels a vintage, not an
isolated change, and the 2x2's column heading is weaker than it looks.

**Also registered:** if the verdict produces no surviving pairs, no `--summary-json`
is written and the family split is **UNAVAILABLE, not zero.** v4's run wrote none,
which is why this is stated in advance rather than discovered.

### The dose-response, third version, with a per-episode interval

Withdrawn twice: first as epoch-wise minima of checkpoints never saved, then as a
0.57% single-family prefix. Held a third time pending an error bar. **It now has
one, and it survives.**

`last.pt` for all three, full validation set (2,244,752 windows over **615 of 621**
episodes -- the other six are shorter than the 128-step window and yield none),
bootstrap resampling **whole episodes**, 5,000 draws:

| contrast | stratum | n eps | difference | 95% CI |
|---|---|---|---|---|
| exc25 − base | straight | 227 | **+0.001067** | [+0.000723, +0.001416] |
| exc25 − base | turning | 388 | **+0.000926** | [+0.000603, +0.001312] |
| exc25 − base | **ALL** | 615 | **+0.000978** | [+0.000737, +0.001251] |
| exc50 − base | straight | 227 | **+0.003269** | [+0.002668, +0.003920] |
| exc50 − base | turning | 388 | **+0.003759** | [+0.002968, +0.004733] |
| exc50 − base | **ALL** | 615 | **+0.003578** | [+0.003015, +0.004252] |

**Every interval excludes zero. Monotone in dose, and uniform across strata --
not a cancellation.**

> **Adding excitation data degrades one-step accuracy on the walking validation
> distribution, monotonically with dose: +0.00098 at 25% and +0.00358 at 50%,
> both significant at the episode level.**

**The interval is the point.** Window counts would have given ~2.24M "observations"
and an interval roughly 60x too narrow; windows within an episode overlap and share
a trajectory, so the effective n is 615. **The prefix result proved that before this
was run** -- five arc episodes read 30x different from the full arc stratum, which
is between-episode variance dominating.

**NAME THE ESTIMAND -- the two defensible summaries differ by 25%:**

    pooled over WINDOWS    base 0.004813  exc25 0.005594   diff +0.000781
    mean over EPISODES     (the bootstrap above)           diff +0.000978

Long episodes carry more weight in the first; every episode carries equal weight in
the second. **The figures above are per-EPISODE** -- "how much worse on a typical
episode" -- which is the estimand a bootstrap over episodes estimates. The
per-window figure answers "how much worse at a typical timestep" and is what the
trainer optimises. Neither is wrong; **two numbers 25% apart under one name is how
the preceding four hours went.**

**Scope, stated because this has been overstated twice:** one-step prediction error
on the WALKING distribution only.

### The three measurements differ in PRECISION as much as in direction

| measurement | n | interval | |
|---|---|---|---|
| one-step, walking dist. | 615 episodes | [+0.00074, +0.00125] | **tight** |
| one-step, excitation dist. | same checkpoints | ~6x better | large effect |
| closed-loop tracking | 16-20 episodes | [-0.0593, -0.0104] | **wide**, and does not exclude the -0.020 criterion |

**They are NOT three equally-established results.** The two one-step measurements
are tight and point opposite ways *by distribution*; the closed-loop measurement is
imprecise and is being re-measured on ~300 fresh episodes. Whether it belongs in the
comparison at all depends on that run.

### What the one-step results actually say, and the sentence that survives

Stated in full they are not surprising: **adding out-of-distribution data makes the
model better there and worse here, monotonically in dose.** That is what mixing data
does; these numbers measure it cleanly for the first time in this project.

**The finding is that neither one predicts the closed-loop behaviour.**

> **One-step accuracy, on any distribution, does not predict what a policy trained
> inside that model will do in the plant.**

Six instruments have now hit that wall: the gate, the apparatus ratio, rho(J),
the flattening measure, lambda, and this. **It is the one claim tonight with a
tight measurement behind it rather than a string of nulls.**

### Registered BEFORE the v4 replication's verdict: how to read a positive result

After nine dead mechanisms, a non-zero result will be the most scrutinised number of
the session, so its reading is fixed now.

```
  ~20 of 43  -> v4 REPRODUCES on its own data with current code. The variable
                separating v4 from every other arm is its DATASET, not its channel
                count and not the pipeline. The fourth cell then becomes meaningful
                and asks whether that holds at 36-D.
  ~0-2       -> v4 does NOT reproduce. The one non-zero fine-tune result this
                project has is a single unreproducible run, and every comparison
                that used it as the positive reference -- including the entire
                34-vs-36 line -- rested on it.
  in between -> no clean reading; report the number and the margin, claim nothing.
```

**A ~20 would NOT establish that the unwrap is the variable.** v4's dataset predates
the provenance instrumentation entirely, so it differs from the current one in an
unknown bundle of ways; "pre-unwrap" names a vintage. **It would establish
reproducibility and localise the cause to the dataset, which is a strictly weaker
and more defensible claim than the one the 2x2's axis label implies.**

### v4 DOES NOT REPRODUCE, and both blind predictions were correct

34-D trained on **v4's own dataset** (`go2_corrected_34d_excl`) with today's
pipeline. Predictions committed at `15f4d69` before the verdict was opened.

```
  REGISTERED          rho 0.9264, |a| 5.3074   -> 0 of 43
                      k* = 0.600               -> 0 of 43
  ACTUAL              0 of 43, 43/43 diverged, median max|raw action| 6.46e34
                      harness: NOT MEASURABLE -- every treated episode failed the predicate
```

**Both predictors correct, and on the literal count this time rather than only the
class.** Second out-of-sample test; the first (replicate, predicted 0, actual 2) was
correct on class only.

### The ladder, all at nominal gain, one ruler

| arm | dataset | surviving | diverged >1e3 | k* |
|---|---|---|---|---|
| base policy | -- | **43 of 43** | 0/43 | 1.450 |
| **v4 (original)** | v4's, OLD pipeline | **20 of 43** | 23/43 | 1.075 |
| **v4 REPLICATION** | **v4's, current pipeline** | **0 of 43** | **43/43** | **0.600** |
| replicate 34-D | current | 2 of 43 | 41/43 | 0.925 |
| base36 | current | 0 of 43 | 43/43 | 0.906 |
| armA / armB | current | 0 of 43 | 43/43, 41/43 | 0.925 |

> **v4's 20 of 43 does not reproduce on v4's own data with current code.** The one
> non-zero fine-tune result this project has is a **single unreproducible run**.

**Everything that used v4 as the positive reference rested on it:** the 34-vs-36
line, the 17-and-20-of-43 figures, and the postmortem's reading of v4 as the
configuration that works.

**Unregistered observation, labelled as such:** `k* = 0.600` is the lowest margin of
any arm -- below every 36-D arm. The current pipeline given v4's *exact data*
produces less stability margin than the current pipeline given any other data. **So
the dataset is not the variable either.** What changed between v4 and now is the
pipeline, and that is not something the 2x2 was built to test.

**The fourth cell is not run.** It was gated on the replication returning ~20;
nothing reproduces, so the axis is dead.

**Verified before this number joined the ladder:** scoring equivalence across two
harness versions (258 episodes, six arms, 0 disagreements), `NEDM_ACTION_MULT`
absent from the verdict process by `/proc` read, pychrono `d1d0bd0a` in the
verdict's own output, and replay 5/5 bit-identical.

### k* has a measurement noise floor of at least 0.15, found by accident

Bisection cell 1 fine-tuned inside v4's **own** surrogate with current code and
reproduced v4's weights exactly -- **14/14 tensors identical, `||cell1 - v4|| =
0.000000`**, on both the state dict and the exported policy. So its `k*` run
measured **the same policy twice**:

```
  v4       k* = 1.075   "STABLE as deployed (k* > 1)"
  cell 1   k* = 0.925   "UNSTABLE as deployed (k* < 1)"
```

> **A spread of 0.15 on identical weights, straddling the threshold the predictor
> classifies against.**

**What this invalidates:** the gap between v4 (1.075) and armA / armB / the 34-D
replicate (0.925) **is the noise floor**, so `k*` does not separate them. `base` at
1.450 and the v4 replication at 0.600 lie outside the band and survive; the middle
of the ladder does not.

**The threshold is still principled and the resolution is not.** `k*` is read off an
8-episode screen at each of six gains with the 0.5 crossing interpolated -- a
binomial on 8 samples cannot resolve a crossing between rungs 0.15 apart. Fixing it
means more episodes per rung or a finer grid near the crossing.

**Effect on the blind tests:** both remain correct, but for the first (replicate)
`k* = 0.925` sat inside the noise band and its call was luck; the v4 replication's
0.600 is far outside it and stands. **rho/|a| is unaffected** -- it is autodiff over
weights, and cell 1 demonstrated it returns identical values on identical weights.

**Nobody designed this check.** It fell out of a bisection cell that happened to
reproduce a policy bit-for-bit, and only because the identical rho and |a| were
questioned rather than accepted as "the fine-tune reproduces."

### Scope note: which commits are inert, and for which runs

Six commits touched the training path after v4's surrogate was trained
(2026-09-05T01:47). Three are preprocessing-only (`c699338`, `8633631`, `3f5a86c`)
and **inert for the bisection cells only because those consume v4's
already-processed dataset.** That is a property of the run, not of the commits: **a
cell using current preprocessing would carry all three**, including the circular
unwrap. Of the three that touch `trainer.py`, `212a787` is a stdout print inside a
try/except and `96a4811` is an additive guard that either raises or does nothing
(it never raised -- every run completed), leaving `519ad1d`'s `input_noise_sigma`,
which is gated behind `if self.input_noise_sigma > 0.0`.

### REGISTERED before the repaired k* re-measurement

The broken criterion **under-detected failure at high gain** -- it scored falls with
bounded commands as passes, which is why cell1's k=1.50 rung read 0/8 and now reads
8/8. **So every old `k*` is biased UPWARD**: the 0.5 crossing appeared to sit at a
higher `k` than it should, because failures above the true crossing were scored as
survivals.

> **Registered: repaired `k*` values should come in LOWER than their old
> counterparts, across the board. If any comes in higher, something else is wrong.**

**And the old ladder's status is stronger than "0.15 resolution":**

> Every `k*` in the ladder -- base 1.450, v4 1.075, armA/armB 0.925, base36 0.906,
> v4rep 0.600 -- was measured on a criterion now known to mis-score an entire failure
> mode. **They are not noisy readings of the right quantity; they are readings of a
> different one.** `base` and `v4rep`, which I said survived the noise floor, do not
> survive this: both were read off curves the broken criterion could have turned over
> anywhere.

**The deciding measurement is one sweep, not six.** Cell1's weights are v4's, so
re-running the repaired sweep on v4's checkpoint gives a second repaired reading of
the *identical* policy:

    agrees with cell1's 0.925   -> the 0.15 spread WAS the criterion; k* is
                                   repeatable and re-measuring the ladder is worth it
    differs by ~0.15 again      -> 8-episode rungs are the limit; the repair fixed
                                   monotonicity, not resolution, and re-measuring
                                   the ladder at this rung count is pointless

**This is also the first *designed* repeatability check `k*` has had** -- the first
came free from a bisection accident.

### REGISTERED: does k* discriminate at all? v4 vs armA, n=40 per rung

**The ladder was never worth rebuilding at n=8.** Binomial sd at p=0.5 is
`sqrt(0.25/8) = 0.177`, and the rate moves only ~0.25 across the whole 0.85-1.00
bracket the crossing is interpolated inside -- **so sampling noise is 71% of the
bracket, and two genuinely different arms will often return the same k\*.** That is
the identical-number tell again: agreement produced by the instrument.

Resolving `dk* = 0.05` needs the rate pinned to 0.083, i.e. **n ~= 36 per rung**.

**So run the extreme pair, not six sweeps:**

    v4     scores 20 of 43     the only arm that partially works
    armA   scores  0 of 43

**If k\* cannot separate those two it cannot separate anything, and the other four
sweeps are wasted. Rungs 0.50 / 0.70 / 0.85 / 0.90 / 0.95 / 1.00 / 1.20 / 1.50** so
the crossing bracket is not a single step, at 40 episodes per rung.

```
  intervals DISJOINT    -> k* discriminates; rebuild the full ladder at this resolution
  intervals OVERLAP     -> k* does not distinguish 20/43 from 0/43, and the stability
                           half needs a different instrument, not more sweeps
```

**Per-rung standard errors are reported, not just the interpolated crossing** -- the
crossing hides the sampling noise that produced it, which is what let 0.925 look
exact when both bracketing pairs happened to straddle 0.5 symmetrically.

### REGISTERED before cell 2's verdict returns: the intermediate branch

The original registration covers only the ends:

    ~20 of 43   -> input_noise_sigma IS the regression
    ~0-2        -> it is a correlate
    ~5-15       -> NOT COVERED

**Registered now, while the verdict is still running:**

> **An intermediate count means the key is PARTIALLY CAUSAL** -- removing it recovers
> some of the gap and not all. That is a third claim, distinct from both branches,
> and its follow-up is a **dose-response across `input_noise_sigma` values**, not
> another binary cell.

### And the abstention does NOT predict an intermediate result

I wrote that cell 2's landing in the no-prediction gap suggested "intermediate, not a
restoration." **That is wrong and it is the reading pre-registration exists to
prevent.** Abstention means **the features do not place it**, not that the answer
lies between. Nothing was ever measured inside the gap; it is empty space between two
labelled clusters, and the verdict could be 20 or 0 with equal consistency.

**There is a concrete reason the axes cannot order this**, and it is visible in the
only two points whose outcomes differ:

| policy | rho | max abs action | verdict |
|---|---|---|---|
| v4 | 0.4591 | 3.9550 | 20 of 43 |
| base | 0.4919 | 2.4569 | 43 of 43 |

**On `rho`, base is HIGHER than v4 and does BETTER. On `|a|`, base is LOWER and does
better.** The two features order the survivor class in *opposite directions*. Cell2-A
sits above v4 on both -- which argues "better" by one axis and "worse" by the other.

**Every other reference point is 0 of 43**, so the failure class carries no
within-class ordering either.

> **The predictor separates survivors from failures and has essentially no resolution
> on "how many of 43" -- which is the quantity both branches are stated in. It was
> built to classify and is being read for a count.**

### CELL 2 SEED A SCORED: 22 of 43 -- lands on the registered "~20" branch

    control (base policy)                              43/43
    v4        original, trained pre-519ad1d, NO noise  20/43
    v4rep     same dataset, same 34-D, noise ON         0/43   <- matched counterfactual
    cell2 A   same dataset, same 34-D, noise OFF       22/43   <- this measurement

**Scored against the branch registered at `5c669d3`, unamended: `~20 of 43 ->
input_noise_sigma IS the regression`.** Removing the noise recovers v4's score from a
with-noise counterpart at zero.

**The flag was verified active rather than assumed** -- final surrogate `val_loss`
0.007559 with noise against 0.002806 without, a 2.7x gap, so this is not a repeat of
the bisection cell that reproduced a policy bit-for-bit. The three fine-tuned policies
are also distinct by md5.

**The recovery is spread across all five command families, not carried by one:**

    family      v4 (no noise)   cell2 A (noise off)   v4rep (noise on)
    arc               2                  4                  0
    constant          7                  7                  0
    vel_step          6                  3                  0
    weave             3                  5                  0
    yaw_step          2                  3                  0

#### Correction: the 2/43 arm is NOT the counterfactual

`finetune_go2_34d_replicate` (2/43) trained on **`go2_mix34_base_replicate`** -- the
base dataset, not v4's. I nearly read it as a second with-noise draw on the matched
dataset. **It differs in two variables and licenses nothing.** The matched with-noise
arm is `v4rep` at 0/43, and it is n=1.

#### What can still overturn this, and it is already running

**Neither surrogate run dir records its training seed** -- `metrics.jsonl` carries only
per-epoch losses and there is no `config.json`. So "these two differ only in the noise
flag" is an intent, **not a fact recoverable from the artifacts.**

**Seed B is the guard, and its blind prediction is committed: `0 of 43`.** If that
verifies, the two cell-2 seeds disagree 22 against 0 on the same flag, the seed
dominates, and this reading dies. **The prediction actively points away from the
result I just scored**, which is the strongest form the check could have taken.

### Cell 2's seed caveat is DISCHARGED -- from git, not from the run dirs

The configs are committed and diffable. `go2_mix34_v4dataset` and
`go2_mix34_v4ds_nonoise` both carry `training.seed = 2026061801` and the same
`processed_dataset_dir` (`go2_corrected_34d_excl`); the entire diff is `output_dir`,
`input_noise_sigma` 0.05 -> 0.0, and the metadata block.

> **So 22-against-0 is not two draws that happened to differ. It is one seed, one
> dataset, one architecture, with one flag flipped.** "Differ only in the noise flag"
> is verified, not asserted.

**The run dirs still record no seed** -- `metrics.jsonl` has only per-epoch losses.
That provenance gap is real and worth fixing, but it is not load-bearing here.

#### The remaining n=1 is on the noise-ON side, and seed B does not touch it

    sigma 0.05   seed 2026061801   v4rep     0/43
    sigma 0.0    seed 2026061801   cell2 A  22/43
    sigma 0.0    seed 2026061802   cell2 B  running
    sigma 0.05   seed 2026061802   DOES NOT EXIST   <- the missing cell

**Seed B doubles the noise-off arm and leaves noise-on at n=1**, so a ~20 from B gives
3-against-1, not a 2x2. Partial mitigation exists -- every `sigma=0.05` arm this
project has run (v4rep, armA, armB, base36 at 0/43, 34d_replicate at 2/43) sits at
0-2 across varied datasets and dims -- **but none of those is seed-matched, which is
the whole point of the cell.**

### REGISTERED before the symmetric cell runs

`go2_mix34_v4dataset` at `training.seed = 2026061802`, flag ON, everything else
matched:

    ~0 of 43    -> the flag's effect holds on two seeds in BOTH directions; surrogate
                   seed variance cannot produce 22-against-0 and the 2x2 is closed
    ~20 of 43   -> the noise-ON arm at seed ...801 was the outlier, the flag is NOT
                   the variable, and cell 2's reading is withdrawn

**`rollout_sel` varied 2.5x across the two noise-off seeds**, so surrogate-to-surrogate
spread is exactly the thing that could still produce this at n=1 on one side.

### CELL 2 SEED B: 27 of 43 -- and the blind predictor is REFUTED

    sigma 0.05   seed 2026061801   v4rep     0/43
    sigma 0.0    seed 2026061801   cell2 A  22/43
    sigma 0.0    seed 2026061802   cell2 B  27/43
    v4           original, no noise          20/43

**Both noise-off seeds recover, and both exceed v4.** The `~20` branch registered at
`5c669d3` is confirmed on two independent surrogate seeds, spread across all five
families in each (B: arc 3, constant 8, vel_step 8, weave 5, yaw_step 3).

#### The predictor called seed B at 0 of 43 and it returned 27

    seed A   rho 0.5164  |a| 4.2741   -> NO PREDICTION (gap)   actual 22/43
    seed B   rho 0.7003  |a| 4.0517   -> 0 of 43               actual 27/43

**This is the predictor's first out-of-sample call and it is wrong by 27 episodes**,
in the direction that would have overturned the cell. It was committed before the
verdict at `ab21d78`, so the failure is recorded rather than reconstructed.

**Why it failed is visible in its own training data.** It was fit on five points --
two survivors (rho 0.459/0.492) and three failures (0.652/0.714/1.023) -- and seed B's
0.7003 sits inside the failure cluster on rho while its |a| of 4.0517 sits with the
survivors. **The two features order the classes in opposite directions**, which is the
same thing that produced seed A's abstention. A boundary fit to five points, read as
though it generalised.

> **Retire the rho/|a| predictor.** It has now abstained once and been decisively
> wrong once, on the only two out-of-sample cases it has ever seen. Nothing downstream
> should cite it, and the k* ladder's readings that leaned on it inherit the doubt.

**The n=1 that remains is unchanged and is on the noise-ON side** -- the symmetric
cell (`70627b8`, queued) is what closes it. Seed B does not touch it.

### REGISTERED before the base reference curve reports its failure-mode split

`NEDM_ACTION_MULT` multiplies the action (`imported_policy.py:296`), so **k < 1 is
REDUCED control authority and k > 1 is amplified.** The screen's failure criterion is
an OR over two modes it already records separately in `per_episode`:

    short      episode ends before 1500 rows, commands bounded   -> the robot FELL
    unbounded  max|raw action| > 1e6                             -> it DIVERGED

**These have opposite gain dependence, and the pooled rate hides it.** Under-authority
makes a controller fall; over-authority makes it oscillate and diverge.

> **Registered: base's 22-33% failures at k=0.85-0.90 are predominantly `short`, not
> `unbounded` -- falls from under-actuation, the expected behaviour of any controller
> whose actions are scaled to 85-90%.** If they come back predominantly `unbounded`,
> the screen is detecting something that is not under-actuation and my reading of the
> instrument is wrong.

**This is what decides whether "v4 is base plus half again" is a legal sentence.** If
base's rate at k=1.00 is falls and v4's is divergences, the two rates are different
quantities and their ratio means nothing -- the pooling error one level up from the
denominator errors.

**A second reason the comparison is not yet legal:** base scores 43/43 on the verdict
and some non-zero rate on the screen, but **the two instruments do not share an
episode set.** The verdict scores 43 dataset-derived episodes; the screen runs 8
synthetic standing conditions x 5. A rate from one is not a denominator for the other
until that is reconciled.

#### The anomaly this framing exposes

    v4    condition 1, k 0.90 -> 0.95    5/5 fail -> 0/5     MORE authority, better
    armA  all conditions, k 0.90 -> 0.95  23/40  -> 40/40    MORE authority, WORSE

**armA degrading as gain rises toward nominal is the direction that under-actuation
cannot explain**, and it is the one result in the sweep that needs a mechanism rather
than a rate.

### Base's failures separate exactly on SUSTAINED COMMANDED YAW, not on family

The base curve's two 5/5 conditions are `yaw_step` and `arc`. Read by family that is
two unrelated families; read by the commanded yaw rate it is a clean split:

    cond  family              commanded yaw                     base @ k=1.00
     0-3  constant x4         none                                   0/5
     4    vel_step            none                                   3/5
     6    weave               oscillating wz_amp 0.4, MEAN ZERO      0/5
     5    yaw_step            SUSTAINED wz 0.5 after t=8             5/5
     7    arc                 SUSTAINED wz 0.3                       5/5

**The two conditions base fails are exactly the two with sustained nonzero commanded
yaw, and `weave` -- which commands yaw but with zero mean -- passes 0/5.** Under
random labelling P = 1/28 = 0.036 for that exact pairing.

**It also has an independent mechanism.** `corpus_coverage.py` found six coverage
holes, all body-motion channels, with `yaw_rate` showing a **101x density gap** in
the tail. Sustained yaw is the regime the training corpus is thinnest in.

#### But the eight cells confound four factors, so no label is safe yet

Each condition carries a unique (family, peak force, roll, pitch). **There is one cell
per combination, so family, disturbance magnitude and both tilts are perfectly
confounded across n=8.** Force does worse than yaw as an explanation -- `arc` fails at
36 N while `constant` passes at 72 N, and corr(peak, failures) is only +0.52 -- but
"does worse" on eight confounded cells is not an attribution.

> **REGISTERED: `arc` with `wz` set to 0.0, at the SAME 36 N peak and the same
> +1.0/-3.0 tilts, base checkpoint, 5 seeds.**
>
>     passes ~0/5  -> the sustained yaw command is the driver, not the arc family
>                     and not its disturbance; the condition list is probing a known
>                     corpus hole and the base "failure rate" is partly an artifact
>                     of which conditions were chosen
>     fails ~5/5   -> yaw is a correlate, the arc cell fails for its force or tilt,
>                     and the exact separation above is a coincidence at P=0.036

#### What this does to the proposed statistic

Excluding conditions 5 and 7 because the reference fails them **removes precisely the
sustained-yaw regime** -- the one physically distinct regime in the list and the one
the corpus is thinnest in. That is not removing a constant offset; it is changing the
estimand to "performance away from the known coverage hole."

**Prefer the paired excess over base.** Saturated conditions contribute zero discordant
pairs and drop out of a McNemar automatically, with no data-dependent selection rule
and no change to what is being estimated.

### The registered `arc wz=0` test REFUTES the sustained-yaw reading

    arc  wz=0.3  (as listed)   failed 4/5    max|raw| ~1e35 at rows 89-97
    arc  wz=0.0  (yaw removed) failed 4/5    max|raw| ~1e35 at rows 89-99
    the single pass in each: rows 1775, max|raw| 9.4 and 10.2

**Removing the yaw command changes nothing.** The exact separation at P=0.036 was a
coincidence, exactly as the second branch registered at `4f8f64d` said it would be if
yaw were a correlate. **Base does not fail `arc` because it is turning.**

**And the failure is not under-actuation.** `warmup_s` is discarded before recording
(`collect_go2_smoke.py:502,510`), so row 95 is ~1.07 s of *policy-controlled* time --
these blow up to 1e35 almost immediately, and the episodes that survive run the full
1775 rows with max|raw| near 10. **Bimodal: instant divergence or clean completion,
nothing between.**

### REGISTERED: the conditions separate exactly on NOSE-DOWN PITCH

    cond  roll  pitch   base @ k=1.00
     0     0.0   0.0        0/5
     2    -2.0   1.5        0/5
     3     2.5   2.0        0/5
     6    -2.5   2.5        0/5
     1     1.5  -1.0        0/5
     4    -1.5  -2.5        3/5
     5     3.0  -1.5        5/5
     7     1.0  -3.0        5/5

**Every condition with pitch >= -1.0 passes; every condition with pitch <= -1.5
fails.** Perfect separation, and unlike yaw it is a property of the ground the robot
is standing on rather than of the command.

> **Test: `arc` at the identical 36 N and roll +1.0, with pitch -3.0 (as listed),
> 0.0, and +3.0. Base, 5 seeds each.**
>
>     +3.0 and 0.0 pass, -3.0 fails  -> nose-down ground tilt is the driver; the eight
>                                       conditions are NOT matched in difficulty and
>                                       the pooled "failure rate" is substantially a
>                                       terrain artifact
>     all three fail ~4/5            -> pitch sign is not the driver either, and the
>                                       arc cell fails for its force or for something
>                                       not yet identified

### PITCH TEST RESULT: the ground tilt is the driver, and the conditions are not matched

    arc, identical family / 36 N / roll +1.0 / wz 0.3, only the ground pitch varies:

      pitch -3.0    failed 4/5     rows  97, 91, 97, 1775, 95
      pitch  0.0    failed 0/5     rows  1775 x5
      pitch +3.0    failed 1/5     rows  1775, 135, 1775, 1775, 1775

**One variable changed; the rate goes 4/5 to 0/5.** The first registered branch is
confirmed: **negative ground pitch is what base fails, and the effect is asymmetric --
+3.0 is nearly clean while -3.0 is nearly total.** (Sign convention not verified
against the collector's frame; the asymmetry is the finding, not its physical name.)

#### This is a defect in the condition list, and I wrote it

The eight conditions were built to carry the verdict's disturbances rather than a
still command -- which was the right correction to v2. **But each cell got its own
(family, peak force, roll, pitch) with nothing crossed**, so all four vary together
across n=8 and the pooled rate is dominated by the factor nobody was tracking.

    conditions with pitch <= -1.5 : 4, 5, 7   -> base fails 3/5, 5/5, 5/5
    conditions with pitch >= -1.0 : all rest  -> base fails 0/5

**Base's "33-38% failure at nominal gain" is essentially three nose-down cells.** The
number is a property of the condition list at least as much as of the controller.

#### What survives and what does not

**Survives: every PAIRED comparison.** Arms are run on the identical condition list, so
a fixed-rung comparison between arms holds the tilt constant and the difference is
still a difference. armA's cliff in conditions 2, 3 and 6 -- all of which have pitch
>= -1.0 and which base never fails -- is untouched and if anything sharpened, since
those are cells with no terrain confound at all.

**Does not survive: any absolute rate read as policy quality**, including "base fails
35% at nominal gain", the pooled rate as a `k*` replacement, and the rate-vs-`k` curve
whose crossing `k*` interpolates -- **that curve's height is set by how many nose-down
cells are in the list.**

> **Do not adopt the pooled rate as the `k*` replacement.** Paired excess over base on
> matched conditions is unaffected by this and should be used instead. A condition list
> that crosses tilt against family would fix the instrument, but that is a rebuild.

### REGISTERED: pitch x roll crossed on one cell, to settle what the rebuild must cross

Within the nose-down group neither pitch magnitude nor peak force orders severity:

    cond  pitch  roll  peak    failures (of 20, x3 seeds)
     5    -1.5   3.0   120         20 20 20
     7    -3.0   1.0    36         19 16 17
     4    -2.5  -1.5    96          4 14  8
     1    -1.0   1.5    24          0  4  1

**By pitch magnitude the order is 7 > 4 > 5; by rate it is 5 > 7 > 4. By peak it is
5 > 4 > 7; by rate 5 > 7 > 4.** Neither works. **ROLL is monotone with the rate across
the three deep cells** (+3.0 > +1.0 > -1.5) -- **which is the fourth label tried on
these same eight cells, and by the rule from the last one it is worth a test and not a
conclusion.**

> **Test: `arc`, base, peak 36 N and `wz` 0.3 held fixed, pitch x roll fully crossed
> at pitch in {-3.0, -1.5, 0.0} and roll in {-3.0, 0.0, +3.0}, 5 seeds per cell.**
>
>     roll matters only when pitch < 0    -> the factors INTERACT; a rebuild must cross
>                                            pitch x roll, and neither alone suffices
>     roll matters at every pitch         -> roll is a main effect and the nose-down
>                                            grouping is incomplete
>     roll does not matter                -> the within-group ordering is noise at
>                                            n=20, and pitch sign is the whole story
>     pitch -1.5 ~= pitch -3.0            -> confirms sign not magnitude, independently
>                                            of the eight-cell coincidence

**This is a crossed design on one cell, so it is the rebuild in miniature** -- if it
resolves, the full rebuild's factor list is known rather than guessed.

**One correction to my own arc test while registering this:** pitch +3.0 gave 1/5, not
0/5 (one episode ended at row 135). **Nose-up is not strictly immune**, which their
zero-over-240 does not contradict -- no nose-up `arc` cell exists in the eight.

### CROSSED GRID RESULT: pitch x roll INTERACT, and peak is not needed at all

`arc`, base, **peak held at 36 N and `wz` at 0.3 throughout**, 5 seeds per cell:

    pitch \ roll     -3.0     0.0    +3.0
        -3.0          0/5     5/5     5/5
        -1.5          0/5     5/5     5/5
         0.0          0/5     0/5     0/5

**Three findings, each from one variable moving:**

1. **Roll -3.0 completely protects a nose-down cell.** 0/5 at pitch -3.0, where roll 0.0
   and +3.0 both give 5/5. The registered "roll matters only when pitch < 0" branch is
   confirmed -- at pitch 0.0 roll does nothing.
2. **Pitch magnitude is irrelevant** -- -1.5 and -3.0 are identical at every roll.
   Sign, not magnitude, confirmed independently of the eight-cell coincidence.
3. **Peak force is not required to span the range.** It was fixed at 36 N while the
   rate went 0/5 to 5/5. **The pitch x peak crossing the fleet data argued for would
   have missed the factor that actually does the work.**

**And it is not the turn direction.** At pitch -3.0 / roll -3.0, `wz` = +0.3, 0.0 and
-0.3 all give 0/5, all 1775 rows -- so roll's protection is not cancelling the arc's
turn.

    cond4 (roll -1.5, nose-down) failing 4-14/20 fits the same monotone axis:
    roll -3.0 -> 0/5,  roll -1.5 -> ~20-70%,  roll 0.0 -> 5/5,  roll +3.0 -> 5/5

> **The rebuild must cross pitch x roll.** Family, peak and yaw have each now been
> tested one-at-a-time and none of them moves the rate; the two tilt axes move it
> completely and only in combination.

#### Consequence for a 1-D pitch dose-response

**The pitch boundary is roll-conditional** -- at roll -3.0 there is no boundary
anywhere in [-3.0, 0.0]. **A pitch sweep at a single fixed roll measures where that
roll's boundary is, not where the boundary is**, and the arc cell's +1.0 roll is an
arbitrary choice inherited from the defective list. Fix roll at 0.0 and say so, or
sweep the plane. **Arm-vs-arm comparison at a fixed roll stays valid** -- it is paired,
and the confound is held constant.

### REGISTERED: does protective roll SHIFT the boundary or REMOVE it?

At roll +1.0 the boundary is a hard step in (-1.5, -1.0]. At roll -3.0 there is no
boundary anywhere in [-3.0, 0.0]. **Two structures fit that equally:**

    SHIFTED   failure depends on one effective tilt, some f(pitch, roll); roll -3.0
              just moves the step past -3.0 and it reappears at a steeper pitch
    REMOVED   a genuine interaction; protective roll eliminates the mode entirely

**These have opposite consequences for the rebuild.** If shifted, the whole thing
collapses to ONE axis and the rebuild sweeps a scalar. If removed, two axes must be
crossed and no scalar summarises them.

**Neither the eight cells nor a3's grid can tell them apart -- both stop at pitch
-3.0, which is inside the region roll -3.0 already protects.**

> **Test: `arc`, base, peak 36 N, `wz` +0.3, roll -3.0, at pitch -4.5, -6.0 and -9.0.
> Control: roll 0.0 at pitch -4.5, which must fail if the instrument is behaving.**
>
>     fails at some deeper pitch  -> SHIFTED; one effective tilt axis; rebuild is 1-D
>     0/5 at all three           -> REMOVED; real interaction; rebuild must cross both

**Independent replication worth recording:** their pitch +3.0 read 1/5 at roll +1.0
and my arc test read 1/5 at roll +1.0, on different boxes with different Chrono
builds. **The nose-up edge is real and reproduces across the build difference.**

### RESULT: protective roll REMOVES the boundary, it does not shift it

    arc, base, peak 36 N, wz +0.3, 5 seeds per cell

      roll -3.0  pitch -4.5    0/5     rows 1775 x5
      roll -3.0  pitch -6.0    0/5     rows 1775 x5
      roll -3.0  pitch -9.0    0/5     rows 1775 x5
      roll  0.0  pitch -4.5    5/5     rows 87, 87, 87, 91, 89   <- control, as required

**At roll -3.0 there is no boundary out to -9.0, six times deeper than where the step
sits at roll +1.0.** The REMOVED branch registered above is confirmed: this is a
genuine interaction and **no scalar effective-tilt summarises it. The rebuild crosses
two axes.**

**And the protection is sign-asymmetric** -- roll +3.0 gives 5/5 at pitch -3.0 while
roll -3.0 gives 0/5 at -9.0.

#### The plumbing was verified before reporting this, because the result is odd

`standing_screen.py:143-144` passes `roll` to `--ground-tilt-roll-deg` and `pitch` to
`--ground-tilt-pitch-deg`; not swapped.

#### But "ground tilt" does not tilt the ground -- it rotates GRAVITY

`collect_go2_smoke.py:366-371`:

    _grav_world = [9.81*sin(pitch), -9.81*sin(roll), -9.81*cos(roll)*cos(pitch)]
    system.SetGravitationalAcceleration(...)

**The ground stays horizontal and its contact normal stays vertical.** On a real slope
the friction cone rotates with the surface; here it does not, so the two are not
equivalent at the contacts even though the body-frame acceleration matches.

**This makes the surprise arithmetic rather than physics.** At roll -3.0 / pitch -9.0
the longitudinal component is `9.81*sin(-9deg)` = **-1.53**, twice the **-0.77** of the
control at roll 0.0 / pitch -4.5 that fails 5/5. **The passing cell has double the
destabilising gravity of the failing one**, so this is not a magnitude effect in any
form and the lateral component is doing something specific.

> **Nothing measured on this axis should be described as slope or terrain
> performance.** It is a gravity-direction sweep on flat ground, and the name in the
> flag has been carrying an assumption none of us checked.

#### armB's identity, for the clean-cell table

`run_armB_finetune.sh` writes `--out /home/kyle/sbel-artifacts/finetune_go2_base_matchw`.
**`base_matchw` IS armB** (md5 `7c0020ec3829`, surrogate `go2_mix36_base_bothweights`),
and its own header says it is deliberately NOT base36, whose surrogate is
`go2_mix36_base`. **They are two distinct arms and both belong in the table.**

### THE POLICY CANNOT SEE THE TILT: `_projected_gravity` is hardcoded to world -Z

`imported_policy.py:223-227` computes the gravity observation **from the base
quaternion alone**. It never queries the system, so `SetGravitationalAcceleration`
cannot reach it. The 45-vector has no other world-frame term -- no base linear
velocity, no height.

Combined with the yaw-only spawn (`collect_go2_smoke.py:392`), the robot starts
upright, `q = identity`, and:

    roll  0.0 pitch  0.0   policy SEES [0,0,-1]  TRUE [0,0,-1]              error 0.00 deg
    roll  0.0 pitch -4.5   policy SEES [0,0,-1]  TRUE [-0.078,0,-0.997]     error 4.50 deg
    roll -3.0 pitch -9.0   policy SEES [0,0,-1]  TRUE [-0.156,0.052,-0.986] error 9.49 deg

**The observation error equals the full tilt.**

#### This corrects my earlier objection, which was wrong for the reason I gave

The rotation argument is sound: plane + gravity + upright robot is rotation-equivalent
to a slope, and `tan(theta)` is the same in either frame. **The friction-cone objection
is withdrawn -- the DYNAMICS map.**

**The OBSERVATION does not map.** Rotate the world so gravity is vertical and the robot
stands normal to a slope with quaternion `R`; a correct `projected_gravity` would
return the tilted vector and the policy would see the slope. Ours returns `[0,0,-1]`.
**The rotation carries the dynamics and not the sensing, because the sensing is
computed under an assumption the rotation invalidates.**

#### And it is worse than blindness -- it is a wrong attitude reference

The quaternion does change as the robot tips, so the policy sees *some* gravity signal.
But it is the direction of world -Z, not of the actual field. **The policy holds
attitude against a vertical that is not vertical**, fighting to stand normal to a
horizontal ground plane while the field pulls it sideways. On a real slope with correct
sensing it would lean into the slope; here it cannot, because it does not know there is
one.

> **This axis measures rejection of an UNOBSERVED, MIS-SENSED gravity disturbance. It
> is not slope walking.** Arm comparisons on it stay valid -- every arm is equally
> mis-informed -- but no result from it should be described as terrain or slope
> performance.

**A one-line fix exists** (normalise the system's gravity into the body frame instead
of assuming world -Z) **and it would change every number on this axis, so it must not
be applied underneath a running comparison.** It is a different experiment.

### The corpus's pitch cap rests on `corr(|roll|, fell)`, which is blind to the roll effect

`drive_go2_collection.py:84-94` records why pitch is capped at +-1.5 while roll runs
+-3.0, measured on 2,000 episodes:

    corr(|pitch|, fell) = +0.427   rising 2% -> 48% across the band
    corr(|roll|,  fell) = +0.057   and flat
    "Capping the combined magnitude instead would sacrifice roll range for nothing,
     since roll does not drive failures."

**The measured quantity is `|roll|`. The effect is antisymmetric in roll.**

    pitch -3.0, roll -3.0  ->  0/5        same |roll| = 3.0
    pitch -3.0, roll +3.0  ->  5/5        opposite outcomes

**Taking the absolute value cancels the two exactly**, so `corr(|roll|, fell)` is near
zero *whatever* the sign effect's size. The statistic used to rule roll out cannot
represent the way roll acts. **`corr(|pitch|, fell)` worked because the pitch effect
IS roughly symmetric in magnitude; the same transform applied to roll destroyed it.**

> **The corpus's tilt envelope was shaped by a measurement that was blind by
> construction to the factor it was used to dismiss** -- and the resulting asymmetry
> (roll fully sampled, pitch capped at half the tested depth) is exactly where the two
> policies turn out to differ.

#### Which also means the clean-cell statistic is mostly out-of-distribution

Training pitch envelope is +-1.5. The four "clean" conditions:

    cond 0   pitch  0.0    INSIDE
    cond 2   pitch +1.5    at the CAP
    cond 3   pitch +2.0    BEYOND
    cond 6   pitch +2.5    BEYOND

**Only one of the four sits strictly inside the pitch range the surrogate was trained
over.** Roll is in-distribution throughout (all four within +-3.0). So the headline
numbers -- base 0%, v4 44%, armA 57%, armB 75%, base36 79% -- are **largely a measure
of out-of-distribution pitch generalisation**, which is a legitimate and interesting
thing to measure but is not what the table's column heading says.

**This does not weaken the ordering** -- every arm is equally out-of-distribution, and
base at a hard 0/240 while fine-tuned arms fail 44-79% OOD is a sharper statement than
the in-distribution version would be. **It changes the caption, not the result.**

### RETRACTED: the OOD framing. The pitch cap postdates the entire training corpus.

    corpus s2000000   1741 episodes   2026-09-04 18:59 .. 23:07
    corpus s3000000   1762 episodes   2026-09-04 22:44 .. 23:10
    go2_corrected_34d_excl built                        2026-09-05 01:33
    pitch capped to +-1.5   commit e09e45b              2026-09-05 17:07   <- AFTER
    parameters first recorded  commit 574a6d2           2026-09-07 02:39

`e09e45b` changed `uniform(-3.0, 3.0)` to `uniform(-1.5, 1.5)` **sixteen hours after
the dataset the surrogate trains on was already built.** The surrogate's
`metadata.json` names `go2_comprehensive_merged/flat` as its raw root, and every
episode in it predates the cap.

> **The training pitch envelope is +-3.0, not +-1.5.** My claim at `d525b5f` that three
> of the four clean cells sit beyond the trained range is **wrong** -- pitch 0.0, +1.5,
> +2.0 and +2.5 are all inside +-3.0, and the grid's -3.0 is at the edge rather than at
> double the depth.

**Also retracted: "the arms are indistinguishable inside the envelope and diverge at
its boundary."** There was no boundary in the training data. **The arms diverge at
pitch -1.5 to -3.0, which is squarely IN distribution.**

#### The corrected reading is stronger, not weaker

The surrogate saw tilt drawn over the full +-3.0 pitch range **and has no channel that
represents it** -- `grav_body_*` is derived from the stored quaternion, carrying the
same hardcoded world -Z as the policy. **So the arms differ on a latent that was
sampled across exactly the range where they differ, and that the model could not
represent.** No distribution shift is needed to explain it: the data was there and the
state definition threw it away.

#### And this exact failure is documented in the codebase, in the commit that fixed it

`collect_go2_smoke.py:746-752`, added by `574a6d2`:

    "The verdict harness used to reconstruct these from a seeded RNG duplicated in
     its own source. That contract broke silently when the driver's pitch range was
     capped to +-1.5 and the harness kept deriving +-3.0."

**Previously: harness derived +-3.0, driver had capped to +-1.5. Now: a reconstruction
drew +-1.5 from a corpus collected at +-3.0.** The same mismatch, mirrored, eighteen
hours after a comment was written describing it. **A reconstruction is only as good as
the version of the source it duplicates, and nothing in the corpus pins that version.**

### The 500 excluded episodes: verified, and the mechanism is narrower than "no failures"

    dataset_index.json                 3003     <- what the surrogate trains on
    dataset_index.json.pre_physical    3503
    dropped                             500     mean 131.5 rows vs 3760.7 kept

`PROVENANCE_NOTE.md:44-48` gives the criterion -- physical admissibility, bounds `2x
URDF per joint, |dq| <= 60.2, |v| <= 15.0, |w| <= 50.0` -- and records that **499 of
the 500 had status `diverged`, 1 `fell`, and 344 were inadmissible in 100% of their
frames.**

**But the training set is NOT failure-free**, which the "the model never saw failure"
reading would require:

    IN the training index   n=3003   diverged   78 ( 2.6%)   fell  363 (12.1%)
    EXCLUDED                n= 500   diverged  499 (99.8%)   fell  500 (100.0%)

> **The surrogate saw 363 falls and 78 divergences. What it never saw is the
> CATASTROPHIC tail -- the episodes where the integrator left the physical envelope.**
> The exclusion is not "failures removed"; it is "the worst 14% of failures removed,
> selected with near-perfect correlation to the outcome."

#### Why that is still the sharpest mechanism available

**The mode the fine-tuned policies exhibit is exactly the mode that was filtered.**
Every screen failure runs to `max|raw action| ~ 1e35` at row ~90 -- non-physical
magnitudes, the same regime the admissibility bounds excise. **The surrogate was
trained with 99.8% of that mode removed, and a policy optimised inside it has no
gradient away from a failure the model cannot represent.**

**Two refinements to the version I was sent:**

1. **It is whole-episode, not per-frame.** 156 of the 500 were admissible in *some*
   frames, so their pre-divergence prefixes -- the run-up, which is the informative
   part -- were discarded along with the blown-up remainder.
2. **The steep-pitch claim rests on reconstructed tilts**, which are the half-scale
   values from the range error above. The exclusion's outcome-correlation (499/500
   `diverged`) is recorded in the provenance note and needs no reconstruction; the
   pitch association does.

**And the exclusion is defensible on its own terms** -- training a dynamics model on
`|v| = 1e35` states would poison it. **The defect is that a bound chosen for numerical
sanity selects on the outcome, and nothing downstream records that the training
distribution is conditioned on not having blown up.**

### Base's 43/43 on the verdict is guaranteed by the selection rule, not measured

The baseline corpus is collected with `CKPT = go2_cts_150k.pt` -- **the base controller**
(`drive_go2_collection.py:33`). The verdict's cell is then built by applying
`scored()` to each baseline CSV, which returns `None` when
`len(rows) < SCORED_ROWS + 500` (`run_go2_finetune_verdict.py:157`).

> **So the 43 eligible episodes are exactly the episodes on which the BASE controller
> ran to completion.** An episode where base failed cannot enter the cell. Base then
> "scores" 43/43 because it is being re-run on episodes selected for base having
> survived them -- **up to determinism, which the replay check confirms. The number
> cannot come out any other way.**

**This is not a defect in the verdict.** It asks whether a fine-tuned arm tracks better
than base where base works, and conditioning on that is the right cell for that
question. **The defect is using base's 43/43 as evidence of anything.**

#### What it costs, concretely

**The verdict is structurally incapable of detecting an arm that is BETTER than base
where base fails.** The two-arm grid contains exactly such a cell:

    pitch -1.5, roll 0.0 and +1.5     base 5/5 fail     v4 0/5 fail

**v4 outperforms base there, and no episode of that kind can ever be in the verdict's
43.** The verdict's silence about it is a property of the selection, not a finding.

#### And it weakens one leg of the screen's validation

The argument for the clean-cell statistic was that it "agrees with the verdict wherever
the verdict discriminates: base 43/43 -> 0%, v4 20/43 -> 44%." **The base end of that
agreement is tautological on the verdict's side.** The v4 and armA points still carry
it, and **the screen's own base 0/240 on the clean cells is a real measurement** --
those episodes were not selected for base surviving them. **The validation survives on
the arms; it should not be quoted with base as its anchor.**

### "Rejected by `scored()`" is not "base failed" -- the set is 59% that and 41% other

`scored()` has five rejection paths: short episode, non-finite values, non-constant
command over the scored window, insufficient lead-in (clause 3b), and joint
admissibility. **Only the first and last mean base failed.**

    episodes on disk                            3503
    rows < 1500  (base did not finish)           572
    verdict's "failed predicate or admissibility" 967
    rejected for reasons OTHER than short         395

**So re-running arms on everything `scored()` rejects would mix 572 base-failures with
395 episodes base completed fine** -- and an arm "completing an episode base could not"
would be unfalsifiable, since 41% of the set was never a base failure.

> **Select on `rows < 1500`, not on `scored() is None`.**

#### The verdict cannot classify the episodes it is blind to

Cell membership is `-cell_hi < cmd <= -cell_lo`, where `cmd` comes from **the constant
tail of the scored window** -- a statistic over rows a failed episode never reached.
**`scored()` returns `None` before `cmd` exists, so a base-failed episode has no
defined cell.** The complement experiment needs the command taken from the episode's
early rows or its spec, and that is a different rule, not the same one applied further.

#### And three command families never enter the verdict at all

Short episodes span eight families:

    weave 90   yaw_step 82   arc 81   pivot 72   vel_step 71
    lateral 65   constant 63   stop_and_go 48

**The verdict's eligible set contains only five** -- `pivot`, `lateral` and
`stop_and_go` are absent, excluded by the cell rather than by failure (their `cmd_vx`
is zero or out of range). **Those three are invisible to the verdict for a reason that
has nothing to do with the arms**, so they must be reported separately from the
base-failure region or the two blind spots get pooled.

### The 1500-row threshold does sit in a gap -- but the short set is not homogeneous

Over all 3503 corpus episodes:

    short (<1500)  572     min 50   median 121   max 1492
    next value at or above 1500:                      1553
    => EMPTY between 1492 and 1553, a 61-row gap

**So the threshold is not arbitrary**, unlike the earlier case where I reused "site the
threshold in the gap" on a set spanning 248 to 3958 with no gap in it. Here there is
one, and it is thin but real.

**But the set has a tail that the median hides:**

    [   0,  200)   530     93%   dead within ~5% of the episode
    [ 200,  500)    26
    [ 500,  900)    10
    [ 900, 1200)     3
    [1200, 1500)     3     failed around 35-40% of the way through

**Full corpus episodes average 3760 rows**, so a 1492-row episode is a robot that ran
for two fifths of the run before losing it, and a 121-row one is gone in about a
second. **These are not the same event and an arm completing one of each would not be
the same result.**

> **Any "arms complete N of the base failures" figure should be stratified by base's
> own length**, or a single marginal recovery and a genuine rescue report identically.

### The complement fires: v4 completes 14 of 38 episodes base cannot

    corpus go2_cell_a3, 38 baseline episodes with rows < 1500

      base    0 of 38    0.0%    deterministic control
      v4     14 of 38   36.8%    6 of 14 re-verified with the harness's own scored()
      armA    0 of 38    0.0%

**The registered ">0" branch fires**, and armA's zero shows it is not a generic
property of fine-tuning. **The verdict can never contain one of these episodes.**

#### But the reframe needs its other half, and the number is net negative

    region                            n    base    v4
    base SUCCEEDED (verdict cell)    43      43    20
    base FAILED (complement)         38       0    14
    -------------------------------------------------
    union                            81      43    34

**v4 gains 14 episodes base cannot complete and loses 23 that it can. Net -9.** So
"v4 is different, not worse" is right as a description of the *structure* and wrong if
it is heard as "the verdict understates v4." **On the two regions together v4
completes fewer episodes than base**, and the verdict's `20/43` is directionally
consistent with that even though it is measured on only one region.

#### And the union is not a population, so the -9 is an illustration and not a statistic

- **The 43 are cell-filtered** (`-0.18 < cmd <= -0.02`); **the 38 cannot be**, because
  cell membership is read from a scored window a failed episode never reaches.
- **They are from different corpora** -- `go2_comprehensive_merged/flat` against
  `go2_cell_a3`, which is not on this box.
- The two regions are **not sampled in their natural proportion**; 43 and 38 are
  artifacts of what each selection happened to yield.

> **Nothing here licenses a pooled rate.** The defensible statement is the pair of
> conditional rates, reported together: **on base's successes v4 completes 47% against
> base's 100%; on base's failures v4 completes 37% against base's 0%.** Quoting either
> alone mischaracterises it, and quoting their sum invents a population.

### Fine-tuning SHIFTED the tilt tolerance rather than shrinking it

Measured on two machines and two seeds, tilt set explicitly per cell:

    nose-down side   base fails from about -1.5    v4 holds to about -3.0    v4 BETTER
    nose-up   side   base clean out to +3.0        v4 fails from +2.0        v4 WORSE

> **v4 buys roughly 1.5 degrees of downhill tolerance and pays about 1 degree of
> uphill.** This is the first mechanistic account of what fine-tuning did to these
> policies rather than a description of how often they fail.

**armA saturates at 15/15 and 20/20 across both grids** -- no measurable band at
nominal gain, so it is *uninformative* on this axis rather than worse on it. Same
distinction as a saturated cell contributing no discordant pairs.

#### The band prediction, and a provenance question it rests on

The claim that v4's 14 rescues sit between base's boundary and v4's needs the failed
episodes' pitch values. **Those were reported from sidecars, spanning -2.97 to +2.99 --
which two commit dates make hard to hold together:**

    e09e45b   2026-09-05 17:07   pitch capped, uniform(-3,3) -> uniform(-1.5,1.5)
    574a6d2   2026-09-07 02:39   tilt first RECORDED in the episode sidecar

**A corpus with a tilt sidecar was collected after the cap, so its pitch should lie
inside +-1.5.** Either the corpus was built by a purpose-made script setting tilt
directly -- in which case it is a constructed distribution, not a found one -- or the
values are the half-scale reconstruction, in which case the median is -0.93 and the
band argument collapses. **`go2_cell_a3` is not on this box and I cannot resolve it
here.**

**The grids in the first table do not depend on this** -- their tilts were set per cell
by the experimenter. **Only the band-clustering claim does.**

#### And the separation is not yet a separation

    rescued      n=14   -2.21 .. -0.79   median -1.52
    not rescued  n=24   -2.97 .. +2.99   median -2.15

**Ranges overlap almost entirely; medians differ by 0.63; the non-rescued range is
inflated by a single nose-up episode.** At these n it needs a rank test. **The sign was
registered before the split was seen, which is what makes it evidence at all; the
magnitude is not quotable.**

### The 2x2 at the SURROGATE level, before the symmetric cell's verdict

    cell                          epochs   val_loss   rollout_sel   verdict
    A  sigma 0.0   seed ...801       80    0.002806      8.8339      22/43
    B  sigma 0.0   seed ...802       80    0.002803      3.5918      27/43
    C  sigma 0.05  seed ...801       80    0.007559      0.5796       0/43
    D  sigma 0.05  seed ...802       80    0.007922      2.0275     running

    noise effect  2.69x at seed ...801,  2.83x at seed ...802
    seed  effect  1.001x at sigma 0.0,   1.048x at sigma 0.05

**On `val_loss` the design does exactly what it was built to test: the flag moves the
surrogate 2.7-2.8x on both seeds, and the seed moves it under 5%.**

#### But the two surrogate metrics disagree about cell D, and I am recording that now

`rollout_sel` is not stable across seeds -- 8.83 against 3.59 at sigma 0.0 (2.5x) and
0.58 against 2.03 at sigma 0.05 (3.5x). **It is the multi-step rollout metric, and the
fine-tune optimises through branch rollouts**, so it is not obviously the less relevant
of the two.

    by val_loss     D looks like C  (0.0079 vs 0.0076)          -> expect ~0/43
    by rollout_sel  D looks unlike C (2.03 vs 0.58, 3.5x better) -> expect better than 0

**And `rollout_sel` does not order the three known verdicts** -- 8.83 gave 22, 3.59
gave 27, 0.58 gave 0 -- so it is not a clean predictor either. **The point is only that
the surrogate-level evidence is not unanimous, and I would rather say so before the
number arrives than explain afterwards which metric I had been watching.**

**The registered branches at `d645afd` are unchanged and this does not amend them.**
They are about the surviving-pair count, not about either surrogate metric.

### The symmetric cell's blind prediction is from the RETIRED predictor. It counts for nothing.

The chain script still calls `predict_replicate.py`, so a prediction of **0 of 43** was
committed before this verdict. **The predictor was retired at `16754ee`** after its
only two out-of-sample calls: an abstention on cell 2 seed A, and a confident `0 of 43`
on seed B that returned **27**.

> **Recording this BEFORE the verdict, because the registered branch for cell D is also
> `~0 of 43`.** If D comes back near zero, the retired predictor will have been "right",
> and that must not be read as rehabilitating it. **One correct call from a model that
> was decisively wrong on its previous one, on a case where the designed hypothesis
> predicts the same answer, is not evidence about the model.**

The prediction is committed for provenance only. **Nothing downstream should cite it in
either direction**, and the verdict is scored against `d645afd` alone.

### SYMMETRIC CELL: 11 of 43. The 2x2 closes, and it hits NEITHER registered branch.

                    seed ...801   seed ...802
      sigma 0.0          22            27
      sigma 0.05          0            11

    noise effect at seed ...801   22 -> 0    -22 pairs
    noise effect at seed ...802   27 -> 11   -16 pairs
    seed  effect at sigma 0.0     22 vs 27    +5 pairs
    seed  effect at sigma 0.05     0 vs 11   +11 pairs

**Registered at `d645afd` were `~0` and `~20`. The answer is 11, which is neither**,
and unlike cell 2 I did not register an intermediate branch here. **That is a gap in my
registration, not a result to be read into one of the two arms of it.**

#### What the 2x2 establishes anyway

**`input_noise_sigma` is causal.** It costs 22 surviving pairs at one seed and 16 at
the other -- **same sign, large at both, and larger than the seed effect at either
sigma.** The direction replicates and that was the question the cell existed to answer.

**But "noise-ON gives ~0" was an overreading of n=1.** Cell C's zero is the extreme of
the four, not the typical value: at the other seed the same flag gives 11. **The
seed moves the outcome by 11 pairs at sigma 0.05 against 5 at sigma 0.0**, so surrogate
seed variance is larger than the `val_loss` 2x2 suggested (1.048x across seeds) and
larger than I assumed when I called C the counterfactual.

> **Corrected claim: removing `input_noise_sigma` recovers roughly 16-22 surviving
> pairs, on two seeds. Not "22 against 0" -- that pair was the widest of the four
> available and quoting it overstates the effect by about a third.**

#### The pre-registered metric disagreement resolved toward `rollout_sel`

At `1f9fee3`, before the verdict: `val_loss` said D looks like C (expect ~0);
`rollout_sel` said D is 3.5x better than C (expect better than 0). **D returned 11.**

    by rollout_sel   C 0.58 -> 0,  D 2.03 -> 11,  B 3.59 -> 27,  A 8.83 -> 22
    by val_loss      B -> 27, A -> 22, C -> 0, D -> 11

**`rollout_sel` orders three of four and inverts A/B; `val_loss` orders three of four
and inverts C/D.** Both are 3-of-4 at n=4. **The direction call on D was correct and
was made in advance -- but it is one binary call, which chance gets right half the
time. It is not evidence that `rollout_sel` is the better metric.**

#### The retired predictor said 0 and the answer was 11

`34171f8` registered that its call counted for nothing either way. **It was wrong
again, on inputs (rho 1.2597, |a| 6.0899) far outside its fitting range.** Two wrong
calls and one abstention, out of three out-of-sample cases. **It stays retired.**

### My projection of the unfiltered pass was wrong, and the reason is a second selection

    projection at 47% (the verdict cell's rate)   v4 121/269   net -110
    projection at 40%                             v4 106/269   net -125
    projection at 55%                             v4 141/269   net  -90
    MEASURED                                      v4 167/269   net  -64

**All three were wrong and the answer is outside the range I quoted.** The sign was
determined, as I said; **the magnitudes were not mine to give.**

**The error is entirely in one imported rate:**

    v4 on base-successes, verdict cell (cell-filtered)     20/43    46.5%
    v4 on base-successes, unfiltered (no cell)            153/231   66.2%

**I transplanted 47% from a cell-filtered population onto an unfiltered one** -- the
exact move I spent the night objecting to in others, and the lesson
*"a fact travels; the conditions that made it true do not"* is already in the file
under my name.

#### But the 19.7-point gap is itself a finding

**The verdict's cell selects a region where v4 is unusually bad.** Base's completion
rate is comparable across the two corpora (85.9% in `go2_cell_a3`, 83.7% in the merged
set), so this is not a difficulty difference between corpora -- **it is the cell.**

> **So the verdict carries TWO selection effects in OPPOSITE directions.** It conditions
> on base having succeeded, which hides v4's 37% completion rate on base's failures and
> flatters base. And it conditions on `-0.18 < cmd <= -0.02`, a band where v4 completes
> 47% rather than 66%, which flatters base again.
>
> **`20/43` is not simply "one region of v4's behaviour". It is the region of that
> region where v4 does worst.**

**What this does not change:** the net is still solidly negative at -64 over 269
episodes, and v4 still completes fewer episodes than base on any population measured.
**The direction survives all of it. The magnitude has now been wrong twice, both times
from a rate carried across a selection boundary.**

### REGISTERED before running v4 in a neighbouring command band on the merged root

**Their cell test has a weak arm, and the reason is corpus composition.** Sampling 417
base-completed episodes from `go2_comprehensive_merged/flat`:

    (0.02,0.18]  THE CURRENT CELL      9    2.2%
    (0.18,0.40]                       26    6.2%
    (0.40,0.80]                       22    5.3%
    forward or zero commanded vx     305   73.1%
    non-constant tail                 55   13.2%

**The verdict's cell is 2.2% of this corpus.** `go2_cell_a3` is 78% inside it (209 of
269) -- it is a cell-TARGETED corpus, as its name says. **So their "outside the cell"
arm is 22 episodes that a cell-targeted generator happened to miss, not a sample of the
outside-cell population.** Their 1.3-point result is measured where they have almost no
outside-cell data.

**The merged root has the opposite composition and can answer it.** Running the verdict
harness unchanged with `--cell-lo 0.18 --cell-hi 0.40` gives v4's surviving-pair rate
in a neighbouring band -- same corpus, same machine, same code path, same predicate,
only the band moved.

    ~47%, like 20/43   -> the cell is NOT the explanation; the 46.5-vs-66.2 gap is the
                          corpus or v4 generally, and my `733734e` claim is withdrawn
    ~66%               -> the cell IS the explanation on the root where 20/43 was
                          measured, and their a3 null was an artifact of having 22
                          outside-cell episodes
    52-60%             -> partial; the cell carries some of the gap and not most of it,
                          and neither the two-selection claim nor its withdrawal is clean

**Registering the middle branch explicitly, because the last cell I ran did not have
one and the answer landed in it.**

### The 15.8-point corpus gap at fixed cell is the PITCH CAP

    v4 surviving pairs, cell (0.02,0.18] held identical on both sides:
      merged root      20 / 43     46.5%
      go2_cell_a3     129 / 207    62.3%

**The cell is not a variable in that contrast -- it is the harness default on both
sides -- so the 15.8 points is a corpus difference.** And the corpora sit on opposite
sides of `e09e45b`:

    merged root      collected 2026-09-04                pitch ~ U(-3.0, +3.0)
    go2_cell_a3      HAS tilt sidecars => after 574a6d2 (09-07 02:39)
                     => after the cap e09e45b (09-05 17:07)  => U(-1.5, +1.5)

**The tilt grid says v4 fails 5/5 at pitch +2.0 and +3.0, at roll 0.0 and roll -3.0,
where base is clean.** Conditioning on base-eligibility as the verdict does:

    base-eligible pitch span, merged root   (-1.5, +3.0)   width 4.5
    of which v4 fails, pitch > +2.0                        width 1.0
    predicted deficit from nose-up alone                   22.2 points
    OBSERVED                                               15.8 points

**An upper bound overshooting by six points is what a roll-modulated boundary predicts.**

> **Neither 46.5% nor 62.3% is an outlier. One corpus samples v4's nose-up failure
> region and the other cannot reach it.** Two of tonight's findings meeting: the
> shifted tilt tolerance, and a collection parameter that changed between the two
> corpora.

**Which makes `20/43` partly a measurement of a collection parameter that changed after
it was taken.** The verdict's headline number would have been materially higher had
that cell been collected two days later.

**Registered:** `go2_cell_sliger` is also post-cap, so **v4 should land near 62% there,
not 46%.** That is this account's prediction, NOT evidence against it -- the fleet's
own registration read it the other way. **The structural check is
`max(|ground_tilt_pitch_deg|)` over both post-cap corpora's sidecars: ~1.5 confirms it
without any inference from commit dates.**

### armA is at the floor on an unselected population

    go2_cell_a3, all 269 episodes, no cell, no predicate:
      base   231/269   85.9%      v4  167/269  62.1%      armA  0/269  0.0%

**269 CSVs produced, median 411 rows against a 1500 threshold** -- a plumbing failure
gives zero files, not 269 short ones. **The floor is real and not an artifact of the
eligibility rule.**

**And armA's failures run three times LONGER than base's** -- median 411 against 116.
**armA stays up longer and then always goes down; base goes down fast and rarely.**
Different modes, invisible in a completion rate.

> **So the selection story separates cleanly: the eligibility rule hides 14 real
> completions for v4 and hides nothing for armA.** It does not merely flatter base --
> it conceals capability exactly where capability exists.

### WITHDRAWN: the pitch-cap account, and the gap it explained was 1.9 sigma

**Both `go2_cell_a3` and `go2_cell_sliger` reach pitch -2.99..+3.00.** Neither is
capped, so the pre-cap/post-cap inference at `2940da7` is wrong, and the `-2.97..+2.99`
sidecar figures I had flagged as "the number that does not fit" were correct.

**And the gap did not need a mechanism:**

    root      20/43   = 46.5%   binomial sd 7.6   95% CI 31.6% .. 61.4%
    a3       129/207  = 62.3%   binomial sd 3.4
    difference 15.8 pts, se 8.3, z = 1.90, two-sided p = 0.057

**I fitted a mechanism to a marginal difference and it appeared to fit** -- 22.2
predicted against 15.8 observed. **That agreement is what fitting to noise produces:
the gap is small enough that any plausible mechanism lands near it.** Both `733734e`
(the cell attribution) and `2940da7` (the cap account) are withdrawn.

### The finding that survives: the crossover, at population scale

    pitch band       n     base      v4      delta
    [-3.0,-2.0)     29    41.4%     6.9%    -34.5
    [-2.0,-1.0)     30    46.7%    76.7%    +30.0    <- v4 BETTER
    [-1.0, 0.0)     57    93.0%    78.9%    -14.0
    [ 0.0,+1.0)     46   100.0%   100.0%      0.0
    [+1.0,+2.0)     58   100.0%    62.1%    -37.9
    [+2.0,+3.1)     49    98.0%    30.6%    -67.3    <- v4 collapses, base clean

**Marginals verified: the six bands sum to n=269, base 231, v4 167, matching the
reported totals exactly.** The synthetic `arc` grid's crossover reproduced on eight
families in a collected corpus -- same signs, same ordering, same crossover point, a
completely different instrument.

#### And the aggregate is a weighted average across a sign change

Same per-band rates, reweighted:

    as measured (this corpus's mix)   base 85.9%  v4 62.1%  net -23.8 pts
    pitch capped to +-1.0             base 96.5%  v4 89.5%  net  -7.1 pts
    pitch capped to +-2.0             base 84.9%  v4 79.4%  net  -5.5 pts
    nose-down half only               base 60.4%  v4 54.2%  net  -6.2 pts
    nose-up half only                 base 99.3%  v4 64.2%  net -35.1 pts

> **`net -64` and `62.1%` are properties of this corpus's tilt distribution, not of
> v4.** The aggregate ranges over a factor of five across defensible mixes.

**This is the condition-list defect one level up.** The unfiltered pass removed
SELECTION -- no cell, no predicate, no conditioning on either arm's outcome -- **but it
did not remove COMPOSITION, and the tilt mix is as arbitrary as the eight conditions
were.** "Unfiltered population" felt like it had solved the problem.

**Report the band table, not the net.** The net needs its mix stated, exactly as the
`k*` rate needed its condition list stated.

#### Unresolved provenance

`drive_go2_collection.py:94` draws `uniform(-1.5, 1.5)`, yet both corpora reach +-3.0.
**Neither was produced by the current driver as it stands**, so they are constructed
evaluation corpora rather than draws from the collection distribution. **They do match
the TRAINING corpus's +-3.0**, which is the comparison that matters, but how they were
generated is not recorded anywhere I can see.

### CORRECTION: my own reweighting table weighted bands equally, not by episode count

The table at `de5e1ea` is wrong in every row except the first. **The "as measured" row
passed the band counts and is right; the restricted rows passed 0/1 indicators**, which
averages over BANDS rather than over EPISODES -- a different estimand, and an unstated
composition of exactly the kind the entry was about.

    restriction        WRONG (equal bands)      CORRECT (episode counts)
    as collected +-3.0        -23.8                     -23.8
    +-2.0                      -5.5                     -11.0
    +-1.0 / +-1.5              -7.1                      -9.9   (theirs, exact: -6.1)

**The corrected +-2.0 figure matches the fleet's independently computed -11.0
exactly.** My +-1.5 of -9.9 linearly interpolates the boundary bands; **theirs at -6.1
uses per-episode pitch and is the one to use.**

> **I made a composition error while writing up a composition error.** Substituting a
> uniform-over-bands mix for the episode mix is the same move as picking a tilt range
> without noticing -- silent, defensible-looking, and it changed the number by half.

**The qualitative conclusion survives**: the net ranges from -6 to -24 across
defensible mixes, a factor of four rather than five.

### The tilt mix was chosen, not collected

`fresh_a3.py:28-29` and `fresh_sliger.py:28-29` draw **pitch from `uniform(-3.0, 3.0)`**
while `drive_go2_collection.py:94` draws `uniform(-1.5, 1.5)` -- the roll range copied
across to pitch, past a comment explaining why the two differ.

    as collected  (+-3.0, chosen)   n=269   base 85.9%   v4 62.1%   net -23.8
    driver's current +-1.5          n=147   base 94.6%   v4 88.4%   net  -6.1

**Under the distribution the project actually collects under, v4's deficit is 6 points.
Under the one in the scratch generator, 24.** Every aggregate quoted from the
unfiltered pass -- `net -64`, `62.1%`, "v4 loses more than it gains" -- is a property
of that choice.

**Third instance of the same defect at a third level:** the screen's hand-picked
conditions, the verdict's selection rule, and now a composition in a pass that had
removed all selection. **Removing selection does not remove composition.**

### BAND TEST RESULT: 34/55 = 61.8%, and the three measurements point at the 43

Registered at `f51770b`: `~47%` -> cell is not the explanation; `~66%` -> it is;
`52-60%` -> partial. **The answer is 61.8%, closest to the `~66%` branch and
statistically consistent with it (95% CI 49.0-74.7%).**

    measurement                        k/n      rate     sd    95% CI
    root, cell (0.02,0.18]           20/43    46.5%    7.6%   31.6 .. 61.4
    root, band (0.18,0.40]   NEW     34/55    61.8%    6.6%   49.0 .. 74.7
    go2_cell_a3, cell (0.02,0.18]   129/207   62.3%    3.4%   55.7 .. 68.9

#### But the pairwise tests say something neither branch anticipated

    band vs cell, same corpus        +15.3 pts   z=1.52   p=0.127
    a3 vs root,   same cell          +15.8 pts   z=1.90   p=0.057
    a3 vs band,   NEITHER shared      +0.5 pts   z=0.07   p=0.946

> **The two measurements that differ in EVERYTHING -- different corpus, different
> command band, different machine -- agree to half a point. The only measurement that
> disagrees with anything is the 43, and it is the shared term in both "significant"
> comparisons.**

**So the parsimonious reading is not a cell effect and not a corpus effect. It is that
`20/43` is a low draw**, and the two 15-point gaps are one gap counted twice, because
both subtract the same 46.5%.

**This does not license "the true value is 62%"** -- 61.8 and 62.3 agreeing so closely
is itself a coincidence at these n, and I have spent the night warning about exactly
that kind of agreement. **What it does say is that no mechanism is needed: the 43's own
interval reaches 61.4%.**

**The registered `~66%` branch nominally fires, but the reading it was attached to --
"the cell IS the explanation" -- does not survive the third comparison.** Recording it
that way rather than claiming the branch.

**Both prior accounts stay withdrawn:** `733734e` (cell attribution) and `2940da7`
(pitch cap). **The band test was launched to resolve a difference that three
measurements now suggest was never there.**

### Which tilt distribution is "the project's"? Not the driver's.

Every `flat`-derived training dataset -- `go2_corrected_34d`, `go2_corrected_34d_excl`
(the surrogate's), `go2_corrected_36d_pose`, `go2_contact_40d` and its variants --
resolves to `go2_comprehensive_merged/flat`, **collected 2026-09-04, sixteen hours
before the cap.** The verdict's baseline corpus is that same root.

    training corpus, all surrogates      pitch +-3.0     (pre-cap)
    verdict baseline corpus              pitch +-3.0     (same root)
    the fleet's evaluation corpora       pitch +-3.0     (chosen by copying the roll bound)
    drive_go2_collection.py today        pitch +-1.5     (no in-use dataset came from it)

> **The accidental +-3.0 happens to match the distribution every surrogate was trained
> on and every verdict scored against. The driver's +-1.5 matches nothing currently in
> use.** So "under the distribution the project actually collects under, the deficit is
> 1.4-6.1 points" is the wrong reference: **nothing in the project has been collected
> under it yet.**

**Neither range is correct a priori** -- the point is that the correction swapped one
unstated choice for another, and the one it swapped to is less connected to the
existing artifacts, not more.

#### And the cap removes exactly the discriminating region

The band tables put the arms' largest differences at `[+1.0,+2.0)` (-38 and -40) and
`[+2.0,+3.1)` (-67 and -63), with v4's advantage at `[-2.0,-1.0)` (+30 and +21).
**Capping pitch to +-1.5 excises most of the first two entirely and half the third.**

**The cap's own rationale was that combined tilt reached 4.24 deg, "inside the 3-5 deg
band where the gait was separately measured to collapse."** That is a reasonable
objective for a corpus meant to teach a dynamics model. **It is the wrong objective for
a corpus meant to distinguish policies, and the same corpus is used for both.**

### Band table with denominators, both corpora, marginals verified

    go2_cell_a3                          go2_cell_sliger
    band          n   base  v4  delta    band          n   base  v4  delta     p(a3)  p(sl)
    [-3.1,-2.0)  29    12    2  -34.5    [-3.1,-2.0)  32    10    2  -25.0     0.001  0.007
    [-2.0,-1.0)  30    14   23  +30.0    [-2.0,-1.0)  39    22   30  +20.5     0.012  0.049
    [-1.0,+0.0)  57    53   45  -14.0    [-1.0,+0.0)  53    49   46   -5.7     0.028  0.337
    [+0.0,+1.0)  46    46   46   +0.0    [+0.0,+1.0)  45    45   45   +0.0       --     --
    [+1.0,+2.0)  58    58   36  -37.9    [+1.0,+2.0)  43    43   26  -39.5     0.000  0.000
    [+2.0,+3.1)  49    48   15  -67.3    [+2.0,+3.1)  65    64   23  -63.1     0.000  0.000
    TOTAL       269   231  167           TOTAL       277   233  172

**Marginals verified independently against both reported totals: exact match.**

**v4's ADVANTAGE band replicates and is significant on both** -- `[-2.0,-1.0)` at
p=0.012 and p=0.049. That is the half of the shift most likely to be dismissed as
noise. **`[-1.0,0.0)` does NOT replicate (0.028 against 0.337) and should not be
carried.**

#### Caveat on those p-values, which are mine

**Both arms run the same episodes, so the comparisons are PAIRED.** The tests above are
unpaired two-proportion z, which ignores that; **McNemar needs discordance counts, not
marginals.** I raised exactly this objection about the gain rungs and then used the
wrong test myself for want of the per-episode data. With positive within-episode
correlation McNemar is usually more powerful, **so the true significance is probably
stronger -- but "probably stronger" is not a result.**

#### And my scaling check made the night's own error

I estimated sliger's v4 total at ~178 by scaling a3's band sizes; the true value is
172. **`[+2.0,+3.1)` is 65 on sliger against 49 on a3, and `[+1.0,+2.0)` is 43 against
58** -- same generator, same range, different draw. **A transplant, inside a check run
to catch transplants.**

### Reference class: the fleet's +-1.5 correction is withdrawn too

**"An improvement in form and a regression in reference class."** No aggregate without
its mix stands; what changes is the default a reader should assume when none is stated,
and that is **+-3.0**, because it is what every surrogate trained on and every verdict
scored against.

> **A corpus tuned to keep the robot inside its stable envelope is a corpus that cannot
> see which policy has the wider envelope. Two objectives, one collection parameter,
> and the parameter was set for the first.**

### PAIRED analysis, and the tilt shift is settled

Discordant counts per band, `b` = base completes and v4 does not, `c` = the reverse:

    band          a3: b, c   McNemar    sliger: b, c   McNemar
    [-3.1,-2.0)     11,  1    0.0063       10,  2       0.0386
    [-2.0,-1.0)      2, 11    0.0225        3, 11       0.0574
    [-1.0,+0.0)     10,  2    0.0386        7,  4       0.5488
    [+0.0,+1.0)      0,  0      --          0,  0         --
    [+1.0,+2.0)     22,  0    0.0000       17,  0       0.0000
    [+2.0,+3.1)     33,  0    0.0000       41,  0       0.0000

**My earlier unpaired p-values are struck.** They treated positively correlated
observations as independent and were anti-conservative: 0.012 and 0.049 against the
correct 0.023 and 0.057. **Concordant pairs carry no information about the difference,
which is why McNemar conditions on the discordant ones -- that is the test being right,
not the pairing "costing power."**

#### Pooling the two corpora IS valid, and settles it

Different episodes, independent draws, same generator -- **unlike the gain rungs, where
the same episodes recurred and pooling would have double-counted.**

    v4's ADVANTAGE   [-2.0,-1.0)    pooled b=5   c=22   p = 0.0015
    v4's DEFICIT     pitch > +1.0   pooled b=113 c=0    p = 1.9e-34
    the ZERO         [0.0,+1.0)     91 episodes, not one failure by either arm

**Neither corpus alone gets the advantage band past 0.02; pooled it is unambiguous.**

**Attainable floors, so neither end is overread:**

     13 discordant -> smallest two-sided p = 2.4e-04      27 -> 1.5e-08
     14 discordant -> 1.2e-04                            113 -> 1.9e-34

**sliger's 0.057 is 3 against 11 on 14 flips -- modest per corpus, decisive pooled.**

**DROPPED: `[-1.0,0.0)`** -- b=10,c=2 against b=7,c=4 is a direction difference in the
discordance, not a magnitude difference. It does not replicate.

> **Final form of the tilt result: v4 trades nose-up tolerance for nose-down. The gain
> is p=0.0015 on 27 discordant pairs; the loss is p=1.9e-34 on 113, every one in base's
> favour. Both measured on paired episodes across two independently drawn corpora, and
> reproducing a synthetic sweep run on a different instrument.**
