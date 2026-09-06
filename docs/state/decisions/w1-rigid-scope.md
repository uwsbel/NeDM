# W1 on rigid: scope (not implementation)

CRM is tabled ([crm-tabled.md](crm-tabled.md)). On rigid the terrain query is a
closed-form plane query, so contact geometry can move OUT of the model and into
a deterministic `G()`. This scopes that. Nothing here is trained yet.

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
