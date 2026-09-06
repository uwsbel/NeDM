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
