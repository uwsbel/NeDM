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

**What is NOT yet marked on this curve is the surrogate's own joint error.**
Until that is measured at 0.1 s and beyond, the curve says where the cliff is but
not which side of it we are on. That measurement is the remaining gate on W1 and
it is not an argument, it is a rollout.

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
