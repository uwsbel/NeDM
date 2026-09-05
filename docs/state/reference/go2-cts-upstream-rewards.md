# CTS upstream reward spec, as actually configured

Source: `wty-yy/go2_rl_gym`, **branch `master`** (there is no `main`; a fetch of
`main` returns 404 and it is easy to mistake that for the file being absent).
Files: `legged_gym/envs/base/legged_robot.py`, `legged_gym/envs/go2/go2_config.py`,
`legged_gym/envs/go2/go2_env.py`. Read 2026-09-05. Implements CTS, arXiv 2405.10830.

This is the reference for implementing the reward against our surrogate. **The
headline weight table is not the trained objective** — see the curriculum below.

## The weights change during training

`go2_config.py` sets `curriculum_rewards`, applied by `update_reward_curriculum()`
as a multiplier on the configured scale:

| term | multiplier | over |
|---|---|---|
| `lin_vel_z` | 1.0 -> **0.0** | iters 0–1500 |
| `correct_base_height` | 1.0 -> **10.0** | iters 0–5000 |

So the **trained** function, which is what `go2_cts_150k` optimised, ranks:

```
  correct_base_height  -10.0   <- largest by 5x
  dof_pos_limits        -2.0
  lin_vel_z              0.0   <- INERT, not -2.0
  tracking_lin_vel      +1.0     tracking_ang_vel +0.5
  ang_vel_xy   -0.05  dof_acc -2.5e-7  dof_power -2e-5  torques -1e-4
  action_rate  -0.01  action_smoothness -0.01  collision -1.0
  feet_regulation -0.05  hip_to_default -0.05
```

Quoting the config's `scales` block alone gives the iteration-0 snapshot.

## Formulas, verbatim in substance

```
tracking_lin_vel    exp(-[ (cmd_x-v_x)^2 + (cmd_y-v_y)^2 ] / sigma)      sigma=0.25
tracking_ang_vel    exp(-(cmd_yaw - w_z)^2 / sigma)
lin_vel_z           v_z^2
ang_vel_xy          sum(w_xy^2)
dof_acc             sum( ((qd_prev - qd)/dt)^2 )        dt = POLICY dt = 0.02 s
dof_power           sum(|tau * qd|)
torques             sum(tau^2)
correct_base_height (base_height - 0.38)^2              base_height = z - ground_z
action_rate         sum((a_{t-1} - a_t)^2)              a = NORMALISED action
action_smoothness   sum((a_t - 2a_{t-1} + a_{t-2})^2)
collision           count of penalised bodies with |contact force| > 0.1
dof_pos_limits      sum( -(q-lo).clip(max=0) + (q-hi).clip(min=0) )   LINEAR, not squared
feet_regulation     sum_feet( |v_foot_xy|^2 * exp(-h_foot / (0.025*0.38)) )
hip_to_default      sum_4hips |q_hip - q_hip_default|
upright             (-1 - g_z)/2        only under turn_over_scales, |roll| > pi/4
```

`only_positive_rewards = False`, so the total is not clipped at zero.

## The soft limit is applied to the LIMITS, not in the reward

`_process_dof_props` shrinks the range about its midpoint before the reward ever
sees it, with `soft_dof_pos_limit = 0.9`:

```
  m = (lo+hi)/2 ;  r = hi-lo
  lo' = m - 0.45*r ;  hi' = m + 0.45*r
```

**Implementing `dof_pos_limits` against raw URDF limits makes it nearly inert** — the
measured violation rate against the hard limits on our plant is 0.059%, so the
third-largest weight would contribute nothing and no log would show it.

## The sign trap, which cuts in both directions

`SIGN = -1.0` in `imported_policy.py` applies to all 12 joints. Recorded
`joint_*_pos_rad` are **Chrono convention**; the policy sees the negation. Verified
against data, not the constant: corr(recorded, upstream defaults) = **-0.976**,
corr(recorded, negated defaults) = **+0.976**.

| term | reference frame | negate recorded q? |
|---|---|---|
| `hip_to_default` | imported-convention constant (+/-0.1) | **YES** |
| `dof_pos_limits` | URDF limit | **YES** |

**Corrected 2026-09-05.** An earlier version of this table said `dof_pos_limits`
must NOT be negated, on the reasoning that Chrono parses the URDF so recorded angles
share its convention. That is wrong, and measurement settles it: the URDF calf range
is entirely negative, `[-2.72, -0.84]`, while recorded calf positions are entirely
POSITIVE, about `[+0.52, +3.00]`. Checking observed ranges against both conventions
across 120 episodes:

```
  thighs  4/4 fit the NEGATED URDF range, 0/4 fit the raw range
  calves  0/4 fit raw; sign matches negated
  hips    symmetric range, so uninformative either way
```

**So the rule is simpler than the earlier table implied: there is ONE convention
shift, not two.** The policy/imported convention IS the URDF convention (upstream's
defaults sit inside the URDF ranges: calf -1.5 in [-2.72,-0.84], thigh 0.8 in
[-1.57,3.49]), and recorded Chrono angles are its negation because `SIGN = -1`.

**Negate recorded joint positions once, into policy/URDF convention, then apply every
term normally.** Equivalently, compare against `[-upper, -lower]`. The failure mode
is not "one term needs it and one does not" — it is applying it to neither, or to one
of the two.

(The residual calf excursions beyond even the negated range, +3.00 against +2.72, are
real limit violations; the measured rate against hard limits is 0.059%.)

`hip_to_default` sums over four hips, which is order-invariant **only if each hip is
paired with its own default**. Upstream defaults: FL +0.1, RL +0.1, FR -0.1, RR -0.1
(and thighs FL/FR 0.8, RL/RR 1.0; calves all -1.5). A leg mismatch is a systematic
0.2 rad offset, not a wash. Upstream indexes hips `[0,3,6,9]` in **FL/FR/RL/RR**
order — a fourth ordering, distinct from our `MOTOR_NAMES` RR/RL/FR/FL.

## Computability from our recorded channels

| term | computable | note |
|---|---|---|
| tracking_lin_vel / ang_vel | yes | cmd_* and vel_body_* |
| lin_vel_z, ang_vel_xy | yes | but lin_vel_z is inert in the trained function |
| dof_acc | yes | **use the 0.02 s policy dt, not the 0.01 s record step** (4x) |
| dof_power, torques | yes | joint torques are recorded |
| correct_base_height | yes | pos_z_m minus terrain top (0.05) |
| action_rate, action_smoothness | yes | **policy_raw_\***, not joint_\*_target_rad (16x) |
| dof_pos_limits | yes | limits from the URDF, soft-shrunk as above |
| feet_regulation | **yes** | foot_\*_vel_x/y_mps and foot_\*_pos_z_m. We have foot height directly; upstream derives it from projected gravity only because it lacks ground height |
| collision | **NO** | upstream penalises thigh/calf/base contacts; we record feet only |
| hip_to_default | yes | with the sign flip above |

**Computable from the data is not computable from the surrogate.** Terms needing
torque, joint velocity or foot state can only be optimised if the surrogate predicts
those channels; confirm before implementing.

## Two terms that exist but do not fire

- `_reward_x_command_hip_regular` (go2_env.py) is defined but absent from `scales`.
- `turn_over_scales` (`upright = 1.0`) activates at |roll| > pi/4 = 0.785 rad. Our
  termination bounds roll at 0.40, so it never fires for us.

## Action space: these terms DO transfer

Upstream's terms are defined on a **joint-space** policy at 50 Hz emitting 12 joint
actions. **The fine-tune matches that**: it trains the imported policy's own 460,972
weights and emits 12 joint actions at 50 Hz, so `action_rate`, `action_smoothness`,
`dof_acc` and the rest are defined on the same quantity upstream defines them on and
transfer directly.

**Corrected 2026-09-05.** An earlier version of this file claimed a structural
mismatch — that our fine-tune acts in command space (cmd_vx, cmd_vy, cmd_wz) into a
frozen low-level policy, making `action_rate` a command rate and not the upstream
quantity. That describes `Go2NeuralTrackingEnv`, the command-level wrapper in this
repo, which is **not** what the fine-tune uses; that architecture was rejected and a
joint-space environment built instead. The claim was true of the older design and
false of the current one. **Check which environment a policy actually trains in
before reasoning about what its action space means** — the repo contains both.

The scale is worth keeping in view: at `action_scale = 0.25`, the observed exploit
(mean action 1.033 -> 5.244, max 5.850 -> 20.180) is a mean joint offset of 1.31 rad
and a maximum of 5.05 rad from the default pose. Go2 joints move within roughly
+/-1.05 (hip), -1.5..3.5 (thigh), -2.7..-0.8 (calf), so **this particular exploit is
gross enough that `dof_pos_limits` would fire against raw URDF limits too**. The
soft-limit correctness above matters for catching subtler drift, not for this case —
worth stating so the fix is not credited with more than it does.

## What the term-check suite establishes, and what it does not

`scripts/evaluation/check_go2_reward_terms.py` reports a discrimination matrix:
each candidate implementation against each test, with deliberately-wrong variants
included so the suite is shown to reject them rather than merely to exist.

**It validates implementations against the reference transcribed in it. It does not
validate that reference against reality.** Were the single-negation convention above
itself wrong, the reference would encode the error and every cell would still read
green. The convention rests on a separate and independent check — the URDF calf range
`[-2.72, -0.84]` against recorded calf near `[+0.52, +3.00]`, and 4/4 thighs fitting
the negated range against 0/4 fitting the raw one across 120 episodes.

Two validations, of two different things, and only one of them is in the suite.

**Known limit:** the two `hip_to_default` wrong-variants score identically (1.5000),
because for this default pattern reversing the leg pairing is arithmetically the same
as negating. A red cell there is a **gate, not a diagnosis** — it says the convention
is wrong, not which mistake was made. An asymmetric displacement pattern would
separate them if a diagnosis is ever needed.

**Not covered, deliberately:** the `dof_acc` timestep (0.02 s policy dt, not the
0.01 s record step) and the `action_rate` input (`policy_raw_*`, not
`joint_*_target_rad`). Those are 4x and 16x SCALE errors, and a synthetic pose cannot
see them — right and wrong both return a plausible positive number. They must be
checked against real arrays, as a ratio between the two candidate computations rather
than against a threshold.
