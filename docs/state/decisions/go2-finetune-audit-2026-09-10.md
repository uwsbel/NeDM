# Four-way audit of the CRM fine-tune, 2026-09-10

Four agents audited the pipeline in parallel against the corpus the base policy itself
generated and against the upstream source. This records what they found, what they
CLEARED, and which earlier conclusions in this directory are now wrong.

The single most useful outcome is negative: the surrogate is fine, the reward is fine,
and the observation vector is fine. Three of the four defects that mattered were in how
the fine-tune *assembles a branch*, and all three were invisible to every guard we had.

## The upstream source is on disk

`sbel-artifacts/reference/go2_rl_gym/` (fetched 2026-09-05 from `wty-yy/go2_rl_gym`).
All 11 implemented reward terms match it -- forms, converged curriculum weights, and
buffer timing. `TRACKING_SIGMA = 0.25` is confirmed at `go2_config.py:166`, not assumed.
Four of the terms (`action_smoothness`, `dof_power`, `correct_base_height`,
`hip_to_default`) are upstream ADDITIONS to stock legged_gym, and they match too.

Nothing is missing. `feet_air_time` is defined upstream but absent from `scales`, so it
is not part of this reward; the `reward_airtime` in `chrono_crmenv.py` is SBEL's own
addition to a different 7-term reward and is not evidence about the CTS policy. The
"missing gait-shaping term" hypothesis is dead.

## Defects, in the order they cost us

### 1. The branch fed the surrogate a state that never occurs in training

`newa` is built to REPLACE the last action; it was APPENDED, so the action sequence ran
one step ahead of the states and a duplicated state token was inserted to compensate.
One-step, driving RECORDED actions so the policy is not involved:

| channel | aligned | fine-tune | ratio |
|---|---|---|---|
| `vel_body_x_mps` | 0.119 | 0.684 | **5.75x** |
| `yaw_rate_radps` | 0.110 | 0.153 | 1.39x |
| joint pos (12) | 0.034 | 0.039 | 1.15x |
| joint vel (12) | 0.263 | 0.280 | 1.06x |

The damage lands on velocity and spares the joint channels. Branch gradient rotated
~59 deg, opposite half-space on 1 branch in 4, against a null control of +0.92 for
dropping a whole context token -- and worst on upright, normally-walking branches.

This is why the reward and the gradient disagreed. Penalties read joint channels, which
survived. Tracking reads velocity, which did not. Fixed in `0df6995`.

### 2. The previous-action channel was invented

`rollout()` started every branch at `prev = zeros` and self-fed the policy its own output
through five warm-up steps, while the true previous raw action sat in the `policy_raw_*`
columns. First branch action vs the logged action for that control step: RMS **1.6455**,
larger than the raw-action standard deviation of 1.539. Seeded from the record: 0.2672.

Two agents measured this independently (corr +0.52 vs +0.986). It also explains the
"~0.117 fidelity limit, origin unexplained" in the module docstring: that IS the seeded
number. Fixed in `491e7fe`.

### 3. Control-row parity was a hardcode; it is a per-episode property

`# control acts on ODD rows` is true of `go2_merged` (200/200) and false for half of the
newer corpora. Measured live on a 30-episode CRM pool: **16 even, 14 odd**. Parity is now
read off the data -- the raw action is held across the decimation, so the rows where it
changes are the control rows. Fixed in `491e7fe`.

Defects 2 and 3 were INVISIBLE to `check_action_roundtrip`: they move its correlation by
0.002 against a self-fed divergence that swamps them. That script's pooled 12-joint
correlation is also inflated by the between-joint `IMPORTED_DEFAULTS` offsets -- same
trajectory reads +0.996 pooled, +0.949 per-joint-centred, +0.52 on the raw action. Its
`--min-corr 0.5` gate would pass a substantially wrong conversion.

### 4. `--target-dw` performed no checkpoint selection at all

It wrote `val_neg_reward = NaN` into `best.pt` and set `best = (NaN, u)`, overwriting
whatever validation had selected, and emitted a bare `NaN` into history.json, which is
not valid JSON. Every displacement-controlled arm shipped whatever the last Adam step
produced. Fixed in `491e7fe`: the stop point is still the deliverable, it is now measured.

## The pathology that SURVIVED the fixes

Re-measured after defect 1, at the base policy on CRM (`--probe-grad-share`):

| | dof_acc | tracking | tracking share of VALUE |
|---|---|---|---|
| branch 5 (0.10 s) | **47.4%** of grad norm | 28.2% | 89.2% |
| branch 15 (0.30 s) | **63.2%** | 18.1% | 89.6% |

`dof_acc` carries weight -2.5e-7 and 6.7% of the reward. It dominates the gradient
because it is a squared finite difference divided by dt, so its path through the
surrogate is stiffer by (1/0.02)^2 -- and backprop weights a term by PATH STIFFNESS, not
by contribution to return. It gets worse with horizon, which is a direct argument against
"longer branch = better credit assignment".

PPO is structurally immune: its score-function estimator multiplies grad-log-pi by a
SCALAR reward, so no term can be over-weighted by its Jacobian. This is a real difference
between our objective and the one that produced the policy, and it is not a tuning
problem -- reweighting to fix the gradient breaks the value composition and vice versa.

## Cleared, with evidence -- do not re-audit these

- **The surrogate's Jacobian is physically sound.** `d(joint_pos)/d(target)` = +0.035 to
  +0.061 rad/rad per 10 ms, positive on all 12 joints. Held for 0.1 s the joint moves
  0.39-0.58 of the offset, identically at offsets of 0.02 / 0.05 / 0.10 / 0.20 rad, and
  0 of 192 branches move the wrong way. Off-diagonal structure is physical: thigh-calf
  coupling negative, cross-leg hip-hip uniformly positive (trunk reaction). Anchored
  against the recorded torque satisfying `tau = 20(target-q) - 0.5 qd` to 0.148 Nm.
  Caveat: per-step calf VELOCITY gradient is sign-unstable (wrong sign on 28-56% of
  states); it averages out over the branch but is close to noise per step.
- **Normalisation and delta construction are clean.** Applied once, identically in
  training and inference, across all 18 go2 checkpoints. `targets = states[1:] -
  states[:-1]`, correctly aligned, on raw state. No off-by-one.
- **The 45-channel observation is correct.** Order, scales, defaults, SIGN and C2I all
  match `go2_mujoco.yaml`, the imported policy's own deploy config, at all 9 sites.
  `BatchedGo2Policy` reproduces the TorchScript to max|diff| = 0.0.
- **No observation normalisation exists on either side.** The CTS `normalizer` is
  literally the identity; `empirical_normalization: null` upstream.
- **Standing still does not pay under this reward.** Freezing costs -0.28 on average;
  penalties would have to be 2.3x larger to make stillness profitable. It wins only in
  the zero-command bin, 8.9% of rows, which is what upstream intends.
- **`hip_to_default` is not a lever.** Correct constant, and insensitive to the
  alternatives on real data (0.4% of total).

## Corrections to earlier documents

- `correct_base_height` is the largest WEIGHT and the 8th of 11 largest TERM (0.67%).
  Its target is upstream's FIXED 0.38 above ground = **0.58** here, not the base policy's
  own mean. Passing the mean, as every arm before today did, turns a restoring force into
  a pure variance penalty.
- `go2-finetune-postmortem.md` named the autoregressive horizon as the blocker. Horizon
  was never tested at a correct forward pass, and the gradient measurement above argues
  the opposite direction. Treat that conclusion as unsupported.
- There is **no critic** in `go2_cts_150k.pt`. `model_2999.pt` has one, but for a
  different policy in Genesis joint order with zero hip defaults and a different reward;
  it is not transplantable. The "bootstrap the branch with the original value function"
  plan is closed.
- The reference `rslrl` code did NOT produce this policy, and no training config ships
  with the checkpoint. Its gamma / lambda / episode length are not evidence about the CTS
  policy's.

## Known and not yet fixed

- Branch pool has no upright filter by default: 9.9% of flat starts begin fully inverted
  (`grav_body_z > 0`). `--min-upright` added, off by default.
- `preprocess.py` unwraps roll/pitch and records `circular_unwrapped`; the fine-tune reads
  raw CSVs without it. 4.14% of flat rows, 0% of CRM, measured impact ~0. Hygiene.
- `check_action_roundtrip.py` should gate on the raw-action correlation, not the pooled
  one. Its current gate is too weak to catch what it exists to catch.
