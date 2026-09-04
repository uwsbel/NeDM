# The Go2 policy environment is a subclass, not a second implementation

**Status:** planned, not implemented. Contingent on level-1/2 validation landing.
**Date:** 2026-09-04

## The finding

`hmmwv_tracking_env.py` is 871 lines and reads like a study-specific file. It is
not. Almost all of it is generic machinery -- terrain mixing, one-hot conditioning,
reference-domain resolution, pose integration, observation assembly -- and the
parts that are genuinely HMMWV-specific are **one config function and one reward
term.**

So the Go2 policy environment is roughly a 150-line subclass. This was worth
establishing before writing anything, because the obvious plan was to mirror 871
lines into a parallel file, and a hand-mirrored 871 lines diverges.

## Why it works out that way

**The two studies share a state vocabulary, deliberately.** `DEFAULT_STATE_FIELDS`
is the same seven body channels for both -- `vel_body_x_mps`, `vel_body_y_mps`,
`roll_rad`, `pitch_rad`, `roll_rate_radps`, `ang_vel_body_y_radps`,
`yaw_rate_radps`. The Go2 collector took the names from `hmmwv_data.BASE_FIELDS`
verbatim wherever the quantity exists for a legged base. `_integrate_pose` looks
its channels up by name through `self.state_index`, so **it runs on Go2 unchanged.**

**Both actions are 3-D.** The Go2 action is the velocity command
`[cmd_vx, cmd_vy, cmd_wz]`, and the HMMWV's is
`[steering, throttle, braking]`. `_scale_policy_actions` is
`clamp(center + scale * tanh(a), low, high)` -- entirely generic, so the Go2 case
is `action_low` / `action_high` / `action_scale` in config, with no code change.

**`FrozenDynamics` takes every dimension from checkpoint metadata**
(`state_dim=len(metadata["state_fields"])`). Nothing in the env hardcodes 15. So an
env written now survives a change of state preset, which matters because the
`quadruped_contact` (15-D) versus `quadruped_full` (23-D) choice is an open
ablation.

**The level-3 pattern is already subclassing.** `HMMWVChronoCRMTrackingEnv` is 104
lines extending `HMMWVChronoTrackingEnv`, overriding terrain creation, stepping,
and state capture. The Go2 Chrono-side env follows it.

## What must actually be overridden

1. **`default_env_cfg()`** -- action bounds for velocity commands rather than
   pedals (`cmd_vy` and `cmd_wz` are signed; `throttle` and `braking` are not),
   reward weights, and termination thresholds. A quadruped's roll/pitch limits are
   a real fall test, not a rollover proxy, so they are tighter and they mean
   something different.

2. **`_compute_reward`** -- exactly one term does not carry over:
   `throttle_brake = driver_actions[:, 1] * driver_actions[:, 2]`, which penalizes
   pressing both pedals. For the Go2 that index pair is `cmd_vy * cmd_wz`,
   penalizing turning while strafing, which is not a thing we want to discourage.
   Setting `throttle_brake_weight: 0.0` zeroes its contribution but the term is
   still computed and still logged into `extras`, so a reader would see a
   meaningless "throttle_brake" number on a legged robot. Override the method.

Everything else is inherited, and inherits future fixes to the shared machinery.

## The reference set is not a blocker

`build_combined_flat_crm_rl_references.py` needs an existing flat reference file as
a template, and no Go2 one exists -- but that script is the HMMWV *combiner*, not
the builder. `rl/references.py:153 build_reference_set(processed_root, ...)` reads
the processed cache directly and needs no template. The Go2 path uses it once
preprocess has run.

Note that `select_reference_episode_indices` round-robins over
`split_metadata["scenario_families"]`, making it the **third** consumer of that
field (with `trainer.py:796` and `build_combined_*:135`). All three are fixed by
the data repair; see [experiment-design.md](../lessons/experiment-design.md).

## What is genuinely open, and waits for validation

- **Which state preset.** 15-D `quadruped_contact` is the HMMWV counterpart and
  the anchor; 23-D `quadruped_full` adds sinkage and surface displacement. The
  paper's rule is deletion: a channel earns its place by degrading rollout when
  removed. `surface_disp` reads 0.17-0.23 mm under a foot that floats clear of the
  bed, so it is the leading deletion candidate.
- **Whether the 3-D command action can track a trajectory at all**, given the
  three plant nonlinearities we measured: a forward dead zone below ~0.35 m/s,
  terrain-dependent tracking (0.030 rigid versus 0.145 soil at 0.30 command,
  converging by 0.50), and a sign asymmetry where backward tracks about 4x better
  than forward in the same band. A tracking policy has to learn to stay out of the
  dead zone, and that is a result rather than an obstacle -- it is the soil
  degrading control authority, which is what the case study is about.
- **`max_episode_steps`.** The HMMWV's 180 steps at `action_repeat` 5 and dt 0.01
  is 9 s, which sits inside our 8-11 s episodes, but the reference segment length
  has to be checked against what the Go2 episodes actually support after boundary
  terminations are excluded.

## Order

Level 1 and 2 first (one-step error, open-loop rollout). The env is only worth
writing against a reduced state that has been shown to propagate, and if
validation moves the state preset, the config moves with it -- but by design, not
the code.
