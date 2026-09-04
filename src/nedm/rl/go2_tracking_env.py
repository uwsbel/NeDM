"""Go2 path-tracking environment inside the frozen NRD model.

A SUBCLASS, NOT A PARALLEL IMPLEMENTATION. hmmwv_tracking_env.py reads
study-specific and almost none of it is: the Go2 collector took its state field
names from hmmwv_data.BASE_FIELDS verbatim, so _integrate_pose resolves
yaw_rate_radps, vel_body_x_mps and vel_body_y_mps by NAME through state_index and
runs here unchanged. Both actions are 3-D. FrozenDynamics takes every dimension
from checkpoint metadata. Two overrides are enough, and everything else --
including future fixes to the shared machinery -- is inherited.

WHAT CHANGES, AND WHY EACH ONE HAS TO:

1. THE ACTION SPACE IS SIGNED ON ALL THREE CHANNELS. The HMMWV's are steering in
   [-1, 1] and throttle and braking in [0, 1] -- two pedals that cannot be
   negative. Ours are cmd_vx, cmd_vy, cmd_wz, and all three take both signs.
   Inheriting action_low [-1, 0, 0] would have made lateral and yaw commands
   one-directional, which is the same defect the collection had before the
   stratified rewrite and would have been just as invisible.

   Bounds are the policy's TRAINED ranges, +/-0.5 and +/-1.0, not its wider
   deployment clips.

2. throttle_brake IS REMOVED, NOT ZERO-WEIGHTED. The term is
   driver_actions[:,1] * driver_actions[:,2], which for the HMMWV penalises
   pressing both pedals at once. For us those channels are cmd_vy and cmd_wz, so
   it penalises TURNING WHILE STRAFING -- a perfectly reasonable quadruped
   manoeuvre. Setting its weight to zero would leave the term computed and
   logged, and a reader would find a throttle_brake number on a legged robot and
   reasonably wonder what it meant.

3. TERMINATION ROLL AND PITCH ARE A REAL FALL TEST. On the HMMWV they proxy a
   rollover; on a quadruped they are the actual failure. Tightened accordingly.

THREE MEASURED PLANT NONLINEARITIES THE POLICY MUST LEARN AROUND, all from the
collection and all reproduced by the NRD model it trains inside:
  a forward dead zone below ~0.35 m/s (commanded 0.30 achieves 0.030 on rigid)
  terrain dependence -- the same 0.30 command achieves 0.145 on CRM, ~5x
  backward tracking ~4x better than forward in the same magnitude band
A policy that declines to command slow forward motion is the plant being
learned, not a bug in the policy.
"""

from __future__ import annotations

from typing import Any

import torch

from nedm.rl.hmmwv_tracking_env import HMMWVNeuralTrackingEnv, default_env_cfg


def go2_default_env_cfg() -> dict[str, Any]:
    cfg = default_env_cfg()
    cfg.update({
        # cmd_vx, cmd_vy, cmd_wz -- the policy's TRAINED ranges. All signed.
        "action_low": [-0.5, -0.5, -1.0],
        "action_high": [0.5, 0.5, 1.0],
        "action_scale": [0.5, 0.5, 1.0],
        # Our episodes are 14.75 s at 100 Hz with action_repeat 5, so 180 steps
        # would run past the end of every reference segment.
        "max_episode_steps": 120,
    })
    cfg["reward"] = {
        **cfg["reward"],
        # CHOSEN FROM MEASURED ERROR, not from the trajectory length. The Go2
        # covers ~1.0-1.3 m in a 10 s rollout against the HMMWV's 30-53 m, so the
        # inherited 2.0 m is wider than our whole trajectory: exp(-(e/2)^2) is
        # 0.99 at 0.2 m and 0.94 at 0.5 m, flat with no gradient. But over-
        # correcting is the same failure mirrored -- at sigma 0.25 the reward is
        # 0.037 by 0.45 m, which is where an untrained policy actually sits.
        # A random policy on this env produces (measured, 256 envs x 120 steps):
        #     p10 0.079   p50 0.295   p90 0.454 m
        # giving reward across that band of:
        #     sigma 0.25 -> 0.905 / 0.249 / 0.037   dead at p90
        #     sigma 0.40 -> 0.962 / 0.581 / 0.275   graded throughout
        #     sigma 0.50 -> 0.975 / 0.707 / 0.438   compressed at the top
        # 0.40 keeps a usable gradient over the whole range early training visits.
        "position_sigma_m": 0.40,
        "yaw_sigma_rad": 0.35,
    }
    # KEPT AT 0.0 RATHER THAN REMOVED. The parent indexes this key directly
    # (hmmwv_tracking_env.py:741), so deleting it raises on the first step. The
    # term is neutralised here, dropped from the reward terms in _compute_reward,
    # and stripped from the log in _make_extras -- so nothing downstream ever
    # sees a throttle_brake number, which is the point. Zeroing the weight ALONE
    # would not have achieved that.
    cfg["reward"]["throttle_brake_weight"] = 0.0
    # THESE ARE A MODEL-VALIDITY BOUNDARY, NOT A FALL TEST. The collection
    # recorded ZERO falls in 1,120 episodes, so the NRD model has never seen one
    # and cannot predict one -- past the edge of its data it extrapolates, and a
    # policy optimising inside it would be free to exploit that. Terminating
    # where the training data ends is what stops it.
    # Measured coverage of the training split:
    #     |roll|  max 0.395 rad (flat), 0.262 (crm)
    #     |pitch| max 0.334 rad (crm),  0.173 (flat)
    # Thresholds sit just above those maxima: legitimate states are not
    # terminated, extrapolated ones are.
    cfg["termination"] = {
        # 20 m is a third of an HMMWV trajectory and several times ours.
        "max_position_error_m": 2.0,
        "max_abs_roll_rad": 0.40,
        "max_abs_pitch_rad": 0.35,
    }
    return cfg


class Go2NeuralTrackingEnv(HMMWVNeuralTrackingEnv):
    """Velocity-command tracking for the Go2 inside the frozen NRD model."""

    def _compute_reward(self, driver_actions: torch.Tensor):
        reward, terms = super()._compute_reward(driver_actions)
        # cmd_vy * cmd_wz is turning-while-strafing, which we do not penalise.
        # The weight is 0.0 so the reward is already unaffected; dropping the term
        # stops it reaching any consumer that would report it.
        terms.pop("throttle_brake", None)
        return reward, terms

    def _make_extras(self, reward_terms, dones, time_outs):  # type: ignore[override]
        # The parent logs "/tracking/throttle_brake" straight out of reward_terms
        # (hmmwv_tracking_env.py:782), so having removed the term we must not let
        # it index a missing key. Pass a zero through for the parent's benefit,
        # then strip the entry so no throttle_brake number is ever logged for a
        # legged robot.
        if "throttle_brake" not in reward_terms:
            reward_terms = {**reward_terms,
                            "throttle_brake": torch.zeros_like(reward_terms["track_reward"])}
        extras = super()._make_extras(reward_terms, dones, time_outs)
        log = extras.get("log")
        if isinstance(log, dict):
            log.pop("/tracking/throttle_brake", None)
        return extras
