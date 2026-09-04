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

3. TERMINATION ROLL AND PITCH ARE A MODEL-VALIDITY BOUND, NOT A FALL TEST. That
   was the first framing and it was wrong: the collection recorded ZERO falls in
   1,120 episodes, so the model has never seen one and cannot predict one. What
   the thresholds actually do is stop the policy optimising past the edge of the
   training data, where the model extrapolates freely. Set just above the
   observed maxima -- see the measurement at the cfg below.

4. THE REWARD BASELINE IS THE ANCHOR'S RL RUN, NOT default_env_cfg(). The
   anchor overrode the module defaults on its command line, so the values it
   actually trained with -- weights 2.0/1.6/0.2/0.2, state error restricted to
   the three controllable velocities -- are not the ones a fresh
   default_env_cfg() hands you. Inheriting the module defaults and calling that
   parity would have been wrong by 10x on the action-rate weight alone.

   The sigmas beneath those weights are then solved from OUR measured error
   scales, because weight * (error/sigma)^2 lets a mis-scaled sigma silently
   overrule the weight beside it. At the anchor's own sigmas, yaw_loss is 40x
   position_loss on Go2-scale errors and the 2.0-vs-1.6 weight ratio expresses
   nothing at all.

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
    # THE BASELINE HERE IS THE ANCHOR'S RL RUN, NOT default_env_cfg(). Those are
    # different configurations and I had been inheriting the wrong one:
    # hmmwv_rl_15d_crm2000mix25_onehot_ofatL8_bestval51 overrode the module
    # defaults from the command line, so its weights are position 2.0 / yaw 1.6 /
    # state 0.2 / action_rate 0.2, against the module's 1.0 / 0.8 / 0.2 / 0.02.
    # The action-rate weight alone differs by 10x. Parity means the run's values.
    cfg["reward"] = {
        **cfg["reward"],
        "position_weight": 2.0,
        "yaw_weight": 1.6,
        "state_weight": 0.2,
        "action_rate_weight": 0.2,
        "state_sigma": 1.0,
        # THE ANCHOR RESTRICTS THE STATE TERM TO THE THREE CONTROLLABLE
        # VELOCITIES, and all three exist verbatim in our state vector. Left at
        # None it averages all 15 fields, eight of which are per-foot normal
        # forces and slip speeds -- gait-phase quantities that a velocity command
        # cannot steer, since our footfall phase relative to the reference's is
        # arbitrary. Measured, those eight are 47% of the 15-field mean, which is
        # about their share by count: they do NOT dominate the value. That is the
        # trap. Including them barely moves the number while pointing half the
        # gradient at an objective the action space cannot reach, so the
        # difference would never appear in a log.
        "state_error_fields": ["vel_body_x_mps", "vel_body_y_mps", "yaw_rate_radps"],
        # SIGMAS ARE CHOSEN SO THE ANCHOR'S WEIGHTS ACTUALLY SET THE BALANCE.
        # Each channel enters as weight * (error / sigma)^2, so a sigma carrying
        # the wrong unit scale silently overrides the weight beside it -- the same
        # failure as any normalisation that hides what it divided by. At the
        # anchor's own sigmas (position 2.0 m, yaw 0.35 rad) and our measured
        # errors, position_loss spans 0.001-0.103 while yaw_loss spans 0.29-4.13:
        # yaw is 40x position and the 2.0-vs-1.6 weight ratio expresses nothing.
        #
        # Measured under a random policy on this env, 256 envs x 120 steps:
        #     position  p10 0.079  p50 0.295  p90 0.454 m
        #     yaw       mean 0.562 rad
        #     state     mean squared normalised error 2.138 over the three fields
        # Solving for equal-to-weight-ratio contributions at the median gives:
        "position_sigma_m": 0.55,
        "yaw_sigma_rad": 0.70,
        # which yields, at p50: position 0.575, yaw 0.522, state 0.428 -- a 1.10
        # position:yaw ratio against the weights' 1.25, and track_reward graded
        # 0.60 / 0.58 / 0.22 / 0.06 from a well-tracked state out to the random
        # policy. The previous 0.40 was solved against the module default weight
        # of 1.0; at the anchor's 2.0 it collapses to 0.0008 at p90.
    }
    # KEPT AT 0.0 RATHER THAN REMOVED (the anchor runs it at 0.05). The parent
    # indexes this key directly (hmmwv_tracking_env.py:741), so deleting it
    # raises on the first step. The term is driver_actions[:,1]*driver_actions[:,2],
    # which for the HMMWV penalises pressing both pedals at once; for us those
    # channels are cmd_vy and cmd_wz, so it would penalise TURNING WHILE
    # STRAFING. It is neutralised here, dropped from the reward terms in
    # _compute_reward, and stripped from the log in _make_extras, so nothing
    # downstream ever sees a throttle_brake number. Zeroing the weight ALONE
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
