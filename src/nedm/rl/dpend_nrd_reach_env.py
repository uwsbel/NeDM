"""Goal-reaching for the planar double pendulum inside the frozen, decoder-free NRD.

Implements docs/vision/double_pen/NRD_double_pendulum_RL_task.md. The joint
NRD model advances ``[z1, z2, a]`` at 50 Hz; the policy acts every
``action_repeat`` transitions (5 -> 10 Hz) and its action is held in between.
Reward and success are evaluated at every 20 ms transition from the predicted
``z1`` and summed over the hold. The decoder is never called.

Two observation variants share everything else (plan section 5):

    z1 policy:     [normalize_state(z1), g, e]
    z1+z2 policy:  [normalize_state(z1), normalize_z2(z2), g, e]

with ``g = goal / L`` and ``e = (goal - tip) / L``. ``build_observation`` is the
single place that assembles it, so the Chrono transfer evaluation feeds true
state + camera-encoded latent through exactly the same code path.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from rsl_rl.env import VecEnv

from nedm.nrd.checkpoint import load_nrd_model
from nedm.nrd.context_bank import load_context_bank
from nedm.training.trainer import pendulum_tip_positions

DPEND_STATE_FIELDS = ["cos_q1", "sin_q1", "cos_q2", "sin_q2", "omega1_radps", "omega2_radps"]
TRIG_PAIRS = ((0, 1), (2, 3))
OMEGA_SLICE = slice(4, 6)

DEFAULT_NRD_CHECKPOINT = Path("artifacts/training_runs/dpend_nrd_full_v1/checkpoints/best_val.pt")
DEFAULT_TRAIN_CONTEXT_BANK = Path("artifacts/rl_reference_sets/dpend_nrd_full_v1_train_contexts_16384_seed20260826.npz")
DEFAULT_EVAL_CONTEXT_BANK = Path("artifacts/rl_reference_sets/dpend_nrd_full_v1_val_contexts_512_seed20260826.npz")


def default_env_cfg() -> dict[str, Any]:
    return {
        "num_envs": 4096,
        "device": "cuda",
        "seed": 1,
        "nrd_checkpoint": str(DEFAULT_NRD_CHECKPOINT),
        "context_bank": str(DEFAULT_TRAIN_CONTEXT_BANK),
        "observe_z2": True,
        "action_repeat": 5,
        "max_episode_steps": 50,
        "auto_reset": True,
        "link_lengths_m": [0.3, 0.3],
        "goal": {
            "theta_range_rad": [0.0, 2.0 * math.pi],
            "r_min_frac": 0.5,
            "r_max_frac": 0.8,
            "max_sample_attempts": 8,
        },
        "reward": {
            # "plan": the task document's shaping (distance + progress + smoothness,
            #         summed over the 20 ms transitions of a hold).
            # "exponential": the arm reach study's recipe -- exp(-d / ee_error_scale_m)
            #         averaged over the hold, an action-rate penalty per policy step,
            #         and the success bonus; every term is >= 0 except the action
            #         penalty, so failure terminations need no extra charge.
            "type": "plan",
            "ee_error_scale_m": 0.06,
            "action_rate_weight": 0.02,
            "distance_weight": 1.0,
            "progress_weight": 5.0,
            "success_bonus": 25.0,
            "angular_velocity_change_weight": 0.01,
            "success_tolerance_m": 0.01,
            # Failure terminations (spin guard, OOD latent, non-finite) charge the
            # distance penalty the episode would have kept paying at the failing
            # distance for its remaining horizon. Without this, the plan's negative
            # distance shaping makes "spin past 35 rad/s in 0.6 s" (return ~ -24)
            # cheaper than surviving 5 s (~ -150): observed in the first paired
            # runs, both policies collapsed to it. "remaining_distance" | "none".
            "failure_penalty_mode": "remaining_distance",
        },
        "termination": {
            "omega_limit_radps": 35.0,
            "z2_guard_margin": 1.5,
        },
        "logging": {
            "close_thresholds_m": [0.01, 0.02, 0.05, 0.1],
        },
    }


def merge_env_cfg(overrides: dict[str, Any] | None) -> dict[str, Any]:
    cfg = default_env_cfg()
    if not overrides:
        return cfg
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            merged_child = dict(cfg[key])
            merged_child.update(value)
            cfg[key] = merged_child
        else:
            cfg[key] = value
    return cfg


def renormalize_trig(states: torch.Tensor) -> torch.Tensor:
    """Project each (cos, sin) pair back onto the unit circle (plan section 4)."""
    out = states.clone()
    for cos_index, sin_index in TRIG_PAIRS:
        norm = torch.sqrt(out[..., cos_index] ** 2 + out[..., sin_index] ** 2).clamp_min(1e-6)
        out[..., cos_index] = out[..., cos_index] / norm
        out[..., sin_index] = out[..., sin_index] / norm
    return out


def draw_polar_goals(
    count: int, goal_cfg: dict[str, Any], total_length_m: float, generator: torch.Generator
) -> torch.Tensor:
    """Plan section 3: theta ~ U(range), r ~ U(r_min, r_max); x = r cos, z = r sin. CPU tensor (count, 2)."""
    theta_lo, theta_hi = (float(v) for v in goal_cfg["theta_range_rad"])
    r_lo = float(goal_cfg["r_min_frac"]) * total_length_m
    r_hi = float(goal_cfg["r_max_frac"]) * total_length_m
    theta = theta_lo + (theta_hi - theta_lo) * torch.rand(count, generator=generator)
    radius = r_lo + (r_hi - r_lo) * torch.rand(count, generator=generator)
    return torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1)


def make_eval_pairs(
    bank: dict[str, Any],
    num_pairs: int,
    seed: int,
    goal_cfg: dict[str, Any],
    link_lengths_m: list[float],
    success_tolerance_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed held-out (context, goal) pairs shared by every policy and by the NRD/Chrono evals."""
    num_contexts = int(bank["states"].shape[0])
    if num_pairs > num_contexts:
        raise ValueError(f"requested {num_pairs} pairs but the bank holds {num_contexts} contexts")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    context_ids = torch.randperm(num_contexts, generator=generator)[:num_pairs]
    last_states = torch.as_tensor(bank["states"][context_ids.numpy(), -1, :], dtype=torch.float32)
    tips = pendulum_tip_positions(last_states, [float(v) for v in link_lengths_m])
    total_length = float(sum(link_lengths_m))
    goals = draw_polar_goals(num_pairs, goal_cfg, total_length, generator)
    for _ in range(int(goal_cfg.get("max_sample_attempts", 8))):
        too_close = torch.linalg.norm(goals - tips, dim=-1) <= success_tolerance_m
        if not bool(too_close.any()):
            break
        goals[too_close] = draw_polar_goals(int(too_close.sum()), goal_cfg, total_length, generator)
    return context_ids, goals


class DPendNRDReachEnv(VecEnv):
    """RSL-RL VecEnv: elbow-torque goal reaching inside the frozen joint NRD."""

    def __init__(self, cfg: dict[str, Any] | None = None, device: str | torch.device | None = None) -> None:
        self.cfg = merge_env_cfg(cfg)
        if device is not None:
            self.cfg["device"] = str(device)
        self.device = torch.device(self.cfg["device"])
        self.num_envs = int(self.cfg["num_envs"])
        self.num_actions = 1
        self.action_repeat = int(self.cfg["action_repeat"])
        self.max_episode_length = int(self.cfg["max_episode_steps"])
        self.auto_reset = bool(self.cfg.get("auto_reset", True))
        self.observe_z2 = bool(self.cfg["observe_z2"])

        self.model, self.payload = load_nrd_model(Path(self.cfg["nrd_checkpoint"]), self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.metadata = self.payload["metadata"]
        self.state_fields = list(self.metadata["state_fields"])
        if self.state_fields != DPEND_STATE_FIELDS:
            raise ValueError(f"expected double-pendulum state layout {DPEND_STATE_FIELDS}, got {self.state_fields}")
        if len(self.metadata["action_fields"]) != 1:
            raise ValueError(f"expected a single elbow action, got {self.metadata['action_fields']}")
        self.dt_s = float(self.metadata["dt_s"])
        self.policy_dt_s = self.dt_s * self.action_repeat
        self.block_size = int(self.model.backbone.config.block_size)
        self.state_dim = len(self.state_fields)
        self.z2_dim = int(self.model.z2_dim)
        self.link_lengths = [float(v) for v in self.cfg["link_lengths_m"]]
        self.total_length = float(sum(self.link_lengths))

        self.state_mean = self.model.state_mean.to(self.device)
        self.state_std = torch.clamp(self.model.state_std.to(self.device), min=1.0e-6)
        self.z2_mean = self.model.z2_mean.to(self.device)
        self.z2_std = torch.clamp(self.model.z2_std.to(self.device), min=1.0e-6)
        self.omega_sigma = self.state_std[OMEGA_SLICE]

        bank = load_context_bank(Path(self.cfg["context_bank"]))
        self.bank_meta = bank["meta"]
        if int(self.bank_meta["block_size"]) != self.block_size:
            raise ValueError(f"context bank block {self.bank_meta['block_size']} != model block {self.block_size}")
        bank_z2_mean = torch.as_tensor(bank["z2_mean"], device=self.device)
        if not torch.allclose(bank_z2_mean, self.z2_mean, atol=1e-5):
            raise ValueError("context bank was encoded with a different NRD checkpoint (z2_mean mismatch)")
        self.bank_states = torch.as_tensor(bank["states"], dtype=torch.float32, device=self.device)
        self.bank_actions = torch.as_tensor(bank["actions"], dtype=torch.float32, device=self.device)
        self.bank_z2 = torch.as_tensor(bank["z2"], dtype=torch.float32, device=self.device)
        self.num_contexts = int(self.bank_states.shape[0])

        termination_cfg = self.cfg["termination"]
        self.omega_limit = float(termination_cfg["omega_limit_radps"])
        z2_margin = termination_cfg.get("z2_guard_margin")
        self.z2_guard = (
            float(z2_margin) * torch.as_tensor(bank["z2_norm_absmax"], dtype=torch.float32, device=self.device)
            if z2_margin is not None
            else None
        )
        reward_cfg = self.cfg["reward"]
        self.success_tolerance = float(reward_cfg["success_tolerance_m"])
        self.reward_type = str(reward_cfg.get("type", "plan")).lower()
        if self.reward_type not in {"plan", "exponential"}:
            raise ValueError(f"reward.type must be 'plan' or 'exponential', got {self.reward_type!r}")
        self.failure_penalty_mode = str(reward_cfg.get("failure_penalty_mode", "remaining_distance"))
        if self.failure_penalty_mode not in {"remaining_distance", "none"}:
            raise ValueError(f"reward.failure_penalty_mode must be 'remaining_distance' or 'none', got {self.failure_penalty_mode!r}")
        self.close_thresholds_m = [float(v) for v in self.cfg["logging"]["close_thresholds_m"]]

        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(self.cfg["seed"]))

        n, b = self.num_envs, self.block_size
        dev = self.device
        self.state_hist = torch.zeros(n, b, self.state_dim, dtype=torch.float32, device=dev)
        self.z2_hist = torch.zeros(n, b, self.z2_dim, dtype=torch.float32, device=dev)
        self.action_hist = torch.zeros(n, b, 1, dtype=torch.float32, device=dev)
        self.goal = torch.zeros(n, 2, dtype=torch.float32, device=dev)
        self.actions = torch.zeros(n, 1, dtype=torch.float32, device=dev)
        self.prev_actions = torch.zeros(n, 1, dtype=torch.float32, device=dev)
        self.dist_buf = torch.zeros(n, dtype=torch.float32, device=dev)
        self.initial_dist = torch.zeros(n, dtype=torch.float32, device=dev)
        self.min_dist = torch.zeros(n, dtype=torch.float32, device=dev)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=dev)
        self.rew_buf = torch.zeros(n, dtype=torch.float32, device=dev)
        self.reset_buf = torch.zeros(n, dtype=torch.long, device=dev)
        self.time_out_buf = torch.zeros(n, dtype=torch.bool, device=dev)
        self.success_buf = torch.zeros(n, dtype=torch.bool, device=dev)
        self.success_time_s = torch.zeros(n, dtype=torch.float32, device=dev)
        self.spin_buf = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ood_buf = torch.zeros(n, dtype=torch.bool, device=dev)
        self.nonfinite_buf = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ep_reward = torch.zeros(n, dtype=torch.float32, device=dev)
        self.ep_distance_term = torch.zeros(n, dtype=torch.float32, device=dev)
        self.ep_progress_term = torch.zeros(n, dtype=torch.float32, device=dev)
        self.ep_smooth_term = torch.zeros(n, dtype=torch.float32, device=dev)
        self.ep_bonus_term = torch.zeros(n, dtype=torch.float32, device=dev)
        self.ep_failure_term = torch.zeros(n, dtype=torch.float32, device=dev)
        self.ep_reach_term = torch.zeros(n, dtype=torch.float32, device=dev)
        self.ep_action_rate_term = torch.zeros(n, dtype=torch.float32, device=dev)
        self.ep_domega_sq_sum = torch.zeros(n, dtype=torch.float32, device=dev)
        self.ep_domega_max = torch.zeros(n, dtype=torch.float32, device=dev)
        self.ep_substeps = torch.zeros(n, dtype=torch.float32, device=dev)
        self.ep_action_abs_sum = torch.zeros(n, dtype=torch.float32, device=dev)
        self.ep_action_slew_sum = torch.zeros(n, dtype=torch.float32, device=dev)
        self.ep_action_saturated_sum = torch.zeros(n, dtype=torch.float32, device=dev)
        self.step_terms: dict[str, torch.Tensor] = {}
        self.extras: dict[str, Any] = {}

        self.num_obs = self.state_dim + (self.z2_dim if self.observe_z2 else 0) + 4
        self.obs_buf = torch.zeros(n, self.num_obs, dtype=torch.float32, device=dev)
        self.reset()

    @property
    def unwrapped(self) -> "DPendNRDReachEnv":
        return self

    # -- kinematics / observation -------------------------------------------------
    def tip_positions(self, states: torch.Tensor) -> torch.Tensor:
        return pendulum_tip_positions(states[..., :4], self.link_lengths)

    def build_observation(self, states: torch.Tensor, latents: torch.Tensor | None, goals: torch.Tensor) -> torch.Tensor:
        """Plan section 5 observation from RAW state (N, 6), RAW latent (N, z2) and goal (N, 2) in metres."""
        state_norm = (states - self.state_mean) / self.state_std
        tips = self.tip_positions(states)
        parts = [state_norm]
        if self.observe_z2:
            if latents is None:
                raise ValueError("this policy observes z2 but no latent was supplied")
            parts.append((latents - self.z2_mean) / self.z2_std)
        parts.append(goals / self.total_length)
        parts.append((goals - tips) / self.total_length)
        return torch.cat(parts, dim=-1)

    def _compute_observations(self) -> None:
        self.obs_buf = self.build_observation(self.state_hist[:, -1, :], self.z2_hist[:, -1, :], self.goal)
        self.extras = {"observations": {"critic": self.obs_buf}}

    # -- reset ----------------------------------------------------------------------
    def reset(self) -> tuple[torch.Tensor, dict]:
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_idx(env_ids)
        self._compute_observations()
        return self.obs_buf, self.extras

    def _sample_goals(self, env_ids: torch.Tensor) -> torch.Tensor:
        count = env_ids.numel()
        tips = self.tip_positions(self.state_hist[env_ids, -1, :])
        goals = draw_polar_goals(count, self.cfg["goal"], self.total_length, self.generator).to(self.device)
        for _ in range(int(self.cfg["goal"].get("max_sample_attempts", 8))):
            too_close = torch.linalg.norm(goals - tips, dim=-1) <= self.success_tolerance
            if not bool(too_close.any()):
                break
            goals[too_close] = draw_polar_goals(
                int(too_close.sum()), self.cfg["goal"], self.total_length, self.generator
            ).to(self.device)
        return goals

    def reset_idx(
        self,
        env_ids: torch.Tensor,
        context_ids: torch.Tensor | None = None,
        goals: torch.Tensor | None = None,
    ) -> None:
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if context_ids is None:
            context_ids = torch.randint(0, self.num_contexts, (env_ids.numel(),), generator=self.generator)
        context_ids = context_ids.to(device=self.device, dtype=torch.long)
        self.state_hist[env_ids] = self.bank_states[context_ids]
        self.z2_hist[env_ids] = self.bank_z2[context_ids]
        self.action_hist[env_ids] = self.bank_actions[context_ids]
        if goals is None:
            goals = self._sample_goals(env_ids)
        self.goal[env_ids] = goals.to(device=self.device, dtype=torch.float32)

        dist = torch.linalg.norm(self.goal[env_ids] - self.tip_positions(self.state_hist[env_ids, -1, :]), dim=-1)
        self.dist_buf[env_ids] = dist
        self.initial_dist[env_ids] = dist
        self.min_dist[env_ids] = dist
        # The last recorded action is what was in force when the context ended; the
        # first policy action's slew is measured against it.
        self.actions[env_ids] = self.action_hist[env_ids, -1, :]
        self.prev_actions[env_ids] = self.action_hist[env_ids, -1, :]
        self.episode_length_buf[env_ids] = 0
        self.rew_buf[env_ids] = 0.0
        self.reset_buf[env_ids] = 0
        self.time_out_buf[env_ids] = False
        self.success_buf[env_ids] = False
        self.success_time_s[env_ids] = 0.0
        self.spin_buf[env_ids] = False
        self.ood_buf[env_ids] = False
        self.nonfinite_buf[env_ids] = False
        for buffer in (
            self.ep_reward,
            self.ep_distance_term,
            self.ep_progress_term,
            self.ep_smooth_term,
            self.ep_bonus_term,
            self.ep_failure_term,
            self.ep_reach_term,
            self.ep_action_rate_term,
            self.ep_domega_sq_sum,
            self.ep_domega_max,
            self.ep_substeps,
            self.ep_action_abs_sum,
            self.ep_action_slew_sum,
            self.ep_action_saturated_sum,
        ):
            buffer[env_ids] = 0.0

    # -- rsl_rl interface -----------------------------------------------------------
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        self._compute_observations()
        return self.obs_buf, self.extras

    def get_privileged_observations(self) -> None:
        return None

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        raw = actions.to(device=self.device, dtype=torch.float32).view(self.num_envs, 1)
        saturated = (raw.abs() > 1.0).float().squeeze(-1)
        action = raw.clamp(-1.0, 1.0)
        self.prev_actions = self.actions
        self.actions = action
        slew = (action - self.prev_actions).abs().squeeze(-1)

        reward_cfg = self.cfg["reward"]
        w_dist = float(reward_cfg["distance_weight"])
        w_prog = float(reward_cfg["progress_weight"])
        w_smooth = float(reward_cfg["angular_velocity_change_weight"])
        bonus = float(reward_cfg["success_bonus"])
        length = self.total_length
        exponential = self.reward_type == "exponential"
        reach_scale = float(reward_cfg.get("ee_error_scale_m", 0.06))
        w_action_rate = float(reward_cfg.get("action_rate_weight", 0.0))

        active = ~(self.success_buf | self.spin_buf | self.ood_buf | self.nonfinite_buf)
        started_active = active.clone()
        reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        distance_term = torch.zeros_like(reward)
        progress_term = torch.zeros_like(reward)
        smooth_term = torch.zeros_like(reward)
        bonus_term = torch.zeros_like(reward)
        failure_term = torch.zeros_like(reward)
        reach_term = torch.zeros_like(reward)
        total_transitions = float(self.max_episode_length * self.action_repeat)
        newly_success_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for substep in range(self.action_repeat):
            self.action_hist[:, -1, :] = action
            previous_state = self.state_hist[:, -1, :]
            next_state, next_z2 = self.model.predict_next(self.state_hist, self.z2_hist, self.action_hist)
            next_state = renormalize_trig(next_state)
            self.state_hist = torch.roll(self.state_hist, shifts=-1, dims=1)
            self.z2_hist = torch.roll(self.z2_hist, shifts=-1, dims=1)
            self.action_hist = torch.roll(self.action_hist, shifts=-1, dims=1)
            self.state_hist[:, -1, :] = next_state
            self.z2_hist[:, -1, :] = next_z2
            self.action_hist[:, -1, :] = action

            nonfinite = ~(torch.isfinite(next_state).all(dim=-1) & torch.isfinite(next_z2).all(dim=-1))
            safe_state = torch.where(nonfinite.unsqueeze(-1), previous_state, next_state)
            omega = safe_state[:, OMEGA_SLICE]
            domega_norm = (omega - previous_state[:, OMEGA_SLICE]) / self.omega_sigma
            spin = (omega.abs() > self.omega_limit).any(dim=-1)
            if self.z2_guard is not None:
                z2_norm = (next_z2 - self.z2_mean) / self.z2_std
                ood = (torch.nan_to_num(z2_norm, nan=float("inf")).abs() > self.z2_guard).any(dim=-1)
            else:
                ood = torch.zeros_like(spin)

            tip = self.tip_positions(safe_state)
            dist = torch.linalg.norm(self.goal - tip, dim=-1)
            reached = dist <= self.success_tolerance
            newly_success = reached & active & ~nonfinite
            newly_spin = spin & active & ~newly_success
            newly_ood = ood & active & ~newly_success & ~newly_spin
            newly_nonfinite = nonfinite & active

            if exponential:
                r_dist = torch.zeros_like(dist)
                r_prog = torch.zeros_like(dist)
                r_reach = torch.exp(-dist / reach_scale) / self.action_repeat
            else:
                r_dist = -w_dist * dist / length
                r_prog = w_prog * (self.dist_buf - dist) / length
                r_reach = torch.zeros_like(dist)
            r_smooth = -w_smooth * (domega_norm**2).sum(dim=-1)
            r_bonus = bonus * newly_success.float()
            newly_failed = newly_spin | newly_ood | newly_nonfinite
            if self.failure_penalty_mode == "remaining_distance":
                elapsed = self.episode_length_buf.float() * self.action_repeat + substep + 1
                remaining = (total_transitions - elapsed).clamp_min(0.0)
                r_fail = -w_dist * (dist / length) * remaining * newly_failed.float()
            else:
                r_fail = torch.zeros_like(dist)
            mask = active.float()
            distance_term += r_dist * mask
            progress_term += r_prog * mask
            smooth_term += r_smooth * mask
            bonus_term += r_bonus
            failure_term += r_fail
            reach_term += r_reach * mask
            reward += (r_dist + r_prog + r_reach + r_smooth + r_bonus + r_fail) * mask

            self.dist_buf = torch.where(active, dist, self.dist_buf)
            self.min_dist = torch.where(active, torch.minimum(self.min_dist, dist), self.min_dist)
            self.ep_domega_sq_sum += (domega_norm**2).sum(dim=-1) * mask
            self.ep_domega_max = torch.maximum(self.ep_domega_max, domega_norm.abs().amax(dim=-1) * mask)
            self.ep_substeps += mask
            elapsed_s = (self.episode_length_buf.float() * self.action_repeat + substep + 1) * self.dt_s
            self.success_time_s = torch.where(newly_success, elapsed_s, self.success_time_s)
            self.success_buf |= newly_success
            self.spin_buf |= newly_spin
            self.ood_buf |= newly_ood
            self.nonfinite_buf |= newly_nonfinite
            newly_success_any |= newly_success
            active = active & ~(newly_success | newly_spin | newly_ood | newly_nonfinite)

        # Action-rate penalty once per policy step (arm recipe), only for envs that
        # were still running when this action was applied.
        action_rate_term = -w_action_rate * (action - self.prev_actions).pow(2).squeeze(-1) * started_active.float()
        reward += action_rate_term

        self.episode_length_buf += 1
        self.ep_action_abs_sum += action.abs().squeeze(-1)
        self.ep_action_slew_sum += slew
        self.ep_action_saturated_sum += saturated
        self.ep_reward += reward
        self.ep_distance_term += distance_term
        self.ep_progress_term += progress_term
        self.ep_smooth_term += smooth_term
        self.ep_bonus_term += bonus_term
        self.ep_failure_term += failure_term
        self.ep_reach_term += reach_term
        self.ep_action_rate_term += action_rate_term

        failed = self.spin_buf | self.ood_buf | self.nonfinite_buf
        time_outs = (self.episode_length_buf >= self.max_episode_length) & ~self.success_buf & ~failed
        dones = self.success_buf | failed | time_outs
        self.rew_buf = reward
        self.reset_buf = dones.long()
        self.time_out_buf = time_outs

        self.step_terms = {
            "distance_m": self.dist_buf,
            "reached": newly_success_any.float(),
            "action_abs": action.abs().squeeze(-1),
            "action_slew": slew,
            "action_saturated": saturated,
        }
        extras = self._make_extras(dones, time_outs)
        return_rewards = reward.clone()
        return_dones = dones.long().clone()
        if self.auto_reset:
            done_env_ids = dones.nonzero(as_tuple=False).flatten()
            if done_env_ids.numel() > 0:
                self.reset_idx(done_env_ids)
        self._compute_observations()
        extras["observations"] = {"critic": self.obs_buf}
        self.extras = extras
        return self.obs_buf, return_rewards, return_dones, self.extras

    # -- logging ----------------------------------------------------------------------
    def episode_records(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Per-env episode statistics (call before auto-reset, or with auto_reset=False)."""
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        substeps = torch.clamp(self.ep_substeps[ids], min=1.0)
        steps = torch.clamp(self.episode_length_buf[ids].float(), min=1.0)
        return {
            "success": self.success_buf[ids].float(),
            "spin": self.spin_buf[ids].float(),
            "ood": self.ood_buf[ids].float(),
            "nonfinite": self.nonfinite_buf[ids].float(),
            "time_out": self.time_out_buf[ids].float(),
            "episode_steps": self.episode_length_buf[ids].float(),
            "success_time_s": self.success_time_s[ids],
            "initial_distance_m": self.initial_dist[ids],
            "final_distance_m": self.dist_buf[ids],
            "min_distance_m": self.min_dist[ids],
            "reward": self.ep_reward[ids],
            "distance_term": self.ep_distance_term[ids],
            "progress_term": self.ep_progress_term[ids],
            "smooth_term": self.ep_smooth_term[ids],
            "bonus_term": self.ep_bonus_term[ids],
            "failure_term": self.ep_failure_term[ids],
            "reach_term": self.ep_reach_term[ids],
            "action_rate_term": self.ep_action_rate_term[ids],
            "domega_rms": torch.sqrt(self.ep_domega_sq_sum[ids] / substeps),
            "domega_max": self.ep_domega_max[ids],
            "action_abs_mean": self.ep_action_abs_sum[ids] / steps,
            "action_slew_mean": self.ep_action_slew_sum[ids] / steps,
            "action_saturated_frac": self.ep_action_saturated_sum[ids] / steps,
            "goal_x_m": self.goal[ids, 0],
            "goal_z_m": self.goal[ids, 1],
        }

    def _make_extras(self, dones: torch.Tensor, time_outs: torch.Tensor) -> dict[str, Any]:
        terms = self.step_terms
        log = {
            "/reach/distance_m": terms["distance_m"].mean(),
            "/reach/reached_rate": terms["reached"].mean(),
            "/reach/action_abs": terms["action_abs"].mean(),
            "/reach/action_slew": terms["action_slew"].mean(),
            "/reach/action_saturated": terms["action_saturated"].mean(),
        }
        extras: dict[str, Any] = {"observations": {"critic": self.obs_buf}, "time_outs": time_outs, "log": log}
        done_env_ids = dones.nonzero(as_tuple=False).flatten()
        if done_env_ids.numel() > 0:
            rec = self.episode_records(done_env_ids)
            episode = {
                "/episode/reward": rec["reward"].mean(),
                "/episode/length": rec["episode_steps"].mean(),
                "/episode/success_rate": rec["success"].mean(),
                "/episode/timeout_rate": rec["time_out"].mean(),
                "/episode/ood_rate": rec["ood"].mean(),
                "/episode/spin_rate": rec["spin"].mean(),
                "/episode/nonfinite_rate": rec["nonfinite"].mean(),
                "/episode/final_distance_m": rec["final_distance_m"].mean(),
                "/episode/min_distance_m": rec["min_distance_m"].mean(),
                "/episode/distance_term": rec["distance_term"].mean(),
                "/episode/progress_term": rec["progress_term"].mean(),
                "/episode/smooth_term": rec["smooth_term"].mean(),
                "/episode/bonus_term": rec["bonus_term"].mean(),
                "/episode/failure_term": rec["failure_term"].mean(),
                "/episode/reach_term": rec["reach_term"].mean(),
                "/episode/action_rate_term": rec["action_rate_term"].mean(),
                "/episode/domega_rms": rec["domega_rms"].mean(),
                "/episode/domega_max": rec["domega_max"].mean(),
                "/episode/action_abs_mean": rec["action_abs_mean"].mean(),
                "/episode/action_slew_mean": rec["action_slew_mean"].mean(),
                "/episode/action_saturated_frac": rec["action_saturated_frac"].mean(),
            }
            for threshold in self.close_thresholds_m:
                episode[f"/episode/min_distance_within_{self._threshold_label(threshold)}"] = (
                    rec["min_distance_m"] <= threshold
                ).float().mean()
            successes = rec["success"] > 0.5
            if bool(successes.any()):
                episode["/episode/time_to_success_s"] = rec["success_time_s"][successes].mean()
            upper = rec["goal_z_m"] > 0.0
            if bool(upper.any()):
                episode["/episode/success_rate_upper_goals"] = rec["success"][upper].mean()
            if bool((~upper).any()):
                episode["/episode/success_rate_lower_goals"] = rec["success"][~upper].mean()
            episode.update(log)
            extras["episode"] = episode
        return extras

    @staticmethod
    def _threshold_label(threshold_m: float) -> str:
        centimeters = threshold_m * 100.0
        if abs(centimeters - round(centimeters)) < 1.0e-6:
            return f"{int(round(centimeters))}cm"
        return f"{centimeters:.1f}cm".replace(".", "p")


__all__ = [
    "DEFAULT_EVAL_CONTEXT_BANK",
    "DEFAULT_NRD_CHECKPOINT",
    "DEFAULT_TRAIN_CONTEXT_BANK",
    "DPEND_STATE_FIELDS",
    "DPendNRDReachEnv",
    "default_env_cfg",
    "draw_polar_goals",
    "make_eval_pairs",
    "merge_env_cfg",
    "renormalize_trig",
]
