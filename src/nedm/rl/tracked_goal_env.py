"""Tracked-vehicle goal-reaching environment backed by the frozen drive NN dynamics.

The frozen ROM maps ``[vx, vy, r] + [steer, throttle, brake] -> delta[vx, vy, r]``.
This env wraps it as an RSL-RL ``VecEnv``: it keeps a rolling context window, steps the
ROM, integrates a kinematic pose ``(x, y, yaw)`` outside the network (the ROM has no
pose state), samples a 2-D goal, and rewards progress toward that goal. It mirrors the
proven machinery in ``hmmwv_tracking_env.py`` (pose integration, action scaling, NN
substep) and ``arm_reaching_env.py`` (real-data prefix seeding), with the goal-reach
reward design adapted from the earlier 4-DOF ``hmmwv_rl`` project.

Coverage caveat (see the v2 dataset scan): the ROM only saw throttle <= 0.6, forward
motion, and near-straight driving, so the action space is capped to that envelope to
keep the policy inside the region where the ROM is trustworthy.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rsl_rl.env import VecEnv

from nedm.rl.defaults import (
    DEFAULT_TRACKED_DYNAMICS_CHECKPOINT,
    DEFAULT_TRACKED_PROCESSED_DATASET_DIR,
)
from nedm.rl.dynamics import load_frozen_dynamics


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def default_env_cfg() -> dict[str, Any]:
    return {
        "num_envs": 2048,
        "device": "cuda",
        "dynamics_checkpoint": str(DEFAULT_TRACKED_DYNAMICS_CHECKPOINT),
        "processed_dataset_dir": str(DEFAULT_TRACKED_PROCESSED_DATASET_DIR),
        "seed_split": "train",
        # Warm-start the ROM context from real prefixes whose terminal speed is below
        # this, so each episode begins nearly at rest (goal-reach-from-rest). A constant
        # zero context is out-of-distribution for the transformer; real prefixes are not.
        "seed_speed_max_mps": 0.5,
        "dynamics_context_steps": None,
        "action_repeat": 5,           # 5 * dt(0.02) = 0.1 s policy control period
        "max_episode_steps": 200,     # 200 * 0.1 s = 20 s
        "auto_reset": True,
        # Action envelope — clamp to what the ROM actually saw in v2. Actions are
        # centered at the dataset-mean driver command (throttle ~0.24, brake ~0), so a
        # zero policy output is a gentle forward cruise (not a throttle/brake fight).
        "action": {
            "center": "dataset_mean",       # or explicit [steer, throttle, brake]
            "scale": [1.0, 0.4, 0.6],
            "throttle_max": 0.6,
            "brake_max": 0.6,
            "throttle_brake_conflict": True,  # effective_throttle = throttle*(1-brake) when braking
            "brake_conflict_deadzone": 0.2,
        },
        "goal": {
            "radius_m": [3.0, 8.0],
            # angle relative to the vehicle's initial forward heading (yaw=0 at reset).
            # Forward cone only: the ROM has no reverse and weak hard-turn authority.
            "angle_rad": [-1.5708, 1.5708],
        },
        "observation": {
            "cartesian_scale_m": 10.0,
        },
        "reward": {
            "progress_weight": 10.0,       # per-step (d_prev - d_curr) in metres
            "distance_weight": 0.0,        # optional -k * d_curr shaping (off by default)
            "heading_weight": 0.05,        # -k * |heading_error| (rad)
            "action_rate_weight": 0.01,    # -k * ||a - a_prev||^2
            "throttle_brake_weight": 0.05, # -k * throttle*brake
            "spin_weight": 0.5,            # -k * relu(|r| - r_safe)^2
            "spin_safe_radps": 0.4,
            "time_penalty": 0.0,           # -k per step (encourage speed; off by default)
            "success_bonus": 50.0,
            "success_tolerance_m": 0.75,
        },
        "termination": {
            "workspace_radius_m": 30.0,
        },
        "logging": {
            "close_thresholds_m": [0.75, 1.5, 3.0],
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


class TrackedGoalReachingEnv(VecEnv):
    """RSL-RL VecEnv for 2-D goal reaching inside a frozen tracked-vehicle ROM."""

    def __init__(self, cfg: dict[str, Any] | None = None, device: str | torch.device | None = None) -> None:
        self.cfg = merge_env_cfg(cfg)
        if device is not None:
            self.cfg["device"] = str(device)
        self.device = torch.device(self.cfg["device"])
        self.num_envs = int(self.cfg["num_envs"])
        self.num_actions = 3
        self.action_repeat = int(self.cfg["action_repeat"])
        self.max_episode_length = int(self.cfg["max_episode_steps"])
        self.auto_reset = bool(self.cfg.get("auto_reset", True))

        self.dynamics = load_frozen_dynamics(
            checkpoint_path=self.cfg["dynamics_checkpoint"],
            device=self.device,
            processed_dataset_dir=self.cfg.get("processed_dataset_dir"),
        )
        self.model = self.dynamics.model
        self.metadata = self.dynamics.metadata
        self.dt_s = self.dynamics.dt_s
        self.step_dt = self.dt_s * self.action_repeat
        self.context_steps = self.dynamics.context_steps
        if int(getattr(self.model, "num_terrains", 0)) != 0:
            raise ValueError("TrackedGoalReachingEnv expects a non-terrain-conditioned dynamics checkpoint")
        dynamics_context_cfg = self.cfg.get("dynamics_context_steps")
        if dynamics_context_cfg is None:
            self.dynamics_context_steps = self.context_steps
        else:
            self.dynamics_context_steps = int(dynamics_context_cfg)
            if not 1 <= self.dynamics_context_steps <= self.context_steps:
                raise ValueError(
                    f"dynamics_context_steps={self.dynamics_context_steps} must be in [1, {self.context_steps}]"
                )

        self.state_fields = list(self.metadata["state_fields"])
        self.action_fields = list(self.metadata["action_fields"])
        self.state_index = {name: i for i, name in enumerate(self.state_fields)}
        self._validate_fields()
        self.vx_idx = self.state_index["vel_body_x_mps"]
        self.vy_idx = self.state_index["vel_body_y_mps"]
        self.r_idx = self.state_index["yaw_rate_radps"]

        self.state_mean = self.model.state_mean.to(self.device)
        self.state_std = torch.clamp(self.model.state_std.to(self.device), min=1.0e-6)
        self.action_mean = self.model.action_mean.to(self.device)
        self.action_std = torch.clamp(self.model.action_std.to(self.device), min=1.0e-6)

        action_cfg = self.cfg["action"]
        self.throttle_max = float(action_cfg["throttle_max"])
        self.brake_max = float(action_cfg["brake_max"])
        self.throttle_brake_conflict = bool(action_cfg.get("throttle_brake_conflict", True))
        self.brake_conflict_deadzone = float(action_cfg.get("brake_conflict_deadzone", 0.2))
        self.action_low = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(
            [1.0, self.throttle_max, self.brake_max], dtype=torch.float32, device=self.device
        )
        center_cfg = action_cfg.get("center", "dataset_mean")
        if center_cfg == "dataset_mean":
            self.action_center = self.action_mean.clone()
        else:
            self.action_center = torch.tensor(center_cfg, dtype=torch.float32, device=self.device)
        self.action_scale = torch.tensor(action_cfg["scale"], dtype=torch.float32, device=self.device)

        goal_cfg = self.cfg["goal"]
        self.goal_radius = tuple(float(v) for v in goal_cfg["radius_m"])
        self.goal_angle = tuple(float(v) for v in goal_cfg["angle_rad"])
        self.cart_scale = float(self.cfg["observation"]["cartesian_scale_m"])
        self.workspace_radius = float(self.cfg["termination"]["workspace_radius_m"])
        self.success_tolerance = float(self.cfg["reward"]["success_tolerance_m"])
        self.close_thresholds_m = [float(v) for v in self.cfg.get("logging", {}).get("close_thresholds_m", [])]

        self.seed_states, self.seed_actions = self._load_seed_prefixes()
        self.num_seed_prefixes = int(self.seed_states.shape[0])

        state_dim = len(self.state_fields)
        action_dim = len(self.action_fields)
        self.state_hist = torch.zeros(self.num_envs, self.context_steps, state_dim, dtype=torch.float32, device=self.device)
        self.action_hist = torch.zeros(self.num_envs, self.context_steps, action_dim, dtype=torch.float32, device=self.device)
        self.pose = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.goal = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)
        self.distance_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float32, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_reward_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_progress_reward_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_min_distance = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.extras: dict[str, Any] = {}

        self.num_obs = self._observation_dim()
        self.num_privileged_obs = None
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float32, device=self.device)
        self.reset()

    @property
    def unwrapped(self) -> "TrackedGoalReachingEnv":
        return self

    def _validate_fields(self) -> None:
        required_state = ["vel_body_x_mps", "vel_body_y_mps", "yaw_rate_radps"]
        required_action = ["driver_steering", "driver_throttle", "driver_braking"]
        missing_state = [f for f in required_state if f not in self.state_index]
        if missing_state or list(self.action_fields) != required_action:
            raise ValueError(
                "TrackedGoalReachingEnv requires state fields "
                f"{required_state} and action fields {required_action}; "
                f"got state={self.state_fields}, action={self.action_fields}"
            )

    def _load_seed_prefixes(self) -> tuple[torch.Tensor, torch.Tensor]:
        processed_root = Path(self.cfg.get("processed_dataset_dir") or self.dynamics.config["processed_dataset_dir"])
        split = str(self.cfg.get("seed_split", "train"))
        states = np.load(processed_root / f"{split}_states.npy", mmap_mode="r")
        actions = np.load(processed_root / f"{split}_actions.npy", mmap_mode="r")
        starts = np.load(processed_root / f"{split}_episode_starts.npy")
        lengths = np.load(processed_root / f"{split}_episode_lengths.npy")
        valid = np.nonzero(lengths >= self.context_steps)[0]
        if valid.size == 0:
            raise ValueError(f"No {split} episodes have length >= context_steps={self.context_steps}")
        seed_states = np.stack(
            [np.asarray(states[starts[i] : starts[i] + self.context_steps], dtype=np.float32) for i in valid], axis=0
        )
        seed_actions = np.stack(
            [np.asarray(actions[starts[i] : starts[i] + self.context_steps], dtype=np.float32) for i in valid], axis=0
        )
        seed_states_t = torch.as_tensor(seed_states, dtype=torch.float32, device=self.device)
        seed_actions_t = torch.as_tensor(seed_actions, dtype=torch.float32, device=self.device)

        speed_max = float(self.cfg.get("seed_speed_max_mps", 0.0) or 0.0)
        if speed_max > 0.0:
            terminal_speed = seed_states_t[:, -1, self.vx_idx].abs()
            near_rest = terminal_speed < speed_max
            if int(near_rest.sum().item()) >= max(64, self.num_envs // 16):
                seed_states_t = seed_states_t[near_rest]
                seed_actions_t = seed_actions_t[near_rest]
        return seed_states_t, seed_actions_t

    def _observation_dim(self) -> int:
        # [vx, vy, r] + [goal_x_body, goal_y_body] + [dist] + [sin, cos heading] + last_action(3)
        return 3 + 2 + 1 + 2 + 3

    def reset(self) -> tuple[torch.Tensor, dict]:
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_idx(env_ids)
        self._compute_observations()
        return self.obs_buf, self.extras

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        seed_ids = torch.randint(0, self.num_seed_prefixes, (env_ids.numel(),), device=self.device)
        self.state_hist[env_ids] = self.seed_states[seed_ids]
        self.action_hist[env_ids] = self.seed_actions[seed_ids]
        self.pose[env_ids] = 0.0  # start at origin, yaw = 0
        self.goal[env_ids] = self._sample_goals(env_ids.numel())
        self.actions[env_ids] = self.action_hist[env_ids, -1, :]
        self.last_actions[env_ids] = self.actions[env_ids]
        self.distance_buf[env_ids] = torch.linalg.norm(self.goal[env_ids] - self.pose[env_ids, :2], dim=-1)
        self.episode_min_distance[env_ids] = self.distance_buf[env_ids]
        self.episode_length_buf[env_ids] = 0
        self.rew_buf[env_ids] = 0.0
        self.reset_buf[env_ids] = 0
        self.time_out_buf[env_ids] = False
        self.episode_reward_sum[env_ids] = 0.0
        self.episode_progress_reward_sum[env_ids] = 0.0

    def _sample_goals(self, count: int) -> torch.Tensor:
        r = self.goal_radius[0] + torch.rand(count, device=self.device) * (self.goal_radius[1] - self.goal_radius[0])
        a = self.goal_angle[0] + torch.rand(count, device=self.device) * (self.goal_angle[1] - self.goal_angle[0])
        return torch.stack([r * torch.cos(a), r * torch.sin(a)], dim=-1)

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        self._compute_observations()
        return self.obs_buf, self.extras

    def get_privileged_observations(self) -> None:
        return None

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        actions = actions.to(device=self.device, dtype=torch.float32)
        driver_actions = self._scale_policy_actions(actions)
        self.actions = driver_actions

        distance_before = self.distance_buf.clone()
        with torch.no_grad():
            for _ in range(self.action_repeat):
                self._nn_substep(driver_actions)

        self.episode_length_buf += 1
        rewards, reward_terms = self._compute_reward(driver_actions, distance_before)
        dones, time_outs, done_terms = self._compute_dones(reward_terms)
        self.rew_buf = rewards
        self.reset_buf = dones.long()
        self.time_out_buf = time_outs
        self.episode_reward_sum += rewards
        self.episode_progress_reward_sum += reward_terms["progress_reward"]
        self.episode_min_distance = torch.minimum(self.episode_min_distance, reward_terms["distance_m"])

        extras = self._make_extras(reward_terms | done_terms, dones, time_outs)
        self.last_actions = driver_actions.clone()
        return_rewards = rewards.clone()
        return_dones = dones.long().clone()
        if self.auto_reset:
            done_env_ids = dones.nonzero(as_tuple=False).flatten()
            if done_env_ids.numel() > 0:
                self.reset_idx(done_env_ids)
        self._compute_observations()
        extras["observations"] = {"critic": self.obs_buf}
        self.extras = extras
        return self.obs_buf, return_rewards, return_dones, self.extras

    def _scale_policy_actions(self, policy_actions: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(policy_actions)  # each in (-1, 1)
        driver = torch.clamp(self.action_center + self.action_scale * bounded, self.action_low, self.action_high)
        if self.throttle_brake_conflict:
            braking = driver[:, 2] > self.brake_conflict_deadzone
            driver = driver.clone()
            driver[:, 1] = torch.where(braking, driver[:, 1] * (1.0 - driver[:, 2]), driver[:, 1])
        return driver

    def _nn_substep(self, driver_actions: torch.Tensor) -> None:
        self.action_hist[:, -1, :] = driver_actions
        k = self.dynamics_context_steps
        delta = self.model.predict_next_delta(self.state_hist[:, -k:, :], self.action_hist[:, -k:, :])
        next_state = self.state_hist[:, -1, :] + delta
        self.pose = self._integrate_pose(self.pose, next_state)
        self.state_hist = torch.roll(self.state_hist, shifts=-1, dims=1)
        self.action_hist = torch.roll(self.action_hist, shifts=-1, dims=1)
        self.state_hist[:, -1, :] = next_state
        self.action_hist[:, -1, :] = driver_actions

    def _integrate_pose(self, pose: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        yaw_rate = next_state[:, self.r_idx]
        vx_body = next_state[:, self.vx_idx]
        vy_body = next_state[:, self.vy_idx]
        yaw_next = pose[:, 2] + self.dt_s * yaw_rate
        cos_yaw = torch.cos(yaw_next)
        sin_yaw = torch.sin(yaw_next)
        vx_world = cos_yaw * vx_body - sin_yaw * vy_body
        vy_world = sin_yaw * vx_body + cos_yaw * vy_body
        x_next = pose[:, 0] + self.dt_s * vx_world
        y_next = pose[:, 1] + self.dt_s * vy_world
        return torch.stack([x_next, y_next, yaw_next], dim=-1)

    def _goal_body_frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx = self.goal[:, 0] - self.pose[:, 0]
        dy = self.goal[:, 1] - self.pose[:, 1]
        cos_yaw = torch.cos(self.pose[:, 2])
        sin_yaw = torch.sin(self.pose[:, 2])
        gx_body = cos_yaw * dx + sin_yaw * dy
        gy_body = -sin_yaw * dx + cos_yaw * dy
        distance = torch.sqrt(dx * dx + dy * dy)
        return gx_body, gy_body, distance

    def _compute_reward(
        self, driver_actions: torch.Tensor, distance_before: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reward_cfg = self.cfg["reward"]
        gx_body, gy_body, distance = self._goal_body_frame()
        self.distance_buf = distance
        heading_error = torch.atan2(gy_body, gx_body)
        yaw_rate = self.state_hist[:, -1, self.r_idx]

        progress = distance_before - distance
        progress_reward = float(reward_cfg["progress_weight"]) * progress
        distance_penalty = -float(reward_cfg["distance_weight"]) * distance
        heading_penalty = -float(reward_cfg["heading_weight"]) * heading_error.abs()
        action_rate = torch.sum(torch.square(driver_actions - self.last_actions), dim=-1)
        action_rate_penalty = -float(reward_cfg["action_rate_weight"]) * action_rate
        throttle_brake = driver_actions[:, 1] * driver_actions[:, 2]
        throttle_brake_penalty = -float(reward_cfg["throttle_brake_weight"]) * throttle_brake
        spin = torch.clamp(yaw_rate.abs() - float(reward_cfg["spin_safe_radps"]), min=0.0)
        spin_penalty = -float(reward_cfg["spin_weight"]) * torch.square(spin)
        time_penalty = -float(reward_cfg["time_penalty"]) * torch.ones_like(distance)

        success = distance < self.success_tolerance
        success_bonus = float(reward_cfg["success_bonus"]) * success.float()

        reward = (
            progress_reward
            + distance_penalty
            + heading_penalty
            + action_rate_penalty
            + throttle_brake_penalty
            + spin_penalty
            + time_penalty
            + success_bonus
        )
        terms = {
            "distance_m": distance,
            "progress_reward": progress_reward,
            "heading_error_abs_rad": heading_error.abs(),
            "action_rate": action_rate,
            "throttle_brake": throttle_brake,
            "spin_excess_radps": spin,
            "success": success.float(),
            "success_bonus": success_bonus,
        }
        for threshold in self.close_thresholds_m:
            terms[f"close_{self._threshold_label(threshold)}"] = (distance < threshold).float()
        return reward, terms

    def _compute_dones(
        self, reward_terms: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        distance = reward_terms["distance_m"]
        success = distance < self.success_tolerance
        out_of_bounds = torch.linalg.norm(self.pose[:, :2], dim=-1) > self.workspace_radius
        invalid = (~torch.isfinite(self.state_hist[:, -1, :]).all(dim=-1)) | (~torch.isfinite(self.pose).all(dim=-1))
        time_outs = self.episode_length_buf >= self.max_episode_length
        failed = out_of_bounds | invalid
        dones = success | failed | time_outs
        return dones, time_outs, {
            "success_done": success.float(),
            "out_of_bounds_done": out_of_bounds.float(),
            "invalid_done": invalid.float(),
            "time_out": time_outs.float(),
        }

    def _make_extras(
        self, terms: dict[str, torch.Tensor], dones: torch.Tensor, time_outs: torch.Tensor
    ) -> dict[str, Any]:
        log = {
            "/goal/distance_m": terms["distance_m"].mean(),
            "/goal/progress_reward": terms["progress_reward"].mean(),
            "/goal/heading_error_abs_rad": terms["heading_error_abs_rad"].mean(),
            "/goal/action_rate": terms["action_rate"].mean(),
            "/goal/throttle_brake": terms["throttle_brake"].mean(),
            "/goal/spin_excess_radps": terms["spin_excess_radps"].mean(),
            "/goal/success_rate_step": terms["success"].mean(),
        }
        extras: dict[str, Any] = {
            "observations": {"critic": self.obs_buf},
            "time_outs": time_outs,
            "log": log,
        }
        done_env_ids = dones.nonzero(as_tuple=False).flatten()
        if done_env_ids.numel() > 0:
            lengths = torch.clamp(self.episode_length_buf[done_env_ids].float(), min=1.0)
            episode = {
                "/episode/reward": self.episode_reward_sum[done_env_ids].mean(),
                "/episode/length": lengths.mean(),
                "/episode/mean_progress_reward": (self.episode_progress_reward_sum[done_env_ids] / lengths).mean(),
                "/episode/final_distance_m": terms["distance_m"][done_env_ids].mean(),
                "/episode/min_distance_m": self.episode_min_distance[done_env_ids].mean(),
                "/episode/success_rate": terms["success_done"][done_env_ids].mean(),
                "/episode/out_of_bounds_rate": terms["out_of_bounds_done"][done_env_ids].mean(),
                "/episode/timeout_rate": terms["time_out"][done_env_ids].mean(),
            }
            for threshold in self.close_thresholds_m:
                label = self._threshold_label(threshold)
                episode[f"/episode/close_{label}_rate"] = terms[f"close_{label}"][done_env_ids].mean()
            episode.update(log)
            extras["episode"] = episode
        return extras

    @staticmethod
    def _threshold_label(threshold_m: float) -> str:
        if abs(threshold_m - round(threshold_m)) < 1.0e-6:
            return f"{int(round(threshold_m))}m"
        return f"{threshold_m:.2f}m".replace(".", "p")

    def _compute_observations(self) -> None:
        current_state = self.state_hist[:, -1, :]
        vel = current_state[:, [self.vx_idx, self.vy_idx, self.r_idx]]
        vel_norm = vel / self.state_std[[self.vx_idx, self.vy_idx, self.r_idx]]
        gx_body, gy_body, distance = self._goal_body_frame()
        heading_error = torch.atan2(gy_body, gx_body)
        last_action_norm = (self.actions - self.action_mean) / self.action_std
        self.obs_buf = torch.cat(
            [
                vel_norm,
                (gx_body / self.cart_scale).unsqueeze(-1),
                (gy_body / self.cart_scale).unsqueeze(-1),
                (distance / self.cart_scale).unsqueeze(-1),
                torch.sin(heading_error).unsqueeze(-1),
                torch.cos(heading_error).unsqueeze(-1),
                last_action_norm,
            ],
            dim=-1,
        )
        self.extras = {"observations": {"critic": self.obs_buf}}

    def current_pose(self) -> torch.Tensor:
        return self.pose

    def current_state(self) -> torch.Tensor:
        return self.state_hist[:, -1, :]


__all__ = ["TrackedGoalReachingEnv", "default_env_cfg", "merge_env_cfg"]
