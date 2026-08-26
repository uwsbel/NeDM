"""Teacher-student distillation of the double-pendulum NRD reaching policy.

Plan: docs/vision/double_pen/NRD_double_pendulum_teacher_student_distillation_plan.md.

Teacher: the frozen privileged policy (observes normalized z1 + goal + exact goal
error; the environment's own ``build_observation`` with ``observe_z2=False``).
Student: observes only a fixed history of ``H`` normalized camera latents, taken
once per policy decision (10 Hz), plus the normalized goal:

    o_S = [z2~_{k-H+1}, ..., z2~_k, g / L]        (H * z2_dim + 2 = 258 for H=4)

Inside the frozen NRD the latents are the recursively predicted ones the
environment already carries; on reset the history is filled from the recorded
context at indices ``[block-1 - action_repeat*(H-1), ..., block-1]`` (= [0, 5,
10, 15]), i.e. four latents 0.1 s apart. Histories never cross episodes.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from rsl_rl.runners import OnPolicyRunner

from nedm.rl.dpend_nrd_reach_env import DPendNRDReachEnv


# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------
def load_runner_checkpoint(runner: OnPolicyRunner, checkpoint_path: Path, device: str) -> int:
    loaded = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)
    runner.alg.actor_critic.load_state_dict(loaded["model_state_dict"])
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded["critic_obs_norm_state_dict"])
    runner.current_learning_iteration = int(loaded["iter"])
    return int(loaded["iter"])


def load_teacher(train_cfg: dict[str, Any], checkpoint_path: Path, env: DPendNRDReachEnv, device: str) -> tuple[Callable, int]:
    """Deterministic teacher (act_inference through its stored empirical normalizer)."""
    if env.observe_z2:
        raise ValueError("the teacher environment must be built with observe_z2=False (privileged z1 observation)")
    # The runner pops class_name keys out of the nested cfg dicts: give it its own deep copy.
    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir=None, device=device)
    iteration = load_runner_checkpoint(runner, checkpoint_path, device)
    policy = runner.get_inference_policy(device=device)

    @torch.no_grad()
    def teacher(obs: torch.Tensor) -> torch.Tensor:
        return policy(obs).clamp(-1.0, 1.0)

    return teacher, iteration


# ---------------------------------------------------------------------------
# Student
# ---------------------------------------------------------------------------
class StudentPolicy(nn.Module):
    """Fixed-history MLP: [H x normalized z2, normalized goal] -> tanh action."""

    def __init__(self, history_len: int, z2_dim: int, hidden_dims: tuple[int, ...] = (256, 128, 64), goal_dim: int = 2) -> None:
        super().__init__()
        self.history_len = int(history_len)
        self.z2_dim = int(z2_dim)
        self.goal_dim = int(goal_dim)
        self.input_dim = self.history_len * self.z2_dim + self.goal_dim
        layers: list[nn.Module] = []
        width = self.input_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(width, int(hidden)), nn.ELU()]
            width = int(hidden)
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class StudentHistory:
    """Per-env rolling window of ``history_len`` NORMALIZED latents at the policy rate."""

    def __init__(self, env: DPendNRDReachEnv, history_len: int) -> None:
        self.env = env
        self.history_len = int(history_len)
        block, repeat = env.block_size, env.action_repeat
        self.context_indices = [block - 1 - repeat * (self.history_len - 1 - i) for i in range(self.history_len)]
        if self.context_indices[0] < 0:
            raise ValueError(f"history_len={history_len} x action_repeat={repeat} does not fit in block_size={block}")
        self.buf = torch.zeros(env.num_envs, self.history_len, env.z2_dim, dtype=torch.float32, device=env.device)

    def _normalize(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.env.z2_mean) / self.env.z2_std

    def reset_from_env(self, env_ids: torch.Tensor | None = None) -> None:
        """Fill from the env's current context window (call right after reset_idx)."""
        if env_ids is None:
            self.buf[:] = self._normalize(self.env.z2_hist[:, self.context_indices, :])
            return
        if env_ids.numel() == 0:
            return
        self.buf[env_ids] = self._normalize(self.env.z2_hist[env_ids][:, self.context_indices, :])

    def push_from_env(self) -> None:
        """Append the env's latest (recursively predicted) latent for every env."""
        self.buf = torch.roll(self.buf, shifts=-1, dims=1)
        self.buf[:, -1, :] = self._normalize(self.env.z2_hist[:, -1, :])

    def after_step(self, dones: torch.Tensor) -> None:
        """Advance histories after env.step(): push for all, then re-seed the auto-reset envs."""
        self.push_from_env()
        if self.env.auto_reset:
            done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
            self.reset_from_env(done_ids)

    def observation(self) -> torch.Tensor:
        return torch.cat([self.buf.flatten(1), self.env.goal / self.env.total_length], dim=-1)


def build_student_observation(history: torch.Tensor, goals: torch.Tensor, total_length: float) -> torch.Tensor:
    return torch.cat([history.flatten(1), goals / total_length], dim=-1)


# ---------------------------------------------------------------------------
# Replay buffer (FIFO over samples)
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.obs = torch.empty(self.capacity, obs_dim, dtype=torch.float32, device=device)
        self.actions = torch.empty(self.capacity, 1, dtype=torch.float32, device=device)
        self.ptr = 0
        self.size = 0

    def add(self, obs: torch.Tensor, actions: torch.Tensor) -> None:
        n = obs.shape[0]
        if n >= self.capacity:
            obs, actions = obs[-self.capacity :], actions[-self.capacity :]
            n = self.capacity
        end = self.ptr + n
        if end <= self.capacity:
            self.obs[self.ptr : end] = obs
            self.actions[self.ptr : end] = actions
        else:
            first = self.capacity - self.ptr
            self.obs[self.ptr :] = obs[:first]
            self.actions[self.ptr :] = actions[:first]
            self.obs[: n - first] = obs[first:]
            self.actions[: n - first] = actions[first:]
        self.ptr = end % self.capacity
        self.size = min(self.size + n, self.capacity)

    def batches(self, batch_size: int, epochs: int, generator: torch.Generator | None = None):
        for _ in range(epochs):
            order = torch.randperm(self.size, device=self.obs.device, generator=generator)
            for start in range(0, self.size, batch_size):
                idx = order[start : start + batch_size]
                yield self.obs[idx], self.actions[idx]


# ---------------------------------------------------------------------------
# Closed-loop rollouts on fixed held-out pairs (teacher or student in control)
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout_pairs(
    env: DPendNRDReachEnv,
    teacher: Callable,
    student: StudentPolicy | None,
    controller: str,
    context_ids: torch.Tensor,
    goals: torch.Tensor,
    max_steps: int,
    history_len: int,
) -> dict[str, np.ndarray]:
    """Run ``controller`` ('teacher' | 'student') on the pairs; both policies are
    queried at every step so action agreement on the visited states is recorded."""
    if controller not in {"teacher", "student"}:
        raise ValueError(controller)
    if controller == "student" and student is None:
        raise ValueError("student controller requested without a student")
    if env.auto_reset:
        raise ValueError("rollout_pairs needs auto_reset=False")
    num = context_ids.numel()
    env_ids = torch.arange(num, device=env.device)
    env.reset_idx(env_ids, context_ids, goals)
    history = StudentHistory(env, history_len)
    history.reset_from_env()
    obs_teacher, _ = env.get_observations()

    tips = [env.tip_positions(env.state_hist[:, -1, :]).cpu().numpy()]
    executed, teacher_actions, student_actions, active_mask = [], [], [], []
    done_step = np.full(num, -1, dtype=np.int64)
    for step in range(max_steps):
        # auto_reset is off, so every termination flag persists for the rest of the loop.
        active = ~(env.success_buf | env.spin_buf | env.ood_buf | env.nonfinite_buf | env.time_out_buf)
        a_t = teacher(obs_teacher)
        a_s = student(history.observation()) if student is not None else torch.zeros_like(a_t)
        action = a_t if controller == "teacher" else a_s
        obs_teacher, _, dones, _ = env.step(action)
        history.push_from_env()
        executed.append(action.squeeze(-1).cpu().numpy())
        teacher_actions.append(a_t.squeeze(-1).cpu().numpy())
        student_actions.append(a_s.squeeze(-1).cpu().numpy())
        active_mask.append(active.cpu().numpy())
        tips.append(env.tip_positions(env.state_hist[:, -1, :]).cpu().numpy())
        newly = (dones.cpu().numpy() > 0) & (done_step < 0)
        done_step[newly] = step + 1
        if bool(dones.all()):
            break
    records = {key: value.cpu().numpy() for key, value in env.episode_records().items()}
    records["episode_steps"] = np.where(done_step > 0, done_step, len(executed)).astype(np.float32)
    records["tips"] = np.stack(tips, axis=1)
    records["executed_actions"] = np.stack(executed, axis=1)
    records["teacher_actions"] = np.stack(teacher_actions, axis=1)
    records["student_actions"] = np.stack(student_actions, axis=1)
    records["active_mask"] = np.stack(active_mask, axis=1)
    records["min_distance_fine_m"] = records["min_distance_m"].copy()
    return records


def action_agreement(records: dict[str, np.ndarray]) -> dict[str, float]:
    """Teacher-vs-student action statistics over the steps where the episode was still running."""
    mask = records["active_mask"].astype(bool)
    t = records["teacher_actions"][mask]
    s = records["student_actions"][mask]
    err = np.abs(s - t)
    return {
        "samples": int(mask.sum()),
        "mae": float(err.mean()) if err.size else float("nan"),
        "rmse": float(np.sqrt((err**2).mean())) if err.size else float("nan"),
        "p95_abs_error": float(np.percentile(err, 95)) if err.size else float("nan"),
        "sign_disagreement_rate": float((np.sign(s) != np.sign(t)).mean()) if err.size else float("nan"),
        "teacher_action_abs_mean": float(np.abs(t).mean()) if t.size else float("nan"),
        "student_action_abs_mean": float(np.abs(s).mean()) if s.size else float("nan"),
    }


__all__ = [
    "ReplayBuffer",
    "StudentHistory",
    "StudentPolicy",
    "action_agreement",
    "build_student_observation",
    "load_runner_checkpoint",
    "load_teacher",
    "rollout_pairs",
]
