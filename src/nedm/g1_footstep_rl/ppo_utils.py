"""Small additions to RSL-RL 2.2.4 PPO used by scripts/g1_footstep_rl/train_footsteps_ppo.py (ported from the SO101
trainer, scripts/so101_push_rl/train_so101_planar_ppo.py):

| Function | What it does |
|---|---|
| split_optimizer | Two Adam groups: actor + std with PPO's adaptive rate, critic with a fixed rate (RSL-RL adapts one rate to the actor KL; with a small action noise that rate collapses and starves the critic) |
| install_std_floor | The action noise std never falls below a per-action floor (clamped after every PPO update) |
| set_std | Set the per-action std (initial values, or "raise to at least" at a curriculum promotion) |
| patch_bootstrap | Time-out bootstrap from the observation of the last state before the reset (extras["final_observations"]) with the current critic, instead of RSL-RL's V(observation before the step) |
| patch_curve | One CSV row per iteration (losses, timing, std, the logged env keys) |
| sha256, write_json | Run manifests |
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def split_optimizer(alg, critic_lr):
    """Replace alg.optimizer by Adam with two groups: actor + std (lr = PPO's adaptive alg.learning_rate) and critic
    (fixed critic_lr; RSL-RL's update() writes alg.learning_rate into every group, the patched step() resets group 1)."""
    ac = alg.actor_critic
    actor_params = list(ac.actor.parameters()) + [ac.std]
    ids = {id(p) for p in actor_params}
    critic_params = [p for p in ac.parameters() if id(p) not in ids]
    opt = torch.optim.Adam([{"params": actor_params, "lr": alg.learning_rate}, {"params": critic_params, "lr": critic_lr}])
    orig_step = opt.step

    def step(*args, **kw):
        opt.param_groups[1]["lr"] = critic_lr
        return orig_step(*args, **kw)

    opt.step = step
    alg.optimizer = opt


def per_action(values, groups, device):
    """Expand per-group values to a per-action tensor. values: a number (all actions) or a list with one value per
    group; groups: list of action counts per group (e.g. [3, 12] = command, residual)."""
    if isinstance(values, (int, float)):
        return torch.full((sum(groups),), float(values), device=device)
    if len(values) != len(groups):
        raise ValueError(f"{values}: need one value per action group {groups}")
    return torch.cat([torch.full((n,), float(v), device=device) for v, n in zip(values, groups)])


def set_std(ac, values, groups, at_least=False):
    """Set the action std of an RSL-RL ActorCritic (noise_std_type "scalar") per action group; at_least: only raise."""
    v = per_action(values, groups, ac.std.device)
    with torch.no_grad():
        ac.std.copy_(torch.maximum(ac.std, v) if at_least else v)


def install_std_floor(alg, floor, groups):
    """Clamp the action std to >= floor (per action group) after every PPO update and now."""
    ac = alg.actor_critic
    fl = per_action(floor, groups, ac.std.device)
    update = alg.update

    def update_with_floor(*args, **kwargs):
        out = update(*args, **kwargs)
        with torch.no_grad():
            ac.std.copy_(torch.maximum(ac.std, fl))
        return out

    alg.update = update_with_floor
    with torch.no_grad():
        ac.std.copy_(torch.maximum(ac.std, fl))


def patch_bootstrap(runner):
    """Time-out bootstrap: reward += gamma V(final observation) for envs with extras["time_outs"], with the current
    critic (RSL-RL itself adds gamma V(observation before the step), which is one step stale). With
    empirical_normalization the final observation is normalised like the critic's inputs (the runner's
    critic_obs_normalizer statistics, without updating them)."""
    alg = runner.alg
    orig = alg.process_env_step
    norm = getattr(runner, "critic_obs_normalizer", None)

    def normalise(x):
        if norm is not None and hasattr(norm, "_mean"):      # rsl_rl EmpiricalNormalization (Identity otherwise)
            return (x - norm._mean) / (norm._std + norm.eps)
        return x

    def process_env_step(rewards, dones, infos):
        to = infos.get("time_outs")
        if to is not None and "final_observations" in infos:
            to = to.to(alg.device).bool()
            rewards = rewards.clone()
            if to.any():
                idx = torch.nonzero(to).flatten()
                v = alg.actor_critic.evaluate(normalise(infos["final_observations"][idx].to(alg.device))).squeeze(-1)
                rewards[idx] += alg.gamma * v
            infos = {k: v for k, v in infos.items() if k != "time_outs"}
        return orig(rewards, dones, infos)

    alg.process_env_step = process_env_step


def patch_curve(runner, path, extra=None):
    """Append one CSV row per iteration: losses, timing, mean action std, learning rates, env steps/s, and the mean of
    every logged env key. extra(): optional dict of more columns."""
    orig = runner.log
    path = Path(path)
    state = {"header": None}
    if path.exists() and path.stat().st_size:          # a resumed run appends under the existing header
        with open(path, newline="") as f:
            state["header"] = next(csv.reader(f))

    def log(locs, width=80, pad=35):
        orig(locs, width, pad)
        n = runner.num_steps_per_env * runner.env.num_envs
        row = {"iteration": locs["it"], "collection_s": locs["collection_time"], "learn_s": locs["learn_time"],
               "env_steps_per_s": n / max(locs["collection_time"], 1e-9),
               "value_loss": locs["mean_value_loss"], "surrogate_loss": locs["mean_surrogate_loss"],
               "entropy": locs["mean_entropy"], "action_std_mean": float(runner.alg.actor_critic.std.mean()),
               "learning_rate": runner.alg.learning_rate, "critic_lr": runner.alg.optimizer.param_groups[-1]["lr"]}
        if extra is not None:
            row.update(extra())
        if locs["ep_infos"]:
            for k in locs["ep_infos"][0]:
                vals = [float(torch.as_tensor(e[k]).double().mean()) for e in locs["ep_infos"] if k in e]
                row[k.strip("/").replace("/", ".")] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        new = state["header"] is None
        if new:
            state["header"] = list(row)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=state["header"], extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)

    runner.log = log
