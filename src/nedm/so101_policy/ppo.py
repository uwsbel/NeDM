"""PPO for the SO101 + cube goal pick-and-place policy (``vec_env.PickPlaceVecEnv``; trainer
``scripts/so101_policy/train_ppo.py``), adapted from the G1 study (worktree t3code-7d8d61fa:
``src/nedm/g1_policy/ppo_utils.py`` + ``scripts/g1_policy/train_planks_ppo.py`` on RSL-RL 2.2.4; read only, copied
and changed here):

| Piece | Here | G1 |
|---|---|---|
| PPO update | own loop with the RSL-RL 2.2.4 PPO equations: GAE(lambda) with done cuts, advantages normalised over the batch, clipped surrogate, clipped value loss (clip_param), adaptive-KL learning rate (x 1.5 / 1.5 at 2 x / 0.5 x desired_kl, 1e-5..1e-2), minibatch epochs, grad-norm clip over all parameters (``tests/so101_policy/test_ppo.py`` checks GAE and the losses against rsl_rl) | rsl_rl.algorithms.PPO |
| optimizer | ``split_optimizer``: Adam, actor + std at the adaptive rate, critic at a fixed rate | ppo_utils.split_optimizer |
| action std | state-independent, per action group (``vec_env.ACTION_GROUPS``: move 4, trigger 1, lift 1), clamped to [floor, cap] after every optimizer step; ``set_std`` (init, or raise to at least at a curriculum promotion) | install_std_floor, set_std (no cap) |
| observation normaliser | ``RunningNorm`` inside the policy module (float64 running mean / variance over every observation the rollouts see, (x - mean) / (std + 1e-2) clipped to +-``norm_clip``; saved with the policy, frozen outside ``collect``) | RSL-RL EmpiricalNormalization (off in G1); its runner skipped it on the first observation of each learn() call |
| time-out bootstrap | reward += gamma V(final observation) at time-outs (current critic, normaliser not updated) | patch_bootstrap |
| action mask | ``vec_env`` marks the action dims a decision used; the others leave the log-likelihood, the entropy and the KL (review P1-10) | none |
| learning curve | ``CurveWriter``: one CSV row per iteration (losses, KL, rates, std per group, episode outcomes, end codes, world steps, wall clock, GPU memory) | patch_curve |

Policy (``Policy``): actor and critic MLPs (default 512-256-128 ELU, separate), Gaussian on the pre-tanh action (the env
squashes with tanh; the trigger is a[4] > 0); the actor's last layer x ``actor_out_gain`` and its bias = ``action_bias``
(default trigger -2: P(trigger) = 2.3 % per decision at std 1). ``act_inference(obs)`` = the deterministic pre-tanh
mean. Checkpoints (``PPO.save``) hold the policy, its constructor arguments, the optimizer, the learning rate, the
counters and the RNG states; ``load_policy(path)`` rebuilds the policy for evaluation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from nedm.so101_policy.task import ENDS

LOG2PI = math.log(2 * math.pi)
ACT = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}


def train_cfg_default(seed=1) -> dict:
    """PPO defaults (G1 train_cfg_default values; gamma per 0.1 s decision)."""
    return {
        "algorithm": {"clip_param": 0.2, "desired_kl": 0.01, "entropy_coef": 0.002, "gamma": 0.99, "lam": 0.95,
                      "learning_rate": 3e-4, "max_grad_norm": 1.0, "num_learning_epochs": 5, "num_mini_batches": 8,
                      "schedule": "adaptive", "use_clipped_value_loss": True, "value_loss_coef": 1.0},
        "policy": {"actor_hidden_dims": [512, 256, 128], "critic_hidden_dims": [512, 256, 128], "activation": "elu",
                   "actor_out_gain": 0.01, "action_bias": [0.0, 0.0, 0.0, 0.0, -2.0, 0.0], "norm_clip": 10.0},
        "num_steps_per_env": 32,
        "critic_lr": 3e-4,
        "init_std": [0.3, 1.0, 0.5],        # [move (pre-tanh, review P1-10: <= 0.3), trigger, lift height]
        "std_floor": [0.15, 0.3, 0.2],
        "std_cap": [0.3, 2.0, 1.0],
        "std_reset": [0.3, 1.0, 0.5],       # raised to at least this at a curriculum promotion
        "seed": seed,
    }


def merge(cfg: dict, over: dict | None) -> dict:
    """Recursive dict update (nested dicts merged, other values replaced)."""
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            merge(cfg[k], v)
        else:
            cfg[k] = v
    return cfg


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def per_action(values, groups, device="cpu"):
    """Per-group values (a number, or one per group) -> per-action tensor [sum(groups)] (float32)."""
    if isinstance(values, (int, float)):
        return torch.full((sum(groups),), float(values), device=device)
    if len(values) != len(groups):
        raise ValueError(f"{values}: need one value per action group {groups}")
    return torch.cat([torch.full((n,), float(v), device=device) for v, n in zip(values, groups)])


# ---- policy -----------------------------------------------------------------------------------------------------
class RunningNorm(nn.Module):
    """Running mean / variance of observations (float64, Chan's parallel update); forward = (x - mean) / (std + eps)
    clipped to +-clip, float32."""

    def __init__(self, dim, eps=1e-2, clip=10.0):
        super().__init__()
        self.eps, self.clip = float(eps), float(clip)
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def update(self, x):
        x = x.detach().to(torch.float64).reshape(-1, self.mean.shape[0])
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        d = m - self.mean
        self.mean.add_(d * n / tot)
        self.var.copy_((self.var * self.count + v * n + d * d * self.count * n / tot) / tot)
        self.count.copy_(tot)

    def forward(self, x):
        y = (x.to(torch.float64) - self.mean) / (self.var.sqrt() + self.eps)
        return y.clamp(-self.clip, self.clip).to(torch.float32)


def mlp(n_in, hidden, n_out, act="elu"):
    layers, k = [], n_in
    for h in hidden:
        layers += [nn.Linear(k, h), ACT[act]()]
        k = h
    layers.append(nn.Linear(k, n_out))
    return nn.Sequential(*layers)


class Policy(nn.Module):
    """Observation normaliser + Gaussian actor (pre-tanh mean, per-action std parameter) + critic (module docstring)."""

    def __init__(self, obs_dim, act_dim=6, groups=(4, 1, 1), actor_hidden_dims=(512, 256, 128),
                 critic_hidden_dims=(512, 256, 128), activation="elu", init_std=(0.3, 1.0, 0.5), actor_out_gain=0.01,
                 action_bias=None, norm_clip=10.0):
        super().__init__()
        self.kwargs = dict(obs_dim=int(obs_dim), act_dim=int(act_dim), groups=[int(g) for g in groups],
                           actor_hidden_dims=list(actor_hidden_dims), critic_hidden_dims=list(critic_hidden_dims),
                           activation=activation, init_std=list(init_std) if not isinstance(init_std, (int, float)) else init_std,
                           actor_out_gain=float(actor_out_gain),
                           action_bias=None if action_bias is None else [float(x) for x in action_bias],
                           norm_clip=float(norm_clip))
        if sum(groups) != act_dim:
            raise ValueError(f"groups {groups} do not sum to {act_dim}")
        self.groups = list(groups)
        self.norm = RunningNorm(obs_dim, clip=norm_clip)
        self.actor = mlp(obs_dim, actor_hidden_dims, act_dim, activation)
        self.critic = mlp(obs_dim, critic_hidden_dims, 1, activation)
        self.std = nn.Parameter(per_action(init_std, self.groups))
        last = self.actor[-1]
        with torch.no_grad():
            last.weight.mul_(float(actor_out_gain))
            last.bias.zero_()
            if action_bias is not None:
                last.bias.copy_(torch.as_tensor(action_bias, dtype=last.bias.dtype))

    def value(self, obs_n):
        return self.critic(obs_n).squeeze(-1)

    @torch.no_grad()
    def act_inference(self, obs):
        """Raw observation [B, D] -> deterministic pre-tanh action [B, A] (the normaliser is not updated)."""
        return self.actor(self.norm(obs))

    def std_groups(self):
        out, k = [], 0
        for n in self.groups:
            out.append(float(self.std[k:k + n].detach().mean()))
            k += n
        return out


def log_prob(mu, std, a, mask):
    """Masked Gaussian log-likelihood [B] (sum over the used dims)."""
    lp = -0.5 * ((a - mu) / std) ** 2 - torch.log(std) - 0.5 * LOG2PI
    return (lp * mask).sum(-1)


def entropy(std, mask):
    """Masked Gaussian entropy [B]."""
    return ((0.5 + 0.5 * LOG2PI + torch.log(std)).expand_as(mask) * mask).sum(-1)


def kl_div(mu0, std0, mu1, std1, mask):
    """Masked KL(old || new) [B] (RSL-RL form)."""
    k = torch.log(std1 / std0 + 1e-5) + (std0 ** 2 + (mu0 - mu1) ** 2) / (2 * std1 ** 2) - 0.5
    return (k * mask).sum(-1)


def gae(rewards, values, dones, last_values, gamma, lam):
    """GAE over [T, B] (RSL-RL RolloutStorage.compute_returns): -> (returns, advantages), not normalised."""
    T = rewards.shape[0]
    adv = torch.zeros_like(last_values)
    out = torch.empty_like(rewards)
    for t in reversed(range(T)):
        nxt = last_values if t == T - 1 else values[t + 1]
        nd = 1.0 - dones[t].to(rewards.dtype)
        delta = rewards[t] + nd * gamma * nxt - values[t]
        adv = delta + nd * gamma * lam * adv
        out[t] = adv
    return out + values, out


def split_optimizer(policy: Policy, lr, critic_lr):
    """Adam: group 0 = actor + std (PPO's adaptive rate), group 1 = critic (fixed critic_lr)."""
    actor = list(policy.actor.parameters()) + [policy.std]
    return torch.optim.Adam([{"params": actor, "lr": lr}, {"params": list(policy.critic.parameters()), "lr": critic_lr}])


# ---- rollouts and updates -------------------------------------------------------------------------------------------
class Rollout:
    def __init__(self, T, B, obs_dim, act_dim, device):
        f = dict(device=device)
        self.T, self.B = T, B
        self.obs = torch.zeros(T, B, obs_dim, **f)
        self.act = torch.zeros(T, B, act_dim, **f)
        self.mask = torch.zeros(T, B, act_dim, dtype=torch.bool, **f)
        self.logp = torch.zeros(T, B, **f)
        self.mu = torch.zeros(T, B, act_dim, **f)
        self.values = torch.zeros(T, B, **f)
        self.rewards = torch.zeros(T, B, **f)
        self.dones = torch.zeros(T, B, dtype=torch.bool, **f)


class PPO:
    """PPO on a ``PickPlaceVecEnv`` (module docstring). cfg = ``train_cfg_default`` layout."""

    def __init__(self, env, policy: Policy, cfg: dict, device=None):
        self.env, self.policy, self.cfg = env, policy, cfg
        self.dev = torch.device(device or env.dev)
        al = cfg["algorithm"]
        self.al = al
        self.gamma, self.lam = float(al["gamma"]), float(al["lam"])
        self.lr = float(al["learning_rate"])
        self.critic_lr = float(cfg["critic_lr"])
        self.opt = split_optimizer(policy, self.lr, self.critic_lr)
        self.groups = policy.groups
        self.floor = per_action(cfg["std_floor"], self.groups, self.dev)
        self.cap = per_action(cfg["std_cap"], self.groups, self.dev)
        self.clamp_std()
        self.T = int(cfg["num_steps_per_env"])
        self.roll = Rollout(self.T, env.num_envs, env.num_obs, env.num_actions, self.dev)
        self.gen = torch.Generator(self.dev).manual_seed(int(cfg.get("seed", 1)) + 7)
        self.obs = None
        self.iteration = 0                # completed iterations
        self.decisions = 0                # PPO transitions (all envs)
        self.world_steps = 0              # 20 ms world steps (all envs)
        self.episodes = 0
        self.wall = 0.0                   # training wall clock (s), summed over resumed jobs

    @torch.no_grad()
    def clamp_std(self):
        self.policy.std.copy_(torch.minimum(torch.maximum(self.policy.std, self.floor), self.cap))

    @torch.no_grad()
    def set_std(self, values, at_least=False):
        v = per_action(values, self.groups, self.dev)
        self.policy.std.copy_(torch.maximum(self.policy.std, v) if at_least else v)
        self.clamp_std()

    # -- rollout --
    @torch.no_grad()
    def collect(self):
        """num_steps_per_env decisions of every env -> episode / step statistics (GPU tensors)."""
        env, pol, R = self.env, self.policy, self.roll
        if self.obs is None:
            self.obs = env.reset()
        obs = self.obs
        B, dev = env.num_envs, self.dev
        nE = len(ENDS)
        st = {k: torch.zeros((), dtype=torch.float64, device=dev) for k in
              ("episodes", "success", "success10", "lifted", "ret", "length", "end_xy_succ", "reward", "time_outs",
               "grasp_starts", "move_used", "trigger_used")}
        st["ends"] = torch.zeros(nE, dtype=torch.float64, device=dev)
        for t in range(self.T):
            pol.norm.update(obs)
            on = pol.norm(obs)
            mu = pol.actor(on)
            v = pol.value(on)
            std = pol.std.detach()
            a = mu + std * torch.randn(mu.shape, generator=self.gen, device=dev)
            obs2, r, done, ex = env.step(a)
            mask = ex["action_mask"]
            r = r.to(dev)
            to = ex["time_outs"]
            vf = pol.value(pol.norm(ex["final_obs"]))
            r = r + torch.where(to, self.gamma * vf, torch.zeros_like(vf))
            R.obs[t], R.act[t], R.mask[t], R.mu[t], R.values[t] = on, a, mask, mu, v
            R.logp[t] = log_prob(mu, std, a, mask)
            R.rewards[t], R.dones[t] = r, done
            ep = ex["episode"]
            d = done.to(torch.float64)
            st["episodes"] += d.sum()
            st["success"] += (ep["success"] & done).sum()
            st["success10"] += (ep["success10"] & done).sum()
            st["lifted"] += (ep["lifted"] & done).sum()
            st["ret"] += (ep["ret"] * d).sum()
            st["length"] += (ep["length"] * d).sum()
            st["end_xy_succ"] += torch.where(ep["success"] & done, ep["end_xy"], torch.zeros_like(ep["end_xy"])).sum()
            st["time_outs"] += to.sum()
            st["ends"] += (torch.nn.functional.one_hot(ep["end"].clamp(0, nE - 1), nE) * d[:, None]).sum(0)
            st["reward"] += r.sum()
            st["grasp_starts"] += mask[:, 5].sum()
            st["move_used"] += mask[:, 0].sum()
            st["trigger_used"] += mask[:, 4].sum()
            for k, v in ex.get("counts", {}).items():
                key = "count_" + k
                if key not in st:
                    st[key] = torch.zeros((), dtype=torch.float64, device=dev)
                st[key] += v.sum()
            obs = obs2
        self.obs = obs
        self.last_values = pol.value(pol.norm(obs))
        self.decisions += self.T * B
        self.world_steps += self.T * B * env.steps_per_decision
        return st

    # -- update --
    def update(self):
        al, pol, R = self.al, self.policy, self.roll
        with torch.no_grad():
            returns, adv = gae(R.rewards, R.values, R.dones, self.last_values, self.gamma, self.lam)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        N = R.T * R.B
        flat = lambda x: x.reshape(N, *x.shape[2:])                                  # noqa: E731
        obs, act, mask, old_lp, old_mu = flat(R.obs), flat(R.act), flat(R.mask).float(), flat(R.logp), flat(R.mu)
        old_v, ret, adv = flat(R.values), flat(returns), flat(adv)
        old_std = pol.std.detach().clone()
        clip = float(al["clip_param"])
        n_mb = int(al["num_mini_batches"])
        mb = N // n_mb
        out = dict(value_loss=0.0, surrogate_loss=0.0, entropy=0.0, kl=0.0, clip_frac=0.0)
        n_up = 0
        for _ in range(int(al["num_learning_epochs"])):
            perm = torch.randperm(N, generator=self.gen, device=self.dev)
            for k in range(n_mb):
                i = perm[k * mb:(k + 1) * mb]
                m = mask[i]
                mu = pol.actor(obs[i])
                v = pol.value(obs[i])
                std = pol.std
                lp = log_prob(mu, std, act[i], m)
                with torch.no_grad():
                    kl = kl_div(old_mu[i], old_std, mu, std, m)
                    act_rows = (m.sum(-1) > 0).float()
                    kl_mean = float((kl * act_rows).sum() / act_rows.sum().clamp_min(1.0))
                if al.get("desired_kl") is not None and al.get("schedule") == "adaptive":
                    if kl_mean > al["desired_kl"] * 2.0:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif 0.0 < kl_mean < al["desired_kl"] / 2.0:
                        self.lr = min(1e-2, self.lr * 1.5)
                    self.opt.param_groups[0]["lr"] = self.lr
                    self.opt.param_groups[1]["lr"] = self.critic_lr
                ratio = torch.exp(lp - old_lp[i])
                a_ = adv[i]
                surr = torch.max(-a_ * ratio, -a_ * ratio.clamp(1 - clip, 1 + clip)).mean()
                if al.get("use_clipped_value_loss", True):
                    vc = old_v[i] + (v - old_v[i]).clamp(-clip, clip)
                    vloss = torch.max((v - ret[i]) ** 2, (vc - ret[i]) ** 2).mean()
                else:
                    vloss = ((v - ret[i]) ** 2).mean()
                ent = entropy(std, m).mean()
                loss = surr + float(al["value_loss_coef"]) * vloss - float(al["entropy_coef"]) * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(pol.parameters(), float(al["max_grad_norm"]))
                self.opt.step()
                self.clamp_std()
                with torch.no_grad():
                    out["clip_frac"] += float(((ratio - 1).abs() > clip).float().mean())
                out["value_loss"] += float(vloss.detach())
                out["surrogate_loss"] += float(surr.detach())
                out["entropy"] += float(ent.detach())
                out["kl"] += kl_mean
                n_up += 1
        return {k: v / max(n_up, 1) for k, v in out.items()}

    def iterate(self):
        """One collect + update -> stats dict (host floats)."""
        t0 = time.time()
        st = self.collect()
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)
        t1 = time.time()
        up = self.update()
        t2 = time.time()
        self.iteration += 1
        self.wall += t2 - t0
        s = {k: v.cpu().numpy() for k, v in st.items()}
        n = float(s["episodes"])
        B, T, spd = self.env.num_envs, self.T, self.env.steps_per_decision
        self.episodes += int(n)
        row = dict(iteration=self.iteration, decisions=self.decisions, world_steps=self.world_steps,
                   wall_s=self.wall, collection_s=t1 - t0, learn_s=t2 - t1,
                   world_steps_per_s=B * T * spd / max(t1 - t0, 1e-9),
                   world_steps_per_s_total=B * T * spd / max(t2 - t0, 1e-9),
                   reward_per_decision=float(s["reward"]) / (B * T),
                   episodes=n, success=float(s["success"]) / n if n else math.nan,
                   success10=float(s["success10"]) / n if n else math.nan,
                   lifted=float(s["lifted"]) / n if n else math.nan,
                   ep_return=float(s["ret"]) / n if n else math.nan,
                   ep_length=float(s["length"]) / n if n else math.nan,
                   end_xy_success_mm=1e3 * float(s["end_xy_succ"]) / max(float(s["success"]), 1.0),
                   grasp_starts_per_episode=float(s["grasp_starts"]) / n if n else math.nan,
                   move_used=float(s["move_used"]) / (B * T), trigger_used=float(s["trigger_used"]) / (B * T),
                   lr=self.lr, critic_lr=self.critic_lr, **up)
        for k in s:
            if k.startswith("count_"):
                row[k[6:] + "_per_episode"] = float(s[k]) / n if n else math.nan
        for j, g in enumerate(self.policy.std_groups()):
            row[f"std_{('move', 'trigger', 'lift')[j] if j < 3 else j}"] = g
        for k, name in enumerate(ENDS[1:], start=1):
            row[f"end_{name}"] = float(s["ends"][k]) / n if n else math.nan
        if self.dev.type == "cuda":
            row["gpu_mem_gb"] = torch.cuda.max_memory_allocated(self.dev) / 1e9     # torch allocator peak
            free, total = torch.cuda.mem_get_info(self.dev)
            row["gpu_used_gb"] = (total - free) / 1e9      # whole device (other processes and Warp's pool included)
        return row

    # -- checkpoints --
    def state(self, extra=None) -> dict:
        return dict(policy=self.policy.state_dict(), policy_kwargs=self.policy.kwargs, optimizer=self.opt.state_dict(),
                    lr=self.lr, iteration=self.iteration, decisions=self.decisions, world_steps=self.world_steps,
                    episodes=self.episodes, wall=self.wall, gen=self.gen.get_state(),
                    env_gen=self.env.gen.get_state(), extra=extra or {})

    def save(self, path, extra=None):
        tmp = Path(str(path) + ".tmp")
        torch.save(self.state(extra), tmp)
        tmp.replace(path)

    def load(self, path, optimizer=True):
        d = torch.load(path, map_location=self.dev, weights_only=False)
        self.policy.load_state_dict(d["policy"])
        if optimizer:
            self.opt.load_state_dict(d["optimizer"])
            self.lr = float(d["lr"])
            self.iteration, self.decisions, self.world_steps = d["iteration"], d["decisions"], d["world_steps"]
            self.episodes, self.wall = d.get("episodes", 0), d.get("wall", 0.0)
            self.gen.set_state(d["gen"].cpu())
            if "env_gen" in d:
                self.env.gen.set_state(d["env_gen"].cpu())
        return d


def load_policy(path, device="cpu") -> Policy:
    """A saved checkpoint (``PPO.save``) -> Policy in eval mode on device."""
    d = torch.load(path, map_location=device, weights_only=False)
    pol = Policy(**d["policy_kwargs"]).to(device)
    pol.load_state_dict(d["policy"])
    return pol.eval()


class CurveWriter:
    """learning_curve.csv: one row per iteration; a resumed run appends under the existing header (new keys are
    dropped, missing ones left empty)."""

    def __init__(self, path):
        self.path = Path(path)
        self.header = None
        if self.path.exists() and self.path.stat().st_size:
            with open(self.path, newline="") as f:
                self.header = next(csv.reader(f))

    def write(self, row: dict):
        new = self.header is None
        if new:
            self.header = list(row)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.header, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})


def read_curve(path) -> dict:
    """learning_curve.csv -> {column: numpy array}."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in (rows[0] if rows else {}):
        try:
            out[k] = np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])
        except ValueError:
            out[k] = np.array([r[k] for r in rows])
    return out
