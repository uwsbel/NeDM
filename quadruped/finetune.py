#!/usr/bin/env python3
"""Fine-tune the policy inside the NN-ROM.

Two methods, one interface, one objective, one stopping rule -- so the comparison
between them is about the METHOD and not about how each was tuned.

`--method analytic` backpropagates through the frozen model to the policy weights.
`--method ppo` does not: it treats the model as an ordinary environment, rolls it under
no_grad, and learns from reward alone. That difference is the point. Analytic leans on
the plant being differentiable, which a general method should not need, and the NN-ROM
happens to be differentiable only because it is a neural network -- Chrono is not.

WHAT ANALYTIC POLICY GRADIENT ACTUALLY DOES, since the name misleads

The NN-ROM is frozen and differentiable. A branch starts from a state RECORDED IN THE
CORPUS, rolls the policy forward inside the model for K steps, scores the result against
the command, and backpropagates through the whole rollout to the POLICY WEIGHTS. The
gradient is d(loss)/d(theta). It is not a gradient with respect to the velocity command,
which is the usual first guess.

The consequence worth stating plainly: branch starts are drawn from recorded states, so
the method never explores outside the corpus. It is closer to local improvement over a
dataset than to reinforcement learning, and that is exactly why it transfers only as far
as the corpus covers.

THE FAILURE MODE THIS IS BUILT AROUND

In-model gain anti-correlates with transfer. A policy that scores brilliantly inside the
NN-ROM has usually found somewhere the model is wrong, not somewhere the robot is fast --
the optimiser's curse, and on this project it has bitten more than once. Two consequences
are wired in rather than left to discipline:

  - Training stops on WEIGHT DISPLACEMENT, ||theta - theta_0||, not on in-model reward.
    A budget in weight space cannot be gamed by a model error the way a reward can.
  - In-model gain is reported but never used to select. The only number that decides
    anything is measured in Chrono by evaluate.py, against the base policy, on the same
    machine, over replicates.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))


# --------------------------------------------------------------- observation

class ObsBuilder:
    """Build the policy's 45-vector from an NN-ROM state, differentiably.

    Every term the policy reads must be a function of the propagated state, or the policy
    cannot be rolled inside the model at all. That is why the 36-D preset carries
    `grav_body_*`: the previous study found the 34-D state could not express the base
    height term the policy's dominant reward uses, and the first fine-tune fell in 43 of
    43 Chrono episodes at a median of 1.52 s.

    The layout is rl_sar's, in rl_sar's joint order, with rl_sar's scales -- the same
    construction as `lib/policy.py`, rebuilt here in torch so it carries gradient. If the
    two ever disagree the fine-tune optimises against an observation the robot will not
    see, so the column indices come from the preset rather than being restated.
    """

    def __init__(self, torch, state_fields, pol_cfg):
        self.torch = torch
        ix = {f: i for i, f in enumerate(state_fields)}
        need = ["roll_rate_radps", "ang_vel_body_y_radps", "yaw_rate_radps",
                "grav_body_x", "grav_body_y", "grav_body_z"]
        missing = [n for n in need if n not in ix]
        if missing:
            raise SystemExit(
                f"the policy observation needs {missing}, absent from this preset. A "
                f"state the policy cannot be rebuilt from cannot be fine-tuned in.")
        self.i_ang = [ix[n] for n in need[:3]]
        self.i_grav = [ix[n] for n in need[3:]]

        legs = ("rr", "rl", "fr", "fl")
        jp = [f"joint_{l}_{s}_pos_rad" for l in legs for s in ("hip", "thigh", "calf")]
        jv = [f"joint_{l}_{s}_vel_radps" for l in legs for s in ("hip", "thigh", "calf")]
        for n in jp + jv:
            if n not in ix:
                raise SystemExit(f"preset lacks {n!r}, needed for the policy observation")
        self.i_jp = [ix[n] for n in jp]      # CHRONO order
        self.i_jv = [ix[n] for n in jv]

        sc = pol_cfg["observation"]["scales"]
        self.s_ang = float(sc["ang_vel"])
        self.s_qpos = float(sc["dof_pos"])
        self.s_qvel = float(sc["dof_vel"])
        self.s_cmd = torch.tensor(sc["commands"], dtype=torch.float32)
        self.clip = float(pol_cfg["observation"].get("clip", 100.0))
        self.p2c = list(pol_cfg["joints"]["policy_to_chrono"])
        self.sign = float(pol_cfg["sign"]["value"])
        self.default = torch.tensor(pol_cfg["joints"]["default_pos"], dtype=torch.float32)
        self.act_scale = torch.tensor(pol_cfg["action"]["scale"], dtype=torch.float32)
        self.act_clip = tuple(pol_cfg["action"]["clip"])

    def to(self, dev):
        for a in ("s_cmd", "default", "act_scale"):
            setattr(self, a, getattr(self, a).to(dev))
        return self

    def observe(self, state, cmd, last_action):
        t = self.torch
        ang = state[..., self.i_ang] * self.s_ang
        grav = state[..., self.i_grav]
        q_c = state[..., self.i_jp]          # chrono order
        qd_c = state[..., self.i_jv]
        # chrono -> policy order, then the global sign, matching lib/policy.py exactly.
        q_p = self.sign * q_c[..., self.p2c]
        qd_p = self.sign * qd_c[..., self.p2c]
        dof_pos = (q_p - self.default) * self.s_qpos
        dof_vel = qd_p * self.s_qvel
        obs = t.cat([ang, grav, cmd * self.s_cmd, dof_pos, dof_vel, last_action], dim=-1)
        return t.clamp(obs, -self.clip, self.clip)

    def action_from_raw(self, raw):
        """Policy output -> the 12 joint targets in CHRONO order, as the model expects."""
        t = self.torch
        raw = t.clamp(raw, *self.act_clip)
        targets_p = self.default + raw * self.act_scale
        # p2c is self-inverse on this robot, but invert explicitly rather than rely on it.
        inv = [0] * len(self.p2c)
        for pi, ci in enumerate(self.p2c):
            inv[ci] = pi
        return self.sign * targets_p[..., inv]


# --------------------------------------------------------------------- pieces

def load_nnrom(torch, path, dev, allow_smoke=False):
    ck = torch.load(path, map_location=dev, weights_only=False)
    if ck.get("smoke") and not allow_smoke:
        raise SystemExit(
            f"{path} is stamped smoke=True. It was trained with the selection guards "
            f"relaxed on a corpus too small to select on, so it was never selected on a "
            f"usable rollout metric and must not be fine-tuned in.")
    if ck.get("worse_than_no_motion") and not allow_smoke:
        raise SystemExit(
            f"{path} is stamped worse_than_no_motion=True. Its best smoothed rollout "
            f"errdist was at or above 1.0, which is what a model that predicts the robot "
            f"does not move scores. Fine-tuning a policy inside a plant model that is "
            f"worse than assuming nothing happens cannot produce a transferable result, "
            f"however good the in-model reward gets. The usual cause is too little data.")
    if ck.get("selection_no_trend") and not allow_smoke:
        raise SystemExit(
            f"{path} is stamped selection_no_trend=True: the rollout metric never "
            f"meaningfully improved during training, so its best epoch is the luckiest "
            f"rather than the most trained. Fine-tuning inside it would measure that "
            f"luck. The usual cause is an undersized corpus -- compare val_loss against "
            f"train_loss in metrics.jsonl before adding epochs.")
    if ck.get("selection_lottery") and not allow_smoke:
        raise SystemExit(
            f"{path} is stamped selection_lottery=True: during training the rollout "
            f"metric moved as far between adjacent epochs as it did across the whole run "
            f"({ck.get('selection_lag1'):.4f} against {ck.get('selection_range'):.4f}), "
            f"so which epoch became 'best' was close to arbitrary.\n"
            f"  Fine-tuning inside it would attribute to the method whatever that "
            f"arbitrary draw happened to be. Collect more long held-out segments, or "
            f"raise --select-window, and retrain.")
    sys.path.insert(0, str(HERE))
    import train as T  # noqa: PLC0415
    cfg = ck["config"]
    stats = {k: np.asarray(v) for k, v in ck["stats"].items()}
    model = T.build_model(torch, torch.nn, len(ck["state_fields"]),
                          len(ck["action_fields"]),
                          {"block_size": cfg["block_size"], "n_layer": cfg["n_layer"],
                           "n_head": cfg["n_head"], "n_embd": cfg["n_embd"],
                           "dropout": 0.0}, stats).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():      # FROZEN. The plant is not what is being fitted.
        p.requires_grad_(False)
    return model, ck


def load_policy_torch(torch, path, dev):
    net = torch.jit.load(str(path), map_location=dev)
    net.train()
    params = [p for p in net.parameters() if p.requires_grad]
    if not params:
        raise SystemExit(
            f"{path} exposes no trainable parameters. A TorchScript module frozen at "
            f"export has none, and there is nothing to fine-tune.")
    return net, params


# Columns the fine-tune reads beside the model's inputs. policy_raw is in CHRONO order in
# a corpus stamped policy_raw_order="chrono"; ObsBuilder wants the network's own order.
RAW_FIELDS = [f"policy_raw_{l}_{s}" for l in ("rr", "rl", "fr", "fl")
              for s in ("hip", "thigh", "calf")]
CMD_FIELDS = ["cmd_vx_mps", "cmd_vy_mps", "cmd_wz_radps"]
EXTRA_FIELDS = ["time_s"] + CMD_FIELDS + RAW_FIELDS
_T, _CMD, _RAW = 0, slice(1, 4), slice(4, 16)


def build_pool(corpus, ctx, ctrl_dt):
    """Every window of `ctx` real rows whose LAST row is a control instant.

    RECORDED STATES, not sampled ones. This is the property that makes the method local:
    the policy is improved on states the corpus actually contains, so coverage of the
    corpus bounds what can be learned. Drawn from the TRAIN split, since the val split is
    what any honest in-model number would have to be read on.

    WHY THE LAST ROW MUST BE A CONTROL ROW. Rows are 100 Hz and the policy acts at 50 Hz.
    The branch's first act is the policy choosing an action at the window's last row, so
    that row has to be one where the real policy chose one too -- otherwise the branch
    starts half a control period out of phase with every recorded action before it, and
    the recorded last action it observes is not the one the policy would have seen.
    """
    pool = []
    for si, r in enumerate(corpus.train):
        ph = r["extra"][:, _T] / ctrl_dt
        ctl = np.abs(ph - np.round(ph)) < 1e-6
        for k in range(max(r["n"] - ctx - 1, 0)):
            if ctl[k + ctx - 1]:
                pool.append((si, k))
    if not pool:
        raise SystemExit("no segment is long enough to start a branch from on a control row")
    return pool


def start_batch(torch, corpus, pool, n, rng, ctx, p2c, dev):
    """Draw `n` branch starts and everything the rollout needs at each.

    states   (B, ctx, S)   the recorded window
    acts     (B, ctx-1, A) the recorded actions for every row BUT the last. The last row's
                           action is the one the branch chooses, so it is left open; the
                           model pairs state row j with action row j, and filling the last
                           slot with the recorded action and then appending the policy's
                           would shift the whole history by one row.
    last_raw (B, 12)       the network output held at the start row, i.e. the one the
                           policy observed as its last action there (policy order)
    raw_rec  (B, 12)       what the network actually output at the start row -- used once,
                           to prove the observation rebuild reproduces it
    cmd      (B, 3)        the command the robot was following at the start row
    """
    picks = [pool[rng.randrange(len(pool))] for _ in range(n)]
    S = np.stack([corpus.train[si]["state"][k:k + ctx] for si, k in picks])
    A = np.stack([corpus.train[si]["action"][k:k + ctx - 1] for si, k in picks])
    E = np.stack([corpus.train[si]["extra"][k + ctx - 2:k + ctx] for si, k in picks])
    f = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)  # noqa: E731
    return {"picks": picks, "states": f(S), "acts": f(A),
            "last_raw": f(E[:, 0, _RAW][:, p2c]), "raw_rec": f(E[:, 1, _RAW][:, p2c]),
            "cmd": f(E[:, 1, _CMD])}


def check_start_reproduction(torch, obs, policy, b):
    """The rebuilt observation must reproduce the policy's recorded output. Refuse if not.

    This is the test the first fine-tunes never ran, and the reason they were meaningless.
    The corpus was captured one physics step late, so the recorded state at a control row
    already carried the PD kick from the action being chosen there; the rebuilt
    observation then moved the policy's output by 36-60% of its spread before a single
    weight changed, and the fine-tune optimised a policy for inputs it never receives.
    On a corpus captured before the step, from the policy being fine-tuned, this agrees to
    float rounding -- so anything above 1% means the rollout is not rolling this policy.
    """
    with torch.no_grad():
        o = obs.observe(b["states"][:, -1], b["cmd"], b["last_raw"])
        raw = torch.clamp(policy(o), *obs.act_clip)
    rms = float(torch.sqrt(((raw - b["raw_rec"]) ** 2).mean()))
    spread = float(b["raw_rec"].std())
    print(f"start check: the base policy on the rebuilt observation reproduces its recorded "
          f"output to RMS {rms:.2e} against a spread of {spread:.3f} "
          f"({100 * rms / max(spread, 1e-12):.3f}%)")
    if rms > 0.01 * spread:
        raise SystemExit(
            f"the observation rebuilt from corpus rows does not reproduce what the policy "
            f"actually output ({100 * rms / spread:.1f}% of its spread). Fine-tuning would "
            f"optimise the policy for an input it never sees in Chrono. Check that the "
            f"corpus was collected by THIS policy with row_capture=pre_step, and that "
            f"ObsBuilder matches lib/policy.py (obs_truth.py tests exactly that).")
    return rms / max(spread, 1e-12)


# ------------------------------------------------------------------------ ppo

def make_stochastic(torch, nn, policy, act_dim, init_log_std):
    """A Gaussian head on the deterministic actor, plus a fresh critic.

    The base policy is a deterministic TorchScript MLP: it maps an observation to an
    action, with no notion of a distribution. PPO needs a stochastic policy to have a
    likelihood ratio at all, so the mean comes from the existing network -- keeping
    everything the base policy already knows -- and a learnable log_std is added beside
    it. Starting it small means the first rollouts stay near the base behaviour rather
    than flailing, which matters because a random-looking policy on CRM falls over and
    then the rollouts are all about falling over.

    The critic is new. There is nothing to inherit: the base policy was trained elsewhere
    with its own value function, which was not exported and would be wrong for this
    reward anyway.
    """
    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = policy
            self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

        def forward(self, obs):
            return self.net(obs)

        def dist(self, obs):
            mu = self.net(obs)
            return torch.distributions.Normal(mu, self.log_std.exp())

    class Critic(nn.Module):
        def __init__(self, obs_dim):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(obs_dim, 256), nn.ELU(),
                                   nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))

        def forward(self, obs):
            return self.f(obs).squeeze(-1)

    return Actor(), Critic


def step_reward(torch, nxt, cmd, ix, upright_weight):
    """The SAME objective the analytic path minimises, as a reward.

    Comparing two optimisers is only meaningful if they are pointed at one target. This
    is the negative of the analytic loss term for term, so a difference in outcome is a
    difference in method rather than in what each was asked to do.
    """
    v = torch.stack([nxt[:, ix["vel_body_x_mps"]], nxt[:, ix["vel_body_y_mps"]],
                     nxt[:, ix["yaw_rate_radps"]]], dim=-1)
    track = ((v - cmd) ** 2).sum(-1)
    upright = (nxt[:, ix["grav_body_z"]] + 1.0) ** 2
    return -(track + upright_weight * upright)


class OODCost:
    """Penalise the reward when a rollout leaves the region the corpus covers.

    THIS IS THE DIFFERENCE BETWEEN PPO WORKING AND PPO CHEATING, and PPO needs it more
    than the analytic path does.

    The NN-ROM is only a model of the robot where the corpus taught it one. PPO explores
    by sampling actions, so it will find the places the model is wrong faster than any
    method that stays near recorded behaviour -- and a place where the model is wrong
    usually looks like free reward. That is the optimiser's curse this project has been
    bitten by repeatedly: in-model gain anti-correlates with transfer.

    So distance from the corpus is priced into the reward, using the same whitened kNN
    reference that Gate 4 uses to decide whether a corpus covers its own held-out data.
    Beyond the corpus's own 99th-percentile self-distance -- its own notion of "far" --
    every further unit costs `weight`.

    Disabling this does not make the numbers better, it makes them less true.
    """

    def __init__(self, corpus, max_points=4000, seed=0):
        sys.path.insert(0, str(REPO))
        from quadruped.lib import coverage as C  # noqa: PLC0415
        X = np.concatenate([np.concatenate([r["state"][:-1], r["action"][:-1]], axis=1)
                            for r in corpus.train], axis=0)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=min(max_points * 4, len(X)), replace=False)
        self.ref = C.Reference(X[idx], names=corpus.state_fields + corpus.action_fields,
                               rng=rng)
        self.mu = self.ref.mu
        self.sd = self.ref.sd
        self.thresh = self.ref.thresh

    def to_torch(self, torch, dev):
        self.t_mu = torch.tensor(self.mu, dtype=torch.float32, device=dev)
        self.t_sd = torch.tensor(self.sd, dtype=torch.float32, device=dev)
        self.t_ref = torch.tensor(self.ref.ref, dtype=torch.float32, device=dev)
        self.k = self.ref.k
        return self

    def cost(self, torch, state, action):
        """Mean distance to the k nearest corpus points, beyond the threshold."""
        z = (torch.cat([state, action], dim=-1) - self.t_mu) / self.t_sd
        d = torch.cdist(z, self.t_ref)
        knn = d.topk(self.k, dim=-1, largest=False).values.mean(-1)
        return torch.clamp(knn - self.thresh, min=0.0)


def ppo_rollout(torch, model, obs_b, actor, critic, b, steps, hold, cmd, ix,
                upright_weight, ood=None, ood_weight=0.0):
    """Collect one batch of trajectories inside the model. NO GRADIENT THROUGH DYNAMICS.

    This is the whole difference from the analytic path. The model is stepped under
    no_grad and only its OUTPUTS are used, so nothing here requires the plant to be
    differentiable. Swap the NN-ROM for Chrono and this function still works, which is
    the property the analytic method does not have.

    One PPO transition is one CONTROL step: the policy samples, the action is held for
    `hold` model steps (the model steps at the record rate, the policy acts at half it),
    and the reward is the mean over those steps.
    """
    hist_s, hist_a, last_raw = b["states"], b["acts"], b["last_raw"]
    obs_buf, act_buf, logp_buf, rew_buf, val_buf = [], [], [], [], []
    ood_buf = []
    with torch.no_grad():
        for _ in range(steps):
            o = obs_b.observe(hist_s[:, -1], cmd, last_raw)
            d = actor.dist(o)
            raw = d.sample()
            logp = d.log_prob(raw).sum(-1)
            val = critic(o)
            last_raw = torch.clamp(raw, *obs_b.act_clip)
            act = obs_b.action_from_raw(last_raw)
            r = 0.0
            for _h in range(hold):
                a_win = torch.cat([hist_a, act[:, None]], dim=1)
                nxt = hist_s[:, -1] + model.predict_delta(hist_s, a_win)[:, -1]
                hist_s = torch.cat([hist_s[:, 1:], nxt[:, None]], dim=1)
                hist_a = a_win[:, 1:]
                rh = step_reward(torch, nxt, cmd, ix, upright_weight)
                if ood is not None and ood_weight > 0:
                    c = ood.cost(torch, nxt, act)
                    rh = rh - ood_weight * c
                    ood_buf.append(c.mean())
                r = r + rh / hold
            obs_buf.append(o); act_buf.append(raw); logp_buf.append(logp)
            val_buf.append(val); rew_buf.append(r)
        last_val = critic(obs_b.observe(hist_s[:, -1], cmd, last_raw))
    ood_mean = float(torch.stack(ood_buf).mean()) if ood_buf else 0.0
    return (torch.stack(obs_buf), torch.stack(act_buf), torch.stack(logp_buf),
            torch.stack(rew_buf), torch.stack(val_buf), last_val, ood_mean)


def gae(torch, rew, val, last_val, gamma, lam):
    """Generalised advantage estimation over a fixed-length, never-terminating rollout.

    There is no done flag: a branch runs `steps` and stops because the budget ran out,
    not because anything ended. Bootstrapping off the critic at the tail is therefore the
    correct treatment, and inserting a terminal would tell the agent the world ends when
    it does not.
    """
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    nxt_val = last_val
    run = torch.zeros_like(last_val)
    for t in reversed(range(T)):
        delta = rew[t] + gamma * nxt_val - val[t]
        run = delta + gamma * lam * run
        adv[t] = run
        nxt_val = val[t]
    return adv, adv + val


# ------------------------------------------------------------------ the loss

def branch_loss(torch, model, obs, policy, b, cmd, steps, hold, ix):
    """Roll the policy inside the frozen model for `steps` control steps, score against the
    command.

    The score is command tracking in the body frame, which is what the study reports and
    what the base policy is weakest at. Height and attitude are held with a light penalty
    rather than optimised: without them the optimiser discovers that lying down tracks
    zero velocity beautifully, and a fine-tune that falls over is not an improvement
    however good its number is.

    THE LOOP HAS THE ROBOT'S TIMING. The model steps at the record rate (0.01 s) and the
    policy acts at the control rate (0.02 s), so each action is held for `hold` model
    steps and the last-action observation changes only when the policy acts. The first
    version called the policy on every model step: it ran at 100 Hz inside the model,
    fed the model an action stream that changed every row where every recorded one is
    held for two, and labelled 15 model steps "0.30 s" when they were 0.15 s.
    """
    hist_s, hist_a, last_raw = b["states"], b["acts"], b["last_raw"]
    track, upright = [], []
    for _ in range(steps):
        o = obs.observe(hist_s[:, -1], cmd, last_raw)
        last_raw = torch.clamp(policy(o), *obs.act_clip)
        act = obs.action_from_raw(last_raw)
        for _h in range(hold):
            a_win = torch.cat([hist_a, act[:, None]], dim=1)
            nxt = hist_s[:, -1] + model.predict_delta(hist_s, a_win)[:, -1]
            hist_s = torch.cat([hist_s[:, 1:], nxt[:, None]], dim=1)
            hist_a = a_win[:, 1:]
            v = torch.stack([nxt[:, ix["vel_body_x_mps"]], nxt[:, ix["vel_body_y_mps"]],
                             nxt[:, ix["yaw_rate_radps"]]], dim=-1)
            track.append(((v - cmd) ** 2).sum(-1))
            # grav_body_z is -1 when level; anything above that is the trunk pitching over.
            upright.append((nxt[:, ix["grav_body_z"]] + 1.0) ** 2)
    return torch.stack(track).mean(), torch.stack(upright).mean()


def branch_cmd(torch, a, b, ranges, rng, dev):
    """The command each branch tracks.

    `corpus` (default): the command the robot was following at the start row, held for the
    branch. The start state was produced under that command, so the objective is to track
    it BETTER from where the robot actually was.

    `random`: a fresh draw from the collection ranges. Over a 0.30 s branch that mostly
    scores how fast the robot can change speed toward an unrelated command, which is a
    transient the base policy was never asked to win and the model saw little of.
    """
    if a.command == "corpus":
        return b["cmd"]
    return torch.tensor(
        [[rng.uniform(*ranges["vx"]), rng.uniform(*ranges["vy"]),
          rng.uniform(*ranges["wz"])] for _ in range(a.branches)],
        dtype=torch.float32, device=dev)


def run_ppo(torch, nn, a, model, obs_b, policy, params, baseline, base_norm, n_par,
            corpus, ctx, ix, ranges, rng, dev, log, pool, p2c, hold):
    """Clipped-surrogate PPO with the model as an ordinary environment.

    Stops on the SAME weight-displacement budget as the analytic path. That is deliberate
    and it is the only way the comparison means anything: matching iterations would
    compare two optimisers that moved the policy different distances, and matching
    in-model reward would compare how well each gamed the model.
    """
    obs_dim = obs_b.observe(torch.zeros(1, len(corpus.state_fields), device=dev),
                            torch.zeros(1, 3, device=dev),
                            torch.zeros(1, 12, device=dev)).shape[-1]
    actor, Critic = make_stochastic(torch, nn, policy, 12, a.init_log_std)
    actor = actor.to(dev)
    critic = Critic(obs_dim).to(dev)
    # The dw budget is measured on the ACTOR's inherited weights only. log_std is new and
    # the critic is new, so counting them would let the budget be spent on parameters the
    # base policy never had.
    opt = torch.optim.Adam(
        [{"params": params, "lr": a.lr},
         {"params": [actor.log_std], "lr": a.lr},
         {"params": critic.parameters(), "lr": a.critic_lr}])

    ood = None
    if a.ood_penalty > 0:
        ood = OODCost(corpus, seed=a.seed).to_torch(torch, dev)
        print(f"  OOD penalty {a.ood_penalty} beyond corpus self-distance "
              f"{ood.thresh:.3f} (the same kNN reference Gate 4 uses)")
    print(f"ppo: {a.branches} branches x {a.steps} control steps ({a.steps * hold} model "
          f"steps), clip {a.clip_eps}, {a.ppo_epochs} epochs x {a.minibatches} minibatches")
    dw, it = 0.0, 0
    for it in range(1, a.iters + 1):
        b = start_batch(torch, corpus, pool, a.branches, rng, ctx, p2c, dev)
        cmd = branch_cmd(torch, a, b, ranges, rng, dev)
        ob, ac, lp, rw, vl, last_val, ood_mean = ppo_rollout(
            torch, model, obs_b, actor, critic, b, a.steps, hold, cmd, ix,
            a.upright_weight, ood, a.ood_penalty)
        adv, ret = gae(torch, rw, vl, last_val, a.gamma, a.lam)
        # Flatten time and branch: every (t, b) is one independent sample here, since the
        # branches do not interact.
        ob, ac, lp = ob.reshape(-1, ob.shape[-1]), ac.reshape(-1, 12), lp.reshape(-1)
        adv, ret = adv.reshape(-1), ret.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = ob.shape[0]
        idx = torch.randperm(n, device=dev)
        mb = max(n // a.minibatches, 1)
        pl = vf = ent = 0.0
        for _ in range(a.ppo_epochs):
            for st in range(0, n, mb):
                j = idx[st:st + mb]
                d = actor.dist(ob[j])
                new_lp = d.log_prob(ac[j]).sum(-1)
                ratio = (new_lp - lp[j]).exp()
                un = ratio * adv[j]
                cl = torch.clamp(ratio, 1 - a.clip_eps, 1 + a.clip_eps) * adv[j]
                p_loss = -torch.min(un, cl).mean()
                v_loss = ((critic(ob[j]) - ret[j]) ** 2).mean()
                e = d.entropy().sum(-1).mean()
                loss = p_loss + a.vf_coef * v_loss - a.ent_coef * e
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(params) + [actor.log_std] + list(critic.parameters()), 1.0)
                opt.step()
                pl, vf, ent = p_loss.item(), v_loss.item(), e.item()

        with torch.no_grad():
            dw = float(weight_displacement(torch, params, baseline))
        rec = {"iter": it, "reward": rw.mean().item(), "policy_loss": pl,
               "value_loss": vf, "entropy": ent, "dw": dw, "dw_rel": dw / base_norm,
               "log_std": actor.log_std.mean().item(), "ood": ood_mean}
        log.write(json.dumps(rec) + "\n")
        if it % 10 == 0 or it == 1:
            print(f"  iter {it:5d}  reward {rw.mean().item():+.4f}  vloss {vf:.4f}  "
                  f"ent {ent:+.3f}  log_std {actor.log_std.mean().item():+.2f}  "
                  f"dw {dw:.4f} ({100 * dw / base_norm:.2f}%)  ood {ood_mean:.4f}",
                  flush=True)
        if dw >= a.target_dw:
            print(f"  stopping: dw {dw:.4f} reached the {a.target_dw} budget at iter {it}")
            break
    return it, dw



def weight_displacement(torch, params, baseline):
    return torch.sqrt(sum(((p - b) ** 2).sum() for p, b in zip(params, baseline)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a trained NN-ROM checkpoint")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--corpus", required=True, help="for branch start states")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["analytic", "ppo"], default="analytic")
    ap.add_argument("--branches", type=int, default=64)
    ap.add_argument("--steps", type=int, default=15,
                    help="branch length in CONTROL steps; 15 is 0.30 s at 50 Hz. The model "
                         "steps twice per control step, at the 100 Hz record rate.")
    ap.add_argument("--command", choices=["corpus", "random"], default="corpus",
                    help="what each branch tracks; see branch_cmd()")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--target-dw", type=float, default=4.0,
                    help="stop at this ||theta - theta_0||; the budget is in weight "
                         "space because in-model reward can be gamed and this cannot. "
                         "Adam normalises per parameter, so one step moves the vector by "
                         "about lr*sqrt(N): 0.043 here, and dw 4.0 is therefore roughly "
                         "100 steps. A budget of 0.05 stops after ONE.")
    ap.add_argument("--upright-weight", type=float, default=0.5)
    # PPO only. Defaults are the standard continuous-control set; the one choice specific
    # to this problem is init_log_std, kept small so early rollouts stay near the base
    # policy -- a policy that flails on CRM falls over, and then every rollout is about
    # falling over rather than about tracking.
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--ppo-epochs", type=int, default=10)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--critic-lr", type=float, default=1e-3)
    ap.add_argument("--init-log-std", type=float, default=-2.5)
    ap.add_argument("--ood-penalty", type=float, default=1.0,
                    help="reward penalty per unit of normalised kNN distance beyond the "
                         "corpus threshold. 0 disables it, which is not recommended: see "
                         "the note in ood_cost().")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true",
                    help="permit a smoke-stamped NN-ROM so this code path can be "
                         "exercised before a real corpus exists. Stamps the output.")
    a = ap.parse_args()

    import torch
    sys.path.insert(0, str(HERE))
    import train as T

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available())
                       else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model, ck = load_nnrom(torch, a.model, dev, allow_smoke=a.smoke)

    # IS THE MODEL ACCURATE OVER THE HORIZON THIS RUN WILL ROLL IT? Checked here, against
    # this run's own branch length, because a model is not good or bad in general -- it is
    # good out to some horizon. The 550-episode NN-ROM beats "the robot does not move" at
    # 0.30 s (errdist 0.451) and loses to it past about 1.5 s. Analytic fine-tuning rolls
    # 15 steps, 0.30 s, and is fine. PPO benefits from longer rollouts, and a PPO run
    # asked for 100 steps -- 2 s -- would be optimising the policy against a plant model
    # that is worse than assuming nothing happens, and would find that out only in Chrono.
    # THE TWO RATES. The model steps at the record rate it was trained on; the policy acts
    # at the control rate. Taken from the checkpoint and the collection config rather than
    # restated, because restating them is how the first version came to run the policy at
    # 100 Hz and call 15 model steps 0.30 s.
    exc = yaml.safe_load((HERE / "params" / "excitation.yaml").read_text())
    ctrl_dt = 1.0 / float(exc["episode"]["control_hz"])
    dt_s = float(ck["config"].get("dt_s") or 1.0 / float(exc["episode"]["record_hz"]))
    hold = int(round(ctrl_dt / dt_s))
    if hold < 1 or abs(hold * dt_s - ctrl_dt) > 1e-9:
        raise SystemExit(f"the model steps at {dt_s} s and the policy acts every {ctrl_dt} "
                         f"s; the control period must be a whole number of model steps")
    branch_s = a.steps * ctrl_dt
    prof = ck.get("horizon_profile") or {}
    if prof:
        keys = sorted(prof, key=float)
        near = min(keys, key=lambda k: abs(float(k) - branch_s))
        # Take the nearest profiled horizon AT OR BEYOND the branch, erring pessimistic:
        # a branch between two profiled points is judged by the longer, worse one.
        beyond = [k for k in keys if float(k) >= branch_s - 1e-9]
        judge = beyond[0] if beyond else keys[-1]
        e_b = float(prof[judge])
        print(f"model errdist at {float(judge):g} s (this run rolls {branch_s:.2f} s): "
              f"{e_b:.3f}  [usable to ~{ck.get('usable_to_s')} s]")
        if e_b >= 1.0 and not a.smoke:
            raise SystemExit(
                f"this run rolls the model {branch_s:.2f} s ({a.steps} steps), and at that "
                f"horizon the model's errdist is {e_b:.3f} -- at or above the 1.0 a model "
                f"scores for predicting the robot does not move. Optimising against it "
                f"would chase the model's errors, not the robot. Shorten --steps to within "
                f"the usable horizon (~{ck.get('usable_to_s')} s), or train a better model.")
    else:
        print(f"WARNING: {a.model} records no horizon profile, so whether it is accurate "
              f"over this run's {branch_s:.2f} s branch is unknown. Retrain with the "
              f"current train.py, which records one.")
    state_fields = ck["state_fields"]
    ctx = ck["config"]["block_size"]
    ix = {f: i for i, f in enumerate(state_fields)}
    for n in ("vel_body_x_mps", "vel_body_y_mps", "yaw_rate_radps", "grav_body_z"):
        if n not in ix:
            raise SystemExit(f"the tracking objective needs {n!r}, absent from the preset")

    pol_cfg = yaml.safe_load((HERE / "params" / "policy.yaml").read_text())
    obs = ObsBuilder(torch, state_fields, pol_cfg).to(dev)
    policy, params = load_policy_torch(torch, a.policy, dev)
    baseline = [p.detach().clone() for p in params]
    base_norm = float(torch.sqrt(sum((b ** 2).sum() for b in baseline)))
    n_par = sum(p.numel() for p in params)

    corpus = T.Corpus(Path(a.corpus), ck["config"]["preset"], ctx,
                      extra_fields=EXTRA_FIELDS)
    man = corpus.manifest
    if man.get("row_capture") != "pre_step" or man.get("policy_raw_order") != "chrono":
        raise SystemExit(
            f"{a.corpus} records row_capture={man.get('row_capture')!r} and "
            f"policy_raw_order={man.get('policy_raw_order')!r}. Corpora collected before "
            f"those fields existed captured each row one physics step AFTER the policy "
            f"acted, so the state at a control row already carries the kick from the "
            f"action chosen there, and the policy cannot be rolled on it faithfully. "
            f"Recollect with the current collect.py.")
    from quadruped.lib import provenance as PROV  # noqa: PLC0415
    col_sha = (man.get("policy") or {}).get("sha256")
    if col_sha != PROV.sha256(a.policy):
        raise SystemExit(
            f"{a.corpus} was collected by policy sha256 {str(col_sha)[:12]}, and "
            f"{a.policy} is {PROV.sha256(a.policy)[:12]}. Branches start from states that "
            f"policy produced and observe the actions IT took, so fine-tuning a different "
            f"one from them is not local improvement of anything.")
    if corpus.state_fields != state_fields:
        raise SystemExit(
            "the corpus and the model disagree about the state. Fine-tuning would "
            "optimise against channels in a different order than the model was fitted "
            "on, which produces a confident and meaningless result.")

    p2c = list(pol_cfg["joints"]["policy_to_chrono"])
    pool = build_pool(corpus, ctx, ctrl_dt)
    print(f"timing: model step {dt_s} s, policy every {ctrl_dt} s (hold {hold}); "
          f"{len(pool):,} branch starts on control rows; commands from {a.command}")
    b0 = start_batch(torch, corpus, pool, min(len(pool), 4096), random.Random(a.seed + 1),
                     ctx, p2c, dev)
    start_err = check_start_reproduction(torch, obs, policy, b0)
    del b0

    rng = random.Random(a.seed)
    opt = torch.optim.Adam(params, lr=a.lr)
    ranges = exc["commands"]["ranges"]
    log = (out / "finetune.jsonl").open("a")

    import math as _m
    if a.method == "ppo":
        it, dw = run_ppo(torch, torch.nn, a, model, obs, policy, params, baseline,
                         base_norm, n_par, corpus, ctx, ix, ranges, rng, dev, log,
                         pool, p2c, hold)
        log.close()
        torch.jit.save(policy, str(out / "policy_ft.pt"))
        meta = {"smoke": bool(a.smoke) or bool(ck.get("smoke")), "method": "ppo",
                "iters_run": it, "dw": dw, "dw_rel": dw / base_norm,
                "target_dw": a.target_dw, "branches": a.branches, "steps": a.steps,
                "lr": a.lr, "clip_eps": a.clip_eps, "gamma": a.gamma, "lam": a.lam,
                "ppo_epochs": a.ppo_epochs, "init_log_std": a.init_log_std,
                "hold": hold, "ctrl_dt": ctrl_dt, "dt_s": dt_s, "command_source": a.command,
                "start_reproduction": start_err, "corpus_row_capture": man.get("row_capture"),
                "policy_params": n_par, "theta0_norm": base_norm, "seed": a.seed,
                "model": str(a.model), "base_policy": str(a.policy),
                "corpus": str(a.corpus)}
        (out / "finetune.json").write_text(json.dumps(meta, indent=2))
        if a.smoke or ck.get("smoke"):
            print("\nSMOKE RUN. The NN-ROM was smoke-stamped; this is a test of the code "
                  "path, not a result.")
        print(f"\nwrote {out}/policy_ft.pt")
        print("IN-MODEL GAIN IS NOT A RESULT. Score it in Chrono with evaluate.py "
              "against the base policy, on one machine, over replicates.")
        return 0

    print(f"analytic fine-tune: {a.branches} branches x {a.steps} control steps "
          f"({a.steps * ctrl_dt:.2f} s, {a.steps * hold} model steps), target dw {a.target_dw}")
    print(f"  policy {n_par:,} params, ||theta_0|| {base_norm:.3f}; at lr {a.lr} one Adam "
          f"step moves dw by about {a.lr * _m.sqrt(n_par):.4f}, so the budget is roughly "
          f"{a.target_dw / max(a.lr * _m.sqrt(n_par), 1e-12):.0f} steps")
    t0 = time.perf_counter()
    dw = 0.0
    for it in range(1, a.iters + 1):
        b = start_batch(torch, corpus, pool, a.branches, rng, ctx, p2c, dev)
        cmd = branch_cmd(torch, a, b, ranges, rng, dev)
        track, upright = branch_loss(torch, model, obs, policy, b, cmd, a.steps, hold, ix)
        loss = track + a.upright_weight * upright
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        with torch.no_grad():
            dw = float(weight_displacement(torch, params, baseline))
        # Reported relative to the baseline norm as well, because an absolute displacement
        # means nothing without knowing how big theta is: the same 4.0 is a rounding error
        # on one policy and a rewrite on another.
        rec = {"iter": it, "track": track.item(), "upright": upright.item(),
               "dw": dw, "dw_rel": dw / base_norm}
        log.write(json.dumps(rec) + "\n")
        if it % 25 == 0 or it == 1:
            print(f"  iter {it:5d}  track {track.item():.5f}  upright "
                  f"{upright.item():.5f}  dw {dw:.4f} ({100 * dw / base_norm:.2f}% of "
                  f"||theta_0||)", flush=True)
        if dw >= a.target_dw:
            print(f"  stopping: dw {dw:.4f} reached the {a.target_dw} budget at iter {it}")
            break

    log.close()
    # Already a ScriptModule; re-scripting it is a no-op at best and an error
    # at worst. Saved in the same form the collector and evaluator load.
    torch.jit.save(policy, str(out / "policy_ft.pt"))
    meta = {"smoke": bool(a.smoke) or bool(ck.get("smoke")),
            "dw_rel": dw / base_norm, "policy_params": n_par, "theta0_norm": base_norm,
            "method": a.method, "iters_run": it, "dw": dw, "target_dw": a.target_dw,
            "branches": a.branches, "steps": a.steps, "lr": a.lr, "seed": a.seed,
            "hold": hold, "ctrl_dt": ctrl_dt, "dt_s": dt_s, "command_source": a.command,
            "start_reproduction": start_err, "corpus_row_capture": man.get("row_capture"),
            "model": str(a.model), "base_policy": str(a.policy),
            "corpus": str(a.corpus), "seconds": round(time.perf_counter() - t0, 1)}
    (out / "finetune.json").write_text(json.dumps(meta, indent=2))
    if a.smoke or ck.get("smoke"):
        print("\nSMOKE RUN. The NN-ROM was smoke-stamped, so this policy was fine-tuned "
              "inside a model that was never properly selected. Stamped smoke=True; it "
              "is a test of this code path, not a result.")
    print(f"\nwrote {out}/policy_ft.pt")
    print("IN-MODEL GAIN IS NOT A RESULT. In-model gain anti-correlates with transfer; "
          "score this in Chrono with evaluate.py against the base policy, on one "
          "machine, over replicates, before it means anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
