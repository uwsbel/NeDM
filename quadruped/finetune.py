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
# The OU noise the collector added to the policy's target, chrono order, already signed.
INJ_FIELDS = [f"action_injection_{l}_{s}_rad" for l in ("rr", "rl", "fr", "fl")
              for s in ("hip", "thigh", "calf")]
EXTRA_FIELDS = ["time_s"] + CMD_FIELDS + RAW_FIELDS + INJ_FIELDS
_T, _CMD, _RAW, _INJ = 0, slice(1, 4), slice(4, 16), slice(16, 28)


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
            f"ObsBuilder matches lib/policy.py (diagnostics/obs_truth.py tests exactly that).")
    return rms / max(spread, 1e-12)


def advance(model, torch, hist_s, hist_a, act):
    """One model step. THE ONLY PLACE the state and action histories are aligned.

    `hist_a` holds the actions for every row of `hist_s` but the last; `act` is the one
    applied at the last row. The model pairs state row j with action row j, exactly as
    `train.py` fits and rolls it. Every rollout in this file steps through here, so the
    alignment is written once and tested once.
    """
    a_win = torch.cat([hist_a, act[:, None]], dim=1)
    nxt = hist_s[:, -1] + model.predict_delta(hist_s, a_win)[:, -1]
    return nxt, torch.cat([hist_s[:, 1:], nxt[:, None]], dim=1), a_win[:, 1:]


class Ensemble:
    """Several NN-ROMs behind the one-model interface advance() expects.

    WHY. Across eleven single-surrogate PPO runs the yaw gain held everywhere, but the
    forward-speed gain ran from -54% to +80% between surrogates of equal accuracy, and no
    training statistic (selected epoch, open-loop error at any horizon) predicted which way
    a given surrogate would go. Each surrogate carries its own large, idiosyncratic error,
    and a policy tuned against one inherits it. Two standard defences, both here:

      - Every branch is rolled in a member drawn at random each iteration (MBPO), so no
        single model's error is available to exploit consistently.
      - The members' spread on each predicted step is kept (`last_spread`) for a
        disagreement penalty on the reward (MOPO): where the models disagree, the
        prediction is not knowledge, and reward there should not be believed.

    Every member is evaluated on every row, which the spread needs anyway; the cost is M x
    one model's rollout, and PPO's rollouts are the cheap part of an iteration.
    """

    def __init__(self, torch, members):
        self.torch = torch
        self.members = members
        self.assign = None          # (B,) member index per row, or None for the mean
        self.last_spread = None     # (B,) normalised spread of the last predicted delta
        self.tstd = members[0].target_std

    def assign_random(self, n, gen):
        self.assign = self.torch.randint(0, len(self.members), (n,), generator=gen,
                                         device=self.tstd.device)

    def predict_delta(self, hist_s, a_win):
        t = self.torch
        preds = t.stack([m.predict_delta(hist_s, a_win) for m in self.members])
        last = preds[:, :, -1]                                    # (M, B, S)
        self.last_spread = ((last.std(0) / self.tstd) ** 2).mean(-1).sqrt()
        if self.assign is None:
            return preds.mean(0)
        return preds[self.assign, t.arange(preds.shape[1], device=preds.device)]


def closed_loop(torch, model, obs, policy, b, cmd, steps, hold, inj=None):
    """Roll the deterministic policy inside the model: `steps` control steps, each action
    held for `hold` model steps. Returns the predicted states, (B, steps*hold, S).

    The analytic loss differentiates through this; the fidelity check runs it with the
    recorded injection noise added (`inj`, (B, steps*hold, 12), chrono order) so that on a
    faithful loop it takes the very actions the robot took.
    """
    hist_s, hist_a, last_raw = b["states"], b["acts"], b["last_raw"]
    out = []
    for c in range(steps):
        o = obs.observe(hist_s[:, -1], cmd, last_raw)
        last_raw = torch.clamp(policy(o), *obs.act_clip)
        act = obs.action_from_raw(last_raw)
        if inj is not None:
            act = act + inj[:, c * hold]
        for _h in range(hold):
            nxt, hist_s, hist_a = advance(model, torch, hist_s, hist_a, act)
            out.append(nxt)
    return torch.stack(out, dim=1)


def closed_loop_actions(torch, model, obs, policy, b, cmd, steps, hold):
    """The actions `closed_loop` takes, aligned with the states it returns.

    Same rollout, recording the action instead of the state. Separate rather than an extra
    return value so the loss path, which differentiates through closed_loop, is untouched.
    """
    hist_s, hist_a, last_raw = b["states"], b["acts"], b["last_raw"]
    out = []
    for _c in range(steps):
        o = obs.observe(hist_s[:, -1], cmd, last_raw)
        last_raw = torch.clamp(policy(o), *obs.act_clip)
        act = obs.action_from_raw(last_raw)
        for _h in range(hold):
            _nxt, hist_s, hist_a = advance(model, torch, hist_s, hist_a, act)
            out.append(act)
    return torch.stack(out, dim=1)


def check_loop_fidelity(torch, model, obs, policy, corpus, ctx, steps, hold, ctrl_dt, p2c,
                        ix, dev, n=256, seed=0):
    """Does the closed loop inside the model reproduce the recorded closed loop?

    The start check proves the FIRST observation is right. This proves the whole branch:
    from held-out starts, the base policy is rolled inside the model exactly as the
    fine-tune rolls it -- same hold, same alignment, same observation -- but with the
    recorded injection noise added back, so that on a faithful loop it takes the very
    actions the robot took. Its predicted velocities are compared with the recorded ones,
    and so are those of an OPEN-loop rollout of the recorded actions from the same starts.

    Open-loop error is the model's own error. Closed-loop error adds whatever the loop
    gets wrong -- timing, alignment, observation -- compounded through the policy's
    feedback. On a faithful loop the two are close; the ratio is the number to read.

    REPORTED, NOT GATED, because its power is unproven. A negative control on a barely
    trained rigid smoke model -- the policy stepped every model step, the v1 timing fault
    -- scored the same ratio (0.99) as the correct loop: over 0.30 s the model's own error
    swamped the difference. The start check is what refuses; this is a number to read
    beside it until a negative control on a real model shows it can tell the two apart.

    WHY A CURVE AND NOT ONE RATIO. On 2 s branches the closed-loop error came out at
    0.27-0.30 in every surrogate while open-loop error ran from 0.12 to 0.23, so the ratio
    tracked the DENOMINATOR: the most accurate surrogate scored the worst ratio (2.41) and
    the least accurate the best (1.28), and every good 2 s fine-tune tripped the old fixed
    warning at 2.0. The loop has NOT lost the recording by then: scored against a
    decorrelated reference (each start against another start's recording) seed 7's 0.5 s
    rollout-trained surrogate reads 0.29 closed-loop against 0.64 at 2 s, and 0.15 against
    0.63 at 0.30 s. What does not shrink with a better model is the part the policy's
    feedback adds, so over long branches the ratio mostly measures how small the open-loop
    error has become. The errors are therefore printed along the branch beside that
    reference, and the warning reads the ratio at 0.30 s, the horizon at which it was set.
    """
    need = steps * hold
    pool = []
    for si, r in enumerate(corpus.val):
        ph = r["extra"][:, _T] / ctrl_dt
        for k in range(max(r["n"] - ctx - need, 0)):
            j = k + ctx - 1
            if abs(ph[j] - round(ph[j])) < 1e-6:
                pool.append((si, k))
    if not pool:
        print("loop check: skipped, no held-out segment is long enough")
        return None
    rng = random.Random(seed)
    picks = [pool[rng.randrange(len(pool))] for _ in range(min(n, len(pool)))]
    f = lambda x: torch.tensor(np.stack(x), dtype=torch.float32, device=dev)  # noqa: E731
    segs = corpus.val
    S0 = f([segs[si]["state"][k:k + ctx] for si, k in picks])
    A_hist = f([segs[si]["action"][k:k + ctx - 1] for si, k in picks])
    A_fut = f([segs[si]["action"][k + ctx - 1:k + ctx - 1 + need] for si, k in picks])
    S_fut = f([segs[si]["state"][k + ctx:k + ctx + need] for si, k in picks])
    E = f([segs[si]["extra"][k + ctx - 2:k + ctx - 1 + need] for si, k in picks])
    cmd = E[:, 1, _CMD]
    vel = [ix["vel_body_x_mps"], ix["vel_body_y_mps"], ix["yaw_rate_radps"]]

    b = {"states": S0, "acts": A_hist, "last_raw": E[:, 0, _RAW][:, p2c]}
    with torch.no_grad():
        pred_closed = closed_loop(torch, model, obs, policy, b, cmd, steps, hold,
                                  inj=E[:, 1:1 + need, _INJ])[..., vel]
        hs, ha, pred_open = S0, A_hist, []
        for m in range(need):
            nxt, hs, ha = advance(model, torch, hs, ha, A_fut[:, m])
            pred_open.append(nxt[:, vel])
        pred_open = torch.stack(pred_open, dim=1)

    truth = S_fut[..., vel]
    e_open = float(torch.sqrt(((pred_open - truth) ** 2).mean()))
    e_closed = float(torch.sqrt(((pred_closed - truth) ** 2).mean()))
    ratio = e_closed / max(e_open, 1e-12)
    print(f"loop check ({len(picks)} held-out starts, {need} model steps): velocity RMSE "
          f"open-loop {e_open:.4f}, closed-loop {e_closed:.4f}  (ratio {ratio:.2f})")
    # Error up to each mark along the branch, and the decorrelated reference: every start
    # scored against the NEXT start's recording, i.e. the error of a real trajectory that
    # knows nothing about this one. Closed-loop error reaching it means the loop no longer
    # follows the recording at that horizon. No randomness is drawn here.
    other = torch.roll(truth, 1, dims=0)
    step_s = ctrl_dt / hold
    curve = []
    for t_s in (0.1, 0.3, 0.5, 1.0, 2.0, 5.0):
        m = int(round(t_s / step_s))
        if m > need:
            break
        rm = lambda x, y: float(torch.sqrt(((x[:, :m] - y[:, :m]) ** 2).mean()))  # noqa: E731
        curve.append({"t_s": t_s, "open": rm(pred_open, truth),
                      "closed": rm(pred_closed, truth), "decorrelated": rm(other, truth)})
    for c in curve:
        print(f"  to {c['t_s']:3.1f} s: open {c['open']:.4f}  closed {c['closed']:.4f}  "
              f"decorrelated {c['decorrelated']:.4f}  (closed/open "
              f"{c['closed'] / max(c['open'], 1e-12):.2f}, closed/decorrelated "
              f"{c['closed'] / max(c['decorrelated'], 1e-12):.2f})")
    short = next((c for c in curve if c["t_s"] == 0.3), None)
    short_ratio = short["closed"] / max(short["open"], 1e-12) if short else None
    if short_ratio is not None and short_ratio > 2.0:
        print("  WARNING: within 0.30 s the closed loop inside the model is far worse than "
              "the model itself. The fine-tune will be optimising a loop that is not the "
              "robot's.")
    return {"open": e_open, "closed": e_closed, "ratio": ratio, "n": len(picks),
            "ratio_at_0.3s": short_ratio, "curve": curve}


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


def branch_kicks(torch, n_branch, steps, prob, dv, yaw_dv, ix, dim, dev, gen):
    """Which branches get shoved, when, and by how much.

    The base policy was trained in robot_lab with a push event every 10-15 s (a velocity
    kick of +/-0.5 m/s in x and y) on top of randomised friction, masses, COM and actuator
    gains. Our fine-tune has none of that: one surrogate, one soil, no disturbance, and a
    reward made only of tracking error and uprightness. Anything the base policy knows that
    only pays off when disturbed therefore has no gradient protecting it -- and the push
    test shows it eroding, robustness at 300 N falling from the base policy's level at
    iteration 500 to clearly below it by 1500 while tracking keeps improving.

    So the disturbance goes back in, in the model: one kick per chosen branch, at a uniform
    control step, added to the body-frame velocity the model carries forward. The surrogate
    can roll the recovery because recoveries ARE in the corpus -- only the force windows
    themselves were cut out of it (collect.py's PushSchedule).
    """
    if prob <= 0.0:
        return None
    hit = torch.rand(n_branch, generator=gen, device=dev) < prob
    when = torch.randint(0, steps, (n_branch,), generator=gen, device=dev)
    kick = torch.zeros(n_branch, dim, device=dev)
    u = (torch.rand(n_branch, 3, generator=gen, device=dev) * 2 - 1)
    kick[:, ix["vel_body_x_mps"]] = u[:, 0] * dv
    kick[:, ix["vel_body_y_mps"]] = u[:, 1] * dv
    if yaw_dv > 0:
        kick[:, ix["yaw_rate_radps"]] = u[:, 2] * yaw_dv
    return {"when": when, "kick": kick * hit[:, None].float()}


def ppo_rollout(torch, model, obs_b, actor, critic, b, steps, hold, cmd, ix,
                upright_weight, ood=None, ood_weight=0.0, dis_weight=0.0, gen=None,
                kicks=None):
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
    ood_buf, dis_buf = [], []
    is_ens = isinstance(model, Ensemble)
    if is_ens:
        model.assign_random(hist_s.shape[0], gen)
    with torch.no_grad():
        for _t in range(steps):
            o = obs_b.observe(hist_s[:, -1], cmd, last_raw)
            d = actor.dist(o)
            raw = d.sample()
            logp = d.log_prob(raw).sum(-1)
            val = critic(o)
            last_raw = torch.clamp(raw, *obs_b.act_clip)
            act = obs_b.action_from_raw(last_raw)
            r = 0.0
            for _h in range(hold):
                nxt, hist_s, hist_a = advance(model, torch, hist_s, hist_a, act)
                if kicks is not None and _h == 0:
                    # Applied to the state the model carries forward, so the policy sees it
                    # in the next observation exactly as it would see a real shove.
                    add = kicks["kick"] * (kicks["when"] == _t)[:, None].float()
                    nxt = nxt + add
                    hist_s = torch.cat([hist_s[:, :-1], nxt[:, None]], dim=1)
                rh = step_reward(torch, nxt, cmd, ix, upright_weight)
                if ood is not None:
                    # MEASURED whether or not it is PRICED. The guard reads this number, and
                    # a run with the penalty switched off is exactly the run whose health is
                    # most worth watching; computing it only when it is charged left the
                    # monitor blind in that case.
                    c = ood.cost(torch, nxt, act)
                    if ood_weight > 0:
                        rh = rh - ood_weight * c
                    ood_buf.append(c.mean())
                if is_ens:
                    dis_buf.append(model.last_spread.mean())
                    if dis_weight > 0:
                        rh = rh - dis_weight * model.last_spread
                r = r + rh / hold
            obs_buf.append(o); act_buf.append(raw); logp_buf.append(logp)
            val_buf.append(val); rew_buf.append(r)
        last_val = critic(obs_b.observe(hist_s[:, -1], cmd, last_raw))
    if is_ens:
        model.assign = None         # anything after this (the loop check) sees the mean
    ood_mean = float(torch.stack(ood_buf).mean()) if ood_buf else 0.0
    dis_mean = float(torch.stack(dis_buf).mean()) if dis_buf else 0.0
    return (torch.stack(obs_buf), torch.stack(act_buf), torch.stack(logp_buf),
            torch.stack(rew_buf), torch.stack(val_buf), last_val, ood_mean, dis_mean)


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
    pred = closed_loop(torch, model, obs, policy, b, cmd, steps, hold)
    v = pred[..., [ix["vel_body_x_mps"], ix["vel_body_y_mps"], ix["yaw_rate_radps"]]]
    track = ((v - cmd[:, None]) ** 2).sum(-1)
    # grav_body_z is -1 when level; anything above that is the trunk pitching over.
    upright = (pred[..., ix["grav_body_z"]] + 1.0) ** 2
    return track.mean(), upright.mean()


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
    if a.ood_penalty > 0 or a.guard_spike < 1.0 or a.monitor_ood:
        ood = OODCost(corpus, seed=a.seed).to_torch(torch, dev)
        how = (f"penalty {a.ood_penalty}" if a.ood_penalty > 0 else "measured only, not priced")
        print(f"  OOD {how}, beyond corpus self-distance {ood.thresh:.3f} "
              f"(the same kNN reference Gate 4 uses)")
    print(f"ppo: {a.branches} branches x {a.steps} control steps ({a.steps * hold} model "
          f"steps), clip {a.clip_eps}, {a.ppo_epochs} epochs x {a.minibatches} minibatches")
    if a.branch_push_prob > 0:
        print(f"  disturbance: {100 * a.branch_push_prob:.0f}% of branches take one kick of "
              f"+/-{a.branch_push_dv:g} m/s (yaw +/-{a.branch_push_yaw:g} rad/s) at a "
              f"random control step, as robot_lab trained the base policy")
    dw, it = 0.0, 0
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed + 99)
    # Snapshots along the run, every --snapshot-every iterations and whenever dw first
    # passes a --snapshot-dw mark, for scoring how transfer changes with training length.
    # Saving draws no randomness, so the run is the same one it would be without them.
    marks = sorted(a.snapshot_dw)
    snaps, snap_dir = [], Path(a.out) / "snapshots"
    # THE GUARD. One run in about fifty has come apart: PPO found a region its surrogate
    # scored well and the robot's real dynamics do not support, in-model reward fell from
    # -0.046 to -1.01, value loss reached 480,000, and the policy ended up tumbling. Every
    # sign of it was in this log hours before any Chrono scoring, so the failure does not
    # need to be caught by eye. A rolling window watches the same signals; the last window
    # that looked healthy is kept, and when the window trips, the run stops and that
    # checkpoint is the result. The thresholds are defaults, not laws: see docs/STATE.md
    # for how they were calibrated and on how many induced failures.
    guard, good, ema = [], None, None

    def snapshot(name, rec):
        snap_dir.mkdir(exist_ok=True)
        torch.jit.save(policy, str(snap_dir / name))
        snaps.append(rec)
        (snap_dir / "snapshots.json").write_text(json.dumps(snaps, indent=2))
    if isinstance(model, Ensemble):
        print(f"  ensemble of {len(model.members)} surrogates: each branch in a random "
              f"member; disagreement penalty {a.disagreement_penalty}")
    for it in range(1, a.iters + 1):
        b = start_batch(torch, corpus, pool, a.branches, rng, ctx, p2c, dev)
        cmd = branch_cmd(torch, a, b, ranges, rng, dev)
        kicks = branch_kicks(torch, a.branches, a.steps, a.branch_push_prob,
                             a.branch_push_dv, a.branch_push_yaw, ix,
                             b["states"].shape[-1], dev, gen)
        ob, ac, lp, rw, vl, last_val, ood_mean, dis_mean = ppo_rollout(
            torch, model, obs_b, actor, critic, b, a.steps, hold, cmd, ix,
            a.upright_weight, ood, a.ood_penalty, a.disagreement_penalty, gen, kicks)
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
               "log_std": actor.log_std.mean().item(), "ood": ood_mean,
               "disagreement": dis_mean}
        log.write(json.dumps(rec) + "\n")
        if it % 10 == 0 or it == 1:
            print(f"  iter {it:5d}  reward {rw.mean().item():+.4f}  vloss {vf:.4f}  "
                  f"ent {ent:+.3f}  log_std {actor.log_std.mean().item():+.2f}  "
                  f"dw {dw:.4f} ({100 * dw / base_norm:.2f}%)  ood {ood_mean:.4f}  "
                  f"dis {dis_mean:.4f}",
                  flush=True)
        while marks and dw >= marks[0]:
            m = marks.pop(0)
            snapshot(f"policy_dw{m:g}.pt", {"file": f"policy_dw{m:g}.pt", "dw_mark": m,
                                            "iter": it, "dw": dw, "reward": rw.mean().item()})
            print(f"  snapshot: dw {dw:.4f} passed {m:g} at iter {it}", flush=True)
        # Rolling health: the spike rate PPO's own sampling produces, and whether the
        # reward is still going the right way.
        guard.append(rec)
        if len(guard) > a.guard_window:
            guard.pop(0)
        if len(guard) == a.guard_window:
            spike = sum(1 for r in guard if r["ood"] > a.guard_ood) / a.guard_window
            mean_r = sum(r["reward"] for r in guard) / a.guard_window
            ema = mean_r if ema is None else max(ema, mean_r)
            drop = (ema - mean_r) / max(abs(ema), 1e-9)
            # BOTH have to be bad. A burst of spikes on its own is PPO sampling near the
            # edge, which healthy runs do; reward falling on its own is a hard patch of
            # branch starts. The failure is the pair: leaving the corpus AND getting worse.
            healthy = not (spike > a.guard_spike and drop > a.guard_drop)
            if healthy:
                good = {"iter": it, "dw": dw, "reward": mean_r, "spike_rate": spike}
                if a.guard_spike < 1.0:
                    snap_dir.mkdir(exist_ok=True)
                    torch.jit.save(policy, str(snap_dir / "policy_lastgood.pt"))
            elif a.guard_spike < 1.0:
                print(f"  GUARD: over the last {a.guard_window} iterations {100 * spike:.0f}% "
                      f"left the corpus region (limit {100 * a.guard_spike:.0f}%) and reward "
                      f"is {100 * drop:.0f}% off its best (limit {100 * a.guard_drop:.0f}%). "
                      f"Stopping. Last healthy window: {good}", flush=True)
                if good is not None:
                    import shutil  # noqa: PLC0415
                    shutil.copy(snap_dir / "policy_lastgood.pt", Path(a.out) / "policy_guard.pt")
                return it, dw, {"tripped": True, "at_iter": it, "spike_rate": spike,
                                "reward_drop": drop, "last_good": good}
        if a.snapshot_every and it % a.snapshot_every == 0:
            snapshot(f"policy_it{it}.pt", {"file": f"policy_it{it}.pt", "iter_mark": it,
                                           "iter": it, "dw": dw, "reward": rw.mean().item()})
            print(f"  snapshot: iter {it}, dw {dw:.4f}", flush=True)
        if dw >= a.target_dw:
            print(f"  stopping: dw {dw:.4f} reached the {a.target_dw} budget at iter {it}")
            break
    return it, dw, {"tripped": False, "last_good": good}



def weight_displacement(torch, params, baseline):
    return torch.sqrt(sum(((p - b) ** 2).sum() for p, b in zip(params, baseline)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, nargs="+",
                    help="one or more trained NN-ROM checkpoints. More than one is an "
                         "ENSEMBLE: PPO rolls each branch in a random member (see Ensemble)")
    ap.add_argument("--disagreement-penalty", type=float, default=0.0,
                    help="PPO with an ensemble: reward penalty per unit of the members' "
                         "normalised spread on each predicted step. 0 = members are only "
                         "randomised over, not penalised on.")
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
    ap.add_argument("--snapshot-dw", type=float, nargs="*", default=[],
                    help="also save the policy as snapshots/policy_dw<m>.pt when dw first "
                         "passes each mark (PPO only). With a large --target-dw, one run "
                         "gives the policy at every budget, for testing the budget itself")
    ap.add_argument("--monitor-ood", action="store_true",
                    help="compute the OOD cost for the log even when it is not priced into "
                         "the reward and the guard is off")
    ap.add_argument("--guard-window", type=int, default=50,
                    help="iterations in the health window (see the guard in run_ppo)")
    ap.add_argument("--guard-ood", type=float, default=0.005,
                    help="an iteration counts as a spike when its mean OOD cost exceeds this")
    ap.add_argument("--guard-spike", type=float, default=0.6,
                    help="stop when more than this fraction of the window spikes AND the "
                         "reward has fallen; 1.0 disables the guard but keeps the log")
    ap.add_argument("--guard-drop", type=float, default=0.25,
                    help="stop when the window's mean reward is this far below its best")
    ap.add_argument("--branch-push-prob", type=float, default=0.0,
                    help="fraction of branches given one velocity kick, at a uniformly "
                         "drawn control step (PPO only). The base policy was trained with "
                         "a push every 10-15 s; a 2 s branch is ~15% of that interval")
    ap.add_argument("--branch-push-dv", type=float, default=0.5,
                    help="kick size in m/s, uniform in +/-dv on body x and y; 0.5 matches "
                         "robot_lab's randomize_push_robot")
    ap.add_argument("--branch-push-yaw", type=float, default=0.0,
                    help="kick size in rad/s on yaw rate; upstream pushes do not turn the "
                         "robot, so this is off by default")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="also save the policy as snapshots/policy_it<N>.pt every N "
                         "iterations (PPO only); 0 disables")
    ap.add_argument("--upright-weight", type=float, default=0.5)
    ap.add_argument("--accum", type=int, default=2,
                    help="analytic only: split the branches into this many micro-batches, "
                         "backpropagated one at a time and summed before one step. The "
                         "gradient is the same as one batch of --branches; only memory "
                         "changes. Needed since the policy holds each action for two model "
                         "steps: a 0.30 s branch is 30 model steps, not 15, the graph "
                         "doubled, and 64 branches ran a 16 GB card out of memory.")
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

    if len(a.model) > 1 and a.method != "ppo":
        raise SystemExit("an ensemble is supported for --method ppo only: the analytic path "
                         "backpropagates through every member, M x the memory of one")
    loaded = [load_nnrom(torch, m, dev, allow_smoke=a.smoke) for m in a.model]
    ck = loaded[0][1]
    for mp, (_m, c) in zip(a.model[1:], loaded[1:]):
        if (c["state_fields"] != ck["state_fields"] or c["action_fields"] != ck["action_fields"]
                or c["config"]["block_size"] != ck["config"]["block_size"]
                or c["config"].get("dt_s") != ck["config"].get("dt_s")):
            raise SystemExit(f"{mp} does not share fields, context or timestep with "
                             f"{a.model[0]}; an ensemble must be one model class")
        for k in ck["stats"]:
            if not np.allclose(np.asarray(c["stats"][k]), np.asarray(ck["stats"][k])):
                raise SystemExit(f"{mp} was fitted under different normalisation ({k})")

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

    # IS EACH MODEL ACCURATE OVER THE HORIZON THIS RUN WILL ROLL IT? Checked per member,
    # against this run's own branch length, because a model is not good or bad in general
    # -- it is good out to some horizon. A branch between two profiled points is judged by
    # the longer, worse one.
    for mp, (_m, c) in zip(a.model, loaded):
        prof = c.get("horizon_profile") or {}
        if not prof:
            print(f"WARNING: {mp} records no horizon profile, so whether it is accurate over "
                  f"this run's {branch_s:.2f} s branch is unknown.")
            continue
        keys = sorted(prof, key=float)
        beyond = [k for k in keys if float(k) >= branch_s - 1e-9]
        judge = beyond[0] if beyond else keys[-1]
        e_b = float(prof[judge])
        # train.py records None when the profile never reaches the floor.
        usable = c.get("usable_to_s")
        usable = f"~{usable} s" if usable is not None else f"past {float(keys[-1]):g} s"
        print(f"model errdist at {float(judge):g} s (this run rolls {branch_s:.2f} s): "
              f"{e_b:.3f}  [usable {usable}]  {Path(mp).parent.name}")
        if e_b >= 1.0 and not a.smoke:
            raise SystemExit(
                f"this run rolls {branch_s:.2f} s ({a.steps} steps), and {mp} scores errdist "
                f"{e_b:.3f} there -- at or above the 1.0 a model scores for predicting the "
                f"robot does not move. Optimising against it would chase its errors, not "
                f"the robot. Shorten --steps to within {usable}, drop that "
                f"member, or train a better model.")
    model = loaded[0][0] if len(loaded) == 1 else Ensemble(torch, [m for m, _c in loaded])
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
    loop = check_loop_fidelity(torch, model, obs, policy, corpus, ctx, a.steps, hold,
                               ctrl_dt, p2c, ix, dev, seed=a.seed)

    rng = random.Random(a.seed)
    opt = torch.optim.Adam(params, lr=a.lr)
    ranges = exc["commands"]["ranges"]
    log = (out / "finetune.jsonl").open("a")

    import math as _m
    if a.method == "ppo":
        it, dw, guard = run_ppo(torch, torch.nn, a, model, obs, policy, params, baseline,
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
                "start_reproduction": start_err, "loop_fidelity": loop, "guard": guard,
                "guard_settings": {"window": a.guard_window, "spike": a.guard_spike,
                                   "drop": a.guard_drop, "ood": a.guard_ood},
                "corpus_row_capture": man.get("row_capture"),
                "policy_params": n_par, "theta0_norm": base_norm, "seed": a.seed,
                "model": [str(m) for m in a.model], "base_policy": str(a.policy),
                "disagreement_penalty": a.disagreement_penalty,
                "branch_push": {"prob": a.branch_push_prob, "dv": a.branch_push_dv,
                                "yaw": a.branch_push_yaw},
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
        opt.zero_grad(set_to_none=True)
        n_mb = max(1, a.accum)
        mb = a.branches // n_mb
        track_sum = upright_sum = 0.0
        for _m in range(n_mb):
            b = start_batch(torch, corpus, pool, mb, rng, ctx, p2c, dev)
            cmd = branch_cmd(torch, a, b, ranges, rng, dev)
            tr, up = branch_loss(torch, model, obs, policy, b, cmd, a.steps, hold, ix)
            # Each micro-batch is a mean over its branches, so dividing by their number
            # makes the summed gradient the mean over all of them.
            ((tr + a.upright_weight * up) / n_mb).backward()
            track_sum += tr.item() / n_mb
            upright_sum += up.item() / n_mb
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        with torch.no_grad():
            dw = float(weight_displacement(torch, params, baseline))
        # Reported relative to the baseline norm as well, because an absolute displacement
        # means nothing without knowing how big theta is: the same 4.0 is a rounding error
        # on one policy and a rewrite on another.
        rec = {"iter": it, "track": track_sum, "upright": upright_sum,
               "dw": dw, "dw_rel": dw / base_norm}
        log.write(json.dumps(rec) + "\n")
        if it % 25 == 0 or it == 1:
            print(f"  iter {it:5d}  track {track_sum:.5f}  upright "
                  f"{upright_sum:.5f}  dw {dw:.4f} ({100 * dw / base_norm:.2f}% of "
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
            "branches": a.branches, "accum": a.accum, "steps": a.steps, "lr": a.lr,
            "seed": a.seed, "hold": hold, "ctrl_dt": ctrl_dt, "dt_s": dt_s, "command_source": a.command,
            "start_reproduction": start_err, "loop_fidelity": loop,
                "corpus_row_capture": man.get("row_capture"),
            "model": [str(m) for m in a.model], "base_policy": str(a.policy),
                "disagreement_penalty": a.disagreement_penalty,
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
