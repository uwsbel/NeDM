#!/usr/bin/env python3
"""Fine-tune the policy inside the NN-ROM.

Two methods, one interface. `--method analytic` is the one that works today;
`--method ppo` is the one we want to work, because analytic leans on a property -- a
differentiable model of the plant -- that a general method should not need.

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


def branch_starts(corpus, ctx, n, rng):
    """Windows of `ctx` real states, drawn from the corpus, to start branches from.

    RECORDED STATES, not sampled ones. This is the property that makes the method local:
    the policy is improved on states the corpus actually contains, so coverage of the
    corpus bounds what can be learned. Drawn from the TRAIN split, since the val split is
    what any honest in-model number would have to be read on.
    """
    pool = [(si, k) for si, r in enumerate(corpus.train)
            for k in range(max(r["n"] - ctx - 1, 0))]
    if not pool:
        raise SystemExit("no segment is long enough to start a branch from")
    return [pool[rng.randrange(len(pool))] for _ in range(n)]


# ------------------------------------------------------------------ the loss

def branch_loss(torch, model, obs, policy, states0, actions0, cmd, steps, ix):
    """Roll the policy inside the frozen model for `steps`, score against the command.

    The score is command tracking in the body frame, which is what the study reports and
    what the base policy is weakest at. Height and attitude are held with a light penalty
    rather than optimised: without them the optimiser discovers that lying down tracks
    zero velocity beautifully, and a fine-tune that falls over is not an improvement
    however good its number is.
    """
    B, ctx, S = states0.shape
    hist_s = states0.clone()
    hist_a = actions0.clone()
    last_raw = torch.zeros(B, 12, device=states0.device)
    track, upright = [], []
    for _ in range(steps):
        cur = hist_s[:, -1]
        o = obs.observe(cur, cmd, last_raw)
        raw = policy(o)
        last_raw = raw
        act = obs.action_from_raw(raw)
        hist_a = torch.cat([hist_a[:, 1:], act[:, None]], dim=1)
        delta = model.predict_delta(hist_s, hist_a)[:, -1]
        nxt = cur + delta
        hist_s = torch.cat([hist_s[:, 1:], nxt[:, None]], dim=1)
        v = torch.stack([nxt[:, ix["vel_body_x_mps"]], nxt[:, ix["vel_body_y_mps"]],
                         nxt[:, ix["yaw_rate_radps"]]], dim=-1)
        track.append(((v - cmd) ** 2).sum(-1))
        # grav_body_z is -1 when level; anything above that is the trunk pitching over.
        upright.append((nxt[:, ix["grav_body_z"]] + 1.0) ** 2)
    return torch.stack(track).mean(), torch.stack(upright).mean()


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
                    help="branch length; 15 steps is 0.30 s at 50 Hz control")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--target-dw", type=float, default=4.0,
                    help="stop at this ||theta - theta_0||; the budget is in weight "
                         "space because in-model reward can be gamed and this cannot. "
                         "Adam normalises per parameter, so one step moves the vector by "
                         "about lr*sqrt(N): 0.043 here, and dw 4.0 is therefore roughly "
                         "100 steps. A budget of 0.05 stops after ONE.")
    ap.add_argument("--upright-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true",
                    help="permit a smoke-stamped NN-ROM so this code path can be "
                         "exercised before a real corpus exists. Stamps the output.")
    a = ap.parse_args()

    if a.method == "ppo":
        raise SystemExit(
            "PPO is not implemented here yet, and a stub that silently did something "
            "else would be worse than this message. The analytic path is the one with "
            "evidence behind it; PPO is the one that removes the dependence on a "
            "differentiable plant, and it needs its own rollout buffer, advantage "
            "estimation and clipped objective rather than a rename of this loop.")

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

    corpus = T.Corpus(Path(a.corpus), ck["config"]["preset"], ctx)
    if corpus.state_fields != state_fields:
        raise SystemExit(
            "the corpus and the model disagree about the state. Fine-tuning would "
            "optimise against channels in a different order than the model was fitted "
            "on, which produces a confident and meaningless result.")

    rng = random.Random(a.seed)
    opt = torch.optim.Adam(params, lr=a.lr)
    ranges = yaml.safe_load((HERE / "params" / "excitation.yaml").read_text())["commands"]["ranges"]
    log = (out / "finetune.jsonl").open("a")

    import math as _m
    print(f"analytic fine-tune: {a.branches} branches x {a.steps} steps "
          f"({a.steps * 0.02:.2f} s), target dw {a.target_dw}")
    print(f"  policy {n_par:,} params, ||theta_0|| {base_norm:.3f}; at lr {a.lr} one Adam "
          f"step moves dw by about {a.lr * _m.sqrt(n_par):.4f}, so the budget is roughly "
          f"{a.target_dw / max(a.lr * _m.sqrt(n_par), 1e-12):.0f} steps")
    t0 = time.perf_counter()
    dw = 0.0
    for it in range(1, a.iters + 1):
        picks = branch_starts(corpus, ctx, a.branches, rng)
        S = np.stack([corpus.train[si]["state"][k:k + ctx] for si, k in picks])
        A = np.stack([corpus.train[si]["action"][k:k + ctx] for si, k in picks])
        states0 = torch.tensor(S, dtype=torch.float32, device=dev)
        actions0 = torch.tensor(A, dtype=torch.float32, device=dev)
        cmd = torch.tensor(
            [[rng.uniform(*ranges["vx"]), rng.uniform(*ranges["vy"]),
              rng.uniform(*ranges["wz"])] for _ in range(a.branches)],
            dtype=torch.float32, device=dev)

        track, upright = branch_loss(torch, model, obs, policy, states0, actions0,
                                     cmd, a.steps, ix)
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
