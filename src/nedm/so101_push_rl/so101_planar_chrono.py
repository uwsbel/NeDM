"""Chrono controller for planar pusher policies (SO-101 push-T; spec SPEC_lead_v1.md A9): the same decoder (n = 1, CPU,
float64), observation, action map and rules as the learned environment (nedm.so101_push_rl.so101_planar_env) on the MEASURED
history (step-average rates on the 20 ms grid). Episodes run with so101_goal_chrono.run_goal_episode (recorded prefix
replay up to the start frame j_a, then the controller every 20 ms).

PlanarPolicy.act(ctx) at control step t:
  geometry of the measured state (DirectGeometry, dense fingertip points); contact timer (fingertip-T signed distance
  <= touch_m, as the env); every decision_steps steps (t % 5 == 0): observation (PlanarObs) -> actor mean -> action map
  (displacement in the T frame of the measured T, rotated to the world; gripper yaw change) -> decoder.decide (yaw rate 0
  where the measured fingertip-T distance < yaw_free_gap_m); then decoder.step(measured q) -> q_cmd (between decisions the
  decoder steps open loop).
Hold / random baselines: the same controller with a fixed latent action (0: the target stays; random: N(0, std) per
decision, seeded).

measured_rules(S, ...): the env's validity rules (so101_planar_common.support_rules, every A5 rule) on the measured
states S[1:] with the env's ring-buffer semantics (TCP planar speed of the last kick_window_steps states, fingertip-T
signed distance of the last unexplained_window_steps states, current state last, front-padded with the start state S[0]
as SupportBuffers.reset does).
In Chrono only the keep-out rule is terminal (plus the Chrono facts of run_goal_episode: arm-table contact, non-finger
link contact, T height / tilt); the other rules are logged per episode (steps flagged, first flagged step).
"""
from __future__ import annotations

import numpy as np
import torch

from nedm.so101_push_rl import so101_direct_common as D
from nedm.so101_push_rl import so101_planar_common as P
from nedm.so101_push_rl.so101_planar_decoder import PlanarDecoder
from nedm.so101_push_rl.so101_planar_env import merge


def _windows(x, n):
    """x [T+1] -> [T, n] windows ending at entries 1..T (current last), front-padded with x[0]."""
    xp = torch.cat((x[:1].expand(n - 1), x))
    return xp.unfold(0, n, 1)[1:]


@torch.no_grad()
def measured_rules(S, geom, vc, kin):
    """S [T+1, 23] measured states from the start frame -> (bad [T] bool, reasons dict name -> [T] bool) for S[1:]."""
    vc = {**P.DEFAULT_VALIDITY, **(vc or {})}
    s = torch.as_tensor(np.asarray(S, float), dtype=torch.float64)
    g = geom(s)
    bufs = dict(tcp_speed=_windows(P.tcp_planar_speed(g), int(vc["kick_window_steps"])),
                finger_sd=_windows(g["finger_sd"].to(torch.float64), int(vc["unexplained_window_steps"])))
    return P.support_rules(s[1:], bufs, {k: v[1:] for k, v in g.items()}, vc, kin)


class PlanarPolicy:
    def __init__(self, name, actor, env_cfg, kin, fk, obs_builder, baseline=None, std=1.0, seed=0):
        """actor: obs [1, 139] -> latent [1, 3] (None for the baselines). baseline: None, "hold" (latent 0) or "random"
        (latent N(0, std) per decision, generator seeded with seed)."""
        self.name, self.actor, self.kin, self.fk, self.obs = name, actor, kin, fk, obs_builder
        self.cfg = cfg = merge(env_cfg)
        if not hasattr(fk, "pts_dense"):
            fk.load_dense(cfg["dense_points"])
        self.geom = D.DirectGeometry(kin, fk, dense=True)
        self.amap = P.PlanarActionMap(cfg["action"])
        self.dec_cfg = {**cfg["action"], **cfg["decoder"]}
        self.decoder = PlanarDecoder(kin, 1, torch.device("cpu"), self.dec_cfg)
        self.every = int(cfg["decoder"]["decision_steps"])
        self.centres, self.halves = fk.t_centres[:, :2].to(torch.float64), fk.t_half[:, :2].to(torch.float64)
        self.baseline, self.std, self.seed = baseline, float(std), int(seed)
        self.interventions = 0

    def reset(self, ctx):
        s = ctx["start"]
        t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64)[None]
        self.decoder.reset(torch.tensor([0]), t(s["p_line"]), torch.tensor([float(s["gripper_yaw"])], dtype=torch.float64),
                           t(s["q_des_prev"]), t(s["q_des_cur"]), torch.tensor([int(s["j_a"])]))
        self.prev_inc = t(s["prev_inc"])
        self.t_since = 10.0
        self.interventions = 0
        self.max_ik_err = 0.0                   # largest IK position error of the decoder (m), as the collector's max_ik_err_m
        self.flags = {}
        self.decisions = []
        self.targets = []
        self.gen = torch.Generator().manual_seed(self.seed)

    @torch.no_grad()
    def observation(self, ctx, h, ok, geo):
        prev = torch.as_tensor(np.asarray(ctx["prev_cmd"], float))[None]
        goal = torch.as_tensor(np.asarray(ctx["goal"], float))[None]
        H = float(ctx["horizon"])
        tl = torch.tensor([max(H - ctx["t"], 0.0)], dtype=torch.float64)
        cur = h[:, -1]
        cov = P.coverage(cur, goal, self.centres, self.halves, self.cfg["obs"]["coverage_step_m"]).to(torch.float64)
        rest = P.t_at_rest(cur, self.cfg["obs"]["rest_v_mps"], self.cfg["obs"]["rest_w_radps"])
        return self.obs(h, ok, prev, self.prev_inc, goal, tl, torch.tensor([H], dtype=torch.float64), geo, self.decoder.features(),
                        torch.tensor([self.t_since], dtype=torch.float64), rest, cov)

    @torch.no_grad()
    def latent(self, ctx, h, ok, geo):
        """Latent action [1, 3] of a decision: the actor mean, or the hold / random baseline (subclasses add noise or
        scripted decisions; nedm.so101_push.policy_rollout)."""
        if self.baseline == "hold":
            return torch.zeros(1, 3, dtype=torch.float64)
        if self.baseline == "random":
            return self.std * torch.randn(1, 3, generator=self.gen, dtype=torch.float64)
        return self.actor(self.observation(ctx, h, ok, geo)).to(torch.float64)

    @torch.no_grad()
    def act(self, ctx):
        h = torch.as_tensor(ctx["hist"][None], dtype=torch.float64)
        ok = torch.as_tensor(ctx["ok"][None])
        geo = self.geom(h[:, -1])
        touch = bool(geo["finger_sd"][0] <= self.cfg["touch_m"])
        self.t_since = 0.0 if touch else (10.0 if ctx["t"] == 0 else self.t_since + 0.02)
        if ctx["t"] % self.every == 0:
            a = self.latent(ctx, h, ok, geo)
            d_w, dyaw, d_t = self.amap(a, geo["t_yaw"])
            gap = float(self.dec_cfg.get("yaw_free_gap_m", 0.0) or 0.0)
            yb = (geo["finger_sd"] < gap) if gap > 0 else None
            self.decoder.decide(torch.tensor([0]), d_w, dyaw, yb)
            self.decisions.append(dict(t=int(ctx["t"]), a=a[0].tolist(), d_world=d_w[0].tolist(), d_t=d_t[0].tolist(), dyaw=float(dyaw[0]),
                                       yaw_block=bool(yb[0]) if yb is not None else False, finger_sd=float(geo["finger_sd"][0])))
        prev = torch.as_tensor(np.asarray(ctx["prev_cmd"], float))[None]
        cmd, info = self.decoder.step(h[:, -1, :5])
        for k, v in info.items():
            if v.dtype == torch.bool:
                self.flags[k] = self.flags.get(k, 0) + int(v[0])
        self.interventions += int(bool((info["ws_clip"] | info["dq_scaled"] | info["ik_err"])[0]))
        self.max_ik_err = max(self.max_ik_err, float(info["ik_pos_err_m"][0]))
        self.targets.append(np.r_[info["target_xy"][0].numpy(), float(info["target_yaw"][0])])
        self.prev_inc = cmd - prev
        return cmd[0].numpy()

    def summary(self):
        out = dict(decisions=len(self.decisions), interventions=int(self.interventions), flags=dict(self.flags),
                   yaw_blocked=int(sum(d["yaw_block"] for d in self.decisions)), max_ik_err_m=float(self.max_ik_err))
        if "yaw_blocked" in self.decoder.counts:
            out["yaw_blocked_decoder"] = int(self.decoder.counts["yaw_blocked"][0])
        return out
