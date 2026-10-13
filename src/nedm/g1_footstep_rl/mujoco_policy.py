"""Shared parts of the MuJoCo evaluation of policies trained inside the G1 contact NRD: the run files next to a
checkpoint and the mapping of their data paths to this machine, the action mapping of the NRD env (ActionMap), a start
bank made of MuJoCo hand-over snapshots (SnapshotBank) and the statistics of the evaluation tables.

The released runs record cluster paths in env_cfg.json and manifest.json. resolve() maps them to the released copies
under artifacts/ (NRD checkpoints and HOMIE weights) and checks their SHA-256 against the run manifest.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from nedm.g1_footstep_rl import REPO_ROOT
from nedm.g1_footstep_rl.homie_policy import DEFAULT_ANGLES, N_LEG
from nedm.g1_footstep_rl.planks_env import NUM_ACTIONS, merge_cfg
from nedm.g1_footstep_rl.sim import HEADLINE

F64 = torch.float64
LOCAL_HOMIE_NPZ = REPO_ROOT / "artifacts/g1_footstep_rl/homie/homie_weights.npz"
LOCAL_NRD = {f"full_g1_ridge01_s{s}/best.pt": p for s, p in zip((61, 62), HEADLINE)}   # run name -> released copy


# ---- files -------------------------------------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def resolve(path, want_sha=None, kind="file"):
    """A data path of a run config on this machine: the path itself if it exists, else the released copy of a
    cluster file (NRD checkpoint full_g1_ridge01_s<seed>/best.pt -> LOCAL_NRD, HOMIE weights homie_weights.npz ->
    LOCAL_HOMIE_NPZ). want_sha (the run's manifest) is checked when given."""
    if path is None:
        return None
    p = Path(path)
    cands = [p]
    if kind == "checkpoint" and f"{p.parent.name}/{p.name}" in LOCAL_NRD:
        cands.append(LOCAL_NRD[f"{p.parent.name}/{p.name}"])
    elif kind == "homie" and p.name == LOCAL_HOMIE_NPZ.name:
        cands.append(LOCAL_HOMIE_NPZ)
    for c in cands:
        if c.exists():
            if want_sha and sha256(c) != want_sha:
                raise ValueError(f"{c}: sha256 differs from the run manifest ({path})")
            return str(c)
    raise FileNotFoundError(f"{path}: not found here (tried {[str(c) for c in cands]})")


def run_files(ckpt):
    """(env_cfg, train_cfg, manifest or {}) saved next to an RSL-RL checkpoint by train_footsteps_ppo.py."""
    d = Path(ckpt).resolve().parent
    rd = lambda n: json.loads((d / n).read_text()) if (d / n).exists() else None
    ec, tc = rd("env_cfg.json"), rd("train_cfg.json")
    if ec is None or tc is None:
        raise FileNotFoundError(f"{d}: env_cfg.json and train_cfg.json must sit next to the checkpoint")
    return ec, tc, rd("manifest.json") or {}


# ---- action mapping of the NRD env ---------------------------------------------------------------------------------
class ActionMap:
    """The NRD env's action mapping for one env config: a [B, n] (n = 3 if action.alpha == 0 else 15) ->
    clipped to +-action.clip, zero-padded to 15, u = tanh(a); command = lo + (hi - lo) (u[:3] + 1) / 2 with the height;
    target = cat(teacher legs + alpha u[3:], upper body default)."""

    def __init__(self, env_cfg):
        ac = merge_cfg(env_cfg)["action"]
        self.lo = torch.tensor([ac["vx"][0], ac["vy"][0], ac["wz"][0]], dtype=F64)
        self.hi = torch.tensor([ac["vx"][1], ac["vy"][1], ac["wz"][1]], dtype=F64)
        self.alpha, self.clip, self.height = float(ac["alpha"]), float(ac["clip"]), float(ac["height"])
        self.num_actions = 3 if self.alpha == 0 else NUM_ACTIONS
        self.upper = torch.as_tensor(DEFAULT_ANGLES[N_LEG:], dtype=F64)

    def squash(self, a):
        a = torch.as_tensor(a).to(F64).clamp(-self.clip, self.clip)
        if a.shape[1] != self.num_actions:
            raise ValueError(f"actions [B, {a.shape[1]}], the policy has {self.num_actions}")
        if self.num_actions < NUM_ACTIONS:
            a = torch.cat((a, torch.zeros(a.shape[0], NUM_ACTIONS - self.num_actions, dtype=F64)), 1)
        return torch.tanh(a)

    def command(self, u):
        cmd3 = self.lo + (self.hi - self.lo) * (u[:, :3] + 1) / 2
        return torch.cat((cmd3, torch.full((u.shape[0], 1), self.height, dtype=F64)), 1)

    def target(self, legs, u):
        return torch.cat((legs + self.alpha * u[:, 3:], self.upper.expand(u.shape[0], -1)), 1)

    def action_of_command(self, cmd3):
        """The env's action_of_command: pre-tanh actions [B, num_actions] of commands [B, 3], residual 0."""
        c = torch.as_tensor(cmd3, dtype=F64)
        u = (2 * (c - self.lo) / (self.hi - self.lo) - 1).clamp(-0.999999, 0.999999)
        return torch.cat((torch.atanh(u), torch.zeros(c.shape[0], self.num_actions - 3, dtype=F64)), 1)


# ---- the same episodes inside the NRD ------------------------------------------------------------------------------
class SnapshotBank:
    """Duck-typed planks_env.StartBank whose starts are MuJoCo hand-over snapshots (all 'stand' starts)."""

    def __init__(self, snaps, device="cpu"):
        t = lambda k, dt=F64: torch.as_tensor(np.stack([s[k] for s in snaps]), device=device).to(dt)
        self.n = len(snaps)
        self.hist = t("hist")
        self.ok = torch.ones(self.n, 4, dtype=torch.bool, device=device)
        self.teacher_history = t("teacher_history", torch.float32)
        self.teacher_last_action = t("teacher_last_action", torch.float32)
        self.command = t("prev_cmd")
        self.contact_before = t("gate") > 0.5
        self.stand = torch.ones(self.n, dtype=torch.bool, device=device)
        self.stand_ids = torch.arange(self.n, device=device)
        self.move_ids = torch.zeros(0, dtype=torch.long, device=device)
        self.index = np.arange(self.n)
        self.summary = dict(path="mujoco hand-over snapshots", starts=self.n, stand=self.n, available=self.n)


# ---- statistics ----------------------------------------------------------------------------------------------------
def wilson(k, n, z=1.959964):
    """Wilson 95 % interval of k / n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def phi(x, y):
    """Phi coefficient of two boolean sequences (None if a margin is empty)."""
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    n11, n00 = int((x & y).sum()), int((~x & ~y).sum())
    n10, n01 = int((x & ~y).sum()), int((~x & y).sum())
    den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return None if den == 0 else (n11 * n00 - n10 * n01) / den
