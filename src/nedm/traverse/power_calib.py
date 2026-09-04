"""Calibrated drive-power model from kinematic state (plan open item: energy).

Shared between the fit script (numpy, recorded data) and the imagination rollout
(torch, predicted z1) so the two never disagree on the feature definition.
Acceleration is the BACKWARD difference of vx (the only one available online).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

def _primitives(z1, act, ax, xp) -> dict:
    vx, pitch, yaw_rate = z1[..., 0], z1[..., 3], z1[..., 6]
    relu = lambda a: (a + abs(a)) / 2
    fz = z1[..., 7:11].sum(-1) / 1000.0
    omega = z1[..., 11:15].mean(-1)
    sp = xp.sin(pitch)
    thr, brk = act[..., 1], act[..., 2]
    return {"one": xp.ones_like(vx), "vx": vx, "vx2": vx ** 2, "vx3": vx ** 3, "vax": vx * ax,
            "vax_p": vx * relu(ax), "vsp": vx * sp, "vsp_p": vx * relu(sp), "corner": abs(vx * yaw_rate) * vx,
            "fzv": fz * vx, "om": omega, "om2": omega * abs(omega), "thr": thr, "thrv": thr * vx,
            "throm": thr * omega, "brk": brk, "brkv": brk * vx}


_KIN = ["one", "vx", "vx2", "vx3", "vax", "vax_p", "vsp", "vsp_p", "corner", "fzv", "om", "om2"]
FEATURE_SETS = {
    "kin": _KIN,
    "kin+act": _KIN + ["thr", "thrv", "throm", "brk", "brkv"],
    "phys": ["one", "vx", "vx3", "vax_p", "vsp"],
    "phys+act": ["one", "vx", "vx3", "vax_p", "vsp", "thr", "thrv", "brkv"],
    "vx": ["one", "vx", "vx2", "vx3"],
    "act": ["one", "vx", "thr", "thrv", "brk", "brkv"],
}
KINDS = tuple(FEATURE_SETS)


def power_features(z1, act, ax, kind: str, xp):
    """z1 (..., 15), act (..., 3), ax (...): -> (..., F). ``xp`` is numpy or torch."""
    prim = _primitives(z1, act, ax, xp)
    return xp.stack([prim[n] for n in FEATURE_SETS[kind]], -1)


def backward_accel(vx: np.ndarray, dt: float) -> np.ndarray:
    ax = np.zeros_like(vx)
    ax[..., 1:] = (vx[..., 1:] - vx[..., :-1]) / dt
    return ax


class PowerModel:
    """Linear model on standardized features; ``predict`` works for numpy or torch inputs."""

    def __init__(self, kind: str, w, mu, sd):
        self.kind = kind
        self.w, self.mu, self.sd = np.asarray(w, np.float32), np.asarray(mu, np.float32), np.asarray(sd, np.float32)

    @classmethod
    def load(cls, path: Path, kind: str) -> "PowerModel":
        d = json.loads(Path(path).read_text())[kind]
        return cls(kind, d["w"], d["mu"], d["sd"])

    def predict(self, z1, act, ax, xp):
        f = power_features(z1, act, ax, self.kind, xp)
        if xp is np:
            return ((f - self.mu) / self.sd) @ self.w
        import torch
        w = torch.as_tensor(self.w, device=f.device); mu = torch.as_tensor(self.mu, device=f.device)
        sd = torch.as_tensor(self.sd, device=f.device)
        return ((f - mu) / sd) @ w
