"""Views of a unified model in each earlier study's state layout, so it is scored
by exactly the same metric code (and the same episodes) as the earlier models.

Pool: [A x, y, vx, vy, wx, wy, wz, B same] (14). Ball: [x, z, vx, vz, omega_y] (5).
"""
from __future__ import annotations

import torch
from torch import nn


class PoolView(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dt = model.dt
        self.radius = float(model.system["bodies"][0]["radius"])
        # Version-5 models and their ensembles have no state_mean; it only carries dtype and device here.
        self.state_mean = getattr(model, "state_mean", None)
        if self.state_mean is None:
            self.state_mean = torch.zeros(9, dtype=torch.float64, device=next(model.parameters()).device)

    def to_unified(self, s14):
        out = torch.zeros(*s14.shape[:-1], 2, 9, dtype=s14.dtype, device=s14.device)
        for b in (0, 1):
            x = s14[..., 7 * b:7 * b + 7]
            out[..., b, 0:2] = x[..., 0:2]
            out[..., b, 2] = self.radius
            out[..., b, 3:5] = x[..., 2:4]
            out[..., b, 6:9] = x[..., 4:7]
        return out

    @staticmethod
    def to_pool(u):
        return torch.cat([torch.cat((u[..., b, 0:2], u[..., b, 3:5], u[..., b, 6:9]), -1) for b in (0, 1)], -1)

    def rollout(self, initial, steps):
        return self.to_pool(self.model.rollout(self.to_unified(initial), steps)).to(initial.dtype)


class BallView(nn.Module):
    contact_enabled = False

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dt = model.dt
        self.config = {}

    @staticmethod
    def to_unified(s5):
        out = torch.zeros(*s5.shape[:-1], 1, 9, dtype=s5.dtype, device=s5.device)
        out[..., 0, 0], out[..., 0, 2], out[..., 0, 3], out[..., 0, 5], out[..., 0, 7] = s5.unbind(-1)
        return out

    @staticmethod
    def to_ball(u):
        return u[..., 0, [0, 2, 3, 5, 7]]

    def rollout(self, initial, steps):
        return self.to_ball(self.model.rollout(self.to_unified(initial), steps)).to(initial.dtype)
