"""The state contract shared by the NRD world and the MuJoCo worlds of the SO101 + cube policy study.

Both worlds take 6 absolute joint targets per world (5 arm joints + jaw, rad, held for one 20 ms control step) and
return a ``WorldState``. Every field has the definition of the grasp NRD data (``nedm.contact_nrd.collect.so101_grasp``):
rates are 20 ms step averages, the cube velocity and spin are in the world frame. The cube orientation is the TRUE
one, not the NRD's canonical relabelling of the cube's symmetry.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

CONTROL_DT = 0.02
JOINTS = 6
CONTACT_PAIRS = ("fixed_jaw_cube", "moving_jaw_cube", "cube_table")


@dataclass
class WorldState:
    q: torch.Tensor           # [B, 6] joint angles (rad); index 5 = jaw
    qd: torch.Tensor          # [B, 6] 20 ms step-average joint rates (rad/s)
    cube_pos: torch.Tensor    # [B, 3] cube centre, world (m); table top at z = 0
    cube_quat: torch.Tensor   # [B, 4] true cube orientation, w x y z
    cube_v: torch.Tensor      # [B, 3] cube velocity, world (m/s), data definition
    cube_w: torch.Tensor      # [B, 3] cube spin, world (rad/s), data definition
    contact: torch.Tensor     # [B, 3] bool, order CONTACT_PAIRS
    arm_table: torch.Tensor   # [B] bool, arm touches the table (MuJoCo: real contact; NRD: lowest-point guard)
    invalid: torch.Tensor     # [B] bool, any other end condition (non-finite state, cube off table, ...)

    def index(self, idx) -> "WorldState":
        return WorldState(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    def to(self, device=None, dtype=None) -> "WorldState":
        out = {}
        for f in fields(self):
            t = getattr(self, f.name)
            out[f.name] = t.to(device=device, dtype=dtype) if (dtype is not None and t.is_floating_point()) else t.to(device)
        return WorldState(**out)

    @property
    def batch(self) -> int:
        return self.q.shape[0]

    @staticmethod
    def cat(states: list["WorldState"]) -> "WorldState":
        return WorldState(**{f.name: torch.cat([getattr(s, f.name) for s in states]) for f in fields(WorldState)})
