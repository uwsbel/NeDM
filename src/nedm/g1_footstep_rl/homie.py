"""The frozen HOMIE G1 walking teacher (OpenHomie HomieDeploy/deploy.onnx) in torch, batched and device-agnostic.

The contract is that of nedm.g1_footstep_rl.homie_policy (the collector's numpy/onnxruntime teacher). This
module rebuilds the same network from the ONNX initializers, the same 76-value observation frame and the same
6-frame history, so the teacher can run inside the NRD on any torch device:

    input [B, 456] = 6 frames x 76, oldest first (all zeros after a reset)
    estimator: Linear 456->256, ELU, Linear 256->256, ELU, Linear 256->35
    v = est[:, :3] (velocity estimate), z = normalize(est[:, 3:35]) (unit latent, eps 1e-12)
    actor input [B, 111] = cat(input[:, -76:], v, z)
    actor: Linear 111->512, ELU, 512->256, ELU, 256->256, ELU, 256->12 -> raw leg output [B, 12]
    leg targets = DEFAULT_ANGLES[:12] + 0.25 x clip(raw, -100, 100)

Frame [76]: command (vx, vy, wz) x (2, 2, 0.5) (heading frame), height command (m, unscaled), pelvis angular
velocity in the PELVIS frame x 0.5, gravity direction in the pelvis frame, 27 joint angles - default, 27 joint rates
x 0.05, previous clipped output (12). The frame is computed in float64 and rounded once to float32, as
homie_policy.frame does; the network runs in float32 like onnxruntime.

Rates. In MuJoCo the collector's teacher saw instantaneous rates. In the NRD the state carries 20 ms step-average
rates (nedm.contact_nrd.data.step_average_rates), so HomieTeacher.step sees averaged rates there. A new policy that
uses this teacher must also be fed averaged rates in MuJoCo, so both worlds give identical inputs.

Weights: ``load_weights()`` reads the initializers of deploy.onnx (env nedm has the onnx package; the file's
SHA-256 is checked) or a .npz of the same arrays (artifacts/g1_footstep_rl/homie/homie_weights.npz, written by
scripts/g1_footstep_rl/fetch_homie.py).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nedm.g1_footstep_rl import REPO_ROOT
from nedm.g1_footstep_rl.homie_policy import (ACTION_CLIP, ACTION_SCALE, ANG_VEL_SCALE, COMMAND_SCALE,
                                                   DEFAULT_ANGLES, DOF_VEL_SCALE, FRAME, HISTORY, N_LEG)

ONNX_PATH = REPO_ROOT / "artifacts/g1_footstep_rl/vendor/OpenHomie/HomieDeploy/deploy.onnx"
ONNX_SHA256 = "afe6363c83c8b9ab8b1a2fed598fce7de35ff045eee1a95b336fd6720e3d3d49"   # collection config sources
ESTIMATOR = ("estimator.0", "estimator.2", "estimator.4")
ACTOR = ("actor.0", "actor.2", "actor.4", "actor.6")
SIZES = {"estimator": (HISTORY * FRAME, 256, 256, 35), "actor": (FRAME + 3 + 32, 512, 256, 256, N_LEG)}


# ---- weights -----------------------------------------------------------------------------------------------
def load_weights(path=None, sha256=ONNX_SHA256):
    """{initializer name: float32 array} of deploy.onnx (default: the vendor checkout of this repository) or of a
    .npz with the same names. For an .onnx file the SHA-256 is checked (sha256=None skips the check)."""
    path = Path(path or ONNX_PATH)
    if path.suffix == ".npz":
        with np.load(path) as f:
            return {k: f[k].astype(np.float32) for k in f.files}
    if sha256:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sha256:
            raise ValueError(f"{path}: sha256 {digest}, expected {sha256}")
    import onnx   # env nedm; only needed for .onnx files
    from onnx import numpy_helper
    graph = onnx.load(str(path)).graph
    w = {t.name: numpy_helper.to_array(t).astype(np.float32) for t in graph.initializer}
    gemm = [n for n in graph.node if n.op_type == "Gemm"]
    if len(gemm) != 7 or [n.op_type for n in graph.node if n.op_type in ("Gemm", "Elu")] != \
            ["Gemm", "Elu", "Gemm", "Elu", "Gemm", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"]:
        raise ValueError(f"{path}: not the HOMIE estimator/actor graph")
    for n in gemm:   # y = x W^T + b (the nn.Linear layout)
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        if attrs.get("alpha", 1.0) != 1.0 or attrs.get("beta", 1.0) != 1.0 or attrs.get("transB", 0) != 1:
            raise ValueError(f"{path}: Gemm attributes {attrs}")
    return w


def _mlp(sizes):
    layers = []
    for k in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[k], sizes[k + 1]))
        if k < len(sizes) - 2:
            layers.append(nn.ELU(alpha=1.0))
    return nn.Sequential(*layers)


class HomieNet(nn.Module):
    """deploy.onnx as a torch module: forward(x [B, 456]) -> raw leg output [B, 12] (no clipping)."""

    def __init__(self):
        super().__init__()
        self.estimator = _mlp(SIZES["estimator"])
        self.actor = _mlp(SIZES["actor"])

    @classmethod
    def load(cls, path=None, device="cpu"):
        """The frozen teacher network (eval mode, no parameter gradients) from deploy.onnx or a weights .npz."""
        net = cls()
        net.load_state_dict({k: torch.from_numpy(v) for k, v in load_weights(path).items()}, strict=True)
        for p in net.parameters():
            p.requires_grad_(False)
        return net.to(device).eval()

    def forward(self, x):
        est = self.estimator(x)
        z = F.normalize(est[:, 3:], dim=-1, eps=1e-12)
        return self.actor(torch.cat((x[:, -FRAME:], est[:, :3], z), -1))


# ---- observation frame ----------------------------------------------------------------------------------------
def projected_gravity(quat):
    """[B, 4] pelvis quaternions wxyz -> [B, 3] gravity direction in the pelvis frame (homie_policy.projected_gravity:
    the quaternion is used as given, not re-normalised)."""
    w, x, y, z = quat.unbind(-1)
    return -torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)


def world_to_body(quat, v_world):
    """Rotate world-frame vectors [B, 3] into the pelvis frame with the (normalised) quaternions [B, 4] wxyz:
    v_body = R(q)^T v_world (the inverse of MuJoCo's mju_rotVecQuat body -> world)."""
    q = quat / quat.norm(dim=-1, keepdim=True)
    w, u = q[..., :1], -q[..., 1:]                       # the conjugate quaternion rotates world -> body
    t = 2 * torch.cross(u, v_world, dim=-1)
    return v_world + w * t + torch.cross(u, t, dim=-1)


def homie_frame(command, quat, ang_vel_body, q, dq, last_action):
    """One observation frame [B, 76] float32, as homie_policy.frame: command [B, 4] (vx, vy, wz in the heading
    frame, height), quat [B, 4] wxyz, ang_vel_body [B, 3] (pelvis frame), q, dq [B, 27], last_action [B, 12] (the
    previous clipped output). Computed in float64, rounded once to float32."""
    f64 = torch.float64
    command = command.to(f64)
    scale = torch.as_tensor(COMMAND_SCALE, dtype=f64, device=command.device)
    default = torch.as_tensor(DEFAULT_ANGLES, dtype=f64, device=command.device)
    return torch.cat((command[:, :3] * scale, command[:, 3:4], ang_vel_body.to(f64) * ANG_VEL_SCALE,
                      projected_gravity(quat.to(f64)), q.to(f64) - default, dq.to(f64) * DOF_VEL_SCALE,
                      last_action.to(f64)), -1).to(torch.float32)


def state_frame(command, state, last_action):
    """homie_frame of 67-channel NRD states [B, 67]: the world angular velocity (channels 37-39) is rotated into
    the pelvis frame; the joint rates are the state's rates (20 ms averages in the NRD)."""
    quat = state[:, 3:7]
    return homie_frame(command, quat, world_to_body(quat, state[:, 37:40]), state[:, 7:34], state[:, 40:67],
                       last_action)


def leg_targets(act):
    """Clipped output [B, 12] -> leg joint position targets [B, 12] float64 (default + 0.25 x output)."""
    default = torch.as_tensor(DEFAULT_ANGLES[:N_LEG], dtype=torch.float64, device=act.device)
    return default + (ACTION_SCALE * act).to(torch.float64)


# ---- batched teacher -----------------------------------------------------------------------------------------
class HomieTeacher:
    """The teacher with its memory for B environments: history [B, 6, 76] float32 (oldest first) and last_action
    [B, 12] float32 (the previous clipped output). Call step() exactly once per 20 ms control step, also when its
    output is not used, or its history goes stale.

    No in-place writes on the memory tensors (step and reset build new tensors), so a step can sit inside an autograd
    graph (gradients to the state through the frame) when it runs with gradients enabled."""

    def __init__(self, batch, device="cpu", net=None, weights=None):
        self.device = torch.device(device)
        self.net = net if net is not None else HomieNet.load(weights, self.device)
        self.batch = int(batch)
        self.history = torch.zeros(self.batch, HISTORY, FRAME, dtype=torch.float32, device=self.device)
        self.last_action = torch.zeros(self.batch, N_LEG, dtype=torch.float32, device=self.device)

    def _mask(self, env_ids):
        """env_ids (None = all, bool mask [B], or indices) -> bool mask [B]."""
        if env_ids is None:
            return torch.ones(self.batch, dtype=torch.bool, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        if env_ids.dtype == torch.bool:
            return env_ids
        return torch.zeros(self.batch, dtype=torch.bool, device=self.device).index_fill(0, env_ids.long(), True)

    def reset(self, env_ids=None, history=None, last_action=None):
        """Reset the memory of env_ids (None = all; bool mask or indices): zeros (the deploy reset), or the given
        history [n, 6, 76] and last_action [n, 12] (n = number of chosen environments, in index order)."""
        m = self._mask(env_ids)
        n = int(m.sum())
        h = torch.zeros(n, HISTORY, FRAME, dtype=torch.float32, device=self.device) if history is None else \
            torch.as_tensor(history, dtype=torch.float32, device=self.device)
        a = torch.zeros(n, N_LEG, dtype=torch.float32, device=self.device) if last_action is None else \
            torch.as_tensor(last_action, dtype=torch.float32, device=self.device)
        hist, last = self.history.clone(), self.last_action.clone()
        hist[m], last[m] = h, a
        self.history, self.last_action = hist, last

    def observe(self, command, state):
        """The history with the frame of this state appended [B, 6, 76], without changing the memory."""
        f = state_frame(command, state, self.last_action)
        return torch.cat((self.history[:, 1:], f[:, None]), 1)

    def step(self, command, state):
        """command [B, 4] (vx, vy, wz, height), state [B, 67] (NRD layout, world angular velocity) ->
        (raw output [B, 12] float32, leg targets [B, 12] float64). Appends the frame and stores the clipped output."""
        hist = self.observe(command, state)
        raw = self.net(hist.reshape(len(hist), -1))
        act = raw.clamp(-ACTION_CLIP, ACTION_CLIP)
        self.history, self.last_action = hist, act
        return raw, leg_targets(act)

    def state_dict(self):
        """The memory (for snapshots of an environment batch)."""
        return {"history": self.history.clone(), "last_action": self.last_action.clone()}

    def load_state_dict(self, sd):
        self.history, self.last_action = sd["history"].clone(), sd["last_action"].clone()
