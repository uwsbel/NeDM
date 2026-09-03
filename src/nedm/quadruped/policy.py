"""The policy and its observation contract.

The observation is built by calling the training harness's own
_compute_observations unmodified, so the leg permutation, the sign negation, the
rest-pose subtraction and the four scale factors are inherited rather than
reimplemented. See _harness_observer."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

from .constants import (ANG_VEL_SCALE, CHRONO_TO_POLICY, DOF_POS_SCALE,
                          DOF_VEL_SCALE, LIN_VEL_SCALE, POLICY_DEFAULTS)
from .robot import Go2Robot


def _try_config(root):
    try:
        if str(root / "simulation") not in sys.path:
            sys.path.append(str(root / "simulation"))
        import Config
        return Config
    except Exception:  # noqa: BLE001
        return None


def _harness_observer(torch):
    """Bind chrono_crmenv's own _compute_observations, or return (None, None).

    The harness file is imported BYTE-IDENTICAL -- chrono_crm_compat's
    install_display_stubs satisfies its display-only imports from outside rather
    than patching it, so the file that defines the checkpoint's input contract is
    provably the original. What comes back is its unbound method plus a
    duck-typed object carrying exactly the thirteen attributes that method reads.
    """
    from nedm import chrono_crm_compat as crm_compat

    root = Path("/home/kyle/Documents/sbel/sbel-reproducibility/2025/multi-terrain-RL")
    if not (root / "rl_examples/rslrl/chrono_crmenv.py").is_file():
        return None, None
    crm_compat.install_display_stubs()
    for extra in (root, root / "rl_examples/rslrl"):
        if str(extra) not in sys.path:
            sys.path.append(str(extra))
    try:
        import chrono_crmenv
    except Exception:  # noqa: BLE001
        return None, None

    Config = _try_config(root)
    cfg = Config.Go2Config() if Config is not None else None

    class _Adapter:
        pass

    a = _Adapter()
    a.num_envs = 1
    a.device = "cpu"
    a.obs_buf = torch.zeros(1, 45, dtype=torch.float32)
    a.dof_pos = torch.zeros(1, 12, dtype=torch.float32)
    a.dof_vel = torch.zeros(1, 12, dtype=torch.float32)
    a.actions = torch.zeros(1, 12, dtype=torch.float32)
    for name in ("base_ang_vel", "base_lin_vel", "projected_gravity"):
        setattr(a, name, torch.zeros(1, 3, dtype=torch.float32))
    # Scales taken from the harness's own Config where available, so even these
    # four numbers are inherited rather than retyped.
    a.lin_vel_scale = getattr(cfg, "lin_vel_scale", LIN_VEL_SCALE)
    a.ang_vel_scale = getattr(cfg, "ang_vel_scale", ANG_VEL_SCALE)
    a.dof_pos_scale = getattr(cfg, "dof_pos_scale", DOF_POS_SCALE)
    a.dof_vel_scale = getattr(cfg, "dof_vel_scale", DOF_VEL_SCALE)
    return chrono_crmenv.ChronoQuadrupedEnv._compute_observations, a


class PolicyController:
    """model_2999.pt, the CRM-finetuned checkpoint. 45 obs in, 12 actions out."""

    def __init__(self, ckpt: Path, cfgs: Path, harness: bool = True):
        import torch
        import torch.nn as nn

        self.torch = torch
        self._harness_obs = None
        self._harness_adapter = None
        if harness:
            self._harness_obs, self._harness_adapter = _harness_observer(torch)
        with cfgs.open("rb") as fh:
            self.env_cfg, self.train_cfg = pickle.load(fh)
        hidden = self.train_cfg["policy"]["actor_hidden_dims"]

        layers, in_dim = [], 45
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 12))
        self.actor = nn.Sequential(*layers)

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.actor.load_state_dict({
            k.removeprefix("actor."): v
            for k, v in state["model_state_dict"].items() if k.startswith("actor.")
        })
        self.actor.eval()
        self.last_actions = np.zeros(12, dtype=np.float32)
        # Hardcoded, NOT env_cfg['target_lin_vel']. See docstring note 4.
        self.command = np.array([0.5, 0.0, 0.0], dtype=np.float32) * LIN_VEL_SCALE

    @staticmethod
    def _projected_gravity(q) -> np.ndarray:
        qw, qx, qy, qz = q.e0, q.e1, q.e2, q.e3
        return np.array([-2 * (qx * qz - qw * qy),
                         -2 * (qy * qz + qw * qx),
                         -(1 - 2 * (qx * qx + qy * qy))], dtype=np.float32)

    def observe(self, robot: Go2Robot) -> np.ndarray:
        """Build the 45-wide observation the checkpoint expects.

        Two implementations. The default DELEGATES to the training harness's own
        _compute_observations, called unmodified on a duck-typed adapter, so the
        reorder, the sign negation, the rest-pose subtraction and the four scale
        factors are INHERITED rather than reimplemented. The sign negation in
        particular has no recorded justification anywhere in the source we have,
        so not maintaining it by hand is the point.

        THE MATH IS THEIRS; THE PLUMBING IS OURS. Populating thirteen named
        attributes is much less to get wrong than reproducing a permutation, a
        sign convention and four scales, but it is not nothing -- which is why
        the port was gated bit-exactly against the hand-written path before that
        path was removed (max abs diff 0.0 over 45 elements on random input, and
        travel/tilt/base_z reproduced to the digit on a bit-reproducible run).

        THERE IS NO FALLBACK, DELIBERATELY. A hand-written second implementation
        used to stand behind this and was selected whenever the harness failed to
        import. That is a silent degradation: the guarantee "the conventions are
        inherited" would have quietly become "inherited if an import happened to
        succeed", with no signal at the point it stopped being true. Refusing is
        the same call as the render-options guard above.
        """
        if self._harness_obs is None:
            raise RuntimeError(
                "The training harness's _compute_observations is unavailable, so "
                "the checkpoint's input convention cannot be inherited. Refusing "
                "to fall back to a hand-written reimplementation -- the joint "
                "sign negation has no recorded justification anywhere in the "
                "source, so a local copy of it cannot be verified against "
                "anything. Make chrono_crmenv.py importable; see "
                "docs/state/decisions/reuse-chrono-crmenv.md.")
        return self._observe_via_harness(robot)

    def _observe_via_harness(self, robot: Go2Robot) -> np.ndarray:
        torch = self.torch
        base = robot.base()
        w = base.GetAngVelLocal()
        a = self._harness_adapter
        # dof_pos is filled ABSOLUTE here, exactly as the harness fills it at
        # chrono_crmenv.py:413 from get_joint_pos(). The rest-pose subtraction is
        # a LOCAL inside _compute_observations (line 495), not an attribute, which
        # is why it does not appear in the adapter's field list.
        a.dof_pos[0, :] = torch.from_numpy(robot.joint_pos().astype("float32"))
        a.dof_vel[0, :] = torch.from_numpy(robot.joint_vel().astype("float32"))
        a.base_ang_vel[0, :] = torch.tensor([w.x, w.y, w.z], dtype=torch.float32)
        a.projected_gravity[0, :] = torch.from_numpy(self._projected_gravity(base.GetRot()))
        a.actions[0, :] = torch.from_numpy(self.last_actions)
        self._harness_obs(a)          # their code, unmodified, mutates a.obs_buf
        return a.obs_buf[0].numpy().astype(np.float32)

    def act(self, robot: Go2Robot) -> np.ndarray:
        obs = self.torch.from_numpy(self.observe(robot)).unsqueeze(0)
        with self.torch.no_grad():
            action = self.actor(obs).squeeze(0).numpy().astype(np.float32)
        self.last_actions = action
        targets_policy_frame = action * 0.25 + POLICY_DEFAULTS
        return -targets_policy_frame[CHRONO_TO_POLICY].astype(np.float64)

