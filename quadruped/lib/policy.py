"""The rl_sar robot_lab actor, wired to a Chrono Go2.

Everything about this policy that the pipeline depends on is ASSERTED at load, because a
checkpoint that quietly differs from its config reads exactly like a working one. The
previous policy's sign convention was inherited without a source and a sign-flip bug
followed; this one refuses to run until the convention has been established by test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

PARAMS = Path(__file__).resolve().parents[1] / "params"


class PolicyGateFailed(Exception):
    pass


def _cfg() -> dict:
    return yaml.safe_load((PARAMS / "policy.yaml").read_text())


class Go2Policy:
    """Memoryless MLP actor. Pure function of the observation it is handed.

    The 45-vector has NO base linear velocity block, which is what makes it a 45 and not
    the 48 of the legged_gym base config. That is a feature here: a policy that never had
    base linear velocity is one we can supply honestly from a simulator.
    """

    OBS_BLOCKS = ("ang_vel", "gravity", "commands", "dof_pos", "dof_vel", "actions")

    def __init__(self, ckpt: str | Path, cfg: dict | None = None, device: str = "cpu"):
        import torch

        self.torch = torch
        self.cfg = cfg or _cfg()
        self.device = device

        sign = self.cfg.get("sign", {}).get("value")
        if sign is None:
            raise PolicyGateFailed(
                "params/policy.yaml has sign.value unset. The joint sign convention must "
                "be established by the action round-trip test against Chrono before this "
                "policy may drive a collection. The previous harness inherited a negation "
                "with no source recording why, and a sign-flip bug followed."
            )
        self.sign = float(sign)

        self.net = torch.jit.load(str(ckpt), map_location=device)
        self.net.eval()
        self._gate(ckpt)

        o = self.cfg["observation"]
        self.obs_dim = int(o["dim"])
        self.clip_obs = float(o["clip"])
        s = o["scales"]
        self.s_ang, self.s_lin = float(s["ang_vel"]), float(s["lin_vel"])
        self.s_qpos, self.s_qvel = float(s["dof_pos"]), float(s["dof_vel"])
        # COMMANDS USE commands_scale, NOT lin_vel_scale/ang_vel_scale. Those two scale
        # the OBSERVED base velocity, and this policy has no base-linear-velocity term at
        # all -- that is what makes its observation 45 wide and not 48. Applying
        # lin_vel_scale here doubles the command: measured, a 0.5 m/s request produced
        # 0.989 m/s achieved, which is exactly the factor of 2.0.
        self.s_cmd = np.asarray(s.get("commands", [1.0, 1.0, 1.0]), dtype=np.float32)

        j = self.cfg["joints"]
        self.p2c = np.asarray(j["policy_to_chrono"], dtype=np.int64)
        self.default_pos = np.asarray(j["default_pos"], dtype=np.float32)

        a = self.cfg["action"]
        self.act_scale = np.asarray(a["scale"], dtype=np.float32)
        self.act_clip = tuple(a["clip"])

        self.last_actions = np.zeros(12, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)

    # ---------------------------------------------------------------- gates
    def _gate(self, ckpt):
        """Refuse anything that is not the architecture we chose this policy for."""
        sd = self.net.state_dict()
        arch = self.cfg["architecture"]

        banned = ("encoder", "estimator", "him", "latent", "lstm", "gru", "rnn", "memory")
        found = [k for k in sd if any(b in k.lower() for b in banned)]
        if found:
            raise PolicyGateFailed(
                f"{ckpt} carries {found}, so it is not a plain actor. A history encoder "
                f"or estimator would be co-adapted with the policy during fine-tuning and "
                f"no result could be attributed to either."
            )

        want = int(arch.get("state_dict_entries", 0))
        if want and len(sd) != want:
            raise PolicyGateFailed(
                f"{ckpt} has {len(sd)} state_dict entries, config says {want}. A "
                f"different checkpoint than the one this pipeline was specified against."
            )

        if self.cfg["observation"].get("history"):
            raise PolicyGateFailed(
                "observation.history is non-empty. This pipeline requires a memoryless "
                "policy: a branch rollout would otherwise have to seed the history from "
                "the corpus, and the first steps of every branch would be mis-conditioned."
            )

        in_dim = None
        for k, v in sd.items():
            if k.endswith("0.weight") and v.dim() == 2:
                in_dim = int(v.shape[1])
                break
        if in_dim is not None and in_dim != int(self.cfg["observation"]["dim"]):
            raise PolicyGateFailed(
                f"first layer takes {in_dim} inputs, config declares "
                f"{self.cfg['observation']['dim']}"
            )

    def assert_stateless(self, rng=None):
        """Same input twice must give the same output, with an unrelated call between.

        Verified rather than trusted: this is the single property the whole gradient path
        depends on, and it is cheap to check.
        """
        torch = self.torch
        rng = rng or np.random.default_rng(0)
        x = torch.as_tensor(rng.normal(size=(1, self.obs_dim)), dtype=torch.float32)
        with torch.no_grad():
            a = self.net(x)
            _ = self.net(torch.randn(1, self.obs_dim))
            b = self.net(x)
        if not torch.equal(a, b):
            raise PolicyGateFailed(
                "policy is NOT stateless: an interleaved call changed its output."
            )
        return True

    # ---------------------------------------------------------- observation
    def observe(self, robot) -> np.ndarray:
        """Assemble the 45-vector in rl_sar block order, scales and joint order."""
        from ..params import transforms as T  # noqa: PLC0415

        base = robot.base()
        w = base.GetAngVelLocal()
        ang = np.array([w.x, w.y, w.z], dtype=np.float32) * self.s_ang

        r = base.GetRot()
        grav = T.projected_gravity(r.e0, r.e1, r.e2, r.e3).astype(np.float32)

        cmd = (self.command * self.s_cmd).astype(np.float32)

        q = self.sign * robot.joint_pos().astype(np.float32)[self.p2c]
        qd = self.sign * robot.joint_vel().astype(np.float32)[self.p2c]
        dof_pos = (q - self.default_pos) * self.s_qpos
        dof_vel = qd * self.s_qvel

        obs = np.concatenate([ang, grav, cmd, dof_pos, dof_vel, self.last_actions])
        obs = np.clip(obs, -self.clip_obs, self.clip_obs).astype(np.float32)
        if obs.shape[0] != self.obs_dim:
            raise PolicyGateFailed(f"built a {obs.shape[0]}-vector, need {self.obs_dim}")
        return obs

    # --------------------------------------------------------------- action
    def act(self, robot) -> np.ndarray:
        """Return joint targets in CHRONO order, ready for robot.actuate()."""
        torch = self.torch
        obs = self.observe(robot)
        with torch.no_grad():
            raw = self.net(torch.as_tensor(obs).unsqueeze(0)).squeeze(0).numpy()
        raw = np.clip(raw, *self.act_clip).astype(np.float32)
        self.last_actions = raw.copy()

        targets_policy = self.default_pos + raw * self.act_scale
        out = np.empty(12, dtype=np.float32)
        out[self.p2c] = targets_policy          # policy order -> chrono order
        return self.sign * out

    def reset(self):
        """Only the action memory; the network itself carries no state."""
        self.last_actions = np.zeros(12, dtype=np.float32)
