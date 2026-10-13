"""Shared parts of the G1 environments inside the contact NRD (torch): the env config with its defaults and merge
rule, the start bank, the HOMIE preview, the path frame helpers and the plank observation.

nedm.g1_footstep_rl.footsteps_env builds the footstep task on these parts. The plank environment of the study that
first used them (stepping planks without footstep targets) is not part of this package; its definitions stay where
the footstep task reuses them: default_cfg / merge_cfg (the option names the footstep env extends), obs_names /
planks_observation (the first 99 observation values of the footstep env are the proprioception block of this
observation), path_coords, wrap and plank_index (the path frame of a plank layout, nedm.g1_footstep_rl.planks_task).

Start bank (StartBank; file artifacts/g1_footstep_rl/rl_inputs/start_bank/bank_v1.npz, training split only): a
recorded 4-state NRD history and the HOMIE memory at a control step of a recorded episode, filtered by family,
command kind, height and speed (start_filter).

State convention: 67 channels float64, rates are 20 ms step averages (nedm.g1_footstep_rl.sim). True world x, y stay in
float64 in the history; the NRD only sees re-centred histories (NrdSim.step).

Observation (OBS_NAMES, float32, clipped to +-obs.clip; no world x, y; the pure function planks_observation):
    HOMIE frame quantities of the newest state: gravity in the pelvis frame (3), pelvis angular velocity in the pelvis
    frame x 0.5 (3), 27 joint angles - default, 27 joint rates x 0.05 (averaged rates);
    HOMIE preview (12): 0.25 x clip(raw) of the frozen teacher at this state with the previous command, computed
    without changing its memory (obs.homie_preview; False: its last clipped output) = the leg offsets it would apply;
    previous action u (15); switches of the last step (2); foot centres relative to the pelvis in the heading frame
    (forward, left, z) (2 x 3); pelvis velocity in the heading frame (forward, left, up) (3); pelvis z (1);
    path frame: pelvis lateral offset l (1), pelvis yaw - path heading (1), foot centre s - pelvis s (2);
    planks: the n_ahead plank intervals [start - s_pelvis, end - s_pelvis] from the plank under or behind the
    rear-most of (pelvis, foot centres), clipped to obs.dist_clip (2 x n_ahead); per foot the index of the plank under
    its centre minus that first plank's index (-1 over a gap) (2) and the share of its 4 sphere centres over a plank (2).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from nedm.g1_footstep_rl import REPO_ROOT
from nedm.g1_footstep_rl import planks_task as P
from nedm.g1_footstep_rl.bank import FAMILIES, KINDS
from nedm.g1_footstep_rl.fk import foot_points
from nedm.g1_footstep_rl.homie import projected_gravity, world_to_body
from nedm.g1_footstep_rl.homie_policy import ACTION_CLIP, ACTION_SCALE, DEFAULT_ANGLES, N_LEG
from nedm.g1_footstep_rl.npz_rows import npz_members
from nedm.g1_footstep_rl.sim import HEADLINE, yaw

BANK = REPO_ROOT / "artifacts/g1_footstep_rl/rl_inputs/start_bank/bank_v1.npz"
ENVELOPE = REPO_ROOT / "artifacts/g1_footstep_rl/rl_inputs/start_bank/envelope_v1.json"
NUM_ACTIONS = 15


def default_cfg():
    """All env options (the json configs override any subset; merge_cfg)."""
    return dict(
        device="cpu", num_envs=64, seed=0,
        checkpoints=[str(p) for p in HEADLINE],     # NRD members (seed 61, seed 62)
        members=[0],                                # members drawn uniformly per episode
        homie_weights=None,                         # None: deploy.onnx of the vendor checkout; or a weights .npz
        bank_path=str(BANK), bank_max=0,            # bank_max > 0: keep at most this many filtered starts (tests)
        start_filter=dict(families=["nominal"], kinds=["stand", "straight", "walk"], vx=[-0.1, 1.2], vy_abs_max=0.3,
                          wz_abs_max=0.5, height_min=0.735),
        stand_share=0.3,
        episode_steps=400, auto_reset=True,
        levels=None,                                # None: planks_task.LEVELS; else {name: {"plank": [lo, hi], "gap": [lo, hi]}}
        level_weights={"L1": 1.0},
        layout=dict(n_planks=P.N_PLANKS, platform_len=P.PLATFORM_LEN, half_width=P.HALF_WIDTH, ahead=[0.05, 0.25],
                    ahead_speed_s=0.5, behind_margin=0.10),
        action=dict(vx=[-0.5, 1.2], vy=[-0.5, 0.5], wz=[-0.8, 0.8], height=0.74, alpha=0.10, clip=5.0),
        obs=dict(n_ahead=4, clip=5.0, dist_clip=[-1.0, 3.0], homie_preview=True),
        rules=dict(support="spheres", support_min_off=2, two_state=True, fall_z=0.35, fall_tilt_deg=45.0, min_z=0.55,
                   max_tilt_deg=30.0, feet_min_dist=0.03, joint_range=None, edge_margin=0.0),
        success=dict(margin=P.SUCCESS_MARGIN),
        reward=dict(progress=5.0, progress_clip=0.03, alive=0.02, success=10.0, fail=10.0, charge_per_step="auto", reach=0.0,
                    gamma=0.99, place=0.5, place_scale=0.05, min_swing_steps=P.MIN_SWING_STEPS, residual=0.02,
                    action_rate=0.02, tilt=0.05, slip=0.02, slip_cap=1.0, disagreement=0.0, disagreement_cap=0.2,
                    lateral=0.0, lateral_scale=0.10, heading=0.0, heading_scale_deg=20.0),
    )


FREE_KEYS = ("levels", "level_weights")       # dict options whose keys are not fixed


def merge_cfg(cfg):
    """default_cfg() with the given (nested, one level deep) overrides. Unknown option names raise (keys starting
    with "_" and "note" / "notes" are comments)."""
    out = default_cfg()
    for k, v in (cfg or {}).items():
        if k.startswith("_") or k in ("note", "notes"):
            continue
        if k not in out:
            raise KeyError(f"unknown planks env option {k!r}")
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in FREE_KEYS:
            bad = sorted(set(v) - set(out[k]))
            if bad:
                raise KeyError(f"unknown planks env option {k}.{bad}")
            out[k] = {**out[k], **copy.deepcopy(v)}
        else:
            out[k] = copy.deepcopy(v)
    return out


def obs_names(n_ahead=4):
    """Names of the observation entries (OBS_DIM = len)."""
    j = [f"q{i}" for i in range(27)]
    return (["grav_x", "grav_y", "grav_z", "wb_x", "wb_y", "wb_z"] + [f"{x}_rel" for x in j] + [f"d{x}" for x in j]
            + [f"homie_leg{i}" for i in range(N_LEG)] + [f"prev_u{i}" for i in range(NUM_ACTIONS)]
            + ["switch_l", "switch_r"]
            + [f"foot_{f}_{c}" for f in "lr" for c in ("fwd", "left", "z")] + ["v_fwd", "v_left", "v_up", "pelvis_z"]
            + ["path_l", "path_yaw", "foot_l_ds", "foot_r_ds"]
            + [f"plank{k}_{e}" for k in range(n_ahead) for e in ("start", "end")]
            + ["foot_l_plank", "foot_r_plank", "foot_l_support", "foot_r_support"])


OBS_NAMES = obs_names()


# ---- geometry helpers (torch, batched) -----------------------------------------------------------------------
def path_coords(xy, origin, heading):
    """World x, y [B, ..., 2] -> path (s, l) [B, ..., 2] with per-env origin [B, 2] and heading [B]."""
    sh = (-1,) + (1,) * (xy.dim() - 2)
    d = xy - origin.view(*sh, 2)
    c, s = torch.cos(heading).view(sh), torch.sin(heading).view(sh)
    return torch.stack((c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]), -1)


def plank_index(intervals, s, l, half_width):
    """Index of the plank under path points (s, l) [B, K] for per-env intervals [B, P, 2] -> [B, K] long; -1 over a
    gap, beside the path (|l| > half_width) or outside the layout (planks_task.Layout.plank_of, batched)."""
    st, en = intervals[:, None, :, 0], intervals[:, None, :, 1]
    inside = (s[..., None] >= st) & (s[..., None] <= en) & (l.abs() <= half_width)[..., None]
    k = torch.arange(intervals.shape[1], device=s.device)
    return torch.where(inside.any(-1), (inside.long() * k).sum(-1), torch.full_like(s, -1, dtype=torch.long))


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


# ---- observation (pure functions: the NRD env and the MuJoCo runner call the same code) -------------------------
def homie_preview(teacher, net, prev_cmd, state, enabled=True):
    """The HOMIE preview [B, 12] of the observation: the frozen teacher's clipped raw output for its stored history
    plus the frame of (prev_cmd [B, 4], state [B, 67], its last output), without changing its memory; enabled False:
    its last clipped output (obs.homie_preview)."""
    if enabled:
        hist = teacher.observe(prev_cmd, state)
        return net(hist.reshape(state.shape[0], -1)).clamp(-ACTION_CLIP, ACTION_CLIP)
    return teacher.last_action


def planks_observation(state, preview, prev_u, gate, intervals, origin, heading, obs_cfg, half_width):
    """The planks observation [B, OBS_DIM] float32 (module docstring, OBS_NAMES) of the newest states [B, 67]
    float64 (20 ms averaged rates): preview [B, 12] = homie_preview(...); prev_u [B, 15] the previous tanh-squashed
    action (residual part 0 in the command-only env); gate [B, 2] the contact switches (labels) of the last step;
    intervals [B, P, 2], origin [B, 2], heading [B] the layout in the path frame; obs_cfg = cfg["obs"];
    half_width = cfg["layout"]["half_width"]. Pure: it reads only its arguments."""
    st = state
    f64 = dict(dtype=torch.float64, device=st.device)
    quat = st[:, 3:7]
    default = torch.as_tensor(DEFAULT_ANGLES, **f64)
    frame = [projected_gravity(quat), 0.5 * world_to_body(quat, st[:, 37:40]), st[:, 7:34] - default, 0.05 * st[:, 40:67]]
    prev = [ACTION_SCALE * preview.to(torch.float64), prev_u, gate]
    th = yaw(st)
    c, s = torch.cos(th)[:, None], torch.sin(th)[:, None]
    n = st.shape[0]
    fp = foot_points(st)                       # foot spheres and centres of the newest states
    sl = path_coords(fp["spheres"][..., :2].reshape(n, 8, 2), origin, heading)
    idx = plank_index(intervals, sl[..., 0], sl[..., 1], half_width).view(n, 2, 4)
    centre_sl = path_coords(fp["center"][..., :2], origin, heading)
    cidx = plank_index(intervals, centre_sl[..., 0], centre_sl[..., 1], half_width)
    d = fp["center"][..., :2] - st[:, None, 0:2]
    feet = torch.stack((c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1], fp["center"][..., 2]), -1)
    v = st[:, 34:37]
    vel = torch.stack((c[:, 0] * v[:, 0] + s[:, 0] * v[:, 1], -s[:, 0] * v[:, 0] + c[:, 0] * v[:, 1], v[:, 2]), -1)
    psl = path_coords(st[:, 0:2], origin, heading)
    ps = psl[:, 0]
    path = torch.stack((psl[:, 1], wrap(th - heading)), -1)
    foot_ds = centre_sl[..., 0] - ps[:, None]
    iv = intervals
    s_ref = torch.minimum(ps, centre_sl[..., 0].amin(-1))
    k0 = ((iv[..., 0] <= s_ref[:, None]).sum(-1) - 1).clamp(min=0)
    n_ahead = int(obs_cfg["n_ahead"])
    kk = (k0[:, None] + torch.arange(n_ahead, device=st.device)[None]).clamp(max=iv.shape[1] - 1)
    win = torch.gather(iv, 1, kk[..., None].expand(-1, -1, 2)) - ps[:, None, None]
    lo, hi = obs_cfg["dist_clip"]
    win = win.clamp(lo, hi).flatten(1)
    foot_k = torch.where(cidx >= 0, cidx - k0[:, None], torch.full_like(cidx, -1)).to(torch.float64)
    support = (idx >= 0).to(torch.float64).mean(-1)
    out = torch.cat(frame + prev + [feet.flatten(1), vel, st[:, 2:3], path, foot_ds, win, foot_k, support], 1)
    clip = float(obs_cfg["clip"])
    return out.clamp(-clip, clip).to(torch.float32)


class StartBank:
    """Filtered starts of a bank file (scripts/g1_footstep_rl/build_start_bank.py), held as tensors on the device:
    hist [N, 4, 67] float64, ok [N, 4], teacher_history [N, 6, 76], teacher_last_action [N, 12], command [N, 4],
    contact_before [N, 2], stand [N] bool (command kind stand)."""

    KEYS = ("hist", "ok", "teacher_history", "teacher_last_action", "command", "contact_before")

    def __init__(self, path, flt, max_n=0, seed=0, device="cpu"):
        path = Path(path)
        meta = path.with_suffix(".json")
        if meta.exists() and "training" not in json.loads(meta.read_text()).get("split", ""):
            raise ValueError(f"{path}: not a training-split bank")
        a = npz_members(path)
        fam, kind = np.asarray(a["family"]), np.asarray(a["kind"])
        cmd, vel = np.asarray(a["command"], np.float64), np.asarray(a["vel_avg"], np.float64)
        sel = (np.isin(fam, [FAMILIES.index(f) for f in flt["families"]]) & np.isin(kind, [KINDS.index(k) for k in flt["kinds"]])
               & (cmd[:, 3] >= flt["height_min"]) & (vel[:, 0] >= flt["vx"][0]) & (vel[:, 0] <= flt["vx"][1])
               & (np.abs(vel[:, 1]) <= flt["vy_abs_max"]) & (np.abs(vel[:, 2]) <= flt["wz_abs_max"]))
        idx = np.flatnonzero(sel)
        if max_n and len(idx) > max_n:
            idx = np.sort(np.random.default_rng(seed).choice(idx, int(max_n), replace=False))
        if not len(idx):
            raise ValueError(f"{path}: no start passes the filter {flt}")
        self.index = idx
        self.n = len(idx)
        for k in self.KEYS:
            x = np.asarray(a[k][idx])
            setattr(self, k, torch.as_tensor(x, device=device))
        self.hist = self.hist.to(torch.float64)
        self.command = self.command.to(torch.float64)
        self.stand = torch.as_tensor(kind[idx] == KINDS.index("stand"), device=device)
        self.stand_ids, self.move_ids = torch.nonzero(self.stand).flatten(), torch.nonzero(~self.stand).flatten()
        self.summary = dict(path=str(path), filter=flt, starts=int(self.n), stand=int(self.stand.sum()),
                            available=int(sel.sum()))


def joint_range_from_envelope(path=ENVELOPE, margin=0.05):
    """rules.joint_range from the training-data envelope (min / max of the 27 joint angles of all training nominal
    and difficult grid states, nedm.g1_footstep_rl.sim.envelope_features)."""
    e = json.loads(Path(path).read_text())
    j = [e["features"].index(f"q{i}") for i in range(27)]
    return dict(min=[round(e["min"][i], 4) for i in j], max=[round(e["max"][i], 4) for i in j], margin=margin,
                source=str(path))
