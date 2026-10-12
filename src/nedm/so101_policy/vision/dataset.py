"""Demo states -> training batches of the camera student, with the images drawn on the fly by the batch renderer.

Inputs of one sample (decision k of an episode; ``demos`` module docstring for the stored keys):

| input | shape | content |
|---|---|---|
| images | [C, 3, H, W] | ``CAMERAS`` (top, wrist) drawn from (q, cube pose with the TRUE orientation, goal plate) |
| state ("proprio") | [30] | measured joint angles q (6) + ``DecoderV2.observation()`` (24): known to the robot's own controller |
| goal | [4] | goal cube centre x, y (m), sin 4 psi, cos 4 psi (psi = goal yaw; the cube's 90 deg symmetry) |
| target | [chunk, 6] | the expert's decoder actions of decisions k .. k + chunk - 1 (after the episode end: padded, masked) |

Cameras (written down here, build4_vision.md sec. 2): ``top`` lowered to 0.48 m above the table over (0.19, 0) and the
MJCF wrist camera, 288 x 288 px both (one renderer has one resolution and one 61 deg field of view for all its cameras;
non-square Madrona images were not centred in a probe, so square only). The top view covers |y| <= 0.283 m at the
table (the annulus out to 75 deg plus the cube's half diagonal); a 30 mm cube on the table is 15 px wide there
(focal 244.5 px; measured median 15.5 px with the 384 x 256 probe at 0.40 m). The arm's wrist reaches z = 0.32 m in the
gate runs, so the camera stays 16 cm above it. The goal plate (``SO101Renderer`` goal "plate") is drawn too.

Images enter the network as float, ImageNet mean / std normalised (the ResNet18 backbone is ImageNet-pretrained).
Augmentation (``shift_px``): DrQ-style random shift of each image by up to +-shift_px pixels (edge padding), the same
shift for all cameras of a sample is NOT assumed (independent per camera); 0 = off.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nedm.so101_policy.vision import demos as D

CAMERAS = ("top", "wrist")
RES = 288
TOP_EYE = (0.19, 0.0, 0.48)
TOP_TARGET = (0.19, 0.0, 0.0)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
STATE_DIM, GOAL_DIM, ACTION_DIM = 30, 4, 6
PRIV_DIM = 5          # privileged cube state (ablation): cube centre x, y, z (m), sin 4 yaw, cos 4 yaw


def cube_vec(cube_pos, cube_quat):
    """PRIVILEGED (ablation only: a camera policy must not see it): [n, 5] float32 = cube x, y, z, sin 4 yaw, cos 4 yaw
    (yaw = ``task.cube_yaw_tilt``)."""
    from nedm.so101_policy.task import cube_yaw_tilt
    yaw, _ = cube_yaw_tilt(torch.as_tensor(cube_quat).to(torch.float64))
    return torch.cat((torch.as_tensor(cube_pos).to(yaw), torch.sin(4 * yaw)[:, None], torch.cos(4 * yaw)[:, None]), -1).float()


def goal_vec(goal_xy, goal_yaw):
    """[n, 2], [n] -> [n, 4] float32: x, y, sin 4 psi, cos 4 psi."""
    gy = torch.as_tensor(goal_yaw, dtype=torch.float64)
    gxy = torch.as_tensor(goal_xy, dtype=torch.float64)
    return torch.cat((gxy, torch.sin(4 * gy)[:, None], torch.cos(4 * gy)[:, None]), -1).float()


def proprio(q, dec_obs):
    """[n, 6], [n, 24] -> [n, 30] float32."""
    return torch.cat((torch.as_tensor(q).float(), torch.as_tensor(dec_obs).float()), -1)


class Views:
    """The student's cameras: ``SO101Renderer`` (Madrona by default; NVIDIA, one per process) for up to num_worlds states
    per call. ``render(q, cube_pos, cube_quat, goal_pose[n, 3]) -> uint8 [n, C, H, W, 3]`` on cuda."""

    def __init__(self, num_worlds, res=RES, cameras=CAMERAS, backend="madrona", top_eye=TOP_EYE, top_target=TOP_TARGET,
                 shadows=True, goal="plate"):
        from nedm.render.so101 import SO101Renderer
        self.r = SO101Renderer(int(num_worlds), width=int(res), height=int(res), backend=backend, cameras=tuple(cameras),
                               top_eye=tuple(top_eye), top_target=tuple(top_target), shadows=shadows, goal=goal)
        self.num_worlds, self.res, self.cameras = int(num_worlds), int(res), tuple(cameras)
        self.calls, self.views, self.seconds = 0, 0, 0.0

    @torch.no_grad()
    def render(self, q, cube_pos, cube_quat, goal_pose):
        import time
        t0 = time.time()
        out = []
        for s in range(0, q.shape[0], self.num_worlds):
            e = s + self.num_worlds
            out.append(self.r.render(q[s:e], cube_pos[s:e], cube_quat[s:e], goal_pose[s:e]))
        rgb = torch.cat(out) if len(out) > 1 else out[0]
        self.calls += 1
        self.views += q.shape[0] * len(self.cameras)
        self.seconds += time.time() - t0
        return rgb


def to_input(rgb, shift_px=0, gen=None):
    """uint8 [n, C, H, W, 3] -> float [n, C, 3, H, W], ImageNet-normalised; optional random shift (training only)."""
    x = rgb.permute(0, 1, 4, 2, 3).float().div_(255.0)
    n, C, _, H, W = x.shape
    if shift_px:
        x = random_shift(x.reshape(n * C, 3, H, W), shift_px, gen).reshape(n, C, 3, H, W)
    m = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 1, 3, 1, 1)
    s = torch.tensor(IMAGENET_STD, device=x.device).view(1, 1, 3, 1, 1)
    return (x - m) / s


def random_shift(x, pad, gen=None):
    """DrQ random shift: replicate-pad by ``pad`` and crop back at a random integer offset per image ([N, 3, H, W])."""
    N, _, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (N,), device=x.device, generator=gen)
    oy = torch.randint(0, 2 * pad + 1, (N,), device=x.device, generator=gen)
    iy = (oy[:, None] + torch.arange(H, device=x.device)[None]).clamp_max(H + 2 * pad - 1)    # [N, H]
    ix = (ox[:, None] + torch.arange(W, device=x.device)[None]).clamp_max(W + 2 * pad - 1)    # [N, W]
    b = torch.arange(N, device=x.device)[:, None, None]
    return xp.permute(0, 2, 3, 1)[b, iy[:, :, None], ix[:, None, :]].permute(0, 3, 1, 2)


def sustain_place_labels(d):
    """Training targets with the PLACE trigger held through the CARRY line that precedes it: for every decision in mode
    hold with a[4] > 0 (the expert's PLACE trigger, taken at rest), the contiguous earlier decisions of the same
    episode in mode hold with a moving decoder target (|target velocity| or |yaw rate| > 1e-6: a CARRY line runs)
    get a[4] = +1. The decoder does not read a[4] while a CARRY line runs (no PLACE, no arming change from a value
    > 0), so the expert's executed episode is unchanged; the label no longer flips between the line's last decision
    (target speed p50 13 mm/s, p10 0.8 mm/s in the MuJoCo demos) and the first decision at rest (0), which the
    student cannot tell apart (measured 2026-10-11: the v1 student never triggered a PLACE). -> action array copy."""
    from nedm.so101_policy.decoder import M
    a = np.array(d["action"], copy=True)
    mode = d["mode"].astype(np.int64)
    dec = d["dec_obs"]
    moving = (mode == M["hold"]) & ((np.linalg.norm(dec[:, 5:8], axis=1) > 1e-6) | (np.abs(dec[:, 8]) > 1e-6))
    trig = np.flatnonzero((mode == M["hold"]) & (a[:, 4] > 0))
    ep = d["ep_index"]
    for k in trig:
        j = k - 1
        while j >= 0 and ep[j] == ep[k] and moving[j]:
            a[j, 4] = 1.0
            j -= 1
    return a


class DemoSet:
    """Demo files (``demos.collect`` output) on one device: the successful episodes (``success_only``), split by episode
    into train / validation (``val_frac``, seeded), at most ``max_episodes`` (the first ones after a seeded shuffle);
    plus every episode of DAgger files (meta ``dagger``; training only). ``sustain_place``: ``sustain_place_labels``."""

    def __init__(self, paths, device="cuda", chunk=20, success_only=True, max_episodes=0, val_frac=0.05, seed=0,
                 sustain_place=False, privileged=False):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        files = [D.load(p) for p in paths]
        self.meta = [f["meta"] for f in files]
        decs = {json.dumps(m.get("decoder_cfg", {}), sort_keys=True) for m in self.meta}
        if len(decs) > 1:
            raise ValueError(f"demo files with different decoder configs: {decs}")
        self.decoder_cfg = self.meta[0].get("decoder_cfg", {})
        keys = set.intersection(*[set(f) for f in files]) - {"meta"}
        d = D.concat([{k: v for k, v in f.items() if k in keys} for f in files])
        E = len(d["ep_end"])
        # DAgger files (meta "dagger"): every episode (expert labels on the student's states); demo files: successes
        dag = np.concatenate([np.full(len(f["ep_end"]), bool(f["meta"].get("dagger", False))) for f in files])
        keep = d["ep_success"].astype(bool) if success_only else np.ones(E, bool)
        self.dagger_eps = int(dag.sum())
        # expert-demo episodes: seeded shuffle, the first max_episodes, val_frac of them for validation; DAgger
        # episodes: all of them, training only (so the demo split is the same in every DAgger round)
        ep_ids = np.flatnonzero(keep & ~dag)
        rng = np.random.default_rng(int(seed))
        ep_ids = ep_ids[rng.permutation(len(ep_ids))]
        if max_episodes:
            ep_ids = ep_ids[: int(max_episodes)]
        n_val = int(round(val_frac * len(ep_ids)))
        self.val_eps = np.sort(ep_ids[:n_val])
        self.train_eps = np.sort(np.concatenate([ep_ids[n_val:], np.flatnonzero(dag)]))
        self.chunk, self.dev = int(chunk), torch.device(device)
        t = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=self.dev)   # noqa
        self.q, self.cube_pos, self.cube_quat = t(d["q"]), t(d["cube_pos"]), t(d["cube_quat"])
        self.state = proprio(self.q, t(d["dec_obs"]))
        act = sustain_place_labels(d) if sustain_place else d["action"]
        self.relabelled = int((act[:, 4] != d["action"][:, 4]).sum())
        self.action = t(act)
        self.ep_index = t(d["ep_index"], torch.long)
        self.t = t(d["t"], torch.long)
        self.ep_start = t(d["ep_start"], torch.long)
        self.ep_len = t(d["ep_length"], torch.long)
        self.goal_pose = torch.cat((t(d["ep_goal_xy"]), t(d["ep_goal_yaw"])[:, None]), -1)     # [E, 3]
        self.goal = goal_vec(d["ep_goal_xy"], d["ep_goal_yaw"]).to(self.dev)                    # [E, 4]
        self.privileged = bool(privileged)                  # ablation: the goal vector also holds the true cube state
        self.priv = cube_vec(self.cube_pos, self.cube_quat).to(self.dev) if self.privileged else None
        rows = lambda eps: np.concatenate([np.arange(d["ep_start"][e], d["ep_start"][e] + d["ep_length"][e])  # noqa
                                           for e in eps]) if len(eps) else np.zeros(0, np.int64)
        self.train_rows = t(rows(self.train_eps), torch.long)
        self.val_rows = t(rows(self.val_eps), torch.long)
        self.n_train_eps, self.n_val_eps = len(self.train_eps), len(self.val_eps)

    def stats(self, state_floor=1e-2, action_floor=5e-2):
        """Mean / std of state, goal and action over the training rows (std floored)."""
        r = self.train_rows
        out = {}
        g = self.goal[self.ep_index[r]]
        if self.privileged:
            g = torch.cat((g, self.priv[r]), -1)
        for k, x, fl in (("state", self.state[r], state_floor), ("goal", g, state_floor),
                         ("action", self.action[r], action_floor)):
            out[f"{k}_mean"] = x.mean(0).cpu()
            out[f"{k}_std"] = x.std(0).clamp_min(fl).cpu()
        return out

    def sample(self, n, gen=None, split="train"):
        rows = self.train_rows if split == "train" else self.val_rows
        return rows[torch.randint(len(rows), (int(n),), device=self.dev, generator=gen)]

    def batch(self, rows):
        """Rows [n] -> dict: q, cube_pos, cube_quat, goal_pose [n, 3] (render inputs), state [n, 30], goal [n, 4],
        action [n, chunk, 6], action_is_pad [n, chunk]."""
        e = self.ep_index[rows]
        k = torch.arange(self.chunk, device=self.dev)
        end = self.ep_start[e] + self.ep_len[e]                       # one past the episode's last row
        idx = rows[:, None] + k[None]
        pad = idx >= end[:, None]
        idx = torch.minimum(idx, (end - 1)[:, None])
        return dict(q=self.q[rows], cube_pos=self.cube_pos[rows], cube_quat=self.cube_quat[rows],
                    goal_pose=self.goal_pose[e], state=self.state[rows],
                    goal=torch.cat((self.goal[e], self.priv[rows]), -1) if self.privileged else self.goal[e],
                    action=self.action[idx],
                    action_is_pad=pad)

