"""VIRAL / DoorMan-style camera student of build9 (``~/so101_policy/docs/build9_viral_student.md``).

Student inputs: ONLY the top and wrist RGB images (Newton / Warp renderer, the goal drawn as a flat red colour block,
``render.so101`` goal "block") and the arm's q (6) and qdot (6). Output: a chunk of ``chunk`` absolute joint targets
(5 arm joints + jaw, 20 ms each) relative to q at the query, one query every 0.1 s, the first 5 executed through the
same action layer as the teacher (``joint_env.safety``). Labels: the Markov joint teacher (``joint_env.JointTeacher``)
at every recorded decision boundary: chunk elements 0-4 = the JT's 5 targets at the query state, 5-9 = the JT's 5
targets at the next boundary of the same episode (on the path that happened), relative to q at the query.

Pieces:

- ``DRViews``: ``SO101Renderer`` (Newton, top + wrist, block goal, table drawn as a body; tiled ray order 8 x 8) with
  visual randomisation (``DRConfig``): per row object colours (arm, cube, block, table), camera pose offsets (top and
  wrist), per call the light direction; ``augment`` per image in torch (brightness, contrast, saturation, hue, noise,
  blur; VIRAL's kinds, one draw per image).
- ``ViralNet``: per camera ResNet18 to layer3 (ImageNet, frozen BN) -> spatial-softmax keypoints + pool (build8's
  per-step encoder); MLP over [image features, normalised q, qdot (+ Gaussian noise on qdot in training)] -> normalised
  chunk; training-only auxiliary head on the image features -> block x, y, sin / cos 4 yaw and cube x, y, z.
- ``StudentSet``: ``joint_teacher.run_tasks`` record files -> query rows, chunk targets and validity, aux targets.
- ``train``: images rendered on the fly (fresh randomisation per sample), masked L1 on the chunk + aux loss.
- ``Actor``: the student as the driver of ``run_tasks`` (DR fixed per world for an episode, or nominal visuals).
- ``OnlineDAgger`` (``OnlineConfig``, ``ReplayBuffer``): VIRAL's online loop on one GPU: NRD worlds with automatic
  resets, fixed teacher / student world slots for the whole run, every world rendered once per decision, the JT
  labels every world, a GPU replay buffer, a few training steps per decision (``vision_viral.py online``).
- ``expand_hist``: a student with a robot-input history (``ViralConfig.hist``) from one without (zero weights on the
  new inputs).

Measured choices (build9 report): qdot noise 0.3 (1.0 and 0.0 were worse), jaw lead 4 at execution (8, build8's
value, closes the jaw during the descent and opens it before the cube is down), no robot history (3 decisions of
history: worse), one training step per new sample in the online loop (two: worse).
"""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nedm.so101_policy import joint_env as JE
from nedm.so101_policy.vision import dataset as V

CAMERAS = ("top", "wrist")


def log_default(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


# ---- rendering with visual randomisation ------------------------------------------------------------------------------
@dataclass
class DRConfig:
    on: bool = True
    # per row (world) object colours: brightness factor and per-channel tint
    arm_bright: tuple = (0.75, 1.25)
    arm_tint: float = 0.08
    cube_bright: tuple = (0.75, 1.25)
    cube_tint: float = 0.10
    block_bright: tuple = (0.75, 1.25)
    block_tint: float = 0.10
    table_bright: tuple = (0.6, 1.35)
    table_tint: float = 0.15
    # camera pose offsets (uniform, world-frame position m, local rotation rad)
    top_pos: float = 0.01
    top_rot: float = 0.03
    wrist_pos: float = 0.002
    wrist_rot: float = 0.02
    # light direction per call: elevation (deg above the horizon) and any azimuth
    light_elev: tuple = (35.0, 85.0)
    # per image (probability, range); VIRAL's kinds
    p_bright: float = 0.4
    bright: tuple = (0.7, 1.5)
    p_contrast: float = 0.4
    contrast: tuple = (0.6, 1.4)
    p_sat: float = 0.4
    sat: tuple = (0.5, 1.6)
    p_hue: float = 0.4
    hue: float = 0.06
    p_noise: float = 0.25
    noise: tuple = (0.0, 0.06)
    p_blur: float = 0.25
    blur: tuple = (0.1, 1.2)

    @staticmethod
    def from_dict(d):
        d = dict(d or {})
        known = set(DRConfig.__dataclass_fields__)
        bad = sorted(set(d) - known - {"note"})
        if bad:
            raise ValueError(f"unknown DR keys {bad}")
        return DRConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in known})


def _quat_axis_angle(v):
    """[..., 3] rotation vectors -> quaternions w x y z."""
    a = v.norm(dim=-1, keepdim=True)
    s = torch.where(a > 1e-12, torch.sin(a / 2) / a.clamp_min(1e-12), torch.full_like(a, 0.5))
    return torch.cat((torch.cos(a / 2), v * s), -1)


class DRViews:
    """The student's cameras with visual randomisation (module docstring). ``render(q, cube_pos, cube_quat, goal_pose,
    dr=None)``: dr = None (nominal visuals) or a dict of per-row draws (``draw``): colors [n, nb, 3], cam_dpos [n, C, 3],
    cam_drot [n, C, 3]; the light direction is drawn per call when ``light`` is True. -> uint8 [n, C, H, W, 3]."""

    def __init__(self, num_worlds, res=224, cameras=CAMERAS, backend="newton", tiled=True, shadows=True,
                 top_eye=V.TOP_EYE, top_target=V.TOP_TARGET, dr: DRConfig | None = None, seed=0, device="cuda"):
        from nedm.render.so101 import SO101Renderer
        self.r = SO101Renderer(int(num_worlds), width=int(res), height=int(res), backend=backend, cameras=tuple(cameras),
                               top_eye=tuple(top_eye), top_target=tuple(top_target), shadows=shadows, goal="block",
                               table_body=True)
        if tiled and backend == "newton":
            from newton._src.sensors.warp_raytrace.types import RenderOrder
            rc = self.r.renderer.backend._camera.render_config
            rc.render_order, rc.tile_width, rc.tile_height = RenderOrder.TILED, 8, 8
        self.num_worlds, self.res, self.cameras = int(num_worlds), int(res), tuple(cameras)
        self.dr = dr or DRConfig()
        self.dev = torch.device(device)
        self.calls, self.views, self.seconds = 0, 0, 0.0
        sc = self.r.scene
        nb = len(sc.body_names)
        self.nb = nb
        self.base = torch.tensor([m.color for m in sc.scene.bodies], dtype=torch.float32, device=self.dev)   # [nb, 3]
        n_arm = len(sc.mesh_body)
        self.groups = dict(arm=slice(0, n_arm), cube=slice(n_arm, n_arm + sc.cube_meshes),
                           block=slice(n_arm + sc.cube_meshes, n_arm + sc.cube_meshes + sc.goal_meshes),
                           table=slice(nb - 1, nb))
        self.gen = torch.Generator(device=self.dev).manual_seed(int(seed) + 911)
        self._colors_nominal = True
        self.light_default = (-0.57735026, 0.57735026, -0.57735026)

    # -- draws --
    def _u(self, n, lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(n, *shape, device=self.dev, generator=self.gen)

    def draw(self, n):
        """Per-row randomisation draws for n rows."""
        d = self.dr
        cols = self.base.expand(n, self.nb, 3).clone()
        for g, (br, ti) in dict(arm=(d.arm_bright, d.arm_tint), cube=(d.cube_bright, d.cube_tint),
                                block=(d.block_bright, d.block_tint), table=(d.table_bright, d.table_tint)).items():
            sl = self.groups[g]
            b = self._u(n, br[0], br[1], 1, 1)
            t = self._u(n, 1 - ti, 1 + ti, 1, 3)
            cols[:, sl] = (cols[:, sl] * b * t).clamp(0, 1)
        C = len(self.cameras)
        pos = torch.tensor([d.wrist_pos if c == "wrist" else d.top_pos for c in self.cameras], device=self.dev)
        rot = torch.tensor([d.wrist_rot if c == "wrist" else d.top_rot for c in self.cameras], device=self.dev)
        dpos = self._u(n, -1.0, 1.0, C, 3) * pos[None, :, None]
        drot = self._u(n, -1.0, 1.0, C, 3) * rot[None, :, None]
        return dict(colors=cols, cam_dpos=dpos, cam_drot=drot)

    def _light(self, nominal):
        import warp as wp
        if nominal:
            d = self.light_default
        else:
            el = math.radians(float(self._u(1, *self.dr.light_elev)[0]))
            az = float(self._u(1, 0.0, 2 * math.pi)[0])
            d = (-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el))
        self.r.renderer.backend._camera.utils.create_default_light(self.r.shadows, direction=wp.vec3f(*d))

    @torch.no_grad()
    def render(self, q, cube_pos, cube_quat, goal_pose, dr=None, light=None):
        from nedm.render.so101 import qmul, wxyz, xyzw
        t0 = time.time()
        nominal = dr is None
        self._light(nominal if light is None else not light)
        out = []
        for s in range(0, q.shape[0], self.num_worlds):
            e = min(q.shape[0], s + self.num_worlds)
            n = e - s
            body_q, cam_q = self.r.scene.transforms(q[s:e], cube_pos[s:e], cube_quat[s:e], goal_pose[s:e])
            if dr is not None:
                cq = cam_q.to(self.dev, torch.float32).clone()
                cq[..., :3] = cq[..., :3] + dr["cam_dpos"][s:e].to(cq)
                dq = _quat_axis_angle(dr["cam_drot"][s:e].to(cq))
                cq[..., 3:] = xyzw(qmul(wxyz(cq[..., 3:]), dq))
                cam_q = cq
                cols = dr["colors"][s:e]
            else:
                cols = self.base.expand(n, self.nb, 3)
            if n < self.num_worlds:
                pad = self.num_worlds - n
                body_q = torch.cat([body_q, body_q[:1].expand(pad, *body_q.shape[1:])])
                cam_q = torch.cat([cam_q, cam_q[:1].expand(pad, *cam_q.shape[1:])])
                cols = torch.cat([cols, cols[:1].expand(pad, *cols.shape[1:])])
            if not (nominal and self._colors_nominal):
                self.r._newton_colors(cols.float().cpu())
                self._colors_nominal = nominal
            frames = self.r.renderer.render(cam_q, body_q=body_q)
            out.append(frames.rgb_torch()[:n].clone(memory_format=torch.contiguous_format))
        rgb = torch.cat(out) if len(out) > 1 else out[0]
        self.calls += 1
        self.views += q.shape[0] * len(self.cameras)
        self.seconds += time.time() - t0
        return rgb


def _gray(x):
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]


def augment(x, d: DRConfig, gen=None):
    """Per-image colour randomisation of float images x [N, 3, H, W] in 0-1 (module docstring) -> same shape."""
    N, dev = x.shape[0], x.device

    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(N, 1, 1, 1, device=dev, generator=gen)

    def on(p):
        return (torch.rand(N, 1, 1, 1, device=dev, generator=gen) < p).float()

    m = on(d.p_bright)
    x = x * (1 + m * (u(*d.bright) - 1))
    m = on(d.p_contrast)
    mu = _gray(x).mean((2, 3), keepdim=True)
    x = mu + (x - mu) * (1 + m * (u(*d.contrast) - 1))
    m = on(d.p_sat)
    g = _gray(x)
    x = g + (x - g) * (1 + m * (u(*d.sat) - 1))
    m = on(d.p_hue)
    th = m * u(-d.hue, d.hue) * 2 * math.pi                    # hue turn in YIQ space
    yiq = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], device=dev)
    inv = torch.linalg.inv(yiq)
    y = torch.einsum("ij,njhw->nihw", yiq, x)
    c, s = torch.cos(th), torch.sin(th)
    i2 = c * y[:, 1:2] - s * y[:, 2:3]
    q2 = s * y[:, 1:2] + c * y[:, 2:3]
    x = torch.einsum("ij,njhw->nihw", inv, torch.cat((y[:, :1], i2, q2), 1))
    m = on(d.p_noise)
    x = x + m * u(*d.noise) * torch.randn(x.shape, device=dev, generator=gen)
    m = on(d.p_blur)
    if bool(m.any()):
        sig = u(*d.blur)[:, 0, 0, 0]
        k = torch.arange(-2, 3, device=dev, dtype=x.dtype)
        w = torch.exp(-(k[None] ** 2) / (2 * sig[:, None] ** 2))
        w = w / w.sum(-1, keepdim=True)                                         # [N, 5]
        xb = x.reshape(1, N * 3, *x.shape[2:])
        wk = w.repeat_interleave(3, 0)
        xb = F.conv2d(F.pad(xb, (2, 2, 0, 0), mode="replicate"), wk[:, None, None, :], groups=N * 3)
        xb = F.conv2d(F.pad(xb, (0, 0, 2, 2), mode="replicate"), wk[:, None, :, None], groups=N * 3)
        x = torch.where(m.bool(), xb.reshape_as(x), x)
    return x.clamp(0, 1)


def to_input(rgb, d: DRConfig | None = None, gen=None):
    """uint8 [n, C, H, W, 3] -> float [n, C, 3, H, W], ImageNet-normalised; ``d``: per-image augmentation (training)."""
    x = rgb.permute(0, 1, 4, 2, 3).float().div_(255.0)
    n, C = x.shape[:2]
    if d is not None and d.on:
        x = augment(x.reshape(n * C, *x.shape[2:]), d, gen).reshape(x.shape)
    m = torch.tensor(V.IMAGENET_MEAN, device=x.device).view(1, 1, 3, 1, 1)
    s = torch.tensor(V.IMAGENET_STD, device=x.device).view(1, 1, 3, 1, 1)
    return (x - m) / s


# ---- the student network -----------------------------------------------------------------------------------------------
@dataclass
class ViralConfig:
    chunk: int = 10
    n_exec: int = 5
    res: int = 224
    cameras: tuple = CAMERAS
    qd_noise: float = 1.0              # training: Gaussian noise (std units) on the normalised qdot input
    hist: int = 0                      # robot input history: [q, qdot] of the previous ``hist`` decisions (online path)
    encoder: str = "resnet18"
    stages: int = 3
    keypoints: int = 32
    pool_dim: int = 64
    hidden: tuple = (512, 512)
    aux_weight: float = 1.0            # block x, y, sin / cos 4 yaw + cube x, y, z from the image features
    aux_hidden: int = 256
    jaw_lead: int = 0                  # execution: jaw target taken this many steps ahead in the chunk
    batch: int = 128
    render_batch: int = 256
    reuse: int = 1                     # training steps per rendered image = reuse (random sub-batches of the buffer)
    lr: float = 3e-4
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    warmup: int = 300
    grad_clip: float = 10.0
    amp: bool = True
    seed: int = 0
    dr: dict = field(default_factory=dict)        # DRConfig overrides (training images)
    dr_frac: float = 1.0                          # share of training samples with randomised visuals

    @staticmethod
    def from_dict(d):
        d = dict(d or {})
        known = set(ViralConfig.__dataclass_fields__)
        bad = sorted(set(d) - known - {"note", "notes"})
        if bad:
            raise ValueError(f"unknown viral config keys {bad}")
        for k in ("cameras", "hidden"):
            if k in d:
                d[k] = tuple(d[k])
        return ViralConfig(**{k: v for k, v in d.items() if k in known})


STAT_KEYS = ("robot_mean", "robot_std", "tgt_mean", "tgt_std", "aux_mean", "aux_std")
AUX_DIM = 7


class ViralNet(nn.Module):
    def __init__(self, cfg: ViralConfig, stats, pretrained=True):
        super().__init__()
        from nedm.so101_policy.vision.perstep import Encoder, PerStepConfig
        self.cfg = cfg
        pc = PerStepConfig(encoder=cfg.encoder, stages=cfg.stages, keypoints=cfg.keypoints, pool_dim=cfg.pool_dim)
        self.enc = nn.ModuleList([Encoder(pc, pretrained) for _ in cfg.cameras])
        fd = sum(e.out_dim for e in self.enc)
        layers, k = [], fd + 12 * (1 + int(cfg.hist))
        for h in cfg.hidden:
            layers += [nn.Linear(k, h), nn.ELU()]
            k = h
        layers.append(nn.Linear(k, cfg.chunk * 6))
        self.head = nn.Sequential(*layers)
        self.aux = nn.Sequential(nn.Linear(fd, cfg.aux_hidden), nn.ELU(), nn.Linear(cfg.aux_hidden, AUX_DIM))
        for k_, v in stats.items():
            self.register_buffer(k_, torch.as_tensor(v, dtype=torch.float32).clone())

    def features(self, images):
        return torch.cat([e(images[:, i]) for i, e in enumerate(self.enc)], -1)

    def robot(self, robot):
        """[B, 12 (1 + hist)] = [q, qd] now, then the previous decisions (newest first) -> normalised (+ qdot noise in
        training on every qdot block)."""
        H = 1 + int(self.cfg.hist)
        r = ((robot.view(-1, H, 12) - self.robot_mean) / self.robot_std).clamp(-10, 10)
        if self.training and self.cfg.qd_noise > 0:
            r = torch.cat((r[..., :6], r[..., 6:] + self.cfg.qd_noise * torch.randn_like(r[..., 6:])), -1)
        return r.reshape(-1, 12 * H)

    def forward(self, images, robot):
        f = self.features(images)
        return self.head(torch.cat((f, self.robot(robot).to(f.dtype)), -1).float()).view(-1, self.cfg.chunk, 6), f

    def loss(self, images, robot, tgt_n, valid, aux_n=None):
        p, f = self(images, robot)
        e = (p.float() - tgt_n).abs() * valid[..., None]
        l_act = e.sum() / (valid.sum() * 6).clamp_min(1)
        info = {"l_act": float(l_act.detach())}
        loss = l_act
        if aux_n is not None and self.cfg.aux_weight > 0:
            pa = self.aux(f).float()
            l_aux = F.smooth_l1_loss(pa, aux_n, beta=0.5)
            loss = loss + self.cfg.aux_weight * l_aux
            info["l_aux"] = float(l_aux.detach())
        return loss, info

    @torch.no_grad()
    def predict(self, images, robot):
        return self(images, robot)[0].float()

    @torch.no_grad()
    def predict_aux(self, images):
        return self.aux(self.features(images)).float() * self.aux_std + self.aux_mean

    def backbone_params(self):
        return [p for e in self.enc if e.is_backbone for p in e.trunk.parameters() if p.requires_grad]


def expand_hist(st: ViralNet, hist: int, device="cuda"):
    """A student with ``hist`` decisions of robot history from one without (or with fewer): the same weights, zero
    weights on the new history inputs of the head's first layer (so it starts as the given student)."""
    from dataclasses import replace as _rep
    old = int(st.cfg.hist)
    if hist == old:
        return st
    if hist < old:
        raise ValueError("cannot shrink the history")
    cfg = _rep(st.cfg, hist=int(hist))
    new = ViralNet(cfg, {k: getattr(st, k) for k in STAT_KEYS}, pretrained=False).to(device)
    sd = st.state_dict()
    nd = new.state_dict()
    for k, v in sd.items():
        if k == "head.0.weight":
            w = torch.zeros_like(nd[k])
            w[:, : v.shape[1]] = v
            nd[k] = w
        else:
            nd[k] = v
    new.load_state_dict(nd)
    return new


def save_student(st, path, **extra):
    torch.save(dict(cfg=asdict(st.cfg), state_dict=st.state_dict(), stats={k: getattr(st, k).cpu() for k in STAT_KEYS},
                    **extra), path)


def load_student(path, device="cuda"):
    ck = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    cfg = ViralConfig.from_dict(ck["cfg"])
    st = ViralNet(cfg, ck["stats"], pretrained=False).to(device)
    st.load_state_dict(ck["state_dict"])
    return st.eval(), ck


# ---- data ----------------------------------------------------------------------------------------------------------------
def aux_targets(goal_pose, cube_pos):
    """[n, 3] (x, y, yaw), [n, 3] -> [n, 7]: block x, y, sin 4 yaw, cos 4 yaw, cube x, y, z."""
    gy = goal_pose[:, 2]
    return torch.cat((goal_pose[:, :2], torch.sin(4 * gy)[:, None], torch.cos(4 * gy)[:, None], cube_pos), -1)


class StudentSet:
    """``joint_teacher.run_tasks`` record files (``row_*``, ``ep_row_start``, ``length``, goals) on one device. A sample
    = one decision boundary: render inputs, robot [q, qd], chunk target [K, 6] = (lab[r] ++ lab[r + 1] ...) - q[r] with
    validity (inside the episode), aux targets. Validation: ``val_frac`` of the FIRST file's episodes (seeded)."""

    def __init__(self, paths, device="cuda", chunk=10, val_frac=0.03, seed=0, max_files=None):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        paths = list(paths)[-max_files:] if max_files else list(paths)
        keys = ("row_q", "row_qd", "row_cube_pos", "row_cube_quat", "row_lab", "row_ep", "length", "ep_row_start",
                "goal_xy", "goal_yaw", "src", "lenient", "success")
        parts, e0, r0 = {k: [] for k in keys}, 0, 0
        is_val = []
        for fi, p in enumerate(paths):
            z = np.load(Path(p).expanduser())
            E = len(z["length"])
            for k in keys:
                v = z[k]
                if k == "row_ep":
                    v = v.astype(np.int64) + e0
                if k == "ep_row_start":
                    v = v.astype(np.int64) + r0
                parts[k].append(v)
            iv = np.zeros(E, bool)
            if fi == 0 and val_frac > 0:
                rng = np.random.default_rng(seed)
                iv[rng.permutation(E)[: int(round(val_frac * E))]] = True
            is_val.append(iv)
            e0 += E
            r0 += len(z["row_ep"])
        d = {k: np.concatenate(v) for k, v in parts.items()}
        self.dev, self.chunk = torch.device(device), int(chunk)
        t = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=self.dev)   # noqa
        self.q, self.qd = t(d["row_q"]), t(d["row_qd"])
        self.cube_pos, self.cube_quat = t(d["row_cube_pos"]), t(d["row_cube_quat"])
        self.lab = t(d["row_lab"])                                       # [N, 5, 6] absolute
        self.ep = t(d["row_ep"], torch.long)
        self.ep_len = t(d["length"], torch.long)
        self.ep_start = t(d["ep_row_start"], torch.long)
        self.goal_pose = torch.cat((t(d["goal_xy"]), t(d["goal_yaw"])[:, None]), -1)
        self.src = t(d["src"], torch.long)
        self.robot = torch.cat((self.q, self.qd), -1)
        iv = t(np.concatenate(is_val), torch.bool)[self.ep]
        self.train_rows = torch.nonzero(~iv).flatten()
        self.val_rows = torch.nonzero(iv).flatten()
        self.n_eps = len(self.ep_len)
        self.n_rows = len(self.q)
        self.paths = [str(p) for p in paths]

    def targets(self, rows):
        K, S = self.chunk, JE.STEPS
        nb = (K + S - 1) // S
        e = self.ep[rows]
        last = self.ep_start[e] + self.ep_len[e] - 1
        tg, ok = [], []
        for b in range(nb):
            r = rows + b
            inside = r <= last
            lab = self.lab[torch.minimum(r, last)]
            tg.append(lab)
            ok.append(inside[:, None].expand(-1, S))
        tgt = torch.cat(tg, 1)[:, :K] - self.q[rows][:, None, :]
        return tgt, torch.cat(ok, 1)[:, :K]

    def batch(self, rows):
        tgt, valid = self.targets(rows)
        gp = self.goal_pose[self.ep[rows]]
        return dict(q=self.q[rows], cube_pos=self.cube_pos[rows], cube_quat=self.cube_quat[rows], goal_pose=gp,
                    robot=self.robot[rows], target=tgt, valid=valid, aux=aux_targets(gp, self.cube_pos[rows]))

    def stats(self, floor=1e-3, robot_floor=1e-2, n=200_000, seed=0):
        g = torch.Generator(device=self.dev).manual_seed(seed)
        rows = self.train_rows[torch.randint(len(self.train_rows), (min(n, len(self.train_rows)),), device=self.dev,
                                             generator=g)]
        b = self.batch(rows)
        v = b["valid"][..., None].float()
        cnt = v.sum(0).clamp_min(1)
        mu = (b["target"] * v).sum(0) / cnt
        sd = (((b["target"] - mu) ** 2 * v).sum(0) / cnt).sqrt().clamp_min(floor)
        x = b["robot"]
        return dict(robot_mean=x.mean(0).cpu(), robot_std=x.std(0).clamp_min(robot_floor).cpu(), tgt_mean=mu.cpu(),
                    tgt_std=sd.cpu(), aux_mean=b["aux"].mean(0).cpu(), aux_std=b["aux"].std(0).clamp_min(1e-3).cpu())

    def sample(self, n, gen=None, split="train"):
        rows = self.train_rows if split == "train" else self.val_rows
        return rows[torch.randint(len(rows), (int(n),), device=self.dev, generator=gen)]


# ---- training -----------------------------------------------------------------------------------------------------------
def normalise(st, tgt):
    return (tgt - st.tgt_mean) / st.tgt_std


def denormalise(st, p):
    return p * st.tgt_std + st.tgt_mean


@torch.no_grad()
def val_metrics(st, data: StudentSet, views: DRViews, rows, bs=256):
    e1, c1, ea, ca = 0.0, 0.0, 0.0, 0.0
    eb, cb = torch.zeros(2, device=data.dev), 0
    for s in range(0, len(rows), bs):
        b = data.batch(rows[s:s + bs])
        img = to_input(views.render(b["q"], b["cube_pos"], b["cube_quat"], b["goal_pose"]))
        with torch.autocast(data.dev.type, dtype=torch.bfloat16, enabled=st.cfg.amp):
            p = denormalise(st, st.predict(img, b["robot"]))
            pa = st.predict_aux(img)
        err = (p - b["target"]).abs() * b["valid"][..., None]
        e1 += float(err[:, :st.cfg.n_exec].sum())
        c1 += float(b["valid"][:, :st.cfg.n_exec].sum()) * 6
        ea += float(err.sum())
        ca += float(b["valid"].sum()) * 6
        eb += (pa[:, :2] - b["aux"][:, :2]).norm(dim=-1).sum() * torch.tensor([1.0, 0.0], device=data.dev) + \
            (pa[:, 4:7] - b["aux"][:, 4:7]).norm(dim=-1).sum() * torch.tensor([0.0, 1.0], device=data.dev)
        cb += len(b["q"])
    return dict(val_mae_exec_mrad=1e3 * e1 / max(c1, 1), val_mae_chunk_mrad=1e3 * ea / max(ca, 1),
                val_block_xy_mm=float(eb[0]) / max(cb, 1) * 1e3, val_cube_mm=float(eb[1]) / max(cb, 1) * 1e3)


def train(data: StudentSet, views: DRViews, cfg: ViralConfig, out_dir, max_steps=6000, init_from=None,
          log=log_default, val_n=1024, val_every=1000, log_every=200, device="cuda", extra_meta=None):
    """Train (or fine-tune ``init_from``: weights and normalisation) -> student (eval mode); writes out_dir/{meta.json,
    learning_curve.csv, last.pt}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cuda = torch.device(device).type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    if init_from:
        st, _ = load_student(init_from, device)
        st.cfg = cfg
    else:
        st = ViralNet(cfg, data.stats(), pretrained=True).to(device)
    st.train()
    drc = DRConfig.from_dict(cfg.dr)
    views.dr = drc
    bb = st.backbone_params()
    bb_ids = {id(p) for p in bb}
    rest = [p for p in st.parameters() if p.requires_grad and id(p) not in bb_ids]
    opt = torch.optim.AdamW([dict(params=rest, lr=cfg.lr), dict(params=bb, lr=cfg.lr_backbone)],
                            weight_decay=cfg.weight_decay)
    base = [g["lr"] for g in opt.param_groups]
    vgen = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    vrows = data.sample(min(val_n, len(data.val_rows)), vgen, "val") if len(data.val_rows) else None
    meta = dict(cfg=asdict(cfg), files=data.paths, n_eps=data.n_eps, rows=data.n_rows, train_rows=len(data.train_rows),
                init_from=str(init_from or ""), **(extra_meta or {}))
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    f = open(out / "learning_curve.csv", "w", newline="")
    wr = None
    t0 = time.time()
    t_render = t_train = 0.0
    buf, pos = None, 0
    agg = {}
    per_buf = max(1, int(cfg.reuse) * cfg.render_batch // cfg.batch)
    for step in range(1, max_steps + 1):
        if buf is None or pos >= per_buf:
            tr = time.time()
            b = data.batch(data.sample(cfg.render_batch, gen))
            n = len(b["q"])
            ndr = int(round(cfg.dr_frac * n)) if drc.on else 0
            if ndr >= n:
                b["rgb"] = views.render(b["q"], b["cube_pos"], b["cube_quat"], b["goal_pose"], dr=views.draw(n))
            elif ndr == 0:
                b["rgb"] = views.render(b["q"], b["cube_pos"], b["cube_quat"], b["goal_pose"])
            else:
                r1 = views.render(b["q"][:ndr], b["cube_pos"][:ndr], b["cube_quat"][:ndr], b["goal_pose"][:ndr],
                                  dr=views.draw(ndr))
                r2 = views.render(b["q"][ndr:], b["cube_pos"][ndr:], b["cube_quat"][ndr:], b["goal_pose"][ndr:])
                b["rgb"] = torch.cat((r1, r2))
            b["aug"] = torch.arange(n, device=data.dev) < ndr
            b["tgt_n"] = normalise(st, b["target"])
            b["aux_n"] = (b["aux"] - st.aux_mean) / st.aux_std
            sync()
            t_render += time.time() - tr
            buf, pos = b, 0
        tt = time.time()
        if cfg.reuse <= 1:
            sl = slice(pos * cfg.batch, (pos + 1) * cfg.batch)
        else:
            sl = torch.randperm(cfg.render_batch, device=data.dev, generator=gen)[:cfg.batch]
        pos += 1
        rgb = buf["rgb"][sl]
        aug = buf["aug"][sl]
        img = to_input(rgb, drc if bool(aug.any()) else None, gen)
        if bool(aug.any()) and not bool(aug.all()):
            img = torch.where(aug[:, None, None, None, None], img, to_input(rgb))
        frac = min(1.0, step / max(1, cfg.warmup))
        if step > cfg.warmup:
            u = (step - cfg.warmup) / max(1, max_steps - cfg.warmup)
            frac = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * u))
        for g_, lr0 in zip(opt.param_groups, base):
            g_["lr"] = lr0 * frac
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=cfg.amp):
            loss, info = st.loss(img, buf["robot"][sl], buf["tgt_n"][sl], buf["valid"][sl], buf["aux_n"][sl])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(st.parameters(), cfg.grad_clip)
        opt.step()
        sync()
        t_train += time.time() - tt
        for k, v in dict(loss=float(loss.detach()), **info).items():
            agg[k] = agg.get(k, 0.0) + v
        agg["n"] = agg.get("n", 0) + 1
        do_val = vrows is not None and (step % val_every == 0 or step == max_steps)
        if step % log_every == 0 or do_val:
            n_ = agg.pop("n")
            row = dict(step=step, wall_min=round((time.time() - t0) / 60, 3), **{k: v / n_ for k, v in agg.items()},
                       grad_norm=float(gn), lr=opt.param_groups[0]["lr"], render_s=round(t_render, 1),
                       train_s=round(t_train, 1), gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else 0)
            agg = {}
            if do_val:
                st.eval()
                row.update(val_metrics(st, data, views, vrows))
                st.train()
            if wr is None:
                keys = list(row) + [k for k in ("val_mae_exec_mrad", "val_mae_chunk_mrad", "val_block_xy_mm",
                                                "val_cube_mm") if k not in row]
                wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                wr.writeheader()
            wr.writerow(row)
            f.flush()
            log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
    st.eval()
    save_student(st, out / "last.pt", step=max_steps, wall_min=(time.time() - t0) / 60)
    f.close()
    (out / "train_summary.json").write_text(json.dumps(dict(steps=max_steps, wall_min=(time.time() - t0) / 60,
                                                            render_s=t_render, train_s=t_train,
                                                            views_per_s=views.views / max(views.seconds, 1e-9))))
    return st


# ---- the student as a driver ---------------------------------------------------------------------------------------------
class Actor:
    """``run_tasks`` student callback: renders the worlds (``dr``: per-world draws fixed for the episode, or None =
    nominal visuals), feeds [q, qdot] -> absolute targets [B, 5, 6] of the executed part of the chunk (jaw target
    ``jaw_lead`` steps ahead)."""

    def __init__(self, student: ViralNet, views: DRViews, randomise=False, jaw_lead=None, device="cuda"):
        self.st, self.views, self.dev = student, views, torch.device(device)
        self.randomise = bool(randomise)
        self.jaw_lead = student.cfg.jaw_lead if jaw_lead is None else int(jaw_lead)
        self.dr = None
        self.t_render = self.t_model = 0.0
        self.calls = 0

    def reset(self, B):
        self.dr = self.views.draw(B) if self.randomise else None
        self.rhist = None

    @torch.no_grad()
    def __call__(self, ws, goal_pose, k=None):
        t0 = time.time()
        q = ws.q.to(self.dev, torch.float32)
        cp, cq = ws.cube_pos.to(self.dev, torch.float32), ws.cube_quat.to(self.dev, torch.float32)
        ok = torch.isfinite(q).all(-1) & torch.isfinite(cp).all(-1) & torch.isfinite(cq).all(-1)
        q = torch.where(ok[:, None], q, torch.zeros_like(q))
        cp = torch.where(ok[:, None], cp, torch.tensor([0.2, 0.0, 0.015], device=self.dev))
        cq = torch.where(ok[:, None], cq, torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.dev))
        rgb = self.views.render(q, cp, cq, goal_pose.to(self.dev, torch.float32), dr=self.dr)
        img = to_input(rgb)
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        robot = torch.cat((q, ws.qd.to(self.dev, torch.float32).nan_to_num()), -1)
        H = int(self.st.cfg.hist)
        if H:
            if self.rhist is None:
                self.rhist = robot[:, None, :].expand(-1, H, -1).clone()
            now = robot
            robot = torch.cat((now, self.rhist.reshape(len(now), -1)), -1)
            self.rhist = torch.cat((now[:, None, :], self.rhist[:, :-1]), 1)
        out = []
        for s in range(0, img.shape[0], 256):
            with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.st.cfg.amp):
                out.append(denormalise(self.st, self.st.predict(img[s:s + 256], robot[s:s + 256])))
        delta = torch.cat(out)
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        self.t_render += t1 - t0
        self.t_model += time.time() - t1
        self.calls += 1
        u = q[:, None, :] + delta                                              # [B, K, 6]
        K, n = u.shape[1], self.st.cfg.n_exec
        jj = torch.arange(n, device=self.dev)
        kk = (jj + self.jaw_lead).clamp_max(K - 1)
        ex = torch.cat((u[:, jj, :5], u[:, kk, 5:]), -1)
        return ex.to(ws.q.device, torch.float64)


# ---- VIRAL-style online DAgger --------------------------------------------------------------------------------------------
@dataclass
class OnlineConfig:
    """VIRAL's loop (``viral_doorman_code.md`` sec. 3): every decision all worlds are rendered once, the student acts in
    the student slots, the JT labels every world, the samples enter a GPU replay buffer and ``steps_per_iter`` training
    steps run. Slots ``[0, teacher_share * envs)`` are JT-driven for the whole run (VIRAL: fixed env indices), the first
    ``noise_share`` of them with Gaussian noise of ``noise`` x the JT's PPO std on its normalised action (DART)."""
    envs: int = 1024
    teacher_share: float = 0.5
    noise_share: float = 0.5
    noise: float = 1.0
    nominal_share: float = 0.2          # slots drawn with the scene's own colours and camera poses
    default_light_p: float = 0.25       # per decision: probability of the scene's own light (else a random one)
    buffer: int = 40000                 # replay samples (uint8 images on the GPU)
    steps_per_iter: int = 8             # training steps per decision (batch = ViralConfig.batch)
    lr: float = 1e-4                    # constant (VIRAL: 2e-4 constant)
    lr_backbone: float = 5e-5
    iterations: int = 5000
    eval_every: int = 500
    save_every: int = 250
    log_every: int = 25
    warm_teacher_only: int = 0          # first iterations: every slot JT-driven (buffer fill)
    lr_decay_start: int = -1            # >= 0: from this iteration the learning rates fall linearly (cosine-free) ...
    lr_final_frac: float = 0.1          # ... to this fraction at the last iteration

    @staticmethod
    def from_dict(d):
        d = dict(d or {})
        known = set(OnlineConfig.__dataclass_fields__)
        bad = sorted(set(d) - known - {"note"})
        if bad:
            raise ValueError(f"unknown online config keys {bad}")
        return OnlineConfig(**{k: v for k, v in d.items() if k in known})


class ReplayBuffer:
    """Ring buffer of student samples on one device: rgb uint8 [C, cams, H, W, 3], q, qd, labels of this decision
    and of the next one (absolute targets [5, 6] each; ``valid2`` = the next decision is in the same episode), aux
    targets, and whether per-image colour augmentation applies (``aug``: not for nominal-slot samples)."""

    def __init__(self, cap, cams, res, device, robot_dim=12):
        f = dict(device=device)
        self.cap, self.n, self.ptr = int(cap), 0, 0
        self.rgb = torch.zeros(self.cap, cams, res, res, 3, dtype=torch.uint8, **f)
        self.q = torch.zeros(self.cap, 6, **f)
        self.robot = torch.zeros(self.cap, robot_dim, **f)
        self.lab1 = torch.zeros(self.cap, JE.STEPS, 6, **f)
        self.lab2 = torch.zeros(self.cap, JE.STEPS, 6, **f)
        self.valid2 = torch.zeros(self.cap, dtype=torch.bool, **f)
        self.aux = torch.zeros(self.cap, AUX_DIM, **f)
        self.aug = torch.zeros(self.cap, dtype=torch.bool, **f)

    def add(self, rgb, q, robot, lab, aux, aug):
        n = rgb.shape[0]
        idx = (self.ptr + torch.arange(n, device=rgb.device)) % self.cap
        self.rgb[idx], self.q[idx], self.robot[idx], self.lab1[idx] = rgb, q.float(), robot.float(), lab.float()
        self.valid2[idx] = False
        self.aux[idx], self.aug[idx] = aux.float(), aug
        self.ptr = int((self.ptr + n) % self.cap)
        self.n = min(self.cap, self.n + n)
        return idx

    def sample(self, b, gen=None):
        return torch.randint(self.n, (int(b),), device=self.rgb.device, generator=gen)


class OnlineDAgger:
    """The online loop (``OnlineConfig``). Tasks: the start states and goals of a task table (``state_*``, goal_xy,
    goal_yaw), drawn uniformly at every reset."""

    def __init__(self, st: ViralNet, jt, tasks, nrd_ckpt, views: DRViews, ocfg: OnlineConfig, device="cuda", seed=0,
                 log=log_default):
        self.st, self.jt, self.views, self.o, self.dev, self.log = st, jt, views, ocfg, torch.device(device), log
        N = int(ocfg.envs)
        cfg = JE.JointEnvConfig(world="nrd", num_envs=N, device=device, ckpts=[str(nrd_ckpt)], auto_reset=True,
                                stagger=True, starts="", seed=seed)
        self.env = JE.JointVecEnv(cfg, jt.codec)
        f64 = dict(dtype=torch.float64, device=self.dev)
        self.env.bank = {k: torch.as_tensor(tasks[f"state_{k}"], **f64) for k in
                         ("q", "qd", "cube_pos", "cube_quat", "cube_v", "cube_w")}
        nt = len(tasks["seeds"])
        self.env.pool = dict(i=torch.arange(nt, device=self.dev), goal_xy=torch.as_tensor(tasks["goal_xy"], **f64),
                             goal_yaw=torch.as_tensor(tasks["goal_yaw"], **f64))
        self.gen = torch.Generator(device=self.dev).manual_seed(int(seed) + 5)
        nt_ = int(round(ocfg.teacher_share * N))
        slot = torch.arange(N, device=self.dev)
        self.teacher = slot < nt_
        self.noisy = slot < int(round(ocfg.noise_share * nt_))
        self.nominal = torch.rand(N, device=self.dev, generator=self.gen) < ocfg.nominal_share
        self.H = int(st.cfg.hist)
        self.buf = ReplayBuffer(ocfg.buffer, len(st.cfg.cameras), st.cfg.res, self.dev, 12 * (1 + self.H))
        self.rhist = torch.zeros(N, max(self.H, 1), 12, device=self.dev)
        self.fresh = torch.ones(N, dtype=torch.bool, device=self.dev)
        self.drc = DRConfig.from_dict(st.cfg.dr)
        views.dr = self.drc
        self.dr = None
        self.prev_slot = torch.full((N,), -1, dtype=torch.long, device=self.dev)
        bb = st.backbone_params()
        bb_ids = {id(p) for p in bb}
        rest = [p for p in st.parameters() if p.requires_grad and id(p) not in bb_ids]
        self.opt = torch.optim.AdamW([dict(params=rest, lr=ocfg.lr), dict(params=bb, lr=ocfg.lr_backbone)],
                                     weight_decay=st.cfg.weight_decay)
        self.it = 0
        self.samples = 0
        self.stats = dict(ep_student=0, len_student=0, ep_teacher=0, len_teacher=0)

    def _draw(self, idx):
        d = self.views.draw(len(idx))
        nom = self.nominal[idx]
        d["colors"] = torch.where(nom[:, None, None], self.views.base.expand_as(d["colors"]), d["colors"])
        d["cam_dpos"] = torch.where(nom[:, None, None], torch.zeros_like(d["cam_dpos"]), d["cam_dpos"])
        d["cam_drot"] = torch.where(nom[:, None, None], torch.zeros_like(d["cam_drot"]), d["cam_drot"])
        if self.dr is None:
            self.dr = d
        else:
            for k in self.dr:
                self.dr[k][idx] = d[k]

    @torch.no_grad()
    def act(self):
        """One decision of every world -> timing dict."""
        env, st, o = self.env, self.st, self.o
        T = {}
        t0 = time.time()
        if env.obs is None:
            env.reset()
            self._draw(torch.arange(env.B, device=self.dev))
        ws, obs = env.ws, env.obs
        q = ws.q.to(torch.float64)
        a = self.jt.act(obs).to(torch.float64)
        lab = env.codec.targets(a, q)                                              # [N, 5, 6] the label
        noise = (self.noisy.double()[:, None] * o.noise) * self.jt.std.to(a)[None] * \
            torch.randn(a.shape, device=self.dev, generator=self.gen, dtype=a.dtype)
        tg = env.codec.targets(a + noise, q)
        gp = torch.cat((env.task.goal_xy, env.task.goal_yaw[:, None]), -1).float()
        ok = torch.isfinite(q).all(-1) & torch.isfinite(ws.cube_pos).all(-1) & torch.isfinite(ws.cube_quat).all(-1)
        qf = torch.where(ok[:, None], q, torch.zeros_like(q)).float()
        cp = torch.where(ok[:, None], ws.cube_pos, torch.tensor([0.2, 0.0, 0.015], **dict(
            dtype=ws.cube_pos.dtype, device=self.dev))).float()
        cq = torch.where(ok[:, None], ws.cube_quat, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=ws.cube_quat.dtype,
                                                                  device=self.dev)).float()
        T["jt"] = time.time() - t0
        t1 = time.time()
        light = bool(torch.rand(1, device=self.dev, generator=self.gen) >= o.default_light_p)
        rgb = self.views.render(qf, cp, cq, gp, dr=self.dr, light=light)
        torch.cuda.synchronize() if self.dev.type == "cuda" else None
        T["render"] = time.time() - t1
        t2 = time.time()
        stud = ~self.teacher if self.it >= o.warm_teacher_only else torch.zeros_like(self.teacher)
        qd = ws.qd.float().nan_to_num()
        now = torch.cat((qf, qd), -1)                                             # [N, 12]
        if self.H:
            fr = self.fresh
            self.rhist[fr] = now[fr][:, None, :].expand(-1, self.H, -1)
            robot_all = torch.cat((now, self.rhist.reshape(len(now), -1)), -1)
        else:
            robot_all = now
        if bool(stud.any()):
            si = stud.nonzero().flatten()
            st.eval()
            img = to_input(rgb[si])
            robot = robot_all[si]
            out = []
            for s in range(0, len(si), 256):
                with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=st.cfg.amp):
                    out.append(denormalise(st, st.predict(img[s:s + 256], robot[s:s + 256])))
            st.train()
            u = qf[si][:, None, :] + torch.cat(out)
            K, n = u.shape[1], st.cfg.n_exec
            jj = torch.arange(n, device=self.dev)
            kk = (jj + st.cfg.jaw_lead).clamp_max(K - 1)
            tg[si] = torch.cat((u[:, jj, :5], u[:, kk, 5:]), -1).double()
        T["student"] = time.time() - t2
        # samples: this decision's label completes the previous sample of every world that did not reset
        live = ok
        pv = self.prev_slot
        has = (pv >= 0) & live
        if bool(has.any()):
            self.buf.lab2[pv[has]] = lab[has].float()
            self.buf.valid2[pv[has]] = True
        idx = live.nonzero().flatten()
        aux = aux_targets(gp[idx], cp[idx])
        slots = self.buf.add(rgb[idx], qf[idx], robot_all[idx], lab[idx], aux, ~self.nominal[idx])
        if self.H:
            self.rhist = torch.cat((now[:, None, :], self.rhist[:, :-1]), 1)
            self.fresh = torch.zeros_like(self.fresh)
        self.prev_slot = torch.full_like(self.prev_slot, -1)
        self.prev_slot[idx] = slots
        self.samples += len(idx)
        t3 = time.time()
        _, _, done, ex = env.step(targets=tg)
        T["env"] = time.time() - t3
        if bool(done.any()):
            di = done.nonzero().flatten()
            self.prev_slot[di] = -1                       # the next decision is a new episode
            self.fresh[di] = True
            ep = ex["episode"]
            le = ep["lenient"][di]
            ts = self.teacher[di]
            self.stats["ep_teacher"] += int(ts.sum())
            self.stats["len_teacher"] += int((le & ts).sum())
            self.stats["ep_student"] += int((~ts).sum())
            self.stats["len_student"] += int((le & ~ts).sum())
            self._draw(di)
        return T

    def train_steps(self):
        st, o, b = self.st, self.o, self.st.cfg.batch
        info_sum, n = {}, 0
        for _ in range(int(o.steps_per_iter)):
            i = self.buf.sample(b, self.gen)
            rgb = self.buf.rgb[i]
            aug = self.buf.aug[i]
            img = to_input(rgb, self.drc, self.gen)
            img = torch.where(aug[:, None, None, None, None], img, to_input(rgb))
            q = self.buf.q[i]
            tgt = torch.cat((self.buf.lab1[i], self.buf.lab2[i]), 1)[:, :st.cfg.chunk] - q[:, None, :]
            valid = torch.cat((torch.ones(len(i), JE.STEPS, dtype=torch.bool, device=self.dev),
                               self.buf.valid2[i][:, None].expand(-1, JE.STEPS)), 1)[:, :st.cfg.chunk]
            robot = self.buf.robot[i]
            aux_n = (self.buf.aux[i] - st.aux_mean) / st.aux_std
            with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=st.cfg.amp):
                loss, info = st.loss(img, robot, normalise(st, tgt), valid, aux_n)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(st.parameters(), st.cfg.grad_clip)
            self.opt.step()
            for k, v in dict(loss=float(loss.detach()), **info).items():
                info_sum[k] = info_sum.get(k, 0.0) + v
            n += 1
        return {k: v / max(n, 1) for k, v in info_sum.items()}
