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

Real2sim (``~/so101_r2s_claude/docs/integrate.md``; all OFF by default, old configs and checkpoints unchanged):

- ``ViralConfig.scene`` = the render agent's real-scene file (``real_scene_v1.json``, read by
  ``render.so101.RealScene.from_scene_dict``; ``load_real_setup``): the calibrated ``external`` + ``wrist`` fisheye
  cameras (crop box of the real 640 x 480 frame per camera by the student-view rules shared with the robot code),
  the light-box scene and its body groups, its per-world randomisation (``RealDR``: colours, camera poses, mat /
  walls, tone map), per call its lights and intrinsics (never ``create_default_light``), its tone map and
  supersampling (inside ``SO101Renderer``), its camera effects (``image_fx``: ``render.camera_fx.ImageFX``) and the
  config's extra ones (``CamFX``: vignetting, motion blur, blur, white balance, gamma, noise, JPEG; ``fixed_*`` /
  nominal effects on every image, the random ones on training images only; ``ViewFX``). The scene / calibration
  sha256 and the crops are pinned in the student config (``pin_real``). ``make_views(cfg, n)`` builds the views of
  any config.
- Image lag (``ViralConfig.lag`` / ``lag_range``): the images show the world ``k`` control steps (20 ms) BEFORE the
  decision (q and qdot stay current); k fixed (evaluation, default ``lag``) or drawn per world per episode
  (training: ``lag_range``). ``LagHistory`` keeps the states, ``step_tap`` feeds it from any ``JointVecEnv.step``
  caller (``joint_teacher.run_tasks``); round files record the states (``row_lag``, ``lag_rows``; a lag above
  ``LAG_REC`` reaches into the decisions before).
- Image latency v8 (``OnlineConfig.img_lag`` -> ``ImgLagConfig``, 2026-10-23, off by default; measurement
  ~/so101_r2s_claude/v8/latency/: scripts/lag_fit.py, lag_dist.py -> lag_fit.json, lag_boot.json, lag_dist.json; the
  robot's images are 28-62 ms older than its joint reading, new every query, independent per camera): in the online
  loop a lag in ms per world (or per camera: one render call each), drawn at the episode start and again per decision
  with ``redraw_p``, the images rendered from the state interpolated between the 20 ms states and the goal of then
  (``LagHistory(goal=True).at``); replaces the integer lag in that loop.
Studentlab (2026-10-22, ~/so101_studentlab_claude/trials/BUILD.md; all OFF by default, old configs, checkpoints and
results bit for bit unchanged): ``ViralConfig.cmd_in`` (the previous decision's 5 executed targets - the measured q
now, / ``cmd_scale``, appended to the robot vector), ``ViralConfig.img_hist`` (the previous decision's images through
the same encoders, features appended to the head input; ``ReplayBuffer`` links to the previous sample instead of
copies), ``expand_student`` (zero-init expansion: the same chunk as the given student), ``OnlineConfig.goal_jump``
(``joint_env.GoalJumpConfig``: goal jumps while the cube is held near the goal). Saved configs keep the old keys while
the new inputs are off (``cfg_dict``). ``OnlineConfig.light_redraw_p`` (< 1; review 2026-10-23): the per-call light and
intrinsics stay for several decisions, so the image pairs of the image history share them as on the robot (a change
cuts the pair). The robot side of the new inputs: ``scripts/so101_policy/real2sim/hw/hw_lib.py`` (``StudentPolicy``,
``Guard.u_layer``).
Background randomisation (studentlab external-camera fix, 2026-10-23, ~/so101_studentlab_claude/trials/bgdr/BUILD.md;
OFF unless the scene file's ``dr`` / ``ViralConfig.real_dr`` sets it; scene v3 does): ``RealDR.floor_turn_deg`` /
``floor_xy`` / ``wall_end`` (floor box and back-wall end per world) and ``RealDR.bg`` (``nedm.render.bg_dr``: texture,
clutter, spots, dark-area profile, floor / mat gradients, low texture, mat folds and boundary blur in linear colour
before the tone map, on the external camera's floor / walls / mat / dark-area pixels only). ``DRViews`` then builds
the renderer with ``bg_cameras`` (segment output), keeps a placement prior (``bg_prior``) and adds the ``bg_*``
layers to every per-world draw (zeros in ``nominal_draw``), so the online loop's per-world reset and
``merge_nominal`` carry them unchanged.
Scene v4 round (2026-10-23, OFF by default, ~/so101_r2s_claude/v8/scene4): ``RealScene.parts`` may sit on the fixed
base (scene v4 draws the real table clamp at the base) and ``RealDR.base_addon`` (``nedm.render.base_addon``) puts
random blocks and cables at the base outline in a share of the worlds; ``load_real_setup`` then adds the slot bodies
to the scene (``RealScene.addon_slots``), nominal draws keep them hidden; ``DRViews`` draws them on their own
generator (seed + 917), so the other draws of every per-world reset stay as with them off.
Gripper look (v12 visual fix, 2026-10-26, OFF by default, ~/so101_r2s_claude/v12/vis): ``RealDR.gripper_look``
(``nedm.render.gripper_look.GripperLookDR``: per world the albedo of the camera bracket plate and the servo bracket,
the scene's fitted screws shown / moved / recoloured, 0-4 light or dark marks on the gripper parts in the bottom half
of the wrist view, a bright plate top edge; exclusion zones for a held-out variant); ``load_real_setup`` adds its
parts to the scene (``RealScene.look_slots``), ``DRViews`` draws it on its own generator (seed + 919) with the draw key
``part_pose`` (nominal draws: the scene's look). In the online loop it is a TRAINING-view option
(``OnlineConfig.gripper_look``, merged into the training views' ``real_dr`` by ``online_views_cfg``, like
``plate_mm``): the student's saved config, its evaluation views and the robot never see it.
``OnlineConfig.require_scene_sha256`` makes a run refuse a student whose scene file is not the named one.

- Robot timing and servo in the online loop (``OnlineConfig.dyn`` -> ``joint_env.RobotDynConfig``: command delay with
  jitter, weaker arm servo, command lag, gravity sag, command noise; per world per episode), robot-like qdot inputs
  (``qd_robot_p``: ``robot_qd``, the slope of the quantised q, exactly 0 when still), longer episodes (``horizon``,
  ``hold_cap_s``) and a per-sample scale of the loss-side qdot noise (``qd_noise_zero_p``). Evaluation:
  ``$NEDM_SO101_DYN`` (``joint_teacher.make_env``).
- The student's own last command (``ViralConfig.lastcmd``; VIRAL feeds its student the integrated command and the
  last raw output): 12 more robot inputs [last target - q (6), (last target - the target before) / 20 ms (6)], the
  targets of the last two 20 ms steps before the decision as executed (sim: ``JointVecEnv.u_exe[:, -1]`` / ``[:, -2]``
  after the action layer, before the robot dynamics; robot: ``hw_lib.Guard.prev``, the command without the integral
  correction), 0 at an episode start (``lastcmd_raw``), fixed scales ``LASTCMD_DQ_SCALE`` / ``LASTCMD_V_SCALE`` (not
  data statistics). Its own zero-initialised input layer (``ViralNet.lastcmd_in``) added to the head's first layer:
  ``expand_lastcmd`` turns a student without it into one that acts exactly like it at the start. Online loop
  (``OnlineDAgger``: acting and the buffer, teacher slots too), evaluation (``Actor.env``), robot (``hw_lib``).

Both "last command" inputs are independent options (merge 2026-10-23): the studentlab ``cmd_in`` (the last decision's 5
executed targets, columns of the head's first layer; robot: ``Guard.u_layer``) and the real2sim ``lastcmd`` (the last
two 20 ms targets, its own zero-init layer; robot: ``Guard.prev``). With both on, the robot vector is [q, qd, history,
cmd (n_exec x 6), lastcmd (12)] (``ViralConfig.robot_dim``).
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import os
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
CMD_SCALE = tuple(float(v) for v in JE.LEAD) + (0.1,)    # cmd input scale (rad): the lead band (arm), 0.1 rad (jaw)


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


# ---- real2sim: image-space camera effects -----------------------------------------------------------------------------
@dataclass
class CamFX:
    """Image-space effects of a real camera on float images in 0-1 (``to_input``), one draw per image (each camera
    separately). ``fixed_*``: what the real camera always has (every sim image: training and both evaluations; 0 =
    off); the others are random, training images only, each with probability ``p_*``. Order: vignetting, motion blur,
    blur (optics), then the colour augmentation of ``DRConfig``, then white balance, gamma, sensor noise, JPEG."""
    fixed_blur: float = 0.0             # Gaussian sigma (output px), e.g. for the real 480 -> 224 area downscale
    fixed_vignette: float = 0.0         # x * (1 - v r^2), r = 1 at the image corner
    fixed_jpeg: int = 0                 # JPEG quality 1-100 (0: off)
    p_vignette: float = 0.0
    vignette: tuple = (0.0, 0.3)
    p_motion: float = 0.0
    motion_px: tuple = (0.0, 4.0)       # motion blur: line length (output px), any direction
    p_blur: float = 0.0
    blur: tuple = (0.2, 1.0)            # Gaussian sigma (output px)
    p_wb: float = 0.0
    wb: float = 0.08                    # white balance: per-channel gain 1 +- wb
    p_gamma: float = 0.0
    gamma: tuple = (0.85, 1.2)          # x ** g
    p_noise: float = 0.0
    shot: tuple = (0.0, 0.004)          # sensor noise: variance = shot * x + read ** 2
    read: tuple = (0.0, 0.02)
    p_jpeg: float = 0.0
    jpeg: tuple = (50.0, 95.0)          # JPEG quality (standard IJG tables, 4:2:0 chroma)

    @staticmethod
    def for_cameras(d, cameras):
        """{CamFX keys, "per_camera": {camera: {keys}}} -> one CamFX per camera (unknown keys refused)."""
        d = dict(d or {})
        per = d.pop("per_camera", {}) or {}
        for k in ("note", "notes"):
            d.pop(k, None)
        known = set(CamFX.__dataclass_fields__)
        bad = sorted((set(d) | {k for v in per.values() for k in v}) - known - {"note"})
        if bad:
            raise ValueError(f"unknown camera effect keys {bad}")
        if set(per) - set(cameras):
            raise ValueError(f"camera effects for {sorted(set(per) - set(cameras))}: not cameras of {cameras}")
        out = []
        for c in cameras:
            kw = {**d, **{k: v for k, v in (per.get(c) or {}).items() if k != "note"}}
            out.append(CamFX(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in kw.items()}))
        return out


_JPEG_Y = torch.tensor([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
                        [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
                        [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
                        [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], dtype=torch.float32)
_JPEG_C = torch.tensor([[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
                        [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99]] + [[99] * 8] * 4,
                       dtype=torch.float32)


def _dct8(device):
    k, n = torch.arange(8, dtype=torch.float32), torch.arange(8, dtype=torch.float32)
    D = torch.cos(math.pi * (2 * n[None] + 1) * k[:, None] / 16) * math.sqrt(2 / 8)
    D[0] /= math.sqrt(2)
    return D.to(device)


def _jpeg_tables(quality, base):
    """IJG quality scaling: quality [N] (1-100) -> tables [N, 8, 8]."""
    q = quality.clamp(1, 100)
    s = torch.where(q < 50, 5000 / q, 200 - 2 * q)
    return ((base.to(q.device)[None] * s[:, None, None] + 50) // 100).clamp(1, 255)


def _block_q(ch, table, D):
    """[N, c, h, w] (0-255) -> each 8 x 8 block DCT-quantised with table [N, 8, 8] and back."""
    N, c, h, w = ch.shape
    ph, pw = (-h) % 8, (-w) % 8
    x = F.pad(ch, (0, pw, 0, ph), mode="replicate") if ph or pw else ch
    H, W = x.shape[2:]
    b = x.reshape(N, c, H // 8, 8, W // 8, 8).permute(0, 1, 2, 4, 3, 5) - 128.0          # [N, c, hb, wb, 8, 8]
    co = D @ b @ D.T
    t = table[:, None, None, None]
    co = torch.round(co / t) * t
    b = D.T @ co @ D + 128.0
    return b.permute(0, 1, 2, 4, 3, 5).reshape(N, c, H, W)[:, :, :h, :w]


def jpeg_sim(x, quality):
    """JPEG round trip of float images x [N, 3, H, W] in 0-1 at ``quality`` [N]: YCbCr (JFIF), 4:2:0 chroma
    (2 x 2 mean, bilinear back), 8 x 8 DCT with the standard tables scaled by quality, 8-bit output."""
    D = _dct8(x.device)
    v = x * 255.0
    r, g, b = v[:, 0:1], v[:, 1:2], v[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    H, W = x.shape[2:]
    c = F.avg_pool2d(torch.cat((cb, cr), 1), 2, ceil_mode=True)
    y = _block_q(y, _jpeg_tables(quality, _JPEG_Y), D)
    c = _block_q(c, _jpeg_tables(quality, _JPEG_C), D)
    c = F.interpolate(c, size=(H, W), mode="bilinear", align_corners=False)
    cb, cr = c[:, :1] - 128.0, c[:, 1:] - 128.0
    out = torch.cat((y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb), 1)
    return torch.round(out.clamp(0, 255)) / 255.0


def _gauss(x, sig):
    """Separable Gaussian blur of x [N, 3, H, W] with sigma [N] (px; <= 0.05: unchanged)."""
    N = x.shape[0]
    rad = int(min(6, max(1, math.ceil(3 * float(sig.max())))))
    k = torch.arange(-rad, rad + 1, device=x.device, dtype=x.dtype)
    s = sig.clamp_min(0.05).to(x.dtype)
    w = torch.exp(-(k[None] ** 2) / (2 * s[:, None] ** 2))
    w = w / w.sum(-1, keepdim=True)
    xb = x.reshape(1, N * 3, *x.shape[2:])
    wk = w.repeat_interleave(3, 0)
    xb = F.conv2d(F.pad(xb, (rad, rad, 0, 0), mode="replicate"), wk[:, None, None, :], groups=N * 3)
    xb = F.conv2d(F.pad(xb, (0, 0, rad, rad), mode="replicate"), wk[:, None, :, None], groups=N * 3)
    return torch.where((sig > 0.05)[:, None, None, None], xb.reshape_as(x), x)


def _motion(x, length, angle, taps=7):
    """Motion blur: the mean of ``taps`` copies of x [N, 3, H, W] shifted along ``angle`` [N] over ``length`` [N] px."""
    N, _, H, W = x.shape
    theta = torch.zeros(N, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = theta[:, 1, 1] = 1.0
    grid = F.affine_grid(theta, (N, 3, H, W), align_corners=False)
    acc = torch.zeros_like(x)
    for t in torch.linspace(-0.5, 0.5, taps).tolist():
        g = grid.clone()
        g[..., 0] += (2.0 / W) * (t * length * torch.cos(angle))[:, None, None]
        g[..., 1] += (2.0 / H) * (t * length * torch.sin(angle))[:, None, None]
        acc += F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=False)
    return acc / taps


def _vignette(x, v):
    N, _, H, W = x.shape
    yy = (torch.arange(H, device=x.device, dtype=x.dtype) + 0.5) / H - 0.5
    xx = (torch.arange(W, device=x.device, dtype=x.dtype) + 0.5) / W - 0.5
    r2 = (xx[None, :] ** 2 + yy[:, None] ** 2) / 0.5                          # 1 at the corners
    return x * (1 - v.to(x.dtype)[:, None, None, None] * r2[None, None])


def _rand(N, gen, dev):
    return lambda lo, hi: lo + (hi - lo) * torch.rand(N, device=dev, generator=gen)


def fx_optics(x, fx: CamFX, gen=None, random=False):
    """Vignetting, motion blur, blur (``CamFX``) of x [N, 3, H, W] in 0-1."""
    N, dev = x.shape[0], x.device
    if fx.fixed_vignette > 0:
        x = _vignette(x, torch.full((N,), float(fx.fixed_vignette), device=dev))
    if random and fx.p_vignette > 0:
        u = _rand(N, gen, dev)
        x = _vignette(x, (u(0, 1) < fx.p_vignette).float() * u(*fx.vignette))
    if random and fx.p_motion > 0:
        u = _rand(N, gen, dev)
        on = u(0, 1) < fx.p_motion
        L, a = on.float() * u(*fx.motion_px), u(0, 2 * math.pi)
        if bool(on.any()):
            x = torch.where(on[:, None, None, None], _motion(x, L, a), x)
    sig = torch.full((N,), float(fx.fixed_blur), device=dev)
    if random and fx.p_blur > 0:
        u = _rand(N, gen, dev)
        on = u(0, 1) < fx.p_blur
        sig = torch.where(on, torch.sqrt(sig ** 2 + u(*fx.blur) ** 2), sig)
    if bool((sig > 0.05).any()):
        x = _gauss(x, sig)
    return x.clamp(0, 1)


def fx_sensor(x, fx: CamFX, gen=None, random=False):
    """White balance, gamma, sensor noise, JPEG (``CamFX``) of x [N, 3, H, W] in 0-1."""
    N, dev = x.shape[0], x.device
    if random and fx.p_wb > 0:
        u = _rand(N, gen, dev)
        on = (u(0, 1) < fx.p_wb).float()[:, None]
        gain = 1 + on * (2 * torch.rand(N, 3, device=dev, generator=gen) - 1) * fx.wb
        x = (x * gain[:, :, None, None]).clamp(0, 1)
    if random and fx.p_gamma > 0:
        u = _rand(N, gen, dev)
        on = u(0, 1) < fx.p_gamma
        g = torch.where(on, u(*fx.gamma), torch.ones(N, device=dev))
        x = x.clamp_min(0) ** g[:, None, None, None]
    if random and fx.p_noise > 0:
        u = _rand(N, gen, dev)
        on = (u(0, 1) < fx.p_noise).float()[:, None, None, None]
        sh, rd = u(*fx.shot)[:, None, None, None], u(*fx.read)[:, None, None, None]
        x = (x + on * torch.sqrt(sh * x.clamp_min(0) + rd ** 2) * torch.randn(x.shape, device=dev, generator=gen))
        x = x.clamp(0, 1)
    q = torch.full((N,), float(fx.fixed_jpeg), device=dev)
    if random and fx.p_jpeg > 0:
        u = _rand(N, gen, dev)
        on = u(0, 1) < fx.p_jpeg
        r = u(*fx.jpeg)
        q = torch.where(on & ((q <= 0) | (r < q)), r, q)        # the stronger of the fixed and the random JPEG
    if bool((q > 0).any()):
        i = (q > 0).nonzero().flatten()
        x = x.clone()
        x[i] = jpeg_sim(x[i], q[i])
    return x


# ---- real2sim: the real scene of a student config ---------------------------------------------------------------------
REAL_CAMS = ("external", "wrist")
# top-level keys of a scene file that ``RealScene.from_scene_dict`` reads (others: a warning) and meta keys
SCENE_KEYS = {"version", "calib", "geometry", "colors", "light", "student_view", "tone", "render", "dr", "image_fx"}
SCENE_META_KEYS = {"note", "notes", "source", "sources", "evidence", "created", "author", "description", "measured",
                   "status", "todo"}


def _sha(path):
    return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()


@dataclass
class RealSetup:
    """A config's resolved real2sim scene (``load_real_setup``)."""
    scene_path: str
    scene_sha256: str
    calib_path: str
    calib_sha256: str
    real: object                # nedm.render.so101.RealScene (crop None: the student-view rules)
    crops: dict                 # camera -> [x, y, w, h] of the real image (the resolved student view)
    dr: object                  # nedm.render.so101.RealDR (the scene's "dr" + cfg.real_dr)
    image_fx: dict | None       # camera -> render.camera_fx.ImageFX dict; None: none
    cam_fx: dict                # CamFX keys (+ per_camera): the config's extra student-side effects


def _merge(base: dict, over: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in (base or {}).items()}
    for k, v in (over or {}).items():
        out[k] = {**out.get(k, {}), **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_real_setup(cfg: "ViralConfig", geometry_over: dict | None = None) -> RealSetup:
    """The real scene of ``cfg`` (``cfg.scene`` set): the render agent's scene file (``real_scene_v1.json`` layout,
    read by ``nedm.render.so101.RealScene.from_scene_dict``: calib, geometry, colors, light, student_view, tone,
    render.supersample, dr, image_fx) or a bare calibration contract (``RealScene`` defaults). ``cfg.calib`` (if set)
    replaces the file's calibration path (a relative path in the file is relative to the file). The crop of every
    calibrated camera follows the student-view rules (``nedm.render.student_view``, shared with the robot code);
    ``cfg.crops`` is only the pinned record (``pin_real``). ``cfg.real_dr`` is merged over the scene's ``dr``,
    ``cfg.cam_fx`` adds student-side image effects (``CamFX``). ``geometry_over``: evaluation-only overrides of the
    scene file's ``geometry`` section (e.g. ``table_plane``), applied after the file is read (the sha256 stays the
    file's own)."""
    from nedm.render.so101 import RealDR, RealScene, load_calibration
    sp = Path(cfg.scene).expanduser()
    raw = json.loads(sp.read_text())
    if geometry_over:
        bad = sorted(set(geometry_over) - set(RealScene.GEOMETRY_KEYS))
        if bad or not str(raw.get("version", "")).startswith("real_scene"):
            raise ValueError(f"geometry_over {sorted(geometry_over)}: needs a scene file and RealScene geometry keys")
        raw["geometry"] = {**(raw.get("geometry") or {}), **geometry_over}
    is_scene = str(raw.get("version", "")).startswith("real_scene")
    if is_scene:
        odd = sorted(k for k in raw if k not in SCENE_KEYS | SCENE_META_KEYS and not k.startswith("_"))
        if odd:
            print(f"WARNING: scene file {sp}: keys {odd} are not read by RealScene.from_scene_dict", flush=True)
        calib_p = cfg.calib or raw.get("calib") or ""
        if not calib_p:
            raise ValueError(f"no calibration: set ViralConfig.calib or the scene file's 'calib' ({sp})")
        cp = Path(calib_p).expanduser()
        if not cp.is_absolute():
            cp = (sp.parent / cp).resolve()
        calib = load_calibration(cp)
        real = RealScene.from_scene_dict(raw, calib=calib, source=str(sp))
    else:
        if cfg.calib:
            raise ValueError(f"{sp} is a calibration contract, not a scene file: leave ViralConfig.calib empty")
        cp, real = sp, RealScene(calib=raw, source=str(sp))
    crops = {}
    for c in cfg.cameras:
        if c not in REAL_CAMS:
            continue
        v = real.view(c, cfg.res)
        if v is not None and (tuple(v.out_wh) != (cfg.res, cfg.res) or v.interp != "area"):
            raise ValueError(f"student view of {c!r}: output {tuple(v.out_wh)} {v.interp!r}, the student needs "
                             f"{cfg.res} x {cfg.res} 'area' (what the renderer matches)")
        crops[c] = [int(x) for x in real.crop_box(c, cfg.res)]
    dr = RealDR.from_dict(_merge(getattr(real, "dr", None) or {}, cfg.real_dr or {}))
    if dr.addon_on:                         # base add-on randomisation (scene v4 round): its slot bodies in the scene
        from dataclasses import replace as _replace
        real = _replace(real, addon_slots=dr.addon_slots())
    if getattr(dr, "look_on", False):       # v12 gripper look: its slot parts and movable screws in the scene
        from dataclasses import replace as _replace
        real = _replace(real, look_slots=dr.look_slots(real))
    ifx = getattr(real, "image_fx", None)
    if ifx:
        ifx = {k: v for k, v in ifx.items() if k not in ("note", "notes")}
        per = set(ifx) <= set(REAL_CAMS) | {"top", "front"}
        if per and [c for c in cfg.cameras if c not in ifx]:
            raise ValueError(f"image_fx: no entry for cameras {[c for c in cfg.cameras if c not in ifx]}")
        ifx = {c: dict(ifx[c]) if per else dict(ifx) for c in cfg.cameras}
    else:
        ifx = None
    cam_fx = dict(cfg.cam_fx or {})
    CamFX.for_cameras(cam_fx, cfg.cameras)                              # validate now
    return RealSetup(str(sp), _sha(sp), str(cp), _sha(cp), real, crops, dr, ifx, cam_fx)


def pin_real(cfg: "ViralConfig", allow_change=False) -> "ViralConfig":
    """Fix the real scene in a (training) config: the resolved crops and the sha256 of the scene and calibration
    files (checked by ``make_views``: a changed file is refused unless ``allow_change``). Old configs: unchanged."""
    if not cfg.scene:
        return cfg
    rs = load_real_setup(cfg)
    for k, v in (("scene_sha256", rs.scene_sha256), ("calib_sha256", rs.calib_sha256)):
        old = getattr(cfg, k)
        if old and old != v and not allow_change:
            raise ValueError(f"{k}: the file changed since this student was configured ({old[:12]} -> {v[:12]}); "
                             "pass allow_change (--allow-scene-change) to use the new file knowingly")
        setattr(cfg, k, v)
    if cfg.crops and {c: list(b) for c, b in cfg.crops.items()} != rs.crops and not allow_change:
        raise ValueError(f"crops {cfg.crops} != the scene's student view {rs.crops}")
    cfg.crops = {c: list(b) for c, b in rs.crops.items()}
    return cfg


def make_views(cfg: "ViralConfig", num_worlds, seed=0, device="cuda", allow_change=False, geometry_over=None, **kw):
    """The student's views for config ``cfg``: build9's sim scene (``cfg.scene`` empty; exactly the old ``DRViews``
    call) or the real scene (``load_real_setup``; the scene / calibration sha256 and the crops pinned in ``cfg`` must
    match the files, unless ``allow_change``). ``geometry_over``: evaluation-only scene geometry overrides (real scene
    only, ``load_real_setup``), e.g. ``{"table_plane": (c0, c1, c2)}`` for a lowered table."""
    if not cfg.scene:
        if geometry_over:
            raise ValueError("geometry_over needs the real scene (cfg.scene)")
        return DRViews(num_worlds, cfg.res, cfg.cameras, dr=DRConfig.from_dict(cfg.dr), seed=seed, device=device, **kw)
    rs = load_real_setup(cfg, geometry_over)
    msgs = []
    for k, v in (("scene_sha256", rs.scene_sha256), ("calib_sha256", rs.calib_sha256)):
        old = getattr(cfg, k)
        if old and old != v:
            msgs.append(f"{k} of the student {old[:12]} != the file's {v[:12]} "
                        f"({rs.scene_path if 'scene' in k else rs.calib_path})")
    if cfg.crops and {c: list(b) for c, b in cfg.crops.items()} != rs.crops:
        msgs.append(f"crops of the student {cfg.crops} != the scene's student view {rs.crops}")
    if msgs:
        if not allow_change:
            raise ValueError("; ".join(msgs) + ": pass --allow-scene-change to render with the new files")
        print("WARNING: " + "; ".join(msgs), flush=True)
    return DRViews(num_worlds, cfg.res, cfg.cameras, dr=DRConfig.from_dict(cfg.dr), seed=seed, device=device, real=rs,
                   **kw)


class ViewFX:
    """Image-space model of the real cameras for ``to_input``: the config's ``CamFX`` (per camera) and the scene
    file's ``image_fx`` (``nedm.render.camera_fx.ImageFX`` per camera; nominal effects always, one random draw per
    image on training images). Order: CamFX optics, the DRConfig colour augmentation, CamFX sensor, ImageFX."""

    def __init__(self, cam_fx: list, image_fx: list | None):
        self.cam, self.image = list(cam_fx), (list(image_fx) if image_fx is not None else None)

    @staticmethod
    def from_setup(rs: RealSetup, cameras):
        img = None
        if rs.image_fx is not None:
            from nedm.render.camera_fx import ImageFX
            img = [ImageFX.from_dict(rs.image_fx[c]) for c in cameras]
            for c, f in zip(cameras, img):
                if f.extra:
                    raise ValueError(f"image_fx[{c!r}]: unknown keys {sorted(f.extra)}")
        return ViewFX(CamFX.for_cameras(rs.cam_fx, cameras), img)

    def apply(self, x, gen=None, random=False, augment_fn=None):
        """x [n, C, 3, H, W] float 0-1 -> same."""
        n, C = x.shape[:2]
        x = torch.stack([fx_optics(x[:, c], self.cam[c], gen, random) for c in range(C)], 1)
        if augment_fn is not None:
            x = augment_fn(x.reshape(n * C, *x.shape[2:])).reshape(x.shape)
        x = torch.stack([fx_sensor(x[:, c], self.cam[c], gen, random) for c in range(C)], 1)
        if self.image is not None:
            out = []
            for c in range(C):
                img = x[:, c].permute(0, 2, 3, 1).contiguous()
                draw = self.image[c].draw(n, gen, x.device) if random else None
                out.append(self.image[c].apply(img, draw, gen).permute(0, 3, 1, 2))
            x = torch.stack(out, 1)
        return x


class DRViews:
    """The student's cameras with visual randomisation (module docstring). ``render(q, cube_pos, cube_quat, goal_pose,
    dr=None)``: dr = None (nominal visuals) or a dict of per-row draws (``draw``): colors [n, nb, 3], cam_dpos [n, C, 3],
    cam_drot [n, C, 3]; the light direction is drawn per call when ``light`` is True. -> uint8 [n, C, H, W, 3].

    ``real`` (a ``RealSetup``; ``make_views``): the real scene instead (module docstring): draws = colors [n, nb, 3],
    cam_offset [n, C, 6], env_pose [n, n_env, 7] (``RealDR``) and exposure [n, C] (tone map only), and with the
    background randomisation on (``RealDR.bg``, scene v3; ``bg_cams``) the background layers ``bg_*`` (per world for
    the episode, ``nedm.render.bg_dr``); the scene's lights (randomised per call by ``RealDR.lights`` when ``light``);
    ``fx`` = the cameras' ``CamFX`` (None: sim scene). With the plate thickness randomisation (v8,
    ``RealDR.plate_thickness``; ``plate_on``) the per-world plate thickness ``plate_dz`` [n] as well (visual only;
    the renderer is built with its plate depth; ``render(..., held=)`` [n] bool: cubes the jaws hold are not lifted).
    ``render(..., plane=)`` [n, 3] (real scene without its own table_plane, ``table_render_ok``): every world's table
    plane (``SO101Scene`` world_plane; v8 ``OnlineConfig.table_render``). With the gripper look (v12,
    ``RealDR.gripper_look``; ``look_on``) the draws carry ``part_pose`` [n, K, 7] (the look parts' local poses).

    Per-call state (studentlab image history): ``render(..., keep=True)`` keeps the last call's random light (and
    intrinsics) instead of a new draw (no random numbers used); ``call_id`` counts the changes of the per-call state
    (a new random draw, or random -> nominal), so a caller can tell whether two calls share it."""

    def __init__(self, num_worlds, res=224, cameras=CAMERAS, backend="newton", tiled=True, shadows=True,
                 top_eye=V.TOP_EYE, top_target=V.TOP_TARGET, dr: DRConfig | None = None, seed=0, device="cuda",
                 real: RealSetup | None = None):
        self.real, self.fx = real, None
        self.table_render_ok = False        # per-world table planes (v8; the real scene without its own table_plane)
        if real is not None:
            self._init_real(num_worlds, res, cameras, backend, tiled, shadows, dr, seed, device, real)
            self.table_render_ok = real.real.table_plane is None
            return
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
        self._light_state = "nominal"
        self.call_id = 0            # changes of the per-call state (class docstring)

    # -- the real scene (real2sim) --
    def _init_real(self, num_worlds, res, cameras, backend, tiled, shadows, dr, seed, device, rs: RealSetup):
        from nedm.render.so101 import SO101Renderer
        if backend != "newton":
            raise NotImplementedError("the real scene needs the newton backend")
        self.num_worlds, self.res, self.cameras = int(num_worlds), int(res), tuple(cameras)
        self.dr = dr or DRConfig()
        self.dev = torch.device(device)
        self.calls, self.views, self.seconds = 0, 0, 0.0
        # the renderer checks at construction that no body sits in front of a calibrated camera (check_near)
        self.bg_cams = rs.dr.bg_cameras(self.cameras)       # background layers (RealDR.bg; none: off, as before)
        bgkw = dict(bg_cameras=self.bg_cams, bg_dark_groups=rs.dr.bg_dr().dark_groups) if self.bg_cams else {}
        self.plate_on = bool(getattr(rs.dr, "plate_on", False))     # v8: plate thickness (RealDR.plate_thickness)
        if self.plate_on:
            bgkw["plate_depth"] = rs.dr.plate_depth()
        self.r = SO101Renderer(self.num_worlds, width=self.res, height=self.res, backend="newton", cameras=self.cameras,
                               shadows=shadows, goal="block", real=rs.real, **bgkw)
        self.bg_prior = self.r.bg_prior().to(device) if self.bg_cams else None     # clutter / spot placement
        if tiled:
            from newton._src.sensors.warp_raytrace.types import RenderOrder
            rc = self.r.renderer.backend._camera.render_config
            rc.render_order, rc.tile_width, rc.tile_height = RenderOrder.TILED, 8, 8
        sc = self.r.scene
        self.nb = len(sc.body_names)
        self.base = torch.tensor([m.color for m in sc.scene.bodies], dtype=torch.float32, device=self.dev)
        self.groups = {g: list(v) for g, v in sc.groups.items()}
        self.env0 = sc.env_pose.to(self.dev)
        self.tone0 = sc.tone_params.to(self.dev) if getattr(sc, "tone_params", None) is not None else None
        self.rdr = rs.dr
        self.gen = torch.Generator(device=self.dev).manual_seed(int(seed) + 911)
        self.light_gen = torch.Generator().manual_seed(int(seed) + 913)     # per-call draws (lights, intrinsics): CPU
        # base add-ons (RealDR.base_addon): their own generator, so every other draw (each per-world reset of the
        # online loop) is the same with them on or off (paired arms); None when off (nothing changes)
        self.addon_gen = torch.Generator(device=self.dev).manual_seed(int(seed) + 917) if rs.dr.addon_on else None
        # v12 gripper look (RealDR.gripper_look): its own generator as well (None when off: nothing changes)
        self.look_on = bool(getattr(rs.dr, "look_on", False))
        self.look_gen = torch.Generator(device=self.dev).manual_seed(int(seed) + 919) if self.look_on else None
        self.lights0 = [dict(lt) for lt in rs.real.lights]
        self.models0 = list(sc.camera_models)
        self.intrinsics_dr = float(getattr(self.rdr, "focal", 0.0)) > 0 or float(getattr(self.rdr, "pp_px", 0.0)) > 0
        self._call_state = "nominal"
        self.call_id = 0            # changes of the per-call state (class docstring)
        self.fx = ViewFX.from_setup(rs, self.cameras)

    def nominal_draw(self, n):
        """Per-row draws with the nominal visuals (real scene)."""
        C = len(self.cameras)
        d = dict(colors=self.base.expand(n, self.nb, 3).clone(),
                 cam_offset=torch.zeros(n, C, 6, device=self.dev),
                 env_pose=self.env0.expand(n, *self.env0.shape).clone())
        if self.tone0 is not None:
            d["tone"] = self.tone0.expand(n, *self.tone0.shape).clone()
        if self.bg_cams:
            d.update(self.rdr.bg_nominal(n, self.cameras, self.res, self.dev))
        if self.plate_on:           # the scene's own plate
            d["plate_dz"] = torch.zeros(n, dtype=torch.float64, device=self.dev)
        if self.look_on:            # v12: the scene's own gripper look (screws in place, marks and edge hidden)
            p0 = self.r.scene.look_pose0.to(self.dev)
            d["part_pose"] = p0.expand(n, *p0.shape).clone()
        return d

    def merge_nominal(self, d, nominal):
        """Rows ``nominal`` [n] bool of the draws ``d`` replaced by the nominal visuals (real scene)."""
        nd = self.nominal_draw(len(nominal))
        return {k: torch.where(nominal.view(-1, *([1] * (v.dim() - 1))), nd[k].to(v), v) for k, v in d.items()}

    def _draw_real(self, n):
        kw = {"addon_generator": self.addon_gen} if self.addon_gen is not None else {}
        if self.look_gen is not None:
            kw["look_generator"] = self.look_gen
        d = self.rdr.draw(self.r.scene, n, self.gen, device=self.dev, res=self.res, bg_prior=self.bg_prior, **kw)
        return {k: v.to(self.dev) for k, v in d.items()}

    def _per_call_real(self, nominal, keep=False):
        """Lights (and the intrinsics, if the scene randomises them) of the call: shared by all worlds. ``keep``: a
        random call keeps the last random draw (no new draw) if the last call was random as well."""
        if nominal:
            if self._call_state != "nominal":
                self.r.set_lights(self.lights0)
                if self.intrinsics_dr:
                    self.r.set_camera_models(self.models0)
                self._call_state = "nominal"
                self.call_id += 1
            return
        if keep and self._call_state == "random":
            return
        self.r.set_lights(self.rdr.lights(self.lights0, self.light_gen))
        if self.intrinsics_dr:
            self.r.set_camera_models(self.rdr.camera_models(self.r.scene, self.light_gen, out_res=self.res))
        self._call_state = "random"
        self.call_id += 1

    def _render_real(self, q, cube_pos, cube_quat, goal_pose, dr=None, light=None, keep=False, plane=None, held=None):
        t0 = time.time()
        if plane is not None and not self.table_render_ok:
            raise ValueError("render(plane=): the real scene has its own table_plane")
        self._per_call_real((dr is None) if light is None else not light, keep)
        out = []
        for s in range(0, q.shape[0], self.num_worlds):
            e = min(q.shape[0], s + self.num_worlds)
            d = None if dr is None else {k: v[s:e] for k, v in dr.items()}
            rkw = {}
            if plane is not None:
                rkw["world_plane"] = plane[s:e]
            if held is not None:
                rkw["cube_held"] = held[s:e]
            out.append(self.r.render_dr(q[s:e], cube_pos[s:e], cube_quat[s:e], goal_pose[s:e], draw=d,
                                        **rkw).to(self.dev))
        rgb = torch.cat(out) if len(out) > 1 else out[0]
        self.calls += 1
        self.views += q.shape[0] * len(self.cameras)
        self.seconds += time.time() - t0
        return rgb

    # -- draws --
    def _u(self, n, lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(n, *shape, device=self.dev, generator=self.gen)

    def draw(self, n):
        """Per-row randomisation draws for n rows."""
        if self.real is not None:
            return self._draw_real(n)
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

    def _light(self, nominal, keep=False):
        import warp as wp
        if nominal:
            d = self.light_default
            if self._light_state != "nominal":
                self._light_state = "nominal"
                self.call_id += 1
        else:
            if keep and self._light_state == "random":
                return                                  # the last call's random light stays
            self._light_state = "random"
            self.call_id += 1
            el = math.radians(float(self._u(1, *self.dr.light_elev)[0]))
            az = float(self._u(1, 0.0, 2 * math.pi)[0])
            d = (-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el))
        self.r.renderer.backend._camera.utils.create_default_light(self.r.shadows, direction=wp.vec3f(*d))

    @torch.no_grad()
    def render(self, q, cube_pos, cube_quat, goal_pose, dr=None, light=None, keep=False, plane=None, held=None):
        """``keep``: keep the last call's random per-call state; ``plane`` / ``held`` (real scene): per-world table
        planes / held cubes (class docstring)."""
        if self.real is not None:
            if plane is None and held is None:
                return self._render_real(q, cube_pos, cube_quat, goal_pose, dr, light, keep)
            return self._render_real(q, cube_pos, cube_quat, goal_pose, dr, light, keep, plane, held)
        if plane is not None or held is not None:
            raise ValueError("render(plane=, held=): the real scene only")
        from nedm.render.so101 import qmul, wxyz, xyzw
        t0 = time.time()
        nominal = dr is None
        self._light(nominal if light is None else not light, keep)
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


def to_input(rgb, d: DRConfig | None = None, gen=None, fx: ViewFX | None = None, fx_random=None):
    """uint8 [n, C, H, W, 3] -> float [n, C, 3, H, W], ImageNet-normalised; ``d``: per-image augmentation (training).
    ``fx`` (real scene, ``DRViews.fx``): the cameras' image-space model; its random effects when ``fx_random``
    (default: when ``d`` augments), its fixed / nominal effects always. Real camera frames: no ``fx``."""
    x = rgb.permute(0, 1, 4, 2, 3).float().div_(255.0)
    n, C = x.shape[:2]
    aug = d is not None and d.on
    if fx is not None:
        rnd = aug if fx_random is None else bool(fx_random)
        x = fx.apply(x, gen, rnd, (lambda y: augment(y, d, gen)) if aug else None)
    elif aug:
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
    qd_input_zero: bool = False        # real2sim: the qdot input is always 0 (every path: training, acting, the robot),
                                       # so the student's step size cannot follow the measured arm speed
    hist: int = 0                      # robot input history: [q, qdot] of the previous ``hist`` decisions (online path)
    lastcmd: bool = False              # real2sim: + the student's last executed command (module docstring, 12 inputs
                                       # after the history and the cmd input; online loop, evaluation and robot; not
                                       # the round files; independent of ``cmd_in``)
    # studentlab (2026-10-22; off by default, online path only): the student's own last command and its last images
    cmd_in: bool = False               # + the previous decision's 5 EXECUTED targets (after the action layer) - the
                                       #   measured q now, [5, 6] / cmd_scale (0 at an episode start; teacher-driven
                                       #   worlds: the teacher's executed targets)
    cmd_scale: tuple = CMD_SCALE       # rad per joint: the lead band (arm) and 0.1 rad (jaw)
    img_hist: bool = False             # + the previous decision's images of both cameras through the same encoders
                                       #   (the current images at an episode start)
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
    # ---- real2sim (module docstring; defaults = build9's sim scene, no lag) ----
    scene: str = ""                               # real-scene file (JSON, ``load_real_setup``); "" = the sim scene
    calib: str = ""                               # calibration contract; "" = the scene file's "calib"
    crops: dict = field(default_factory=dict)     # {camera: [x, y, w, h]} real-image crop (resolved by ``pin_real``)
    real_dr: dict = field(default_factory=dict)   # RealDR overrides (+ exposure_log2) over the scene file's "dr"
    cam_fx: dict = field(default_factory=dict)    # CamFX: extra image-space effects (student side)
    lag: int = 0                                  # image lag (20 ms control steps): evaluation / deployment nominal
    lag_range: tuple = (0, 0)                     # training: lag drawn per world per episode, inclusive
    scene_sha256: str = ""                        # pinned at the start of training (``pin_real``)
    calib_sha256: str = ""

    @staticmethod
    def from_dict(d):
        d = dict(d or {})
        known = set(ViralConfig.__dataclass_fields__)
        bad = sorted(set(d) - known - {"note", "notes"})
        if bad:
            raise ValueError(f"unknown viral config keys {bad}")
        for k in ("cameras", "hidden", "lag_range", "cmd_scale"):
            if k in d:
                d[k] = tuple(d[k])
        cfg = ViralConfig(**{k: v for k, v in d.items() if k in known})
        if len(cfg.cmd_scale) != 6 or min(float(v) for v in cfg.cmd_scale) <= 0:
            raise ValueError(f"cmd_scale {cfg.cmd_scale}: 6 positive values (rad)")
        return cfg

    @property
    def lag_max(self):
        return max(int(self.lag), int(self.lag_range[1]))

    @property
    def robot_dim(self):
        """Length of the robot input vector: [q, qd] now + ``hist`` decisions before (12 each), then the cmd input
        (``cmd_in``: n_exec x 6), then the last command (``lastcmd``: 12)."""
        return 12 * (1 + int(self.hist)) + (int(self.n_exec) * 6 if getattr(self, "cmd_in", False) else 0) + \
            12 * int(bool(getattr(self, "lastcmd", False)))


STAT_KEYS = ("robot_mean", "robot_std", "tgt_mean", "tgt_std", "aux_mean", "aux_std")
AUX_DIM = 7
# the last-command input (``ViralConfig.lastcmd``): fixed scales, NOT data statistics (a lastcmd checkpoint depends on
# them: never change them; another scale = another option). Last target - q: arm 0.05 rad (the action layer's lead
# band is 0.045-0.12 rad; real robot 10-22: |prev - q| p99 0.02 rad, the dead band 0.006-0.013 rad), jaw 0.25 rad (it
# squeezes the cube: real p50 0.12, max 0.37 rad). Target change per 20 ms: arm 1 rad/s (real p99 0.6-0.9, max 1.9),
# jaw 5 rad/s (real p99 3.9 rad/s). Normalised values are clamped to +-10 like the other robot inputs.
LASTCMD_DQ_SCALE = (0.05, 0.05, 0.05, 0.05, 0.05, 0.25)
LASTCMD_V_SCALE = (1.0, 1.0, 1.0, 1.0, 1.0, 5.0)


def lastcmd_raw(u_last, u_prev, q, fresh=None):
    """The last-command input before normalisation (``ViralConfig.lastcmd``; the same function for the sim and the
    robot): u_last, u_prev [B, 6] the targets executed in the last and the second-to-last 20 ms step before the decision
    (sim: ``JointVecEnv.u_exe[:, -1]``, ``[:, -2]``; robot: ``hw_lib.Guard.prev`` of those steps), q [B, 6] the
    measured q at the decision (any joint frame: only differences) -> [B, 12] float64 = [u_last - q, (u_last - u_prev)
    / 20 ms]; ``fresh`` [B] bool: rows at an episode start (no command yet) -> 0; non-finite entries -> 0."""
    u1, u0 = u_last.to(torch.float64), u_prev.to(device=u_last.device, dtype=torch.float64)
    out = torch.cat((u1 - q.to(u1), (u1 - u0) / JE.CONTROL_DT), -1)
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    if fresh is not None:
        out = torch.where(fresh.to(out.device).view(-1, 1), torch.zeros_like(out), out)
    return out


def lastcmd_zeros(B, device):
    """[B, 12] float64: the last-command input at an episode start."""
    return torch.zeros(int(B), 12, dtype=torch.float64, device=device)


NEW_INPUT_KEYS = ("cmd_in", "cmd_scale", "img_hist")      # left out of saved configs while off (``cfg_dict``)
ARCH_KEYS = ("hist", "cmd_in", "img_hist")                 # change the head's input: ``expand_student``


def cfg_dict(cfg: ViralConfig) -> dict:
    """The config as saved in checkpoints / meta files: ``asdict`` without the studentlab input options while they are
    off (so a student without them stays readable by code that predates them); with any of them on, all are saved.
    The same for the last-command input (``lastcmd``): no key while it is off."""
    d = asdict(cfg)
    if not cfg.cmd_in and not cfg.img_hist and tuple(cfg.cmd_scale) == CMD_SCALE:
        for k in NEW_INPUT_KEYS:
            d.pop(k, None)
    if not d.get("lastcmd"):
        d.pop("lastcmd", None)
    return d


class ViralNet(nn.Module):
    def __init__(self, cfg: ViralConfig, stats, pretrained=True):
        super().__init__()
        from nedm.so101_policy.vision.perstep import Encoder, PerStepConfig
        self.cfg = cfg
        pc = PerStepConfig(encoder=cfg.encoder, stages=cfg.stages, keypoints=cfg.keypoints, pool_dim=cfg.pool_dim)
        self.enc = nn.ModuleList([Encoder(pc, pretrained) for _ in cfg.cameras])
        fd = sum(e.out_dim for e in self.enc)
        self.fd = fd
        self.n_cmd = int(cfg.n_exec) * 6 if getattr(cfg, "cmd_in", False) else 0
        if self.n_cmd:
            self.register_buffer("cmd_s", torch.tensor(cfg.cmd_scale, dtype=torch.float32).repeat(int(cfg.n_exec)),
                                 persistent=False)
        layers, k = [], fd + 12 * (1 + int(cfg.hist)) + self.n_cmd + (fd if getattr(cfg, "img_hist", False) else 0)
        for h in cfg.hidden:
            layers += [nn.Linear(k, h), nn.ELU()]
            k = h
        layers.append(nn.Linear(k, cfg.chunk * 6))
        self.head = nn.Sequential(*layers)
        self.aux = nn.Sequential(nn.Linear(fd, cfg.aux_hidden), nn.ELU(), nn.Linear(cfg.aux_hidden, AUX_DIM))
        for k_, v in stats.items():
            self.register_buffer(k_, torch.as_tensor(v, dtype=torch.float32).clone())
        self.lastcmd = bool(getattr(cfg, "lastcmd", False))
        if self.lastcmd:        # the last-command inputs: their own weights into the head's first layer, zero at start
            self.lastcmd_in = nn.Linear(12, self.head[0].out_features, bias=False)
            nn.init.zeros_(self.lastcmd_in.weight)
            self.register_buffer("lastcmd_scale", torch.tensor(LASTCMD_DQ_SCALE + LASTCMD_V_SCALE))  # saved with the checkpoint

    def features(self, images):
        return torch.cat([e(images[:, i]) for i, e in enumerate(self.enc)], -1)

    def lastcmd_norm(self, lc):
        """[B, 12] raw last-command input (``lastcmd_raw``) -> / the fixed scales, clamped to +-10 (no noise)."""
        return (lc.float() / self.lastcmd_scale).clamp(-10, 10)

    def robot(self, robot, qd_scale=None):
        """[B, 12 (1 + hist) (+ 30 cmd)] = [q, qd] now, then the previous decisions (newest first), then (``cmd_in``)
        the previous decision's executed targets - q now ([5, 6] flattened) -> normalised (+ qdot noise in training on
        every qdot block; ``qd_scale`` [B]: per-sample factor on that noise, None = 1; the cmd part: / cmd_scale)."""
        H = 1 + int(self.cfg.hist)
        nc = getattr(self, "n_cmd", 0)
        if nc:
            cmd = (robot[:, 12 * H:12 * H + nc] / self.cmd_s).clamp(-10, 10)
            return torch.cat((ViralNet._robot(self, robot[:, :12 * H].contiguous(), H, qd_scale), cmd), -1)
        return ViralNet._robot(self, robot, H, qd_scale)

    def _robot(self, robot, H, qd_scale):
        if getattr(self.cfg, "qd_input_zero", False):
            robot = robot.view(-1, H, 12)
            robot = torch.cat((robot[..., :6], torch.zeros_like(robot[..., 6:])), -1)
        r = ((robot.view(-1, H, 12) - self.robot_mean) / self.robot_std).clamp(-10, 10)
        if self.training and self.cfg.qd_noise > 0:
            n = self.cfg.qd_noise * torch.randn_like(r[..., 6:])
            if qd_scale is not None:
                n = n * qd_scale.to(n).view(-1, 1, 1)
            r = torch.cat((r[..., :6], r[..., 6:] + n), -1)
        return r.reshape(-1, 12 * H)

    def _split_lastcmd(self, robot):
        """lastcmd students: robot [B, robot_dim] -> (the robot vector without the last command: [q, qd, history,
        (cmd)], the raw last command [B, 12] = the last 12 values)."""
        n = 12 * (1 + int(self.cfg.hist)) + int(getattr(self, "n_cmd", 0))
        if robot.shape[-1] != n + 12:
            raise ValueError(f"lastcmd student: robot input of {robot.shape[-1]} values, needs {n + 12}")
        return robot[:, :n].contiguous(), robot[:, n:n + 12]

    def _head(self, x, lc=None):
        """The head on its input x [B, k] float32 -> [B, chunk, 6]; lastcmd: the first layer's output + the
        last-command term of the raw last command ``lc`` (exactly + 0 while its weights are zero: the student it was
        expanded from, bit for bit)."""
        if not self.lastcmd:
            return self.head(x).view(-1, self.cfg.chunk, 6)
        h = self.head[0](x)
        h = h + self.lastcmd_in(self.lastcmd_norm(lc)).to(h.dtype)
        return self.head[1:](h).view(-1, self.cfg.chunk, 6)

    def forward(self, images, robot, qd_scale=None, prev=None):
        """``prev`` (``img_hist``): the previous decision's images [B, C, 3, H, W] (None: the current ones, as at an
        episode start); both through the same encoders, the previous features appended to the head input.
        ``robot`` = [q, qd, history (12 (1 + hist)), (cmd_in: n_exec x 6), (lastcmd: 12)] (``cfg.robot_dim``)."""
        lc = None
        if self.lastcmd:
            robot, lc = self._split_lastcmd(robot)
        if getattr(self.cfg, "img_hist", False):
            B = images.shape[0]
            if prev is None:
                f = self.features(images)
                fp = f
            else:
                both = self.features(torch.cat((images, prev.to(images)), 0))
                f, fp = both[:B], both[B:]
            x = torch.cat((f, self.robot(robot, qd_scale).to(f.dtype), fp), -1)
            return self._head(x.float(), lc), f
        f = self.features(images)
        return self._head(torch.cat((f, self.robot(robot, qd_scale).to(f.dtype)), -1).float(), lc), f

    def head_from_features(self, f, robot, fp=None):
        """The normalised chunk [B, chunk, 6] from the encoder features ``f`` [B, fd] (``img_hist``: + ``fp``, the
        previous decision's features; None = ``f``, as at an episode start) = ``forward`` without its encoder pass
        (the robot caches the previous query's features: ``hw_lib.StudentPolicy``)."""
        lc = None
        if self.lastcmd:
            robot, lc = self._split_lastcmd(robot)
        parts = [f, self.robot(robot).to(f.dtype)]
        if getattr(self.cfg, "img_hist", False):
            parts.append(f if fp is None else fp.to(f.dtype))
        return self._head(torch.cat(parts, -1).float(), lc)

    def loss(self, images, robot, tgt_n, valid, aux_n=None, qd_scale=None, prev=None):
        p, f = self(images, robot, qd_scale, prev)
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
    def predict(self, images, robot, prev=None):
        return self(images, robot, prev=prev)[0].float()

    @torch.no_grad()
    def predict_aux(self, images):
        return self.aux(self.features(images)).float() * self.aux_std + self.aux_mean

    def backbone_params(self):
        return [p for e in self.enc if e.is_backbone for p in e.trunk.parameters() if p.requires_grad]


def head_layout(cfg: ViralConfig, fd: int) -> dict:
    """Column ranges of the head's first layer: features now, robot [q, qd] (now + history), cmd, previous features."""
    out, k = {}, 0
    for name, n in (("feat", fd), ("robot", 12 * (1 + int(cfg.hist))),
                    ("cmd", int(cfg.n_exec) * 6 if cfg.cmd_in else 0), ("feat_prev", fd if cfg.img_hist else 0)):
        out[name] = (k, k + n)
        k += n
    return out


def expand_student(st: ViralNet, hist=None, cmd_in=None, img_hist=None, device="cuda", cmd_scale=None):
    """A student with more inputs (robot history ``hist``, the command input ``cmd_in``, the image history
    ``img_hist``; None = keep) from the given one: the same weights, ZERO weights on the new inputs' columns of the
    head's first layer, so the new student gives the same chunk as the given one on the same inputs (whatever the new
    inputs are). Inputs cannot be removed. ``cmd_scale``: the cmd input scale of a NEW cmd input (default the
    config's)."""
    from dataclasses import replace as _rep
    c0 = st.cfg
    h1 = int(c0.hist) if hist is None else int(hist)
    m1 = bool(c0.cmd_in) if cmd_in is None else bool(cmd_in)
    i1 = bool(c0.img_hist) if img_hist is None else bool(img_hist)
    if h1 < int(c0.hist) or (c0.cmd_in and not m1) or (c0.img_hist and not i1):
        raise ValueError(f"cannot remove inputs: hist {c0.hist} -> {h1}, cmd_in {c0.cmd_in} -> {m1}, "
                         f"img_hist {c0.img_hist} -> {i1}")
    if cmd_scale is not None and c0.cmd_in and tuple(cmd_scale) != tuple(c0.cmd_scale):
        raise ValueError("cmd_scale of an existing cmd input cannot change")
    if (h1, m1, i1) == (int(c0.hist), bool(c0.cmd_in), bool(c0.img_hist)):
        return st
    kw = dict(hist=h1, cmd_in=m1, img_hist=i1)
    if cmd_scale is not None:
        kw["cmd_scale"] = tuple(float(v) for v in cmd_scale)
    cfg = _rep(c0, **kw)
    new = ViralNet(cfg, {k: getattr(st, k) for k in STAT_KEYS}, pretrained=False).to(device)
    sd = st.state_dict()
    nd = new.state_dict()
    lo, ln = head_layout(c0, st.fd), head_layout(cfg, new.fd)
    for k, v in sd.items():
        if k == "head.0.weight":
            w = torch.zeros_like(nd[k])
            for seg, (a, b) in lo.items():
                if b > a:
                    a1 = ln[seg][0]
                    w[:, a1:a1 + (b - a)] = v[:, a:b]
            nd[k] = w
        else:
            nd[k] = v
    new.load_state_dict(nd)
    return new


def expand_lastcmd(st: ViralNet, device="cuda"):
    """A student with the last-command input (``ViralConfig.lastcmd``) from one without: the same weights (any
    history), zero weights on the new inputs (``lastcmd_in``), so it acts exactly as the given student until trained.
    A student that has it already: returned as is."""
    from dataclasses import replace as _rep
    if getattr(st.cfg, "lastcmd", False):
        return st
    new = ViralNet(_rep(st.cfg, lastcmd=True), {k: getattr(st, k) for k in STAT_KEYS}, pretrained=False).to(device)
    miss, extra = new.load_state_dict(st.state_dict(), strict=False)
    if sorted(miss) != ["lastcmd_in.weight", "lastcmd_scale"] or extra:
        raise RuntimeError(f"expand_lastcmd: unexpected state keys (missing {miss}, unexpected {extra})")
    return new.train(st.training)


def expand_hist(st: ViralNet, hist: int, device="cuda"):
    """A student with ``hist`` decisions of robot history from one without (or with fewer): the same weights, zero
    weights on the new history inputs of the head's first layer (so it starts as the given student)."""
    if hist < int(st.cfg.hist):
        raise ValueError("cannot shrink the history")
    return expand_student(st, hist=hist, device=device)


def save_student(st, path, **extra):
    # cfg_dict: no lastcmd / cmd_in / cmd_scale / img_hist keys while they are off (the old checkpoint layout: older
    # code, e.g. a robot bundle, can load it)
    torch.save(dict(cfg=cfg_dict(st.cfg), state_dict=st.state_dict(), stats={k: getattr(st, k).cpu() for k in STAT_KEYS},
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


# ---- real2sim: image lag ------------------------------------------------------------------------------------------------
LAG_REC = 4            # control steps of state history recorded per round-file row (``row_lag``)


def _state13(q, cp, cq):
    return torch.cat((q, cp, cq), -1)


def lag_interp(z, lag):
    """Round-file rows without ``row_lag``: the states 1..lag control steps before each row, interpolated linearly
    between the row and the previous row (100 ms = ``JE.STEPS`` steps earlier) of the same episode (q, cube position;
    quaternion nlerp); the first row of an episode: the row itself. -> [N, lag, 13] float32."""
    q, cp, cq = (np.asarray(z[k], np.float64) for k in ("row_q", "row_cube_pos", "row_cube_quat"))
    cur = np.concatenate((q, cp, cq), -1)
    prev = np.roll(cur, 1, 0)
    first = np.asarray(z["row_t"]) == 0 if "row_t" in z.files else np.isin(np.arange(len(cur)), z["ep_row_start"])
    prev[first] = cur[first]
    dot = (prev[:, 9:13] * cur[:, 9:13]).sum(-1, keepdims=True)
    prev[:, 9:13] = np.where(dot < 0, -prev[:, 9:13], prev[:, 9:13])
    out = []
    for k in range(1, lag + 1):
        a = min(1.0, k / JE.STEPS)
        x = (1 - a) * cur + a * prev
        x[:, 9:13] /= np.linalg.norm(x[:, 9:13], axis=-1, keepdims=True)
        out.append(x)
    return np.stack(out, 1).astype(np.float32)


def lag_rows(r, sub, kmax=LAG_REC):
    """``run_tasks`` record ``r`` + the states after every 20 ms step ``sub`` [B, D, STEPS, 13] (``SubstepLog``) ->
    ``row_lag`` [N, kmax, 13] float32: the state k = 1..kmax control steps before each row (k >= STEPS reaches into
    the decisions before; before the episode start: its own start state, the arm at rest)."""
    if kmax < 1:
        raise ValueError(f"kmax {kmax}: >= 1 control steps")
    S = JE.STEPS
    rb, rt = np.asarray(r["row_ep"], np.int64), np.asarray(r["row_t"], np.int64)
    cur = np.concatenate((r["row_q"], r["row_cube_pos"], r["row_cube_quat"]), -1).astype(np.float32)
    out = np.repeat(cur[:, None], kmax, 1)
    p = rt >= 1
    if p.any():
        prev = sub[rb[p], rt[p] - 1]                                     # [n, STEPS, 13]: the previous decision's steps
        err = float(np.abs(prev[:, S - 1] - cur[p]).max())
        if err > 1e-4:
            raise RuntimeError(f"substep log does not match the recorded rows (max {err:.2e})")
        first = np.zeros(int(rb.max()) + 1, np.int64)                    # the start row of every episode
        first[rb[rt == 0]] = np.nonzero(rt == 0)[0]
        for k in range(1, kmax + 1):
            i = rt[p] * S - k                                            # global step index of the state (0 = start)
            st = i <= 0
            v = sub[rb[p], np.maximum(i - 1, 0) // S, np.maximum(i - 1, 0) % S]
            out[p, k - 1] = np.where(st[:, None], cur[first[rb[p]]], v)
    return out


class SubstepLog:
    """The world state after every 20 ms step of every ``JointVecEnv.step`` (feed it with ``step_tap``):
    ``array()`` -> [B, decisions, STEPS, 13] float32 (q, cube pos, cube quat w x y z)."""

    def __init__(self):
        self.rows = []

    def __call__(self, ws):
        self.rows.append(_state13(ws.q, ws.cube_pos, ws.cube_quat).float().cpu())

    def array(self):
        S = JE.STEPS
        if len(self.rows) % S:
            raise RuntimeError(f"{len(self.rows)} logged steps: not whole decisions")
        a = torch.stack(self.rows, 1).numpy()                                        # [B, D * S, 13]
        return a.reshape(a.shape[0], -1, S, 13)


@contextlib.contextmanager
def step_tap(env, *fns):
    """Call each ``fn(ws)`` after every 20 ms world step of ``env.step`` (any caller, e.g. ``run_tasks``); restores
    ``env.step`` on exit."""
    had = "step" in vars(env)
    orig = env.step

    def step(actions=None, targets=None, use_targets=None, step_hook=None):
        def hook(e, j):
            for f in fns:
                f(e.ws)
            if step_hook is not None:
                step_hook(e, j)
        return orig(actions, targets=targets, use_targets=use_targets, step_hook=hook)

    env.step = step
    try:
        yield env
    finally:
        if had:
            env.step = orig
        else:
            del env.step


def _sane(q, cp, cq, dev):
    """Non-finite rows -> a harmless pose (as ``Actor``)."""
    q, cp, cq = q.to(dev, torch.float32), cp.to(dev, torch.float32), cq.to(dev, torch.float32)
    ok = torch.isfinite(q).all(-1) & torch.isfinite(cp).all(-1) & torch.isfinite(cq).all(-1)
    q = torch.where(ok[:, None], q, torch.zeros_like(q))
    cp = torch.where(ok[:, None], cp, torch.tensor([0.2, 0.0, 0.015], device=dev))
    cq = torch.where(ok[:, None], cq, torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev))
    return q, cp, cq


class LagHistory:
    """Per world the states of the last ``kmax`` control steps (index 0 = now): ``fill`` at an episode start (the arm
    at rest: every entry = the start state), ``push`` after every 20 ms step, ``get(k)`` [B] -> (q, cube pos, quat).

    ``goal`` (``ImgLagConfig``, real2sim v8): also the goal pose [x, y, yaw] of every entry (``fill`` / ``push`` then
    need it), and ``at(lag, goal_now)``: the state a real-valued ``lag`` back (interpolated) with the goal of then."""

    def __init__(self, B, kmax, device, goal=False):
        self.x = torch.zeros(B, int(kmax) + 1, 13, device=device)
        self.g = torch.zeros(B, int(kmax) + 1, 3, device=device) if goal else None
        self.ready = False

    def fill(self, idx, q, cp, cq, goal=None):
        self.x[idx] = _state13(q, cp, cq).to(self.x)[:, None].expand(-1, self.x.shape[1], -1)
        if self.g is not None:
            if goal is None:
                raise ValueError("LagHistory(goal=True).fill: the goal pose is required")
            self.g[idx] = goal.to(self.g)[:, None].expand(-1, self.g.shape[1], -1)
        self.ready = True

    def push(self, q, cp, cq, goal=None):
        self.x = torch.cat((_state13(q, cp, cq).to(self.x)[:, None], self.x[:, :-1]), 1)
        if self.g is not None:
            if goal is None:
                raise ValueError("LagHistory(goal=True).push: the goal pose is required")
            self.g = torch.cat((goal.to(self.g)[:, None], self.g[:, :-1]), 1)

    def newest(self):
        return self.x[:, 0]

    def get(self, k):
        x = self.x[torch.arange(len(self.x), device=self.x.device), k.to(self.x.device).long()]
        return x[:, :6], x[:, 6:9], x[:, 9:13]

    def at(self, lag, goal_now):
        """``lag`` [B] float control steps (0 <= lag <= kmax; above kmax - 1 only whole steps): the state ``lag``
        steps before now, linear between the two neighbouring 20 ms states (q, cube position; cube quaternion nlerp
        on the same hemisphere); a whole step = that entry exactly. The goal (``goal=True``): ``goal_now`` [B, 3] at
        lag 0, else the goal of entry floor(lag) (the goal moves only at a decision boundary, after that decision's
        last push: an image from before the boundary shows the old goal). -> (q, cube pos, cube quat, goal) float32."""
        if self.g is None:
            raise ValueError("LagHistory.at needs goal=True")
        B, dev = len(self.x), self.x.device
        lag = lag.to(dev, torch.float32).clamp(0.0, float(self.x.shape[1] - 1))
        k0 = lag.floor().long()
        a = lag - k0.to(lag)
        k1 = (k0 + 1).clamp_max(self.x.shape[1] - 1)
        ar = torch.arange(B, device=dev)
        s0, s1 = self.x[ar, k0], self.x[ar, k1]
        w = a[:, None]
        q = torch.lerp(s0[:, :6], s1[:, :6], w)
        cp = torch.lerp(s0[:, 6:9], s1[:, 6:9], w)
        q1 = torch.where(((s0[:, 9:13] * s1[:, 9:13]).sum(-1, keepdim=True) < 0), -s1[:, 9:13], s1[:, 9:13])
        cq = torch.lerp(s0[:, 9:13], q1, w)
        cq = cq / cq.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        whole = (a == 0)[:, None]
        q, cp, cq = (torch.where(whole, s0[:, sl], v) for sl, v in ((slice(0, 6), q), (slice(6, 9), cp),
                                                                     (slice(9, 13), cq)))
        g = torch.where((lag == 0)[:, None], goal_now.to(self.g), self.g[ar, k0])
        return q, cp, cq, g


@dataclass
class ImgLagConfig:
    """Image latency of the online loop (``OnlineConfig.img_lag``; real2sim v8, 2026-10-23; measurement and method:
    ~/so101_r2s_claude/v8/latency/: scripts/lag_fit.py, lag_dist.py -> lag_fit.json, lag_boot.json, lag_dist.json).
    On the robot the images of a query are older than its joint reading by the camera pipeline latency before
    LeRobot's frame time stamp (26.5 ms, 95 % CI 25-28, fitted on 47,691 consecutive query pairs of 173 robot
    episodes: thumbnail change vs joint motion between the capture times) + the frame age (uniform 0-33 ms, 30 fps):
    28-62 ms (p1-p99), mean 44 ms = 2.2 control steps; the two cameras' ages are uncorrelated (r -0.13) and change
    from query to query (lag-1 autocorrelation 0.15). The calibration's 42-48 ms (teleop recordings,
    ``time_offset_frames``) is the same latency + the mean age of the recording loop.

    Per world (per camera with ``per_camera``) a lag U(lo_ms, hi_ms) is drawn at every episode start and drawn again at
    each decision with probability ``redraw_p`` (0 = once per episode; the redraw decision is per world, all its
    cameras together); the student's images show the world ``lag /
    20 ms`` control steps before the decision (``LagHistory.at``: interpolated between the 20 ms states, the goal as it
    was then), ``quantize``: whole steps only (the lag rounded). q, qdot and the labels stay current; the aux targets
    follow the image state (first camera). ``per_camera``: one render call per camera (the later calls keep the first
    call's light and intrinsics: ``DRViews.render(keep=True)``), so the render time grows with the cameras. A goal
    jump (``OnlineConfig.goal_jump``) that the lagged images do not show yet: no buffer sample that decision (its label
    is for the new goal). Replaces the student's integer lag (``ViralConfig.lag`` / ``lag_range``) in the online loop;
    the evaluation keeps ``ViralConfig.lag``."""
    lo_ms: float = 0.0
    hi_ms: float = 0.0
    redraw_p: float = 1.0
    per_camera: bool = False
    quantize: bool = False

    MAX_MS = 400.0

    @staticmethod
    def is_on(d) -> bool:
        """An ``OnlineConfig.img_lag`` dict turns the option on when it has any key besides "note"."""
        return bool(set(dict(d or {})) - {"note"})

    @staticmethod
    def from_dict(d):
        d = dict(d or {})
        known = set(ImgLagConfig.__dataclass_fields__)
        bad = sorted(set(d) - known - {"note"})
        if bad:
            raise ValueError(f"unknown img_lag keys {bad}")
        miss = [k for k in ("lo_ms", "hi_ms") if k not in d]
        if ImgLagConfig.is_on(d) and miss:     # (defaults 0 / 0 would silently give no lag and drop lag_range)
            raise ValueError(f"img_lag: give lo_ms and hi_ms explicitly (missing {miss})")
        c = ImgLagConfig(**{k: v for k, v in d.items() if k in known})
        c.lo_ms, c.hi_ms, c.redraw_p = float(c.lo_ms), float(c.hi_ms), float(c.redraw_p)
        c.per_camera, c.quantize = bool(c.per_camera), bool(c.quantize)
        if not 0.0 <= c.lo_ms <= c.hi_ms <= ImgLagConfig.MAX_MS:
            raise ValueError(f"img_lag lo_ms {c.lo_ms}, hi_ms {c.hi_ms}: 0 <= lo <= hi <= {ImgLagConfig.MAX_MS} ms")
        if not 0.0 <= c.redraw_p <= 1.0:
            raise ValueError(f"img_lag redraw_p {c.redraw_p}: a probability in [0, 1]")
        return c

    @property
    def step_ms(self):
        return 1000.0 * JE.CONTROL_DT

    @property
    def kmax(self):
        """Steps of history needed: floor(hi / 20 ms) + 1 (the interpolation's older neighbour); the ring buffer has
        kmax + 1 entries (index 0 = now)."""
        return int(math.floor(self.hi_ms / self.step_ms + 1e-9)) + 1

    def draw(self, n, cams, gen, device):
        """[n, cams] float32 lags in control steps."""
        u = torch.rand(n, cams, device=device, generator=gen)
        x = (self.lo_ms + (self.hi_ms - self.lo_ms) * u) / self.step_ms
        return torch.round(x) if self.quantize else x


# ---- real2sim: robot-like qdot ---------------------------------------------------------------------------------------
QD_TICK = 2 * math.pi / 4096          # STS3215 position step (rad): the robot's q is quantised to it


def qd_weights(window, dt=JE.CONTROL_DT):
    """[window] float64 least-squares slope weights over states i = 0 (now) .. window - 1 steps back (``dt`` apart):
    slope = sum_i w_i (x_i - x_0) (sum w_i = 0, so the reference x_0 changes nothing but keeps a still arm at exactly
    0)."""
    if int(window) < 2:
        raise ValueError(f"qd window {window}: >= 2 states")
    t = -torch.arange(int(window), dtype=torch.float64) * dt
    tc = t - t.mean()
    return tc / (tc ** 2).sum()


def qd_ticks(q):
    """q (rad) -> the robot's position in encoder steps (float64 integers; non-finite -> 0)."""
    return torch.round(torch.nan_to_num(q.to(torch.float64), 0.0, 0.0, 0.0) / QD_TICK)


def robot_qd(ticks, w):
    """The robot's qdot (``hw_lib.QdotEstimator``): ticks [B, W, 6] (index 0 = now, ``qd_ticks``), weights [W]
    (``qd_weights``) -> [B, 6] float64 rad/s, steps of QD_TICK x the weights, exactly 0 when the ticks do not change."""
    return ((ticks - ticks[:, :1]) * w.to(ticks)[None, :, None]).sum(1) * QD_TICK


class StudentSet:
    """``joint_teacher.run_tasks`` record files (``row_*``, ``ep_row_start``, ``length``, goals) on one device. A sample
    = one decision boundary: render inputs, robot [q, qd], chunk target [K, 6] = (lab[r] ++ lab[r + 1] ...) - q[r] with
    validity (inside the episode), aux targets. Validation: ``val_frac`` of the FIRST file's episodes (seeded)."""

    def __init__(self, paths, device="cuda", chunk=10, val_frac=0.03, seed=0, max_files=None, lag=0):
        """``lag`` > 0: also the states 1..lag control steps before each row (image lag): the file's ``row_lag``
        (recorded, ``lag_rows``), else linear interpolation between the row and the previous row of its episode
        (q, cube position; quaternion nlerp; the first row of an episode: the row itself) (``lag_exact`` per file)."""
        if isinstance(paths, (str, Path)):
            paths = [paths]
        paths = list(paths)[-max_files:] if max_files else list(paths)
        keys = ("row_q", "row_qd", "row_cube_pos", "row_cube_quat", "row_lab", "row_ep", "length", "ep_row_start",
                "goal_xy", "goal_yaw", "src", "lenient", "success")
        parts, e0, r0 = {k: [] for k in keys}, 0, 0
        is_val = []
        lag = int(lag)
        lag_parts, self.lag_exact = [], []
        for fi, p in enumerate(paths):
            z = np.load(Path(p).expanduser())
            if "goal_jumps" in z.files and np.asarray(z["goal_jumps"]).any():
                raise ValueError(f"{p}: episodes with goal jumps (StudentSet reads one goal per episode)")
            E = len(z["length"])
            if lag > 0:
                if "row_lag" in z.files and z["row_lag"].shape[1] >= lag:
                    lag_parts.append(z["row_lag"][:, :lag].astype(np.float32))
                    self.lag_exact.append(True)
                else:
                    lag_parts.append(lag_interp(z, lag))
                    self.lag_exact.append(False)
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
        self.lag_x = t(np.concatenate(lag_parts)) if lag > 0 else None      # [N, lag, 13] q, cube pos, quat

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

    def batch(self, rows, lag=None):
        """``lag`` [n] long (0 = now): the render inputs ``img_q``, ``img_cube_pos``, ``img_cube_quat`` and the aux
        targets come from the state that many control steps before the row; q / qd / targets stay current."""
        tgt, valid = self.targets(rows)
        gp = self.goal_pose[self.ep[rows]]
        q, cp, cq = self.q[rows], self.cube_pos[rows], self.cube_quat[rows]
        if lag is not None:
            if self.lag_x is None or int(lag.max()) > self.lag_x.shape[1]:
                raise ValueError("StudentSet built without (enough) lag states")
            k = lag.to(self.dev).long()
            x = self.lag_x[rows, (k - 1).clamp_min(0)]
            has = (k > 0)[:, None]
            q, cp, cq = torch.where(has, x[:, :6], q), torch.where(has, x[:, 6:9], cp), torch.where(has, x[:, 9:13], cq)
        return dict(q=self.q[rows], cube_pos=self.cube_pos[rows], cube_quat=self.cube_quat[rows], goal_pose=gp,
                    robot=self.robot[rows], target=tgt, valid=valid, aux=aux_targets(gp, cp), img_q=q,
                    img_cube_pos=cp, img_cube_quat=cq)

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
    lag = int(st.cfg.lag) if data.lag_x is not None else 0
    for s in range(0, len(rows), bs):
        r_ = rows[s:s + bs]
        b = data.batch(r_, lag=torch.full((len(r_),), lag, device=data.dev) if lag > 0 else None)
        img = to_input(views.render(b["img_q"], b["img_cube_pos"], b["img_cube_quat"], b["goal_pose"]), fx=views.fx)
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
    if getattr(cfg, "lastcmd", False):
        raise NotImplementedError("ViralConfig.lastcmd: the online loop only (round files have no executed targets)")
    if cfg.cmd_in or cfg.img_hist:
        raise ValueError("cmd_in / img_hist: the online loop only (round files have no executed targets or image links)")
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
    meta = dict(cfg=cfg_dict(cfg), files=data.paths, n_eps=data.n_eps, rows=data.n_rows, train_rows=len(data.train_rows),
                init_from=str(init_from or ""), **(extra_meta or {}))
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    f = open(out / "learning_curve.csv", "w", newline="")
    wr = None
    t0 = time.time()
    t_render = t_train = 0.0
    buf, pos = None, 0
    agg = {}
    per_buf = max(1, int(cfg.reuse) * cfg.render_batch // cfg.batch)
    lo, hi = (int(v) for v in cfg.lag_range)
    if hi > 0 and data.lag_x is None:
        raise ValueError("cfg.lag_range needs a StudentSet with lag states (StudentSet(..., lag=cfg.lag_max))")
    for step in range(1, max_steps + 1):
        if buf is None or pos >= per_buf:
            tr = time.time()
            rows_ = data.sample(cfg.render_batch, gen)
            lag = torch.randint(lo, hi + 1, (len(rows_),), device=data.dev, generator=gen) if hi > 0 else None
            b = data.batch(rows_, lag=lag)
            n = len(b["q"])
            ndr = int(round(cfg.dr_frac * n)) if drc.on else 0
            iq, icp, icq = b["img_q"], b["img_cube_pos"], b["img_cube_quat"]
            if ndr >= n:
                b["rgb"] = views.render(iq, icp, icq, b["goal_pose"], dr=views.draw(n))
            elif ndr == 0:
                b["rgb"] = views.render(iq, icp, icq, b["goal_pose"])
            else:
                r1 = views.render(iq[:ndr], icp[:ndr], icq[:ndr], b["goal_pose"][:ndr], dr=views.draw(ndr))
                r2 = views.render(iq[ndr:], icp[ndr:], icq[ndr:], b["goal_pose"][ndr:])
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
        img = to_input(rgb, drc if bool(aug.any()) else None, gen, fx=views.fx)
        if bool(aug.any()) and not bool(aug.all()):
            img = torch.where(aug[:, None, None, None, None], img, to_input(rgb, fx=views.fx, fx_random=False))
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
# evaluation only (real2sim debugging, 2026-10-22): the evaluation Actor's qdot input times this gain (env
# NEDM_SO101_EVAL_QD_GAIN; 1 = off): tests whether a slow arm's small measured qdot makes the student command smaller
# steps (speed follows the measured speed). Not used in training or on the robot.
EVAL_QD_GAIN = float(os.environ.get("NEDM_SO101_EVAL_QD_GAIN", "1"))

class Actor:
    """``run_tasks`` student callback: renders the worlds (``dr``: per-world draws fixed for the episode, or None =
    nominal visuals), feeds [q, qdot] -> absolute targets [B, 5, 6] of the executed part of the chunk (jaw target
    ``jaw_lead`` steps ahead).

    Image lag (real2sim): the images show the state ``lag`` control steps before the decision (default the student's
    ``cfg.lag``; ``lag_range`` (lo, hi): drawn per world per episode instead); q and qdot stay current. Lag > 0 needs
    the env's 20 ms states: run the episodes inside ``with step_tap(env, actor.push)`` (``needs_tap``).

    ``qd_window`` > 0 (evaluation of the robot-like qdot, ``OnlineConfig.qd_robot_p``): the qdot input is the robot's
    slope over the last ``qd_window`` 20 ms states of the quantised q (``robot_qd``; needs the tap as well, else the
    second decision raises); 0 = off. ``qd_log``: that input per decision ([B, 6] float32 on the CPU; evaluation rows
    ``row_qd_input``).

    Last-command input (``student.cfg.lastcmd``): set ``env`` (``bind(env)``: the ``JointVecEnv`` that ``run_tasks``
    steps) before the episodes; the first decision after ``reset`` gets 0, every later one ``lastcmd_raw`` of
    ``env.u_exe`` (the targets of the decision before, after the action layer); a decision without exactly one
    ``env.step`` since the last one raises.

    Studentlab inputs: ``cfg.cmd_in`` needs the env (``bind(env)``: its ``u_exe``, the targets the action layer
    executed in the last decision); ``cfg.img_hist`` keeps the last decision's rendered images. Both start from zero /
    the current images at the first call after ``reset``. ``light_redraw_p`` < 1 (randomised visuals): per decision
    the probability of a new per-call render state (random light and intrinsics, ``DRViews.render(keep=)``), else the
    last one stays (the robot's frames share light and intrinsics); an image-history student gets the current images
    as its previous ones when the state changed (as ``OnlineDAgger``). 1 = a new draw every decision (the old
    behaviour)."""

    def __init__(self, student: ViralNet, views: DRViews, randomise=False, jaw_lead=None, device="cuda", lag=None,
                 lag_range=None, seed=0, qd_window=0, light_redraw_p=1.0):
        self.st, self.views, self.dev = student, views, torch.device(device)
        self.randomise = bool(randomise)
        self.jaw_lead = student.cfg.jaw_lead if jaw_lead is None else int(jaw_lead)
        self.dr = None
        self.t_render = self.t_model = 0.0
        self.calls = 0
        self.lag = int(student.cfg.lag if lag is None else lag)
        self.lag_range = tuple(int(v) for v in lag_range) if lag_range is not None and lag_range[1] > 0 else None
        self.kmax = max(self.lag, self.lag_range[1] if self.lag_range else 0)
        self.qd_window = int(qd_window)
        self.qd_w = qd_weights(self.qd_window).to(self.dev) if self.qd_window > 0 else None
        self.qd_hist = None                 # [B, W, 6] q in encoder ticks (index 0 = now)
        self.qd_log = []                    # per decision the robot-like qdot input [B, 6] (qd_window > 0)
        self.needs_tap = self.kmax > 0 or self.qd_window > 0
        self.hist = None
        self.lag_gen = torch.Generator(device=self.dev).manual_seed(int(seed) + 4243)
        self.q_offset = None    # [6] rad: robot input q + q_offset and targets in that frame (JointVecEnv.set_table)
        self.env = None         # cmd_in / lastcmd: the env whose u_exe holds the last decision's executed targets
                                # (``bind``)
        self.lastcmd = bool(getattr(student.cfg, "lastcmd", False))
        self.lc_dec = None      # lastcmd: env.decisions at the last decision (None: an episode start)
        self.n_dec = 0          # decisions since ``reset``
        self.prev_rgb = None    # img_hist: the last decision's rendered images
        self.light_redraw_p = float(light_redraw_p)
        if not 0.0 <= self.light_redraw_p <= 1.0:
            raise ValueError(f"light_redraw_p {light_redraw_p}: a probability in [0, 1]")
        self.redraw_gen = torch.Generator().manual_seed(int(seed) + 4249) if self.light_redraw_p < 1.0 else None
        self.call_id = None     # the views' per-call state id at the last decision
        self.state_changes = 0  # decisions (after the first) whose per-call render state differs from the last one

    def bind(self, env):
        self.env = env
        return self

    def reset(self, B):
        self.dr = self.views.draw(B) if self.randomise else None
        self.n_dec = 0
        self.prev_rgb = None
        self.call_id = None
        self.rhist = None
        self.qd_hist = None
        self.qd_log = []
        self.lc_dec = None
        if self.kmax > 0:
            self.hist = LagHistory(B, self.kmax, self.dev)
            if self.lag_range:
                self.k = torch.randint(self.lag_range[0], self.lag_range[1] + 1, (B,), device=self.dev,
                                       generator=self.lag_gen)
            else:
                self.k = torch.full((B,), self.lag, dtype=torch.long, device=self.dev)

    def push(self, ws):
        """After every 20 ms world step (``step_tap``)."""
        if self.hist is not None and self.hist.ready:
            self.hist.push(*_sane(ws.q, ws.cube_pos, ws.cube_quat, self.dev))
        if self.qd_hist is not None:
            self.qd_hist = torch.cat((qd_ticks(ws.q.to(self.dev))[:, None], self.qd_hist[:, :-1]), 1)

    @torch.no_grad()
    def __call__(self, ws, goal_pose, k=None):
        t0 = time.time()
        q = ws.q.to(self.dev, torch.float32)
        cp, cq = ws.cube_pos.to(self.dev, torch.float32), ws.cube_quat.to(self.dev, torch.float32)
        ok = torch.isfinite(q).all(-1) & torch.isfinite(cp).all(-1) & torch.isfinite(cq).all(-1)
        q = torch.where(ok[:, None], q, torch.zeros_like(q))
        cp = torch.where(ok[:, None], cp, torch.tensor([0.2, 0.0, 0.015], device=self.dev))
        cq = torch.where(ok[:, None], cq, torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.dev))
        iq, icp, icq = q, cp, cq
        if self.hist is not None:
            if not self.hist.ready:                       # the first decision: the arm at rest since the start
                self.hist.fill(torch.arange(len(q), device=self.dev), q, cp, cq)
            elif float((self.hist.newest() - _state13(q, cp, cq)).abs().max()) > 1e-4:
                raise RuntimeError("image lag: the 20 ms states are missing (run the episodes inside step_tap(env, "
                                   "actor.push))")
            iq, icp, icq = self.hist.get(self.k)
        kw = {}
        if self.redraw_gen is not None and self.dr is not None and self.n_dec > 0 and \
                bool(torch.rand(1, generator=self.redraw_gen) >= self.light_redraw_p):
            kw["keep"] = True                             # the last decision's random light / intrinsics stay
        rgb = self.views.render(iq, icp, icq, goal_pose.to(self.dev, torch.float32), dr=self.dr, **kw)
        cid = getattr(self.views, "call_id", None)
        changed = cid is not None and self.call_id is not None and cid != self.call_id
        self.call_id = cid
        self.state_changes += int(changed)
        img = to_input(rgb, fx=self.views.fx)
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        qd = ws.qd.to(self.dev, torch.float32).nan_to_num()
        if self.qd_w is not None:       # the robot's qdot estimate (the first decision: the arm at rest, 0)
            tk = qd_ticks(ws.q.to(self.dev))
            if self.qd_hist is None:
                self.qd_hist = tk[:, None].expand(-1, self.qd_window, -1).clone()
            elif not torch.equal(self.qd_hist[:, 0], tk):
                raise RuntimeError("robot qdot: the 20 ms states are missing (run the episodes inside step_tap(env, "
                                   "actor.push))")
            qd = robot_qd(self.qd_hist, self.qd_w).float()
            self.qd_log.append(qd.cpu())
        if self.q_offset is not None:   # the policy's joint frame (images stay the true state)
            q = q + torch.as_tensor(self.q_offset, dtype=q.dtype, device=self.dev)
        if EVAL_QD_GAIN != 1.0:         # evaluation only (real2sim debugging): the student's qdot input x gain
            qd = qd * EVAL_QD_GAIN
        robot = torch.cat((q, qd), -1)
        H = int(self.st.cfg.hist)
        if H:
            if self.rhist is None:
                self.rhist = robot[:, None, :].expand(-1, H, -1).clone()
            now = robot
            robot = torch.cat((now, self.rhist.reshape(len(now), -1)), -1)
            self.rhist = torch.cat((now[:, None, :], self.rhist[:, :-1]), 1)
        if getattr(self.st.cfg, "cmd_in", False):      # the last decision's executed targets - the measured q now
            if self.env is None:
                raise RuntimeError("cmd_in: bind the env (Actor.bind(env)) for its executed targets")
            u = getattr(self.env, "u_exe", None)
            qm = ws.q.to(self.dev, torch.float32).nan_to_num()
            if self.n_dec == 0 or u is None:
                cmd = torch.zeros(len(q), int(self.st.cfg.n_exec) * 6, device=self.dev)
            else:
                cmd = (u.to(self.dev, torch.float32) - qm[:, None, :]).nan_to_num().reshape(len(q), -1)
            robot = torch.cat((robot, cmd), -1)
        if self.lastcmd:                # the student's last executed command (0 at the episode start; after cmd)
            robot = torch.cat((robot, self.lastcmd_input(ws).to(robot)), -1)
        prev = None
        if getattr(self.st.cfg, "img_hist", False):    # the last decision's images (the current ones at the first
            own = self.prev_rgb is None or changed      # call and after a change of the per-call render state)
            prev = img if own else to_input(self.prev_rgb, fx=self.views.fx)
            self.prev_rgb = rgb
        out = []
        for s in range(0, img.shape[0], 256):
            with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.st.cfg.amp):
                p_ = self.st.predict(img[s:s + 256], robot[s:s + 256]) if prev is None else \
                    self.st.predict(img[s:s + 256], robot[s:s + 256], prev[s:s + 256])
                out.append(denormalise(self.st, p_))
        self.n_dec += 1
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

    def lastcmd_input(self, ws):
        """[B, 12] float64 raw last-command input of this decision (``lastcmd_raw``; class docstring)."""
        e = self.env
        if e is None:
            raise RuntimeError("lastcmd student: set actor.env (Actor.bind(env)) = the JointVecEnv that runs the "
                               "episodes")
        B = ws.q.shape[0]
        if self.lc_dec is None:
            out = lastcmd_zeros(B, self.dev)
        else:
            if e.u_exe is None or int(e.decisions) != self.lc_dec + int(e.B) or e.u_exe.shape[0] != B:
                raise RuntimeError("lastcmd: the env did not step exactly once since the last decision (actor.env is "
                                   "not the env that runs the episodes?)")
            out = lastcmd_raw(e.u_exe[:, -1], e.u_exe[:, -2], ws.q.to(e.u_exe.device)).to(self.dev)
        self.lc_dec = int(e.decisions)
        return out


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
    # joint-reading offset (real2sim: a joint-map / table-height error of the real robot): with probability q_bias_p
    # per world per episode, the student's q INPUT is the true q + U(-h, h) per joint (q_bias_deg = 6 half-widths,
    # deg; jaw normally 0). The labels, the buffer's q (relative targets) and the student's executed targets stay in
    # the true frame (target = true q + predicted delta = what a robot whose joint map is off by the same offset does).
    q_bias_deg: tuple = ()
    q_bias_p: float = 0.0
    # robot timing and servo (real2sim; all off by default): the online env's robot dynamics stage
    # (``JE.RobotDynConfig`` dict: delay, gain, lag, sag, noise, the guard's integral correction; {} = off), drawn per
    # world per episode; the JT labels the reached states
    dyn: dict = field(default_factory=dict)
    dyn_student_only: bool = False      # the stage only in the student-driven slots (the JT-driven slots stay nominal)
    pause: dict = field(default_factory=dict)   # the robot's hold state (``JE.PauseConfig``; {} = off)
    # robot-like qdot of the student's INPUT (acting and the buffer's robot state) with probability qd_robot_p per world
    # per episode: the least-squares slope over the last qd_window 20 ms states of q quantised to the STS3215 step
    # 2 pi / 4096 (``robot_qd``; the robot's ``QdotEstimator``): exactly 0 when the arm is still
    qd_robot_p: float = 0.0
    qd_window: int = 3
    # training episodes: horizon (20 ms steps) and the hold_cap rule (s held in the air; ``JE.hold_cap_kw``), longer for
    # slow worlds; hold_obs_s: the held-time scale of the JT's observation (0 with a new cap = the old envs' 3.5 s, the
    # scale of every run so far; jt_h110 was trained with 2.5 s; a 20 s scale breaks it)
    horizon: int = 400
    hold_cap_s: float = 3.5
    hold_obs_s: float = 0.0
    # the student's image lag range for this run (``ViralConfig.lag_range``, saved with the checkpoints; () = keep the
    # student's own, e.g. from --config / --set)
    lag_range: tuple = ()
    # the loss-side qdot noise (``ViralConfig.qd_noise``): None = as before; a share p -> per sample the noise times
    # U(0, 1), exactly 0 for the share p of the samples (the robot's exact-zero qdot)
    qd_noise_zero_p: float | None = None
    # real2sim (speed follows the measured speed): training-only per-sample factor on the student's qdot INPUT,
    # U(qd_in_lo, qd_in_hi), exactly 0 for the share qd_in_zero_p (1, 1, 0 = off); acting keeps the measured qdot
    qd_in_lo: float = 1.0
    qd_in_hi: float = 1.0
    qd_in_zero_p: float = 0.0
    # real2sim: in robot-dynamics worlds (``dyn``), the JT labels with the commanded speed (the last decision's
    # action-layer target change / 20 ms) in its qdot observation, so its labels keep the nominal pace on a slow arm
    relabel_cmd_speed: bool = False
    # ... only in the place phase (> 0): the cube held, lifted and within relabel_near_m (xy) of the goal; the grasp and
    # the carry keep the measured-speed labels (robot 10-22: relabel everywhere carried the cube ~4 cm lower and grasped
    # worse); 0 = everywhere
    relabel_near_m: float = 0.0
    # ... not in the grasp zone (> 0): no relabel while the cube is not held and the grasp point is within
    # relabel_grasp_zone_m (xy) of it (grasp diagnosis 2026-10-23: relabel at the cube -> the student hovers above an
    # off-centre cube with the jaw open on the robot); the approach, lift, carry and place keep the relabel; 0 = off
    relabel_grasp_zone_m: float = 0.0
    # the student must have the last-command input (``ViralConfig.lastcmd``, e.g. --config viral_r2s_v1_lastcmd.json):
    # a guard against a launch with the wrong student config (the input itself follows the student's config)
    require_lastcmd: bool = False
    # studentlab (2026-10-22): goal jumps in the online env (``JE.GoalJumpConfig`` dict; {} = off): the goal moves
    # while the cube is held near it, so the cube is past (or before) the new goal; the rendered goal, the JT
    # observation and the aux targets read the moved goal from the next decision on; the sample of the decision
    # before a jump gets no second-half label (``valid2`` False: its images showed the old goal)
    goal_jump: dict = field(default_factory=dict)
    # studentlab review (2026-10-23): per decision the probability of a NEW per-call render state (the light choice
    # ``default_light_p``, the random lights and the random intrinsics; one state for all worlds of a call); else the
    # last decision's state stays (``DRViews.render(keep=)``). 1 = a new draw every decision (the old behaviour). For
    # the image history (``ViralConfig.img_hist``): the robot's consecutive frames share light and intrinsics, so a
    # pair of images should as well; when the state changes, the decision's previous images are its current ones
    # (acting) and its buffer link is cut (training), as at an episode start. Also used by the in-run evaluation.
    light_redraw_p: float = 1.0
    # real2sim (2026-10-23): the wrist-roll target cap of the online env (sim deg; ``JE.JointEnvConfig.roll_hi_deg``,
    # the real roll ends at +101.2 sim deg); the JT's labels get the same cap, task rows whose start roll is above it
    # are dropped from the pool, and the in-run evaluations use it too. None = off
    roll_hi_deg: float | None = None
    # real2sim v8 (2026-10-23; off by default): table height / tilt randomisation of the student's JOINT INPUT
    # (``nedm.so101_policy.table_bias``, ``TableBiasConfig`` dict; {} = off): per world per episode with probability p
    # a table plane (drop at x_ref, slopes in x / y); per decision the student reads q_obs = Phi(q), the joints whose
    # gripper is at the same place relative to a world lying on that plane (under a lower table: FK lower by the drop
    # under the gripper, the same gripper rotation; exact on the 3 pitch joints). Every joint quantity of the student
    # is in that observed frame (``table_frame``): q input (+ the q_bias draw), qdot input J_Phi qdot, history, cmd /
    # last-command inputs, the buffer's q and labels (Phi(q), Phi(teacher label)); its applied targets are
    # Phi^-1(Phi(q) + predicted delta). Physics, the action layer and the teacher stay true; images and the aux
    # targets too unless ``table_render``. Training samples are exact (a world whose map rows are not all exact adds
    # no sample that decision; ``table_frame``). A dict with only a "note" is off (``table_bias.is_on``).
    table_bias: dict = field(default_factory=dict)
    # ... and the goal plate ("mat") thickness of the TRAINING renders (real scene only; () = off): per world per
    # episode U(lo, hi) mm in the non-nominal slots (the real plate 3.9 mm, the scenes 1 mm), visual only
    # (``RealDR.plate_thickness``: a cube over the plate that the jaws do not hold is drawn on its top; a held cube at
    # its physics pose, sinking into a thicker plate); the training views get it from ``online_views_cfg``, the
    # student's saved config and the evaluation views do not
    plate_mm: tuple = ()
    # ... and consistent images for the table planes (needs table_bias and a real scene without its own table_plane;
    # False = off): in the worlds with a plane the training renders draw the arm at the student's reading (Phi of the
    # image state, also with image lag) and the world on the plane (the cube sheared by the plane height under it and
    # tilted with the plane; plate, floor, mat, walls, strip and ceiling on the plane; ``SO101Scene`` world_plane), so
    # images and joint reading agree as on the robot (the arm in the image IS the reading); the aux cube target is the
    # drawn cube (z + plane height). Off: the true arm and the sim-height world (the image-vs-reading mismatch tells
    # the drop, a cue that exists only in sim)
    table_render: bool = False
    # real2sim v12 (2026-10-26): the action-layer floor of the online env with ``table_bias``: "true" (default, the v8
    # coupling rule) = FLOOR above the TRUE (sim) table in every world, i.e. above the plane in the student's observed
    # frame (robot: floor mode plane); "observed" = flat FLOOR above z = 0 of the OBSERVED frame, the robot's flat
    # arm-frame floor: the online env's floor raised by -plane at the grasp point (``JE.JointEnvConfig.floor_raise`` =
    # ``JE.raise_of_table``; every slot, student and JT), needs ``table_bias`` with a fixed ``plane``; the labelling JT
    # must have been trained with the same floor raise (vision_viral.py online checks its env record); the checkpoint
    # records it (``ckpt_record``: v8_online.student_floor) and the robot deploys it with the flat floor
    # (``hw_lib.student_floor_mode``)
    student_floor: str = "true"
    # real2sim v8 (2026-10-23): the robot's image latency (``ImgLagConfig`` dict: lo_ms, hi_ms, redraw_p, per_camera,
    # quantize; {} = off, the student's integer lag as before): a lag in ms per world (and camera), drawn again per
    # decision, the images rendered from the interpolated past state and goal; replaces lag_range in this run
    img_lag: dict = field(default_factory=dict)
    # real2sim v12 visual fix (2026-10-26; {} = off): the gripper-look randomisation of the TRAINING renders
    # (``nedm.render.gripper_look.GripperLookDR`` dict, ``RealDR.gripper_look``; per world per episode: albedo of the
    # camera bracket plate and the servo bracket, the scene's fitted screws, extra marks, a bright plate top edge);
    # the training views get it from ``online_views_cfg`` (as plate_mm), the student's saved config, the evaluation
    # views and the robot do not; the nominal slots (nominal_share) keep the scene's look
    gripper_look: dict = field(default_factory=dict)
    # ... and a guard ("" = off): the run refuses a student whose scene file (sha256, after pin_real) is not this one
    # (a held-out variant must not train on the fitted real look; a real-look variant must have it)
    require_scene_sha256: str = ""

    @staticmethod
    def from_dict(d):
        d = dict(d or {})
        known = set(OnlineConfig.__dataclass_fields__)
        bad = sorted(set(d) - known - {"note"})
        if bad:
            raise ValueError(f"unknown online config keys {bad}")
        if d.get("table_bias"):
            from nedm.so101_policy.table_bias import TableBiasConfig
            TableBiasConfig.from_dict(d["table_bias"])                         # validate now
        if d.get("table_render"):
            from nedm.so101_policy.table_bias import is_on
            if not is_on(d.get("table_bias")):
                raise ValueError("table_render: needs table_bias (the planes to draw)")
            d["table_render"] = bool(d["table_render"])
        if d.get("student_floor", "true") != "true":
            if d["student_floor"] != "observed":
                raise ValueError(f"student_floor {d['student_floor']!r}: 'true' (default) or 'observed'")
            from nedm.so101_policy.table_bias import TableBiasConfig, is_on
            if not is_on(d.get("table_bias")) or not TableBiasConfig.from_dict(d["table_bias"]).plane:
                raise ValueError("student_floor observed: needs table_bias with a fixed plane (table_bias.plane; one "
                                 "floor raise for all worlds)")
            # the floor raise -plane within the env's limits (JE.FLOOR_RAISE_MAX, tighter than the plane's own): fail
            # here, not when the online env is built (v12 review finding 6)
            JE.raise_of_table(TableBiasConfig.from_dict(d["table_bias"]).plane)
        if d.get("plate_mm"):
            d["plate_mm"] = tuple(float(v) for v in d["plate_mm"])
            if len(d["plate_mm"]) != 2 or not 0.0 < d["plate_mm"][0] <= d["plate_mm"][1] <= 20.0:
                raise ValueError(f"plate_mm {list(d['plate_mm'])}: (lo, hi) mm with 0 < lo <= hi <= 20")
        if ImgLagConfig.is_on(d.get("img_lag")):
            ImgLagConfig.from_dict(d["img_lag"])                               # validate now
            if d.get("lag_range"):
                raise ValueError("online config: img_lag replaces lag_range in the online loop; set only one")
        if d.get("lag_range"):
            d["lag_range"] = tuple(int(v) for v in d["lag_range"])
            if len(d["lag_range"]) != 2 or not 0 <= d["lag_range"][0] <= d["lag_range"][1]:
                raise ValueError(f"online lag_range {d['lag_range']}: (lo, hi) control steps, 0 <= lo <= hi")
        if d.get("goal_jump"):
            JE.goal_jump_config(d["goal_jump"])                                # validate now
        if "light_redraw_p" in d and not 0.0 <= float(d["light_redraw_p"]) <= 1.0:
            raise ValueError(f"light_redraw_p {d['light_redraw_p']}: a probability in [0, 1]")
        if d.get("roll_hi_deg") is not None:
            d["roll_hi_deg"] = JE.roll_hi_config(d["roll_hi_deg"])          # validate now
        if d.get("gripper_look"):
            from nedm.render.gripper_look import GripperLookDR
            GripperLookDR.from_dict(d["gripper_look"])                         # validate now
        if d.get("require_scene_sha256") and not (isinstance(d["require_scene_sha256"], str)
                                                  and len(d["require_scene_sha256"]) == 64):
            raise ValueError(f"require_scene_sha256 {d['require_scene_sha256']!r}: a sha256 hex string or ''")
        return OnlineConfig(**{k: v for k, v in d.items() if k in known})


def check_scene_requirement(cfg: ViralConfig, ocfg: OnlineConfig) -> None:
    """``OnlineConfig.require_scene_sha256`` (v12 visual fix; '' = no check): the student's scene file (its pinned
    sha256, after ``pin_real``) must be the named one, else ValueError (e.g. a held-out run on the real-look scene, or a
    real-look run with the old scene because of a wrong --config / R2S_SCENE)."""
    want = getattr(ocfg, "require_scene_sha256", "") or ""
    if want and (cfg.scene_sha256 or "") != want:
        raise ValueError(f"online config require_scene_sha256 {want[:12]}: the student's scene {cfg.scene!r} has "
                         f"sha256 {(cfg.scene_sha256 or '-')[:12]} (wrong --config / R2S_SCENE?)")


def online_floor_raise(ocfg: OnlineConfig):
    """The online env's floor raise (``JE.JointEnvConfig.floor_raise``; None = off): -plane of the fixed table plane
    (``JE.raise_of_table``) with ``student_floor`` "observed", else None (a zero plane: None)."""
    if getattr(ocfg, "student_floor", "true") != "observed":
        return None
    from nedm.so101_policy.table_bias import TableBiasConfig
    return JE.raise_of_table(TableBiasConfig.from_dict(ocfg.table_bias).plane)


def online_views_cfg(cfg: ViralConfig, ocfg: OnlineConfig) -> ViralConfig:
    """The student config of the online loop's TRAINING views (``make_views``): ``cfg`` itself, or a copy whose
    ``real_dr`` adds the plate thickness randomisation (``ocfg.plate_mm`` -> ``RealDR.plate_thickness``, m) and / or
    the gripper look (v12, ``ocfg.gripper_look`` -> ``RealDR.gripper_look``, replacing a student's own). The student
    keeps ``cfg`` (its checkpoints and the evaluation views never see the options)."""
    look = dict(getattr(ocfg, "gripper_look", None) or {})
    if not ocfg.plate_mm and not look:
        return cfg
    if not cfg.scene:
        raise ValueError("plate_mm / gripper_look: the real scene only (ViralConfig.scene)")
    from dataclasses import replace as _rep
    over = {}
    if ocfg.plate_mm:
        lo, hi = (float(v) * 1e-3 for v in ocfg.plate_mm)
        over["plate_thickness"] = [lo, hi]
    rdr = _merge(cfg.real_dr or {}, over)
    if look:
        rdr["gripper_look"] = look
    return _rep(cfg, real_dr=rdr)


class ReplayBuffer:
    """Ring buffer of student samples on one device: rgb uint8 [C, cams, H, W, 3], q, qd, labels of this decision
    and of the next one (absolute targets [5, 6] each; ``valid2`` = the next decision is in the same episode), aux
    targets, and whether per-image colour augmentation applies (``aug``: not for nominal-slot samples).

    ``prev_link`` (image history, ``ViralConfig.img_hist``): per sample the slot of the same world's previous-decision
    sample (``prev``; -1 = none: an episode start) and that sample's id (``prev_sid``); every sample gets a running id
    (``sid``). ``prev_valid(i)``: the previous sample exists (same episode) and its slot was not overwritten since."""

    def __init__(self, cap, cams, res, device, robot_dim=12, prev_link=False):
        f = dict(device=device)
        self.cap, self.n, self.ptr = int(cap), 0, 0
        self.sid = self.prev = self.prev_sid = None
        if prev_link:
            self.sid = torch.full((self.cap,), -1, dtype=torch.long, **f)
            self.prev = torch.full((self.cap,), -1, dtype=torch.long, **f)
            self.prev_sid = torch.full((self.cap,), -1, dtype=torch.long, **f)
            self.count = 0
        self.rgb = torch.zeros(self.cap, cams, res, res, 3, dtype=torch.uint8, **f)
        self.q = torch.zeros(self.cap, 6, **f)
        self.robot = torch.zeros(self.cap, robot_dim, **f)
        self.lab1 = torch.zeros(self.cap, JE.STEPS, 6, **f)
        self.lab2 = torch.zeros(self.cap, JE.STEPS, 6, **f)
        self.valid2 = torch.zeros(self.cap, dtype=torch.bool, **f)
        self.aux = torch.zeros(self.cap, AUX_DIM, **f)
        self.aug = torch.zeros(self.cap, dtype=torch.bool, **f)

    def add(self, rgb, q, robot, lab, aux, aug, prev=None):
        """-> the slots [n]. ``prev`` [n] long (``prev_link``): slots of the same worlds' previous samples (-1: none)."""
        n = rgb.shape[0]
        idx = (self.ptr + torch.arange(n, device=rgb.device)) % self.cap
        if self.sid is not None:          # read the previous samples' ids BEFORE this write may overwrite them
            p = torch.full((n,), -1, dtype=torch.long, device=rgb.device) if prev is None else \
                prev.to(rgb.device, torch.long)
            psid = torch.where(p >= 0, self.sid[p.clamp_min(0)], torch.full_like(p, -1))
        self.rgb[idx], self.q[idx], self.robot[idx], self.lab1[idx] = rgb, q.float(), robot.float(), lab.float()
        self.valid2[idx] = False
        self.aux[idx], self.aug[idx] = aux.float(), aug
        if self.sid is not None:
            self.sid[idx] = self.count + torch.arange(n, device=rgb.device)
            self.count += n
            self.prev[idx], self.prev_sid[idx] = p, psid
        self.ptr = int((self.ptr + n) % self.cap)
        self.n = min(self.cap, self.n + n)
        return idx

    def sample(self, b, gen=None):
        return torch.randint(self.n, (int(b),), device=self.rgb.device, generator=gen)

    def prev_valid(self, i):
        """[len(i)] bool: sample i has its world's previous-decision sample in the buffer (same episode, not
        overwritten)."""
        p = self.prev[i]
        return (self.prev_sid[i] >= 0) & (self.sid[p.clamp_min(0)] == self.prev_sid[i])

    def prev_rgb(self, i):
        """The previous-decision images of samples i (their own images where ``prev_valid`` is False)."""
        ok = self.prev_valid(i)
        return torch.where(ok[:, None, None, None, None], self.rgb[self.prev[i].clamp_min(0)], self.rgb[i]), ok


def relabel_mask(o, env):
    """[B] bool: the worlds whose JT label gets the commanded speed (``OnlineConfig.relabel_cmd_speed``): robot-dynamics
    worlds, not the first decision of an episode (``env.u_exe`` still holds the previous episode's targets after an
    auto reset), only in the place phase if ``relabel_near_m`` > 0, and not in the grasp zone if
    ``relabel_grasp_zone_m`` > 0."""
    m = env.dyn.on & (env.ep_len > 0)
    F = env.task.F
    if o.relabel_near_m > 0 and F is not None:                                             # place phase only
        m = m & F["held"] & env.task.tr.lifted & ((F["c"][:, :2] - env.task.goal_xy).norm(dim=-1) < o.relabel_near_m)
    if o.relabel_grasp_zone_m > 0 and F is not None:                                       # not at the grasp
        m = m & ~(~F["held"] & (F["dxy"] < o.relabel_grasp_zone_m))
    return m


class OnlineDAgger:
    """The online loop (``OnlineConfig``). Tasks: the start states and goals of a task table (``state_*``, goal_xy,
    goal_yaw), drawn uniformly at every reset. Image lag (``st.cfg.lag_range``, else ``st.cfg.lag``): drawn per world
    per episode; every world's 20 ms states feed a ``LagHistory`` (``JointVecEnv.step`` hook). Robot timing / servo
    (``OnlineConfig.dyn``: the env's ``RobotDyn``) and robot-like qdot inputs (``qd_robot_p``: per-world tick history of
    the 20 ms states, same hook) are drawn per world per episode as well. Last-command input (``st.cfg.lastcmd``): every
    world's (student and teacher slots) ``lastcmd_raw`` of the env's ``u_exe``, 0 at its episode start (``lc_fresh``),
    in the acting input and the buffer's robot vector (``ViralConfig.robot_dim``). Wrist-roll cap
    (``OnlineConfig.roll_hi_deg``): the env caps every executed target, the JT's labels are capped the same way (the
    student never learns a roll above the cap), task rows that start above it leave the pool. Image latency
    (``OnlineConfig.img_lag``, ``ImgLagConfig``; real2sim v8): instead of the student's integer lag, a lag in ms per
    world (and camera), drawn at the episode start and again per decision (``redraw_p``), the images rendered from the
    interpolated past state and goal (``LagHistory.at``; ``lag_x`` [N, 1 or C] control steps)."""

    def __init__(self, st: ViralNet, jt, tasks, nrd_ckpt, views: DRViews, ocfg: OnlineConfig, device="cuda", seed=0,
                 log=log_default, world="nrd"):
        """``world``: the online env's world (``nrd``; tests: ``kin``)."""
        if ocfg.require_lastcmd and not getattr(st.cfg, "lastcmd", False):
            raise ValueError("online config require_lastcmd: the student has no last-command input "
                             "(ViralConfig.lastcmd; --config viral_r2s_v1_lastcmd.json)")
        self.st, self.jt, self.views, self.o, self.dev, self.log = st, jt, views, ocfg, torch.device(device), log
        N = int(ocfg.envs)
        cfg = JE.JointEnvConfig(world=world, num_envs=N, device=device, ckpts=[str(nrd_ckpt)], auto_reset=True,
                                stagger=True, starts="", seed=seed, horizon=int(ocfg.horizon), dyn=ocfg.dyn or None,
                                pause=ocfg.pause or None, goal_jump=ocfg.goal_jump or None,
                                roll_hi_deg=ocfg.roll_hi_deg, floor_raise=online_floor_raise(ocfg),
                                **JE.hold_cap_kw(ocfg.hold_cap_s, ocfg.hold_obs_s))
        self.env = JE.JointVecEnv(cfg, jt.codec)
        f64 = dict(dtype=torch.float64, device=self.dev)
        self.env.bank = {k: torch.as_tensor(tasks[f"state_{k}"], **f64) for k in
                         ("q", "qd", "cube_pos", "cube_quat", "cube_v", "cube_w")}
        nt = len(tasks["seeds"])
        self.env.pool = dict(i=torch.arange(nt, device=self.dev), goal_xy=torch.as_tensor(tasks["goal_xy"], **f64),
                             goal_yaw=torch.as_tensor(tasks["goal_yaw"], **f64))
        self.roll_cap = self.env.roll_cap()       # the wrist-roll cap (rad; None = off): labels, pool, statistics
        if self.roll_cap is not None:             # no start above the cap (no row: the pool and its draws stay)
            keep = ~(self.env.bank["q"][:, JE.ROLL] > self.roll_cap)
            if not bool(keep.all()):
                ki = keep.nonzero().flatten()
                self.env.pool = {k: v[ki] for k, v in self.env.pool.items()}
            self.env.roll_dropped = int((~keep).sum())     # the record (roll_info: bank_dropped = task rows here)
            log(f"roll cap {ocfg.roll_hi_deg:g} deg: {int((~keep).sum())} of {nt} training tasks start above it "
                f"(dropped); labels capped")
        self.gen = torch.Generator(device=self.dev).manual_seed(int(seed) + 5)
        nt_ = int(round(ocfg.teacher_share * N))
        slot = torch.arange(N, device=self.dev)
        self.teacher = slot < nt_
        self.noisy = slot < int(round(ocfg.noise_share * nt_))
        if ocfg.dyn_student_only and self.env.dyn is not None:
            self.env.dyn.set_eligible(~self.teacher)
        self.nominal = torch.rand(N, device=self.dev, generator=self.gen) < ocfg.nominal_share
        self.H = int(st.cfg.hist)
        self.n_cmd = int(st.cfg.n_exec) * 6 if getattr(st.cfg, "cmd_in", False) else 0     # cmd input (studentlab)
        self.img_hist = bool(getattr(st.cfg, "img_hist", False))                       # image history (studentlab)
        self.LC = bool(getattr(st.cfg, "lastcmd", False))                              # last-command input (real2sim)
        self.lc_fresh = torch.ones(N, dtype=torch.bool, device=self.dev)    # lastcmd: no command yet in the episode
        # robot vector [q, qd, history, (cmd), (lastcmd)] = 12 (1 + H) + n_cmd + 12 LC (ViralConfig.robot_dim)
        self.buf = ReplayBuffer(ocfg.buffer, len(st.cfg.cameras), st.cfg.res, self.dev, st.cfg.robot_dim,
                                prev_link=self.img_hist)
        self.rhist = torch.zeros(N, max(self.H, 1), 12, device=self.dev)
        self.fresh = torch.ones(N, dtype=torch.bool, device=self.dev)            # the next decision starts an episode
        self.prev_rgb = None                                                     # img_hist: the last decision's images
        self.last = {}                                                           # the last decision's inputs (tests)
        self.light_prev = None                                                   # light_redraw_p < 1: the last light
        self.redraw_gen = torch.Generator().manual_seed(int(seed) + 919) if float(ocfg.light_redraw_p) < 1.0 else None
        self.call_id = None                                                      # the views' per-call state id
        if self.img_hist and self.redraw_gen is None and getattr(views, "call_id", None) is not None:
            log("NOTE: img_hist with light_redraw_p 1: a new light / intrinsics draw at most decisions, so most image "
                "pairs are cut (previous images = current ones); light_redraw_p 0.1 keeps ~91 % of them")
        self.kmax = int(st.cfg.lag_max)
        self.lag_hist = LagHistory(N, self.kmax, self.dev) if self.kmax > 0 else None
        lo, hi = (int(v) for v in st.cfg.lag_range)
        self.lag_lo, self.lag_hi = (lo, hi) if hi > 0 else (int(st.cfg.lag), int(st.cfg.lag))
        self.lagk = torch.zeros(N, dtype=torch.long, device=self.dev)
        self.lag_gen = torch.Generator(device=self.dev).manual_seed(int(seed) + 77)
        self.il = ImgLagConfig.from_dict(ocfg.img_lag) if ImgLagConfig.is_on(ocfg.img_lag) else None   # v8 img lag
        self.lag_x = None                                  # [N, 1 or C] float control steps (img_lag)
        self.stale = 0                                     # img_lag + goal jumps: decisions left out (old goal shown)
        if self.il is not None:
            self.kmax = self.il.kmax
            self.lag_hist = LagHistory(N, self.kmax, self.dev, goal=True)
            self.lag_x = torch.zeros(N, len(st.cfg.cameras) if self.il.per_camera else 1, device=self.dev)
            self.il_gen = torch.Generator(device=self.dev).manual_seed(int(seed) + 177)
            log(f"image lag: img_lag {asdict(self.il)} (history {self.kmax} steps) replaces the student's lag "
                f"{st.cfg.lag} / lag_range {list(st.cfg.lag_range)} in the online loop")
        self.qb = torch.zeros(N, 6, device=self.dev)       # per-world offset of the student's q input (rad)
        self.qb_h = None
        if ocfg.q_bias_p > 0 and ocfg.q_bias_deg:
            if len(ocfg.q_bias_deg) != 6:
                raise ValueError("q_bias_deg: 6 half-widths (deg)")
            self.qb_h = torch.deg2rad(torch.tensor([float(v) for v in ocfg.q_bias_deg], device=self.dev))
            self.qb_gen = torch.Generator(device=self.dev).manual_seed(int(seed) + 91)
        self.tb = None              # v8: table height / tilt of the student's q input (OnlineConfig.table_bias)
        self.tb_render = False      # ... and the consistent images (OnlineConfig.table_render)
        from nedm.so101_policy import table_bias as TBI
        if TBI.is_on(ocfg.table_bias):
            from nedm.so101_policy.kin import SO101Kin
            self.tb = TBI.TableBiasConfig.from_dict(ocfg.table_bias)
            self.tb_gen = torch.Generator(device=self.dev).manual_seed(int(seed) + 97)
            self.tb_plane = torch.zeros(N, 3, dtype=torch.float64, device=self.dev)    # per world (zeros: off)
            self.tb_on = torch.zeros(N, dtype=torch.bool, device=self.dev)
            self.tb_kin = SO101Kin(self.dev)
            # log statistics since the last log row (``tb_note``): largest residual of the rows below TOL (frame map,
            # applied targets), counts [rows above TOL, rows] of the planed worlds (frame map, student-slot applied
            # targets), [samples skipped, samples] (a world whose frame map was not exact)
            self.tb_res = torch.zeros(2, dtype=torch.float64, device=self.dev)
            self.tb_cnt = torch.zeros(6, dtype=torch.long, device=self.dev)
            self.tb_render = bool(ocfg.table_render)
            if self.tb_render and not getattr(views, "table_render_ok", False):
                raise ValueError("table_render: the training views must draw per-world table planes (DRViews of a real "
                                 "scene without its own table_plane)")
            log(f"table bias of the student's q input: {json.dumps(self.tb.to_dict())}"
                + (" + consistent images (arm at the reading, world on the plane)" if self.tb_render else ""))
            if self.env.cfg.floor_raise is not None or getattr(ocfg, "student_floor", "true") != "true":
                log(f"student floor {ocfg.student_floor}: the online env's floor 1.5 mm above z = "
                    f"{self.env.cfg.floor_raise} (c0 + c1 x + c2 y at the grasp point, true frame; None = z 0) = flat "
                    "in the observed frame (the robot's flat floor)")
        elif ocfg.table_render:
            raise ValueError("table_render: needs table_bias")
        if ocfg.plate_mm and not getattr(views, "plate_on", False):
            raise ValueError("plate_mm: the training views need the plate thickness randomisation "
                             "(make_views(online_views_cfg(st.cfg, ocfg), ...))")
        if getattr(ocfg, "gripper_look", None):
            from nedm.render.gripper_look import GripperLookDR
            if GripperLookDR.from_dict(ocfg.gripper_look).on and not getattr(views, "look_on", False):
                raise ValueError("gripper_look: the training views need the gripper-look randomisation "
                                 "(make_views(online_views_cfg(st.cfg, ocfg), ...))")
        self.qd_rob = torch.zeros(N, dtype=torch.bool, device=self.dev)     # worlds with the robot-like qdot input
        self.qd_hist = None                                                 # [N, W, 6] q in encoder ticks, 0 = now
        if ocfg.qd_robot_p > 0:
            self.qd_w = qd_weights(ocfg.qd_window).to(self.dev)
            self.qd_hist = torch.zeros(N, int(ocfg.qd_window), 6, dtype=torch.float64, device=self.dev)
            self.qd_gen = torch.Generator(device=self.dev).manual_seed(int(seed) + 131)
        if ocfg.qd_noise_zero_p is not None and not 0.0 <= float(ocfg.qd_noise_zero_p) <= 1.0:
            raise ValueError(f"qd_noise_zero_p {ocfg.qd_noise_zero_p}: a share in [0, 1] (None = off)")
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
        if self.env.goal_jump is not None:      # episodes with >= 1 goal jump (and their lenient successes)
            self.stats.update(ep_jump_student=0, len_jump_student=0, ep_jump_teacher=0, len_jump_teacher=0)
        if self.img_hist:                       # in-episode image pairs, and those with the same per-call render state
            self.stats.update(img_pairs=0, img_pairs_same=0)
        if self.roll_cap is not None:           # episodes whose commanded wrist roll went above the cap
            self.stats.update(roll_over_student=0, roll_over_teacher=0)

    def _lag_reset(self, idx):
        """Episode start of worlds ``idx`` (the env already holds their start states): lag draw, history = start."""
        if self.lag_hist is None:
            return
        ws = self.env.ws
        q, cp, cq = _sane(ws.q[idx], ws.cube_pos[idx], ws.cube_quat[idx], self.dev)
        if self.il is not None:            # img_lag: history = start state and goal, a new lag per world (camera)
            self.lag_hist.fill(idx, q, cp, cq, self._goal_pose()[idx])
            self.lag_x[idx] = self.il.draw(len(idx), self.lag_x.shape[1], self.il_gen, self.dev)
            return
        self.lag_hist.fill(idx, q, cp, cq)
        self.lagk[idx] = torch.randint(self.lag_lo, self.lag_hi + 1, (len(idx),), device=self.dev,
                                       generator=self.lag_gen)

    def _goal_pose(self, task=None):
        """[N, 3] float32 goal pose (x, y, yaw) of every world (as rendered)."""
        task = self.env.task if task is None else task
        return torch.cat((task.goal_xy, task.goal_yaw[:, None]), -1).float()

    def _lag_redraw(self):
        """img_lag, decision start: every world (camera) draws a new lag with probability redraw_p (the same random
        numbers are used whatever the share is)."""
        il = self.il
        if il is None or il.redraw_p <= 0.0:
            return
        N, C = self.lag_x.shape
        m = torch.rand(N, device=self.dev, generator=self.il_gen) < il.redraw_p
        self.lag_x = torch.where(m[:, None], il.draw(N, C, self.il_gen, self.dev), self.lag_x)

    def _lag_states(self, gp):
        """img_lag: per lag column (1, or one per camera with ``per_camera``) the image state ``lag_x`` steps back
        (``LagHistory.at``: q, cube position, cube quaternion, goal pose of then) -> list of tuples."""
        return [self.lag_hist.at(self.lag_x[:, c], gp) for c in range(self.lag_x.shape[1])]

    def _render_lagged(self, gp, light, kw, states=None, iqs=None, rkw=None):
        """img_lag: the images of every world from its state ``lag_x`` steps back (``LagHistory.at``; ``states``: the
        ``_lag_states`` already taken) -> (rgb, cube position and goal pose of the (first camera's) image state, [N]
        bool: every camera shows the current goal). ``per_camera``: one render call per camera; the later calls keep
        the first call's light and intrinsics. ``iqs`` (v8 ``table_render``): per lag column the arm state to draw
        instead (Phi of the image state); ``rkw``: the other render inputs (``plane``, ``held``)."""
        states = self._lag_states(gp) if states is None else states
        rkw = rkw or {}
        if len(states) == 1:
            iq, icp, icq, igp = states[0]
            iq = iq if iqs is None else iqs[0]
            rgb = self.views.render(iq, icp, icq, igp, dr=self.dr, light=light, **kw, **rkw)
            return rgb, icp, igp, (igp == gp).all(-1)
        outs, icp, igp, cur = [], None, None, None
        for c, (iq, cp_c, icq, gp_c) in enumerate(states):
            iq = iq if iqs is None else iqs[c]
            r = self.views.render(iq, cp_c, icq, gp_c, dr=self.dr, light=light, **(kw if c == 0 else dict(keep=True)),
                                  **rkw)
            outs.append(r[:, c].clone())
            same = (gp_c == gp).all(-1)
            if c == 0:
                icp, igp, cur = cp_c, gp_c, same
            else:
                cur = cur & same
        return torch.stack(outs, 1), icp, igp, cur

    def _bias_reset(self, idx):
        """Episode start of worlds ``idx``: draw the offset of the student's q input (``OnlineConfig.q_bias_*``) and
        the world's table plane (``OnlineConfig.table_bias``; own generator)."""
        if getattr(self, "tb", None) is not None:
            from nedm.so101_policy import table_bias as TBI
            pl, on = TBI.sample(len(idx), self.tb, self.tb_gen, self.dev)
            self.tb_plane[idx], self.tb_on[idx] = pl, on
        if self.qb_h is None:
            return
        n = len(idx)
        on = torch.rand(n, device=self.dev, generator=self.qb_gen) < self.o.q_bias_p
        u = torch.rand(n, 6, device=self.dev, generator=self.qb_gen) * 2 - 1
        self.qb[idx] = u * self.qb_h[None] * on[:, None]

    def table_frame(self, q, qf, qd, lab, u_exe=None, iq=None, u_used=None):
        """The observed (table) frame of this decision (``OnlineConfig.table_bias``; ``table_bias`` module docstring):
        one batched Phi under each world's plane of the true q [N, 6] (float64, finite; ``qf`` = its float32 input
        copy), of the labels [N, 5, 6], of the last decision's executed targets [N, 5, 6] (the cmd / last-command
        inputs; None: not needed; ``u_used`` [N] bool: the worlds whose inputs use them, not at an episode start) and
        of the lagged image state ``iq`` [G N, 6] (``table_render`` with image lag; G = 1, or one block of N rows per
        lag column of ``OnlineConfig.img_lag`` per_camera; None: not needed); the qdot reading J_Phi qd analytic
        (``table_bias.observe_qd``) -> dict b [N, 6] float32 (reading - q), b64, q [N, 6] float32 (the reading qf + b),
        q64 [N, 6] float64 (Phi(q)), qd [N, 6] float32, lab [N, 5, 6] float64, u_exe [N, 5, 6] float64 or None, iq
        [G N, 6] float32 (Phi(iq); None if not asked), skip [N] bool (a world
        with a plane whose map rows are not all exact, ``table_bias.EXACT``: no training sample this decision),
        lab_bad [N] bool (its label rows are not exact: it does not complete the previous sample). Rows above
        ``table_bias.TOL`` (no image) are acting-only fallbacks: x + b. Worlds without a plane get every value back as
        given."""
        from nedm.so101_policy import table_bias as TBI
        N, f64 = qf.shape[0], torch.float64
        pl, on = self.tb_plane, self.tb_on
        K = lab.shape[1]
        rows, planes = [q, lab.reshape(-1, 6).to(f64)], [pl, pl.repeat_interleave(K, 0)]
        ons = [on, on.repeat_interleave(K)]                         # rows of the worlds with a plane (statistics)
        if u_exe is not None:
            rows.append(u_exe.to(self.dev, f64).reshape(-1, 6))
            planes.append(pl.repeat_interleave(u_exe.shape[1], 0))
            ons.append(on.repeat_interleave(u_exe.shape[1]))
        G = 0 if iq is None else iq.shape[0] // N         # image-state blocks (img_lag per_camera: one per camera)
        if iq is not None:
            if iq.shape[0] != G * N:
                raise ValueError(f"table_frame: iq rows {iq.shape[0]} not a multiple of {N} worlds")
            rows.append(iq.to(self.dev, f64))
            planes.append(pl if G == 1 else pl.repeat(G, 1))
            ons.append(on if G == 1 else on.repeat(G))
        out, res = TBI.observe(self.tb_kin, torch.cat(rows), torch.cat(planes))
        bad, inexact = res > TBI.TOL, res > TBI.EXACT     # no image (acting fallback) / not exact (no training)
        q64 = out[:N]
        b64 = q64 - q
        qd_obs = torch.where(on[:, None], TBI.observe_qd(self.tb_kin, q, qd.to(f64), pl, q64), qd.to(f64)).to(qd.dtype)
        k1 = N + N * K
        k2 = k1 + (0 if u_exe is None else u_exe.shape[0] * u_exe.shape[1])

        def mapped(x, lo, hi):            # rows [lo, hi) of the batch, fallback x + b where they have no image
            m = bad[lo:hi].reshape(x.shape[:-1])[..., None]
            return torch.where(m, x.to(f64) + b64[:, None, :], out[lo:hi].reshape(x.shape))

        per = lambda m, lo, hi: m[lo:hi].reshape(N, -1).any(-1)                         # noqa: E731
        lab_bad = per(inexact, N, k1)
        skip = inexact[:N] | lab_bad
        if u_exe is not None:
            skip = skip | (per(inexact, k1, k2) & (torch.ones_like(on) if u_used is None else u_used))
        iq_obs = None
        if iq is not None:
            skip = skip | inexact[k2:].reshape(G, N).any(0)
            bg = b64 if G == 1 else b64.repeat(G, 1)
            iq_obs = torch.where(bad[k2:, None], iq.to(self.dev, f64) + bg, out[k2:]).float()
        self.tb_note(res, torch.cat(ons), frame=True, skip=(skip & on, on))
        return dict(b=b64.float(), b64=b64, q=qf + b64.float(), q64=q64, qd=qd_obs, lab=mapped(lab, N, k1),
                    u_exe=None if u_exe is None else mapped(u_exe, k1, k2), iq=iq_obs, skip=skip & on,
                    lab_bad=lab_bad & on)

    def table_applied(self, u_obs, si, b64):
        """Applied (true-frame) targets of the student slots ``si``: Phi^-1 of their observed-frame targets u_obs
        [n, n_exec, 6] (``table_bias.to_true``; the jaw unchanged); rows without a preimage (near the reach limit,
        residual above ``table_bias.TOL``): u_obs - b of the decision (``b64`` [n, 6] float64) -> float64, same
        shape."""
        from nedm.so101_policy import table_bias as TBI
        k = u_obs.shape[1]
        u, res = TBI.to_true(self.tb_kin, u_obs.reshape(-1, 6).to(torch.float64),
                             self.tb_plane[si].repeat_interleave(k, 0))
        self.tb_note(res, self.tb_on[si].repeat_interleave(k, 0), frame=False)
        u = u.reshape(u_obs.shape)
        bad = (res > TBI.TOL).reshape(u_obs.shape[:2])[..., None]
        return torch.where(bad, u_obs.to(torch.float64) - b64[:, None, :], u)

    def tb_note(self, res, row_on, frame=True, skip=None):
        """Log statistics of the joint map (``log_extra``, since the last log row; rows of the worlds with a plane
        only): tb_res_um = the largest residual of the rows below ``table_bias.TOL`` (frame map; tb_res_applied_um:
        the student slots' applied targets), tb_bad_frame / tb_bad_applied = the share of rows above TOL (acting
        fallback), tb_skip = the share of samples of planed worlds not added (map not exact)."""
        from nedm.so101_policy import table_bias as TBI
        k = 0 if frame else 1
        m = row_on.to(res.device)
        bad = (res > TBI.TOL) & m
        good = torch.where(m & ~bad, res, torch.zeros_like(res))
        self.tb_res[k] = torch.maximum(self.tb_res[k], good.max() if good.numel() else self.tb_res[k])
        self.tb_cnt[2 * k] += bad.sum()
        self.tb_cnt[2 * k + 1] += m.sum()
        if skip is not None:
            self.tb_cnt[4] += skip[0].sum()
            self.tb_cnt[5] += skip[1].sum()

    def _qd_reset(self, idx):
        """Episode start of worlds ``idx``: robot-like qdot draw (``OnlineConfig.qd_robot_p``), tick history = start."""
        if self.qd_hist is None:
            return
        self.qd_rob[idx] = torch.rand(len(idx), device=self.dev, generator=self.qd_gen) < self.o.qd_robot_p
        self.qd_hist[idx] = qd_ticks(self.env.ws.q[idx]).to(self.qd_hist)[:, None].expand(-1, self.qd_hist.shape[1],
                                                                                          -1)

    def qd_input(self, qd):
        """[N, 6] float32 qdot of the student's input: the robot-like slope in the ``qd_rob`` worlds, else ``qd``."""
        if self.qd_hist is None:
            return qd
        return torch.where(self.qd_rob[:, None], robot_qd(self.qd_hist, self.qd_w).to(qd), qd)

    def lastcmd_input(self, q, u=None):
        """[N, 12] float64 raw last-command input of every world (``lastcmd_raw``): the env's ``u_exe`` (the targets of
        the decision before, after the action layer; ``u``: those in another frame, the table frame) vs q [N, 6]; 0 in
        the worlds at an episode start (``lc_fresh``; ``env.u_exe`` is not cleared at an automatic reset)."""
        u = self.env.u_exe if u is None else u
        if u is None:
            return lastcmd_zeros(q.shape[0], self.dev)
        return lastcmd_raw(u[:, -1], u[:, -2], q, self.lc_fresh)

    def _lag_hook(self, e, j):
        if self.il is not None:            # img_lag: the goal of every 20 ms state as well
            self.lag_hist.push(*_sane(e.ws.q, e.ws.cube_pos, e.ws.cube_quat, self.dev), goal=self._goal_pose(e.task))
            return
        self.lag_hist.push(*_sane(e.ws.q, e.ws.cube_pos, e.ws.cube_quat, self.dev))

    def _step_hook(self, e, j):
        """After every 20 ms step: the image-lag history and / or the tick history of the robot-like qdot."""
        if self.lag_hist is not None:
            self._lag_hook(e, j)
        if self.qd_hist is not None:
            self.qd_hist = torch.cat((qd_ticks(e.ws.q).to(self.qd_hist)[:, None], self.qd_hist[:, :-1]), 1)

    def ckpt_record(self) -> dict:
        """Checkpoint keys of the v8 training options (``save_student(**extra)``): ``v8_online`` = the image lag
        (``img_lag``) and the table randomisation (``table_bias``, ``table_render``, ``plate_mm``) the student was
        trained with; {} while all of them are off (the old checkpoint layout bit for bit). The evaluation does not
        read it (it renders with ``ViralConfig.lag``); the robot reads ``table_bias`` and ``student_floor`` for its
        floor mode (``hw_lib.student_floor_mode``: table_bias on = the floor above the calibrated mat plane, unless
        ``student_floor`` "observed" (v12; only written when not the default): the flat arm-frame floor). v12 visual
        fix: ``v12_vis`` = the gripper look of the training views and the student's scene sha256 (only when on)."""
        rec = {}
        if getattr(self, "il", None) is not None:
            rec["img_lag"] = asdict(self.il)
        if getattr(self, "tb", None) is not None:
            rec.update(table_bias=self.tb.to_dict(), table_render=bool(self.tb_render))
            if getattr(self.o, "student_floor", "true") != "true":      # v12: the flat observed-frame floor
                rec["student_floor"] = self.o.student_floor
        if getattr(self.o, "plate_mm", ()):
            rec["plate_mm"] = [float(v) for v in self.o.plate_mm]
        out = {"v8_online": rec} if rec else {}
        if getattr(self.o, "gripper_look", None):       # v12: the training look (its own key; the robot ignores it)
            out["v12_vis"] = dict(gripper_look=dict(self.o.gripper_look),
                                  scene_sha256=getattr(self.st.cfg, "scene_sha256", ""))
        return out

    def log_extra(self):
        """Columns of the online log for the robot options that are on (none: {} = the old columns)."""
        out = {}
        if self.env.dyn is not None:
            out.update(self.env.dyn.summary())
        if self.env.pause is not None:
            out.update(self.env.pause.summary())
        if self.qd_hist is not None:
            out["qd_robot"] = round(float(self.qd_rob.float().mean()), 3)
        if self.tb is not None:     # the worlds' current table planes: share with one, mean drop at x_ref; the map's
            on = self.tb_on         # statistics since the last row (``tb_note``)
            drop = -(self.tb_plane[:, 0] + self.tb_plane[:, 1] * float(self.tb.x_ref_m))
            out["tb_on"] = round(float(on.float().mean()), 3)
            out["tb_drop_mm"] = round(float(drop[on].mean()) * 1e3, 2) if bool(on.any()) else float("nan")
            r, c = self.tb_res.tolist(), [int(v) for v in self.tb_cnt.tolist()]
            out["tb_res_um"], out["tb_res_applied_um"] = round(r[0] * 1e6, 4), round(r[1] * 1e6, 4)
            out["tb_bad_frame"] = round(c[0] / max(c[1], 1), 6)
            out["tb_bad_applied"] = round(c[2] / max(c[3], 1), 6)
            out["tb_skip"] = round(c[4] / max(c[5], 1), 6)
            self.tb_res, self.tb_cnt = torch.zeros_like(self.tb_res), torch.zeros_like(self.tb_cnt)
        if self.env.goal_jump is not None:
            out.update(self.env.goal_jump.summary())
        if self.img_hist:       # in-episode image pairs since the last log row: the share with the same light etc.
            n = self.stats.get("img_pairs", 0)
            out["img_pair_same"] = round(self.stats.get("img_pairs_same", 0) / n, 4) if n else float("nan")
        if self.il is not None:     # the image lags of this decision (ms) and the samples left out since the last row
            ms = self.lag_x.double() * self.il.step_ms
            out.update(img_lag_ms=round(float(ms.mean()), 2), img_lag_ms_max=round(float(ms.max()), 2),
                       img_lag_stale=int(self.stale))
            self.stale = 0
        return out

    def _draw(self, idx):
        d = self.views.draw(len(idx))
        nom = self.nominal[idx]
        if self.views.real is not None:
            d = self.views.merge_nominal(d, nom)
            if self.dr is None:
                self.dr = d
            else:
                for k in self.dr:
                    self.dr[k][idx] = d[k]
            return
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
            self._lag_reset(torch.arange(env.B, device=self.dev))
            self._bias_reset(torch.arange(env.B, device=self.dev))
            self._qd_reset(torch.arange(env.B, device=self.dev))
        ws, obs = env.ws, env.obs
        q = ws.q.to(torch.float64)
        if o.relabel_cmd_speed and env.dyn is not None and env.u_exe is not None and bool(env.dyn.on.any()):
            v = ((env.u_exe[:, -1] - env.u_exe[:, -2]) / JE.CONTROL_DT).to(obs.dtype)      # commanded speed [B, 6]
            obs = obs.clone()
            m = relabel_mask(o, env)
            obs[:, 6:12] = torch.where(m[:, None], v, obs[:, 6:12])                        # obs layout: q, qd, ...
        a = self.jt.act(obs).to(torch.float64)
        lab = env.codec.targets(a, q)                                              # [N, 5, 6] the label
        if self.roll_cap is not None:                                              # the roll cap: capped labels
            lab = env.cap_roll(lab)
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
        kw = {}
        if self.redraw_gen is not None:   # light_redraw_p < 1: the last decision's per-call state stays unless redrawn
            if self.light_prev is not None and bool(torch.rand(1, generator=self.redraw_gen) >= o.light_redraw_p):
                light, kw = self.light_prev, dict(keep=True)
            self.light_prev = light
        icp, igp, cur_goal = cp, gp, None
        lagged = None
        if self.il is not None:                           # img_lag: per decision lags in ms, interpolated states
            self._lag_redraw()
            lagged = self._lag_states(gp)
        TF = None
        if self.tb is not None:     # the table plane's observed frame: reading, qdot, labels, last executed targets
            ttb = time.time()       # (before the render: table_render draws the arm at the reading; own timing)
            qd = self.qd_input(ws.qd.float().nan_to_num())
            iq_ = None              # the image state(s) to map (table_render with image lag)
            if self.tb_render and lagged is not None:     # img_lag: one state per lag column (camera-major rows)
                iq_ = torch.cat([s_[0] for s_ in lagged])
            elif self.tb_render and self.lag_hist is not None:
                iq_ = self.lag_hist.get(self.lagk)[0]
            TF = self.table_frame(torch.where(ok[:, None], q, torch.zeros_like(q)), qf, qd, lab,
                                  env.u_exe if (self.n_cmd or self.LC) and env.u_exe is not None else None, iq_,
                                  ~self.fresh)       # (an auto-reset world's u_exe is the last episode's: unused)
            torch.cuda.synchronize() if self.dev.type == "cuda" else None
            T["table"] = time.time() - ttb
        rkw = {}                    # v8 render inputs (only when on: other views keep their old call)
        if self.tb_render:          # consistent images: the world on each world's plane, the arm at the reading
            rkw["plane"] = self.tb_plane
        if getattr(self.views, "plate_on", False):      # the plate lift spares a cube the jaws hold
            F = env.task.F
            held = F["held"].to(self.dev) if F is not None else torch.zeros_like(self.fresh)
            rkw["held"] = held & ~self.fresh            # (F of an auto-reset world is the last episode's)
        if lagged is not None:                            # img_lag (+ table_render: the arm at Phi of each state)
            iqs = TF["iq"].split(len(qf)) if self.tb_render else None
            rgb, icp, igp, cur_goal = self._render_lagged(gp, light, kw, lagged, iqs, rkw)
        elif self.lag_hist is not None:                   # the images lag; q, qd, the labels stay current
            iq, icp, icq = self.lag_hist.get(self.lagk)
            rgb = self.views.render(TF["iq"] if self.tb_render else iq, icp, icq, gp, dr=self.dr, light=light, **kw,
                                    **rkw)
        else:
            rgb = self.views.render(TF["q"] if self.tb_render else qf, cp, cq, gp, dr=self.dr, light=light, **kw,
                                    **rkw)
        cid = getattr(self.views, "call_id", None)
        same_call = cid is None or self.call_id is None or cid == self.call_id     # the last decision's light etc.
        self.call_id = cid
        torch.cuda.synchronize() if self.dev.type == "cuda" else None
        T["render"] = time.time() - t1 - T.get("table", 0.0)
        t2 = time.time()
        stud = ~self.teacher if self.it >= o.warm_teacher_only else torch.zeros_like(self.teacher)
        if TF is None:
            qd = self.qd_input(ws.qd.float().nan_to_num())                     # robot-like qdot in the qd_rob worlds
        qin = qf + self.qb if self.qb_h is not None else qf                    # q input offset (q_bias)
        if TF is not None:          # (table frame)
            qin, qd = qin + TF["b"], TF["qd"]
        now = torch.cat((qin, qd), -1)                                         # [N, 12]
        fr = self.fresh
        if self.H:
            self.rhist[fr] = now[fr][:, None, :].expand(-1, self.H, -1)
            robot_all = torch.cat((now, self.rhist.reshape(len(now), -1)), -1)
        else:
            robot_all = now
        if self.n_cmd:      # the last decision's executed targets (student or JT) - the measured q now; 0 at a start
            if env.u_exe is None:
                cmd = torch.zeros(len(now), self.n_cmd, device=self.dev)
            else:
                ue, qr = (env.u_exe, qf) if TF is None else (TF["u_exe"], TF["q"])       # (table frame)
                cmd = (ue.to(self.dev, torch.float32) - qr[:, None, :]).nan_to_num().reshape(len(now), -1)
                cmd = torch.where(fr[:, None], torch.zeros_like(cmd), cmd)
            robot_all = torch.cat((robot_all, cmd), -1)
        if self.LC:                     # the last executed command (every slot; 0 at the start, non-finite worlds;
                                        # after cmd)
            lc = torch.where(ok[:, None], self.lastcmd_input(q) if TF is None else
                             self.lastcmd_input(TF["q64"], TF["u_exe"]),
                             torch.zeros(1, 12, dtype=torch.float64, device=self.dev))
            robot_all = torch.cat((robot_all, lc.to(robot_all)), -1)
            self.lc_fresh = torch.zeros_like(self.lc_fresh)
        prgb = None
        if self.img_hist:   # the last decision's images of the same episode (an episode start, or a new per-call render
            own = fr if same_call else torch.ones_like(fr)                         # state: the current ones)
            prgb = rgb if self.prev_rgb is None else torch.where(own[:, None, None, None, None], rgb, self.prev_rgb)
            if self.prev_rgb is not None:
                self.stats["img_pairs"] += int((~fr).sum())
                self.stats["img_pairs_same"] += int((~fr).sum()) if same_call else 0
        self.last = dict(robot=robot_all, fresh=fr.clone(), prev_rgb=prgb, rgb=rgb, goal_pose=gp, aux_cp=icp)
        if TF is not None:
            self.last["tb"] = TF
        if self.il is not None:
            self.last.update(img_lag=self.lag_x.clone(), img_goal=igp, cur_goal=cur_goal)
        if bool(stud.any()):
            si = stud.nonzero().flatten()
            st.eval()
            img = to_input(rgb[si], fx=self.views.fx)
            pimg = to_input(prgb[si], fx=self.views.fx) if prgb is not None else None
            robot = robot_all[si]
            out = []
            for s in range(0, len(si), 256):
                with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=st.cfg.amp):
                    p_ = st.predict(img[s:s + 256], robot[s:s + 256]) if pimg is None else \
                        st.predict(img[s:s + 256], robot[s:s + 256], pimg[s:s + 256])
                    out.append(denormalise(st, p_))
            st.train()
            u = (qf if TF is None else TF["q"])[si][:, None, :] + torch.cat(out)      # relative to its reading
            K, n = u.shape[1], st.cfg.n_exec
            jj = torch.arange(n, device=self.dev)
            kk = (jj + st.cfg.jaw_lead).clamp_max(K - 1)
            tg[si] = torch.cat((u[:, jj, :5], u[:, kk, 5:]), -1).double()
            if TF is not None:                              # observed-frame targets -> applied (true) targets
                tg[si] = self.table_applied(tg[si], si, TF["b64"][si])
        T["student"] = time.time() - t2
        # samples: this decision's label completes the previous sample of every world that did not reset
        live = ok
        pv = self.prev_slot
        has = (pv >= 0) & live
        if TF is not None:      # table frame: only exact samples / labels (``table_frame``: skip, lab_bad)
            has = has & ~TF["lab_bad"]
            live = ok & ~TF["skip"]
        if env.goal_jump is not None:     # the goal moved since the previous sample: no second-half label for it
            has = has & ~env.goal_jump.just
        lab_b, q_b = (lab, qf) if TF is None else (TF["lab"], TF["q"])          # (table frame: Phi(lab), Phi(q))
        if bool(has.any()):
            self.buf.lab2[pv[has]] = lab_b[has].float()
            self.buf.valid2[pv[has]] = True
        if cur_goal is not None and not bool(cur_goal.all()):   # img_lag: the images still show the goal before a
            live = live & cur_goal                              # jump, the label is for the new one: no sample
            self.stale += int((ok & ~cur_goal).sum())
        idx = live.nonzero().flatten()
        if self.tb_render:      # the drawn cube: sheared by the plane height under it (``table_render``)
            from nedm.so101_policy import table_bias as TBI
            hz = torch.where(self.tb_on, TBI.height(self.tb_plane, icp[:, :2].double()), 0.0).to(icp.dtype)
            icp = icp + torch.cat((torch.zeros_like(icp[:, :2]), hz[:, None]), -1)
        aux = aux_targets(igp[idx], icp[idx])
        plink = None
        if self.img_hist:     # the link to the previous sample (cut when the per-call render state changed)
            plink = pv[idx] if same_call else torch.full_like(pv[idx], -1)
        slots = self.buf.add(rgb[idx], q_b[idx], robot_all[idx], lab_b[idx], aux, ~self.nominal[idx], prev=plink)
        if self.H:
            self.rhist = torch.cat((now[:, None, :], self.rhist[:, :-1]), 1)
        self.fresh = torch.zeros_like(self.fresh)
        if self.img_hist:
            self.prev_rgb = rgb
        self.prev_slot = torch.full_like(self.prev_slot, -1)
        self.prev_slot[idx] = slots
        self.samples += len(idx)
        t3 = time.time()
        hook = self._step_hook if self.qd_hist is not None else self._lag_hook if self.lag_hist is not None else None
        _, _, done, ex = env.step(targets=tg, step_hook=hook)
        T["env"] = time.time() - t3
        if bool(done.any()):
            di = done.nonzero().flatten()
            self.prev_slot[di] = -1                       # the next decision is a new episode
            self.fresh[di] = True
            self.lc_fresh[di] = True
            ep = ex["episode"]
            le = ep["lenient"][di]
            ts = self.teacher[di]
            self.stats["ep_teacher"] += int(ts.sum())
            self.stats["len_teacher"] += int((le & ts).sum())
            self.stats["ep_student"] += int((~ts).sum())
            self.stats["len_student"] += int((le & ~ts).sum())
            if "roll_cmd_max" in ep:
                ro = ep["roll_cmd_max"][di] > self.roll_cap + 1e-9
                self.stats["roll_over_teacher"] += int((ro & ts).sum())
                self.stats["roll_over_student"] += int((ro & ~ts).sum())
            if "goal_jumps" in ep:
                jm = ep["goal_jumps"][di] > 0
                self.stats["ep_jump_teacher"] += int((jm & ts).sum())
                self.stats["len_jump_teacher"] += int((jm & le & ts).sum())
                self.stats["ep_jump_student"] += int((jm & ~ts).sum())
                self.stats["len_jump_student"] += int((jm & le & ~ts).sum())
            self._draw(di)
            self._lag_reset(di)
            self._bias_reset(di)
            self._qd_reset(di)
        return T

    def qd_noise_scale(self, n):
        """[n] per-sample factor on the loss-side qdot noise (``OnlineConfig.qd_noise_zero_p``): U(0, 1), exactly 0
        for that share; None = off (the old loss)."""
        p = self.o.qd_noise_zero_p
        if p is None:
            return None
        s = torch.rand(n, device=self.dev, generator=self.gen)
        return torch.where(torch.rand(n, device=self.dev, generator=self.gen) < float(p), torch.zeros_like(s), s)

    def train_steps(self):
        st, o, b = self.st, self.o, self.st.cfg.batch
        info_sum, n = {}, 0
        for _ in range(int(o.steps_per_iter)):
            i = self.buf.sample(b, self.gen)
            rgb = self.buf.rgb[i]
            aug = self.buf.aug[i]
            fx = self.views.fx
            if self.img_hist:         # the previous images get the SAME per-image augmentation draws as the current
                gs = self.gen.get_state()
            img = to_input(rgb, self.drc, self.gen, fx=fx)
            img = torch.where(aug[:, None, None, None, None], img, to_input(rgb, fx=fx, fx_random=False))
            pimg = None
            if self.img_hist:
                ge = self.gen.get_state()
                prgb, pok = self.buf.prev_rgb(i)
                info_sum["prev_ok"] = info_sum.get("prev_ok", 0.0) + float(pok.float().mean())
                self.gen.set_state(gs)
                pimg = to_input(prgb, self.drc, self.gen, fx=fx)
                pimg = torch.where(aug[:, None, None, None, None], pimg, to_input(prgb, fx=fx, fx_random=False))
                self.gen.set_state(ge)
            q = self.buf.q[i]
            tgt = torch.cat((self.buf.lab1[i], self.buf.lab2[i]), 1)[:, :st.cfg.chunk] - q[:, None, :]
            valid = torch.cat((torch.ones(len(i), JE.STEPS, dtype=torch.bool, device=self.dev),
                               self.buf.valid2[i][:, None].expand(-1, JE.STEPS)), 1)[:, :st.cfg.chunk]
            robot = self.buf.robot[i]
            if o.qd_in_zero_p > 0 or o.qd_in_lo != 1.0 or o.qd_in_hi != 1.0:      # qdot input scale (training only)
                k = o.qd_in_lo + (o.qd_in_hi - o.qd_in_lo) * torch.rand(len(i), device=self.dev, generator=self.gen)
                k = torch.where(torch.rand(len(i), device=self.dev, generator=self.gen) < o.qd_in_zero_p,
                                torch.zeros_like(k), k).to(robot.dtype)
                robot = robot.clone()
                robot[:, 6:12] = robot[:, 6:12] * k[:, None]
            aux_n = (self.buf.aux[i] - st.aux_mean) / st.aux_std
            qs = self.qd_noise_scale(len(i))
            with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=st.cfg.amp):
                loss, info = st.loss(img, robot, normalise(st, tgt), valid, aux_n, qd_scale=qs, prev=pimg)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(st.parameters(), st.cfg.grad_clip)
            self.opt.step()
            for k, v in dict(loss=float(loss.detach()), **info).items():
                info_sum[k] = info_sum.get(k, 0.0) + v
            n += 1
        return {k: v / max(n, 1) for k, v in info_sum.items()}
