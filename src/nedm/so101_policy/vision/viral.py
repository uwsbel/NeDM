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
Studentlab (2026-10-22, ~/so101_studentlab_claude/trials/BUILD.md; all OFF by default, old configs, checkpoints and
results bit for bit unchanged): ``ViralConfig.cmd_in`` (the previous decision's 5 executed targets - the measured q
now, / ``cmd_scale``, appended to the robot vector), ``ViralConfig.img_hist`` (the previous decision's images through
the same encoders, features appended to the head input; ``ReplayBuffer`` links to the previous sample instead of
copies), ``expand_student`` (zero-init expansion: the same chunk as the given student), ``OnlineConfig.goal_jump``
(``joint_env.GoalJumpConfig``: goal jumps while the cube is held near the goal). Saved configs keep the old keys while
the new inputs are off (``cfg_dict``).

- Robot timing and servo in the online loop (``OnlineConfig.dyn`` -> ``joint_env.RobotDynConfig``: command delay with
  jitter, weaker arm servo, command lag, gravity sag, command noise; per world per episode), robot-like qdot inputs
  (``qd_robot_p``: ``robot_qd``, the slope of the quantised q, exactly 0 when still), longer episodes (``horizon``,
  ``hold_cap_s``) and a per-sample scale of the loss-side qdot noise (``qd_noise_zero_p``). Evaluation:
  ``$NEDM_SO101_DYN`` (``joint_teacher.make_env``).
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
    cam_offset [n, C, 6], env_pose [n, n_env, 7] (``RealDR``) and exposure [n, C] (tone map only); the scene's lights
    (randomised per call by ``RealDR.lights`` when ``light``); ``fx`` = the cameras' ``CamFX`` (None: sim scene)."""

    def __init__(self, num_worlds, res=224, cameras=CAMERAS, backend="newton", tiled=True, shadows=True,
                 top_eye=V.TOP_EYE, top_target=V.TOP_TARGET, dr: DRConfig | None = None, seed=0, device="cuda",
                 real: RealSetup | None = None):
        self.real, self.fx = real, None
        if real is not None:
            self._init_real(num_worlds, res, cameras, backend, tiled, shadows, dr, seed, device, real)
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
        self.r = SO101Renderer(self.num_worlds, width=self.res, height=self.res, backend="newton", cameras=self.cameras,
                               shadows=shadows, goal="block", real=rs.real)
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
        self.lights0 = [dict(lt) for lt in rs.real.lights]
        self.models0 = list(sc.camera_models)
        self.intrinsics_dr = float(getattr(self.rdr, "focal", 0.0)) > 0 or float(getattr(self.rdr, "pp_px", 0.0)) > 0
        self._call_state = "nominal"
        self.fx = ViewFX.from_setup(rs, self.cameras)

    def nominal_draw(self, n):
        """Per-row draws with the nominal visuals (real scene)."""
        C = len(self.cameras)
        d = dict(colors=self.base.expand(n, self.nb, 3).clone(),
                 cam_offset=torch.zeros(n, C, 6, device=self.dev),
                 env_pose=self.env0.expand(n, *self.env0.shape).clone())
        if self.tone0 is not None:
            d["tone"] = self.tone0.expand(n, *self.tone0.shape).clone()
        return d

    def merge_nominal(self, d, nominal):
        """Rows ``nominal`` [n] bool of the draws ``d`` replaced by the nominal visuals (real scene)."""
        nd = self.nominal_draw(len(nominal))
        return {k: torch.where(nominal.view(-1, *([1] * (v.dim() - 1))), nd[k].to(v), v) for k, v in d.items()}

    def _draw_real(self, n):
        d = self.rdr.draw(self.r.scene, n, self.gen, device=self.dev)
        return {k: v.to(self.dev) for k, v in d.items()}

    def _per_call_real(self, nominal):
        """Lights (and the intrinsics, if the scene randomises them) of the call: shared by all worlds."""
        if nominal:
            if self._call_state != "nominal":
                self.r.set_lights(self.lights0)
                if self.intrinsics_dr:
                    self.r.set_camera_models(self.models0)
                self._call_state = "nominal"
            return
        self.r.set_lights(self.rdr.lights(self.lights0, self.light_gen))
        if self.intrinsics_dr:
            self.r.set_camera_models(self.rdr.camera_models(self.r.scene, self.light_gen, out_res=self.res))
        self._call_state = "random"

    def _render_real(self, q, cube_pos, cube_quat, goal_pose, dr=None, light=None):
        t0 = time.time()
        self._per_call_real((dr is None) if light is None else not light)
        out = []
        for s in range(0, q.shape[0], self.num_worlds):
            e = min(q.shape[0], s + self.num_worlds)
            d = None if dr is None else {k: v[s:e] for k, v in dr.items()}
            out.append(self.r.render_dr(q[s:e], cube_pos[s:e], cube_quat[s:e], goal_pose[s:e], draw=d).to(self.dev))
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
        if self.real is not None:
            return self._render_real(q, cube_pos, cube_quat, goal_pose, dr, light)
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


STAT_KEYS = ("robot_mean", "robot_std", "tgt_mean", "tgt_std", "aux_mean", "aux_std")
AUX_DIM = 7
NEW_INPUT_KEYS = ("cmd_in", "cmd_scale", "img_hist")      # left out of saved configs while off (``cfg_dict``)
ARCH_KEYS = ("hist", "cmd_in", "img_hist")                 # change the head's input: ``expand_student``


def cfg_dict(cfg: ViralConfig) -> dict:
    """The config as saved in checkpoints / meta files: ``asdict`` without the studentlab input options while they are
    off (so a student without them stays readable by code that predates them); with any of them on, all are saved."""
    d = asdict(cfg)
    if not cfg.cmd_in and not cfg.img_hist and tuple(cfg.cmd_scale) == CMD_SCALE:
        for k in NEW_INPUT_KEYS:
            d.pop(k, None)
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

    def features(self, images):
        return torch.cat([e(images[:, i]) for i, e in enumerate(self.enc)], -1)

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

    def forward(self, images, robot, qd_scale=None, prev=None):
        """``prev`` (``img_hist``): the previous decision's images [B, C, 3, H, W] (None: the current ones, as at an
        episode start); both through the same encoders, the previous features appended to the head input."""
        if getattr(self.cfg, "img_hist", False):
            B = images.shape[0]
            if prev is None:
                f = self.features(images)
                fp = f
            else:
                both = self.features(torch.cat((images, prev.to(images)), 0))
                f, fp = both[:B], both[B:]
            x = torch.cat((f, self.robot(robot, qd_scale).to(f.dtype), fp), -1)
            return self.head(x.float()).view(-1, self.cfg.chunk, 6), f
        f = self.features(images)
        return self.head(torch.cat((f, self.robot(robot, qd_scale).to(f.dtype)), -1).float()).view(
            -1, self.cfg.chunk, 6), f

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


def expand_hist(st: ViralNet, hist: int, device="cuda"):
    """A student with ``hist`` decisions of robot history from one without (or with fewer): the same weights, zero
    weights on the new history inputs of the head's first layer (so it starts as the given student)."""
    if hist < int(st.cfg.hist):
        raise ValueError("cannot shrink the history")
    return expand_student(st, hist=hist, device=device)


def save_student(st, path, **extra):
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
    at rest: every entry = the start state), ``push`` after every 20 ms step, ``get(k)`` [B] -> (q, cube pos, quat)."""

    def __init__(self, B, kmax, device):
        self.x = torch.zeros(B, int(kmax) + 1, 13, device=device)
        self.ready = False

    def fill(self, idx, q, cp, cq):
        self.x[idx] = _state13(q, cp, cq).to(self.x)[:, None].expand(-1, self.x.shape[1], -1)
        self.ready = True

    def push(self, q, cp, cq):
        self.x = torch.cat((_state13(q, cp, cq).to(self.x)[:, None], self.x[:, :-1]), 1)

    def newest(self):
        return self.x[:, 0]

    def get(self, k):
        x = self.x[torch.arange(len(self.x), device=self.x.device), k.to(self.x.device).long()]
        return x[:, :6], x[:, 6:9], x[:, 9:13]


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

    Studentlab inputs: ``cfg.cmd_in`` needs the env (``bind(env)``: its ``u_exe``, the targets the action layer
    executed in the last decision); ``cfg.img_hist`` keeps the last decision's rendered images. Both start from zero /
    the current images at the first call after ``reset``."""

    def __init__(self, student: ViralNet, views: DRViews, randomise=False, jaw_lead=None, device="cuda", lag=None,
                 lag_range=None, seed=0, qd_window=0):
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
        self.env = None         # cmd_in: the env whose u_exe holds the last decision's executed targets (``bind``)
        self.n_dec = 0          # decisions since ``reset``
        self.prev_rgb = None    # img_hist: the last decision's rendered images

    def bind(self, env):
        self.env = env
        return self

    def reset(self, B):
        self.dr = self.views.draw(B) if self.randomise else None
        self.n_dec = 0
        self.prev_rgb = None
        self.rhist = None
        self.qd_hist = None
        self.qd_log = []
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
        rgb = self.views.render(iq, icp, icq, goal_pose.to(self.dev, torch.float32), dr=self.dr)
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
        prev = None
        if getattr(self.st.cfg, "img_hist", False):    # the last decision's images (the first call: the current ones)
            prev = img if self.prev_rgb is None else to_input(self.prev_rgb, fx=self.views.fx)
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
    # studentlab (2026-10-22): goal jumps in the online env (``JE.GoalJumpConfig`` dict; {} = off): the goal moves
    # while the cube is held near it, so the cube is past (or before) the new goal; the rendered goal, the JT
    # observation and the aux targets read the moved goal from the next decision on; the sample of the decision
    # before a jump gets no second-half label (``valid2`` False: its images showed the old goal)
    goal_jump: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d):
        d = dict(d or {})
        known = set(OnlineConfig.__dataclass_fields__)
        bad = sorted(set(d) - known - {"note"})
        if bad:
            raise ValueError(f"unknown online config keys {bad}")
        if d.get("lag_range"):
            d["lag_range"] = tuple(int(v) for v in d["lag_range"])
            if len(d["lag_range"]) != 2 or not 0 <= d["lag_range"][0] <= d["lag_range"][1]:
                raise ValueError(f"online lag_range {d['lag_range']}: (lo, hi) control steps, 0 <= lo <= hi")
        if d.get("goal_jump"):
            JE.goal_jump_config(d["goal_jump"])                                # validate now
        return OnlineConfig(**{k: v for k, v in d.items() if k in known})


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


class OnlineDAgger:
    """The online loop (``OnlineConfig``). Tasks: the start states and goals of a task table (``state_*``, goal_xy,
    goal_yaw), drawn uniformly at every reset. Image lag (``st.cfg.lag_range``, else ``st.cfg.lag``): drawn per world
    per episode; every world's 20 ms states feed a ``LagHistory`` (``JointVecEnv.step`` hook). Robot timing / servo
    (``OnlineConfig.dyn``: the env's ``RobotDyn``) and robot-like qdot inputs (``qd_robot_p``: per-world tick history of
    the 20 ms states, same hook) are drawn per world per episode as well."""

    def __init__(self, st: ViralNet, jt, tasks, nrd_ckpt, views: DRViews, ocfg: OnlineConfig, device="cuda", seed=0,
                 log=log_default, world="nrd"):
        """``world``: the online env's world (``nrd``; tests: ``kin``)."""
        self.st, self.jt, self.views, self.o, self.dev, self.log = st, jt, views, ocfg, torch.device(device), log
        N = int(ocfg.envs)
        cfg = JE.JointEnvConfig(world=world, num_envs=N, device=device, ckpts=[str(nrd_ckpt)], auto_reset=True,
                                stagger=True, starts="", seed=seed, horizon=int(ocfg.horizon), dyn=ocfg.dyn or None,
                                pause=ocfg.pause or None, goal_jump=ocfg.goal_jump or None,
                                **JE.hold_cap_kw(ocfg.hold_cap_s, ocfg.hold_obs_s))
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
        if ocfg.dyn_student_only and self.env.dyn is not None:
            self.env.dyn.set_eligible(~self.teacher)
        self.nominal = torch.rand(N, device=self.dev, generator=self.gen) < ocfg.nominal_share
        self.H = int(st.cfg.hist)
        self.n_cmd = int(st.cfg.n_exec) * 6 if getattr(st.cfg, "cmd_in", False) else 0     # cmd input (studentlab)
        self.img_hist = bool(getattr(st.cfg, "img_hist", False))                       # image history (studentlab)
        self.buf = ReplayBuffer(ocfg.buffer, len(st.cfg.cameras), st.cfg.res, self.dev, 12 * (1 + self.H) + self.n_cmd,
                                prev_link=self.img_hist)
        self.rhist = torch.zeros(N, max(self.H, 1), 12, device=self.dev)
        self.fresh = torch.ones(N, dtype=torch.bool, device=self.dev)            # the next decision starts an episode
        self.prev_rgb = None                                                     # img_hist: the last decision's images
        self.last = {}                                                           # the last decision's inputs (tests)
        self.kmax = int(st.cfg.lag_max)
        self.lag_hist = LagHistory(N, self.kmax, self.dev) if self.kmax > 0 else None
        lo, hi = (int(v) for v in st.cfg.lag_range)
        self.lag_lo, self.lag_hi = (lo, hi) if hi > 0 else (int(st.cfg.lag), int(st.cfg.lag))
        self.lagk = torch.zeros(N, dtype=torch.long, device=self.dev)
        self.lag_gen = torch.Generator(device=self.dev).manual_seed(int(seed) + 77)
        self.qb = torch.zeros(N, 6, device=self.dev)       # per-world offset of the student's q input (rad)
        self.qb_h = None
        if ocfg.q_bias_p > 0 and ocfg.q_bias_deg:
            if len(ocfg.q_bias_deg) != 6:
                raise ValueError("q_bias_deg: 6 half-widths (deg)")
            self.qb_h = torch.deg2rad(torch.tensor([float(v) for v in ocfg.q_bias_deg], device=self.dev))
            self.qb_gen = torch.Generator(device=self.dev).manual_seed(int(seed) + 91)
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

    def _lag_reset(self, idx):
        """Episode start of worlds ``idx`` (the env already holds their start states): lag draw, history = start."""
        if self.lag_hist is None:
            return
        ws = self.env.ws
        q, cp, cq = _sane(ws.q[idx], ws.cube_pos[idx], ws.cube_quat[idx], self.dev)
        self.lag_hist.fill(idx, q, cp, cq)
        self.lagk[idx] = torch.randint(self.lag_lo, self.lag_hi + 1, (len(idx),), device=self.dev,
                                       generator=self.lag_gen)

    def _bias_reset(self, idx):
        """Episode start of worlds ``idx``: draw the offset of the student's q input (``OnlineConfig.q_bias_*``)."""
        if self.qb_h is None:
            return
        n = len(idx)
        on = torch.rand(n, device=self.dev, generator=self.qb_gen) < self.o.q_bias_p
        u = torch.rand(n, 6, device=self.dev, generator=self.qb_gen) * 2 - 1
        self.qb[idx] = u * self.qb_h[None] * on[:, None]

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

    def _lag_hook(self, e, j):
        self.lag_hist.push(*_sane(e.ws.q, e.ws.cube_pos, e.ws.cube_quat, self.dev))

    def _step_hook(self, e, j):
        """After every 20 ms step: the image-lag history and / or the tick history of the robot-like qdot."""
        if self.lag_hist is not None:
            self._lag_hook(e, j)
        if self.qd_hist is not None:
            self.qd_hist = torch.cat((qd_ticks(e.ws.q).to(self.qd_hist)[:, None], self.qd_hist[:, :-1]), 1)

    def log_extra(self):
        """Columns of the online log for the robot options that are on (none: {} = the old columns)."""
        out = {}
        if self.env.dyn is not None:
            out.update(self.env.dyn.summary())
        if self.env.pause is not None:
            out.update(self.env.pause.summary())
        if self.qd_hist is not None:
            out["qd_robot"] = round(float(self.qd_rob.float().mean()), 3)
        if self.env.goal_jump is not None:
            out.update(self.env.goal_jump.summary())
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
            obs[:, 6:12] = torch.where(env.dyn.on[:, None], v, obs[:, 6:12])               # obs layout: q, qd, ...
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
        icp = cp
        if self.lag_hist is not None:                     # the images lag; q, qd, the labels stay current
            iq, icp, icq = self.lag_hist.get(self.lagk)
            rgb = self.views.render(iq, icp, icq, gp, dr=self.dr, light=light)
        else:
            rgb = self.views.render(qf, cp, cq, gp, dr=self.dr, light=light)
        torch.cuda.synchronize() if self.dev.type == "cuda" else None
        T["render"] = time.time() - t1
        t2 = time.time()
        stud = ~self.teacher if self.it >= o.warm_teacher_only else torch.zeros_like(self.teacher)
        qd = self.qd_input(ws.qd.float().nan_to_num())                         # robot-like qdot in the qd_rob worlds
        now = torch.cat((qf + self.qb if self.qb_h is not None else qf, qd), -1)   # [N, 12] (q input offset)
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
                cmd = (env.u_exe.to(self.dev, torch.float32) - qf[:, None, :]).nan_to_num().reshape(len(now), -1)
                cmd = torch.where(fr[:, None], torch.zeros_like(cmd), cmd)
            robot_all = torch.cat((robot_all, cmd), -1)
        prgb = None
        if self.img_hist:   # the last decision's images of the same episode (an episode start: the current ones)
            prgb = rgb if self.prev_rgb is None else torch.where(fr[:, None, None, None, None], rgb, self.prev_rgb)
        self.last = dict(robot=robot_all, fresh=fr.clone(), prev_rgb=prgb, rgb=rgb, goal_pose=gp, aux_cp=icp)
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
        if env.goal_jump is not None:     # the goal moved since the previous sample: no second-half label for it
            has = has & ~env.goal_jump.just
        if bool(has.any()):
            self.buf.lab2[pv[has]] = lab[has].float()
            self.buf.valid2[pv[has]] = True
        idx = live.nonzero().flatten()
        aux = aux_targets(gp[idx], icp[idx])
        slots = self.buf.add(rgb[idx], qf[idx], robot_all[idx], lab[idx], aux, ~self.nominal[idx],
                             prev=pv[idx] if self.img_hist else None)
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
            ep = ex["episode"]
            le = ep["lenient"][di]
            ts = self.teacher[di]
            self.stats["ep_teacher"] += int(ts.sum())
            self.stats["len_teacher"] += int((le & ts).sum())
            self.stats["ep_student"] += int((~ts).sum())
            self.stats["len_student"] += int((le & ~ts).sum())
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
                prgb, _ = self.buf.prev_rgb(i)
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
