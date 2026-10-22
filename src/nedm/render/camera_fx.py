"""Camera response and image-space camera effects for the real-setup renderer (torch, any device).

Tone map (per camera, optionally per world): the linear colour of the ray tracer (Newton ``hdr_color``: albedo times
the summed light, no exposure, no gamma) -> sRGB in 0-1::

    x = hdr * exposure * wb[c] * cos(theta) ** vignette        (theta: the pixel's ray angle to the optical axis)
    x = x + black[c]                                           (flare / black level, linear)
    y = srgb_oetf(clamp(x, 0, 1)) ** gamma                     (gamma 1: plain sRGB)

Parameter vector per camera, :data:`TONE_KEYS` order: exposure, wb_r, wb_g, wb_b, black_r, black_g, black_b, gamma,
vignette.

Image-space effects (:class:`ImageFX`), applied to uint8 / float images [N, H, W, 3] after rendering, one draw per
image: nominal per-camera blur and chroma blur (applied by the real-scene renderer itself), Gaussian sensor noise,
Gaussian blur, linear motion blur, JPEG
compression (the robot's cameras stream MJPG;
own DCT implementation: 8 x 8 blocks, YCbCr 4:2:0, the IJG quality scaling of the standard tables), each with a
probability and a range. Nominal values and ranges come from ``real_scene_v1.json`` (``"image_fx"``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

TONE_KEYS = ("exposure", "wb_r", "wb_g", "wb_b", "black_r", "black_g", "black_b", "gamma", "vignette")
TONE_DEFAULT = {"exposure": 1.0, "wb": (1.0, 1.0, 1.0), "black": (0.0, 0.0, 0.0), "gamma": 1.0, "vignette": 0.0}


def tone_vector(d: dict | None) -> list[float]:
    """Tone-map dict (``exposure``, ``wb`` (3,), ``black`` (3,), ``gamma``, ``vignette``) -> the 9 values."""
    d = {**TONE_DEFAULT, **(d or {})}
    return [float(d["exposure"]), *map(float, d["wb"]), *map(float, d["black"]), float(d["gamma"]),
            float(d["vignette"])]


def tone_dict(v) -> dict:
    v = [float(x) for x in v]
    return {"exposure": v[0], "wb": v[1:4], "black": v[4:7], "gamma": v[7], "vignette": v[8]}


def srgb_oetf(x: torch.Tensor) -> torch.Tensor:
    """Linear 0-1 -> sRGB 0-1 (IEC 61966-2-1)."""
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(0.0031308) ** (1 / 2.4) - 0.055)


def srgb_eotf(y: torch.Tensor) -> torch.Tensor:
    """sRGB 0-1 -> linear 0-1."""
    return torch.where(y <= 0.04045, y / 12.92, ((y.clamp_min(0.04045) + 0.055) / 1.055) ** 2.4)


def tone_map(hdr: torch.Tensor, params: torch.Tensor, cos_theta: torch.Tensor | None = None) -> torch.Tensor:
    """hdr [B, C, H, W, 3] linear; params [C, 9] or [B, C, 9] (:data:`TONE_KEYS`); cos_theta [C, H, W] (needed when a
    vignette exponent is not 0) -> sRGB float 0-1 [B, C, H, W, 3]."""
    p = params.to(hdr.device, hdr.dtype)
    if p.dim() == 2:
        p = p[None]
    p = p[:, :, None, None, :]                                   # [B|1, C, 1, 1, 9]
    x = hdr * p[..., 0:1] * p[..., 1:4]
    if cos_theta is not None and bool((p[..., 8] != 0).any()):
        x = x * cos_theta.to(hdr.device, hdr.dtype)[None, :, :, :, None].clamp_min(1e-3) ** p[..., 8:9]
    x = (x + p[..., 4:7]).clamp(0.0, 1.0)
    return srgb_oetf(x) ** p[..., 7:8]


def to_uint8(y: torch.Tensor) -> torch.Tensor:
    return (y * 255.0 + 0.5).clamp(0, 255).to(torch.uint8)


# ---- image-space effects -----------------------------------------------------------------------------------------
def _as_float(img: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if img.dtype == torch.uint8:
        return img.float() / 255.0, True
    return img.float(), False


def gaussian_noise(img: torch.Tensor, sigma: torch.Tensor, generator=None) -> torch.Tensor:
    """img [N, H, W, 3] float 0-1; sigma [N] (0-1 units). Random numbers on the generator's device."""
    gdev = generator.device if generator is not None else img.device
    n = torch.randn(img.shape, device=gdev, generator=generator, dtype=img.dtype).to(img.device)
    return (img + n * sigma.to(img)[:, None, None, None]).clamp(0, 1)


def _conv_per_image(img: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """img [N, H, W, 3]; kernels [N, k, k] (normalised) -> [N, H, W, 3] (reflect padding)."""
    N, H, W, _ = img.shape
    k = kernels.shape[-1]
    x = img.permute(0, 3, 1, 2).reshape(1, N * 3, H, W)
    x = F.pad(x, (k // 2,) * 4, mode="reflect")
    w = kernels.to(img)[:, None].repeat_interleave(3, 0)         # [N * 3, 1, k, k]
    return F.conv2d(x, w, groups=N * 3).reshape(N, 3, H, W).permute(0, 2, 3, 1)


def gaussian_blur(img: torch.Tensor, sigma: torch.Tensor, k: int = 7) -> torch.Tensor:
    """img [N, H, W, 3] float; sigma [N] px (0 = unchanged)."""
    r = torch.arange(k, device=img.device, dtype=img.dtype) - k // 2
    s = sigma.to(img).clamp_min(1e-3)[:, None]
    g = torch.exp(-0.5 * (r[None] / s) ** 2)
    g = g / g.sum(-1, keepdim=True)
    return _conv_per_image(img, g[:, :, None] * g[:, None, :])


def chroma_blur(img: torch.Tensor, sigma: torch.Tensor, k: int = 9) -> torch.Tensor:
    """Blur only the colour (JFIF Cb, Cr) of img [N, H, W, 3] float 0-1 by a Gaussian of ``sigma`` [N] px; the luma
    stays sharp (the cameras' chroma subsampling / chroma denoising: small saturated objects lose colour)."""
    r, g, b = img.unbind(-1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb, cr = b - y, r - y
    c = gaussian_blur(torch.stack([cb, cr, torch.zeros_like(cb)], -1), sigma, k)
    cb, cr = c[..., 0], c[..., 1]
    r2, b2 = y + cr, y + cb
    g2 = (y - 0.299 * r2 - 0.114 * b2) / 0.587
    return torch.stack([r2, g2, b2], -1).clamp(0, 1)


def motion_blur(img: torch.Tensor, length: torch.Tensor, angle: torch.Tensor, k: int = 9) -> torch.Tensor:
    """img [N, H, W, 3] float; a line kernel of ``length`` px [N] at ``angle`` rad [N] (anti-aliased by distance)."""
    r = torch.arange(k, device=img.device, dtype=img.dtype) - k // 2
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    c, s = torch.cos(angle).to(img)[:, None, None], torch.sin(angle).to(img)[:, None, None]
    along = xx[None] * c + yy[None] * s
    across = -xx[None] * s + yy[None] * c
    half = (length.to(img) / 2).clamp_min(0.0)[:, None, None]
    w = (1 - across.abs()).clamp_min(0) * (half + 0.5 - along.abs()).clamp(0, 1)
    w = w / w.sum((-1, -2), keepdim=True).clamp_min(1e-6)
    return _conv_per_image(img, w)


_LUMA_Q = torch.tensor([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
                        [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
                        [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
                        [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], dtype=torch.float32)
_CHROMA_Q = torch.full((8, 8), 99.0)
_CHROMA_Q[:4, :4] = torch.tensor([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]])


def _dct_matrix(device, dtype) -> torch.Tensor:
    n = torch.arange(8, device=device, dtype=dtype)
    m = torch.cos((2 * n[None, :] + 1) * n[:, None] * math.pi / 16) * math.sqrt(2 / 8)
    m[0] = m[0] / math.sqrt(2)
    return m


def _qtable(base: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    """IJG scaling: [N, 8, 8] tables for qualities [N] (1-100)."""
    q = quality.clamp(1, 100)
    scale = torch.where(q < 50, 5000 / q, 200 - 2 * q)
    return ((base[None] * scale[:, None, None] + 50) / 100).floor().clamp(1, 255)


def _blocks_q(x: torch.Tensor, table: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """x [N, H, W] (0-255 minus 128), H, W multiples of 8 -> quantised and reconstructed."""
    N, H, W = x.shape
    b = x.reshape(N, H // 8, 8, W // 8, 8).permute(0, 1, 3, 2, 4)            # [N, by, bx, 8, 8]
    c = D @ b @ D.T
    t = table[:, None, None]
    c = torch.round(c / t) * t
    return (D.T @ c @ D).permute(0, 1, 3, 2, 4).reshape(N, H, W)


def jpeg(img: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    """JPEG round trip of img [N, H, W, 3] float 0-1 at ``quality`` [N] (1-100): YCbCr (JFIF), 4:2:0 chroma (2 x 2
    mean, bilinear back), 8 x 8 DCT with the IJG-scaled standard tables. Edge padding to multiples of 16."""
    N, H, W, _ = img.shape
    ph, pw = (-H) % 16, (-W) % 16
    x = img.permute(0, 3, 1, 2) * 255.0
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    D = _dct_matrix(img.device, torch.float32)
    qq = quality.to(img.device, torch.float32)
    y = _blocks_q(y.float() - 128, _qtable(_LUMA_Q.to(img.device), qq), D) + 128
    sub = lambda c: F.avg_pool2d(c[:, None].float(), 2)[:, 0]                # noqa: E731
    tc = _qtable(_CHROMA_Q.to(img.device), qq)
    cb = _blocks_q(sub(cb) - 128, tc, D) + 128
    cr = _blocks_q(sub(cr) - 128, tc, D) + 128
    up = lambda c: F.interpolate(c[:, None], scale_factor=2, mode="bilinear", align_corners=False)[:, 0]  # noqa
    cb, cr = up(cb) - 128, up(cr) - 128
    out = torch.stack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], 1)
    out = out[:, :, :H, :W].permute(0, 2, 3, 1) / 255.0
    return out.clamp(0, 1).to(img.dtype)


@dataclass
class ImageFX:
    """Per-image camera effects after rendering, one draw per image. Each effect: probability ``p_*`` and a range;
    the nominal effects (always on: the real camera's sharpness ``nominal_blur_px`` and colour bleeding
    ``nominal_chroma_px``, a number or a dict per camera name) come first. Values in px or 0-1 units."""

    nominal_blur_px: float | dict = 0.0
    nominal_chroma_px: float | dict = 0.0
    p_noise: float = 0.8
    noise: tuple = (0.004, 0.02)            # std (0-1 units)
    p_blur: float = 0.3
    blur_px: tuple = (0.3, 1.0)             # Gaussian sigma (px at the output size)
    p_motion: float = 0.2
    motion_px: tuple = (1.0, 3.0)           # line length (px)
    p_jpeg: float = 0.8
    jpeg_quality: tuple = (50, 95)
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> "ImageFX":
        d = dict(d or {})
        known = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known, extra={k: v for k, v in d.items() if k not in cls.__dataclass_fields__})

    def draw(self, n: int, generator=None, device="cpu") -> dict:
        """Per-image draws (random numbers on the generator's device, results on ``device``)."""
        gdev = generator.device if generator is not None else torch.device(device)

        def u(lo, hi):
            return (lo + (hi - lo) * torch.rand(n, generator=generator, device=gdev)).to(device)

        def on(p):
            return (torch.rand(n, generator=generator, device=gdev) < p).to(device)

        return {"noise": torch.where(on(self.p_noise), u(*self.noise), 0.0),
                "blur": torch.where(on(self.p_blur), u(*self.blur_px), 0.0),
                "motion": torch.where(on(self.p_motion), u(*self.motion_px), 0.0),
                "motion_angle": u(0.0, math.pi),
                "jpeg": torch.where(on(self.p_jpeg), u(*self.jpeg_quality).round(), 0.0)}

    @staticmethod
    def _per_camera(v, camera):
        if isinstance(v, dict):
            return float(v.get(camera, 0.0)) if camera is not None else 0.0
        return float(v)

    def nominal(self, x: torch.Tensor, camera: str | None) -> torch.Tensor:
        """The nominal (always-on) effects of ``camera`` on float images [N, H, W, 3]: chroma blur, then blur.
        :class:`~nedm.render.so101.SO101Renderer` applies them itself for a real scene with ``image_fx``."""
        chroma = self._per_camera(self.nominal_chroma_px, camera)
        if chroma > 0:
            x = chroma_blur(x, torch.full((len(x),), chroma, device=x.device))
        blur = self._per_camera(self.nominal_blur_px, camera)
        if blur > 0:
            x = gaussian_blur(x, torch.full((len(x),), blur, device=x.device))
        return x

    def apply(self, img: torch.Tensor, draw: dict | None = None, generator=None, camera: str | None = None,
              nominal: bool = False) -> torch.Tensor:
        """img [N, H, W, 3] uint8 or float 0-1 -> same dtype: the random effects of ``draw`` (None: none). The
        nominal effects only with ``nominal`` (the real-scene renderer already applied them to its output);
        ``camera``: the camera name for per-camera nominal effects (all images of one call from one camera)."""
        x, was_u8 = _as_float(img)
        dev = x.device
        if nominal:
            x = self.nominal(x, camera)
        if draw is not None:
            d = {k: v.to(dev) for k, v in draw.items()}
            sel = d["blur"] > 0
            if sel.any():
                x[sel] = gaussian_blur(x[sel], d["blur"][sel])
            sel = d["motion"] > 0
            if sel.any():
                x[sel] = motion_blur(x[sel], d["motion"][sel], d["motion_angle"][sel])
            sel = d["noise"] > 0
            if sel.any():
                x[sel] = gaussian_noise(x[sel], d["noise"][sel], generator)
            sel = d["jpeg"] > 0
            if sel.any():
                x[sel] = jpeg(x[sel], d["jpeg"][sel])
        return to_uint8(x) if was_u8 else x
