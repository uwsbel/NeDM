"""Background randomisation of the real-setup renderer (studentlab external-camera fix, 2026-10-23;
``~/so101_studentlab_claude/diag/extcam/VERDICT.md`` sec. 4): image-space layers in LINEAR colour, applied by
:class:`nedm.render.so101.SO101Renderer` to Newton's linear output BEFORE the tone map (so the exposure, white balance,
black level, gamma and vignette randomisation act on them as well), only on the pixels whose body (the renderer's
segment output) is in a background group. The plate, the cube, the arm and the camera board are never changed.

Off by default: ``RealDR.bg = {}``. A scene file's ``"dr"`` (or ``ViralConfig.real_dr``) turns it on with a ``"bg"``
dict of :class:`BgDR` fields. Per world and camera, one draw for the whole episode (:meth:`BgDR.draw`, called by
``RealDR.draw``), stored as tensors in the draw dict (rows = worlds, so ``DRViews.merge_nominal`` and the online loop's
per-world reset work unchanged):

- ``bg_tex`` [n, Cb, H, W, 3] float16: added to the dark area (groups ``dark_groups``: the room and the lab floor
  outside the light box): procedural texture (``tex_*``, probability ``tex_p``), clutter bars and rectangles
  (``clutter_*``), dim colour spots (``spot_*``, red / orange / green / blue, also in the image corners).
- ``bg_low`` [n, Cb, H, W] float16: relative brightness change of the floor and the walls (low texture, ``low_*``).
- ``bg_mat`` [n, Cb, H, W] float16: the same for the mat (low texture + fold lines, ``fold_*``).
- ``bg_par`` [n, Cb, NPAR] float32: on, flat dark-area profile (else the rendered shading times a linear ramp),
  ramp amplitude, ramp direction, boundary blur sigma (px), floor gradient amplitude, floor gradient direction,
  floor radial exponent, mat gradient amplitude, mat gradient direction, spare x 2.

At render (:func:`apply_bg`): dark area -> (flat mean or ramp-shaded rendered colour) + ``bg_tex``, clamped at 0;
floor x (1 + floor gradient) x cos(theta) ** -floor_radial x (1 + ``bg_low``); walls x (1 + ``bg_low``); mat x
(1 + mat gradient) x (1 + ``bg_mat``); then the floor / dark-area boundary band (3 px each side) gets a normalised
Gaussian blur over the floor and dark pixels only (sigma ``edge_blur_px``). Gradients are linear ramps from -1 to 1
across the image in a random direction. ``floor_radial`` (theta = the pixel ray's angle to the optical axis: the
tone map's vignette term) undoes part of the tone-map vignette on the floor only: the real floor stays bright up to
its edge in the lower-left corner while the fitted vignette (exponent 2.11) darkens the sim floor there (coverage
check, trials/bgdr/BUILD.md). Layers are at the student resolution (``res``); with supersampling they are repeated
per sub-pixel.

Texture family ``train_noise_v1`` (the ONLY family of training; held-out gate probes must use another one, e.g.
Gaussian-filtered white noise as the forward tests did, Perlin gradient noise, cells or stripes):
value noise = unit Gaussian values on square lattices of ``tex_cells`` px cells (1, 2, 4, 8 px at 224 px), each
lattice bilinearly interpolated with a random sub-cell phase (cell 1 = white noise), each octave normalised to unit
std, octave weights U(0, 1) normalised to unit norm; monochrome field times a per-world channel tint 1 +- ``tex_tint``.
Clutter: soft-edged (0.5 px) bars (length 20-160 px, width 1.5-6 px, any angle) or rectangles (6-50 px sides,
+-0.2 rad), "over" composited, neutral colour with +- ``clutter_tint`` per channel. Spots: soft discs. Folds: lines
of Gaussian profile. Placement of the clutter and of the spots that are not in a corner: uniform over the image, or,
with a ``prior`` map per camera (``draw(..., prior=)``; ``SO101Renderer.bg_prior``: the dark area of one nominal
frame), from the prior (+ ``prior_floor`` of uniform weight), so that most items land in the dark area (the layer is
masked to it at render).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

FAMILY = "train_noise_v1"
KEYS = ("bg_tex", "bg_low", "bg_mat", "bg_par")
NPAR = 12
# linear colours of the spots (red, orange, green, blue), times the brightness draw
SPOT_COLORS = ((0.9, 0.08, 0.05), (0.95, 0.4, 0.05), (0.1, 0.7, 0.15), (0.08, 0.2, 0.9))
BAND_PX = 3                 # half width of the floor / dark-area boundary band (output px)
CHUNK = 128                 # worlds per generation chunk (bounded temporaries)


@dataclass
class BgDR:
    """Ranges (``(lo, hi)`` uniform, ``*_p`` probabilities per world and camera; module docstring)."""

    family: str = FAMILY
    cameras: tuple = ("external",)
    p: float = 1.0                          # per world: background randomisation at all (else the rendered one)
    dark_groups: tuple = ("room", "room_floor")
    tex_p: float = 0.8
    tex_amp: tuple = (0.02, 0.16)           # std of the added texture, linear colour
    tex_cells: tuple = (1, 2, 4, 8)
    tex_tint: float = 0.25
    clutter_p: float = 0.5
    clutter_n: tuple = (1, 6)
    clutter_bright: tuple = (0.02, 0.3)
    clutter_tint: float = 0.3
    spot_p: float = 0.5
    spot_n: tuple = (1, 3)
    spot_px: tuple = (3.0, 15.0)            # diameter
    spot_bright: tuple = (0.05, 0.3)
    spot_corner_p: float = 0.5              # per spot: within 30 px of a random image corner
    flat_p: float = 0.5                     # dark-area profile flat (its mean colour), else rendered x linear ramp
    ramp_amp: tuple = (0.0, 0.5)
    edge_blur_px: tuple = (0.0, 1.5)
    floor_grad: float = 0.15                # +- relative, from one side of the image to the other
    floor_radial: tuple = (0.0, 0.0)        # floor x cos(theta) ** -a, a ~ U(lo, hi) (0: off)
    mat_grad: float = 0.0                   # +- relative mat brightness gradient (0: off)
    low_p: float = 0.5
    low_amp: tuple = (0.01, 0.04)
    fold_p: float = 0.5
    fold_n: tuple = (1, 4)
    fold_px: tuple = (0.35, 1.0)            # Gaussian sigma of a fold line
    fold_gain: tuple = (-0.3, 0.8)
    prior_floor: float = 0.1                # share of uniform placement when a prior map is given

    @classmethod
    def from_dict(cls, d: dict | None) -> "BgDR":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        bad = sorted(set(d) - known - {"note"})
        if bad:
            raise ValueError(f"unknown RealDR.bg keys {bad}")
        d = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in known}
        out = cls(**d)
        if out.family != FAMILY:
            raise ValueError(f"RealDR.bg family {out.family!r}: training implements only {FAMILY!r}")
        if any(int(c) < 1 for c in out.tex_cells):
            raise ValueError("tex_cells: lattice cell sizes >= 1 px")
        return out

    def cams(self, cameras) -> tuple:
        """The scene's cameras (in its order) that get the layers."""
        return tuple(c for c in cameras if c in self.cameras)

    # ---- draws ------------------------------------------------------------------------------------------------------
    @staticmethod
    def nominal(n: int, cb: int, res: int, device) -> dict:
        """Layers of n worlds without background randomisation (zeros; ``on`` 0)."""
        H = W = int(res)
        return dict(bg_tex=torch.zeros(n, cb, H, W, 3, dtype=torch.float16, device=device),
                    bg_low=torch.zeros(n, cb, H, W, dtype=torch.float16, device=device),
                    bg_mat=torch.zeros(n, cb, H, W, dtype=torch.float16, device=device),
                    bg_par=torch.zeros(n, cb, NPAR, dtype=torch.float32, device=device))

    def draw(self, n: int, cb: int, res: int, generator: torch.Generator | None = None, device=None,
             prior: torch.Tensor | None = None) -> dict:
        """Per-world, per-camera layers [n, cb, ...] (module docstring). Random numbers on the generator's device
        (CPU without one); results on ``device``. ``prior`` [cb, res, res] >= 0: placement weights of the clutter and
        spots (None: uniform)."""
        gdev = torch.device(generator.device) if generator is not None else torch.device("cpu")
        dev = torch.device(device) if device is not None else gdev
        out = self.nominal(n, cb, res, dev)
        if n == 0 or cb == 0:
            return out
        pw = None
        if prior is not None:
            pr = torch.as_tensor(prior, dtype=torch.float32).to(gdev)
            if tuple(pr.shape) != (cb, res, res):
                raise ValueError(f"prior {tuple(pr.shape)}: expected ({cb}, {res}, {res})")
            pr = pr.clamp_min(0).flatten(1)
            pr = pr / pr.sum(1, keepdim=True).clamp_min(1e-12)
            pw = (1 - self.prior_floor) * pr + self.prior_floor / pr.shape[1]             # [cb, res * res]
        for s in range(0, n, CHUNK):
            e = min(n, s + CHUNK)
            d = self._draw(e - s, cb, int(res), generator, gdev, pw)
            for k in KEYS:
                out[k][s:e] = d[k].to(dev, out[k].dtype)
        return out

    def _draw(self, n, cb, res, gen, gdev, pw=None) -> dict:
        m, H, W = n * cb, res, res
        f32 = dict(device=gdev, dtype=torch.float32)

        def place(K):
            """Centres (x, y) [m, K] px: uniform, or from the prior of each row's camera."""
            if pw is None:
                return u(0, W, m, K), u(0, H, m, K)
            i = torch.multinomial(pw, n * K, replacement=True, generator=gen)            # [cb, n * K]
            i = i.reshape(cb, n, K).transpose(0, 1).reshape(m, K)                        # rows world-major
            return (i % W).float() + u(0, 1, m, K), (i // W).float() + u(0, 1, m, K)

        def u(lo, hi, *shape):
            return lo + (hi - lo) * torch.rand(*shape, generator=gen, **f32)

        def coin(p, *shape):
            return torch.rand(*shape, generator=gen, **f32) < float(p)

        on = coin(self.p, m)
        yy, xx = torch.meshgrid(torch.arange(H, **f32) + 0.5, torch.arange(W, **f32) + 0.5, indexing="ij")
        # dark area: texture + clutter + spots
        tex = torch.zeros(m, H, W, 3, **f32)
        t_on = on & coin(self.tex_p, m)
        amp = u(*self.tex_amp, m) * t_on
        tint = 1 + u(-self.tex_tint, self.tex_tint, m, 3)
        tex += (amp[:, None, None] * self.noise(m, H, W, gen, gdev))[..., None] * tint[:, None, None, :]
        c_on = on & coin(self.clutter_p, m)
        tex += self._clutter(m, H, W, c_on, xx, yy, gen, u, place)
        s_on = on & coin(self.spot_p, m)
        tex += self._spots(m, H, W, s_on, xx, yy, gen, u, place)
        # floor / walls / mat
        l_on = on & coin(self.low_p, m)
        low = (u(*self.low_amp, m) * l_on)[:, None, None] * self.noise(m, H, W, gen, gdev)
        f_on = on & coin(self.fold_p, m)
        mat = low + self._folds(m, H, W, f_on, xx, yy, gen, u)
        par = torch.zeros(m, NPAR, **f32)
        par[:, 0] = on.float()
        par[:, 1] = (coin(self.flat_p, m) & on).float()
        par[:, 2] = u(*self.ramp_amp, m) * on
        par[:, 3] = u(0.0, 2 * math.pi, m) * on
        par[:, 4] = u(*self.edge_blur_px, m) * on
        par[:, 5] = u(-self.floor_grad, self.floor_grad, m) * on
        par[:, 6] = u(0.0, 2 * math.pi, m) * on
        par[:, 7] = u(*self.floor_radial, m) * on
        par[:, 8] = u(-self.mat_grad, self.mat_grad, m) * on
        par[:, 9] = u(0.0, 2 * math.pi, m) * on
        r = lambda t: t.reshape(n, cb, *t.shape[1:])                                              # noqa: E731
        return dict(bg_tex=r(tex), bg_low=r(low), bg_mat=r(mat), bg_par=r(par))

    def noise(self, m, H, W, gen, gdev) -> torch.Tensor:
        """``train_noise_v1`` value noise [m, H, W], unit std per field (module docstring)."""
        cells = [int(c) for c in self.tex_cells]
        w = torch.rand(m, len(cells), generator=gen, device=gdev)
        w = w / w.norm(dim=1, keepdim=True).clamp_min(1e-6)
        out = torch.zeros(m, H, W, device=gdev)
        for j, c in enumerate(cells):
            if c == 1:
                v = torch.randn(m, H, W, generator=gen, device=gdev)
            else:
                gh, gw = H // c + 2, W // c + 2
                g = torch.randn(m, 1, gh, gw, generator=gen, device=gdev)
                up = F.interpolate(g, size=(gh * c, gw * c), mode="bilinear", align_corners=False)[:, 0]
                oy, ox = (int(v) for v in torch.randint(0, c, (2,), generator=gen, device=gdev))
                v = up[:, oy:oy + H, ox:ox + W]
            v = (v - v.mean((1, 2), keepdim=True)) / v.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            out += w[:, j, None, None] * v
        return out / out.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)

    def _clutter(self, m, H, W, on, xx, yy, gen, u, place) -> torch.Tensor:
        lo, hi = (int(v) for v in self.clutter_n)
        K = hi
        k = torch.randint(lo, hi + 1, (m,), generator=gen, device=xx.device)
        live = (torch.arange(K, device=xx.device)[None] < k[:, None]) & on[:, None]       # [m, K]
        bar = torch.rand(m, K, generator=gen, device=xx.device) < 0.5
        cx, cy = place(K)
        ln, wd = u(20, 160, m, K), u(1.5, 6.0, m, K)
        rw, rh = u(6, 50, m, K), u(6, 50, m, K)
        hw = torch.where(bar, ln, rw) / 2
        hh = torch.where(bar, wd, rh) / 2
        ang = torch.where(bar, u(0, math.pi, m, K), u(-0.2, 0.2, m, K))
        b = u(*self.clutter_bright, m, K)
        col = b[..., None] * (1 + u(-self.clutter_tint, self.clutter_tint, m, K, 3))          # [m, K, 3]
        out = torch.zeros(m, H, W, 3, device=xx.device)
        for j in range(K):
            c, s = torch.cos(ang[:, j])[:, None, None], torch.sin(ang[:, j])[:, None, None]
            dx, dy = xx[None] - cx[:, j, None, None], yy[None] - cy[:, j, None, None]
            qx, qy = (c * dx + s * dy).abs(), (-s * dx + c * dy).abs()
            cov = (hw[:, j, None, None] - qx + 0.5).clamp(0, 1) * (hh[:, j, None, None] - qy + 0.5).clamp(0, 1)
            cov = cov * live[:, j, None, None]
            out = out * (1 - cov[..., None]) + col[:, j, None, None, :] * cov[..., None]
        return out

    def _spots(self, m, H, W, on, xx, yy, gen, u, place) -> torch.Tensor:
        lo, hi = (int(v) for v in self.spot_n)
        K = hi
        dev = xx.device
        k = torch.randint(lo, hi + 1, (m,), generator=gen, device=dev)
        live = (torch.arange(K, device=dev)[None] < k[:, None]) & on[:, None]
        corner = torch.rand(m, K, generator=gen, device=dev) < self.spot_corner_p
        which = torch.randint(0, 4, (m, K), generator=gen, device=dev)
        ox, oy = u(0, 30, m, K), u(0, 30, m, K)
        ccx = torch.where(which % 2 == 0, ox, W - ox)
        ccy = torch.where(which // 2 == 0, oy, H - oy)
        px, py = place(K)
        cx = torch.where(corner, ccx, px)
        cy = torch.where(corner, ccy, py)
        rad = u(*self.spot_px, m, K) / 2
        pal = torch.tensor(SPOT_COLORS, device=dev)
        col = pal[torch.randint(0, len(SPOT_COLORS), (m, K), generator=gen, device=dev)] * \
            u(*self.spot_bright, m, K)[..., None]
        out = torch.zeros(m, H, W, 3, device=dev)
        for j in range(K):
            d = torch.sqrt((xx[None] - cx[:, j, None, None]) ** 2 + (yy[None] - cy[:, j, None, None]) ** 2)
            cov = (rad[:, j, None, None] - d + 0.5).clamp(0, 1) * live[:, j, None, None]
            out = out + col[:, j, None, None, :] * cov[..., None]
        return out

    def _folds(self, m, H, W, on, xx, yy, gen, u) -> torch.Tensor:
        lo, hi = (int(v) for v in self.fold_n)
        K = hi
        dev = xx.device
        k = torch.randint(lo, hi + 1, (m,), generator=gen, device=dev)
        live = (torch.arange(K, device=dev)[None] < k[:, None]) & on[:, None]
        px, py, ang = u(0, W, m, K), u(0, H, m, K), u(0, math.pi, m, K)
        sig, gain = u(*self.fold_px, m, K), u(*self.fold_gain, m, K)
        out = torch.zeros(m, H, W, device=dev)
        for j in range(K):
            d = -torch.sin(ang[:, j, None, None]) * (xx[None] - px[:, j, None, None]) + \
                torch.cos(ang[:, j, None, None]) * (yy[None] - py[:, j, None, None])
            prof = torch.exp(-0.5 * (d / sig[:, j, None, None]) ** 2)
            out = out + (gain[:, j] * live[:, j])[:, None, None] * prof
        return out


# ---- render-time application -------------------------------------------------------------------------------------
def _ramp(angle: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """[m] direction (rad) -> [m, H, W] linear ramp, -1 .. 1 across the image."""
    dev = angle.device
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32) - (H - 1) / 2,
                            torch.arange(W, device=dev, dtype=torch.float32) - (W - 1) / 2, indexing="ij")
    c, s = torch.cos(angle)[:, None, None], torch.sin(angle)[:, None, None]
    norm = (c.abs() * (W - 1) / 2 + s.abs() * (H - 1) / 2).clamp_min(1e-6)
    return (c * xx[None] + s * yy[None]) / norm


def _dilate(mask: torch.Tensor, r: int) -> torch.Tensor:
    """[m, H, W] bool -> dilated by a (2r + 1) square."""
    return F.max_pool2d(mask[:, None].float(), 2 * r + 1, stride=1, padding=r)[:, 0] > 0.5


def _blur(x: torch.Tensor, sigma: torch.Tensor, k: int) -> torch.Tensor:
    """x [m, H, W, ch] float, sigma [m] px -> per-image Gaussian blur (zero padding: normalised convolution)."""
    m, H, W, ch = x.shape
    r = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    s = sigma.to(x).clamp_min(1e-3)[:, None]
    g = torch.exp(-0.5 * (r[None] / s) ** 2)
    g = g / g.sum(-1, keepdim=True)
    y = x.permute(0, 3, 1, 2).reshape(1, m * ch, H, W)
    gx = g.repeat_interleave(ch, 0)[:, None, None, :]
    gy = g.repeat_interleave(ch, 0)[:, None, :, None]
    y = F.conv2d(F.pad(y, (k // 2, k // 2, 0, 0)), gx, groups=m * ch)
    y = F.conv2d(F.pad(y, (0, 0, k // 2, k // 2)), gy, groups=m * ch)
    return y.reshape(m, ch, H, W).permute(0, 2, 3, 1)


def apply_bg(x: torch.Tensor, gid: torch.Tensor, layers: dict, codes: dict, dark_codes, ss: int = 1,
             cos_theta: torch.Tensor | None = None) -> torch.Tensor:
    """One camera: linear colour x [m, Hs, Ws, 3], group code per (sub-)pixel gid [m, Hs, Ws] (-1: none), the
    camera's layers (``bg_tex`` [m, H, W, 3], ``bg_low`` / ``bg_mat`` [m, H, W], ``bg_par`` [m, NPAR]) at the output
    resolution (Hs = H * ss), ``cos_theta`` [Hs, Ws] of the camera's rays (needed for ``floor_radial``) -> the new
    linear colour (module docstring). Rows with ``on`` 0 are returned unchanged (bit for bit); x is not modified in
    place."""
    par = layers["bg_par"].to(x.device, torch.float32)
    on = par[:, 0] > 0.5
    if not bool(on.any()):
        return x

    def up(t):
        t = t.to(x.device, torch.float32)
        return t if ss == 1 else t.repeat_interleave(ss, 1).repeat_interleave(ss, 2)

    m, Hs, Ws, _ = x.shape
    dark = torch.zeros_like(gid, dtype=torch.bool)
    for c in dark_codes:
        dark |= gid == int(c)
    floor, walls, mat = (gid == int(codes[g]) for g in ("floor", "walls", "mat"))
    # dark area: profile, then the texture layer
    dm = dark[..., None].to(x.dtype)
    mean = (x * dm).sum((1, 2)) / dm.sum((1, 2)).clamp_min(1.0)                         # [m, 3]
    ramp = 1 + par[:, 2, None, None] * _ramp(par[:, 3], Hs, Ws)
    flat = (par[:, 1] > 0.5)[:, None, None, None]
    prof = torch.where(flat, mean[:, None, None, :].expand_as(x), x * ramp[..., None])
    y = torch.where(dark[..., None], (prof + up(layers["bg_tex"])).clamp_min(0.0), x)
    # floor gradient + low texture, mat texture + folds
    low, matl = up(layers["bg_low"]), up(layers["bg_mat"])
    grad = 1 + par[:, 5, None, None] * _ramp(par[:, 6], Hs, Ws)
    if bool((par[:, 7] != 0).any()):
        if cos_theta is None:
            raise ValueError("floor_radial needs the camera's cos_theta")
        ct = cos_theta.to(x.device, torch.float32).clamp_min(1e-3)
        grad = grad * ct[None] ** -par[:, 7, None, None]
    mgrad = 1 + par[:, 8, None, None] * _ramp(par[:, 9], Hs, Ws)
    mul = torch.where(floor, grad * (1 + low), torch.where(walls, 1 + low, torch.where(mat, mgrad * (1 + matl), 1.0)))
    y = y * mul.clamp_min(0.0)[..., None]
    # floor / dark-area boundary: normalised blur over the floor and dark pixels only
    sig = par[:, 4] * ss
    if bool((sig > 0.05).any()):
        r = BAND_PX * ss
        fm = floor | dark
        band = ((floor & _dilate(dark, r)) | (dark & _dilate(floor, r))) & (sig > 0.05)[:, None, None]
        k = 2 * int(math.ceil(3 * float(sig.max()))) + 1
        fmf = fm[..., None].to(y.dtype)
        num = _blur(y * fmf, sig, k)
        den = _blur(fmf, sig, k)
        y = torch.where(band[..., None], num / den.clamp_min(1e-6), y)
    return torch.where(on[:, None, None, None], y, x)
