"""Base add-on randomisation of the real-setup renderer (scene v4 round, 2026-10-23; OFF by default).

Why (~/so101_studentlab_claude/diag/lateral/VERDICT.md): the student reads the plate position relative to the arm base
(the only world-fixed landmark that the camera-pose and background randomisation leave), so an extra part at the real
base (the table clamp: about 8 mm of sideways plate error) moves its estimate. Scene v4 draws the measured clamp; this
randomisation adds random blocks and cables at the base outline in a share of the worlds, so the student learns to read
the base body and not the exact outline of what is fixed to it. The base itself never moves.

Geometry, not image layers: the add-ons are real bodies of the scene (``RealScene.addon_slots``: hidden slot bodies,
group ``addon``, the last env bodies), so they are drawn by both cameras, under the camera-pose randomisation, occluded
by the arm and occluding it exactly like the real parts. Slot sizes are fixed per slot (Newton cannot scale a body per
world): ``block_slots`` boxes and ``cable_slots`` thin bars with sizes spread over the ranges by a fixed
low-discrepancy sequence; per world a random subset is shown, each with a random pose, yaw and grey level.

Per world (``RealDR.draw``, drawn AFTER every other draw, so the other draws of a seed are the same with it on or off):
with probability ``p`` the world has add-ons: ``n_blocks`` (inclusive range) blocks and ``n_cables`` cables (random
slots). A block touches the base footprint's convex hull (``SO101Scene.base_outline``) at a random angle, its centre
``block_gap`` (m; < 0 = into the base) beyond the outline plus its half extent along the outline normal, bottom at
``block_z``, random yaw, a tilt up to ``block_tilt`` rad about a random horizontal axis; grey albedo ``block_bright``
(linear; printed plastic is 0.3 to 0.8) times a per-channel tint 1 +- ``block_tint``. A cable starts above the base
(``cable_in`` x the outline radius from the footprint centre, height ``cable_z``), runs outwards (outline normal +-
``cable_yaw`` rad) and down (``cable_pitch`` rad), dark (``cable_bright``). Hidden slots stay at ``HIDDEN_Z``.

Review round 1 (2026-10-24; every option OFF / the first build's behaviour by default, so a config without them draws
the same numbers and places the same slots):

- ``slot_spread``: how the fixed slot sizes cover the ranges. ``"halton"`` (default, the first build): Halton points
  k = 1..S of bases 2, 3, 5; with few slots they cover only the middle of each range (S = 4: 0.125-0.75 of the range in
  the first size, 0.11-0.67 in the second, 0.2-0.8 in the third; block_size's 15 and 70 mm ends are never drawn).
  ``"grid"``: a Latin-hypercube design with BOTH ends: per size dimension the S values lo + j / (S - 1) (hi - lo),
  j = 0..S-1, assigned to the slots in the rank order of that dimension's Halton sequence (so the dimensions do not
  move together); S = 1: the middle.
- Flat cables (``n_flat`` (inclusive range per world with add-ons), ``flat_slots``; default none): a cable LYING on
  the table next to the base (it changes from session to session; in the robot gate-5 frames of 2026-10-22/23 the
  real cable hangs over the base and the clamp instead, see ``cable_phi``). A straight bar of length ``flat_len``
  (slot sizes), width ``cable_width``, centre height = the table top + width / 2 + 0.5 mm; it starts at the base
  outline (outline angle ``flat_phi`` (lo, hi) rad, world frame from the footprint centre, + ``flat_gap`` m along the
  outline normal) and runs outward (the outline normal +- ``flat_yaw`` rad, < 90 deg, so it never cuts the base
  footprint), optionally limited to the world heading range ``flat_heading`` (lo, hi) rad (None: any; else the
  heading nearest to the drawn one that is in the range and within ``MAX_DEV`` = 1.5 rad of the outline normal).
  Long flat cables can reach the workspace (cube and goal annulus r 0.15-0.27 m, +-75 deg about the pan axis): limit
  ``flat_phi`` / ``flat_heading`` to the rear and the sides. It may run past the table edge: the part beyond the edge
  floats at table height over the dark area (lit, so it looks like the background clutter bars of ``RealDR.bg``).
  With probability ``flat_drop`` it has a sloped segment (slot length ``drop_len``, the same width and grey): from
  the flat cable's start up and back over the base to the height ``drop_z`` above the table (pitch
  asin(height / length), clamped to 0.05..0.97), i.e. the cable coming down from the base top. Colours as the other
  cables (``cable_bright``, ``cable_tint``). Slot names ``addon_flat{k}`` / ``addon_drop{k}`` (after the blocks and
  the cables). Drawn after the blocks and the other cables.
- ``cable_phi`` (lo, hi) rad (default (-pi, pi), the first build) and ``cable_heading`` (lo, hi) rad (default None):
  the outline angle of the sloped cables' start and an optional limit of their world heading (as ``flat_heading``).
  The real servo cable hangs from the arm over the base's rear and the clamp and down past the table edge (gate-5
  frames, ~/so101_r2s_claude/v8/scene4/fix): longer ``cable_len`` and a higher ``cable_z`` (the cable comes from
  above the base top) cover it; ``cable_heading`` keeps the long ones out of the workspace.
- ``block_reach`` (m, 0 = off): a block whose xy extent (half diagonal + half height x sin ``block_tilt``) would reach
  farther than ``block_reach`` from ``reach_xy`` (the pan axis) is pulled towards it until it does not (no extra
  random numbers): keeps the blocks clear of the plate and the carried cube (cube and goal annulus r >= 0.15 m about
  the pan axis).
- ``RealDR.draw(..., addon_generator=)``: the add-on draws on their own generator (``viral.DRViews``: seed + 917), so
  every other draw of the online loop (each per-world reset) stays the same with the add-ons on or off; without it
  they use RealDR's generator after its other draws (then only the first ``RealDR.draw`` call of a seed is paired).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
import torch


def _halton(i: int, base: int) -> float:
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


@dataclass
class BaseAddonDR:
    p: float = 0.5
    n_blocks: tuple = (1, 2)
    block_slots: int = 4
    block_size: tuple = ((0.015, 0.07), (0.010, 0.035), (0.006, 0.025))
    block_gap: tuple = (-0.008, 0.002)
    block_z: tuple = (-0.015, 0.035)
    block_bright: tuple = (0.3, 0.8)
    block_tint: float = 0.08
    block_tilt: float = 0.2
    n_cables: tuple = (0, 2)
    cable_slots: int = 2
    cable_len: tuple = (0.04, 0.10)
    cable_width: tuple = (0.004, 0.010)
    cable_in: tuple = (0.2, 1.0)
    cable_z: tuple = (0.01, 0.065)
    cable_yaw: float = 1.0
    cable_pitch: tuple = (0.4, 1.3)
    cable_bright: tuple = (0.02, 0.12)
    cable_tint: float = 0.1
    # review round 1 (module docstring): off / the first build's behaviour by default
    slot_spread: str = "halton"
    block_reach: float = 0.0
    reach_xy: tuple = (0.0388, 0.0)
    n_flat: tuple = (0, 0)
    flat_slots: int = 0
    flat_len: tuple = (0.08, 0.25)
    flat_gap: tuple = (-0.004, 0.004)
    flat_phi: tuple = (-math.pi, math.pi)
    flat_yaw: float = 1.3
    flat_heading: tuple | None = None
    cable_phi: tuple = (-math.pi, math.pi)
    cable_heading: tuple | None = None
    flat_drop: float = 1.0
    drop_len: tuple = (0.04, 0.08)
    drop_z: tuple = (0.02, 0.07)

    ANGLES = 720                    # outline lookup resolution (0.5 deg)
    SPREADS = ("halton", "grid")
    FLAT_LIFT = 0.0005              # m, a flat cable's bottom above the table top
    MAX_DEV = 1.5                   # rad, a limited cable heading stays within this of the outline normal

    @classmethod
    def from_dict(cls, d: dict | None) -> "BaseAddonDR":
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        bad = sorted(set(d) - names)
        if bad:
            raise ValueError(f"unknown base_addon keys {bad}")
        fix = lambda v: tuple(fix(x) for x in v) if isinstance(v, (list, tuple)) else v  # noqa: E731
        out = cls(**{k: fix(v) for k, v in d.items()})
        out.check()
        return out

    def check(self) -> None:
        def rng(name, v, lo_min=None, hi_max=None, pos=False):
            try:
                lo, hi = (float(x) for x in v)
            except (TypeError, ValueError):
                raise ValueError(f"base_addon.{name} {v}: a (lo, hi) pair") from None
            if not lo <= hi or (pos and not lo > 0) or (lo_min is not None and lo < lo_min) or \
                    (hi_max is not None and hi > hi_max):
                lim = " ".join(x for x in ("0 <" if pos else "", f">= {lo_min}" if lo_min is not None else "",
                                            f"<= {hi_max}" if hi_max is not None else "") if x)
                raise ValueError(f"base_addon.{name} {v}: needs lo <= hi {lim}".rstrip())

        if not 0.0 <= float(self.p) <= 1.0:
            raise ValueError(f"base_addon.p {self.p}: 0..1")
        if not 0.0 <= float(self.flat_drop) <= 1.0:
            raise ValueError(f"base_addon.flat_drop {self.flat_drop}: 0..1")
        if self.slot_spread not in self.SPREADS:
            raise ValueError(f"base_addon.slot_spread {self.slot_spread!r}: one of {self.SPREADS}")
        for name, n, slots in (("n_blocks", self.n_blocks, self.block_slots),
                               ("n_cables", self.n_cables, self.cable_slots),
                               ("n_flat", self.n_flat, self.flat_slots)):
            if int(slots) < 0:
                raise ValueError(f"base_addon: the slot count for {name} {slots} < 0")
            lo, hi = (int(x) for x in n)
            if not 0 <= lo <= hi <= int(slots):
                raise ValueError(f"base_addon.{name} {n}: needs 0 <= lo <= hi <= the slot count {slots}")
        if len(self.block_size) != 3:
            raise ValueError("base_addon.block_size: three (lo, hi) ranges (length, width, height)")
        for name, rngs in (("block_size", self.block_size), ("cable_len", (self.cable_len,)),
                           ("cable_width", (self.cable_width,)), ("flat_len", (self.flat_len,)),
                           ("drop_len", (self.drop_len,))):
            for lo, hi in rngs:
                if not 0 < float(lo) <= float(hi):
                    raise ValueError(f"base_addon.{name} {getattr(self, name)}: 0 < lo <= hi")
        for name in ("block_gap", "block_z", "flat_gap"):
            rng(name, getattr(self, name))
        for name in ("block_bright", "cable_bright"):
            rng(name, getattr(self, name), lo_min=0.0)
        rng("cable_in", self.cable_in, lo_min=0.0)
        rng("cable_z", self.cable_z)
        rng("cable_pitch", self.cable_pitch, lo_min=0.0, hi_max=math.pi / 2)
        rng("drop_z", self.drop_z, pos=True)
        for name in ("flat_phi", "cable_phi", "flat_heading", "cable_heading"):
            v = getattr(self, name)
            if v is None and name.endswith("_heading"):
                continue
            rng(name, v, lo_min=-2 * math.pi, hi_max=2 * math.pi)
            if float(v[1]) - float(v[0]) > 2 * math.pi + 1e-9:
                raise ValueError(f"base_addon.{name} {v}: at most 2 pi wide")
        for name in ("block_tint", "cable_tint", "block_tilt", "cable_yaw", "flat_yaw", "block_reach"):
            if not float(getattr(self, name)) >= 0:
                raise ValueError(f"base_addon.{name} {getattr(self, name)}: >= 0")
        if float(self.flat_yaw) >= math.pi / 2:
            raise ValueError(f"base_addon.flat_yaw {self.flat_yaw}: < pi / 2 (a flat cable runs outward)")
        if len(tuple(self.reach_xy)) != 2:
            raise ValueError(f"base_addon.reach_xy {self.reach_xy}: (x, y)")

    @property
    def on(self) -> bool:
        return float(self.p) > 0 and (int(self.n_blocks[1]) > 0 or int(self.n_cables[1]) > 0 or self.flat_on)

    @property
    def flat_on(self) -> bool:
        """Flat cables (with their sloped segments) in the scene's slots."""
        return int(self.flat_slots) > 0 and int(self.n_flat[1]) > 0

    def _spread(self, S: int, ranges) -> list:
        """S points (lists) over the box of the (lo, hi) ``ranges`` (at most 3), ``slot_spread`` (module docstring)."""
        bases = (2, 3, 5)
        if self.slot_spread == "halton":
            return [[round(float(lo + _halton(k + 1, b) * (hi - lo)), 6) for (lo, hi), b in zip(ranges, bases)]
                    for k in range(S)]
        out = [[0.0] * len(ranges) for _ in range(S)]
        for d, ((lo, hi), b) in enumerate(zip(ranges, bases)):
            rank = np.argsort(np.argsort([_halton(k + 1, b) for k in range(S)], kind="stable"), kind="stable")
            for k in range(S):
                t = 0.5 if S == 1 else float(rank[k]) / (S - 1)
                out[k][d] = round(float(lo + t * (hi - lo)), 6)
        return out

    def slots(self) -> dict:
        """Slot sizes (deterministic, ``slot_spread``) for ``RealScene.addon_slots``: ``blocks`` [(lx, ly, lz)],
        ``cables`` [(length, width)] and, with flat cables, ``flats`` [(length, width)] and ``drops`` [(length,
        width)] (drop k has flat k's width)."""
        out = {"blocks": self._spread(int(self.block_slots), self.block_size),
               "cables": self._spread(int(self.cable_slots), (self.cable_len, self.cable_width))}
        if int(self.flat_slots) > 0:
            fl = self._spread(int(self.flat_slots), (self.flat_len, self.cable_width, self.drop_len))
            out["flats"] = [[x[0], x[1]] for x in fl]
            out["drops"] = [[x[2], x[1]] for x in fl]
        return out

    # -- outline lookup --
    @classmethod
    def outline_table(cls, hull: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Hull (k, 2) ccw -> (centre (2,), radius r [A], outward normal angle [A]) on A angles from the centre
        (angle a_i = -pi + 2 pi i / A)."""
        h = np.asarray(hull, np.float64)
        c = h.mean(0)
        a = -math.pi + 2 * math.pi * np.arange(cls.ANGLES) / cls.ANGLES
        d = np.stack([np.cos(a), np.sin(a)], -1)                                   # [A, 2]
        p0, p1 = h, np.roll(h, -1, 0)                                              # edges [k, 2]
        e = p1 - p0
        nrm = np.stack([e[:, 1], -e[:, 0]], -1)                                    # outward for a ccw hull
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        # ray c + t d hits edge p0 + s e: solve [d, -e] [t, s] = p0 - c
        det = d[:, None, 0] * -e[None, :, 1] - d[:, None, 1] * -e[None, :, 0]      # [A, k]
        w = (p0 - c)[None]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[..., 0] * -e[None, :, 1] - w[..., 1] * -e[None, :, 0]) / det
            s = (d[:, None, 0] * w[..., 1] - d[:, None, 1] * w[..., 0]) / det
        ok = np.isfinite(t) & (t > 0) & (s >= -1e-9) & (s <= 1 + 1e-9)
        t = np.where(ok, t, np.inf)
        j = t.argmin(1)
        r = t[np.arange(len(a)), j]
        if not np.isfinite(r).all():
            raise ValueError("base outline: a ray from the centre misses the hull")
        return c, r, np.arctan2(nrm[j, 1], nrm[j, 0])

    # -- the per-world draw --
    def draw(self, scene, n: int, u, dev, out: dict) -> None:
        """In place on ``out["colors"]`` [n, bodies, 3] and ``out["env_pose"]`` [n, n_env, 7] (RealDR.draw's dict):
        place the slot bodies of ``scene`` (an SO101Scene with ``real.addon_slots``). ``u(lo, hi, *shape)``: RealDR's
        uniform draw [n, *shape] (float64 on ``dev``)."""
        from nedm.render.so101 import HIDDEN_Z, qmul, xyzw, yaw_quat
        slots = getattr(scene.real, "addon_slots", None) or {}
        want = self.slots()
        if any([list(map(float, b)) for b in slots.get(kind, ())] != want.get(kind, [])
               for kind in ("blocks", "cables", "flats", "drops")):
            raise ValueError("RealDR.base_addon: the scene's addon_slots differ from this randomisation's slots "
                             "(build the scene with RealScene(addon_slots=RealDR.addon_slots()); "
                             "viral.load_real_setup does it)")
        names = list(scene.env_names)
        nb_total = out["colors"].shape[1]
        first_env = nb_total - len(names)
        kb = [names.index(f"addon_block{k}") for k in range(len(want["blocks"]))]
        kc = [names.index(f"addon_cable{k}") for k in range(len(want["cables"]))]
        hull, ztop = scene.base_outline()
        c, r_tab, n_tab = self.outline_table(hull)
        c = torch.tensor(c, dtype=torch.float64, device=dev)
        r_tab = torch.tensor(r_tab, dtype=torch.float64, device=dev)
        n_tab = torch.tensor(n_tab, dtype=torch.float64, device=dev)
        A = len(r_tab)

        def outline(phi):
            # phi in any range (cable_phi / flat_phi may cross pi); for phi in [-pi, pi) the same index as before
            i = (torch.remainder(phi + math.pi, 2 * math.pi) / (2 * math.pi) * A).floor().long().clamp(0, A - 1)
            return r_tab[i], n_tab[i]

        def pick(n_range, S):
            """[n, S] bool: which slots are shown (a random subset of size U{lo..hi} in the worlds with add-ons)."""
            k = u(float(n_range[0]), float(n_range[1]) + 1.0).floor().clamp(max=float(n_range[1]))
            rank = u(0.0, 1.0, S).argsort(-1).argsort(-1).to(torch.float64)
            return (rank < k[:, None]) & on[:, None]

        def tilt_quat(axis_ang, ang):
            ax = torch.stack([torch.cos(axis_ang), torch.sin(axis_ang), torch.zeros_like(ang)], -1)
            return torch.cat([torch.cos(ang / 2)[..., None], torch.sin(ang / 2)[..., None] * ax], -1)

        env, cols = out["env_pose"], out["colors"]
        on = u(0.0, 1.0) < float(self.p)
        # blocks
        S = len(kb)
        show = pick(self.n_blocks, S)
        phi, yaw = u(-math.pi, math.pi, S), u(-math.pi, math.pi, S)
        gap = u(*self.block_gap, S)
        zb = u(*self.block_z, S)
        tax, tang = u(-math.pi, math.pi, S), u(0.0, float(self.block_tilt), S)
        grey = u(*self.block_bright, S, 1) * u(1 - self.block_tint, 1 + self.block_tint, S, 3)
        dims = torch.tensor(want["blocks"], dtype=torch.float64, device=dev).reshape(-1, 3)
        r, th = outline(phi)
        hn = dims[:, 0] / 2 * torch.cos(yaw - th).abs() + dims[:, 1] / 2 * torch.sin(yaw - th).abs()
        dirv = torch.stack([torch.cos(phi), torch.sin(phi)], -1)
        nv = torch.stack([torch.cos(th), torch.sin(th)], -1)
        xy = c + r[..., None] * dirv + (hn + gap)[..., None] * nv
        if float(self.block_reach) > 0:         # pulled towards the pan axis until the block stays within the reach
            ctr = torch.tensor([float(x) for x in self.reach_xy], dtype=torch.float64, device=dev)
            rel = xy - ctr
            dist = rel.norm(dim=-1)
            ext = (dims[:, 0] ** 2 + dims[:, 1] ** 2).sqrt() / 2 + dims[:, 2] / 2 * math.sin(float(self.block_tilt))
            keep = (float(self.block_reach) - ext).clamp(min=0.0)
            xy = torch.where((dist > keep)[..., None], ctr + rel * (keep / dist.clamp(min=1e-12))[..., None], xy)
        pos = torch.cat([xy, (zb + dims[:, 2] / 2)[..., None]], -1)
        q = qmul(tilt_quat(tax, tang), yaw_quat(yaw))
        self._place(env, cols, show, pos, q, grey, kb, first_env, HIDDEN_Z, xyzw)
        # cables
        S = len(kc)
        show = pick(self.n_cables, S)
        phi, rin, zc = u(*self.cable_phi, S), u(*self.cable_in, S), u(*self.cable_z, S)
        dyaw, pitch = u(-self.cable_yaw, self.cable_yaw, S), u(*self.cable_pitch, S)
        grey = u(*self.cable_bright, S, 1) * u(1 - self.cable_tint, 1 + self.cable_tint, S, 3)
        L = torch.tensor([cb[0] for cb in want["cables"]], dtype=torch.float64, device=dev)
        r, th = outline(phi)
        start = c + (rin * r)[..., None] * torch.stack([torch.cos(phi), torch.sin(phi)], -1)
        head = self._heading(th, dyaw, self.cable_heading)
        d = torch.stack([torch.cos(head) * torch.cos(pitch), torch.sin(head) * torch.cos(pitch), -torch.sin(pitch)], -1)
        pos = torch.cat([start, zc[..., None]], -1) + d * (L / 2)[..., None]
        zero = torch.zeros_like(pitch)
        qy = torch.stack([torch.cos(pitch / 2), zero, torch.sin(pitch / 2), zero], -1)     # x axis -> down by pitch
        q = qmul(yaw_quat(head), qy)
        self._place(env, cols, show, pos, q, grey, kc, first_env, HIDDEN_Z, xyzw)
        if self.flat_on:
            self._draw_flat(scene, want, names, u, dev, env, cols, pick, outline, c, first_env)

    def _draw_flat(self, scene, want, names, u, dev, env, cols, pick, outline, c, first_env) -> None:
        """Flat cables on the table + their sloped segments (module docstring); after the blocks and the cables."""
        from nedm.render.so101 import HIDDEN_Z, qmul, xyzw, yaw_quat
        kf = [names.index(f"addon_flat{k}") for k in range(len(want["flats"]))]
        kd = [names.index(f"addon_drop{k}") for k in range(len(want["drops"]))]
        S = len(kf)
        show = pick(self.n_flat, S)
        phi, gap = u(*self.flat_phi, S), u(*self.flat_gap, S)
        dyaw = u(-self.flat_yaw, self.flat_yaw, S)
        hz = u(*self.drop_z, S)
        drop = u(0.0, 1.0, S) < float(self.flat_drop)
        grey = u(*self.cable_bright, S, 1) * u(1 - self.cable_tint, 1 + self.cable_tint, S, 3)
        L = torch.tensor([f[0] for f in want["flats"]], dtype=torch.float64, device=dev)
        W = torch.tensor([f[1] for f in want["flats"]], dtype=torch.float64, device=dev)
        Ld = torch.tensor([d[0] for d in want["drops"]], dtype=torch.float64, device=dev)
        r, th = outline(phi)
        head = self._heading(th, dyaw, self.flat_heading)
        dirv = torch.stack([torch.cos(phi), torch.sin(phi)], -1)
        nv = torch.stack([torch.cos(th), torch.sin(th)], -1)
        hv = torch.stack([torch.cos(head), torch.sin(head)], -1)
        start = c + r[..., None] * dirv + gap[..., None] * nv
        z0 = -float(scene.real.mat_thickness) + W / 2 + self.FLAT_LIFT                 # the white table top + lift
        pos = torch.cat([start + (L / 2)[..., None] * hv, z0.expand_as(phi)[..., None]], -1)
        self._place(env, cols, show, pos, yaw_quat(head), grey, kf, first_env, HIDDEN_Z, xyzw)
        # the sloped segment: from the flat cable's start up and back over the base to the height hz
        pitch = torch.asin((hz / Ld).clamp(0.05, 0.97))
        zero = torch.zeros_like(pitch)
        qy = torch.stack([torch.cos(pitch / 2), zero, torch.sin(pitch / 2), zero], -1)     # x axis -> down by pitch
        ctr = start - (Ld / 2 * torch.cos(pitch))[..., None] * hv
        pos = torch.cat([ctr, (z0 + Ld / 2 * torch.sin(pitch))[..., None]], -1)
        self._place(env, cols, show & drop, pos, qmul(yaw_quat(head), qy), grey, kd, first_env, HIDDEN_Z, xyzw)

    def _heading(self, th, dyaw, rng):
        """Heading (rad) = the outline normal ``th`` + ``dyaw``; with a world range ``rng`` (lo, hi): the heading
        nearest to it that lies in the range AND within ``MAX_DEV`` of the normal (outward); if no such heading exists
        at this outline point, ``th + clamp(dyaw, +-MAX_DEV)``. None: ``th + dyaw`` (the first build)."""
        if rng is None:
            return th + dyaw
        lo, hi = (float(x) for x in rng)
        M = float(self.MAX_DEV)
        rlo = torch.remainder(lo - th + math.pi, 2 * math.pi) - math.pi        # the range relative to the normal
        rhi = rlo + (hi - lo)
        best = dyaw.clamp(-M, M)
        bestd = torch.full_like(dyaw, math.inf)
        for sh in (0.0, -2 * math.pi):                                         # the arc and its copy one turn lower
            a, b = (rlo + sh).clamp(min=-M), (rhi + sh).clamp(max=M)
            d = torch.maximum(torch.minimum(dyaw, b), a)
            dist = (d - dyaw).abs()
            better = (a <= b) & (dist < bestd)
            best, bestd = torch.where(better, d, best), torch.where(better, dist, bestd)
        return th + best

    @staticmethod
    def _place(env, cols, show, pos, q_wxyz, grey, ks, first_env, hidden_z, xyzw):
        hid = torch.zeros_like(env[:, 0])
        hid[:, 2], hid[:, 6] = hidden_z, 1.0
        for s, k in enumerate(ks):
            pose = torch.cat([pos[:, s], xyzw(q_wxyz[:, s])], -1)
            env[:, k] = torch.where(show[:, s, None], pose, hid)
            cols[:, first_env + k] = torch.where(show[:, s, None], grey[:, s].to(cols.dtype).clamp(0, 1),
                                                 cols[:, first_env + k])
