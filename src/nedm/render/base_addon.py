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

    ANGLES = 720                    # outline lookup resolution (0.5 deg)

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
        if not 0.0 <= float(self.p) <= 1.0:
            raise ValueError(f"base_addon.p {self.p}: 0..1")
        for name, n, slots in (("n_blocks", self.n_blocks, self.block_slots),
                               ("n_cables", self.n_cables, self.cable_slots)):
            lo, hi = (int(x) for x in n)
            if not 0 <= lo <= hi <= int(slots):
                raise ValueError(f"base_addon.{name} {n}: needs 0 <= lo <= hi <= the slot count {slots}")
        if len(self.block_size) != 3:
            raise ValueError("base_addon.block_size: three (lo, hi) ranges (length, width, height)")
        for name, rngs in (("block_size", self.block_size), ("cable_len", (self.cable_len,)),
                           ("cable_width", (self.cable_width,))):
            for lo, hi in rngs:
                if not 0 < float(lo) <= float(hi):
                    raise ValueError(f"base_addon.{name} {getattr(self, name)}: 0 < lo <= hi")

    @property
    def on(self) -> bool:
        return float(self.p) > 0 and (int(self.n_blocks[1]) > 0 or int(self.n_cables[1]) > 0)

    def slots(self) -> dict:
        """Slot sizes (deterministic: Halton points over the ranges) for ``RealScene.addon_slots``."""
        blocks = []
        for k in range(int(self.block_slots)):
            blocks.append([round(float(lo + _halton(k + 1, b) * (hi - lo)), 6)
                           for (lo, hi), b in zip(self.block_size, (2, 3, 5))])
        cables = []
        for k in range(int(self.cable_slots)):
            cables.append([round(float(lo + _halton(k + 1, b) * (hi - lo)), 6)
                           for (lo, hi), b in zip((self.cable_len, self.cable_width), (2, 3))])
        return {"blocks": blocks, "cables": cables}

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
        if [list(map(float, b)) for b in slots.get("blocks", ())] != want["blocks"] or \
                [list(map(float, c)) for c in slots.get("cables", ())] != want["cables"]:
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
            i = ((phi + math.pi) / (2 * math.pi) * A).floor().long().clamp(0, A - 1)
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
        pos = torch.cat([xy, (zb + dims[:, 2] / 2)[..., None]], -1)
        q = qmul(tilt_quat(tax, tang), yaw_quat(yaw))
        self._place(env, cols, show, pos, q, grey, kb, first_env, HIDDEN_Z, xyzw)
        # cables
        S = len(kc)
        show = pick(self.n_cables, S)
        phi, rin, zc = u(-math.pi, math.pi, S), u(*self.cable_in, S), u(*self.cable_z, S)
        dyaw, pitch = u(-self.cable_yaw, self.cable_yaw, S), u(*self.cable_pitch, S)
        grey = u(*self.cable_bright, S, 1) * u(1 - self.cable_tint, 1 + self.cable_tint, S, 3)
        L = torch.tensor([cb[0] for cb in want["cables"]], dtype=torch.float64, device=dev)
        r, th = outline(phi)
        start = c + (rin * r)[..., None] * torch.stack([torch.cos(phi), torch.sin(phi)], -1)
        head = th + dyaw
        d = torch.stack([torch.cos(head) * torch.cos(pitch), torch.sin(head) * torch.cos(pitch), -torch.sin(pitch)], -1)
        pos = torch.cat([start, zc[..., None]], -1) + d * (L / 2)[..., None]
        zero = torch.zeros_like(pitch)
        qy = torch.stack([torch.cos(pitch / 2), zero, torch.sin(pitch / 2), zero], -1)     # x axis -> down by pitch
        q = qmul(yaw_quat(head), qy)
        self._place(env, cols, show, pos, q, grey, kc, first_env, HIDDEN_Z, xyzw)

    @staticmethod
    def _place(env, cols, show, pos, q_wxyz, grey, ks, first_env, hidden_z, xyzw):
        hid = torch.zeros_like(env[:, 0])
        hid[:, 2], hid[:, 6] = hidden_z, 1.0
        for s, k in enumerate(ks):
            pose = torch.cat([pos[:, s], xyzw(q_wxyz[:, s])], -1)
            env[:, k] = torch.where(show[:, s, None], pose, hid)
            cols[:, first_env + k] = torch.where(show[:, s, None], grey[:, s].to(cols.dtype).clamp(0, 1),
                                                 cols[:, first_env + k])
