"""Gripper-look randomisation of the real-setup renderer (v12 visual fix, 2026-10-26; OFF by default).

Why (~/so101_r2s_claude/v12/diag/SYNTHESIS.md): the release of the camera students after the first hold comes from the
REAL wrist image. The carrier is the bottom band of the wrist view: the camera bracket plate (real: black with two
bright screw heads and a bright top edge; old sim: a plain grey box), the moving-jaw face with the 4 horn screws to its
right and the fixed jaw to its left. Small high-contrast marks low in the wrist view loosen the plan in general. The
training randomisation changed colours, lights and camera poses, but never put local texture on the robot's own parts.
This randomisation does, per world (and episode):

- the albedo of the camera bracket plate (``plate_parts``, grey ``plate_albedo`` x a per-channel tint) and of the
  servo bracket (``bracket_parts``: the moving-jaw face around the horn screws; the moving jaw mesh in a scene without
  the fitted overlays);
- the fitted screw heads of the scene (``screws``: groups of scene parts named ``<screw>_*``, e.g. the plate screws
  ``plate_screw0`` / ``plate_screw1`` and the jaw horn screws ``jaw_screw0..3`` of real_scene_v2_wb.json): each one
  shown with probability ``p_screw``, moved in its own plane by up to +- ``screw_jitter`` m, ring (``*_head_lo``,
  ``*_rim``) and centre (``*_head_hi``, ``*_head``, ``*_recess``) albedo ``screw_ring`` / ``screw_centre`` (each part
  keeps its fitted colour ratio within its role);
- ``n_marks`` (inclusive range) extra marks (discs or small rectangles, size = disc diameter or long rectangle side in
  ``mark_size`` m; light ``mark_light`` with probability ``p_light``, else dark ``mark_dark``) at random places and
  turns on the surfaces ``mark_surfaces`` (the bracket plate, the servo bracket, the fixed jaw and the moving jaw: the
  parts in the bottom half of the wrist view; geometry :data:`V12_SURFACES`); a mark lies inside its surface (its
  footprint ``MARK_INSET`` m inside every edge), ``mark_standoff`` m in front of it (towards the wrist camera),
  ``mark_thickness`` thick, and outside every exclusion zone of its body (``zones``: part-local boxes
  ``{"body", "lo": (3,), "hi": (3,)}`` m in that MJCF body's frame; a mark whose footprint circle reaches into a zone is
  not placed: the held-out variant keeps the real cues out of training); ``mark_tries`` candidate places per mark, the
  first one that fits is used (none fits: the mark is not shown; the number of random draws never depends on the
  outcome);
- optionally a bright strip along the plate's top edge (``p_edge``; ``edge_width`` m wide, ``edge_inset`` (lo, hi) m
  inside the edge, albedo ``edge_bright``).

With probability 1 - ``p`` a world keeps the scene's look (and RealDR's own colour draws). Every part of it is geometry
(prisms on the arm bodies), so the wrist camera draws it under the camera-pose, light and tone randomisation, occluded
like the real parts. Slot sizes are fixed per slot (Newton cannot scale a body per world): ``mark_slots`` per surface
with sizes over ``mark_size`` (both ends; discs and rectangles alternate, rectangle aspect over ``rect_aspect``).

Renderer side (``nedm.render.so101``): ``RealScene.look_slots`` (set by ``viral.load_real_setup`` from
``RealDR.gripper_look``; not in scene files) = the movable scene parts (the screws) and the slot parts (group
``look``, on moving bodies, hidden until drawn); ``SO101Scene.look_names`` / ``look_index``; the per-world local poses
``part_pose`` [n, K, 7] (x y z qx qy qz qw in the part's body frame; z < ``HIDDEN_Z`` / 2 = hidden) of
:meth:`GripperLookDR.draw`, applied by ``SO101Scene.transforms(part_pose=)``. ``RealDR.draw`` draws it after every
other draw (also after the base add-ons) on its own generator (``look_generator``, ``viral.DRViews``: seed + 919), so
every other draw of a seed is the same with it on or off. Off (``gripper_look`` {} or ``p`` 0): no slot bodies, no
draws, every array bit for bit as before.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
import torch

# The v12 geometry (~/so101_r2s_claude/v12/vis/build_geo.py from the scene-fit parts parts_band_v1.json and the
# calibrated wrist camera, calib_v1.json): per surface its MJCF body, the plane of the part's face towards the wrist
# camera (origin, unit normal towards the camera, in-plane axis e1; e2 = normal x e1) and the face outline in (e1, e2)
# coordinates (m, convex, counter-clockwise): the part of the face that is in the bottom half of the 224 px wrist view
# (rows 112-221, columns 2-221) at >= 80 % of the 30 logged real holds (the bracket plate's lower part is below the
# image: 343 of 604 mm2 kept; the others keep 93-98 %).
V12_SURFACES = (
    {"name": "bracket_plate", "part": "camera_bracket_plate", "body": "camera_mount",
     "origin": [0.00219, 0.027548, -0.020348], "normal": [0.180287, 0.94388, -0.276743],
     "e1": [-0.00419, -0.280614, -0.959812],
     "poly2": [[-0.002839, 0.009713], [-0.000513, -0.010484], [0.014997, -0.010484],
               [0.014997, -0.0062], [0.011895, 0.012161], [0.008018, 0.012161],
               [-0.000513, 0.010937], [-0.002839, 0.010325]]},
    {"name": "servo_bracket", "part": "jaw_face", "body": "moving_jaw_so101_v1",
     "origin": [0.000134, -0.003093, -0.00556], "normal": [-0.001043, -0.023762, -0.999717],
     "e1": [-0.578523, -0.815421, 0.019985],
     "poly2": [[-0.009423, -0.002871], [-0.008727, -0.003891], [-0.007336, -0.005421],
               [-0.00664, -0.005931], [-0.005945, -0.00644], [-0.003162, -0.00797],
               [-0.00038, -0.00899], [0.005881, -0.011029], [0.006576, -0.011029],
               [0.007272, -0.010519], [0.008663, -0.00899], [0.011446, -0.005421],
               [0.014924, -0.000832], [0.014924, 0.000188], [0.014228, 0.001207],
               [0.013533, 0.001717], [0.012141, 0.002737], [0.009359, 0.004776],
               [0.005185, 0.007326], [0.001707, 0.007835], [-0.005249, 0.007835],
               [-0.005945, 0.007326], [-0.00664, 0.006306]]},
    {"name": "fixed_jaw_upper", "part": "fixed_jaw_upper", "body": "camera_mount",
     "origin": [-0.017683, 0.014936, -0.060005], "normal": [-0.257742, 0.914571, -0.311656],
     "e1": [0.090371, 0.343959, 0.934626],
     "poly2": [[-0.026538, -0.003735], [-0.016543, -0.00648], [0.004874, -0.01197],
               [0.010585, -0.013342], [0.016296, -0.014715], [0.017724, -0.014715],
               [0.019152, -0.01197], [0.022008, -0.004422], [0.024863, 0.003813],
               [0.026291, 0.00793], [0.026291, 0.010675], [0.02058, 0.010675],
               [0.012013, 0.009988], [-0.013687, 0.00793], [-0.017971, 0.007244],
               [-0.019399, 0.005871], [-0.023682, 0.001068], [-0.026538, -0.002363]]},
    {"name": "fixed_jaw_lower", "part": "fixed_jaw_lower", "body": "camera_mount",
     "origin": [-0.021351, 0.024718, -0.028267], "normal": [-0.257734, 0.914568, -0.311671],
     "e1": [0.553373, 0.404145, 0.728316],
     "poly2": [[-0.015935, -0.001391], [-0.013084, -0.004617], [-0.012134, -0.005262],
               [-0.008332, -0.006552], [-0.00358, -0.007842], [0.008776, -0.011068],
               [0.009726, -0.010423], [0.01828, 0.00377], [0.01828, 0.004415], [0.017329, 0.00506],
               [0.011627, 0.00764], [0.001172, 0.012156], [-0.000729, 0.012801],
               [-0.001679, 0.012801], [-0.00453, 0.010866], [-0.007382, 0.008285],
               [-0.013084, 0.003125], [-0.015935, 0.000544]]},
    {"name": "moving_jaw", "part": "jaw_upper", "body": "moving_jaw_so101_v1",
     "origin": [-0.0023, -0.03361, 0.001944], "normal": [0.024193, -0.406976, -0.913118],
     "e1": [0.025745, 0.913336, -0.406391],
     "poly2": [[-0.019626, -0.003859], [-0.016423, -0.004222], [-0.012153, -0.004586],
               [0.001725, -0.005675], [0.011333, -0.006401], [0.01667, -0.006764],
               [0.019873, -0.006764], [0.02094, 0.007035], [-0.014288, 0.004493],
               [-0.018558, 0.00413], [-0.019626, 0.003767]]},
)
# the bracket plate's top edge in the wrist view (image rows ~190 of 224): end points and the in-plane direction into
# the plate (camera_mount frame, m)
V12_EDGE = {"body": "camera_mount", "a": [-0.010424, 0.026461, -0.032272], "b": [0.012545, 0.021276, -0.034992],
            "inward": [0.168847, 0.247478, 0.954068], "surface": "bracket_plate"}
# the two fitted plate screw heads (centre, outer radius; camera_mount frame, m): 224-px image (98.9, 199.4) and
# (117.1, 199.9)
V12_PLATE_SCREWS = ({"centre": [-0.003453, 0.027063, -0.02839], "radius": 0.002421},
                    {"centre": [0.005378, 0.025226, -0.028899], "radius": 0.002485})

RING_SUFFIXES = ("_head_lo", "_rim")
CENTRE_SUFFIXES = ("_head_hi", "_head", "_recess")


def _halton(i: int, base: int) -> float:
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def held_out_zones(margin: float = 0.003, edge_width: float = 0.0015) -> list:
    """Exclusion zones (camera_mount boxes, m) of the held-out variant: ``margin`` around the two fitted plate screws
    (centre +- (radius + margin)) and around the plate's top edge strip (``edge_width`` wide, the box + margin)."""
    out = []
    for s in V12_PLATE_SCREWS:
        c, r = np.asarray(s["centre"]), float(s["radius"]) + margin
        out.append({"body": "camera_mount", "lo": [round(float(v), 6) for v in c - r],
                    "hi": [round(float(v), 6) for v in c + r]})
    a, b, d = (np.asarray(V12_EDGE[k], np.float64) for k in ("a", "b", "inward"))
    strip = np.stack([a, b, b + edge_width * d, a + edge_width * d])
    out.append({"body": "camera_mount", "lo": [round(float(v), 6) for v in strip.min(0) - margin],
                "hi": [round(float(v), 6) for v in strip.max(0) + margin]})
    return out


@dataclass
class GripperLookDR:
    p: float = 1.0
    plate_parts: tuple = ("camera_bracket_plate",)
    plate_albedo: tuple | None = (0.02, 0.2)
    plate_tint: float = 0.06
    bracket_parts: tuple = ()
    bracket_albedo: tuple | None = (0.25, 0.8)
    bracket_tint: float = 0.06
    screws: tuple = ()
    p_screw: float = 0.7
    screw_jitter: float = 0.001
    screw_ring: tuple = (0.15, 0.7)
    screw_centre: tuple = (0.03, 0.2)
    screw_tint: float = 0.08
    n_marks: tuple = (0, 4)
    mark_surfaces: tuple = ("bracket_plate", "servo_bracket", "fixed_jaw_upper", "fixed_jaw_lower", "moving_jaw")
    mark_slots: int = 3
    mark_size: tuple = (0.0015, 0.004)
    rect_aspect: tuple = (0.4, 1.0)
    p_light: float = 0.5
    mark_light: tuple = (0.35, 0.9)
    mark_dark: tuple = (0.01, 0.08)
    mark_tint: float = 0.08
    mark_standoff: float = 0.0003
    mark_thickness: float = 0.0002
    mark_tries: int = 16
    zones: tuple = ()
    p_edge: float = 0.0
    edge_width: float = 0.001
    edge_inset: tuple = (0.0, 0.0005)
    edge_bright: tuple = (0.3, 0.9)
    edge_tint: float = 0.05

    DISC_SIDES = 12
    MARK_INSET = 0.0003             # m: a mark's footprint stays this far inside its surface's outline
    EDGE_END = 0.0005               # m: the edge strip stops this short of the plate's corners
    SURFACES = {s["name"]: s for s in V12_SURFACES}

    @classmethod
    def from_dict(cls, d: dict | None) -> "GripperLookDR":
        d = {k: v for k, v in dict(d or {}).items() if k not in ("note", "notes")}
        names = {f.name for f in fields(cls)}
        bad = sorted(set(d) - names)
        if bad:
            raise ValueError(f"unknown gripper_look keys {bad}")
        fix = lambda v: tuple(fix(x) for x in v) if isinstance(v, (list, tuple)) else v  # noqa: E731
        out = cls(**{k: (tuple(dict(z) for z in v) if k == "zones" else fix(v)) for k, v in d.items()})
        out.check()
        return out

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def check(self) -> None:
        def rng(name, v, lo_min=None, hi_max=None, pos=False):
            try:
                lo, hi = (float(x) for x in v)
            except (TypeError, ValueError):
                raise ValueError(f"gripper_look.{name} {v}: a (lo, hi) pair") from None
            if not lo <= hi or (pos and not lo > 0) or (lo_min is not None and lo < lo_min) or \
                    (hi_max is not None and hi > hi_max):
                raise ValueError(f"gripper_look.{name} {v}: needs lo <= hi"
                                 + (", lo > 0" if pos else "") + (f", lo >= {lo_min}" if lo_min is not None else "")
                                 + (f", hi <= {hi_max}" if hi_max is not None else ""))

        for name in ("p", "p_screw", "p_light", "p_edge"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"gripper_look.{name} {getattr(self, name)}: 0..1")
        for name in ("plate_albedo", "bracket_albedo"):
            if getattr(self, name) is not None:
                rng(name, getattr(self, name), 0.0, 1.0)
        for name in ("screw_ring", "screw_centre", "mark_light", "mark_dark", "edge_bright"):
            rng(name, getattr(self, name), 0.0, 1.0)
        rng("mark_size", self.mark_size, pos=True)
        rng("rect_aspect", self.rect_aspect, pos=True, hi_max=1.0)
        rng("edge_inset", self.edge_inset, lo_min=0.0)
        for name in ("plate_tint", "bracket_tint", "screw_tint", "mark_tint", "edge_tint", "screw_jitter"):
            if not float(getattr(self, name)) >= 0:
                raise ValueError(f"gripper_look.{name} {getattr(self, name)}: >= 0")
        for name in ("mark_thickness", "edge_width"):
            if not float(getattr(self, name)) > 0:
                raise ValueError(f"gripper_look.{name} {getattr(self, name)}: > 0")
        if not float(self.mark_standoff) >= float(self.mark_thickness):
            raise ValueError(f"gripper_look.mark_standoff {self.mark_standoff}: >= mark_thickness "
                             f"{self.mark_thickness} (a mark lies in front of its surface)")
        bad = [s for s in self.mark_surfaces if s not in self.SURFACES]
        if bad or len(set(self.mark_surfaces)) != len(self.mark_surfaces):
            raise ValueError(f"gripper_look.mark_surfaces {list(self.mark_surfaces)}: distinct names of "
                             f"{sorted(self.SURFACES)}")
        if int(self.mark_slots) < 0 or int(self.mark_tries) < 1:
            raise ValueError("gripper_look: mark_slots >= 0 and mark_tries >= 1")
        lo, hi = (int(x) for x in self.n_marks)
        if not 0 <= lo <= hi <= self.n_slots:
            raise ValueError(f"gripper_look.n_marks {list(self.n_marks)}: needs 0 <= lo <= hi <= the mark slots "
                             f"{self.n_slots} (mark_slots x surfaces)")
        if len(set(self.screws)) != len(self.screws):
            raise ValueError(f"gripper_look.screws {list(self.screws)}: distinct names")
        for z in self.zones:
            if set(z) - {"body", "lo", "hi", "what"} or not {"body", "lo", "hi"} <= set(z) or len(z["lo"]) != 3 or \
                    len(z["hi"]) != 3 or any(float(a) > float(b) for a, b in zip(z["lo"], z["hi"])):
                raise ValueError(f"gripper_look.zones {z}: {{'body', 'lo': (x, y, z), 'hi': (x, y, z)}} with lo <= hi")
        for name in ("plate_parts", "bracket_parts"):
            if isinstance(getattr(self, name), str):
                raise ValueError(f"gripper_look.{name}: a list of scene body names")

    # -- what is on --
    @property
    def n_slots(self) -> int:
        return int(self.mark_slots) * len(self.mark_surfaces)

    @property
    def edge_on(self) -> bool:
        return float(self.p_edge) > 0

    @property
    def on(self) -> bool:
        return float(self.p) > 0 and bool(
            (self.plate_parts and self.plate_albedo is not None)
            or (self.bracket_parts and self.bracket_albedo is not None)
            or self.screws or int(self.n_marks[1]) > 0 or self.edge_on)

    # -- the slot bodies --
    def mark_specs(self) -> list[dict]:
        """Per mark slot (surface order, then slot order): surface, shape ("disc" / "rect"), size (m: diameter or
        long side), aspect (rect short / long side), footprint radius (m)."""
        S = int(self.mark_slots)
        lo, hi = (float(x) for x in self.mark_size)
        alo, ahi = (float(x) for x in self.rect_aspect)
        out, k_rect = [], 0
        for si, surf in enumerate(self.mark_surfaces):
            for j in range(S):
                t = 0.5 if S == 1 else ((j + si) % S) / (S - 1)        # sizes over both ends, shifted per surface
                size = round(lo + t * (hi - lo), 6)
                shape = "disc" if (j + si) % 2 == 0 else "rect"
                if shape == "rect":
                    k_rect += 1
                    asp = round(alo + _halton(k_rect, 2) * (ahi - alo), 6)
                    rad = 0.5 * size * math.hypot(1.0, asp)
                else:
                    asp, rad = 1.0, 0.5 * size
                out.append({"surface": surf, "shape": shape, "size": size, "aspect": asp, "radius": rad})
        return out

    def _slot_polygon(self, spec) -> list:
        if spec["shape"] == "disc":
            r = spec["size"] / 2
            n = self.DISC_SIDES
            return [[round(r * math.cos(2 * math.pi * k / n), 7), round(r * math.sin(2 * math.pi * k / n), 7), 0.0]
                    for k in range(n)]
        a, b = spec["size"] / 2, spec["size"] * spec["aspect"] / 2
        return [[-a, -b, 0.0], [a, -b, 0.0], [a, b, 0.0], [-a, b, 0.0]]

    def edge_length(self) -> float:
        a, b = (np.asarray(V12_EDGE[k], np.float64) for k in ("a", "b"))
        return float(np.linalg.norm(b - a)) - 2 * self.EDGE_END

    def screw_parts(self, part_names) -> dict:
        """screw -> the scene part names ``<screw>_*`` (in ``part_names`` order); every screw needs at least one."""
        out = {}
        for s in self.screws:
            got = [n for n in part_names if n.startswith(f"{s}_")]
            if not got:
                raise ValueError(f"gripper_look.screws: no scene part named {s}_* (scene parts: {list(part_names)}); "
                                 "a scene without fitted screws needs screws []")
            out[s] = got
        return out

    def slots(self, real) -> dict:
        """``RealScene.look_slots`` for the scene ``real`` (a RealScene): ``{"movable": [scene part names of the
        screws], "slots": [part dicts of the marks and the edge strip (group "look"), each in its own frame: a mark's
        face in z = 0 towards +z, extruded to -mark_thickness]}``."""
        names = [p.get("name", "part") for p in real.parts]
        movable = [n for parts in self.screw_parts(names).values() for n in parts]
        slots = []
        for k, spec in enumerate(self.mark_specs()):
            body = self.SURFACES[spec["surface"]]["body"]
            slots.append({"name": f"look_mark{k}", "body": body, "group": "look",
                          "polygon": self._slot_polygon(spec), "offset": [0.0, 0.0, -float(self.mark_thickness)]})
        if self.edge_on:
            a, b = self.edge_length() / 2, float(self.edge_width) / 2
            slots.append({"name": "look_edge", "body": V12_EDGE["body"], "group": "look",
                          "polygon": [[-a, -b, 0.0], [a, -b, 0.0], [a, b, 0.0], [-a, b, 0.0]],
                          "offset": [0.0, 0.0, -float(self.mark_thickness)]})
        return {"movable": movable, "slots": slots}

    # -- the per-world draw --
    def _prepare(self, scene, dev) -> dict:
        """Index and geometry tensors of the draw for ``scene`` on ``dev`` (checked against the scene's look parts
        once; cached on the scene per config and device)."""
        import json as _json
        key = (_json.dumps(self.to_dict(), sort_keys=True, default=list), str(dev))
        cache = scene.__dict__.setdefault("_gripper_look_prep", {})
        if key in cache:
            return cache[key]
        want = self.slots(scene.real)
        have = getattr(scene.real, "look_slots", None) or {}
        sig = lambda sl: [(s["name"], s["body"], [list(map(float, v)) for v in s["polygon"]],  # noqa: E731
                           list(map(float, s["offset"]))) for s in sl]
        if list(have.get("movable", [])) != want["movable"] or sig(have.get("slots", [])) != sig(want["slots"]) or \
                list(getattr(scene, "look_names", [])) != want["movable"] + [s["name"] for s in want["slots"]]:
            raise ValueError("RealDR.gripper_look: the scene's look_slots differ from this randomisation's slots "
                             "(build the scene with RealScene(look_slots=GripperLookDR.slots(real)); "
                             "viral.load_real_setup does it)")
        f64 = dict(dtype=torch.float64, device=dev)
        names, look = list(scene.body_names), list(scene.look_names)

        def body_idx(part_names, what):
            idx = []
            for nm in part_names:
                k = [i for i, b in enumerate(names) if b == nm]
                if len(k) != 1:
                    raise ValueError(f"gripper_look.{what}: scene body {nm!r} found {len(k)} times (bodies: {names})")
                idx.append(k[0])
            return torch.tensor(idx, dtype=torch.long, device=dev)

        P = {"look_pose0": scene.look_pose0.to(**f64)}
        P["plate_idx"] = body_idx(self.plate_parts, "plate_parts") if self.plate_parts else None
        P["bracket_idx"] = body_idx(self.bracket_parts, "bracket_parts") if self.bracket_parts else None
        # screws: per part its screw, look slot, body, role (0 ring, 1 centre, -1 none) and colour factor
        parts_by_name = {p.get("name", "part"): p for p in scene.real.parts}
        groups = self.screw_parts(list(parts_by_name))
        rows = []
        ax = []
        for s, (screw, pnames) in enumerate(groups.items()):
            _, _, e1, e2 = polygon_plane(parts_by_name[pnames[0]]["polygon"])
            ax.append(np.stack([e1, e2]))
            ref = {}
            for role, suf in ((0, RING_SUFFIXES), (1, CENTRE_SUFFIXES)):
                first = [nm for nm in pnames if nm.endswith(suf)]
                if first:
                    ref[role] = float(np.mean(scene.scene.bodies[names.index(first[0])].color))
            for nm in pnames:
                role = 0 if nm.endswith(RING_SUFFIXES) else (1 if nm.endswith(CENTRE_SUFFIXES) else -1)
                if role >= 0 and not ref.get(role, 0.0) > 0:
                    role = -1
                base = np.asarray(scene.scene.bodies[names.index(nm)].color, np.float64)
                rows.append((s, look.index(nm), names.index(nm), role, base / ref[role] if role >= 0 else base))
        if rows:
            P["screw_s"] = torch.tensor([r[0] for r in rows], dtype=torch.long, device=dev)
            P["screw_kl"] = torch.tensor([r[1] for r in rows], dtype=torch.long, device=dev)
            P["screw_ib"] = torch.tensor([r[2] for r in rows], dtype=torch.long, device=dev)
            P["screw_role"] = torch.tensor([r[3] for r in rows], dtype=torch.long, device=dev)
            P["screw_fac"] = torch.tensor(np.stack([r[4] for r in rows]), **f64)
            P["screw_ax"] = torch.tensor(np.stack(ax), **f64)                          # [S, 2, 3]
        P["n_screws"] = len(groups)
        # marks: per slot its plane, outline edges (padded with the first edge), inset, zones (padded far away)
        specs = self.mark_specs()
        M = len(specs)
        P["n_marks"] = M
        if M:
            kmax = max(len(self.SURFACES[s["surface"]]["poly2"]) for s in specs)
            zs = [[z for z in self.zones if z["body"] == self.SURFACES[s["surface"]]["body"]] for s in specs]
            zmax = max(1, max(len(z) for z in zs))
            A, E = np.zeros((M, kmax, 2)), np.zeros((M, kmax, 2))
            Z = np.full((M, zmax, 2, 3), 1e3)                                           # a box 1 km away: no effect
            geo = np.zeros((M, 4, 3))
            box = np.zeros((M, 2, 2))
            for m, spec in enumerate(specs):
                surf = self.SURFACES[spec["surface"]]
                p = np.asarray(surf["poly2"], np.float64)
                if 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]) < 0:
                    p = p[::-1]
                e = np.roll(p, -1, 0) - p
                A[m], E[m] = np.concatenate([p, np.repeat(p[:1], kmax - len(p), 0)]), \
                    np.concatenate([e, np.repeat(e[:1], kmax - len(p), 0)])
                for j, z in enumerate(zs[m]):
                    Z[m, j] = [z["lo"], z["hi"]]
                n_ = np.asarray(surf["normal"], np.float64)
                e1 = np.asarray(surf["e1"], np.float64)
                geo[m] = [surf["origin"], n_, e1, np.cross(n_, e1)]
                box[m] = [p.min(0), p.max(0)]
            P["A"], P["E"] = torch.tensor(A, **f64), torch.tensor(E, **f64)
            P["Z"] = torch.tensor(Z, **f64)
            P["geo"] = torch.tensor(geo, **f64)
            P["box"] = torch.tensor(box, **f64)
            P["radius"] = torch.tensor([s["radius"] for s in specs], **f64)
            P["mark_kl"] = torch.tensor([look.index(f"look_mark{m}") for m in range(M)], dtype=torch.long, device=dev)
            P["mark_ib"] = torch.tensor([names.index(f"look_mark{m}") for m in range(M)], dtype=torch.long, device=dev)
        if self.edge_on:
            a, b, din = (np.asarray(V12_EDGE[k], np.float64) for k in ("a", "b", "inward"))
            nrm = np.asarray(self.SURFACES[V12_EDGE["surface"]]["normal"], np.float64)
            ex = (b - a) - np.dot(b - a, nrm) * nrm
            ex /= np.linalg.norm(ex)
            q = _mat_quat(torch.tensor(np.stack([ex, np.cross(nrm, ex), nrm], -1)[None], **f64))[0]
            P["edge_q"] = torch.cat([q[1:], q[:1]])                                     # x y z w
            P["edge_c"] = torch.tensor((a + b) / 2 + float(self.mark_standoff) * nrm, **f64)
            P["edge_in"] = torch.tensor(din, **f64)
            P["edge_kl"], P["edge_ib"] = look.index("look_edge"), names.index("look_edge")
        cache[key] = P
        return P

    def draw(self, scene, n: int, u, dev, out: dict) -> None:
        """In place on ``out["colors"]`` [n, bodies, 3]; sets ``out["part_pose"]`` [n, K, 7] (float64, the local
        poses of ``scene.look_names``). ``scene``: an SO101Scene built with ``real.look_slots`` = :meth:`slots`;
        ``u(lo, hi, *shape)``: uniform draws [n, *shape] (float64 on ``dev``). The draws, in this order: world on
        (``p``); plate grey + tint; bracket grey + tint; per screw shown, jitter, ring grey + tint, centre grey + tint;
        mark count, slot ranks, candidate places [M, tries, 2], turns, light or dark, light grey + tint, dark grey +
        tint; edge shown, inset, grey + tint (only the parts that exist: no screws / marks / edge, no draws)."""
        from nedm.render.so101 import HIDDEN_Z
        P = self._prepare(scene, dev)
        cols = out["colors"]
        pose = P["look_pose0"].expand(n, *P["look_pose0"].shape).clone()
        on = u(0.0, 1.0) < float(self.p)                                          # [n]

        def grey(rng, tint, *shape):
            return u(float(rng[0]), float(rng[1]), *shape, 1) * u(1 - float(tint), 1 + float(tint), *shape, 3)

        # 1. albedo of the bracket plate and of the servo bracket (one grey per world for each)
        for idx, rng, tint in ((P["plate_idx"], self.plate_albedo, self.plate_tint),
                               (P["bracket_idx"], self.bracket_albedo, self.bracket_tint)):
            if idx is None or rng is None:
                continue
            g = grey(rng, tint).to(cols.dtype).clamp(0, 1)                         # [n, 3]
            cols[:, idx] = torch.where(on[:, None, None], g[:, None], cols[:, idx])
        # 2. the fitted screws: shown with p_screw, moved in their plane, ring / centre albedo
        S = P["n_screws"]
        if S:
            show = u(0.0, 1.0, S) < float(self.p_screw)
            jit = u(-float(self.screw_jitter), float(self.screw_jitter), S, 2)
            g_ring = grey(self.screw_ring, self.screw_tint, S)
            g_ctr = grey(self.screw_centre, self.screw_tint, S)
            s_, kl, ib, role = P["screw_s"], P["screw_kl"], P["screw_ib"], P["screw_role"]
            d = (jit[..., None] * P["screw_ax"]).sum(-2)[:, s_]                        # [n, parts, 3]
            vis = on[:, None] & show[:, s_]
            hid = on[:, None] & ~show[:, s_]
            pp = pose[:, kl]
            pp[..., :3] = torch.where(vis[..., None], pp[..., :3] + d, pp[..., :3])
            pp[..., 2] = torch.where(hid, torch.full_like(pp[..., 2], HIDDEN_Z), pp[..., 2])
            pose[:, kl] = pp
            g = torch.where((role == 0)[None, :, None], g_ring[:, s_], g_ctr[:, s_]) * P["screw_fac"]
            cols[:, ib] = torch.where((vis & (role >= 0))[..., None], g.to(cols.dtype).clamp(0, 1), cols[:, ib])
        # 3. extra marks on the surfaces (a random subset of the slots, the first candidate place that fits)
        M = P["n_marks"]
        if M:
            nlo, nhi = (int(x) for x in self.n_marks)
            kk = u(float(nlo), float(nhi) + 1.0).floor().clamp(max=float(nhi))
            rank = u(0.0, 1.0, M).argsort(-1).argsort(-1).to(torch.float64)
            pick = (rank < kk[:, None]) & on[:, None]                              # [n, M]
            T = int(self.mark_tries)
            cand = u(0.0, 1.0, M, T, 2)
            turn = u(-math.pi, math.pi, M)
            light = u(0.0, 1.0, M) < float(self.p_light)
            g_light = grey(self.mark_light, self.mark_tint, M)
            g_dark = grey(self.mark_dark, self.mark_tint, M)
            g_mark = torch.where(light[..., None], g_light, g_dark)
            o, nrm, e1, e2 = P["geo"].unbind(1)                                        # [M, 3] each
            box, rad = P["box"], P["radius"]
            xy = box[:, 0, None] + cand * (box[:, 1, None] - box[:, 0, None])         # [n, M, T, 2]
            A, E = P["A"][:, None], P["E"][:, None]                                   # [M, 1, K, 2]
            dist = (E[..., 0] * (xy[..., None, 1] - A[..., 1]) - E[..., 1] * (xy[..., None, 0] - A[..., 0])) \
                / E.norm(dim=-1)                                                       # [n, M, T, K], inside > 0
            ok = (dist >= (rad + self.MARK_INSET)[:, None, None]).all(-1)            # [n, M, T]
            p3 = o[:, None] + xy[..., :1] * e1[:, None] + xy[..., 1:] * e2[:, None]  # [n, M, T, 3]
            Z = P["Z"]                                                                 # [M, Zn, 2, 3]
            gap = torch.maximum(torch.maximum(Z[:, None, :, 0] - p3[..., None, :], p3[..., None, :] - Z[:, None, :, 1]),
                                torch.zeros((), dtype=p3.dtype, device=p3.device)).norm(dim=-1)   # [n, M, T, Zn]
            ok = ok & (gap >= rad[:, None, None]).all(-1)
            first = ok.to(torch.int8).argmax(-1)                                       # [n, M]
            fits = ok.any(-1)
            pt = p3.gather(2, first[..., None, None].expand(n, M, 1, 3))[:, :, 0]      # [n, M, 3]
            a1 = torch.cos(turn)[..., None] * e1 + torch.sin(turn)[..., None] * e2      # the mark's x axis
            nn_ = nrm.expand_as(a1)
            q = _mat_quat(torch.stack([a1, torch.linalg.cross(nn_, a1), nn_], -1))    # columns x, y, z = normal
            shown = pick & fits
            new = torch.cat([pt + float(self.mark_standoff) * nrm, q[..., 1:], q[..., :1]], -1)
            pose[:, P["mark_kl"]] = torch.where(shown[..., None], new, pose[:, P["mark_kl"]])
            ib = P["mark_ib"]
            cols[:, ib] = torch.where(shown[..., None], g_mark.to(cols.dtype).clamp(0, 1), cols[:, ib])
        # 4. the bright strip along the plate's top edge
        if self.edge_on:
            show = (u(0.0, 1.0) < float(self.p_edge)) & on
            inset = u(float(self.edge_inset[0]), float(self.edge_inset[1]))
            g = grey(self.edge_bright, self.edge_tint)
            ctr = P["edge_c"] + (float(self.edge_width) / 2 + inset)[:, None] * P["edge_in"]
            new = torch.cat([ctr, P["edge_q"].expand(n, 4)], -1)
            kl, ib = P["edge_kl"], P["edge_ib"]
            pose[:, kl] = torch.where(show[:, None], new, pose[:, kl])
            cols[:, ib] = torch.where(show[:, None], g.to(cols.dtype).clamp(0, 1), cols[:, ib])
        out["part_pose"] = pose


def _inside(xy, poly, inset):
    """xy [..., 2] inside the convex polygon (k, 2) with every edge at least ``inset`` away -> bool [...]."""
    p = np.asarray(poly, np.float64)
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    if area < 0:
        p = p[::-1]
    a = torch.tensor(p, dtype=xy.dtype, device=xy.device)
    b = torch.roll(a, -1, 0)
    e = b - a
    ln = e.norm(dim=-1)
    d = (e[:, 0] * (xy[..., None, 1] - a[:, 1]) - e[:, 1] * (xy[..., None, 0] - a[:, 0])) / ln     # [..., k], left > 0
    return (d >= float(inset)).all(-1)


def _mat_quat(R):
    """Rotation matrices [..., 3, 3] (columns = the frame's axes) -> unit quaternions [..., 4] w x y z (Shepperd: the
    candidate of the largest diagonal term, stable for every turn)."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    t = m00 + m11 + m22
    cands = torch.stack([torch.stack([1 + t, m21 - m12, m02 - m20, m10 - m01], -1),
                         torch.stack([m21 - m12, 1 + m00 - m11 - m22, m01 + m10, m02 + m20], -1),
                         torch.stack([m02 - m20, m01 + m10, 1 - m00 + m11 - m22, m12 + m21], -1),
                         torch.stack([m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22], -1)], -2)
    k = torch.stack([1 + t, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1).argmax(-1)
    q = cands.gather(-2, k[..., None, None].expand(*k.shape, 1, 4)).squeeze(-2)
    return q / q.norm(dim=-1, keepdim=True)


def polygon_plane(polygon):
    """A planar polygon (k, 3) -> (centroid (3,), unit normal (3,), in-plane unit axes e1, e2 (3,)) (numpy; SVD of
    the k x 3 points with full_matrices=False)."""
    p = np.asarray(polygon, np.float64)
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    n = vt[2] / np.linalg.norm(vt[2])
    e1 = vt[0] - np.dot(vt[0], n) * n
    e1 /= np.linalg.norm(e1)
    return c, n, e1, np.cross(n, e1)
