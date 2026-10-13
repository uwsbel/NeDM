"""Batched, differentiable forward kinematics of the G1 legs (torch) from a 67-channel NRD state.

The constants (g1_leg_fk_constants.json next to this file) were exported from the collector's own MuJoCo model
(nedm.g1_footstep_rl.g1_scene.build_model with the collection config; OpenHomie g1.xml at the pinned commit):
for each leg body (hip pitch to ankle roll) its body_pos, body_quat, joint axis and qpos0, and the 4 foot collision
spheres of each foot (radius 5 mm, in the ankle roll link).
MuJoCo rule for a hinge body with the joint anchor at the body origin:
    R = R_parent R(body_quat) R(axis, q - qpos0),  p = p_parent + R_parent body_pos.
Checked against MuJoCo mj_kinematics on 2,000 validation states: largest difference 1.0e-15 m.

The result depends on channels 0-18 only (pelvis position, quaternion, 12 leg angles).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

CONSTANTS = Path(__file__).with_name("g1_leg_fk_constants.json")
FOOT_RADIUS = 0.005


def quat_to_mat(q):
    """[..., 4] wxyz (any norm) -> [..., 3, 3] rotation matrices."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1).view(*q.shape[:-1], 3, 3)


def axis_angle_mat(axis, angle):
    """axis [3] unit, angle [...] -> [..., 3, 3]."""
    half = angle / 2
    q = torch.cat((torch.cos(half)[..., None], torch.sin(half)[..., None] * axis), -1)
    return quat_to_mat(q)


class G1LegFK(torch.nn.Module):
    """forward(state [..., 67]) -> dict(spheres [..., 2, 4, 3] world sphere centres (left, right), ankle [..., 2, 3]
    ankle roll link origins, center [..., 2, 3] mean of the 4 sphere centres, lowest [..., 2] lowest sphere bottom z)."""

    def __init__(self, constants=CONSTANTS, dtype=torch.float64):
        super().__init__()
        c = json.loads(Path(constants).read_text())
        b = c["bodies"]
        t = lambda v: torch.tensor(v, dtype=dtype)
        self.register_buffer("pos", t([x["pos"] for x in b]).view(2, 6, 3))
        self.register_buffer("rot", quat_to_mat(t([x["quat"] for x in b])).view(2, 6, 3, 3))
        self.register_buffer("axis", t([x["axis"] for x in b]).view(2, 6, 3))
        self.register_buffer("qpos0", t([x["qpos0"] for x in b]).view(2, 6))
        # qposadr 7..18 = state channels of the 12 leg angles (qpos = state[0:34])
        self.chan = [[x["qposadr"] for x in b[:6]], [x["qposadr"] for x in b[6:]]]
        self.register_buffer("foot_pos", t([c["feet"]["left"]["pos"], c["feet"]["right"]["pos"]]))   # [2, 4, 3]
        radius = {float(r) for side in ("left", "right") for r in c["feet"][side]["radius"]}
        if radius != {FOOT_RADIUS}:
            raise ValueError(f"foot sphere radii {radius}, expected {FOOT_RADIUS}")
        if any(x["anchor"] != [0.0, 0.0, 0.0] for x in b):
            raise ValueError("a joint anchor is not at its body origin")
        if self.chan != [list(range(7, 13)), list(range(13, 19))]:
            raise ValueError(f"unexpected leg channels {self.chan}")

    def forward(self, state):
        p0 = state[..., 0:3]
        r0 = quat_to_mat(state[..., 3:7])
        spheres, ankle = [], []
        for s in range(2):
            p, r = p0, r0
            for k in range(6):
                p = p + (r @ self.pos[s, k][:, None])[..., 0]
                r = r @ self.rot[s, k] @ axis_angle_mat(self.axis[s, k], state[..., self.chan[s][k]] - self.qpos0[s, k])
            ankle.append(p)
            spheres.append(p[..., None, :] + (r[..., None, :, :] @ self.foot_pos[s][..., None])[..., 0])
        spheres = torch.stack(spheres, -3)
        return dict(spheres=spheres, radius=FOOT_RADIUS, ankle=torch.stack(ankle, -2), center=spheres.mean(-2),
                    lowest=spheres[..., 2].amin(-1) - FOOT_RADIUS)


_CACHE = {}


def foot_points(state):
    """state [..., 67] -> dict(spheres [..., 2, 4, 3] world centres of the 4 foot spheres of each foot (left, right),
    radius 0.005 m, ankle [..., 2, 3], center [..., 2, 3] = mean of the spheres, lowest [..., 2] = lowest sphere
    bottom z). Differentiable. One G1LegFK per (dtype, device) is kept."""
    key = (state.dtype, state.device)
    if key not in _CACHE:
        with torch.inference_mode(False):   # normal buffers even when first called under inference_mode (the env):
            _CACHE[key] = G1LegFK(dtype=state.dtype).to(state.device)   # later calls may need gradients
    return _CACHE[key](state)


# ---- self-contact proxy of the legs and the pelvis ---------------------------------------------------------------
# Capsules (segment a-b + radius r, in each body frame) fitted to the convex hulls of the collision meshes of the
# pelvis and the leg links, plus the exact foot spheres (a = b): g1_leg_proxy.json, exported from the collector's own
# MuJoCo model and fitted there (its fit_quality gives each hull's under-coverage and protrusion in mm). MuJoCo
# collides every robot geom pair except parent-child pairs; the proxy checks left leg vs right leg (all 6 bodies
# each, feet included) and the pelvis vs the leg bodies that are not its children (hip roll to foot). The pelvis / hip roll link pairs, the only
# pelvis pairs that come close in the data, do not use capsules (they were 7-11 mm too close there): the relative pose
# of these two bodies depends on the hip pitch and hip roll angles only, so g1_leg_proxy.json "pelvis_tables" holds
# their exact MuJoCo hull distance on a grid over both joint ranges (bilinear, error <= 0.5 mm). The frames come from
# the chain of G1LegFK (same constants and operations), checked against mj_kinematics.
PROXY = Path(__file__).with_name("g1_leg_proxy.json")
PROXY_BODIES = ("pelvis",) + tuple(f"{s}_{n}_link" for s in ("left", "right")
                                   for n in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))


def segment_distance(p1, q1, p2, q2, eps=1e-12):
    """Smallest distance between the segments p1-q1 and p2-q2 ([..., 3] each, broadcast). A segment with p = q is a
    point. Ericson, Real-Time Collision Detection 5.1.9, branch-free (torch.where), so it runs batched and its
    gradient is finite everywhere (where the segments cross the distance is clamped at 1e-10 m)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = (d1 * d1).sum(-1), (d2 * d2).sum(-1)
    b, c, f = (d1 * d2).sum(-1), (d1 * r).sum(-1), (d2 * r).sum(-1)
    a_ok, e_ok = a > eps, e > eps
    a_s, e_s = torch.where(a_ok, a, torch.ones_like(a)), torch.where(e_ok, e, torch.ones_like(e))
    den = a * e - b * b
    den_ok = den > eps * a_s * e_s
    s = torch.where(den_ok, ((b * f - c * e) / torch.where(den_ok, den, torch.ones_like(den))).clamp(0, 1),
                    torch.zeros_like(a))
    t = (b * s + f) / e_s
    s = torch.where(t < 0, (-c / a_s).clamp(0, 1), torch.where(t > 1, ((b - c) / a_s).clamp(0, 1), s))
    t = t.clamp(0, 1)
    # degenerate segments (points)
    s = torch.where(a_ok, s, torch.zeros_like(s))
    t = torch.where(a_ok, t, (f / e_s).clamp(0, 1))
    s = torch.where(e_ok, s, torch.where(a_ok, (-c / a_s).clamp(0, 1), torch.zeros_like(s)))
    t = torch.where(e_ok, t, torch.zeros_like(t))
    w = r + s[..., None] * d1 - t[..., None] * d2
    return (w * w).sum(-1).clamp_min(1e-20).sqrt()


class G1LegProxy(torch.nn.Module):
    """Link frames of the pelvis and the 12 leg bodies and the capsule proxy of their collision geometry.

    frames(state [..., 67]) -> (pos [..., 13, 3], rot [..., 13, 3, 3]) = MuJoCo xpos / xmat of PROXY_BODIES.
    capsules(state) -> (a [..., N, 3], b [..., N, 3]) world segment ends; radius [N], body [N] (index in PROXY_BODIES).
    distances(state) -> dict(legs [...], pelvis [...]): smallest capsule surface distance (segment distance - radii,
    negative = overlap) between a left-leg and a right-leg capsule, and between the pelvis and a leg body that is not
    the pelvis' child (capsule pairs; with tables=True, the default, the hip roll links use the exact tables instead
    of their capsule pairs). Depends on channels 0-18 only; the distances on channels 7-18 only."""

    def __init__(self, constants=CONSTANTS, proxy=PROXY, dtype=torch.float64, all_pairs=False, tables=True):
        super().__init__()
        self.fk = G1LegFK(constants, dtype)
        p = json.loads(Path(proxy).read_text())
        if tuple(p["bodies"]) != PROXY_BODIES:
            raise ValueError(f"{proxy}: bodies {list(p['bodies'])}, expected {PROXY_BODIES}")
        caps = [(k, c) for k, n in enumerate(PROXY_BODIES) for c in p["bodies"][n]["capsules"]]
        t = lambda v: torch.tensor(v, dtype=dtype)
        self.register_buffer("cap_a", t([c["a"] for _, c in caps]))
        self.register_buffer("cap_b", t([c["b"] for _, c in caps]))
        self.register_buffer("radius", t([c["r"] for _, c in caps]))
        self.register_buffer("body", torch.tensor([k for k, _ in caps], dtype=torch.long))
        body = self.body.tolist()
        left = [i for i, k in enumerate(body) if 1 <= k <= 6]
        right = [i for i, k in enumerate(body) if 7 <= k <= 12]
        pel = [i for i, k in enumerate(body) if k == 0]
        legs_np = [i for i, k in enumerate(body) if k not in (0, 1, 7)]     # not the pelvis' children (hip pitch)
        ii = lambda v: torch.tensor(v, dtype=torch.long)
        pl, pp = torch.cartesian_prod(ii(left), ii(right)), torch.cartesian_prod(ii(pel), ii(legs_np))
        if "pairs" in p and not all_pairs:   # pruned list: a subset of these
            ql, qp = ii(p["pairs"]["legs"]).view(-1, 2), ii(p["pairs"]["pelvis"]).view(-1, 2)
            full = lambda q, ref: all(tuple(x) in {tuple(y) for y in ref.tolist()} for x in q.tolist())
            if not (full(ql, pl) and full(qp, pp)):
                raise ValueError(f"{proxy}: a listed pair is not a left-right or pelvis-leg pair")
            pl, pp = ql, qp
        # exact pelvis / hip roll link tables: they replace the capsule pairs of those two bodies
        tabs = p.get("pelvis_tables") if tables else None
        self.n_tables = 0
        if tabs:
            names = ("left_hip_roll_link", "right_hip_roll_link")
            hr = [PROXY_BODIES.index(n) for n in names]
            pp = pp[torch.tensor([body[int(j)] not in hr for j in pp[:, 1]], dtype=torch.bool)].view(-1, 2)
            for k, n in enumerate(names):
                tb = tabs[n]
                shape = tuple(tb["shape"])
                if tb["channels"] != [self.fk.chan[k][0], self.fk.chan[k][1]] or len(tb["d"]) != shape[0] * shape[1]:
                    raise ValueError(f"{proxy}: pelvis table {n} does not match the hip pitch / roll channels or its shape")
                self.register_buffer(f"tab{k}", t(tb["d"]).view(shape))
                self.register_buffer(f"tab{k}_lo", t(tb["lo"]))
                self.register_buffer(f"tab{k}_hi", t(tb["hi"]))
                self.register_buffer(f"tab{k}_n1", torch.tensor([s - 1 for s in shape], dtype=torch.long))
            self.tab_channels = [tuple(tabs[n]["channels"]) for n in names]
            self.n_tables = len(names)
        self.register_buffer("pair_legs", pl)
        self.register_buffer("pair_pelvis", pp)
        # fast path: pairs split by kind (sphere-sphere, point-segment, segment-segment); the squared segment lengths
        # are constants of the rigid bodies
        sph = ((self.cap_b - self.cap_a).norm(dim=-1) == 0).tolist()
        self.groups = {}
        for name, pr in (("legs", pl), ("pelvis", pp)):
            g = {"ss": [], "ps": [], "cc": []}
            for n, (i, j) in enumerate(pr.tolist()):
                if sph[i] and sph[j]:
                    g["ss"].append((n, i, j))
                elif sph[i] or sph[j]:
                    g["ps"].append((n, i, j) if sph[i] else (n, j, i))      # point first
                else:
                    g["cc"].append((n, i, j))
            for kind, rows in g.items():
                t3 = torch.tensor(rows, dtype=torch.long).view(-1, 3)
                self.register_buffer(f"g_{name}_{kind}", t3)
            self.groups[name] = len(pr)
        self.register_buffer("seg_len2", ((self.cap_b - self.cap_a) ** 2).sum(-1))

    def frames(self, state):
        fk = self.fk
        p0 = state[..., 0:3]
        r0 = quat_to_mat(state[..., 3:7])
        pos, rot = [p0], [r0]
        for s in range(2):
            p, r = p0, r0
            for k in range(6):
                p = p + (r @ fk.pos[s, k][:, None])[..., 0]
                r = r @ fk.rot[s, k] @ axis_angle_mat(fk.axis[s, k], state[..., fk.chan[s][k]] - fk.qpos0[s, k])
                pos.append(p)
                rot.append(r)
        return torch.stack(pos, -2), torch.stack(rot, -3)

    def capsules(self, state):
        pos, rot = self.frames(state)
        pb, rb = pos[..., self.body, :], rot[..., self.body, :, :]
        a = pb + (rb @ self.cap_a[..., None])[..., 0]
        b = pb + (rb @ self.cap_b[..., None])[..., 0]
        return a, b

    def pair_distances(self, state):
        """state [..., 67] -> dict(legs [..., P_legs], pelvis [..., P_pelvis]) capsule surface distance of every
        checked pair (pair_legs, pair_pelvis: capsule indices), and (with tables) pelvis_table [..., 2] the exact
        pelvis / left and right hip roll link distances."""
        a, b = self.capsules(state)
        out = {}
        for name, n_pairs in self.groups.items():
            d = torch.zeros(*a.shape[:-2], n_pairs, dtype=a.dtype, device=a.device)
            for n, v in self._group_distances(a, b, name):
                d = d.index_copy(-1, n, v)
            out[name] = d
        if self.n_tables:
            out["pelvis_table"] = self.table_distances(state)
        return out

    def table_distances(self, state):
        """state [..., 67] -> [..., 2] exact pelvis / hip roll link distance (left, right) by bilinear interpolation of
        the tables in (hip pitch, hip roll); angles outside the joint ranges are clamped to them. Differentiable in
        the two angles (inside the ranges)."""
        out = []
        for k, (c0, c1) in enumerate(self.tab_channels):
            g, lo, hi, n1 = (getattr(self, f"tab{k}{x}") for x in ("", "_lo", "_hi", "_n1"))
            q = torch.stack((state[..., c0], state[..., c1]), -1)
            x = (torch.minimum(torch.maximum(q, lo), hi) - lo) / (hi - lo) * n1
            i = torch.minimum(x.detach().floor().long().clamp(min=0), n1 - 1)
            t = x - i
            flat, n2 = g.reshape(-1), g.shape[1]
            j = i[..., 0] * n2 + i[..., 1]
            tx, ty = t[..., 0], t[..., 1]
            out.append((1 - tx) * (1 - ty) * flat[j] + tx * (1 - ty) * flat[j + n2] + (1 - tx) * ty * flat[j + 1]
                       + tx * ty * flat[j + n2 + 1])
        return torch.stack(out, -1)

    def _group_distances(self, a, b, name):
        """Capsule surface distances of the pairs of one category, by kind: list of (pair positions, [..., n])."""
        r, l2 = self.radius, self.seg_len2
        out = []
        g = getattr(self, f"g_{name}_ss")
        if len(g):
            n, i, j = g.unbind(1)
            w = a[..., i, :] - a[..., j, :]
            out.append((n, (w * w).sum(-1).clamp_min(1e-20).sqrt() - r[i] - r[j]))
        g = getattr(self, f"g_{name}_ps")
        if len(g):
            n, i, j = g.unbind(1)
            p, a2 = a[..., i, :], a[..., j, :]
            d2 = b[..., j, :] - a2
            w = p - a2
            t = ((w * d2).sum(-1) / l2[j]).clamp(0, 1)
            w = w - t[..., None] * d2
            out.append((n, (w * w).sum(-1).clamp_min(1e-20).sqrt() - r[i] - r[j]))
        g = getattr(self, f"g_{name}_cc")
        if len(g):
            n, i, j = g.unbind(1)
            p1, p2 = a[..., i, :], a[..., j, :]
            d1, d2 = b[..., i, :] - p1, b[..., j, :] - p2
            w = p1 - p2
            ea, ee = l2[i], l2[j]
            bb, c, f = (d1 * d2).sum(-1), (d1 * w).sum(-1), (d2 * w).sum(-1)
            den = ea * ee - bb * bb
            ok = den > 1e-12 * ea * ee
            sv = torch.where(ok, (bb * f - c * ee) / torch.where(ok, den, torch.ones_like(den)), torch.zeros_like(den)).clamp(0, 1)
            tv = (bb * sv + f) / ee
            sv = torch.where(tv < 0, (-c / ea).clamp(0, 1), torch.where(tv > 1, ((bb - c) / ea).clamp(0, 1), sv))
            tv = tv.clamp(0, 1)
            w = w + sv[..., None] * d1 - tv[..., None] * d2
            out.append((n, (w * w).sum(-1).clamp_min(1e-20).sqrt() - r[i] - r[j]))
        return out

    def distances(self, state):
        """state [..., 67] -> dict(legs [...], pelvis [...]) smallest surface distance (fast path, the same values as
        the minimum of pair_distances(...) up to rounding; pelvis includes the exact hip roll tables)."""
        a, b = self.capsules(state)
        out = {}
        for name in self.groups:
            parts = [d for _, d in self._group_distances(a, b, name)]
            if name == "pelvis" and self.n_tables:
                parts.append(self.table_distances(state))
            out[name] = torch.cat(parts, -1).amin(-1)
        return out


_PROXY_CACHE = {}


def _proxy(state):
    key = (state.dtype, state.device)
    if key not in _PROXY_CACHE:
        with torch.inference_mode(False):   # as foot_points: normal buffers, so gradients work after an env step
            _PROXY_CACHE[key] = G1LegProxy(dtype=state.dtype).to(state.device)
    return _PROXY_CACHE[key]


def leg_frames(state):
    """state [..., 67] -> (pos [..., 13, 3], rot [..., 13, 3, 3]): world frames (MuJoCo xpos / xmat) of PROXY_BODIES
    (pelvis, left hip pitch .. ankle roll, right hip pitch .. ankle roll). Differentiable."""
    return _proxy(state).frames(state)


def leg_capsules(state):
    """state [..., 67] -> dict(a [..., N, 3], b [..., N, 3] world segment ends, radius [N], body [N] index into
    PROXY_BODIES) of the self-contact proxy (pelvis, leg links, foot spheres). Differentiable."""
    m = _proxy(state)
    a, b = m.capsules(state)
    return dict(a=a, b=b, radius=m.radius, body=m.body)


def _pelvis_frame(q12, like):
    """[..., 12] leg angles -> [..., 67] states with the pelvis at the origin, identity orientation (the proxy
    distances depend on the 12 leg angles only)."""
    x = torch.zeros(*q12.shape[:-1], 67, dtype=like.dtype, device=like.device)
    x[..., 3] = 1.0
    x[..., 7:19] = q12
    return x


def leg_distances(state):
    """state [..., 67] -> dict(legs [...], pelvis [...]): smallest proxy surface distance (m; negative = the capsules
    overlap) between the left and the right leg, and between the pelvis and the leg bodies that are not its children
    (hip roll to foot). Differentiable; reads channels 7-18 only."""
    return _proxy(state).distances(_pelvis_frame(state[..., 7:19], state))


def min_leg_distance(state, pelvis=True):
    """state [..., 67] -> [...] smallest proxy surface distance (m; negative = the capsules overlap) between the left
    and the right leg, and (pelvis=True) between the pelvis and the leg bodies that are not its children. MuJoCo
    collides these bodies' hulls; the proxy's hull error is in g1_leg_proxy.json fit_quality. Differentiable."""
    d = leg_distances(state)
    return torch.minimum(d["legs"], d["pelvis"]) if pelvis else d["legs"]


def sweep_taus(n_between=2, ahead=1.0, n_ahead=2):
    """Interpolation parameters of swept_leg_distances: k / n_between (k = 1..n_between, the last is the new state)
    and 1 + ahead k / n_ahead (k = 1..n_ahead, the look-ahead)."""
    return [k / n_between for k in range(1, n_between + 1)] + [1 + ahead * k / n_ahead for k in range(1, n_ahead + 1)]


def swept_leg_distances(prev, cur, n_between=2, ahead=1.0, n_ahead=2):
    """Swept self-contact check over one 20 ms step and a look-ahead: leg_distances (legs, pelvis), each the smallest
    over the leg angles q(tau) = q_prev + tau (q_cur - q_prev) for tau in sweep_taus: tau = 1 (the new state), the
    n_between - 1 inner points k / n_between (the motion during the step; tau = 0 was checked one step earlier) and
    n_ahead points 1 + ahead k / n_ahead (linear extrapolation of the step's motion by `ahead` steps: contacts happen
    within one step, so the check must see them coming). prev, cur [..., 67] -> dict(legs [...], pelvis [...]).
    Only channels 7-18 are read. Differentiable."""
    q0, q1 = prev[..., 7:19], cur[..., 7:19]
    tau = torch.tensor(sweep_taus(n_between, ahead, n_ahead), dtype=cur.dtype, device=cur.device)
    q = q0[..., None, :] + tau[:, None] * (q1 - q0)[..., None, :]          # [..., P, 12]
    d = leg_distances(_pelvis_frame(q, cur))
    return {k: v.amin(-1) for k, v in d.items()}


def swept_leg_distance(prev, cur, n_between=2, ahead=1.0, n_ahead=2, pelvis=True):
    """swept_leg_distances as one number: legs, or (pelvis=True) the smaller of legs and pelvis. [..., 67] x 2 -> [...]."""
    d = swept_leg_distances(prev, cur, n_between, ahead, n_ahead)
    return torch.minimum(d["legs"], d["pelvis"]) if pelvis else d["legs"]
