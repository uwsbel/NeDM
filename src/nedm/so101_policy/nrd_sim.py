"""GraspNrdSim: B worlds of the SO101 grasp contact NRD on one device, stepped with fixed shapes (spec 2.2).

One step (20 ms) of ``GraspNrdSim.step(joint_targets [B, 6])``:

1. Dense NRD step of every member on its fixed block of worlds: core change, collision logits of the 3 pairs (switch =
   logit >= 0), contact network on ALL (world, pair) entries, each change multiplied by its switch (``torch.where``,
   so a non-finite output of a switched-off pair cannot leak) and written to BOTH bodies of the pair (the pair's first
   body and its moving partner). Pose integration and ``finish`` (quaternion re-normalised) use the model's own
   channel tables (``model.chan``, ``model.quat``) as host-side slices, so the step has no host sync and can be
   captured in a CUDA graph (``ContactNRD.assemble``/``finish`` call ``torch.nonzero(...).tolist()``).
   ``ContactNRD.details`` runs the contact network only on the switched-on entries (``torch.nonzero``) and gives the
   same next state up to float32 rounding of the contact network (other GEMM batch shapes; measured on 21,506
   recorded validation windows: max 1.8e-7, switches equal; ``scripts/so101_policy/bench_nrd_sim.py``). At
   construction the sim checks that its assemble/finish equal ``model.assemble``/``model.finish`` bit for bit on a
   probe batch; if a future model differs, the sim warns
   and calls the model's own functions (eager only, no CUDA graph).
2. History shift (``ContactNRD.advance``).
3. Cube canonicalisation (``canonical=True``, default): the newest cube quaternion picks one of the 24 cube rotations
   r (``cube_symmetry.canonical_index``); the quaternions of all K history slots of the world are relabelled
   q -> q r and their sign set so that the newest has w >= 0 (one relabel per world: the step differences stay
   consistent), then each relabelled slot is finished again. Positions, world velocity and world spin do not change.
   The refinish is ``model.finish`` restricted to the cube (quaternion re-normalised): the arm channels do not
   change. At
   construction the sim checks that this equals ``model.finish`` bit for bit on relabelled probe states (else it uses
   ``model.finish``). The cumulative relabelling C of each world (an index into the group,
   ``q_internal = q_true G[C]`` up to sign) is kept with the exact group table, and ``WorldState.cube_quat`` is the
   TRUE orientation ``q_internal conj(G[C])``, its sign kept continuous with the previous output.
   ``canonical=False`` leaves the quaternions alone (C = 0); then the sim is ``ContactNRD.rollout`` with the dense
   contact step.

Members: ``members="split"`` loads all checkpoints and gives member j the contiguous worlds
[j B / M, (j + 1) B / M); a name (or list of names) loads only those (one name = every world on that member). Names are
the dict keys of ``ckpts``, or the file stems; a string also matches a name that ends with it ("s61").

WorldState fields (``nedm.so101_policy.world``):

| Field | NRD rule |
|---|---|
| q, qd | arm angle and rate channels of the newest state (rates = 20 ms step averages, the data definition) |
| cube_pos, cube_v, cube_w | cube channels of the newest state (world frame) |
| cube_quat | the TRUE orientation (see 3.), w x y z |
| contact | per CONTACT_PAIRS, from the switches of the model (pair order read from ``system.pair_metadata``). ``contact_rule="gate"`` (default): g[k] of the step just taken, which stands for "the pair touched in step k or in step k-1" (the model's own label convention, ``data.model_labels``); no delay, dilated by one step. ``"physical"``: g[k-1] & g[k], the data's physical contact of step k-1 (``data.physical_switches``): one step late |
| arm_table | ``arm_table_fn(q [B, 6] float64) -> bool [B]`` if given, evaluated at every ``observe`` (the critic's rule, map 3.7: ``lowest_point_guard(kin.SO101Kin(device))`` = lowest gripper / moving-jaw collision point below 0.5 mm); default: all False (the NRD has no arm-table pair; NOTHING checks it) |
| invalid | any channel of the newest state non-finite; cube centre x, y outside the table minus its edge margin (the collector's ``cube_off_table`` rule: x in [-0.30, 0.80], y in [-0.55, 0.55] m for the checkpoint's table); cube centre z < ``cube_z_min`` (0.010 m: sunk > 5 mm into the table; data minimum 11.0 mm in the probe landings, 12.9 mm in the task families); z > ``cube_z_max`` (0.45 m; data maximum 405 mm); cube speed > ``cube_speed_max`` (4 m/s; data maximum 2.77) or spin > ``cube_spin_max`` (100 rad/s; data maximum 66). Data extremes: production_plus, all 37,500 episodes, all records |

The sim does not end or reset worlds: the caller resets invalid worlds. A non-finite world stays non-finite but does
not touch the other worlds (every operation is per world).

Resets (``reset``, ``reset_state``, ``reset_history``) and ``set_cube`` write in place, so a captured CUDA graph stays
valid; each returns the WorldState of ALL worlds (as the MuJoCo worlds do). A reset from a single state uses
``ContactNRD.start``: one valid history entry, the others masked (``ok`` False), as the evaluation's whole-episode
rollouts.

``set_cube`` (disturbance, review_spec_feasibility sec. 5): the 13 base cube channels (position, quaternion,
velocity, spin) are written into ALL K history slots, the arm channels and the masks ``ok`` are kept, and
``model.finish`` runs on each slot. With the default
v = w = 0 all K cube entries are equal: the history of a cube that has rested there, the only disturbance that is in
the data (approach phase: resting cube, arm moving above it). A nonzero v, w gives the constant-rate history
p_j = p - a_j dt v, q_j = exp(-a_j dt w / 2) q (world spin, the model's integration rule), a_j = K-1-j the age of
slot j; a sliding cube is NOT in the data. The caller keeps the new pose in distribution (``cube_rest_envelope``:
flat, at rest, data resting height, on the start annulus) and >= 2 cm from every gripper collision point, before the
GRASP close starts (the 2 cm check needs the arm's collision points: ``kin.SO101Kin``; not done here). The switches
and the contact flags stay those of the last step until the next step.

No gradient path: all methods run under ``torch.no_grad``. For gradients call the model methods directly.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Callable

import torch

from nedm.contact_nrd.model import ContactNRD, load_model
from nedm.so101_policy import cube_symmetry as cs
from nedm.so101_policy.kin import THI, TLO
from nedm.so101_policy.paths import DATA
from nedm.so101_policy.world import CONTACT_PAIRS, JOINTS, WorldState

STUDY_CKPTS = {"c1_s62": DATA / "nrd" / "c1_s62" / "best.pt"}   # the released policy NRD (so101_pick_place/README.md)
LIMITS = dict(cube_z_min=0.010, cube_z_max=0.45, cube_speed_max=4.0, cube_spin_max=100.0)
# Start states of the task families (val_only, 1,803 first records, measured): cube centre z = 0.014993 m in all, flat,
# at rest, on the annulus r 0.150-0.270 m around the pan axis, angle -75.0..74.9 deg (collector sample_cube).
CUBE_REST_Z = 0.014993
ANNULUS = dict(center_xy=(0.0388, 0.0), r_min=0.15, r_max=0.27, angle_max_deg=75.0)
ARM_TABLE_GUARD_M = 0.0005   # critic map 3.7: lowest gripper / moving-jaw collision point below 0.5 mm


def _cols(x) -> list[int]:
    return [int(c) for c in (x.tolist() if torch.is_tensor(x) else x) if int(c) >= 0]


def _index(cols: list[int], device):
    """A contiguous column list -> slice (no index tensor, no host-to-device copy); else a device index tensor."""
    if not cols:
        return None
    if cols == list(range(cols[0], cols[0] + len(cols))):
        return slice(cols[0], cols[0] + len(cols))
    return torch.tensor(cols, dtype=torch.long, device=device)


def _get(start, name):
    return start.get(name) if isinstance(start, dict) else getattr(start, name, None)


def lowest_point_guard(kin, threshold_m: float = ARM_TABLE_GUARD_M) -> Callable[[torch.Tensor], torch.Tensor]:
    """arm_table_fn of the critic's rule (map 3.7): ``kin.lowest_point(q) < threshold_m``, with kin a
    ``kin.SO101Kin`` (lowest world z of the gripper and moving-jaw collision geoms, m). q [B, 6] -> bool [B]."""
    return lambda q: kin.lowest_point(q) < threshold_m


def cube_rest_envelope(pos, quat, v=None, w=None, z_tol_m: float = 5e-4, tilt_max_deg: float = 1.0,
                       rate_max: float = 1e-6) -> torch.Tensor:
    """Is a new cube pose (``set_cube``) inside the start envelope of the data? pos [n, 3] (m), quat [n, 4] (w x y z),
    v, w [n, 3] (world, default 0) -> bool [n]: centre height within z_tol_m of CUBE_REST_Z, tilt (body z against
    world z) <= tilt_max_deg, |v| and |w| <= rate_max, centre on ANNULUS. The 2 cm gripper clearance is NOT checked."""
    pos = torch.as_tensor(pos, dtype=torch.float64)
    quat = torch.as_tensor(quat, dtype=torch.float64, device=pos.device)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    ok = (pos[:, 2] - CUBE_REST_Z).abs() <= z_tol_m
    up = 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)                         # world z of the body z axis
    ok &= up >= math.cos(math.radians(tilt_max_deg))
    for r in (v, w):
        if r is not None:
            ok &= torch.as_tensor(r, dtype=torch.float64, device=pos.device).reshape(-1, 3).norm(dim=-1) <= rate_max
    dx, dy = pos[:, 0] - ANNULUS["center_xy"][0], pos[:, 1] - ANNULUS["center_xy"][1]
    r = torch.hypot(dx, dy)
    ok &= (r >= ANNULUS["r_min"]) & (r <= ANNULUS["r_max"])
    ok &= torch.rad2deg(torch.atan2(dy, dx)).abs() <= ANNULUS["angle_max_deg"]
    return ok


class GraspNrdSim:
    """ckpts: path, list of paths or {name: path} (default STUDY_CKPTS = the released policy NRD c1_s62; a loaded
    ContactNRD may stand in for a path, it is moved to the device and its parameters frozen); num_envs
    B; members "split" | name | [names] | int; canonical: per-step cube canonicalisation; canon_margin: hysteresis of
    the relabelling (0 = the critic's rule); contact_rule "gate" | "physical"; arm_table_fn(q) -> bool [B] (e.g.
    lowest_point_guard);
    cuda_graph: capture the step in one CUDA graph at the first step (CUDA only); limits: overrides of LIMITS."""

    def __init__(self, ckpts=None, num_envs: int = 1, device="cuda", members="split", canonical: bool = True,
                 canon_margin: float = 0.0, contact_rule: str = "gate",
                 arm_table_fn: Callable[[torch.Tensor], torch.Tensor] | None = None, cuda_graph: bool = False,
                 limits: dict | None = None):
        ckpts = STUDY_CKPTS if ckpts is None else ckpts
        if isinstance(ckpts, (str, Path, ContactNRD)):
            ckpts = [ckpts]
        named = dict(ckpts) if isinstance(ckpts, dict) else {
            (f"m{j}" if isinstance(p, ContactNRD) else Path(p).stem): p for j, p in enumerate(ckpts)}
        names = list(named)
        if members in ("split", "all", None):
            pick = names
        else:
            wanted = [members] if isinstance(members, (str, int)) else list(members)
            pick = []
            for w in wanted:
                hit = [names[w]] if isinstance(w, int) else [n for n in names if n == w] or [n for n in names if n.endswith(str(w))]
                if len(hit) != 1:
                    raise ValueError(f"member {w!r} matches {hit} of {names}")
                pick += hit
        self.device = torch.device(device)
        self.member_names = pick
        self.paths = [f"<in-memory {n}>" if isinstance(named[n], ContactNRD) else str(named[n]) for n in pick]
        self.models: list[ContactNRD] = []
        for n, p in zip(pick, self.paths):
            m = named[n].to(self.device).eval() if isinstance(named[n], ContactNRD) else load_model(p, self.device)[0]
            for q in m.parameters():
                q.requires_grad_(False)
            if self.models and (m.config != self.models[0].config or m.system != self.models[0].system
                                or not torch.equal(m.smask, self.models[0].smask)):
                raise ValueError(f"{p}: members must share model_config, system and channel masks")
            self.models.append(m)
        m0 = self.models[0]
        self.B, self.M = int(num_envs), len(self.models)
        if self.B < self.M:
            raise ValueError(f"{self.B} worlds for {self.M} members")
        self.K, self.D, self.S, self.A, self.P, self.dt = m0.k, m0.n_bodies, m0.smax, m0.amax, m0.n_pairs, m0.dt
        self.canonical, self.canon_margin, self.contact_rule = bool(canonical), float(canon_margin), contact_rule
        if contact_rule not in ("gate", "physical"):
            raise ValueError(f"contact_rule {contact_rule!r}")
        self.arm_table_fn = arm_table_fn
        self.limits = {**LIMITS, **(limits or {})}
        self._tlo = torch.tensor(TLO, dtype=torch.float64, device=self.device)
        self._thi = torch.tensor(THI, dtype=torch.float64, device=self.device)
        self.use_graph = bool(cuda_graph) and self.device.type == "cuda"
        self._graph = None
        self._layout(m0)
        dev, f64 = self.device, torch.float64
        # fixed blocks of worlds per member, dense (world, pair) index lists per block
        self.blocks = []
        for j, m in enumerate(self.models):
            s, e = j * self.B // self.M, (j + 1) * self.B // self.M
            n = e - s
            self.blocks.append((m, slice(s, e), torch.arange(n, device=dev).repeat_interleave(self.P),
                                torch.arange(self.P, device=dev).repeat(n)))
        self.member_of_world = torch.cat([torch.full((b[1].stop - b[1].start,), j, dtype=torch.long, device=dev)
                                          for j, b in enumerate(self.blocks)])
        self._g, self._gconj = cs.group(f64, dev), cs.qconj(cs.group(f64, dev))
        self._mul, self._scores = cs.mul_table(dev), cs.score_matrix(dtype=f64, device=dev)
        # state buffers (written in place only: a captured graph keeps their addresses)
        B, K, D, S, P = self.B, self.K, self.D, self.S, self.P
        self.hist = torch.zeros(B, K, D, S, dtype=f64, device=dev)          # newest last; cube quaternion canonical
        self.ok = torch.zeros(B, K, dtype=torch.bool, device=dev)
        self.act = torch.zeros(B, D, self.A, dtype=f64, device=dev)         # row ia = the arm's joint targets
        self.gates = torch.zeros(B, P, dtype=f64, device=dev)               # model pair order
        self.prev_gates = torch.zeros_like(self.gates)
        self.logits = torch.zeros_like(self.gates)
        self.canon = torch.zeros(B, dtype=torch.long, device=dev)           # cumulative relabelling C
        self.true_q = torch.zeros(B, 4, dtype=f64, device=dev)
        self.true_q[:, 0] = 1.0
        self.relabelled = torch.zeros(B, dtype=torch.bool, device=dev)      # relabelled in the last step
        self.contact = torch.zeros(B, len(CONTACT_PAIRS), dtype=torch.bool, device=dev)
        self.bad = torch.zeros(B, dtype=torch.bool, device=dev)
        self.t = torch.zeros(B, dtype=torch.long, device=dev)               # steps since the last reset
        self._self_check(m0)

    # ---- layout ------------------------------------------------------------------------------------------------
    def _layout(self, m):
        """Channel and pair tables from the model (no hard-coded channel indices beyond the 6 + 6 arm and 13 cube
        base channels, which are checked)."""
        dev = self.device
        mov = [b for b in m.system["bodies"] if b["moving"]]
        self.ia = next(n for n, b in enumerate(mov) if int(b.get("action_dim", 0)) > 0)
        self.ic = next(n for n, b in enumerate(mov) if b["name"] == "cube")
        aq, aqd = _cols(m.chan[self.ia, 0]), _cols(m.chan[self.ia, 2])
        cp, cq, cv, cw = (_cols(m.chan[self.ic, k]) for k in range(4))
        if (len(aq), len(aqd), len(cp), len(cq), len(cv), len(cw)) != (JOINTS, JOINTS, 3, 4, 3, 3):
            raise ValueError("expected an arm with 6 angles + 6 rates and a cube with position, quaternion, velocity, spin")
        if int(mov[self.ia]["action_dim"]) != JOINTS:
            raise ValueError("expected 6 arm joint targets")
        self.c_aq, self.c_aqd = _index(aq, dev), _index(aqd, dev)
        self.c_cp, self.c_cq, self.c_cv, self.c_cw = (_index(c, dev) for c in (cp, cq, cv, cw))
        names = [pm.get("name") for pm in m.system.get("pair_metadata", [])]
        self.pair_names = names
        self.contact_order = torch.tensor([names.index(n) for n in CONTACT_PAIRS], dtype=torch.long, device=dev)
        self.slots = [(int(m.pair_pos[k, 0]), int(m.pair_pos[k, 1]) if bool(m.pair_mm[k]) else None) for k in range(m.n_pairs)]
        self._integ = []
        for n in torch.nonzero(m.integ_body).flatten().tolist():
            pc, qc, vc, wc = (_cols(m.chan[n, k]) for k in range(4))
            self._integ.append((n, _index(pc, dev) if pc and vc else None, _index(vc, dev) if pc and vc else None,
                                _index(qc, dev) if len(qc) == 4 and len(wc) == 3 else None,
                                _index(wc, dev) if len(qc) == 4 and len(wc) == 3 else None))
        self._quat = [(n, _index(_cols(torch.nonzero(m.quat[n]).flatten()), dev)) for n in range(m.n_bodies)
                      if bool(m.quat[n].any())]
        table = next((b for b in m.system["bodies"] if not b["moving"] and "size_xy_m" in b), None)
        margin = float(m.system.get("physics_config", {}).get("table", {}).get("edge_margin_m", 0.05))
        c, s = (table["center_xy_m"], table["size_xy_m"]) if table else ([0.25, 0.0], [1.2, 1.2])
        self.table_lo = [c[i] - 0.5 * s[i] + margin for i in range(2)]
        self.table_hi = [c[i] + 0.5 * s[i] - margin for i in range(2)]

    # ---- the model's assemble / finish without host syncs ---------------------------------------------------------
    def _finish(self, m, x):
        """ContactNRD.finish with precomputed column slices (same arithmetic, no torch.nonzero / .tolist())."""
        if not self._quat:
            return x
        out = x.clone()
        for n, qc in self._quat:
            q = x[:, n, qc]
            out[:, n, qc] = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return out

    def _finish_cube(self, m, x):
        """model.finish of states x [n, D, S] whose only change is the cube quaternion (a relabelling): the cube
        quaternion re-normalised."""
        out = x.clone()
        q = x[:, self.ic, self.c_cq]
        out[:, self.ic, self.c_cq] = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return out

    def _assemble(self, m, cur, d_core, change):
        """ContactNRD.assemble with precomputed column slices."""
        nxt = cur + d_core + change
        out = nxt.clone()
        for n, pc, vc, qc, wc in self._integ:
            mk = m.smask[n]
            if pc is not None:
                out[:, n, pc] = cur[:, n, pc] + m.dt * nxt[:, n, vc] * mk[pc]
            if qc is not None:
                half = 0.5 * m.dt * (nxt[:, n, wc] * mk[wc])
                ang = half.norm(dim=-1, keepdim=True)
                dq = torch.cat((torch.cos(ang), torch.where(ang > 1e-12, torch.sin(ang) / ang.clamp_min(1e-12), torch.ones_like(ang)) * half), -1)
                q = ContactNRD._qmul(dq, cur[:, n, qc])
                out[:, n, qc] = torch.where(mk[qc] > 0, q, cur[:, n, qc])
        return self._finish(m, out)

    @torch.no_grad()
    def _self_check(self, m):
        """The sync-free assemble/finish must equal the model's own, bit for bit, on a random probe batch. If not
        (a model with other channels), fall back to ContactNRD.assemble / finish themselves: same results,
        but host syncs, so no CUDA graph (warning)."""
        g = torch.Generator(device="cpu").manual_seed(0)
        cur = (torch.randn(16, self.D, self.S, generator=g, dtype=torch.float64) * 0.3).to(self.device) * m.smask
        d1 = (torch.randn(16, self.D, self.S, generator=g, dtype=torch.float64) * 0.3).to(self.device) * m.smask
        d2 = (torch.randn(16, self.D, self.S, generator=g, dtype=torch.float64) * 0.3).to(self.device) * m.smask
        try:
            self.fast_layout = bool(torch.equal(self._finish(m, cur), m.finish(cur))
                                    and torch.equal(self._assemble(m, cur, d1, d2), m.assemble(cur, d1, d2)))
        except (RuntimeError, IndexError, TypeError):
            self.fast_layout = False
        if not self.fast_layout:
            warnings.warn("GraspNrdSim: the model's finish/assemble differ from the sim's sync-free copy; using "
                          "ContactNRD.finish/assemble (eager only, no CUDA graph)")
            self._finish = lambda m_, x: m_.finish(x)
            self._assemble = lambda m_, cur_, d_core, change: m_.assemble(cur_, d_core, change)
            self.use_graph = False
        # the cube-only refinish of a relabelled state must equal model.finish (finished probes, every group element)
        x = m.finish(cur.repeat(2, 1, 1))
        k = torch.arange(len(x), device=self.device) % cs.ORDER
        x[:, self.ic, self.c_cq] = cs.relabel(x[:, self.ic, self.c_cq], k, self._g, newest_sign=False)
        try:
            self.fast_relabel = self.fast_layout and bool(torch.equal(self._finish_cube(m, x), m.finish(x)))
        except (RuntimeError, IndexError, TypeError):
            self.fast_relabel = False
        if not self.fast_relabel:
            self._finish_cube = lambda m_, x_: self._finish(m_, x_)

    # ---- the step ----------------------------------------------------------------------------------------------
    def _member_step(self, m, hist, ok, act, rows, pairs):
        """Dense step of one member: hist [b, K, D, S], ok [b, K], act [b, D, A] -> next [b, D, S], switches, logits [b, P]."""
        cur, diff, lag_ok = m.code(hist, ok)
        d_core = m.run_core(cur, diff, act, lag_ok)
        b = cur.shape[0]
        inp = m.pair_inputs(cur, diff, act, lag_ok, rows, pairs)            # the same inputs for both pair networks
        logits = m.collision(*inp)[:, 0].reshape(b, self.P)
        g = (logits >= 0).to(cur.dtype)
        o = m.contact(*inp).reshape(b, self.P, 2, self.S)
        gg = g[:, :, None, None]
        o = torch.where(gg > 0, o * gg, torch.zeros_like(o))
        parts = [torch.zeros_like(cur[:, 0]) for _ in range(self.D)]
        for k, (i, j) in enumerate(self.slots):                            # BOTH bodies of a pair get its change
            parts[i] = parts[i] + o[:, k, 0]
            if j is not None:
                parts[j] = parts[j] + o[:, k, 1]
        change = torch.stack(parts, 1) * m.smask
        return self._assemble(m, cur, d_core, change), g, logits

    def _relabel_hist(self, hist, k, changed):
        """Relabel the cube quaternions of all K slots by G[k] (sign of the newest slot: w >= 0) and refinish every
        slot of the worlds where changed (masked slots too; the model never reads them). hist [n, K, D, S] -> new."""
        h = hist.clone()
        h[:, :, self.ic, self.c_cq] = cs.relabel(hist[:, :, self.ic, self.c_cq], k, self._g)
        fin = self._finish_cube(self.models[0], h.flatten(0, 1)).view_as(h)
        return torch.where(changed[:, None, None, None], fin, hist)

    def _invalid(self, st):
        lim = self.limits
        p = st[:, self.ic, self.c_cp]
        bad = ~torch.isfinite(st).all(-1).all(-1)
        bad |= (p[:, 0] < self.table_lo[0]) | (p[:, 0] > self.table_hi[0]) | (p[:, 1] < self.table_lo[1]) | (p[:, 1] > self.table_hi[1])
        bad |= (p[:, 2] < lim["cube_z_min"]) | (p[:, 2] > lim["cube_z_max"])
        bad |= st[:, self.ic, self.c_cv].norm(dim=-1) > lim["cube_speed_max"]
        bad |= st[:, self.ic, self.c_cw].norm(dim=-1) > lim["cube_spin_max"]
        return bad

    def _step_inplace(self):
        outs = [self._member_step(m, self.hist[sl], self.ok[sl], self.act[sl], rows, pairs)
                for m, sl, rows, pairs in self.blocks]
        nxt, g, lg = (torch.cat([o[i] for o in outs]) if self.M > 1 else outs[0][i] for i in range(3))
        hist = torch.cat((self.hist[:, 1:], nxt[:, None]), 1)
        ok = torch.cat((self.ok[:, 1:], torch.ones_like(self.ok[:, :1])), 1)
        if self.canonical:
            qn = nxt[:, self.ic, self.c_cq]
            k = cs.canonical_index(qn, self._scores, self.canon_margin)
            changed = (k != 0) | (qn[:, 0] < 0)
            hist = self._relabel_hist(hist, k, changed)
            self.canon.copy_(self._mul[self.canon, k])
            self.relabelled.copy_(changed)
            qt = ContactNRD._qmul(hist[:, -1, self.ic, self.c_cq], self._gconj[self.canon])
            qt = qt * torch.where((qt * self.true_q).sum(-1, keepdim=True) < 0, -1.0, 1.0).to(qt.dtype)
        else:
            qt = hist[:, -1, self.ic, self.c_cq]
        self.hist.copy_(hist)
        self.ok.copy_(ok)
        self.true_q.copy_(qt)
        self.prev_gates.copy_(self.gates)
        self.gates.copy_(g)
        self.logits.copy_(lg)
        on = g > 0
        if self.contact_rule == "physical":
            on = on & (self.prev_gates > 0)
        self.contact.copy_(on[:, self.contact_order])
        self.bad.copy_(self._invalid(hist[:, -1]))
        self.t.add_(1)

    def _buffers(self):
        return [self.hist, self.ok, self.act, self.gates, self.prev_gates, self.logits, self.canon, self.true_q,
                self.relabelled, self.contact, self.bad, self.t]

    def _capture(self):
        snap = [t.clone() for t in self._buffers()]
        s = torch.cuda.Stream(self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step_inplace()
        torch.cuda.current_stream(self.device).wait_stream(s)
        for t, c in zip(self._buffers(), snap):
            t.copy_(c)
        del snap
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()     # the warm-up's cached blocks; the graph gets its own pool (shared GPUs: one, not two)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._step_inplace()

    @torch.no_grad()
    def step(self, joint_targets: torch.Tensor) -> WorldState:
        """joint_targets [B, 6]: absolute joint position targets (rad; jaw = index 5), held for 20 ms -> WorldState.
        The applied action is float32(clip(target, kin.TLO, kin.THI)), the collector's applied value (as MuJoCo and
        MuJoCo Warp apply it and the data store it; review F3)."""
        u = joint_targets.to(device=self.device, dtype=torch.float64)
        u = torch.minimum(torch.maximum(u, self._tlo), self._thi)
        self.act[:, self.ia, :JOINTS].copy_(u.to(torch.float32).to(torch.float64))
        if self.use_graph:
            if self._graph is None:
                self._capture()
            self._graph.replay()
        else:
            self._step_inplace()
        return self.observe()

    @torch.no_grad()
    def observe(self) -> WorldState:
        """The current WorldState of all worlds (copies, not views of the buffers)."""
        st = self.hist[:, -1]
        q = st[:, self.ia, self.c_aq].clone()
        arm_table = (self.arm_table_fn(q).to(device=self.device, dtype=torch.bool).reshape(self.B)
                     if self.arm_table_fn is not None else torch.zeros(self.B, dtype=torch.bool, device=self.device))
        return WorldState(q=q, qd=st[:, self.ia, self.c_aqd].clone(), cube_pos=st[:, self.ic, self.c_cp].clone(),
                          cube_quat=self.true_q.clone(), cube_v=st[:, self.ic, self.c_cv].clone(),
                          cube_w=st[:, self.ic, self.c_cw].clone(), contact=self.contact.clone(), arm_table=arm_table,
                          invalid=self.bad.clone())

    @property
    def state(self) -> torch.Tensor:
        """The newest NRD state [B, D, S] (model layout; cube quaternion canonical when canonical=True)."""
        return self.hist[:, -1]

    # ---- resets and disturbances -----------------------------------------------------------------------------------
    def _idx(self, idx) -> torch.Tensor:
        if idx is None:
            return torch.arange(self.B, device=self.device)
        idx = torch.as_tensor(idx, device=self.device)
        return torch.nonzero(idx).flatten() if idx.dtype == torch.bool else idx.long().reshape(-1)

    @torch.no_grad()
    def reset(self, idx, start) -> WorldState:
        """Reset worlds idx (long [n], bool [B] or None = all) to a resting-history start: start is a WorldState-like
        object or dict with q [n, 6], cube_pos [n, 3], cube_quat [n, 4] (true orientation, w x y z) and optionally
        qd, cube_v, cube_w (default 0). The quaternion is normalised (w >= 0) by model.finish; one valid history
        entry (ContactNRD.start); the held command = the start angles.
        -> WorldState of all worlds."""
        idx = self._idx(idx)
        f64 = dict(dtype=torch.float64, device=self.device)
        n = len(idx)
        raw = torch.zeros(n, self.D, self.S, **f64)
        z3 = torch.zeros(n, 3, **f64)
        raw[:, self.ia, self.c_aq] = torch.as_tensor(_get(start, "q"), **f64).reshape(n, JOINTS)
        qd = _get(start, "qd")
        raw[:, self.ia, self.c_aqd] = torch.zeros(n, JOINTS, **f64) if qd is None else torch.as_tensor(qd, **f64).reshape(n, JOINTS)
        q = torch.as_tensor(_get(start, "cube_quat"), **f64).reshape(n, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        raw[:, self.ic, self.c_cp] = torch.as_tensor(_get(start, "cube_pos"), **f64).reshape(n, 3)
        raw[:, self.ic, self.c_cq] = q * torch.where(q[:, :1] < 0, -1.0, 1.0).to(q.dtype)
        for name, c in (("cube_v", self.c_cv), ("cube_w", self.c_cw)):
            v = _get(start, name)
            raw[:, self.ic, c] = z3 if v is None else torch.as_tensor(v, **f64).reshape(n, 3)
        return self.reset_state(idx, raw, finish=True)

    @torch.no_grad()
    def reset_state(self, idx, state: torch.Tensor, finish: bool = True) -> WorldState:
        """Reset worlds idx from model-layout states [n, D, S] (one valid history entry). finish=False keeps the
        given quaternion as it is (exact replay of ContactNRD.rollout from a recorded state); a canonical relabelling
        still re-normalises the cube quaternion of the relabelled worlds. -> WorldState of all worlds."""
        state = state.to(device=self.device, dtype=torch.float64)
        h, ok = self.models[0].start(state)
        return self.reset_history(idx, h, ok, finish=finish)

    @torch.no_grad()
    def reset_history(self, idx, hist: torch.Tensor, ok: torch.Tensor, finish: bool = False) -> WorldState:
        """Reset worlds idx from model-layout histories [n, K, D, S] (newest last) with masks ok [n, K] (a recorded
        window, as rollouts.rollouts' "window" mode). The true orientation = the given newest cube quaternion.
        -> WorldState of all worlds."""
        idx = self._idx(idx)
        h = hist.to(device=self.device, dtype=torch.float64).clone()
        ok = ok.to(device=self.device, dtype=torch.bool)
        m0 = self.models[0]
        if finish:
            h = torch.where(ok[..., None, None], m0.finish(h.flatten(0, 1)).view_as(h), h)
        true_q = h[:, -1, self.ic, self.c_cq].clone()
        if self.canonical:
            k = cs.canonical_index(true_q, self._scores, self.canon_margin)
            h = self._relabel_hist(h, k, (k != 0) | (true_q[:, 0] < 0))
            self.canon[idx] = k
        else:
            self.canon[idx] = 0
        self.hist[idx], self.ok[idx], self.true_q[idx] = h, ok, true_q
        self.act[idx] = 0.0
        self.act[idx, self.ia, :JOINTS] = h[:, -1, self.ia, self.c_aq]
        for t in (self.gates, self.prev_gates, self.logits):
            t[idx] = 0.0
        self.relabelled[idx], self.contact[idx], self.t[idx] = False, False, 0
        self.bad[idx] = self._invalid(h[:, -1])
        return self.observe()

    @torch.no_grad()
    def set_cube(self, idx, pos, quat, v=None, w=None) -> WorldState:
        """Move the cube of worlds idx (disturbance, module docstring): pos [n, 3] (m, world), quat [n, 4] (TRUE
        orientation, w x y z), v, w [n, 3] (world, step average; default 0 = a resting cube, the only case in the
        data). The base cube channels go into all K history slots, the arm channels and ok are kept, model.finish
        runs on each slot; the cube is then canonicalised. -> WorldState of all worlds."""
        idx = self._idx(idx)
        n, K = len(idx), self.K
        f64 = dict(dtype=torch.float64, device=self.device)
        pos = torch.as_tensor(pos, **f64).reshape(n, 3)
        quat = torch.as_tensor(quat, **f64).reshape(n, 4)
        quat = quat / quat.norm(dim=-1, keepdim=True)
        v = torch.zeros(n, 3, **f64) if v is None else torch.as_tensor(v, **f64).reshape(n, 3)
        w = torch.zeros(n, 3, **f64) if w is None else torch.as_tensor(w, **f64).reshape(n, 3)
        quat = quat * torch.where((quat * self.true_q[idx]).sum(-1, keepdim=True) < 0, -1.0, 1.0).to(quat.dtype)
        age = torch.arange(K - 1, -1, -1, **f64)[None, :, None] * self.dt          # [1, K, 1] age of each slot (s)
        half = -0.5 * age * w[:, None]                                              # rotation back in time
        ang = half.norm(dim=-1, keepdim=True)
        dq = torch.cat((torch.cos(ang), torch.where(ang > 1e-12, torch.sin(ang) / ang.clamp_min(1e-12), torch.ones_like(ang)) * half), -1)
        h = self.hist[idx].clone()
        h[:, :, self.ic, self.c_cp] = pos[:, None] - age * v[:, None]
        h[:, :, self.ic, self.c_cq] = ContactNRD._qmul(dq, quat[:, None].expand(-1, K, -1))
        h[:, :, self.ic, self.c_cv] = v[:, None].expand(-1, K, -1)
        h[:, :, self.ic, self.c_cw] = w[:, None].expand(-1, K, -1)
        if self.canonical:
            k = cs.canonical_index(quat, self._scores, self.canon_margin)
            h[:, :, self.ic, self.c_cq] = cs.relabel(h[:, :, self.ic, self.c_cq], k, self._g)
            self.canon[idx] = k
        else:
            self.canon[idx] = 0
        h = self.models[0].finish(h.flatten(0, 1)).view_as(h)                      # every slot, masked ones too
        self.hist[idx], self.true_q[idx] = h, quat
        self.bad[idx] = self._invalid(h[:, -1])
        return self.observe()
