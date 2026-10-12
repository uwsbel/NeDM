"""Goal-conditioned pick-and-place task of the SO101 + cube policy study: goals, geometric success / failure / validity
rules, observation and reward (spec_v1 2.6 as changed by spec_v2 B2; rules of review_spec_feasibility sec. 4,
P0-5..8, P1-11, P1-13, P1-14). Every rule is a pure function of the ``WorldState`` (true state), the decoder's macro
phase and the kinematics (``kin.SO101Kin`` + ``TaskGeometry``), batched torch, the same code on CPU and GPU. No rule,
reward or observation reads a contact switch (``WorldState.contact`` is never used).

Frames and units: world = arm base frame (z up, table top z = 0), m, rad, s. Grasp frame = the gripper body frame moved
to the grasp point (TCP): x = closing axis, y = jaw width, z = up along the gripper (a top-down gripper has z up, so a
cube below the TCP has z < 0); ``cube_in_grasp`` = the collector's ``Scene.cube_in_gripper``. Cube tilt = angle of
the most vertical cube axis from the world z; cube yaw = yaw of a side axis after the smallest rotation that makes the
cube flat, mod 90 deg (``cube_yaw_tilt``; both exactly invariant under the 24 cube rotations, so the same with and
without the NRD's canonical relabelling).

Task (one episode, HORIZON 400 steps of 20 ms = 8 s): start = a resting cube and an arm pose (``load_start_bank``: the
first recorded states of task-family episodes; tests use fresh collector starts, spec_v2 B4); goal = cube centre
(x, y) on the workspace annulus (r 0.15-0.27 m about the pan axis, angle +-75 deg, the collector's ``sample_cube``
rule), 6-30 cm from the start, goal yaw = start yaw + U(-30, 30) deg (mod 90), clipped so that the total cube turn
from the grasp stays <= 0.6 rad (``sample_goals``; the C3-b goal change passes the turn made so far).

Per step (``PickPlaceTask.step``) the rules below run in this order; the first that fires sets the end code
(``ENDS``). Thresholds: ``TaskConfig`` (calibrated on validation records by ``scripts/so101_policy/calibrate_rules.py``,
values and false-fire / detection table in ~/so101_policy/docs/build2_task.md).

| End code | Rule (state after the step) |
|---|---|
| world_invalid / world_arm_table | the world's own flags (NRD: non-finite, cube off the table, guard; MuJoCo: real ends) |
| lowest_point | lowest point of all arm collision geoms (the collector's ``Scene.lowest`` set) < ``lowest_min`` |
| link_overlap | a non-jaw link collision box overlaps the cube by > ``link_depth`` (exact box-box separating-axis depth) |
| jaw_overlap | jaw visual meshes in the cube by more than ``jaw_depth`` beyond the jaw collision geoms' own penetration (the collector's ``VisualOverlap`` measure, direction "mesh points inside the cube" only, so it reads up to ~0.3 mm lower; hence 0.2 mm instead of the collector's 1 mm) |
| base_zone | the cube enters the bounding cylinder of the arm base |
| grasp_box | at the GRASP trigger (macro free -> grasp_descend): the cube centre against the decoder's planned descent end (``macro_goal``: x, y, z in its yaw frame, yaw within ``close_yaw``), or without it the horizontal offset from the measured TCP in the gripper yaw frame (x, y) and the measured yaw misalignment <= ``box_yaw``; at the close start (-> grasp_close): cube centre in the grasp frame outside the box (x, y, z) or yaw misalignment > ``close_yaw``. Training: termination; ``log_only`` for MuJoCo evaluation |
| keep_out | outside a macro, TCP z < ``keep_out_z`` while within ``keep_out_r`` (xy) of a resting cube |
| release_high | the cube is not held and its lowest corner is > ``release_z`` above the table while falling (v_z <= ``fall_vz``) |
| unsupported | same, not falling, for ``unsupported_steps`` steps (phantom support: the NRD exploit "cube on one jaw / near-miss lift") |
| hold_cap | held in the air (held and lowest corner > ``air_z``) for more than ``hold_cap_s`` in one run |
| cube_slide | outside a macro, the cube on the table, not between the jaws (outside the held box), horizontal speed > ``slide_v`` for ``slide_steps`` steps, more than ``slide_grace_s`` after it was last held |
| cube_tilted | the cube on the table, not held, tilt > ``tilt_fail`` for ``tilt_steps`` steps |
| arm_self | (``self_rule``, review F1) two arm collision geoms overlap by more than -``self_margin``: ``TaskGeometry.self_depth`` (the collector's kinematic self test ``Scene.clearance``: link / gripper / camera-mount / jaw boxes by the exact separating-axis depth, a box against the gripper or jaw hulls by vertices; MuJoCo's body filter) |
| moved_untouched | (``ghost_rule``, review F9) the cube on the table, no gripper / jaw collision geom within ``touch_d``, and the cube centre moved more than ``ghost_d`` (xy) or ``ghost_dz`` (z), or its yaw more than ``ghost_yaw``, since the last step a point was that near (or it was held or off the table): no touch, no motion |
| success | (terminal) a firm lift happened (held, lowest corner >= ``lift_z`` for ``lift_steps``), cube not held, on the table, tilt <= 5 deg, speed < 5 mm/s, xy error <= ``tol_xy``, yaw error (mod 90) <= ``tol_yaw``, jaw >= 0.4 rad, TCP >= cube top + 3 cm |
| timeout | t = horizon (truncation; bootstrap) |

"Held" = cube centre inside the held box of the grasp frame and the jaw angle inside the firm-grip band. "On the table" =
lowest cube corner < ``table_z``.

Reward (P1-14): progress shaping, not exp(-D). Potential phi = -(w_reach d_reach (until the first firm lift; frozen during
a GRASP macro) + w_goal d_goal), d_reach = |TCP xy - cube xy| + |TCP z - (cube z + pre_h)| + k_yaw |grasp yaw error|,
d_goal = max(xy error, 5 mm) + k_yaw max(yaw error, 5 deg) (the shaping saturates at 5 mm / 5 deg, P1-13). Reward =
phi' - phi + lift bonus (once) + success bonus - time cost; a failure end adds -fail and the discounted time cost of the
remaining steps (charge_remaining). Success ends the episode; timeout is a truncation.

Observation (``observation``, the same function for every world, no contact flags, P1-11): q, qd, TCP position, gripper
rotation (2 columns), cube position, sin / cos of 4 x cube yaw, cube tilt, cube v and w (optional, ``obs_rates``), cube in
the grasp frame, sin / cos of 4 x (cube yaw - gripper yaw), goal xy, sin / cos 4 x goal yaw, goal - cube xy, yaw error,
lifted flag, held-in-air time / cap, one-hot macro phase, and the decoder's own state vector (``dec_obs``).

Macro phases (decoder v2 contract, spec_v2 B1): ``MACROS``; a decoder without macros passes ``macro=None`` (all free;
the grasp-box rule then fires at a geometric close start: the jaw closing through ``close_jaw`` with the cube within
4 cm of the TCP).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from nedm.so101_policy.kin import BODIES, GRIPPER, PAN_AXIS_XY, SO101Kin, quat_mat, wrap, yaw_of
from nedm.so101_policy.world import CONTROL_DT, WorldState

CUBE_HALF = 0.015
CUBE_REST_Z = 0.014993            # resting cube centre height in the data (m)
QUARTER = math.pi / 2
HORIZON = 400
SELF_RULE_DEFAULT = True          # review F1 / F9 rules (calibration: ~/so101_policy/docs/build3_fixes.md)
GHOST_RULE_DEFAULT = True
ANNULUS_R = (0.15, 0.27)          # m about the pan axis (collector sample_cube)
ANNULUS_DEG = (-75.0, 75.0)
GOAL_DIST = (0.06, 0.30)          # m from the start cube centre
GOAL_TURN_DEG = 30.0
MAX_TURN = 0.6                    # rad, total cube (= gripper) yaw turn from the grasp
TASK_FAMILIES = tuple(range(7))   # nominal .. push: the task-family starts of the grasp data
MACROS = ("free", "grasp_descend", "grasp_close", "grasp_grip", "grasp_lift", "place_lower", "place_open",
          "place_retreat")
MC = {n: k for k, n in enumerate(MACROS)}
GRASP_MACROS = (1, 2, 3, 4)
PLACE_MACROS = (5, 6, 7)
# collector phases (so101_grasp.PHASES) -> macro codes, for calibration on recorded data
COLLECTOR_MACRO = {"descend": "grasp_descend", "close": "grasp_close", "grip": "grasp_grip", "lift": "grasp_lift",
                   "lower": "place_lower", "release": "place_open", "retreat": "place_retreat"}
ENDS = ("running", "success", "timeout", "world_invalid", "world_arm_table", "lowest_point", "link_overlap",
        "jaw_overlap", "base_zone", "grasp_box", "keep_out", "release_high", "unsupported", "hold_cap", "cube_slide",
        "cube_tilted", "arm_self", "moved_untouched")     # last two: review F1, F9 (appended: old codes keep values)
EC = {n: k for k, n in enumerate(ENDS)}
RULES = ENDS[3:]


@dataclass
class TaskConfig:
    """Thresholds (m, rad, s). Values marked [cal] come from the validation calibration (build2_task.md)."""
    horizon: int = HORIZON
    # success
    tol_xy: float = 0.015
    tol_yaw: float = math.radians(15.0)
    tol2_xy: float = 0.010                   # secondary success level, reported only
    tol2_yaw: float = math.radians(10.0)
    rest_speed: float = 0.005                # m/s, |cube_v|
    flat_tilt: float = math.radians(5.0)
    open_jaw: float = 0.4                    # rad
    tcp_clear: float = 0.03                  # TCP above the cube top
    lift_z: float = 0.02                     # firm lift: lowest corner (m) ...
    lift_steps: int = 10                     # ... held for 0.2 s
    # geometry of the grasp [cal]
    box_x: float = 0.0055                    # grasp box at the GRASP trigger / close start: |x| (closing axis)
    box_y: float = 0.0125                    # |y| (jaw width)
    box_z: tuple = (-0.0125, 0.0015)         # z of the cube centre in the grasp frame (cube below the TCP: z < 0)
    box_yaw: float = math.radians(4.0)       # |gripper yaw - cube face yaw| (mod 90) at the trigger (tilted gripper) [cal]
    close_yaw: float = math.radians(3.0)     # ... at the close start (top-down gripper) [cal]
    held_x: float = 0.012                    # held box (hold, carry): |x| (MuJoCo grips <= 5.4 mm; the NRD's held
                                             # cube drifts along the closing axis in carries: 7 mm ends 9.8 % of
                                             # nominal NRD replays, 12 mm 0.9 %), |y|, z range [cal]
    held_y: float = 0.015                    # off-centre grips reach 14.6 mm, near misses start at 15 mm
    held_z: tuple = (-0.015, 0.005)
    firm_jaw: tuple = (0.13, 0.185)          # rad, jaw angle band of a firm grip [cal]
    close_jaw: float = 0.35                  # rad, geometric close start without macros
    # failure rules [cal]
    lowest_min: float = 0.0                  # m, lowest arm collision point
    link_depth: float = 0.0                  # m (any overlap; nominal / off-centre max -30 mm) [cal]
    jaw_depth: float = 0.0002                # m; the collector ends at 1 mm of its measure, ours reads lower (no
                                             # cube-points-in-mesh test): 0.2 mm detects 97.8 % of its ends [cal]
    keep_out_z: float = 0.04                 # TCP height (m) ...
    keep_out_r: float = 0.05                 # ... within this xy distance of a resting cube
    table_z: float = 0.002                   # lowest cube corner below this = on the table
    release_z: float = 0.005
    fall_vz: float = -0.2                    # m/s
    unsupported_steps: int = 2
    air_z: float = 0.002
    hold_cap_s: float = 3.5
    slide_v: float = 0.03                    # m/s
    slide_grace_s: float = 0.3
    slide_steps: int = 2
    tilt_fail: float = math.radians(15.0)
    tilt_steps: int = 10
    self_rule: bool = SELF_RULE_DEFAULT      # arm self-contact (review F1): TaskGeometry.self_depth > -self_margin
    self_margin: float = 0.0002              # m [cal]: 18 / 19 MuJoCo arm_self ends of the v2 gate (46 / 49 all
                                             # configs) at their end state, 0 of 10,360 MuJoCo successes, 0 validation
                                             # records
    ghost_rule: bool = GHOST_RULE_DEFAULT    # cube moved on the table with no gripper / jaw point near it (review F9)
    touch_d: float = 0.02                    # m, gripper / jaw collision geoms within this of the cube = near
                                             # (TaskGeometry.jaw_clearance) [cal]
    ghost_d: float = 0.002                   # m, cube centre moved in xy since the last near / held / airborne step
                                             # [cal]
    ghost_dz: float = 0.003                  # m, or in z (the NRD lifts a resting cube by up to 1.8 mm in the first
                                             # step after a reset, then puts it back; sunk cubes: 4-7 mm) [cal]
    ghost_yaw: float = math.radians(5.0)     # rad, or its yaw (mod 90) turned [cal]
    log_only: tuple = ()                     # rule names that only log (MuJoCo evaluation: ("grasp_box",))
    # reward (P1-14)
    w_reach: float = 10.0                    # per m
    w_goal: float = 10.0                     # per m
    k_yaw: float = 0.05                      # m per rad
    pre_h: float = 0.05                      # m, reach target height above the cube centre
    sat_xy: float = 0.005
    sat_yaw: float = math.radians(5.0)
    lift_bonus: float = 1.0
    success_bonus: float = 5.0
    fail_cost: float = 2.0
    time_cost: float = 0.002                 # per step; charged for the remaining steps at a failure end
    gamma: float = 0.99                      # discount of the remaining-step charge
    # observation
    obs_rates: bool = True                   # cube v, w in the observation (P1-11 ablation: False)

    @staticmethod
    def evaluation(**kw) -> "TaskConfig":
        """MuJoCo evaluation: the grasp-box rule only logs (spec_v2 B2)."""
        return TaskConfig(log_only=("grasp_box",), **kw)


# ---- small geometry helpers ----------------------------------------------------------------------------------------
def wrap90(a):
    """Angle -> (-45, 45] deg equivalent under the cube's 90 deg symmetry."""
    return -(torch.remainder(-a + QUARTER / 2, QUARTER) - QUARTER / 2)


def cube_yaw_tilt(quat):
    """Cube orientation (w x y z) [B, 4] -> (yaw mod 90 deg in [0, pi/2), tilt rad). The cube axis nearest the world
    vertical is 'up' (sign: pointing up); tilt = its angle from the vertical; the cube is turned flat by the smallest
    rotation taking 'up' to the vertical, and yaw = the yaw of a side axis then (exactly the same for all 24
    relabellings of the cube's symmetry, also when tilted)."""
    r = quat_mat(quat)
    zr = r[:, 2, :]                                                                # world z of the 3 cube axes
    up = zr.abs().argmax(-1)
    u = r.gather(-1, up[:, None, None].expand(-1, 3, 1))[..., 0]
    u = u * torch.where(u[:, 2:] < 0, -1.0, 1.0)
    tilt = torch.acos(u[:, 2].clamp(max=1.0))
    side = torch.remainder(up + 1, 3)
    sv = r.gather(-1, side[:, None, None].expand(-1, 3, 1))[..., 0]
    # R = I + [w]x + [w]x^2 / (1 + c), w = u x e_z, c = u_z (> 0.57): R u = e_z
    w = torch.stack((u[:, 1], -u[:, 0], torch.zeros_like(u[:, 0])), -1)
    wx = torch.cross(w, sv, dim=-1)
    flat = sv + wx + torch.cross(w, wx, dim=-1) / (1 + u[:, 2:])
    return torch.remainder(torch.atan2(flat[:, 1], flat[:, 0]), QUARTER), tilt


def cube_bottom(pos, rot, half=CUBE_HALF):
    """Lowest cube corner z [B] (m) of a cube at pos [B, 3] with rotation rot [B, 3, 3]."""
    return pos[:, 2] - half * rot[:, 2, :].abs().sum(-1)


def mm3(a, b):
    """a @ b for small matrices [..., 3, 3] by broadcasting (no batched GEMM launch)."""
    return (a[..., :, :, None] * b[..., None, :, :]).sum(-2)


def mv3(a, v):
    """a @ v [..., 3]."""
    return (a * v[..., None, :]).sum(-1)


def mtv3(a, v):
    """a^T @ v [..., 3]."""
    return (a * v[..., :, None]).sum(-2)


def box_sdf(p, half):
    """Signed distance [...] of points p [..., 3] (cube frame) to an axis-aligned box of half size ``half`` (< 0 inside)."""
    q = p.abs() - half
    return q.clamp_min(0).norm(dim=-1) + q.amax(-1).clamp_max(0)


def sat_depth(c1, r1, h1, c2, r2, h2):
    """Exact overlap depth of two boxes by the separating-axis test (15 axes; the classic OBB form in the frame of box
    1): centres c [..., 3], rotations r [..., 3, 3] (columns = box axes), half sizes h [..., 3] -> [...] (m; > 0 =
    overlap by that depth, <= 0 = separated)."""
    R = mm3(r1.transpose(-1, -2), r2)                                           # box-2 axes in the box-1 frame
    t = mtv3(r1, c2 - c1)
    A = R.abs()
    ov = [h1 + (A * h2[..., None, :]).sum(-1) - t.abs(),                       # axes of box 1
          (A * h1[..., :, None]).sum(-2) + h2 - (t[..., :, None] * R).sum(-2).abs()]   # axes of box 2
    i1, i2 = torch.tensor([1, 2, 0]), torch.tensor([2, 0, 1])                   # (i + 1) % 3, (i + 2) % 3
    # axes e_i x b_j: ra = h1[i1] A[i2, j] + h1[i2] A[i1, j]; rb = h2[j1] A[i, j2] + h2[j2] A[i, j1];
    # d = |t[i2] R[i1, j] - t[i1] R[i2, j]|; length sqrt(1 - R[i, j]^2)
    ra = h1[..., i1, None] * A[..., i2, :] + h1[..., i2, None] * A[..., i1, :]
    rb = h2[..., None, i1] * A[..., :, i2] + h2[..., None, i2] * A[..., :, i1]
    d = (t[..., i2, None] * R[..., i1, :] - t[..., i1, None] * R[..., i2, :]).abs()
    n = (1 - R * R).clamp_min(0).sqrt()
    ov.append(torch.where(n > 1e-6, (ra + rb - d) / n.clamp_min(1e-6), torch.full_like(n, math.inf)).flatten(-2))
    return torch.cat(ov, -1).amin(-1)


# ---- arm geometry from the collector scene --------------------------------------------------------------------------
class TaskGeometry:
    """Collision and visual geometry of the collector's SO101 for the kinematic rules (body frames of ``kin.BODIES``).

    - ``low_*``: points and radii of the collector's arm collision geoms (``Scene.arm_geoms``: gripper, moving jaw and
      link geoms, shoulder excluded) -> lowest point (``Scene.lowest``).
    - ``link_*``: the non-jaw link collision boxes (classes LINK incl. the shoulder and the camera mount; all are boxes).
    - per jaw body (gripper = fixed jaw, moving jaw): collision boxes, round geoms (spheres / capsules as segments + radius)
      and convex-hull vertices of collision meshes; visual mesh surface points (vertices + points ``spacing`` apart on
      large triangles, one per ``cell``), as the collector's ``VisualOverlap`` builds them.
    - base cylinder (radius, top z) bounding the base collision hulls.
    Built from MuJoCo (``from_mujoco``, needs mujoco + the vendor Menagerie model) or loaded from ``save``'s .npz."""

    JAWS = ("gripper", "moving_jaw_so101_v1")

    def __init__(self, arrays: dict, device="cpu", dtype=torch.float64):
        self.arrays = arrays
        f = dict(device=torch.device(device), dtype=dtype)
        t = lambda k: torch.as_tensor(arrays[k], **f)                              # noqa: E731
        li = lambda k: torch.as_tensor(arrays[k], dtype=torch.long, device=f["device"])   # noqa: E731
        self.device, self.dtype = f["device"], dtype
        self.low_body, self.low_pts, self.low_rad = li("low_body"), t("low_pts"), t("low_rad")
        self.low_groups = [(int(b), self.low_pts[self.low_body == b], self.low_rad[self.low_body == b])
                           for b in torch.unique(self.low_body).tolist()]
        self.link_body, self.link_pos, self.link_rot, self.link_half = (li("link_body"), t("link_pos"), t("link_rot"),
                                                                        t("link_half"))
        self.jaw = []
        for j in range(2):
            p = f"jaw{j}_"
            self.jaw.append(dict(
                body=BODIES.index(self.JAWS[j]),
                box_pos=t(p + "box_pos"), box_rot=t(p + "box_rot"), box_half=t(p + "box_half"),
                seg_a=t(p + "seg_a"), seg_b=t(p + "seg_b"), seg_r=t(p + "seg_r"),
                hull=t(p + "hull"), vis=t(p + "vis"),
                vis32=torch.as_tensor(arrays[p + "vis"], dtype=torch.float32, device=f["device"])))
            w = torch.linspace(0, 1, 5, **f)[:, None, None]                         # 5 points per segment
            J = self.jaw[-1]
            J["seg5"] = (J["seg_a"] * (1 - w) + J["seg_b"] * w).reshape(-1, 3)
            J["seg5_r"] = J["seg_r"].repeat(5)
        self.base_r, self.base_z = float(arrays["base_r"]), float(arrays["base_z"])
        if "self_pairs" not in arrays:            # a geometry file written before the self-contact rule (review F1)
            arrays.update(_self_arrays_cached())
        self.self_body, self.self_pos = li("self_body"), t("self_pos")
        self.self_rot, self.self_half = t("self_rot"), t("self_half")
        self.self_pairs = li("self_pairs")
        self.self_hull_body, self.self_hull = li("self_hull_body"), t("self_hull")
        self.self_hull_planes, self.self_hull_pairs = t("self_hull_planes"), li("self_hull_pairs")

    @staticmethod
    def build_self_arrays(scene=None, samples=20000, reach=0.01, seed=0) -> dict:
        """Arm self-contact geometry (review F1; the collector's ``Scene.clearance`` self test = MuJoCo contacts of kind
        'arm self' at a joint state): the collision BOXES of the arm links, the gripper (with the camera mount) and the
        moving jaw in their body frames, and every box pair of different bodies that MuJoCo's body filter lets
        collide (same weld body, parent-child weld bodies and the model's excludes are skipped, as in
        mj_collision); plus the pairs of a box with a non-base convex hull (gripper_collision1, moving-jaw hulls).
        Only pairs that come within ``reach`` of each other (MuJoCo distance) in ``samples`` random joint states
        (uniform in the applied target range, numpy seed ``seed``) are kept: the others cannot touch. Not included:
        the base hulls and the jaw tip spheres / capsule (none came within 5 mm of another arm geom in the 1.57 M gate
        and data states of ~/so101_policy/docs/build3_fixes.md)."""
        import mujoco
        import numpy as np

        from nedm.contact_nrd.collect import so101_grasp as g
        s = scene or g.Scene(g.load_config())
        m = s.m
        kidx = {m.body(n).id: k for k, n in enumerate(BODIES)}
        sig = {(int(x) >> 16, int(x) & 0xFFFF) for x in m.exclude_signature[: m.nexclude]}
        dsbl = bool(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)

        def filtered(b1, b2):     # mj_collision's body-pair filter
            w1, w2 = m.body_weldid[b1], m.body_weldid[b2]
            if w1 == w2:
                return True
            p1, p2 = m.body_weldid[m.body_parentid[w1]], m.body_weldid[m.body_parentid[w2]]
            if not dsbl and w1 != 0 and w2 != 0 and (w1 == p2 or w2 == p1):
                return True
            return (b1, b2) in sig or (b2, b1) in sig

        def geom_rot(gid):
            r = np.zeros(9)
            mujoco.mju_quat2Mat(r, m.geom_quat[gid])
            return r.reshape(3, 3)

        arm = [gid for gid in range(m.ngeom) if s.cls[gid] in (g.LINK, g.FIXED, g.MOVING)]
        boxes = [gid for gid in arm if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX]
        hulls = [gid for gid in arm if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH]
        def ok(a, b):             # an arm-self pair that mj_collision would test
            return (s.kind[s.cls[a], s.cls[b]] == 5 and not filtered(m.geom_bodyid[a], m.geom_bodyid[b])
                    and bool(m.geom_contype[a] & m.geom_conaffinity[b] or m.geom_contype[b] & m.geom_conaffinity[a]))
        # pairs that can come within reach (sampled with MuJoCo on a copy of the model with that margin)
        near = set()
        if samples:
            from nedm.so101_policy.kin import THI, TLO
            mm = s.m.__copy__()
            mm.geom_margin[np.isin(s.cls, (g.LINK, g.FIXED, g.MOVING))] = reach
            dd = mujoco.MjData(mm)
            q = np.asarray(TLO) + (np.asarray(THI) - np.asarray(TLO)) * np.random.default_rng(seed).random((samples, 6))
            for qq in q:
                dd.qpos[:6] = qq
                mujoco.mj_kinematics(mm, dd)
                mujoco.mj_collision(mm, dd)
                for k in range(dd.ncon):
                    a, b = int(dd.contact.geom1[k]), int(dd.contact.geom2[k])
                    near.add((min(a, b), max(a, b)))
        close = lambda a, b: not samples or (min(a, b), max(a, b)) in near                 # noqa: E731
        pairs = [(i, j) for i in range(len(boxes)) for j in range(i + 1, len(boxes))
                 if ok(boxes[i], boxes[j]) and close(boxes[i], boxes[j])]
        hv, hp, hb, hpairs = [], [], [], []
        for k, gid in enumerate(hulls):
            mi = m.geom_dataid[gid]
            a = m.mesh_graphadr[mi]
            nv, nf = m.mesh_graph[a], m.mesh_graph[a + 1]
            ids = m.mesh_graph[a + 2 + nv: a + 2 + 2 * nv]
            faces = m.mesh_graph[a + 2 + 3 * nv + 3 * nf: a + 2 + 3 * nv + 6 * nf].reshape(nf, 3)
            R, P = geom_rot(gid), m.geom_pos[gid]
            mv = m.mesh_vert[m.mesh_vertadr[mi]: m.mesh_vertadr[mi] + m.mesh_vertnum[mi]].astype(float) @ R.T + P
            v = mv[ids]
            n = np.cross(mv[faces[:, 1]] - mv[faces[:, 0]], mv[faces[:, 2]] - mv[faces[:, 0]])
            n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-30)
            d = -(n * mv[faces[:, 0]]).sum(1)
            flip = (v.mean(0) @ n.T + d) > 0
            n[flip], d[flip] = -n[flip], -d[flip]
            hv.append(v), hp.append(np.concatenate((n, d[:, None]), 1)), hb.append(kidx[m.geom_bodyid[gid]])
            hpairs += [(k, i) for i, b in enumerate(boxes) if ok(gid, b) and close(gid, b)]
        V, F = max(len(x) for x in hv), max(len(x) for x in hp)
        pad = lambda x, n: np.concatenate((x, np.repeat(x[:1], n - len(x), 0)))      # noqa: E731
        return dict(self_body=np.array([kidx[m.geom_bodyid[gid]] for gid in boxes]),
                    self_pos=np.array([m.geom_pos[gid] for gid in boxes]),
                    self_rot=np.array([geom_rot(gid) for gid in boxes]),
                    self_half=np.array([m.geom_size[gid] for gid in boxes]),
                    self_names=np.array([m.geom(gid).name for gid in boxes]),
                    self_pairs=np.array(pairs, np.int64).reshape(-1, 2),
                    self_hull_body=np.array(hb), self_hull=np.stack([pad(x, V) for x in hv]),
                    self_hull_planes=np.stack([pad(x, F) for x in hp]),
                    self_hull_names=np.array([m.geom(gid).name for gid in hulls]),
                    self_hull_pairs=np.array(hpairs, np.int64).reshape(-1, 2))

    @staticmethod
    def build_arrays(spacing=0.002, cell=0.002) -> dict:
        import mujoco
        import numpy as np

        from nedm.contact_nrd.collect import so101_grasp as g
        s = g.Scene(g.load_config())
        m = s.m
        kidx = {m.body(n).id: k for k, n in enumerate(BODIES)}

        def geom_rot(gid):
            r = np.zeros(9)
            mujoco.mju_quat2Mat(r, m.geom_quat[gid])
            return r.reshape(3, 3)

        out = {}
        # lowest point: Scene.arm_geoms / Scene.points in body frames
        lb, lp, lr = [], [], []
        for gid, (p, rad) in zip(s.arm_geoms, s.points):
            pts = p @ geom_rot(gid).T + m.geom_pos[gid]
            lb += [kidx[m.geom_bodyid[gid]]] * len(pts)
            lp.append(pts)
            lr += [rad] * len(pts)
        out.update(low_body=np.array(lb), low_pts=np.concatenate(lp), low_rad=np.array(lr))
        # link boxes
        link = [gid for gid in range(m.ngeom) if s.cls[gid] == g.LINK]
        if any(m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX for gid in link):
            raise ValueError("expected boxes only for the non-jaw link collision geoms")
        out.update(link_body=np.array([kidx[m.geom_bodyid[gid]] for gid in link]),
                   link_pos=np.array([m.geom_pos[gid] for gid in link]),
                   link_rot=np.array([geom_rot(gid) for gid in link]),
                   link_half=np.array([m.geom_size[gid] for gid in link]),
                   link_names=np.array([m.geom(gid).name for gid in link]))
        # jaw bodies
        for j, name in enumerate(TaskGeometry.JAWS):
            b = m.body(name).id
            cls = g.FIXED if j == 0 else g.MOVING
            boxes, segs, hull = [], [], []
            for gid in np.flatnonzero((m.geom_bodyid == b) & (s.cls == cls)):
                t, size, R, P = int(m.geom_type[gid]), m.geom_size[gid], geom_rot(gid), m.geom_pos[gid]
                if t == mujoco.mjtGeom.mjGEOM_BOX:
                    boxes.append((P, R, size.copy()))
                elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                    segs.append((P, P, size[0]))
                elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    segs.append((P - R[:, 2] * size[1], P + R[:, 2] * size[1], size[0]))
                elif t == mujoco.mjtGeom.mjGEOM_MESH:
                    k = m.geom_dataid[gid]
                    a = m.mesh_graphadr[k]
                    if a < 0:
                        raise ValueError(f"collision mesh {m.geom(gid).name} has no convex hull graph")
                    nv, nf = m.mesh_graph[a], m.mesh_graph[a + 1]
                    ids = m.mesh_graph[a + 2 + nv: a + 2 + 2 * nv]           # vert_globalid
                    fo = a + 2 + 3 * nv + 3 * nf                                # face_globalid
                    faces = m.mesh_graph[fo: fo + 3 * nf].reshape(nf, 3)
                    mv = m.mesh_vert[m.mesh_vertadr[k]: m.mesh_vertadr[k] + m.mesh_vertnum[k]].astype(float) @ R.T + P
                    v = mv[ids]
                    n = np.cross(mv[faces[:, 1]] - mv[faces[:, 0]], mv[faces[:, 2]] - mv[faces[:, 0]])
                    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-30)
                    d = -(n * mv[faces[:, 0]]).sum(1)
                    flip = (v.mean(0) @ n.T + d) > 0                            # outward normals: centroid inside
                    n[flip], d[flip] = -n[flip], -d[flip]
                    hull.append((v, np.concatenate((n, d[:, None]), 1)))
                else:
                    raise ValueError(f"geom type {t}")
            out[f"jaw{j}_box_pos"] = np.array([x[0] for x in boxes]).reshape(-1, 3)
            out[f"jaw{j}_box_rot"] = np.array([x[1] for x in boxes]).reshape(-1, 3, 3)
            out[f"jaw{j}_box_half"] = np.array([x[2] for x in boxes]).reshape(-1, 3)
            out[f"jaw{j}_seg_a"] = np.array([x[0] for x in segs]).reshape(-1, 3)
            out[f"jaw{j}_seg_b"] = np.array([x[1] for x in segs]).reshape(-1, 3)
            out[f"jaw{j}_seg_r"] = np.array([x[2] for x in segs]).reshape(-1)
            out[f"jaw{j}_hull"] = np.concatenate([h[0] for h in hull]).reshape(-1, 3) if hull else np.zeros((0, 3))
            vis = [gid for gid in np.flatnonzero(m.geom_bodyid == b)
                   if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH and not (m.geom_contype[gid] or m.geom_conaffinity[gid])]
            pts = g.surface_points(np.concatenate([g.mesh_triangles(m, gid) for gid in vis]), spacing)
            _, keep = np.unique(np.floor(pts / cell).astype(np.int64), axis=0, return_index=True)
            pts = pts[np.sort(keep)]
            # a visual point inside a collision geom of its body can never be deeper in the cube than that geom's
            # penetration (a ball of radius e around it lies inside the geom): such points cannot fire, drop them
            inside = np.zeros(len(pts), bool)
            for P, R, h in boxes:
                inside |= (np.abs((pts - P) @ R) <= h).all(1)
            for a_, b_, r in segs:
                ab = b_ - a_
                u = np.clip(((pts - a_) @ ab) / max(float(ab @ ab), 1e-30), 0, 1)
                inside |= np.linalg.norm(pts - (a_ + u[:, None] * ab), axis=1) <= r
            for _, planes in hull:
                inside |= (pts @ planes[:, :3].T + planes[:, 3] <= 0).all(1)
            out[f"jaw{j}_vis"] = pts[~inside]
            out[f"jaw{j}_vis_all"] = np.int64(len(pts))
        # base cylinder
        bv = []
        for gid in np.flatnonzero(s.cls == g.BASE):
            k = m.geom_dataid[gid]
            v = m.mesh_vert[m.mesh_vertadr[k]: m.mesh_vertadr[k] + m.mesh_vertnum[k]].astype(float)
            bv.append(v @ geom_rot(gid).T + m.geom_pos[gid])
        bv = np.concatenate(bv)                                                     # base body = world
        out.update(base_r=np.float64(np.linalg.norm(bv[:, :2], axis=1).max()), base_z=np.float64(bv[:, 2].max()),
                   spacing=np.float64(spacing), cell=np.float64(cell))
        out.update(TaskGeometry.build_self_arrays(s))
        return out

    @classmethod
    def from_mujoco(cls, device="cpu", dtype=torch.float64, spacing=0.002, cell=0.002):
        key = (spacing, cell)
        if key not in _GEOM_CACHE:
            _GEOM_CACHE[key] = cls.build_arrays(spacing, cell)
        return cls(_GEOM_CACHE[key], device, dtype)

    @classmethod
    def load(cls, path, device="cpu", dtype=torch.float64):
        import numpy as np
        f = np.load(path)
        return cls({k: f[k] for k in f.files}, device, dtype)

    @classmethod
    def default(cls, device="cpu", dtype=torch.float64, path=None):
        """From ``path`` if it exists, else from MuJoCo (and written to ``path`` when given)."""
        if path is not None and Path(path).exists():
            return cls.load(path, device, dtype)
        geom = cls.from_mujoco(device, dtype)
        if path is not None:
            geom.save(path)
        return geom

    def save(self, path):
        import numpy as np
        np.savez(path, **self.arrays)

    # -- per-state measures (bpos [B, 8, 3], brot [B, 8, 3, 3] = kin.body_poses; c [B, 3], rc [B, 3, 3] = the cube) --
    def lowest(self, bpos, brot):
        """Lowest world z [B] of the arm collision geoms (Scene.lowest), per body one matrix product."""
        z = [(bpos[:, b, 2:3] + brot[:, b, 2, 0:1] * p[:, 0] + brot[:, b, 2, 1:2] * p[:, 1] + brot[:, b, 2, 2:3] * p[:, 2]
              - r).amin(-1) for b, p, r in self.low_groups]
        return torch.stack(z, -1).amin(-1)

    def link_depth(self, bpos, brot, c, rc):
        """Deepest overlap [B] (m) of a non-jaw link collision box with the cube (> 0 = overlap)."""
        R = brot[:, self.link_body]                                                 # [B, G, 3, 3]
        pc = bpos[:, self.link_body] + mv3(R, self.link_pos)
        h = torch.full_like(self.link_half, CUBE_HALF)
        d = sat_depth(c[:, None], rc[:, None], h, pc, mm3(R, self.link_rot), self.link_half)
        return d.amax(-1)

    def jaw_depths(self, bpos, brot, c, rc, chunk=4096):
        """Per jaw body (gripper, moving jaw) -> (visual depth [B, 2], collision depth [B, 2]) (m): the deepest visual
        mesh point inside the cube, and the deepest penetration of that body's collision geoms into the cube (boxes:
        separating-axis depth; spheres / capsules: radius minus the signed distance of 5 points of the segment; hull
        meshes: the deepest hull vertex)."""
        B = c.shape[0]
        vis, col = torch.empty(B, 2, dtype=c.dtype, device=c.device), torch.empty(B, 2, dtype=c.dtype, device=c.device)
        half = torch.full((3,), CUBE_HALF, dtype=c.dtype, device=c.device)
        for j, J in enumerate(self.jaw):
            Rb, pb = brot[:, J["body"]], bpos[:, J["body"]]
            M = mm3(rc.transpose(-1, -2), Rb)                                       # body -> cube frame
            t = mtv3(rc, pb - c)
            v32, M32, t32 = J["vis32"], M.float(), t.float()
            for s in range(0, B, chunk):
                m_, t_ = M32[s: s + chunk, None], t32[s: s + chunk, None]          # [b, 1, 3, 3], [b, 1, 3]
                inf = None
                for a in range(3):                                                  # L-inf norm in the cube frame
                    pa = (t_[..., a] + v32[:, 0] * m_[..., a, 0] + v32[:, 1] * m_[..., a, 1]
                          + v32[:, 2] * m_[..., a, 2]).abs()
                    inf = pa if inf is None else torch.maximum(inf, pa)
                vis[s: s + chunk, j] = CUBE_HALF - inf.amin(-1).to(c.dtype)         # the deepest point
            parts = []
            for pts, rad in ((J["hull"], None), (J["seg5"], J["seg5_r"])):
                if len(pts):
                    p = mv3(M[:, None], pts) + t[:, None]
                    parts.append(((0 if rad is None else rad) - box_sdf(p, half)).amax(-1))
            if len(J["box_half"]):
                parts.append(sat_depth(c[:, None], rc[:, None], half, pb[:, None] + mv3(Rb[:, None], J["box_pos"]),
                                       mm3(Rb[:, None], J["box_rot"]), J["box_half"]).amax(-1))
            col[:, j] = torch.stack(parts, -1).amax(-1)
        return vis, col

    def jaw_clearance(self, bpos, brot, c, rc):
        """Smallest clearance [B] (m) between the cube and the collision geoms of the gripper (with the fixed jaw) and
        the moving jaw: minus the deepest penetration of ``jaw_depths``' collision part (boxes: separating-axis gap, a
        lower bound of the distance; hull vertices and sphere / capsule points: their distance to the cube box)."""
        half = torch.full((3,), CUBE_HALF, dtype=c.dtype, device=c.device)
        out = []
        for J in self.jaw:
            Rb, pb = brot[:, J["body"]], bpos[:, J["body"]]
            M = mm3(rc.transpose(-1, -2), Rb)
            t = mtv3(rc, pb - c)
            for pts, rad in ((J["hull"], None), (J["seg5"], J["seg5_r"])):
                if len(pts):
                    sdf = box_sdf(mv3(M[:, None], pts) + t[:, None], half)
                    out.append((sdf - (0 if rad is None else rad)).amin(-1))
            if len(J["box_half"]):
                out.append(-sat_depth(c[:, None], rc[:, None], half, pb[:, None] + mv3(Rb[:, None], J["box_pos"]),
                                      mm3(Rb[:, None], J["box_rot"]), J["box_half"]).amax(-1))
        return torch.stack(out, -1).amin(-1)

    def jaw_overlap(self, bpos, brot, c, rc, chunk=4096):
        """The collector's VisualOverlap measure [B] (m): max over the 2 jaw bodies of the visual depth minus the
        body's collision penetration (if > 0), at least 0."""
        vis, col = self.jaw_depths(bpos, brot, c, rc, chunk)
        return (vis - col.clamp_min(0)).amax(-1).clamp_min(0)

    def self_depth(self, bpos, brot, cap=0.005):
        """Deepest arm self-overlap [B] (m; > 0 = two arm collision geoms overlap, <= 0 = separated): exact
        separating-axis depth of every box pair (``build_self_arrays``), and for a box against a convex hull the
        deepest vertex of either inside the other (vertex-face only: an edge-edge crossing reads as separated). Pairs
        whose bounding spheres are more than ``cap`` apart are not tested (they read minus that gap)."""
        B = bpos.shape[0]
        G = brot[:, self.self_body]                                                   # [B, G, 3, 3]
        cw = bpos[:, self.self_body] + mv3(G, self.self_pos)                         # box centres, world
        i, j = self.self_pairs[:, 0], self.self_pairs[:, 1]
        rad = self.self_half.norm(dim=-1)
        gap = (cw[:, i] - cw[:, j]).norm(dim=-1) - rad[i] - rad[j]                    # [B, P] sphere gap
        far = lambda x: torch.where(x >= cap, -x, torch.full_like(x, -math.inf))      # noqa: E731
        best = far(gap).amax(-1) if gap.shape[1] else torch.full((B,), -math.inf, dtype=bpos.dtype, device=bpos.device)
        w, k = (gap < cap).nonzero(as_tuple=True)
        if len(w):
            rw = mm3(G[w[:, None], torch.stack((i[k], j[k]), -1)], self.self_rot[torch.stack((i[k], j[k]), -1)])
            d = sat_depth(cw[w, i[k]], rw[:, 0], self.self_half[i[k]], cw[w, j[k]], rw[:, 1], self.self_half[j[k]])
            best = best.scatter_reduce(0, w, d, reduce="amax")
        if len(self.self_hull_pairs):
            hk, bk = self.self_hull_pairs[:, 0], self.self_hull_pairs[:, 1]
            hb = self.self_hull_body[hk]
            hc = self.self_hull.mean(1)                                                # hull centre, body frame
            hr = (self.self_hull - hc[:, None]).norm(dim=-1).amax(-1)
            Rh, Ph = brot[:, hb], bpos[:, hb]                                          # [B, H, 3, 3], [B, H, 3]
            hcw = Ph + mv3(Rh, hc[hk])
            gap = (hcw - cw[:, bk]).norm(dim=-1) - hr[hk] - rad[bk]
            best = torch.maximum(best, far(gap).amax(-1))
            w, k = (gap < cap).nonzero(as_tuple=True)
            if len(w):
                Rk, Pk = Rh[w, k], Ph[w, k]                                            # [n, 3, 3], [n, 3]
                vw = Pk[:, None] + mv3(Rk[:, None], self.self_hull[hk[k]])             # hull vertices, world [n, V, 3]
                bb = bk[k]
                rb = mm3(G[w, bb], self.self_rot[bb])
                cb, hh = cw[w, bb], self.self_half[bb]
                dv = (-box_sdf(mtv3(rb[:, None], vw - cb[:, None]), hh[:, None])).amax(-1)   # hull vertex in the box
                sg = torch.tensor([-1.0, 1.0], dtype=bpos.dtype, device=bpos.device)
                corner = torch.stack(torch.meshgrid(sg, sg, sg, indexing="ij"), -1).reshape(8, 3)
                cl = mtv3(Rk[:, None], cb[:, None] + mv3(rb[:, None], corner * hh[:, None]) - Pk[:, None])   # [n, 8, 3]
                pl = self.self_hull_planes[hk[k]]                                      # [n, F, 4]
                dc = (-(torch.einsum("nkc,nfc->nkf", cl, pl[..., :3]) + pl[:, None, :, 3]).amax(-1)).amax(-1)
                best = best.scatter_reduce(0, w, torch.maximum(dv, dc), reduce="amax")
        return best

    def in_base_zone(self, c, bottom):
        """The cube's bounding circle enters the base cylinder [B] bool."""
        return (c[:, :2].norm(dim=-1) - CUBE_HALF * math.sqrt(2) < self.base_r) & (bottom < self.base_z)


_GEOM_CACHE: dict = {}
_SELF_CACHE: dict = {}


def _self_arrays_cached() -> dict:
    """``TaskGeometry.build_self_arrays`` once per process (needs mujoco and the vendor SO101 model)."""
    if not _SELF_CACHE:
        try:
            _SELF_CACHE.update(TaskGeometry.build_self_arrays())
        except ImportError as e:
            raise ImportError("this TaskGeometry file has no self-contact arrays (review F1) and mujoco is not "
                              "importable to build them: use a geometry file written by TaskGeometry.save after "
                              "2026-10-11 21:00 (e.g. ~/so101_policy/ckpt/task_geom_v2.npz)") from e
    return _SELF_CACHE


# ---- per-step features (pure functions of the state) ---------------------------------------------------------------
def features(state: WorldState, kin: SO101Kin, geom: TaskGeometry | None, cfg: TaskConfig, overlap=True) -> dict:
    """Geometric quantities of one batch of states used by every rule, the reward and the observation (float64)."""
    f64 = dict(dtype=kin.dtype)
    q = state.q.to(**f64)
    c = state.cube_pos.to(**f64)
    quat = state.cube_quat.to(**f64)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    rc = quat_mat(quat)
    if geom is not None:
        bpos, brot = kin.body_poses(q)
        rg = brot[:, GRIPPER]
        tcp = bpos[:, GRIPPER] + mv3(rg, kin.point)
    else:
        tcp, rg = kin.fk(q)
    xg = mtv3(rg, c - tcp)                                                         # cube centre in the grasp frame
    gyaw = yaw_of(rg)
    cyaw, tilt = cube_yaw_tilt(quat)
    bottom = cube_bottom(c, rc)
    v = state.cube_v.to(**f64)
    jaw = q[:, 5]
    near_jaws = ((xg[:, 0].abs() <= cfg.held_x) & (xg[:, 1].abs() <= cfg.held_y) & (xg[:, 2] >= cfg.held_z[0])
               & (xg[:, 2] <= cfg.held_z[1]))
    firm = (jaw >= cfg.firm_jaw[0]) & (jaw <= cfg.firm_jaw[1])
    in_box = ((xg[:, 0].abs() <= cfg.box_x) & (xg[:, 1].abs() <= cfg.box_y) & (xg[:, 2] >= cfg.box_z[0])
              & (xg[:, 2] <= cfg.box_z[1]))
    # horizontal offset of the cube from the TCP in the gripper yaw frame (trigger box; the descent is vertical)
    d = c[:, :2] - tcp[:, :2]
    cy, sy = torch.cos(gyaw), torch.sin(gyaw)
    hx, hy = cy * d[:, 0] + sy * d[:, 1], -sy * d[:, 0] + cy * d[:, 1]
    out = dict(q=q, qd=state.qd.to(**f64), tcp=tcp, rg=rg, c=c, quat=quat, rc=rc, xg=xg, gyaw=gyaw, cyaw=cyaw, tilt=tilt, bottom=bottom,
               v=v, w=state.cube_w.to(**f64), speed=v.norm(dim=-1), vxy=v[:, :2].norm(dim=-1), jaw=jaw,
               held=near_jaws & firm, near_jaws=near_jaws, in_box=in_box, firm=firm, hx=hx, hy=hy, dxy=d.norm(dim=-1),
               yaw_mis=wrap90(gyaw - cyaw).abs(), on_table=bottom < cfg.table_z,
               finite=torch.isfinite(q).all(-1) & torch.isfinite(c).all(-1) & torch.isfinite(quat).all(-1))
    if geom is not None:
        out["lowest"] = geom.lowest(bpos, brot)
        out["base_zone"] = geom.in_base_zone(c, bottom)
        if cfg.self_rule:
            out["self_depth"] = geom.self_depth(bpos, brot)
        if cfg.ghost_rule and not overlap:
            out["touch"] = geom.jaw_clearance(bpos, brot, c, rc)
        if overlap:
            out["link_depth"] = geom.link_depth(bpos, brot, c, rc)
            vis, col = geom.jaw_depths(bpos, brot, c, rc)                       # (= jaw_overlap, jaw_clearance)
            out["jaw_overlap"] = (vis - col.clamp_min(0)).amax(-1).clamp_min(0)
            if cfg.ghost_rule:
                out["touch"] = -col.amax(-1)
    return out


def goal_errors(F, goal_xy, goal_yaw):
    """(xy error [B] m, yaw error [B] rad mod 90) of the cube against the goal."""
    return (F["c"][:, :2] - goal_xy).norm(dim=-1), wrap90(F["cyaw"] - goal_yaw).abs()


def trigger_box_ok(F, cfg: TaskConfig):
    """Grasp box at the GRASP trigger: horizontal offset in the gripper yaw frame and yaw misalignment [B] bool."""
    return (F["hx"].abs() <= cfg.box_x) & (F["hy"].abs() <= cfg.box_y) & (F["yaw_mis"] <= cfg.box_yaw)


def goal_box_ok(F, cfg: TaskConfig, macro_goal):
    """Grasp box at the GRASP trigger against the decoder's planned descent end macro_goal [B, 4] (TCP x, y, z, yaw):
    cube centre in that pose's yaw frame within the box (x, y, z) and its yaw within ``close_yaw`` of the cube's
    (the commanded yaw is not tilted) [B] bool."""
    g = macro_goal.to(F["c"])
    d = F["c"] - g[:, :3]
    cy, sy = torch.cos(g[:, 3]), torch.sin(g[:, 3])
    x, y = cy * d[:, 0] + sy * d[:, 1], -sy * d[:, 0] + cy * d[:, 1]
    return ((x.abs() <= cfg.box_x) & (y.abs() <= cfg.box_y) & (d[:, 2] >= cfg.box_z[0]) & (d[:, 2] <= cfg.box_z[1])
            & (wrap90(g[:, 3] - F["cyaw"]).abs() <= cfg.close_yaw))


def close_box_ok(F, cfg: TaskConfig):
    """Grasp box at the close start: cube centre in the grasp frame and yaw misalignment [B] bool."""
    return F["in_box"] & (F["yaw_mis"] <= cfg.close_yaw)


def keep_out(F, cfg: TaskConfig, free):
    """Outside a macro (free [B] bool), TCP below keep_out_z within keep_out_r (xy) of a resting cube [B] bool."""
    resting = F["on_table"] & ~F["held"]
    return free & resting & (F["dxy"] < cfg.keep_out_r) & (F["tcp"][:, 2] < cfg.keep_out_z)


def success_now(F, cfg: TaskConfig, goal_xy, goal_yaw, lifted, tol_xy=None, tol_yaw=None):
    """Geometric success [B] bool (spec_v2 B2)."""
    ex, ey = goal_errors(F, goal_xy, goal_yaw)
    tol_xy = cfg.tol_xy if tol_xy is None else tol_xy
    tol_yaw = cfg.tol_yaw if tol_yaw is None else tol_yaw
    return (lifted & ~F["held"] & F["on_table"] & (F["tilt"] <= cfg.flat_tilt) & (F["speed"] < cfg.rest_speed)
            & (ex <= tol_xy) & (ey <= tol_yaw) & (F["jaw"] >= cfg.open_jaw)
            & (F["tcp"][:, 2] >= F["c"][:, 2] + CUBE_HALF + cfg.tcp_clear))


def potential(F, cfg: TaskConfig, goal_xy, goal_yaw):
    """(d_reach [B], d_goal [B]) of the progress shaping (m)."""
    ex, ey = goal_errors(F, goal_xy, goal_yaw)
    d_goal = ex.clamp_min(cfg.sat_xy) + cfg.k_yaw * ey.clamp_min(cfg.sat_yaw)
    d_reach = (F["dxy"] + (F["tcp"][:, 2] - (F["c"][:, 2] + cfg.pre_h)).abs() + cfg.k_yaw * F["yaw_mis"])
    return d_reach, d_goal


# ---- the task ------------------------------------------------------------------------------------------------------
@dataclass
class Tracker:
    """Per-world episode state of the task (tensors [B])."""
    t: torch.Tensor
    lifted: torch.Tensor          # a firm lift happened
    lift_run: torch.Tensor        # consecutive steps held with the lowest corner >= lift_z
    air_run: torch.Tensor         # consecutive steps held in the air
    unsup_run: torch.Tensor
    tilt_run: torch.Tensor
    slide_run: torch.Tensor
    since_release: torch.Tensor   # steps since the cube was last held (large = never)
    prev_held: torch.Tensor
    prev_macro: torch.Tensor
    d_reach: torch.Tensor
    d_goal: torch.Tensor
    end: torch.Tensor             # ENDS code (0 running)
    box_log: torch.Tensor         # a grasp-box violation was logged (log_only)
    anchor_xy: torch.Tensor       # cube centre / yaw at the last step the gripper or jaw was near it (or it was held
    anchor_yaw: torch.Tensor      #   or off the table): the moved_untouched rule (anchor_xy [B, 3]: x, y, z)
    success10: torch.Tensor       # latched at the end: the secondary success level (10 mm / 10 deg) also held
    end_xy: torch.Tensor          # latched at the end: xy error (m) and yaw error (rad) of the cube to the goal
    end_yaw: torch.Tensor


class PickPlaceTask:
    """Batched task state over B worlds on one device. ``reset(idx, state, goal_xy, goal_yaw)``;
    ``step(state, macro=None) -> (reward [B], terminated [B], truncated [B], info)`` for every world (worlds that have
    already ended keep their end code until reset; their reward is 0); ``observation(state, dec_obs, macro)``."""

    def __init__(self, num_envs, device="cpu", kin: SO101Kin | None = None, geom: TaskGeometry | None = None,
                 cfg: TaskConfig | None = None, overlap_rules=True):
        self.B, self.dev = int(num_envs), torch.device(device)
        self.cfg = cfg or TaskConfig()
        self.kin = kin or SO101Kin(self.dev)
        self.geom = geom if geom is not None else TaskGeometry.from_mujoco(self.dev, self.kin.dtype)
        self.overlap_rules = overlap_rules
        f, li = dict(dtype=self.kin.dtype, device=self.dev), dict(dtype=torch.long, device=self.dev)
        self.goal_xy = torch.zeros(self.B, 2, **f)
        self.goal_yaw = torch.zeros(self.B, **f)
        z = lambda **k: torch.zeros(self.B, **k)                                    # noqa: E731
        self.tr = Tracker(t=z(**li), lifted=z(dtype=torch.bool, device=self.dev), lift_run=z(**li), air_run=z(**li),
                          unsup_run=z(**li), tilt_run=z(**li), slide_run=z(**li), since_release=z(**li) + 10 ** 6,
                          prev_held=z(dtype=torch.bool, device=self.dev), prev_macro=z(**li), d_reach=z(**f),
                          d_goal=z(**f), end=z(**li), box_log=z(dtype=torch.bool, device=self.dev),
                          anchor_xy=torch.zeros(self.B, 3, **f), anchor_yaw=z(**f),
                          success10=z(dtype=torch.bool, device=self.dev), end_xy=z(**f) + math.nan,
                          end_yaw=z(**f) + math.nan)
        self.prev_jaw = torch.zeros(self.B, **f)        # jaw angle of the last step (geometric close start)
        self.grasp_yaw = torch.zeros(self.B, **f)       # cube yaw at the first firm lift (continuous reference)
        self.turned = torch.zeros(self.B, **f)          # cube yaw turned since the first firm lift (rad)
        self.F = None

    # -- episode control --
    def reset(self, idx, state: WorldState, goal_xy, goal_yaw):
        """Worlds idx (long [n] or bool [B]) start: state = their start states (WorldState [n]), goals [n, 2], [n]."""
        F = features(state, self.kin, None, self.cfg)
        gx = torch.as_tensor(goal_xy, dtype=self.kin.dtype, device=self.dev)
        gy = torch.as_tensor(goal_yaw, dtype=self.kin.dtype, device=self.dev)
        self.goal_xy[idx], self.goal_yaw[idx] = gx, gy
        dr, dg = potential(F, self.cfg, gx, gy)
        tr = self.tr
        for k in ("t", "lift_run", "air_run", "unsup_run", "tilt_run", "slide_run", "prev_macro", "end"):
            getattr(tr, k)[idx] = 0
        for k in ("lifted", "prev_held", "box_log", "success10"):
            getattr(tr, k)[idx] = False
        tr.end_xy[idx], tr.end_yaw[idx] = math.nan, math.nan
        tr.since_release[idx] = 10 ** 6
        tr.d_reach[idx], tr.d_goal[idx] = dr, dg
        tr.anchor_xy[idx], tr.anchor_yaw[idx] = F["c"], F["cyaw"]
        self.grasp_yaw[idx], self.turned[idx], self.prev_jaw[idx] = F["cyaw"], 0.0, F["jaw"]

    def set_goal(self, idx, goal_xy, goal_yaw):
        """Change the goal of worlds idx (C3-b); the shaping potential is re-based (no reward jump)."""
        self.goal_xy[idx] = torch.as_tensor(goal_xy, dtype=self.kin.dtype, device=self.dev)
        self.goal_yaw[idx] = torch.as_tensor(goal_yaw, dtype=self.kin.dtype, device=self.dev)
        if self.F is not None:
            _, dg = potential({k: v[idx] for k, v in self.F.items()}, self.cfg, self.goal_xy[idx], self.goal_yaw[idx])
            self.tr.d_goal[idx] = dg

    @torch.no_grad()
    def step(self, state: WorldState, macro=None, macro_goal=None):
        """Apply the rules to the states after one world step. macro: long [B] (MACROS codes) of the decoder for the
        step just taken (``DecoderV2.macro()``), or None (no macros); macro_goal: [B, 4] the decoder's planned end of
        the GRASP descent (TCP x, y, z, yaw; ``DecoderV2.macro_goal``), used by the grasp-box rule at the trigger
        (None: the measured TCP pose, horizontal offset and yaw only). Returns reward [B] float, terminated [B] bool
        (success or failure this step), truncated [B] bool (time-out this step), info dict: end (ENDS code, latched),
        success / fail / timeout (this step), flags [B, len(RULES)] (rules that fired this step, also log-only ones),
        xy_err / yaw_err (now), success10 / end_xy_err / end_yaw_err (latched at the end; nan while running), lifted,
        held, grasp_box_log, turned (cube yaw turned since the first firm lift), t (steps)."""
        cfg, tr = self.cfg, self.tr
        F = features(state, self.kin, self.geom, cfg, overlap=self.overlap_rules)
        self.F = F
        B, dev = self.B, self.dev
        live = tr.end == 0
        use_macros = macro is not None
        macro = torch.zeros(B, dtype=torch.long, device=dev) if macro is None else macro.to(dev).long()
        free = macro == 0
        in_grasp = (macro >= GRASP_MACROS[0]) & (macro <= GRASP_MACROS[-1])
        tr.t = tr.t + live.long()
        fire = {}
        fire["world_invalid"] = state.invalid.to(dev).bool() | ~F["finite"]
        fire["world_arm_table"] = state.arm_table.to(dev).bool()
        fire["lowest_point"] = F["lowest"] < cfg.lowest_min
        zero = torch.zeros(B, dtype=torch.bool, device=dev)
        fire["link_overlap"] = F["link_depth"] > cfg.link_depth if self.overlap_rules else zero
        fire["jaw_overlap"] = F["jaw_overlap"] > cfg.jaw_depth if self.overlap_rules else zero
        fire["base_zone"] = F["base_zone"]
        # grasp box: at the trigger and at the close start
        trig = (tr.prev_macro == 0) & (macro == MC["grasp_descend"])
        close = (tr.prev_macro != MC["grasp_close"]) & (macro == MC["grasp_close"])
        if not use_macros:   # no macros: geometric close start
            close = (F["jaw"] < cfg.close_jaw) & (self.prev_jaw >= cfg.close_jaw) & (F["xg"].norm(dim=-1) < 0.04)
        self.prev_jaw = F["jaw"].clone()
        trig_ok = trigger_box_ok(F, cfg) if macro_goal is None else goal_box_ok(F, cfg, macro_goal)
        fire["grasp_box"] = (trig & ~trig_ok) | (close & ~close_box_ok(F, cfg))
        fire["keep_out"] = keep_out(F, cfg, free)
        # support: the cube in the air without being held
        air_free = ~F["held"] & (F["bottom"] > cfg.release_z)
        falling = F["v"][:, 2] <= cfg.fall_vz
        fire["release_high"] = air_free & falling
        tr.unsup_run = torch.where(air_free & ~falling, tr.unsup_run + 1, 0)
        fire["unsupported"] = tr.unsup_run >= cfg.unsupported_steps
        tr.air_run = torch.where(F["held"] & (F["bottom"] > cfg.air_z), tr.air_run + 1, 0)
        fire["hold_cap"] = tr.air_run * CONTROL_DT > cfg.hold_cap_s + 1e-9
        tr.since_release = torch.where(F["held"], 0, tr.since_release + 1)
        tr.slide_run = torch.where(free & F["on_table"] & ~F["near_jaws"] & (F["vxy"] > cfg.slide_v)
                                   & (tr.since_release * CONTROL_DT > cfg.slide_grace_s + 1e-9), tr.slide_run + 1, 0)
        fire["cube_slide"] = tr.slide_run >= cfg.slide_steps
        tr.tilt_run = torch.where(F["on_table"] & ~F["held"] & (F["tilt"] > cfg.tilt_fail), tr.tilt_run + 1, 0)
        fire["cube_tilted"] = tr.tilt_run >= cfg.tilt_steps
        fire["arm_self"] = F["self_depth"] > -cfg.self_margin if cfg.self_rule else zero
        if cfg.ghost_rule:    # no touch, no motion: the cube on the table moved since a gripper / jaw point was near
            near = F["touch"] < cfg.touch_d
            anchor = near | F["held"] | ~F["on_table"]
            tr.anchor_xy = torch.where(anchor[:, None], F["c"], tr.anchor_xy)
            tr.anchor_yaw = torch.where(anchor, F["cyaw"], tr.anchor_yaw)
            dc = F["c"] - tr.anchor_xy
            moved = ((dc[:, :2].norm(dim=-1) > cfg.ghost_d) | (dc[:, 2].abs() > cfg.ghost_dz)
                     | (wrap90(F["cyaw"] - tr.anchor_yaw).abs() > cfg.ghost_yaw))
            fire["moved_untouched"] = F["on_table"] & ~near & moved
        else:
            fire["moved_untouched"] = zero
        # firm lift (success precondition)
        tr.lift_run = torch.where(F["held"] & (F["bottom"] >= cfg.lift_z), tr.lift_run + 1, 0)
        new_lift = ~tr.lifted & (tr.lift_run >= cfg.lift_steps) & live
        self.grasp_yaw = torch.where(new_lift, F["cyaw"], self.grasp_yaw)
        tr.lifted = tr.lifted | new_lift
        self.turned = torch.where(tr.lifted & F["held"], wrap90(F["cyaw"] - self.grasp_yaw), self.turned)
        # end codes (rule order = priority)
        fail = torch.zeros(B, dtype=torch.bool, device=dev)
        code = torch.zeros(B, dtype=torch.long, device=dev)
        flags = torch.stack([fire[r] for r in RULES], -1) & live[:, None]
        for r in RULES:
            if r in cfg.log_only:
                continue
            new = fire[r] & live & ~fail
            code = torch.where(new, EC[r], code)
            fail = fail | new
        tr.box_log = tr.box_log | (fire["grasp_box"] & live)
        ok15 = success_now(F, cfg, self.goal_xy, self.goal_yaw, tr.lifted)
        ok10 = success_now(F, cfg, self.goal_xy, self.goal_yaw, tr.lifted, cfg.tol2_xy, cfg.tol2_yaw)
        succ = ok15 & live & ~fail
        code = torch.where(succ, EC["success"], code)
        timeout = live & ~fail & ~succ & (tr.t >= cfg.horizon)
        code = torch.where(timeout, EC["timeout"], code)
        # reward
        d_reach, d_goal = potential(F, cfg, self.goal_xy, self.goal_yaw)
        d_reach = torch.where(in_grasp, tr.d_reach, d_reach)                        # frozen in a GRASP macro
        phi0 = -(cfg.w_reach * tr.d_reach * (~tr.lifted | new_lift) + cfg.w_goal * tr.d_goal)
        phi1 = -(cfg.w_reach * d_reach * ~tr.lifted + cfg.w_goal * d_goal)
        rew = phi1 - phi0 - cfg.time_cost + cfg.lift_bonus * new_lift + cfg.success_bonus * succ
        n_left = (cfg.horizon - tr.t).clamp_min(0).to(rew.dtype)
        charge = cfg.time_cost * (1 - cfg.gamma ** n_left) / (1 - cfg.gamma) if cfg.gamma < 1 else cfg.time_cost * n_left
        rew = rew - fail * (cfg.fail_cost + charge)
        rew = torch.where(live, rew, torch.zeros_like(rew))
        rew = torch.where(torch.isfinite(rew), rew, torch.full_like(rew, -cfg.fail_cost))
        tr.d_reach, tr.d_goal = torch.where(live, d_reach, tr.d_reach), torch.where(live, d_goal, tr.d_goal)
        tr.end = torch.where(live, code, tr.end)
        tr.prev_macro, tr.prev_held = macro, F["held"]
        ex, ey = goal_errors(F, self.goal_xy, self.goal_yaw)
        ended = live & (code != 0)
        tr.success10 = tr.success10 | (succ & ok10)
        tr.end_xy = torch.where(ended, ex, tr.end_xy)
        tr.end_yaw = torch.where(ended, ey, tr.end_yaw)
        info = dict(end=tr.end.clone(), success=succ, fail=fail, timeout=timeout, flags=flags, xy_err=ex, yaw_err=ey,
                    success10=tr.success10.clone(), end_xy_err=tr.end_xy.clone(), end_yaw_err=tr.end_yaw.clone(),
                    lifted=tr.lifted.clone(), held=F["held"], grasp_box_log=tr.box_log.clone(),
                    turned=self.turned.clone(), t=tr.t.clone())
        return rew, (fail | succ), timeout, info

    # -- observation --
    def observation(self, state: WorldState, dec_obs=None, macro=None, F=None):
        """Policy observation [B, D] float32 (module docstring). F: features of ``state`` if already computed."""
        F = F if F is not None else features(state, self.kin, None, self.cfg)
        return observation(F, self.goal_xy, self.goal_yaw, self.tr.lifted, self.tr.air_run, self.cfg, dec_obs, macro)


def observation(F, goal_xy, goal_yaw, lifted, air_run, cfg: TaskConfig, dec_obs=None, macro=None):
    """The observation as a pure function (see the module docstring); float32 [B, D]."""
    B = F["q"].shape[0]
    ex, _ = goal_errors(F, goal_xy, goal_yaw)
    rel_yaw = 4 * (F["cyaw"] - F["gyaw"])
    parts = [F["q"], F["qd"], F["tcp"],
             F["rg"][:, :, 0], F["rg"][:, :, 2], F["c"], torch.sin(4 * F["cyaw"])[:, None],
             torch.cos(4 * F["cyaw"])[:, None], F["tilt"][:, None]]
    if cfg.obs_rates:
        parts += [F["v"], F["w"]]
    parts += [F["xg"], torch.sin(rel_yaw)[:, None], torch.cos(rel_yaw)[:, None], goal_xy,
              torch.sin(4 * goal_yaw)[:, None], torch.cos(4 * goal_yaw)[:, None], goal_xy - F["c"][:, :2],
              wrap90(goal_yaw - F["cyaw"])[:, None], lifted.to(F["q"].dtype)[:, None],
              (air_run.to(F["q"].dtype) * CONTROL_DT / cfg.hold_cap_s).clamp(0, 1)[:, None]]
    m = torch.zeros(B, dtype=torch.long, device=F["q"].device) if macro is None else macro.long()
    parts.append(torch.nn.functional.one_hot(m.clamp(0, len(MACROS) - 1), len(MACROS)).to(F["q"].dtype))
    if dec_obs is not None:
        parts.append(dec_obs.to(F["q"].dtype))
    return torch.cat(parts, -1).float()


def obs_dim(cfg: TaskConfig | None = None, dec_dim: int = 0) -> int:
    cfg = cfg or TaskConfig()
    return 6 + 6 + 3 + 6 + 3 + 3 + (6 if cfg.obs_rates else 0) + 3 + 2 + 2 + 2 + 2 + 1 + 2 + len(MACROS) + dec_dim


# ---- goals and starts ------------------------------------------------------------------------------------------------
def sample_annulus(n, gen=None, device="cpu", dtype=torch.float64):
    """n points (x, y) [n, 2] of the workspace annulus by the collector's sample_cube rule (r and angle uniform)."""
    u = torch.rand(n, 2, generator=gen, device=device, dtype=dtype)
    r = ANNULUS_R[0] + (ANNULUS_R[1] - ANNULUS_R[0]) * u[:, 0]
    a = torch.deg2rad(ANNULUS_DEG[0] + (ANNULUS_DEG[1] - ANNULUS_DEG[0]) * u[:, 1])
    return torch.stack((PAN_AXIS_XY[0] + r * torch.cos(a), PAN_AXIS_XY[1] + r * torch.sin(a)), -1)


def sample_goals(cube_xy, cube_yaw, gen=None, turned=None, dist=GOAL_DIST, turn_deg=GOAL_TURN_DEG, max_turn=MAX_TURN,
                 tries=64):
    """Goals for cubes at cube_xy [B, 2] with yaw cube_yaw [B] (mod 90): goal xy on the annulus at distance dist from
    the cube (rejection sampling, ``tries`` candidates per round), goal yaw = cube yaw + U(lo, hi) mod 90 with
    [lo, hi] = [-turn, turn] intersected with [-max_turn - turned, max_turn - turned] (turned [B]: the cube turn made
    since the grasp, rad; None = 0). -> (goal_xy [B, 2], goal_yaw [B] in [0, pi/2)). Draws on the generator's device."""
    dev = cube_xy.device if gen is None else gen.device
    dt = cube_xy.dtype
    xy0 = cube_xy.to(dev)
    B = xy0.shape[0]
    goal = torch.full((B, 2), math.nan, dtype=dt, device=dev)
    todo = torch.ones(B, dtype=torch.bool, device=dev)
    for _ in range(100):
        cand = sample_annulus(B * tries, gen, dev, dt).reshape(B, tries, 2)
        d = (cand - xy0[:, None]).norm(dim=-1)
        ok = (d >= dist[0]) & (d <= dist[1])
        first = ok.float().argmax(-1)
        pick = cand[torch.arange(B, device=dev), first]
        new = todo & ok.any(-1)
        goal = torch.where(new[:, None], pick, goal)
        todo = todo & ~new
        if not bool(todo.any()):
            break
    else:
        raise RuntimeError("no goal on the annulus at the requested distance")
    t0 = torch.zeros(B, dtype=dt, device=dev) if turned is None else turned.to(dev, dt)
    lo = torch.clamp(-max_turn - t0, min=-math.radians(turn_deg))
    hi = torch.clamp(max_turn - t0, max=math.radians(turn_deg))
    u = torch.rand(B, generator=gen, device=dev, dtype=dt)
    yaw = torch.remainder(cube_yaw.to(dev, dt) + lo + (hi - lo) * u, QUARTER)
    return goal.to(cube_xy.device), yaw.to(cube_xy.device)


def load_start_bank(data_dir, split=0, families=TASK_FAMILIES, device="cpu", limit=None):
    """First recorded states of the task-family episodes of one split of a grasp dataset (unified_data.npz) ->
    dict of float64 tensors q [N, 6], qd [N, 6] (the recorded rates; zero at a resting start), cube_pos [N, 3],
    cube_quat [N, 4], cube_v, cube_w [N, 3], plus rows (long [N], episode rows) and family. NEVER the sealed test."""
    import numpy as np

    from nedm.contact_nrd.rollouts import npz_members
    if "test_sealed" in str(data_dir):
        raise ValueError("the sealed test data must not be used for task starts")
    f = npz_members(Path(data_dir) / "unified_data.npz")
    fam, sp = np.asarray(f["family"]), np.asarray(f["splits"])
    rows = np.flatnonzero((sp == split) & np.isin(fam, list(families)))
    if limit is not None:
        rows = rows[:limit]
    arm = np.asarray(f["arm"][:, 0])[rows].astype(np.float64)
    cube = np.asarray(f["cube"][:, 0])[rows].astype(np.float64)
    t = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)            # noqa: E731
    return dict(q=t(arm[:, :6]), qd=t(arm[:, 6:]), cube_pos=t(cube[:, :3]), cube_quat=t(cube[:, 3:7]),
                cube_v=t(cube[:, 7:10]), cube_w=t(cube[:, 10:13]),
                rows=torch.as_tensor(rows, dtype=torch.long, device=device),
                family=torch.as_tensor(fam[rows].astype(np.int64), device=device))


def make_tasks(bank: dict, n, gen=None):
    """n tasks from a start bank: (start index [n] into the bank, goal_xy [n, 2], goal_yaw [n])."""
    N = bank["q"].shape[0]
    dev = bank["q"].device if gen is None else gen.device
    i = torch.randint(N, (n,), generator=gen, device=dev).to(bank["q"].device)
    yaw, _ = cube_yaw_tilt(bank["cube_quat"][i])
    gxy, gyaw = sample_goals(bank["cube_pos"][i, :2], yaw, gen)
    return i, gxy, gyaw


def start_state(bank: dict, i) -> WorldState:
    """WorldState [n] of the bank starts i (at rest: qd, v, w as recorded; contact / flags False)."""
    n = len(i)
    dev = bank["q"].device
    fb = lambda k: torch.zeros(n, k, dtype=torch.bool, device=dev) if k else torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
    return WorldState(q=bank["q"][i], qd=bank["qd"][i], cube_pos=bank["cube_pos"][i], cube_quat=bank["cube_quat"][i],
                      cube_v=bank["cube_v"][i], cube_w=bank["cube_w"][i], contact=fb(3), arm_table=fb(0),
                      invalid=fb(0))



# ---- recorded data on the 20 ms grid ---------------------------------------------------------------------------------
def grid_from_records(arm, cube, stride=2, dt=CONTROL_DT):
    """Recorded collector states (arm [N, T, 12], cube [N, T, 13], 10 ms records) -> the WorldState fields on the
    20 ms control grid (records 0, stride, 2 stride, ...) with the data's step-average rates (``data.step_average_rates``:
    pose difference over one control step, quaternion as the world-frame rotation vector of q1 conj(q0); the first
    grid point keeps the recorded rates). -> dict of float64 tensors [N, J, ...]: q, qd, cube_pos, cube_quat, cube_v,
    cube_w."""
    arm = torch.as_tensor(arm, dtype=torch.float64)[:, ::stride]
    cube = torch.as_tensor(cube, dtype=torch.float64)[:, ::stride]
    q, pos, quat = arm[..., :6], cube[..., :3], cube[..., 3:7]
    qd, v, w = arm[..., 6:].clone(), cube[..., 7:10].clone(), cube[..., 10:13].clone()
    qd[:, 1:] = (q[:, 1:] - q[:, :-1]) / dt
    v[:, 1:] = (pos[:, 1:] - pos[:, :-1]) / dt
    q1, q0 = quat[:, 1:], quat[:, :-1]
    w0, v0 = q0[..., :1], -q0[..., 1:]
    w1, v1 = q1[..., :1], q1[..., 1:]
    ww = w1 * w0 - (v1 * v0).sum(-1, keepdim=True)
    vv = w1 * v0 + w0 * v1 + torch.cross(v1, v0, dim=-1)
    sign = torch.where(ww < 0, -1.0, 1.0)
    ww, vv = ww * sign, vv * sign
    n = vv.norm(dim=-1, keepdim=True)
    w[:, 1:] = torch.where(n > 1e-12, vv / n.clamp_min(1e-12) * 2 * torch.atan2(n, ww), 2 * vv) / dt
    return dict(q=q, qd=qd, cube_pos=pos, cube_quat=quat, cube_v=v, cube_w=w)


def grid_state(grid: dict, j, device=None) -> WorldState:
    """WorldState [N] of grid point j of ``grid_from_records`` (contact / flags False)."""
    g = {k: v[:, j].to(device) if device is not None else v[:, j] for k, v in grid.items()}
    n, dev = g["q"].shape[0], g["q"].device
    return WorldState(contact=torch.zeros(n, 3, dtype=torch.bool, device=dev),
                      arm_table=torch.zeros(n, dtype=torch.bool, device=dev),
                      invalid=torch.zeros(n, dtype=torch.bool, device=dev), **g)
