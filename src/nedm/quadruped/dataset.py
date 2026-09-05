"""Per-step record schema for Go2 episodes, mirroring the HMMWV CRM collector.

Field names are taken from nedm.hmmwv_data.BASE_FIELDS VERBATIM wherever the
quantity exists for a legged base, so DEFAULT_STATE_FIELDS (vel_body_x_mps,
vel_body_y_mps, roll_rad, pitch_rad, roll_rate_radps, ang_vel_body_y_radps,
yaw_rate_radps) selects the same seven columns here as it does on the vehicle
with no per-robot special-casing in the training package.

TWO PLACES WHERE THE MIRROR IS NOT LITERAL, both deliberate:

1. ACTIONS. The HMMWV's action is three driver channels; ours is the twelve
   joint position targets the policy emits each control step. Same role in
   (state, control) -> next state, different width. Emitted in MOTOR_NAMES order
   -- Chrono order, RR/RL/FR/FL -- because that is the order `act()` returns and
   `actuate()` consumes, and reordering here would put a permutation between the
   dataset and the thing that produced it.

2. PER-FOOT CHANNELS. The vehicle logs tire force plus spindle omega. Omega
   earns its column because vx < omega*R on deformable ground, so the PAIR
   (vx, omega) encodes wheel slip. Sinkage is not that quantity -- it is a
   terrain measurement, not a drivetrain state -- so it is logged in addition to
   a foot kinematic channel rather than in place of one. The actual omega
   analogue is foot_*_slip_mps: a foot in stance should be stationary in world
   frame, so horizontal foot motion under load IS slip.

   SLIP IS LOGGED UNGATED, ON PURPOSE. "Horizontal speed while loaded" needs a
   load threshold to define "loaded", and a threshold baked in here is
   unrecoverable -- every downstream consumer inherits a number chosen now, with
   no way to revisit it without recollecting. force_fz_n sits in the same row, so
   any gating rule is a downstream derivation over data that is already present.
   The HMMWV has the same shape: it logs spindle omega raw AND a derived
   slip_ratio, rather than only the gated form.

SINKAGE IS DEFINED ON BOTH TERRAINS, WHICH TOOK SOME CARE. The obvious
definition, "local SPH surface height minus undisturbed control patch", needs
particles and is therefore NaN for every flat episode -- half the dataset by
terrain condition. Sinkage here is instead foot depth below the UNDISTURBED
surface, which is a length on rigid ground too. The SPH surface statistic is
kept as a separate column, zero-filled on rigid because rigid ground does not
deform, which is a fact rather than a gap.

Likewise the force column: FSI body force on CRM, Chrono contact force on rigid.
Different mechanism, same physical quantity -- exactly the HMMWV's situation,
where tire force comes from the tire model on both terrains.

SINKAGE ZERO IS CALIBRATED ON RIGID AND OFFSET ON CRM. Measured on 8 s episodes,
loaded samples only (foot_*_force_fz_n > 40 N):

    flat  loaded sinkage  mean -0.006 m, max -0.000 m  -- foot origin sits ON the
                                                          ground, so the zero is right
    crm   loaded sinkage  mean -0.028 m, max -0.014 m  -- foot origin sits 28 mm
                                                          ABOVE the nominal surface

The reference is not the error: the undisturbed free surface measured from SPH
particle z is 0.1993 against a nominal soil_top of 0.2000, so the two agree to
0.7 mm. The 28 mm is a real FSI standoff -- about 1.4 particle spacings at the
0.02 m spacing used -- and it means THE FOOT NEVER PENETRATES THE BED in this
configuration; it rides on the coupling layer and the soil responds by deflecting
its surface. That is a better explanation of the millimetre-scale surface
displacement than "the effect is small", and it is why the two columns are not
redundant: sinkage carries the standoff, surface_disp carries the deflection.
"""

from __future__ import annotations

import math
from typing import Any

from .constants import FOOT_BODIES, MOTOR_NAMES

# HMMWV leg order, so a reader who knows tire_fl/fr/rl/rr reads these the same
# way. NOT the FOOT_BODIES order, which is FR/FL/RR/RL.
LEG_ORDER = ["fl", "fr", "rl", "rr"]
LEG_TO_FOOT_BODY = {
    "fl": "FL_foot",
    "fr": "FR_foot",
    "rl": "RL_foot",
    "rr": "RR_foot",
}

JOINT_ACTION_FIELDS = [
    f"joint_{name.removesuffix('_joint').lower()}_target_rad" for name in MOTOR_NAMES
]

# THE SECOND ACTION CANDIDATE, logged alongside the first rather than instead of
# it. Two defensible definitions of the NRD action exist and the choice is a
# training-time config decision, not a collection-time one:
#
#   the 12 JOINT TARGETS -- NRD models the robot, the policy stays external.
#     Direct analogue of the HMMWV's driver_steering/throttle/braking.
#   the 3 VELOCITY COMMANDS -- NRD models (robot + policy) as one plant, exactly
#     as the HMMWV's throttle acts through a powertrain rather than being a wheel
#     torque. A level-3 outer loop then issues commands, which is far more
#     tractable than learning to walk inside a learned model.
#
# Over-capture and select later, which is what the HMMWV pipeline does. Deciding
# now would be free to get wrong and expensive to undo.
COMMAND_ACTION_FIELDS = ["cmd_vx_mps", "cmd_vy_mps", "cmd_wz_radps"]

# Measured joint state, needed to roll an external walking policy forward inside a
# surrogate: its 45-D observation reads q and dq, and a model that predicts only
# body state cannot supply them.
#
# ORDERED BY MOTOR_NAMES (RR, RL, FR, FL), which is CHRONO ORDER -- deliberately,
# and NOT by LEG_ORDER. Four orderings for the same twelve joints exist here:
#
#   MOTOR_NAMES   RR RL FR FL   robot.joint_pos()/joint_vel(), robot.actuate(),
#                               and JOINT_ACTION_FIELDS -- i.e. the target columns
#   LEG_ORDER     fl fr rl rr   the foot_* columns
#   FOOT_BODIES   FR FL RR RL   the body-name list
#   imported      FL FR RL RR   the policy, reached via CHRONO_TO_IMPORTED and SIGN
#
# Chrono order is the only choice that makes q, dq and the previous action line up
# index-for-index, since the target columns already use it. Logging in any other
# order would need a permutation at write time AND a different one at read time,
# to produce columns that silently disagree with the actions beside them.
JOINT_STATE_FIELDS = (
    [f"joint_{name.removesuffix('_joint').lower()}_pos_rad" for name in MOTOR_NAMES]
    + [f"joint_{name.removesuffix('_joint').lower()}_vel_radps" for name in MOTOR_NAMES]
)

# CONVENTION: MOTOR_NAMES order (RR RL FR FL), matching the target and pos/vel
# columns above. The PD law already computes this every physics step and the
# caller threw it away; it cannot be reconstructed afterwards because it depends
# on the target and the state at the moment it was applied, at 400 Hz.
# Clamped to the URDF effort limits, i.e. the torque actually commanded.
JOINT_TORQUE_FIELDS = [
    f"joint_{name.removesuffix('_joint').lower()}_torque_nm" for name in MOTOR_NAMES
]

# CONVENTION: MOTOR_NAMES order. The policy's raw network output, BEFORE
# ACTION_SCALE, IMPORTED_DEFAULTS and the sign/permutation back to Chrono order.
# Logged because the transform is lossy to invert in the presence of clipping and
# because it is the quantity a fine-tuning objective would act on.
POLICY_RAW_ACTION_FIELDS = [
    f"policy_raw_{name.removesuffix('_joint').lower()}" for name in MOTOR_NAMES
]

# CONVENTION: LEG_ORDER (fl fr rl rr), matching the other foot_* columns.
# Tangential force is what traction and slip are made of and is NOT recoverable
# from fz. Contact is Chrono's own contact-container ground truth, not a force
# threshold -- the thing every hysteresis constant so far has been standing in for.
FOOT_VECTOR_FIELDS: list[str] = []
for _leg in LEG_ORDER:
    FOOT_VECTOR_FIELDS += [
        f"foot_{_leg}_force_fx_n", f"foot_{_leg}_force_fy_n",
        f"foot_{_leg}_pos_x_m", f"foot_{_leg}_pos_y_m", f"foot_{_leg}_pos_z_m",
        f"foot_{_leg}_vel_x_mps", f"foot_{_leg}_vel_y_mps", f"foot_{_leg}_vel_z_mps",
        f"foot_{_leg}_in_contact",
    ]

# THREE DISTINCT FRAMES, all logged, because two of them have already been
# confused once: the base COG (pos_x_m et al above), the base REFERENCE frame,
# and the whole-robot centre of mass over all 42 bodies. Measured separation at
# stand: 24 mm COM-to-COG in x, 21 mm COG-to-REF.
# CONVENTION: world frame, applied at the base COG. Zero when no perturbation is
# active. LOGGED, NOT IMPLICIT: an unlogged disturbance is an unexplained
# acceleration -- it widens coverage and makes the data unlearnable at the same
# time. Logging it keeps "is this a model input?" a modelling choice rather than
# a collection one.
PERTURB_FIELDS = [
    "perturb_force_x_n", "perturb_force_y_n", "perturb_force_z_n",
    "perturb_torque_x_nm", "perturb_torque_y_nm", "perturb_torque_z_nm",
]

# THE SAME ARGUMENT AS PERTURB_FIELDS ABOVE, APPLIED TO A DISTURBANCE WE MISSED.
# Terrain slope is implemented by ROTATING GRAVITY on flat ground, so a tilted
# episode carries a horizontal acceleration -- 0.51 m/s^2 at 3 deg -- that nothing
# in the state records. Measured on 1,762 episodes, the applied pitch is NOT
# recoverable from the logged body attitude: corr(applied pitch, mean pitch_rad)
# = -0.030. That is an unlogged disturbance, which is exactly what the comment
# above says makes data unlearnable.
#
# WORLD FRAME, m/s^2, AS SET rather than as reconstructed. A gravity direction
# derived post-hoc from the quaternion computes R^T . [0,0,-1], the world-z axis
# in body frame, which equals gravity only on level ground -- on a tilted episode
# it is wrong by exactly the tilt, in the direction of asserting the ground is
# level. Body-frame gravity is derivable from these three plus the quaternion;
# the reverse is not true, so the primitive is what gets logged.
GRAVITY_FIELDS = [
    "grav_world_x_mps2", "grav_world_y_mps2", "grav_world_z_mps2",
]

BODY_FRAME_FIELDS = [
    # pos_x_m/pos_y_m/pos_z_m above are ALREADY the REF frame (capture_row reads
    # GetFrameRefToAbs). What was missing is the COG, which is what GetPos()
    # returns and what the boundary check used -- the 20.7 mm disagreement that
    # cost an afternoon was between these two.
    "base_cog_x_m", "base_cog_y_m", "base_cog_z_m",
    "com_x_m", "com_y_m", "com_z_m",
    "com_vel_x_mps", "com_vel_y_mps", "com_vel_z_mps",
]

ACTION_FIELDS = JOINT_ACTION_FIELDS + COMMAND_ACTION_FIELDS

BASE_FIELDS = [
    "episode_id",
    "scenario_name",
    "scenario_family",
    "split",
    "sample_index",
    "time_s",
    *ACTION_FIELDS,
    "pos_x_m",
    "pos_y_m",
    "pos_z_m",
    "quat_e0",
    "quat_e1",
    "quat_e2",
    "quat_e3",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    "vel_world_x_mps",
    "vel_world_y_mps",
    "vel_world_z_mps",
    "vel_body_x_mps",
    "vel_body_y_mps",
    "vel_body_z_mps",
    "acc_world_x_mps2",
    "acc_world_y_mps2",
    "acc_world_z_mps2",
    "acc_body_x_mps2",
    "acc_body_y_mps2",
    "acc_body_z_mps2",
    "ang_vel_world_x_radps",
    "ang_vel_world_y_radps",
    "ang_vel_world_z_radps",
    "ang_vel_body_x_radps",
    "ang_vel_body_y_radps",
    "ang_vel_body_z_radps",
    "speed_mps",
    "body_slip_rad",
    "roll_rate_radps",
    "yaw_rate_radps",
    *JOINT_STATE_FIELDS,
    *JOINT_TORQUE_FIELDS,
    *POLICY_RAW_ACTION_FIELDS,
    *BODY_FRAME_FIELDS,
    *PERTURB_FIELDS,
    *GRAVITY_FIELDS,
]


def foot_field_names() -> list[str]:
    fields: list[str] = []
    for leg in LEG_ORDER:
        fields.extend(
            [
                f"foot_{leg}_force_fz_n",
                f"foot_{leg}_force_fx_n",
                f"foot_{leg}_force_fy_n",
                f"foot_{leg}_pos_x_m",
                f"foot_{leg}_pos_y_m",
                f"foot_{leg}_pos_z_m",
                f"foot_{leg}_vel_x_mps",
                f"foot_{leg}_vel_y_mps",
                f"foot_{leg}_vel_z_mps",
                f"foot_{leg}_in_contact",
                f"foot_{leg}_slip_mps",
                f"foot_{leg}_sinkage_m",
                f"foot_{leg}_surface_disp_m",
            ]
        )
    return fields


def csv_field_names(include_feet: bool = True) -> list[str]:
    fields = list(BASE_FIELDS)
    if include_feet:
        fields.extend(foot_field_names())
    return fields


# Schmitt-trigger bounds for contact-mode extraction, in newtons. Measured, not
# chosen: see contact_mode.
CONTACT_RELEASE_N = 5.0
CONTACT_ENGAGE_N = 60.0


def contact_mode(force_fz, release_n: float = CONTACT_RELEASE_N,
                 engage_n: float = CONTACT_ENGAGE_N):
    """Per-foot stance booleans and the packed 4-bit mode, from a force series.

    A DERIVATION, NOT A COLUMN, and deliberately so. Contact mode is a pure
    function of foot_*_force_fz_n, which is already in every row, so it can be
    recomputed at any time with different bounds. Writing it into the CSV would
    freeze two tuned constants into the dataset where no consumer could revisit
    them without recollecting -- the same argument that keeps slip ungated.

    A PLAIN THRESHOLD IS NOT ADEQUATE ON CRM, which is why this is hysteretic.
    Measured on 8 s episodes, comparing the detected switch rate against the
    rate implied by the gait's own spectral peak (2 transitions per foot per
    cycle), where 1.0x means every detected transition is a real one:

        threshold 20 N     flat 1.02x ideal      crm 1.66x ideal
        schmitt 5/60 N     flat 1.01x ideal      crm 0.96x ideal

    On rigid the foot force is EXACTLY 0.0 for 42% of samples, so any threshold
    in a wide band works and the mode is trivially separable. On CRM it never
    reaches zero -- the foot rides on the coupling layer and always feels the
    kernel, which is the standoff documented above -- so the force histogram has
    no gap, decaying monotonically from its peak with no local minimum. A single
    threshold cuts through that noise and invents 66% spurious transitions.

    The gait structure is real on both terrains; only its SEPARABILITY BY FORCE
    MAGNITUDE differs. Hysteresis recovers it because the mode is temporally
    persistent even where it is not amplitude-separable.
    """
    import numpy as np

    f = np.asarray(force_fz, dtype=float)
    if f.ndim == 1:
        f = f[:, None]
    stance = np.zeros(f.shape, dtype=bool)
    for j in range(f.shape[1]):
        on = False
        for i in range(f.shape[0]):
            if on and f[i, j] < release_n:
                on = False
            elif not on and f[i, j] > engage_n:
                on = True
            stance[i, j] = on
    # fl fr rl rr -> bit 3..0, matching LEG_ORDER
    mode = np.zeros(f.shape[0], dtype=int)
    for k in range(min(4, f.shape[1])):
        mode |= stance[:, k].astype(int) << (3 - k)
    return stance, mode


def contact_bodies(chrono, system) -> set:
    """Names of bodies Chrono's contact container resolved a contact for.

    GROUND TRUTH for foot contact, replacing a force threshold. Every hysteresis
    constant tuned so far has been a proxy for this, tuned because it was never
    logged. Empty on CRM, where feet couple through FSI and the contact system
    sees nothing -- the caller passes None there so the column reads NaN rather
    than a confident False.
    """
    class _Rep(chrono.ReportContactCallback):
        def __init__(self):
            super().__init__()
            self.names = set()

        def OnReportContact(self, pA, pB, plane, distance, eff_radius,
                            react_forces, react_torques, cA, cB, offset):
            for c in (cA, cB):
                try:
                    self.names.add(chrono.CastToChBody(c).GetName())
                except Exception:  # noqa: BLE001
                    pass
            return True

    rep = _Rep()
    system.GetContactContainer().ReportAllContacts(rep)
    return rep.names


def whole_robot_com(system):
    """Mass-weighted centre of mass and its velocity over every non-ground body.

    Distinct from the base COG and from the base REF frame: measured separation
    at stand is 24 mm COM-to-COG in x. Not derivable from any logged column,
    because the per-link states are not logged.
    """
    bodies = [b for b in system.GetBodies() if b.GetName() != "ground"]
    m = sum(b.GetMass() for b in bodies)
    if not bodies or m <= 0:
        return (float("nan"),) * 3, (float("nan"),) * 3
    px = sum(b.GetMass() * b.GetPos().x for b in bodies) / m
    py = sum(b.GetMass() * b.GetPos().y for b in bodies) / m
    pz = sum(b.GetMass() * b.GetPos().z for b in bodies) / m
    vx = sum(b.GetMass() * b.GetPosDt().x for b in bodies) / m
    vy = sum(b.GetMass() * b.GetPosDt().y for b in bodies) / m
    vz = sum(b.GetMass() * b.GetPosDt().z for b in bodies) / m
    return (px, py, pz), (vx, vy, vz)


def _foot_force_xy(chrono, body, terrain):
    """Tangential contact force. Not reconstructable from fz, and it is what
    traction and slip are made of."""
    try:
        v = (terrain.GetFsiBodyForce(body) if terrain is not None
             else body.GetContactForce())
        return float(v.x), float(v.y)
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan")


def _foot_force_z(chrono, body, terrain) -> float:
    """Vertical force on a foot: FSI on CRM, rigid contact on flat.

    Two mechanisms because the two terrains couple differently -- feet exchange
    momentum with soil through FSI body forces and with the rigid box through
    the contact system. GetContactForce is ~0 on CRM (nothing touches the feet
    in the contact sense) and GetFsiBodyForce does not exist without a terrain,
    so neither alone spans the dataset.
    """
    if terrain is not None:
        try:
            return float(terrain.GetFsiBodyForce(body).z)
        except Exception:  # noqa: BLE001
            return float("nan")
    try:
        return float(body.GetContactForce().z)
    except Exception:  # noqa: BLE001
        return float("nan")


def capture_row(
    chrono,
    robot: Any,
    terrain: Any,
    soil_top_m: float,
    action: Any,
    command: Any,
    soil_z: Any,
    soil_ctrl: float,
    scenario_name: str,
    scenario_family: str,
    episode_id: str,
    split: str,
    sample_index: int,
    time_s: float,
    tau: Any = None,
    policy_raw: Any = None,
    perturb: Any = None,
    gravity: Any = None,
    contacts: Any = None,
    com: Any = None,
) -> dict[str, Any]:
    """One CSV row: base state in HMMWV field names, plus actions and feet.

    `soil_z` / `soil_ctrl` come from soilprobe.sample in FOOT_BODIES order and
    are NaN off CRM; the surface-displacement column is zero-filled rather than
    NaN in that case, since rigid ground genuinely does not move.
    """
    base = robot.base()
    ref = base.GetFrameRefToAbs()

    pos = ref.GetPos()
    quat = ref.GetRot()
    euler_zyx = quat.GetCardanAnglesZYX()
    vel_world = ref.GetPosDt()
    vel_body = ref.TransformDirectionParentToLocal(vel_world)
    acc_world = base.GetPosDt2()
    acc_body = ref.TransformDirectionParentToLocal(acc_world)
    ang_world = ref.GetAngVelParent()
    ang_body = ref.GetAngVelLocal()

    row: dict[str, Any] = {
        "episode_id": episode_id,
        "scenario_name": scenario_name,
        "scenario_family": scenario_family,
        "split": split,
        "sample_index": sample_index,
        "time_s": float(time_s),
        "pos_x_m": float(pos.x),
        "pos_y_m": float(pos.y),
        "pos_z_m": float(pos.z),
        "quat_e0": float(quat.e0),
        "quat_e1": float(quat.e1),
        "quat_e2": float(quat.e2),
        "quat_e3": float(quat.e3),
        # ChVehicle supplies GetRoll/GetPitch/GetSpeed/GetSlipAngle/GetRollRate/
        # GetYawRate; a URDF body has no such helpers, so these are the same
        # quantities taken from the base frame directly. Cardan ZYX, matching
        # what capture_row uses for yaw on the vehicle side.
        "roll_rad": float(euler_zyx.x),
        "pitch_rad": float(euler_zyx.y),
        "yaw_rad": float(euler_zyx.z),
        "vel_world_x_mps": float(vel_world.x),
        "vel_world_y_mps": float(vel_world.y),
        "vel_world_z_mps": float(vel_world.z),
        "vel_body_x_mps": float(vel_body.x),
        "vel_body_y_mps": float(vel_body.y),
        "vel_body_z_mps": float(vel_body.z),
        "acc_world_x_mps2": float(acc_world.x),
        "acc_world_y_mps2": float(acc_world.y),
        "acc_world_z_mps2": float(acc_world.z),
        "acc_body_x_mps2": float(acc_body.x),
        "acc_body_y_mps2": float(acc_body.y),
        "acc_body_z_mps2": float(acc_body.z),
        "ang_vel_world_x_radps": float(ang_world.x),
        "ang_vel_world_y_radps": float(ang_world.y),
        "ang_vel_world_z_radps": float(ang_world.z),
        "ang_vel_body_x_radps": float(ang_body.x),
        "ang_vel_body_y_radps": float(ang_body.y),
        "ang_vel_body_z_radps": float(ang_body.z),
        "speed_mps": float(vel_world.Length()),
        "body_slip_rad": float(math.atan2(vel_body.y, vel_body.x)),
        "roll_rate_radps": float(ang_body.x),
        "yaw_rate_radps": float(ang_body.z),
    }

    for field, value in zip(JOINT_ACTION_FIELDS, action):
        row[field] = float(value)
    # Raw Chrono-order values, NOT sign-flipped and NOT permuted. The imported
    # policy applies SIGN and CHRONO_TO_IMPORTED itself (imported_policy.py:224);
    # applying either here would bake one consumer's convention into the dataset
    # and silently break every other reader.
    joint_state = list(robot.joint_pos()) + list(robot.joint_vel())
    for field, value in zip(JOINT_STATE_FIELDS, joint_state):
        row[field] = float(value)
    for field, value in zip(COMMAND_ACTION_FIELDS, command):
        row[field] = float(value)

    # Chrono order throughout, matching the target and pos/vel columns.
    for field, value in zip(JOINT_TORQUE_FIELDS,
                            tau if tau is not None else [float("nan")] * 12):
        row[field] = float(value)
    for field, value in zip(POLICY_RAW_ACTION_FIELDS,
                            policy_raw if policy_raw is not None else [float("nan")] * 12):
        row[field] = float(value)

    # Three frames, none derivable from another.
    for field, value in zip(PERTURB_FIELDS,
                            perturb if perturb is not None else [0.0] * 6):
        row[field] = float(value)

    # Default is level gravity, so an episode that never sets it reads as level --
    # which is true for every untilted run and honest for the rest only because the
    # collector always passes it. If this default is ever silently relied on, the
    # column becomes the same reconstruction error it exists to prevent.
    for field, value in zip(GRAVITY_FIELDS,
                            gravity if gravity is not None else [0.0, 0.0, -9.81]):
        row[field] = float(value)

    cg = base.GetPos()          # COG; `pos_*` above is the REF frame
    row["base_cog_x_m"], row["base_cog_y_m"], row["base_cog_z_m"] = (
        float(cg.x), float(cg.y), float(cg.z))
    cpos, cvel = (com if com is not None else ((float("nan"),) * 3, (float("nan"),) * 3))
    row["com_x_m"], row["com_y_m"], row["com_z_m"] = [float(v) for v in cpos]
    row["com_vel_x_mps"], row["com_vel_y_mps"], row["com_vel_z_mps"] = [float(v) for v in cvel]

    probe = {name: z for name, z in zip(FOOT_BODIES, soil_z)}
    for leg in LEG_ORDER:
        body_name = LEG_TO_FOOT_BODY[leg]
        body = robot.body(body_name)
        if body is None:
            for suffix in ("force_fz_n", "force_fx_n", "force_fy_n",
                           "pos_x_m", "pos_y_m", "pos_z_m",
                           "vel_x_mps", "vel_y_mps", "vel_z_mps",
                           "in_contact", "slip_mps", "sinkage_m", "surface_disp_m"):
                row[f"foot_{leg}_{suffix}"] = float("nan")
            continue
        foot_z = float(body.GetPos().z)
        vel = body.GetPosDt()
        row[f"foot_{leg}_force_fz_n"] = _foot_force_z(chrono, body, terrain)
        fxy = _foot_force_xy(chrono, body, terrain)
        row[f"foot_{leg}_force_fx_n"], row[f"foot_{leg}_force_fy_n"] = fxy
        fp = body.GetPos()
        row[f"foot_{leg}_pos_x_m"] = float(fp.x)
        row[f"foot_{leg}_pos_y_m"] = float(fp.y)
        row[f"foot_{leg}_pos_z_m"] = float(fp.z)
        row[f"foot_{leg}_vel_x_mps"] = float(vel.x)
        row[f"foot_{leg}_vel_y_mps"] = float(vel.y)
        row[f"foot_{leg}_vel_z_mps"] = float(vel.z)
        # GROUND TRUTH, not a threshold: membership in the set of bodies Chrono's
        # contact container actually resolved a contact for. NaN off rigid, where
        # feet couple through FSI and the contact system sees nothing.
        row[f"foot_{leg}_in_contact"] = (
            float("nan") if contacts is None else float(body_name in contacts))
        # SLIP IS A MAGNITUDE and therefore carries no direction. Kept for
        # continuity with earlier datasets; the signed components are the
        # foot_*_vel_* columns above, which is what a directional diagnostic needs.
        row[f"foot_{leg}_slip_mps"] = float(math.hypot(vel.x, vel.y))
        # Depth of the foot below the UNDISTURBED surface. Positive is buried.
        # A length on rigid ground too, which the SPH-difference definition is
        # not, which is the whole reason this is the column called sinkage.
        row[f"foot_{leg}_sinkage_m"] = float(soil_top_m) - foot_z
        local = probe.get(body_name, float("nan"))
        if terrain is None:
            disp = 0.0
        elif math.isnan(local) or math.isnan(soil_ctrl):
            disp = float("nan")
        else:
            disp = float(local) - float(soil_ctrl)
        row[f"foot_{leg}_surface_disp_m"] = disp

    return row
