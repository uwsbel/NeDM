from __future__ import annotations

# The collector owns these names; importing keeps one source of truth for the
# column ordering. quadruped.dataset is pure stdlib and pulls in no simulator.
from nedm.quadruped.dataset import (
    COMMAND_ACTION_FIELDS as _COMMAND_ACTION_FIELDS,
    JOINT_ACTION_FIELDS as _JOINT_ACTION_FIELDS,
    JOINT_STATE_FIELDS as _JOINT_STATE_FIELDS,
)

DEFAULT_STATE_FIELDS = [
    "vel_body_x_mps",
    "vel_body_y_mps",
    "roll_rad",
    "pitch_rad",
    "roll_rate_radps",
    "ang_vel_body_y_radps",
    "yaw_rate_radps",
]

TIRE_FORCE_OMEGA_STATE_FIELDS = [
    "tire_fl_force_wheel_fx_n",
    "tire_fl_force_wheel_fy_n",
    "tire_fl_force_wheel_fz_n",
    "tire_fr_force_wheel_fx_n",
    "tire_fr_force_wheel_fy_n",
    "tire_fr_force_wheel_fz_n",
    "tire_rl_force_wheel_fx_n",
    "tire_rl_force_wheel_fy_n",
    "tire_rl_force_wheel_fz_n",
    "tire_rr_force_wheel_fx_n",
    "tire_rr_force_wheel_fy_n",
    "tire_rr_force_wheel_fz_n",
    "tire_fl_spindle_omega_radps",
    "tire_fr_spindle_omega_radps",
    "tire_rl_spindle_omega_radps",
    "tire_rr_spindle_omega_radps",
]

TIRE_NORMAL_FORCE_OMEGA_STATE_FIELDS = [
    "tire_fl_force_wheel_fz_n",
    "tire_fr_force_wheel_fz_n",
    "tire_rl_force_wheel_fz_n",
    "tire_rr_force_wheel_fz_n",
    "tire_fl_spindle_omega_radps",
    "tire_fr_spindle_omega_radps",
    "tire_rl_spindle_omega_radps",
    "tire_rr_spindle_omega_radps",
]

# --- Quadruped (Study 4) -----------------------------------------------------
# Mirrors the HMMWV split: a body-motion base plus a per-contact-patch block.
# The seven body fields are IDENTICAL to DEFAULT_STATE_FIELDS by construction, so
# the same preset selects the same physical quantities on either vehicle.
#
# The per-foot block is the analogue of the HMMWV's terramechanics block, and the
# analogy is exact only for the first two rows:
#   force_fz  <- tire_*_force_wheel_fz_n     vertical load, both terrains
#   slip      <- tire_*_spindle_omega_radps  paired with body velocity it exposes
#                                            SLIP, which is what earns omega its
#                                            place on the vehicle
#   sinkage       geometric foot depth. No wheeled analogue.
#   surface_disp  SPH surface response. No wheeled analogue, CRM-only, and a
#                 deletion candidate -- see quadruped-contact-mode.md, the foot
#                 floats above the bed so this reads 0.17-0.23 mm.
FOOT_NAMES = ["fl", "fr", "rl", "rr"]

QUADRUPED_FOOT_FORCE_FIELDS = [f"foot_{n}_force_fz_n" for n in FOOT_NAMES]
QUADRUPED_FOOT_SLIP_FIELDS = [f"foot_{n}_slip_mps" for n in FOOT_NAMES]
QUADRUPED_FOOT_SINKAGE_FIELDS = [f"foot_{n}_sinkage_m" for n in FOOT_NAMES]
QUADRUPED_FOOT_SURFACE_FIELDS = [f"foot_{n}_surface_disp_m" for n in FOOT_NAMES]

# 15-D, the direct counterpart of the HMMWV's 15-D state: 7 body + 8 contact.
QUADRUPED_CONTACT_STATE_FIELDS = (
    QUADRUPED_FOOT_FORCE_FIELDS + QUADRUPED_FOOT_SLIP_FIELDS
)

# 23-D, adding the two channels with no wheeled analogue. Justified by ablation
# rather than assertion -- the paper's rule is that a channel earns its place only
# if removing it degrades rollout fidelity.
QUADRUPED_FULL_FOOT_FIELDS = (
    QUADRUPED_CONTACT_STATE_FIELDS
    + QUADRUPED_FOOT_SINKAGE_FIELDS
    + QUADRUPED_FOOT_SURFACE_FIELDS
)

# The 3-D velocity command. The alternative action is the twelve joint targets;
# both are collected so the choice is made by ablation. See
# docs/state/decisions/quadruped-case-study-plan.md.
# IMPORTED FROM THE COLLECTOR'S OWN DEFINITIONS, not restated. These name CSV
# columns, so the collector is the only authority on what they are called, and a
# copy here can drift from the files it claims to describe. It already had:
# these were "cmd_vx"/"cmd_vy"/"cmd_wz" against the collector's
# "cmd_vx_mps"/"cmd_vy_mps"/"cmd_wz_radps" -- wrong, and invisible because
# nothing imported them. (Harmless if used: read_episode_csv raises KeyError on a
# missing field, so this would have failed loudly rather than silently.)
QUADRUPED_COMMAND_ACTION_FIELDS = list(_COMMAND_ACTION_FIELDS)
QUADRUPED_JOINT_ACTION_FIELDS = list(_JOINT_ACTION_FIELDS)

# 31-D. Body state plus the twelve measured joint positions and velocities, in
# the collector's Chrono order -- the state a surrogate must predict for an
# external walking policy to be rolled forward inside it, since the imported
# policy's 45-D observation reads q and dq. Body-only channels cannot supply them.
QUADRUPED_JOINT_STATE_FIELDS = list(_JOINT_STATE_FIELDS)

STATE_FIELD_PRESETS = {
    "default": DEFAULT_STATE_FIELDS,
    "tire_force_omega": DEFAULT_STATE_FIELDS + TIRE_FORCE_OMEGA_STATE_FIELDS,
    "tire_normal_force_omega": DEFAULT_STATE_FIELDS + TIRE_NORMAL_FORCE_OMEGA_STATE_FIELDS,
    # Study 4. quadruped_contact is the 15-D HMMWV counterpart; quadruped_full
    # adds sinkage and surface displacement as ablation candidates.
    "quadruped_contact": DEFAULT_STATE_FIELDS + QUADRUPED_CONTACT_STATE_FIELDS,
    "quadruped_full": DEFAULT_STATE_FIELDS + QUADRUPED_FULL_FOOT_FIELDS,
    # 31-D, for the joint-level surrogate the policy fine-tuning pilot needs.
    "quadruped_joint": DEFAULT_STATE_FIELDS + QUADRUPED_JOINT_STATE_FIELDS,
}

DEFAULT_ACTION_FIELDS = [
    "driver_steering",
    "driver_throttle",
    "driver_braking",
]

DEFAULT_ROLLOUT_FIELDS = [
    "pos_x_m",
    "pos_y_m",
    "yaw_rad",
]
