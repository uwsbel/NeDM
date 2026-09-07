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
    # 34-D. Adds the three projected-gravity components, computed from the stored
    # quaternion rather than reconstructed from roll and pitch. The reconstruction
    # carried 0.0288 mean error per component against components of order 1, and
    # the error scales with TILT -- so it was worst during stumbles and falls,
    # which is the regime a low-command fine-tune most depends on. This is the
    # channel telling the policy which way is down, and the policy consumes it
    # directly, so the surrogate now predicts the quantity that is used rather
    # than one it is derived from.
    "quadruped_joint_grav": (DEFAULT_STATE_FIELDS + QUADRUPED_JOINT_STATE_FIELDS
                             + ["grav_body_x", "grav_body_y", "grav_body_z"]),
    # BASE HEIGHT AND VERTICAL VELOCITY, added 2026-09-05 for a measured reason.
    #
    # The imported policy's own training objective is dominated by
    # `correct_base_height`, whose CONVERGED curriculum weight is -10.0 -- five
    # times the next-largest penalty (`dof_pos_limits` at -2.0). The 34-D
    # quadruped_joint_grav state cannot express it: there is no pos_z_m channel.
    #
    # So a policy fine-tuned inside that surrogate has no incentive to hold body
    # height, and the first fine-tune fell in 43 of 43 Chrono episodes at a median
    # of 1.52 s -- a height and posture failure, in a policy with no height term.
    #
    # This is the general point rather than a detail of one run: the reduced state
    # was chosen by asking what the DYNAMICS depend on, and pose is the canonical
    # thing that criterion excludes. That can omit what the OBJECTIVE depends on.
    # Excluding pose does not break the dynamics here; it breaks the control problem.
    "quadruped_joint_grav_pose": (DEFAULT_STATE_FIELDS + QUADRUPED_JOINT_STATE_FIELDS
                                  + ["grav_body_x", "grav_body_y", "grav_body_z"]
                                  + ["pos_z_m", "vel_body_z_mps"]),
    # CONTACT-CONDITIONED. The surrogate fits a function with repeated discontinuities --
    # four feet making and breaking contact several times per gait cycle -- and a single
    # smooth transition model can only average across them. Putting the contact
    # indicators IN THE STATE makes the mode both predicted and conditioned on, which is
    # the requirement: conditioning alone is insufficient because in a rollout the future
    # mode is unknown.
    #
    # The indicators are binary but this is a DELTA model, so predicted contact is
    # prev + delta and can leave [0,1]. That is a relaxation, not a mode classifier.
    # OBSERVED GRAVITY. Every preset above carries grav_body_*, which
    # add_gravity_channels.py computes from the stored QUATERNION as R^T.[0,0,-1].
    # That is the body-frame direction of world -Z, not of gravity, and the two
    # differ by exactly the ground tilt -- so on a tilted episode the channel
    # asserts the ground is level. Measured on 1,762 episodes: the applied pitch is
    # NOT recoverable from the logged attitude, corr = -0.030.
    #
    # Tilt is randomised per episode across the corpus, so it is an unobserved
    # latent folded into the residual. grav_world_*_mps2 is the gravity vector AS
    # SET, and with the quaternion the model can form the true body-frame gravity;
    # the reverse is not possible, which is why the primitive is what gets logged.
    #
    # NeRD's ablation is the outside evidence: replacing the robot-centric frame
    # with a world frame costs 23.3x on Ant, their floating-base walker, and
    # nothing (1.1x) on a base-fixed pendulum. A quadruped that walks out of its
    # training region is the case where the frame matters most.
    #
    # REQUIRES A CORPUS COLLECTED AFTER grav_world_* WAS ADDED TO THE LOGGER.
    # go2_comprehensive_merged/flat (2026-09-04) does NOT carry these columns.
    "quadruped_joint_gravworld_pose": (DEFAULT_STATE_FIELDS + QUADRUPED_JOINT_STATE_FIELDS
                                       + ["grav_body_x", "grav_body_y", "grav_body_z"]
                                       + ["pos_z_m", "vel_body_z_mps"]
                                       + ["grav_world_x_mps2", "grav_world_y_mps2",
                                          "grav_world_z_mps2"]),
    # THE DIMENSIONALITY CONTROL for quadruped_joint_gravworld_pose.
    #
    # That preset adds THREE channels to grav_pose, so a gain over it is
    # attributable to "more inputs" as much as to "tilt is now observable" -- the
    # two-variables problem, and the reason go2_mix34_base_replicate was dropped
    # as a counterfactual earlier.
    #
    # This arm has the same 39 channels and the same three EXTRA columns, but they
    # are grav_world SHUFFLED ACROSS EPISODES: identical marginal distribution,
    # identical scale, identical dimensionality, and no valid correspondence to the
    # episode they sit in. A permutation control rather than a constant, because
    # constants have zero variance and the input normalisation would treat them
    # differently from a real channel.
    #
    #   B - A  =  information + dimensionality
    #   C - A  =  dimensionality alone
    #   B - C  =  tilt observability, which is the claim
    #
    # Built by scripts/ablations/shuffle_gravworld.py, which writes the shuffled
    # columns under these names so the preset selects them by name like any other.
    "quadruped_joint_gravshuf_pose": (DEFAULT_STATE_FIELDS + QUADRUPED_JOINT_STATE_FIELDS
                                      + ["grav_body_x", "grav_body_y", "grav_body_z"]
                                      + ["pos_z_m", "vel_body_z_mps"]
                                      + ["grav_shuf_x_mps2", "grav_shuf_y_mps2",
                                         "grav_shuf_z_mps2"]),
    "quadruped_contact_conditioned": (DEFAULT_STATE_FIELDS + QUADRUPED_JOINT_STATE_FIELDS
                                      + ["grav_body_x", "grav_body_y", "grav_body_z"]
                                      + ["pos_z_m", "vel_body_z_mps"]
                                      + ["foot_fl_in_contact", "foot_fr_in_contact",
                                         "foot_rl_in_contact", "foot_rr_in_contact"]),
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
