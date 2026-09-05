"""Constants and conventions for the Go2-on-CRM runs.

Split out of scripts/quadruped_go2_crm.py unchanged. The comments here are the
most valuable part of the module: the joint permutation and the sign negation
are conventions inherited from the training harness, and one of them nobody has
ever explained."""

from __future__ import annotations

import math
import os

import numpy as np


GRAVITY = 9.81

# PD actuator gains. NOT tuned -- these are what every legged_gym-family Go2
# config specifies, verified against both the Genesis locomotion example
# (kp 20.0, kd 0.5) and unitree_rl_gym (stiffness 20.0 N*m/rad, damping 0.5
# N*m*s/rad). Matching them is the whole point: a policy from that family assumes
# this actuator, so moving these to suit a particular checkpoint would defeat the
# reason for having them.
PD_KP, PD_KD = 20.0, 0.5

# Per-joint effort limits read from the URDF's <limit effort>, in MOTOR_NAMES
# order. Hip and thigh are 23.7 N*m; the calf is 45.43 N*m because the Go2 knee
# carries a reduction. They are NOT uniform, so a single clamp would either
# throttle the knee or let the hips exceed hardware.
JOINT_EFFORT_NM = np.array([23.7, 23.7, 45.43] * 4)

# Chrono joint order. The policy does not use this order; see CHRONO_TO_POLICY.
MOTOR_NAMES = [
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
]
FOOT_BODIES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
SOIL_PROBE_R = 0.05      # radius of the particle patch sampled under each foot
SOIL_CTRL_XY = (0.0, 1.2)  # undisturbed control patch, off the robot's path
EJECTA_BAND = 0.03       # ignore particles this far above the control surface
CALF_BODIES = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]

# Chrono [RR,RL,FR,FL] -> policy [FR,FL,RR,RL]. Swapping two halves of six is
# its own inverse, so this converts observations one way and actions the other.
# THE POLICY FRAME IS GENESIS'S, VERBATIM. Verified field by field against the
# Genesis Go2 locomotion example: joint order FR/FL/RR/RL, the split thigh
# defaults, all four obs scales (2.0 / 0.25 / 1.0 / 0.05), action_scale 0.25, and
# the 3+3+3+12+12+12 observation layout are identical. SBEL reimplemented that
# environment inside Chrono -- chrono_crmenv.py says so in its own comments
# ("Genesis default joint angles", "Reorder from Genesis [FR,FL,RR,RL]").
#
# Two separate things, and conflating them cost a correction in each direction:
#   the WEIGHTS were trained in Chrono, not imported  -- model_2999.pt is in-house
#   the CONVENTION is Genesis's, imported wholesale   -- these names were right
# So *_GENESIS_* was accurate about what it named. Kept as *_POLICY_* because the
# role is what a reader needs at the call site, with provenance recorded here.
CHRONO_TO_POLICY = np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5], dtype=np.int64)

# Policy-frame rest pose, in policy order [FR, FL, RR, RL]. Front and rear thigh
# defaults differ (0.8 vs 1.0); normalising that away breaks the stance.
POLICY_DEFAULTS = np.array([0.0, 0.8, -1.5, 0.0, 0.8, -1.5,
                             0.0, 1.0, -1.5, 0.0, 1.0, -1.5], dtype=np.float32)

# Chrono-order standing pose, held while the robot settles onto the soil.
STAND_ACTION = np.array([0.0, -1.0, 1.5, 0.0, -1.0, 1.5,
                         0.0, -0.8, 1.5, 0.0, -0.8, 1.5], dtype=np.float64)
# EXPERIMENTAL, off by default. The hips here are 0.0 while the imported policy's
# zero-action pose has them at +/-0.1 (upstream: FL/RL +0.1, FR/RR -0.1), so at the
# handover from the scripted stand to the policy the first action commands a 0.1 rad
# splay on all four hips at once. Falls cluster 0.14 s after the ramp ends, which is
# when that handover happens. Set NEDM_STAND_HIP=1 to stand in the policy's own pose
# and measure whether the fall rate changes. Chrono order RR, RL, FR, FL; the sign
# convention is inverted relative to the policy, hence the negation.
if os.environ.get("NEDM_STAND_HIP") == "1":
    STAND_ACTION = STAND_ACTION.copy()
    STAND_ACTION[[0, 3, 6, 9]] = [0.1, -0.1, 0.1, -0.1]

# ~63 deg from upright. A walking Go2 stays well inside this; a tumble
# crosses it decisively, so it separates gait roll from falling over.
FALL_TILT_RAD = math.radians(63.0)

LIN_VEL_SCALE, ANG_VEL_SCALE, DOF_POS_SCALE, DOF_VEL_SCALE = 2.0, 0.25, 1.0, 0.05

# Two soil presets, and which one you pick is a research decision.
#
# "eval" matches configs/hmmwv_crm_eval.json and demo_ROBOT_Viper_CRM.py. It is
# the soil the HMMWV work uses, and it is what this script ran until it was found
# to put a landing body into an undamped limit cycle.
#
# "training" is what chrono_crmenv.py actually used for the CRM policy finetune,
# with Young's modulus HALVED and cohesion cut to 40%, both commented "Reduced"
# in the source. Whoever wrote it evidently hit the same wall and solved it by
# softening the soil rather than by raising artificial dissipation. That is the
# physically honest fix: Young's modulus and cohesion are soil properties, and
# artificial_viscosity is a numerical damping term that changes the foot-soil
# interaction Case Study IV exists to measure.
# WORKING COMBINATION, measured: soil "training" AND artificial_viscosity 2.0.
# Neither alone is sufficient and the pair is better than either, because they
# fix different halves of the problem. Standing, 8 s, depth 0.2:
#   eval soil     + av 0.5 -> flips at 1.4 s, 178 deg
#   training soil + av 0.5 -> falls at 2.4 s, 103 deg
#   eval soil     + av 2.0 -> PASS but drifts to 13.7 deg, 11 cm front-rear split
#   training soil + av 2.0 -> PASS, 6.8 deg peak, 4 cm split, tilt 0.7 deg at t=1
# Soft soil fixes the IMPACT: spike falls 1168 N to 138 N, about one robot weight.
# Viscosity fixes the RINGING: on soft soil at av 0.5 the box force swing halves
# but its vertical excursion nearly TRIPLES, 0.024 m to 0.069 m, and it is the
# movement rather than the force that topples a quadruped.
# SOIL STIFFNESS PRESETS. The names "eval" and "training" name a STIFFNESS, NOT A
# ROLE, and they are actively misleading: **every Go2 episode, the dynamics model,
# and the Chrono transfer eval all run on the one called "training".** The one
# called "eval" has never been used for anything.
#
# Prefer the descriptive aliases below. "training"/"eval" are kept because 304
# collected CRM episodes record `soil_preset: "training"` in their metadata, and
# rewriting a recorded value to fix a naming problem is the wrong-at-source defect
# this project has spent considerable effort cleaning up.
#
#   soft             what we actually use, everywhere
#   hmmwv_reference  byte-identical to the HMMWV study's soil (hmmwv_crm_eval.json)
#
# The two differ by design: `soft` is half the Young's modulus and 40% the cohesion,
# chosen to make sinkage effects visible on a robot far lighter than a vehicle.
# CONSEQUENCE FOR CROSS-STUDY COMPARISON: we match the HMMWV's soil MODEL (same
# constitutive form, friction, grain size) and NOT its soil PARAMETERS. Any
# comparison of Go2 CRM numbers against HMMWV CRM numbers must say so.
_SOIL_SOFT = dict(density=1700.0, young=5.0e5, poisson=0.3, mu_I0=0.04,
                  friction=0.8, diam=0.005, cohesion=2000.0)
_SOIL_HMMWV_REFERENCE = dict(density=1700.0, young=1.0e6, poisson=0.3, mu_I0=0.04,
                             friction=0.8, diam=0.005, cohesion=5000.0)

SOIL_PRESETS = {
    "soft": _SOIL_SOFT,
    "hmmwv_reference": _SOIL_HMMWV_REFERENCE,
    # Deprecated aliases. Same objects, so `is` comparisons and round-trips hold.
    "training": _SOIL_SOFT,
    "eval": _SOIL_HMMWV_REFERENCE,
}

