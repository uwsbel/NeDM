"""The 17-number HMMWV state and the controls used by the traversing dynamics model and route tracker.

Local copy of the ``tire_normal_force_omega_pt`` preset that the experiment branch added to
``src/nedm/training/constants.py``: main's ``tire_normal_force_omega`` preset (15 columns) plus engine speed and
motorshaft torque. Every cache ``z1`` row is this state at one 50 ms frame, in physical units; every ``act`` row is
[steering, throttle, braking] held over the following 50 ms.
"""

from __future__ import annotations

STATE_FIELDS = (
    "vel_body_x_mps",               # 0  vx
    "vel_body_y_mps",               # 1  vy
    "roll_rad",                     # 2
    "pitch_rad",                    # 3
    "roll_rate_radps",              # 4
    "ang_vel_body_y_radps",         # 5  pitch rate
    "yaw_rate_radps",               # 6
    "tire_fl_force_wheel_fz_n",     # 7-10 tire normal loads
    "tire_fr_force_wheel_fz_n",
    "tire_rl_force_wheel_fz_n",
    "tire_rr_force_wheel_fz_n",
    "tire_fl_spindle_omega_radps",  # 11-14 wheel speeds
    "tire_fr_spindle_omega_radps",
    "tire_rl_spindle_omega_radps",
    "tire_rr_spindle_omega_radps",
    "engine_motor_speed_radps",     # 15 engine speed (encodes the gear given the wheel speeds)
    "engine_motorshaft_torque_nm",  # 16 motorshaft torque (the engine's lag behind the throttle)
)
ACTION_FIELDS = ("driver_steering", "driver_throttle", "driver_braking")
Z1_DIM, ACT_DIM = len(STATE_FIELDS), len(ACTION_FIELDS)

DT_S = 0.05  # one frame
VX, VY, ROLL, PITCH, YAW_RATE = 0, 1, 2, 3, 6

# Columns a deployed controller can measure: body motion, wheel speeds and engine speed (no tire loads, no torque).
OBSERVABLE_COLS = (0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15)
# The settle command before a drive starts; also the padding for actions before frame 0.
SETTLE_ACTION = (0.0, 0.0, 1.0)
