"""Upstream CTS reward terms, restricted to what the surrogate can compute.

CONVENTION, applied ONCE. Recorded Chrono joint angles are the NEGATION of the
policy/URDF convention (SIGN = -1 in imported_policy.py). Every term below is
evaluated in the POLICY/URDF frame, so callers pass q_policy = -q_recorded and
nothing else is negated anywhere. The failure mode this avoids is negating for one
term and not another: dof_pos_limits against un-negated calf positions (+1.7 against
a limit of [-2.72,-0.84]) would penalise constantly and look like a term working hard.

WEIGHTS are the CONVERGED curriculum values at 150k iterations, not the iteration-0
snapshot: correct_base_height rises 1.0 -> 10.0 and lin_vel_z decays 1.0 -> 0.0.

FUNCTIONAL FORMS: I was given the weights, not the forms. These are the standard
legged_gym definitions and each is named here so a wrong one is visible rather than
silent. `dof_pos_limits` is the one form I was given explicitly -- a LINEAR, un-squared
excursion, summed -- and it is implemented that way rather than squared.
"""
import numpy as np, torch

TRACKING_SIGMA = 0.25          # legged_gym default; NOT confirmed against upstream
POLICY_DT = 0.02               # 50 Hz control after decimation 4 -- NOT the 0.01 record step

WEIGHTS = {                    # converged; omitted terms are listed as not computable
    "tracking_lin_vel":   1.0,
    "tracking_ang_vel":   0.5,
    "ang_vel_xy":        -0.05,
    "dof_acc":           -2.5e-7,
    "action_rate":       -0.01,
    "action_smoothness": -0.01,
    "dof_pos_limits":    -2.0,
    "hip_to_default":    -0.05,
    "torques":           -1e-4,
    "dof_power":         -2e-5,
}
# PD gains from nedm.quadruped.constants. tau = Kp(target - q) - Kd*qd is a deterministic
# function of the ACTION and the joint state the surrogate already predicts, so these two
# terms need no state channel. Verified against recorded joint_*_torque_nm: 0.507 Nm
# median max-error against |tau| ~ 11.7, i.e. 4.34% relative -- a STATED APPROXIMATION,
# immaterial for a penalty whose job is to resist large effort.
# Magnitudes are invariant under the global SIGN flip, so computing in the policy frame
# gives the same |tau| as Chrono's chrono-frame PD.
PD_KP, PD_KD = 20.0, 0.5
NOT_COMPUTABLE = {             # stated, not silently dropped
    "correct_base_height": -10.0,   # no pos_z_m in the 34-D state (LARGEST weight)
    "lin_vel_z":            0.0,    # inert at convergence anyway
    "collision":           -1.0,    # thigh/calf/base contacts genuinely not recorded
    "feet_regulation":     -0.05,   # foot vel/height not in the 34-D state
}
# torques and dof_power WERE in this dict and should not have been. The classification
# asked "is this a state channel?" when the question is "is this a FUNCTION of the
# channels?" For every term called uncomputable, name the missing quantity and confirm it
# cannot be derived: `collision` survives that test, `torques` did not.

def shrunk_limits(urdf_lo, urdf_hi, frac=0.45):
    """legged_gym shrinks the range about its midpoint in _process_dof_props, NOT in
    the reward. Against the RAW urdf limits this term is nearly inert -- 0.059%
    violations measured on this plant -- so using the raw range would make the
    second-largest penalty contribute nothing, invisibly."""
    m = (urdf_lo + urdf_hi) / 2.0
    r = urdf_hi - urdf_lo
    return m - frac * r, m + frac * r

def terms(q_policy, dq_policy, dq_prev, vxy, wxy, wz, cmd, act, act_prev, act_prev2,
          lo, hi, hip_default, hip_idx, act_target):
    """All tensors (B, ...). q_policy/dq_policy are ALREADY in the policy frame."""
    t = {}
    t["tracking_lin_vel"] = torch.exp(-((cmd[:, :2] - vxy) ** 2).sum(1) / TRACKING_SIGMA)
    t["tracking_ang_vel"] = torch.exp(-((cmd[:, 2] - wz) ** 2) / TRACKING_SIGMA)
    t["ang_vel_xy"]       = (wxy ** 2).sum(1)
    t["dof_acc"]          = (((dq_prev - dq_policy) / POLICY_DT) ** 2).sum(1)
    t["action_rate"]      = ((act_prev - act) ** 2).sum(1)          # NORMALISED actions
    t["action_smoothness"]= ((act - 2 * act_prev + act_prev2) ** 2).sum(1)
    out = -(q_policy - lo).clamp(max=0.0) + (q_policy - hi).clamp(min=0.0)  # LINEAR
    t["dof_pos_limits"]   = out.sum(1)
    # ABSOLUTE VALUE, not squared. Upstream go2_env.py:
    #     return torch.sum(torch.abs(hip_pos - default_hip_pos), dim=1)
    # Squaring understates it by 59% on displacements of 0.10/0.25/0.40/0.55, and the
    # understatement GROWS as displacements shrink -- which is exactly where a posture
    # term catching small persistent drift is supposed to work.
    t["hip_to_default"]   = (q_policy[:, hip_idx] - hip_default).abs().sum(1)
    tau = PD_KP * (act_target - q_policy) - PD_KD * dq_policy
    t["torques"]          = (tau ** 2).sum(1)
    t["dof_power"]        = (tau * dq_policy).abs().sum(1)
    return t

def total(t):
    return sum(WEIGHTS[k] * v for k, v in t.items())
