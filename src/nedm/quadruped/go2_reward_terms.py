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

TRACKING_SIGMA = 0.25          # CONFIRMED against go2_config.py:166 (see below)
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

# THE QUANTITY EACH OMITTED TERM ACTUALLY NEEDS. The dict above is a per-run fact
# frozen into a module-level constant, which cannot be right for two different state
# definitions -- and it was not. It was written for the 34-D state and applied
# unchanged to every 36-D and 40-D run, so `correct_base_height` (the LARGEST weight
# in the whole reward, 10x tracking_lin_vel) was dropped from surrogates that carry
# pos_z_m. v4's omission was correct; every 36-D fine-tune after it inherited a
# reason that had stopped being true.
#
# Naming the requirement per term is what makes the rule above runnable instead of
# advisory. Empty tuple = genuinely underivable from any state we have.
OMITTED_REQUIRES = {
    "correct_base_height": ("pos_z_m",),
    "lin_vel_z":           ("vel_body_z_mps",),
    "collision":           ("foot_fl_in_contact", "foot_fr_in_contact",
                            "foot_rl_in_contact", "foot_rr_in_contact"),
    "feet_regulation":     (),      # foot velocity and height: in no state we have
}


def wrongly_omitted(state_fields):
    """Omitted terms whose required channels are ALL present in this state.

    Returns {term: weight}. Non-empty means the run is about to drop a reward term
    it could compute -- which is the defect this function exists to make impossible
    to repeat, and which cost the fine-tuning line its largest reward term across
    every 36-D run.
    """
    have = set(state_fields)
    return {t: NOT_COMPUTABLE[t] for t, req in OMITTED_REQUIRES.items()
            if req and have.issuperset(req)}

def shrunk_limits(urdf_lo, urdf_hi, frac=0.45):
    """legged_gym shrinks the range about its midpoint in _process_dof_props, NOT in
    the reward. Against the RAW urdf limits this term is nearly inert -- 0.059%
    violations measured on this plant -- so using the raw range would make the
    second-largest penalty contribute nothing, invisibly."""
    m = (urdf_lo + urdf_hi) / 2.0
    r = urdf_hi - urdf_lo
    return m - frac * r, m + frac * r

def terms(q_policy, dq_policy, dq_prev, vxy, wxy, wz, cmd, act, act_prev, act_prev2,
          lo, hi, hip_default, hip_idx, act_target, pos_z=None, height_target=None):
    """All tensors (B, ...). q_policy/dq_policy are ALREADY in the policy frame.

    pos_z / height_target enable `correct_base_height`, the LARGEST weight in the whole
    reward (-10.0, ten times tracking_lin_vel). Pass both or neither.

    HEIGHT IS TERRAIN-RELATIVE AND pos_z_m IS NOT. On rigid ground the bed sits at z=0
    so pos_z IS the height; on CRM the soil surface sits at ~0.20 m (soil_bottom 0.0 +
    depth 0.20) and the base rides at ~0.553 m for the same ~0.35 m stance. A target
    calibrated on rigid would therefore read a normally-standing robot on soil as 0.20 m
    too high and drive it DOWN into the soil -- turning the term that should fix this
    failure into a worse version of it. height_target must be supplied per terrain by
    the caller, which is why it has no default.
    """
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
    if pos_z is not None:
        if height_target is None:
            raise ValueError("pos_z given without height_target: the target is "
                             "terrain-relative and must not be guessed")
        t["correct_base_height"] = (pos_z - height_target) ** 2
    return t

def total_scaled(t, penalty_scale):
    """`total()` with the NEGATIVE-weight terms multiplied by `penalty_scale`.

    A diagnostic, not a tuning knob. Under backprop-through-a-surrogate a term's
    influence on the update is its weight times the STIFFNESS of its path through the
    model, which has nothing to do with its share of the return. Measured here:
    penalties are 13.9% of reward value and 48-57% of gradient norm, and at a 25-step
    branch `tracking_lin_vel` supplies 76% of the reward and 3.9% of the gradient.
    PPO's score-function estimator is structurally immune -- it multiplies
    grad-log-pi by a SCALAR reward, so no term can be over-weighted by its Jacobian.

    Setting penalty_scale = 0 leaves only the tracking terms, which answers a question
    reweighting cannot: whether d(velocity)/d(action) through the surrogate carries
    usable gradient at all.
    """
    w = dict(WEIGHTS)
    w["correct_base_height"] = NOT_COMPUTABLE["correct_base_height"]
    missing = [k for k in t if k not in w]
    if missing:
        raise KeyError(f"reward terms with no weight: {missing}")
    return sum((w[k] if w[k] > 0 else w[k] * penalty_scale) * v for k, v in t.items())


def total(t):
    """Weighted sum. A term present in `t` but absent from WEIGHTS is a silent zero,
    so look it up in the merged table and fail loudly instead."""
    w = dict(WEIGHTS)
    w["correct_base_height"] = NOT_COMPUTABLE["correct_base_height"]
    missing = [k for k in t if k not in w]
    if missing:
        raise KeyError(f"reward terms with no weight: {missing}")
    return sum(w[k] * v for k, v in t.items())
