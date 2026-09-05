"""Discriminating tests for the CTS reward terms, against recorded-convention input.

WHY THESE AND NOT A FORMULA READING. The run that motivated implementing these terms
is gross enough to fire every variant: a 5.05 rad joint offset trips dof_pos_limits
whether or not the soft shrink is applied. So that run CANNOT validate the
implementation, and a wrong one would pass it and then carry the appearance of having
been checked into every later run where it is inert.

Each test below is chosen so a wrong implementation gives a DIFFERENT answer, and
each is paired with the wrong implementations it must reject. A test suite that
passes a broken term is the same failure one level up, so the suite reports its own
discrimination matrix rather than only a verdict.

WHAT THIS SUITE DOES AND DOES NOT ESTABLISH. It validates an IMPLEMENTATION against
the reference transcribed here. It does NOT validate the reference against reality:
if the single-negation convention encoded below were itself wrong, the reference would
encode it and every test would still pass green. The convention has its own, separate
validation -- the URDF calf range is [-2.72, -0.84] while recorded calf sits near
[+0.52, +3.00], and across 120 episodes 4/4 thighs fit the negated range and 0/4 fit
the raw one. Those are two validations of two different things and only the first is
in this file. A green matrix is not confirmation of the convention.

CONVENTION. Inputs are RECORDED (Chrono) joint positions. The policy and the URDF
share one convention and recorded angles are its NEGATION (SIGN = -1): the URDF calf
range is [-2.72, -0.84] while recorded calf sits near [+0.52, +3.00]. Every term
negates once, then applies the upstream formula unchanged.
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
import numpy as np

URDF = ("/home/kyle/Documents/sbel/sbel-reproducibility/2025/multi-terrain-RL"
        "/data/robot/go2_irrvis/urdf/go2_description.urdf")
SOFT = 0.9
# Recorded column order (MOTOR_NAMES): RR, RL, FR, FL, each hip/thigh/calf.
ORDER = [f"{leg}_{k}" for leg in ("RR", "RL", "FR", "FL") for k in ("hip", "thigh", "calf")]
DEFAULTS = {"FL_hip": 0.1, "RL_hip": 0.1, "FR_hip": -0.1, "RR_hip": -0.1,
            "FL_thigh": 0.8, "FR_thigh": 0.8, "RL_thigh": 1.0, "RR_thigh": 1.0,
            "FL_calf": -1.5, "FR_calf": -1.5, "RL_calf": -1.5, "RR_calf": -1.5}


def urdf_limits():
    lim = {}
    for j in ET.parse(URDF).getroot().iter("joint"):
        L = j.find("limit")
        if L is not None:
            lim[j.get("name")] = (float(L.get("lower")), float(L.get("upper")))
    return np.array([lim[f"{n}_joint"] for n in ORDER])


RAW = urdf_limits()
M = RAW.mean(axis=1)
R = RAW[:, 1] - RAW[:, 0]
SOFT_LIM = np.stack([M - 0.5 * R * SOFT, M + 0.5 * R * SOFT], axis=1)
DEF = np.array([DEFAULTS[n] for n in ORDER])
HIP = [i for i, n in enumerate(ORDER) if n.endswith("_hip")]


# ---- reference implementations, transcribed from legged_robot.py -----------------
def ref_dof_pos_limits(q_rec):
    q = -np.asarray(q_rec)                       # into policy/URDF convention
    out = -np.clip(q - SOFT_LIM[:, 0], None, 0.0) + np.clip(q - SOFT_LIM[:, 1], 0.0, None)
    return float(out.sum())


def ref_hip_to_default(q_rec):
    q = -np.asarray(q_rec)
    return float(np.abs(q[HIP] - DEF[HIP]).sum())


# ---- the wrong implementations each test must reject ----------------------------
def wrong_limits_raw(q_rec):          # soft shrink omitted
    q = -np.asarray(q_rec)
    return float((-np.clip(q - RAW[:, 0], None, 0.0) + np.clip(q - RAW[:, 1], 0.0, None)).sum())


def wrong_limits_unsigned(q_rec):     # negation omitted
    q = np.asarray(q_rec)
    return float((-np.clip(q - SOFT_LIM[:, 0], None, 0.0) + np.clip(q - SOFT_LIM[:, 1], 0.0, None)).sum())


def wrong_hip_unsigned(q_rec):        # negation omitted
    q = np.asarray(q_rec)
    return float(np.abs(q[HIP] - DEF[HIP]).sum())


def wrong_hip_squared(q_rec):         # abs replaced by square -- upstream uses torch.abs
    q = -np.asarray(q_rec)
    return float(np.square(q[HIP] - DEF[HIP]).sum())


def wrong_limits_squared(q_rec):      # linear excursion replaced by squared
    q = -np.asarray(q_rec)
    lo = -np.clip(q - SOFT_LIM[:, 0], None, 0.0); hi = np.clip(q - SOFT_LIM[:, 1], 0.0, None)
    return float(np.square(lo + hi).sum())


# ---- tracking, added because sigma is assumable and therefore assumed --------------
TRACKING_SIGMA = 0.25          # go2_config.py line 166, confirmed against source


def ref_tracking_lin_vel(err_xy):
    return float(np.exp(-np.square(np.asarray(err_xy)).sum() / TRACKING_SIGMA))


def wrong_tracking_sigma1(err_xy):    # legged_gym's OTHER common default
    return float(np.exp(-np.square(np.asarray(err_xy)).sum() / 1.0))


def wrong_tracking_nosquare(err_xy):  # error not squared
    return float(np.exp(-np.abs(np.asarray(err_xy)).sum() / TRACKING_SIGMA))


def wrong_hip_legmix(q_rec):          # defaults not paired with their own joint
    q = -np.asarray(q_rec)
    return float(np.abs(q[HIP] - DEF[HIP][::-1]).sum())


# ---- the discriminating cases ---------------------------------------------------
def case_between_limits():
    """Every joint parked at 0.48 of half-range from midpoint: INSIDE raw, OUTSIDE soft."""
    return -(M + 0.48 * R)            # negated, because inputs are recorded-convention


def case_nominal():
    """The default standing pose, in recorded convention. Nothing should be penalised."""
    return -DEF.copy()


# Displacements are UNEQUAL on purpose. With all four hips displaced by the same
# amount the wrong variants coincidentally sum to the right answer -- the two hips
# with default +0.1 and the two with -0.1 cancel -- and the test passes every
# implementation while looking like it discriminates. An unequal set removes the
# cancellation. A test that passes everything is not a test.
HIP_DISPLACEMENT = [0.10, 0.25, 0.40, 0.55]


def case_sign_sensitive():
    """Hips displaced unequally, so each wrong convention gives a different total."""
    q = -DEF.copy()
    for d, i in zip(HIP_DISPLACEMENT, HIP):
        q[i] = -(DEF[i] + d)
    return q


TRACK_ERR = [0.4, 0.3]        # |err| = 0.5, chosen so sigma variants separate clearly
TESTS = [
    ("tracking_lin_vel = exp(-|err|^2/0.25) at |err|=0.5", "track",
     TRACK_ERR, lambda v: abs(v - np.exp(-0.25 / 0.25)) < 1e-9),
    ("dof_pos_limits fires between raw and soft limits", "limits",
     case_between_limits(), lambda v: v > 1e-6),
    # MAGNITUDE, not just sign. With every joint 0.48 of half-range from the midpoint
    # the excursion past the 0.45 soft limit is exactly 0.03*r per joint, so the linear
    # sum is 0.03*sum(r). Checking only ">0" let a SQUARED implementation through --
    # the suite flagged itself unsound until this row existed.
    ("dof_pos_limits equals 0.03*sum(range), i.e. linear not squared", "limits",
     case_between_limits(), lambda v: abs(v - 0.03 * R.sum()) < 1e-6),
    ("dof_pos_limits is silent at the nominal pose", "limits",
     case_nominal(), lambda v: v < 1e-9),
    ("hip_to_default is ~0 at the nominal pose", "hip",
     case_nominal(), lambda v: v < 1e-6),
    ("hip_to_default = sum of unequal hip displacements", "hip",
     case_sign_sensitive(), lambda v: abs(v - sum(HIP_DISPLACEMENT)) < 1e-6),
]
IMPLS = {"limits": [("REFERENCE", ref_dof_pos_limits),
                    ("wrong: raw limits, no soft shrink", wrong_limits_raw),
                    ("wrong: no sign negation", wrong_limits_unsigned),
                    ("wrong: squared not linear", wrong_limits_squared)],
         "hip": [("REFERENCE", ref_hip_to_default),
                 ("wrong: no sign negation", wrong_hip_unsigned),
                 ("wrong: defaults mispaired across legs", wrong_hip_legmix),
                 ("wrong: squared not abs", wrong_hip_squared)],
         "track": [("REFERENCE", ref_tracking_lin_vel),
                   ("wrong: sigma = 1.0", wrong_tracking_sigma1),
                   ("wrong: error not squared", wrong_tracking_nosquare)]}


def main():
    print("DISCRIMINATION MATRIX -- a test is only useful if it REJECTS the wrong ones")
    print("Validates implementations AGAINST THE REFERENCE. Does not validate the")
    print("reference against reality: the negation convention is established separately,")
    print("by measurement (URDF calf [-2.72,-0.84] vs recorded [+0.52,+3.00]).\n")
    ok = True
    for term in ("limits", "hip", "track"):
        rows = [t for t in TESTS if t[1] == term]
        names = [n for n, _ in IMPLS[term]]
        print(f"  {term}:")
        print(f"    {'test':<52}" + "".join(f"{n[:26]:>28}" for n in names))
        for desc, _, q, pred in rows:
            cells = []
            for i, (nm, fn) in enumerate(IMPLS[term]):
                v = fn(q); p = pred(v)
                if i == 0 and not p:
                    ok = False
                cells.append(f"{('PASS' if p else 'fail')} {v:>8.4f}")
            print(f"    {desc:<52}" + "".join(f"{c:>28}" for c in cells))
        for nm, fn in IMPLS[term][1:]:
            if all(pred(fn(q)) for desc, t, q, pred in rows):
                print(f"    *** NOT DISCRIMINATING: '{nm}' passes every test ***")
                ok = False
        # Per-test, not only per-variant: a test that no wrong variant fails is
        # carrying no weight even when the suite as a whole still discriminates.
        for desc, _, q, pred in rows:
            if all(pred(fn(q)) for _, fn in IMPLS[term][1:]):
                print(f"    *** WEIGHTLESS TEST: '{desc}' passes every wrong variant ***")
                ok = False
        print()
    print("  reference passes all tests and every wrong variant is rejected by at least one"
          if ok else "  *** SUITE IS NOT SOUND ***")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
