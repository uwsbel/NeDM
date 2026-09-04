"""The four joint orderings, asserted from the constants rather than from memory.

Four orderings for the same twelve joints coexist in this codebase, and which one
is correct depends on the column family:

    MOTOR_NAMES        RR RL FR FL   accessors, joint targets, torques, pos/vel
    LEG_ORDER          fl fr rl rr   the foot_* columns
    FOOT_BODIES        FR FL RR RL   the body-name list
    imported policy    FL FR RL RR   via CHRONO_TO_IMPORTED

Each of these has now been stated wrongly from recollection at least once, twice on
the same day, which is why the tests below derive everything from the constants and
never restate an order by hand -- with ONE deliberate exception, noted where it
occurs, because that fact lives outside this codebase.
"""
import numpy as np

from nedm.quadruped.constants import CHRONO_TO_POLICY, FOOT_BODIES, MOTOR_NAMES
from nedm.quadruped.dataset import (JOINT_ACTION_FIELDS, JOINT_STATE_FIELDS,
                                    JOINT_TORQUE_FIELDS, LEG_ORDER, LEG_TO_FOOT_BODY)
from nedm.quadruped.imported_policy import CHRONO_TO_IMPORTED, IMPORTED_DEFAULTS


def _legs(perm=None):
    names = [n.removesuffix("_joint") for n in MOTOR_NAMES]
    if perm is not None:
        names = [names[i] for i in perm]
    return [n.split("_")[0] for n in names][::3]


def test_permutations_are_involutions(perm=None):
    """Both are their own inverse, so applying one in either direction is the same
    map. This is asserted rather than assumed because a test written on the belief
    that direction matters would encode a claim the constants do not support."""
    for perm in ([perm] if perm is not None else [CHRONO_TO_IMPORTED, CHRONO_TO_POLICY]):
        p = np.asarray(perm)
        assert (p[p] == np.arange(len(p))).all(), perm


def test_chrono_to_imported_yields_the_policy_leg_order():
    """Indexing MOTOR_NAMES by the permutation must give the imported policy's own
    FL/FR/RL/RR convention.

    Only the LEFT/RIGHT half of this literal is external. It rests on a hip sign
    convention (+0.1 = left) that no array here states. The FRONT/REAR half is
    derivable and is checked separately below, so what remains unverifiable is one
    bit per pair rather than the whole ordering.
    """
    assert _legs(CHRONO_TO_IMPORTED) == ["FL", "FR", "RL", "RR"]


def test_front_rear_grouping_is_derivable_from_the_policy_defaults():
    """The F,F,R,R structure of the policy's leg order follows from its own defaults.

    IMPORTED_DEFAULTS carries thigh values 0.8, 0.8, 1.0, 1.0 -- front and rear
    thigh defaults differ, so legs 0 and 1 must share one and legs 2 and 3 the
    other.

    NOTE, measured rather than assumed: a front/rear transposition of the
    PERMUTATION is already caught by the literal assertion above -- it yields
    RL,RR,FL,FR, which differs from FL,FR,RL,RR. What this test adds is
    independence from that literal: it catches a change to IMPORTED_DEFAULTS
    itself, which the hardcoded order cannot see, and it derives the front/rear
    structure instead of taking it on faith. An earlier draft of this docstring
    claimed the literal assertion missed front/rear swaps. It does not.
    """
    thighs = np.asarray(IMPORTED_DEFAULTS).reshape(4, 3)[:, 1]
    assert thighs[0] == thighs[1], "legs 0,1 should share a thigh default"
    assert thighs[2] == thighs[3], "legs 2,3 should share a thigh default"
    assert thighs[0] != thighs[2], "front and rear thigh defaults should differ"


def test_joint_column_families_all_share_motor_names_order():
    """Targets, torques and pos/vel must agree index-for-index, because a model reads
    q, dq and the previous action from the same row and lines them up by position."""
    base = [f.removeprefix("joint_").removesuffix("_target_rad")
            for f in JOINT_ACTION_FIELDS]
    assert [f.removeprefix("joint_").removesuffix("_torque_nm")
            for f in JOINT_TORQUE_FIELDS] == base
    assert [f.removeprefix("joint_").removesuffix("_pos_rad")
            for f in JOINT_STATE_FIELDS[:12]] == base
    assert [f.removeprefix("joint_").removesuffix("_vel_radps")
            for f in JOINT_STATE_FIELDS[12:]] == base
    assert [n.removesuffix("_joint").lower() for n in MOTOR_NAMES] == base


def test_foot_columns_use_leg_order_not_foot_bodies():
    """The foot_* family is LEG_ORDER, which is NOT the FOOT_BODIES order. Packing
    against the wrong one transposes left and right, and no aggregate check can see
    it: a relabelling leaves every marginal distribution unchanged, and even the trot
    signature survives, since the two dominant diagonal modes swap into each other."""
    assert [b.split("_")[0] for b in FOOT_BODIES] != [l.upper() for l in LEG_ORDER]
    assert [LEG_TO_FOOT_BODY[l] for l in LEG_ORDER] == [f"{l.upper()}_foot" for l in LEG_ORDER]
    assert sorted(LEG_TO_FOOT_BODY.values()) == sorted(FOOT_BODIES)


if __name__ == "__main__":
    # No pytest in the env that holds the code, and two of the existing tests in
    # this directory are plain scripts, so this follows that convention rather
    # than adding a dependency to run four assertions.
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"{len(tests)} passed")
