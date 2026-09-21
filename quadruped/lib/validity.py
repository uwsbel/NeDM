"""Per-row validity checks, applied at collection time.

A diverged episode must never reach the corpus, and the previous study shows both ways
that goes wrong. Quarantining a whole episode for a defect in its last few rows threw away
98-99.9% good data -- 4,300 rows discarded where 32 were bad. And a purely non-finite
check missed divergences that stayed finite: joints that teleported by more than pi in one
control step were numerically valid and physically nonsense.

So: check every row, TRUNCATE at the first failure, keep the valid prefix, and record
which check fired and where. The truncation rate per check is then a corpus-level
statistic that is visible before anything is quoted, rather than a surprise later.

Divergences are not merely discarded. Where the real simulator fails is the boundary a
fine-tuned policy has to be kept inside, so the failing tail is written to a separate
`failures/` set with the same provenance: excluded from dynamics training, available for
asking what the optimiser pushed the system toward.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Physical bounds. Each is deliberately loose: these detect DIVERGENCE, not poor
# locomotion, and a check that also fires on a bad-but-real gait would silently bias the
# corpus toward easy behaviour.
MAX_JOINT_STEP_RAD = math.pi        # a joint cannot cross pi in one control step
MAX_JOINT_ABS_RAD = 4.0             # well outside any Go2 actuator range
MAX_BASE_SPEED_MPS = 12.0           # a Go2 does not travel at 12 m/s
MAX_BASE_RATE_RADPS = 60.0
MIN_BASE_Z_M = -0.5                 # below the bed: it fell through the terrain
MAX_BASE_Z_M = 3.0                  # launched
INVERTED_GRAV_Z = 0.0               # grav_body_z >= 0 means the trunk is past horizontal
OFF_BED_MARGIN_M = 0.25             # how close to the bed edge still counts as on it
SINK_MARGIN_M = 0.05                # below soil_top + this, the base is inside the bed


@dataclass
class Verdict:
    ok: bool
    row: int | None = None
    check: str | None = None
    detail: str = ""


def _f(row, key):
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def first_invalid(rows, joint_pos_fields, *, dt_s: float = 0.01, bed=None,
                  soil_top=None) -> Verdict:
    """Index of the first row that fails any check, or ok.

    Checks are ordered cheapest-first, and the first to fire wins, so the recorded reason
    is the earliest detectable symptom rather than a downstream consequence.

    `bed`, when given, is ((xlo, ylo), (xhi, yhi)) for the soil patch, and rows are
    truncated once the robot leaves it. Without this the only symptom of walking off the
    edge is base_height firing half a second later, once the robot has fallen far enough
    -- by which point the rows between the edge and the trigger are free-fall recorded as
    locomotion. Rigid ground passes None: its floor is sized to the travel instead.

    `soil_top`, when given, catches the robot sinking INTO the bed. Only the feet and
    calves are FSI-coupled (`terrain.py:115-117`); the trunk has no interaction with the
    soil at all, and on CRM there is no rigid ground for its collision model to touch. So
    a robot that pitches far enough to put its belly down has nothing holding it up: it
    descends through the bed, pulls the legs after it, and keeps going. One ensemble case
    recorded a mean base height of -0.303 m, a third of a metre below the bottom of the
    soil, while reporting perfectly finite numbers throughout.

    Without this check the only backstop is MIN_BASE_Z_M at -0.5, which fires roughly half
    a second after the fall begins and lets every row in between into the corpus as
    locomotion. It is not noisy data, it is a robot falling through the world.
    """
    prev_q = None
    prev_t = None

    for i, row in enumerate(rows):
        # 1. non-finite anywhere. The broadest net, and the cheapest.
        for k, v in row.items():
            if isinstance(v, str) and v.strip().lower() in ("nan", "inf", "-inf", "+inf"):
                return Verdict(False, i, "non_finite", f"{k}={v}")

        # 2. time must advance. A duplicated or reversed stamp means the writer or the
        #    solver lost its place, and every downstream difference is meaningless.
        t = _f(row, "time_s")
        if t is not None:
            if prev_t is not None and not (t > prev_t - 1e-12):
                return Verdict(False, i, "time_nonmonotonic", f"{prev_t} -> {t}")
            prev_t = t

        # 3. joint sanity: absolute range, then step size. Step is what catches the
        #    divergences that stay finite.
        q = []
        for f in joint_pos_fields:
            v = _f(row, f)
            if v is None:
                q = None
                break
            if abs(v) > MAX_JOINT_ABS_RAD:
                return Verdict(False, i, "joint_range", f"{f}={v:.3f}")
            q.append(v)
        if q is not None:
            if prev_q is not None:
                for f, a, b in zip(joint_pos_fields, prev_q, q, strict=True):
                    if abs(b - a) > MAX_JOINT_STEP_RAD:
                        return Verdict(False, i, "joint_step",
                                       f"{f} moved {abs(b - a):.3f} rad in one step")
            prev_q = q

        # 4. base state
        vx, vy, vz = _f(row, "vel_body_x_mps"), _f(row, "vel_body_y_mps"), _f(row, "vel_body_z_mps")
        if None not in (vx, vy, vz):
            sp = math.sqrt(vx * vx + vy * vy + vz * vz)
            if sp > MAX_BASE_SPEED_MPS:
                return Verdict(False, i, "base_speed", f"{sp:.2f} m/s")
        for k in ("roll_rate_radps", "ang_vel_body_y_radps", "yaw_rate_radps"):
            v = _f(row, k)
            if v is not None and abs(v) > MAX_BASE_RATE_RADPS:
                return Verdict(False, i, "base_rate", f"{k}={v:.2f}")

        z = _f(row, "pos_z_m")
        if z is not None and not (MIN_BASE_Z_M <= z <= MAX_BASE_Z_M):
            return Verdict(False, i, "base_height", f"pos_z_m={z:.3f}")

        # 4a. sinking into the bed. Fires at the surface rather than 0.5 m below it.
        if soil_top is not None and z is not None and z < soil_top + SINK_MARGIN_M:
            return Verdict(False, i, "sinking",
                           f"pos_z_m={z:.3f} below soil_top {soil_top:.3f}"
                           f"+{SINK_MARGIN_M:.2f}")

        # 4b. still on the soil. Fires BEFORE base_height does, so the free-fall rows
        #     between the edge and the fall are excluded rather than recorded as gait.
        if bed is not None:
            (bxlo, bylo), (bxhi, byhi) = bed
            px, py = _f(row, "pos_x_m"), _f(row, "pos_y_m")
            if px is not None and not (bxlo + OFF_BED_MARGIN_M <= px <= bxhi - OFF_BED_MARGIN_M):
                return Verdict(False, i, "off_bed",
                               f"pos_x_m={px:.3f} outside [{bxlo:+.2f}, {bxhi:+.2f}]")
            if py is not None and not (bylo + OFF_BED_MARGIN_M <= py <= byhi - OFF_BED_MARGIN_M):
                return Verdict(False, i, "off_bed",
                               f"pos_y_m={py:.3f} outside [{bylo:+.2f}, {byhi:+.2f}]")

        # 5. attitude. Past horizontal the CRM coupling is outside anything we intend to
        #    model. This comment used to add "a fallen robot is legitimate data", which is
        #    true on rigid ground and false on CRM: there, the trunk is not coupled to the
        #    soil at all, so a fallen robot does not lie on the bed, it goes through it.
        #    On CRM a fall is the start of a fiction, not a hard-but-real state.
        gz = _f(row, "grav_body_z")
        if gz is not None and gz >= INVERTED_GRAV_Z:
            return Verdict(False, i, "inverted", f"grav_body_z={gz:.3f}")

    return Verdict(True)


def truncate(rows, joint_pos_fields, *, dt_s: float = 0.01, bed=None,
             soil_top=None):
    """Return (kept_rows, failing_tail, verdict).

    The tail is returned rather than dropped so it can be written to the failures set.
    """
    v = first_invalid(rows, joint_pos_fields, dt_s=dt_s, bed=bed,
                      soil_top=soil_top)
    if v.ok:
        return rows, [], v
    return rows[: v.row], rows[v.row:], v


def summarise(verdicts) -> dict:
    """Corpus-level truncation statistics, for the manifest.

    A rising truncation rate on one check is the signal that an excitation level has gone
    past the point where it produces data rather than wreckage, so it belongs in the
    corpus record and not only in a log.
    """
    by = {}
    trunc = 0
    for v in verdicts:
        if not v.ok:
            trunc += 1
            by[v.check] = by.get(v.check, 0) + 1
    return {"episodes": len(verdicts), "truncated": trunc,
            "rate": (trunc / len(verdicts)) if verdicts else 0.0, "by_check": by}


if __name__ == "__main__":
    JP = [f"joint_{leg}_{seg}_pos_rad"
          for leg in ("rr", "rl", "fr", "fl") for seg in ("hip", "thigh", "calf")]

    def mk(n, **over):
        out = []
        for i in range(n):
            r = {"time_s": i * 0.01, "pos_z_m": 0.32, "grav_body_z": -0.98,
                 "vel_body_x_mps": 0.5, "vel_body_y_mps": 0.0, "vel_body_z_mps": 0.0}
            for f in JP:
                r[f] = 0.4
            r.update({k: v(i) if callable(v) else v for k, v in over.items()})
            out.append(r)
        return out

    assert first_invalid(mk(50), JP).ok, "clean rows must pass"

    r = mk(50); r[30]["vel_body_x_mps"] = "nan"
    v = first_invalid(r, JP); assert (v.row, v.check) == (30, "non_finite"), v

    # a step larger than pi but still inside the absolute range, so joint_step is what
    # must fire -- joint_range is checked first and would otherwise mask it
    r = mk(50)
    for f in JP:
        r[30][f] = -3.0                      # 0.4 -> -3.0 is 3.4 rad, and |-3.0| < 4.0
    v = first_invalid(r, JP); assert (v.row, v.check) == (30, "joint_step"), v

    # and the absolute check does fire first when both would
    r = mk(50)
    for f in JP:
        r[30][f] = 4.4
    v = first_invalid(r, JP); assert (v.row, v.check) == (30, "joint_range"), v

    r = mk(50); r[20]["grav_body_z"] = 0.2
    v = first_invalid(r, JP); assert (v.row, v.check) == (20, "inverted"), v

    r = mk(50); r[10]["pos_z_m"] = -1.0
    v = first_invalid(r, JP); assert (v.row, v.check) == (10, "base_height"), v

    r = mk(50); r[40]["time_s"] = 0.05
    v = first_invalid(r, JP); assert (v.row, v.check) == (40, "time_nonmonotonic"), v

    kept, tail, v = truncate(r, JP)
    assert len(kept) == 40 and len(tail) == 10, (len(kept), len(tail))

    s = summarise([first_invalid(mk(10), JP), v, v])
    assert s["truncated"] == 2 and s["by_check"]["time_nonmonotonic"] == 2, s

    print("validity self-test: all checks fire at the right row, truncation keeps the "
          "prefix and returns the tail, summary counts by check")
