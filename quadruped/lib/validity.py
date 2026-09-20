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


def first_invalid(rows, joint_pos_fields, *, dt_s: float = 0.01) -> Verdict:
    """Index of the first row that fails any check, or ok.

    Checks are ordered cheapest-first, and the first to fire wins, so the recorded reason
    is the earliest detectable symptom rather than a downstream consequence.
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

        # 5. attitude. Not a fall detector -- a fallen robot is legitimate data -- but
        #    past horizontal the CRM coupling is outside anything we intend to model.
        gz = _f(row, "grav_body_z")
        if gz is not None and gz >= INVERTED_GRAV_Z:
            return Verdict(False, i, "inverted", f"grav_body_z={gz:.3f}")

    return Verdict(True)


def truncate(rows, joint_pos_fields, *, dt_s: float = 0.01):
    """Return (kept_rows, failing_tail, verdict).

    The tail is returned rather than dropped so it can be written to the failures set.
    """
    v = first_invalid(rows, joint_pos_fields, dt_s=dt_s)
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
