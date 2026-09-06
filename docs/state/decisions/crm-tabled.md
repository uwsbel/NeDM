# CRM is tabled; the two measured reasons

**Decision (Kyle, 2026-09-05, relayed):** "For now, I have just basically table
crm altogether. I just want to show this works on rigid terrain even."

Recorded so this is not re-derived in three months. Both blockers were found
while trying to run W0 on CRM. Neither is solved -- they are **moot**, because
the path they block is not being taken.

## 1. The CRM episodes log commanded joint state, not measured

W0 exits with "no joint pos/vel columns found; check the schema".

| set | columns | `joint_*_pos_rad` | `joint_*_vel_radps` |
|---|---|---|---|
| CRM (seed offset 1,000,000) | 69 | 0 | 0 |
| rigid | 168-171 | 12 | 12 |

The only joint columns on the CRM half are `joint_*_target_rad` -- what the
policy **commanded**, not what the robot **did**. W0 needs the pre-impact
measured state, which that half never recorded.

**This is data age, not a code gate.** `dataset.py:502` emits
`JOINT_STATE_FIELDS` unconditionally; `build_row` takes zero terrain-type
branches. The commit that added measured joint logging is `1d652ae` (2026-09-04),
which postdates the CRM collection. A re-collection today would log it on the
CRM path with no code change. That is worth knowing precisely, because the
cheap-sounding fix is real -- it is the second blocker that makes it not worth
buying.

## 2. The CRM soil is spatially uniform

152 episodes carry **one** distinct soil parameter set:

    cohesion 2000, density 1700, diam 0.005, friction 0.8,
    mu_I0 0.04, poisson 0.3, young 500000      (preset "training")

`terrain.py:30-34` builds a single scalar material and applies it to the entire
bed via `crm_compat.set_crm_soil(terrain, mat)`. There is no per-region
assignment on this path.

So even with measured joint state, W0 could not detect the failure mode it
exists to detect: there is no spatial variation for the robot to walk into.

**And this binds harder on W1 than on W0.** W1's object is a terrain field with
memory, co-evolved with robot state. On a bed with one uniform material there is
no field to learn and no rut to re-encounter -- the terrain side would be
predicting a constant. A uniform-bed CRM dataset cannot support the main line of
the plan, not only the gated one.

## What tabling costs and what it does not

The soil half of the contribution is already carried by the body-level study,
which passes all three validation levels. What moves is the *framing*: from
terramechanics to **contact discontinuity**, which the manuscript's limitations
section already names as future work.
