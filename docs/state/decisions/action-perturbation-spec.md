# Action perturbation: what we collect, and what we decided not to

Kyle authorised a NeRD-style perturbation collection. This records the two
design decisions taken against it and the measurements that settled them.

## Decided: NO gain randomisation. Fixed gains, action stays joint targets.

NeRD randomises `Kp` and `Kd` every step and feeds the network torque rather than
the position target. The proposal was to adopt both.

**Why torque is not a free choice.** With FIXED gains,
`tau = Kp(q* - q) - Kd*qd` is an invertible function of the target given the
state, and `q` and `qd` are already state channels -- so torque and target carry
identical information and the switch buys nothing. With RANDOMISED gains the
target stops determining the outcome and torque becomes REQUIRED. **One decision,
not two: the action encoding is forced by the gain decision.** Adopting torque in
anticipation would have spent an interface change on nothing.

**So the question was only whether to randomise gains, and the answer is no.**
Our surrogate needs validity on ONE controller whose gains never change.
Randomisation trains it on states it will never be asked about, and every
downstream cost -- torque action, PD-instant logging, resample-instant alignment,
reprocessing the existing data, a mid-study action-space change -- exists solely
to serve it.

**The honest cost, measured rather than asserted:**

    corr(e, qd) = +0.393            e = q* - q; NOT collinear, so the Kd
    |e|  mean 0.187 rad             direction is genuinely partly independent
    |qd| mean 1.870 rad/s

    Kd ~ U[0.1,1.5]  ->  0.756 Nm   the direction ONLY gain randomisation adds
    Kp ~ U[10,60]    ->  2.696 Nm   but target noise spans this direction too
    target noise sd 0.02 rad -> 0.400 Nm
    target noise sd 0.05 rad -> 1.000 Nm

**0.756 Nm against 0.4-1.0 Nm is the same order, not an order of magnitude.** So
gain randomisation is not literally redundant, and it is also not worth the
chain. Both halves needed the number.

**What would reverse this:** wanting a stiffness sweep as a case-study dimension.
Then gains become a context input, it is forced, and retrofitting costs the same
chain plus a recollection.

**Spec as built:** in-process reset, randomised initial `q`/`qd`, random target
offsets from the standing pose, fixed gains, action stays 12 joint targets, falls
kept.

**And neither change does the other's job:** torque makes gain randomisation
coherent; perturbation breaks the action-state confound. Two changes with
distinct purposes, landing near each other, and the numbers will move once.

## Found: ~15% of raw episodes diverge, and the filter that catches them is incidental

Measuring the above turned up `joint_*_target_rad` values around 1e34. They are
not a logging defect -- they are **fully diverged episodes**:

    go2_flat_s2000000_weave_050:  58 rows (normal is ~3985)
      joint position wound to -115 rad
      tau pinned at the -45.43 effort limit on EVERY row
      target growing x2.7 per control step -- exponential runaway

    3503 episodes: 2973 full-length, 530 short (15.1%), 307 under 129 rows

**The processed data is clean** -- all 8.96M train and 2.33M val action rows
scanned, max |a| = 8.13, zero above 10.

**But the protection is incidental, and that is the finding.** Those episodes are
dropped because they are shorter than `sequence_length = 128` and therefore yield
no training window -- not because anything detects divergence. **A filter that
works for the wrong reason keeps working until the conditions change, and the new
spec changes exactly those conditions:** keeping falls and randomising initial
`q`/`qd` will produce more divergences, some long enough to yield windows.

Detection layered from meaningful to blunt, and **marked and counted rather than
excluded** -- the divergence rate per sigma is what sizes the sweep, so silently
dropping those episodes would discard the signal the pilot exists to produce:

| tier | signal |
|---|---|
| primary | `\|q\|` outside the URDF joint limits by a margin -- physical, not a tuned constant |
| secondary | `tau` pinned at the effort limit for N steps with `\|q\|` growing |
| backstop | NaN / Inf / `\|state\| > 1e5` -- NeRD's own rule |

### An earlier decision, confirmed by an unrelated measurement

530 short episodes against the ~500 that the physical-admissibility exclusion
removed. **That exclusion was made on other grounds and turns out to have been
removing exactly these divergences.** Worth recording as such: a decision
validated afterwards by a measurement taken for another purpose is stronger
evidence than the reasoning that originally justified it.

### It also nearly produced a false result here

The first pass at the Kd-versus-Kp measurement returned `|e| = 1.3e31 rad` and was
close to being reported. **Corrupt data does not announce itself as corrupt; it
announces itself as a surprising result.** Checking magnitudes before interpreting
a surprise is what caught it.

## Documented, not fixed: the logged torque is not reproducible from any logged row

Testing which record row's `(q, qd, target)` reproduces the logged torque:

| offset (record steps) | -2 | -1 | 0 | +1 | +2 |
|---|---|---|---|---|---|
| mean abs err (Nm) | 1.329 | 0.872 | **0.172** | 0.885 | 1.231 |
| fraction exact | 0 | 0 | 0 | 0 | 0 |

**Same-row is best by 5x and no offset is ever exact.** The residual is
intra-step: `apply_pd` evaluates at `t`, physics advances one exchange step, and
the row records state at `t + exchange`.

> **The logged torque is not reproducible from any logged row; it is evaluated at
> the 500 Hz PD instant while the row records state one exchange step later.**

With gain randomisation dropped this stops being a blocker and becomes a
documented property. **The mechanism is bounded, not characterised** -- predicting
the residual from the `Kp*dq` term alone gave 0.053 Nm against a measured 0.163,
3x low, with a `|qd|` correlation of only 0.285.

## The rate hierarchy, corrected

Three different figures were in circulation and none was right.

    step_size_s   0.0005 s              nominal; DoStepDynamics is called with `exchange`
    exchange      4 x 0.0005 = 0.002 s  -> 500 Hz   the real loop rate; apply_pd runs here
    control_every (1/50)/0.002 = 10     -> 50 Hz    policy and actuate()
    record_step_s 0.01 s                -> 100 Hz   logging

`apply_pd()` sits OUTSIDE the `i % control_every == 0` block, so the PD runs at
500 Hz while `actuate()` only stores the target at 50 Hz.

**The 400 Hz figure came from a stale docstring, not a measurement.**
`robot.py:99` asserted a 2.5e-3 physics step where the config gives 2.0e-3. Two
readers took it for a measured fact because it was stated like one. **A docstring
that states a derived quantity goes stale silently the moment the config moves,
and it is exactly what someone consults in order to avoid measuring.** Corrected
in the same commit as this document.
