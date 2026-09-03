# The Go2 has no actuator model, and that is a fidelity caveat

**Measured 2026-09-03. Not a bug — a modelling choice inherited from
`ChParserURDF`, and it bounds what our numbers mean.**

## The joints are kinematic constraints, not servos

`ChParserURDF` with `ActuationType_POSITION` creates
`ChLinkMotorRotationAngle` (`ChParserURDF.cpp:604`), which **imposes** the
commanded angle rather than driving toward it.

Measured standing, after transients settle — |measured − commanded| joint angle:

| p50 | p99 | max |
|---|---|---|
| 9.6e-15 | 7.2e-08 | 3.8e-07 rad |

**Machine precision.** The joint simply *is* the commanded angle: infinite
stiffness, zero compliance, **unbounded joint torque**, no torque limit.

*(A first probe reported a 16° worst-case error and nearly went in the record as
"tracking is poor." That window included the spawn-drop impact. Restricted to
settled time it is 1e-8. The transient was real and was the robot landing, not
tracking error.)*

## Consequence for the case study

**Our Go2 can apply arbitrary joint torque**, so it can produce motions a real Go2
could not. Every number collected on this plant carries that assumption.

It does **not** invalidate the contact-geometry findings — the standoff, the
foot-vs-kernel resolution, the contact-mode separability — since those are
determined by geometry and the smoothing kernel. Static forces are also sane: the
four feet sum to 158 N against a 158 N robot.

But it is a fidelity caveat that belongs in the writeup, and on the axis the
manuscript cares about it makes the quadruped the **least** physical of the four
case studies. The HMMWV has real tire models and a powertrain; the arm has real
actuation. §7 already concedes that *"whatever bias Chrono's tire, terramechanics,
and actuator models carry is inherited by the surrogate"* — **"no actuator model
at all" is a stronger statement than that sentence covers.**

## It also settles the import question

Every candidate policy assumes **PD torque control**:

| source | stiffness | damping | action scale |
|---|---|---|---|
| Genesis Go2 example | kp 20.0 | kd 0.5 | 0.25 |
| `unitree_rl_gym` (BSD-3) | 20.0 N·m/rad | 0.5 N·m·s/rad | 0.25 |

Same numbers — shared ancestry. For them an action means *"a PD setpoint, tracked
with finite stiffness against finite torque limits, with joint deflection under
load as feedback."* On our plant the identical action means *"this angle, exactly,
now."*

**Those are different plants, not different tunings.** A failed walk would tell us
about the plant and nothing about the policy.

**Strong inference (not fact):** this is almost certainly why SBEL imported the
*convention* and not the *weights*. There is no note in the repo and the history is
three commits with no explanation, but you cannot run a PD-trained policy in a
rigid-motor plant, so they retrained in the plant they had.

## The option, not taken

Switching to `ChLinkMotorRotationTorque` driven by a PD law at kp 20 / kd 0.5
would make imported policies runnable *and* give the simulation bounded joint
torque.

**Not done.** It invalidates every gate and all data collected under the current
plant, and `model_2999` is matched to it. This is a larger decision than the
import question that surfaced it, and it belongs to Kyle.

Note it also bears on retraining: **a policy retrained under a PD plant would be
more transferable** than one retrained under this one.
