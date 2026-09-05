# Scripted trot generator: parked as a negative result

**An open-loop scripted stance is not viable on this Go2 platform**, and the
evidence that this is NOT a kinematics bug is what makes the statement worth
anything. Someone will propose this again.

## What was built and verified

A drop-in replacement for the imported policy at the same interface: body
velocity command in, twelve joint targets out in `MOTOR_NAMES` order, feeding the
existing PD. Geometry from the URDF joint frames, leg IK, gait scheduler, cycloid
foot trajectory, command mapping with the yaw term, and a virtual-model balance
option.

Three independent verifications, all passing:

- **FK against Chrono: 0.00000 m** on all four legs after settling on
  `STAND_ACTION`.
- **FK(IK(target)) round trip: 1.9e-16 m** worst case over 15,214 reachable
  targets across four legs, with 786 correctly rejected as unreachable and a
  negative control confirming an impossible target raises.
- The IK stance differs from the known-good `STAND_ACTION` by **at most 0.21 rad**
  on two joints, with hip identical at 0 and calf within 0.08.

## The gate result

    STAND_ACTION        z 0.2776   tilt   5.99 deg   STANDS
    IK stance h=0.28    z 0.1999   tilt  31.77 deg   falls
    IK stance h=0.30    z 0.2509   tilt  91.32 deg   on its side
    IK stance h=0.32    z 0.1034   tilt 180.00 deg   inverted

The trot run was never reached: the gate stops if the stand fails.

## Why this is not a kinematics bug

The gate's stated rule was "if the stand fails, the IK or the frame convention is
wrong." **That rule is falsified by the verifications above.** A 0.2 rad change
in two joints converting a stable stand into a 91-degree fall is the signature of
an **unstable equilibrium held open-loop**, not of a mistuned constant.
`STAND_ACTION` is a specific tuned point on this plant at `PD_KP = 20`, not a
generic stance.

The virtual-model balance term was swept across an order of magnitude
(`kp_roll` 0.02 → 0.20) and does not close it. Making this work is a balance
control project, not a tuning pass.

## Why it is parked rather than pursued

The generator existed to break the one-gait ceiling: every collected episode was
driven by a single policy whose gait structure never varies. The wide collection
has since delivered thousands of episodes with over half outside the old command
band, so **the diversity argument that would justify paying for balance control
is much weaker than it was.** This is a cost decision, not a verdict on the
approach.

## Three vacuous measurements, and all three passed a robot lying on the ground

The gate's first run reported PASS on a collapsed robot. Every one of its checks
was non-falsifiable:

1. **Attitude read `chrono.Q_to_Euler123` behind a `hasattr` guard.** That
   function does not exist in this build. The guard silently converted a missing
   API into the constant 0.0, so roll and pitch were exactly zero for every step
   of every run and both attitude checks passed without measuring anything.
   **A `hasattr` fallback that returns a value rather than raising turns an
   absent API into a passing measurement.**
2. **Base height was sampled after a 0.75 s settle**, so a robot that fell 0.26 m
   during the settle reported a drop of 0.0000 m — from the floor, to the floor.
3. **The pass condition tested only that motion had stopped.** A collapsed robot
   is perfectly stable.

The fixes: measure attitude with the same call the dataset uses, record height at
spawn, and require the base to be within 0.05 m of the *commanded* standing
height.

## A hypothesis dropped on evidence

Cardan ZYX is degenerate near pitch = ±π and reported `roll −1.5478, pitch
+3.1416` for the h=0.30 case — which looked exactly like a gimbal artifact
concealing an upright robot, at a plausible standing height of 0.2509 m.

Tested it with **tilt** — the angle between body +z and world +z, which has no
singularity. Result: **91.32 degrees.** The robot really is on its side. The
hypothesis was wrong and was dropped.

Recorded because a hypothesis abandoned on evidence is invisible in the final
result, and the temptation to keep a comfortable one is strongest when it would
overturn a negative finding.

The gate now decides on tilt. Cardan is retained only because `dataset.py`
reports those columns.
